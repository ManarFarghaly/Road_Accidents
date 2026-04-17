"""
Data Cleaning.

Eager cleaning function that runs on the raw merged DataFrame *before*
the Spark ML Pipeline is built. Addresses every finding from the
8-dimension validation report.

Order of operations (each step is idempotent and commented with its
MCAR/MAR/MNAR justification where relevant):

    a  Drop 5 columns with >85 % missing     (pure noise)
    b  Null out UK DfT sentinel strings      ("Data missing or out of range", etc.)
    c  Null out numeric -1 "unknown" codes
    d  Snap Speed_limit outliers to the nearest UK legal limit
    e  Drop rows where required fields are null
          - Accident_Severity  (target)
          - Latitude/Longitude (required for geospatial + features)
    f  Structural fill for 2nd_Road_Class   ("Not applicable" — single-road accidents)
    g  Median-impute numerics                (MAR, robust to skew)
    h  Mode-by-group impute for `model`      (model depends on make)
    i  "Unknown" fill for remaining cats     (MAR / MNAR — give them their own level)

Outliers (IQR flagged 470 k in Number_of_Casualties, 47 k in Number_of_Vehicles):
kept as-is. They are legitimate (a coach crash really can have 30 casualties),
tree models (RF/GBT) are outlier-robust, and StandardScaler re-centers LR inputs.
Winsorizing would destroy real severe-accident signal — exactly the minority class
we want the classifier to catch. Documented as a deliberate non-action.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, Window, functions as F
from pyspark.ml.feature import Imputer


# ─── a. columns with >85 % missing  ───────────────────────────────────────
# All five are MNAR (only populated for specific accident types) but at
# >85 % missing they carry almost no signal and would dominate the
# "Unknown" bucket in one-hot encoding.
HIGH_MISSING_COLS = [
    "Carriageway_Hazards",            # 98.07 %
    "Special_Conditions_at_Site",     # 97.45 %
    "Hit_Object_in_Carriageway",      # 95.89 %
    "Hit_Object_off_Carriageway",     # 91.39 %
    "Skidding_and_Overturning",       # 87.19 %
]

# ─── b. UK DfT sentinel string values meaning "missing" ──────────────────
# Note we do NOT include "Not known" here — that is a legitimate category
# for Sex_of_Driver (76 k rows) and carries information (foreign/uninsured
# drivers tend to cluster in certain severity patterns).
SENTINEL_STRINGS = [
    "Data missing or out of range",
    "Unknown",
    "unknown",
    "N/A",
    "",
]

# ─── d. valid UK legal speed limits (mph) ────────────────────────────────
# Used to snap the 36 invalid Speed_limit values (e.g. 10, 15, 999) to the
# nearest legal limit. Cheaper than dropping the rows, and most invalid
# values are typos within 5 mph of a legal limit.
UK_LEGAL_SPEED_LIMITS = [20, 30, 40, 50, 60, 70]

# ─── e. required columns — rows missing any of these are dropped ─────────
REQUIRED_COLS = [
    "Accident_Severity",   # target — null target is useless for supervised learning
    "Latitude",            # needed for features + already used for station join
    "Longitude",           # same
]

# ─── f. structurally-missing categorical ─────────────────────────────────
# 2nd_Road_Class is 41 % missing because **single-road accidents have no
# second road** — this is MNAR / structural, not random. Filling with
# "Unknown" would merge real unknowns with "not applicable" and mask the
# signal. Use an explicit category instead.
STRUCTURAL_NA_CATS = {
    "2nd_Road_Class": "Not applicable",
}

# ─── g. numerics to median-impute ────────────────────────────────────────
# MAR (Missing At Random) — probability of being missing can be predicted
# by other features (e.g. Driver_IMD_Decile is systematically missing for
# foreign postcodes), but the imputed value is stable at population scale.
# Median is robust to the heavy right-skew (skew 2-8) flagged in validation.
NUM_IMPUTE_COLS = [
    "Age_of_Vehicle",        # 16 %  missing — MAR (foreign / unregistered)
    "Engine_Capacity_.CC.",  # 12 %  missing — MAR
    "Driver_IMD_Decile",     # 34 %  missing — MAR (foreign postcodes) depends on driver location
    # Weather features added in ingestion stage e. Nulls occur for
    # (station_id, date) pairs where Meteostat had no observation — MCAR
    # from the accidents' perspective (no selection bias).
    "tavg",
    "tmin",
    "tmax",
    "prcp",
    "snow",
    "wspd",
    "pres",
]

# ─── h. group-wise mode imputation ───────────────────────────────────────
# `model` is missing for 15 % of vehicles. Mode-over-all would collapse
# half the fleet into "Ford Focus". Instead fill with the mode within
# each `make` — more sensible (unknown Volkswagen → most common VW model).
GROUP_MODE_IMPUTE = [
    ("model", "make"),    # fill null model with mode(model) per make
]

# ─── i. remaining categoricals — "Unknown" fill ──────────────────────────
# Pure MAR / low-MNAR where no structural meaning applies. StringIndexer
# treats "Unknown" as its own level, preserving signal rather than dropping
# rows. LSOA is high-cardinality so an "Unknown" level is cheap(negligable cost)
CAT_IMPUTE_COLS = [
    "LSOA_of_Accident_Location",
    "Weather_Conditions",
    "Road_Surface_Conditions",
    "Light_Conditions",
    "Junction_Detail",
    "Junction_Control",
    "Road_Type",
    "Vehicle_Type",
    "make",
]

LABEL_COL = "Accident_Severity"


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════
def _snap_speed_limit(df: DataFrame) -> DataFrame:
    """
    UK speed limits are a fixed set: {20, 30, 40, 50, 60, 70}. The raw
    data contains 36 rows with illegal values (e.g. 10, 15, 999). Snap
    each to the nearest legal limit using a CASE expression built from
    the midpoints between consecutive limits.

    Midpoints: 25, 35, 45, 55, 65 → buckets [-∞, 25, 35, 45, 55, 65, ∞]
    map to                                    20  30  40  50  60  70.
    """
    col = F.col("Speed_limit")
    snapped = (
        F.when(col <= 25, 20)
         .when(col <= 35, 30)
         .when(col <= 45, 40)
         .when(col <= 55, 50)
         .when(col <= 65, 60)
         .otherwise(70)
    )
    # Only snap non-null values outside the legal set; leave legal values
    # and nulls untouched (nulls flow to Imputer if present).
    legal = col.isin(UK_LEGAL_SPEED_LIMITS)
    return df.withColumn(
        "Speed_limit",
        F.when(col.isNull(), col)
         .when(legal, col)
         .otherwise(snapped),
    )


def _fill_mode_by_group(df: DataFrame, target: str, group: str) -> DataFrame:
    """
    Fill null values in `target` with the mode of `target` computed
    within each `group`. Falls back to the global mode when a whole
    group is null for `target`.
    """
    if target not in df.columns or group not in df.columns:
        return df

    # Mode per group
    grp_mode = (
        df.filter(F.col(target).isNotNull())
          .groupBy(group, target)
          .count()
          .withColumn(
              "rk",
              F.row_number().over(
                  Window.partitionBy(group).orderBy(F.col("count").desc())
              ),
          )
          .filter(F.col("rk") == 1)
          .select(group, F.col(target).alias(f"{target}_grp_mode"))
    )

    # Global fallback mode
    global_mode_row = (
        df.filter(F.col(target).isNotNull())
          .groupBy(target).count()
          .orderBy(F.col("count").desc())
          .select(target).head()
    )
    global_mode = global_mode_row[0] if global_mode_row else "Unknown"

    out = (
        df.join(grp_mode, on=group, how="left")
          .withColumn(
              target,
              F.coalesce(
                  F.col(target),
                  F.col(f"{target}_grp_mode"),
                  F.lit(global_mode),
              ),
          )
          .drop(f"{target}_grp_mode")
    )
    return out


# ══════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════
def clean(df: DataFrame) -> DataFrame:
    """
    Run the full cleaning pipeline on the raw merged DataFrame and
    return the cleaned DataFrame. Called once by Member 3 before
    constructing their Spark ML Pipeline.
    """
    # a ─ drop hopelessly-empty columns
    to_drop = [c for c in HIGH_MISSING_COLS if c in df.columns]
    if to_drop:
        df = df.drop(*to_drop)

    # b ─ null out sentinel strings across every string column
    str_cols = [
        f.name for f in df.schema.fields
        if f.dataType.simpleString() == "string"
    ]
    for c in str_cols:
        df = df.withColumn(
            c,
            F.when(F.col(c).isin(SENTINEL_STRINGS), None).otherwise(F.col(c)),
        )

    # c ─ numeric -1 sentinel → NULL (DfT "unknown" code for numerics)
    for c in NUM_IMPUTE_COLS:
        if c in df.columns:
            df = df.withColumn(
                c,
                F.when(F.col(c) < 0, None).otherwise(F.col(c)),
            )

    # d ─ snap 36 invalid Speed_limit values to nearest UK legal limit
    if "Speed_limit" in df.columns:
        df = _snap_speed_limit(df)

    # e ─ drop rows missing any REQUIRED column (target + geo)
    for c in REQUIRED_COLS:
        if c in df.columns:
            df = df.filter(F.col(c).isNotNull())

    # f ─ structural NA fill (2nd_Road_Class has a real "not applicable" meaning)
    for c, fill_value in STRUCTURAL_NA_CATS.items():
        if c in df.columns:
            df = df.fillna(fill_value, subset=[c])

    # g ─ numeric median imputation (MAR, robust to skew)
    num_present = [c for c in NUM_IMPUTE_COLS if c in df.columns]
    if num_present:
        imputer = Imputer(
            inputCols=num_present,
            outputCols=num_present,     # overwrite in place
            strategy="median",
        )
        df = imputer.fit(df).transform(df)

    # h ─ group-wise mode fill (model ← mode per make)
    for target, group in GROUP_MODE_IMPUTE:
        df = _fill_mode_by_group(df, target, group)

    # i ─ remaining categoricals — give nulls their own "Unknown" level
    cat_present = [c for c in CAT_IMPUTE_COLS if c in df.columns]
    if cat_present:
        df = df.fillna("Unknown", subset=cat_present)

    return df
