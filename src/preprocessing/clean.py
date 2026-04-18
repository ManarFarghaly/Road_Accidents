"""
Data Cleaning.

Eager cleaning function that runs on the raw merged DataFrame *before*
the Spark ML Pipeline is built. Addresses every finding from the
8-dimension validation report.

Order of operations (each step is idempotent and commented with its
MCAR/MAR/MNAR justification where relevant):

    a   Drop 6 noise cols (>85 % missing OR 81 %-dominated by "Unclassified")
    a'  Drop target-leakage cols (Number_of_Casualties — STATS19 defines
        severity FROM casualty counts; also unavailable at prediction time)
    b   Null out UK DfT sentinel strings ("Data missing or out of range", etc.)
    c   Null out numeric -1 "unknown" codes
    d   Snap Speed_limit outliers to the nearest UK legal limit
    e   Enforce validity bounds (Lat/Lon UK box, Engine_Capacity 50-8000cc,
        Age_of_Vehicle 0-100, Number_of_Vehicles ≥1) — real DATA ERRORS → NULL
    e'  Drop rows missing required fields (Severity target + Lat/Lon)
    g   Median-impute remaining numeric NULLs (MAR, robust to skew)
    h   Mode-by-group impute for `model` (model depends on make)
    i   "Unknown" fill for remaining cats (MAR / MNAR — own level)

STATISTICAL OUTLIERS (IQR flagged 47 k in Number_of_Vehicles) are kept as-is.
They are legitimate (a coach crash really can have 30 casualties), tree models
(RF/GBT) are outlier-robust, and StandardScaler re-centers LR inputs. Note the
distinction vs step e above: "data error" = physically/legally impossible
value (null out); "outlier" = extreme-but-real observation (keep).

CLASS IMBALANCE is documented here but NOT handled in this module. The target
distribution is roughly Slight 85 % / Serious 14 % / Fatal 1 %. Fatal is
60× rarer than Slight — the model will learn "predict Slight" and score 85 %
accuracy while being useless. Remedies (pick one, Member 3's decision):
   • classWeightCol in LR/GBT   ← cheapest, no data duplication
   • random undersampling of the majority class (see `rebalance_undersample`)
   • SMOTE-style oversampling (Spark doesn't ship it; imblearn on sampled df)
We provide `rebalance_undersample()` as an optional helper; use it only on
the TRAINING split, never on test — resampling the test set gives wrong
performance numbers.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, Window, functions as F
from pyspark.ml.feature import Imputer


# ─── a. columns with so much missing/garbage they carry no signal ─────────
# First five are MNAR (only populated for specific accident types) at >85 %
# missing. 2nd_Road_Class looks different but resolves the same way:
# 41 % NA  + 40 % "Unclassified" = 81 % uninformative — the remaining 19 %
# spread across six classes carries too little signal to justify a column.
HIGH_MISSING_COLS = [
    "Carriageway_Hazards",            # 98.07 %
    "Special_Conditions_at_Site",     # 97.45 %
    "Hit_Object_in_Carriageway",      # 95.89 %
    "Hit_Object_off_Carriageway",     # 91.39 %
    "Skidding_and_Overturning",       # 87.19 %
    "2nd_Road_Class",                 # 41 % NA + 40 % "Unclassified" = 81 % noise
]

# ─── a'. target-leakage columns — drop before training ────────────────────
# UK STATS19 defines Accident_Severity DIRECTLY from casualty counts:
#     Fatal   = at least one fatal casualty
#     Serious = at least one serious injury
#     Slight  = only slight injuries
# So Number_of_Casualties is essentially the target written in another
# column — a model using it achieves near-perfect accuracy by cheating.
# It is also NOT KNOWN AT PREDICTION TIME (a traffic-safety app predicts
# severity BEFORE the accident happens), which is what your mentor
# flagged. Dropping it is mandatory, not a judgement call.
LEAKAGE_COLS = [
    "Number_of_Casualties",
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

# ─── f. validity bounds — real data errors, not statistical outliers ─────
# These are physically/legally impossible values (not extreme-but-real
# observations like a 30-casualty coach crash). We NULL them out here so
# the imputer replaces them with the median; leaving them distorts means
# and scalers.
#
#   Latitude/Longitude : UK geographic bounds
#   Engine_Capacity    : 50 cc (moped) … 8 000 cc (heavy truck)
#   Age_of_Vehicle     : 0 … 100 years
#   Number_of_Vehicles : 1 (must be ≥1) … 100 (safe upper bound)
#
# Note: this is orthogonal to statistical-outlier handling. Skewed-but-
# legitimate values (IQR-flagged large casualty counts, rare high engine
# capacities) are preserved — they carry real signal about severe events.
VALIDITY_BOUNDS = {
    "Latitude":             (49.0, 61.0),
    "Longitude":            (-8.0, 2.0),
    "Engine_Capacity_CC": (50.0, 8000.0),
    "Age_of_Vehicle":       (0.0, 100.0),
    "Number_of_Vehicles":   (1.0, 100.0),
}

# ─── g. numerics to median-impute ────────────────────────────────────────
# MAR (Missing At Random) — probability of being missing can be predicted
# by other features (e.g. Driver_IMD_Decile is systematically missing for
# foreign postcodes), but the imputed value is stable at population scale.
# Median is robust to the heavy right-skew (skew 2-8) flagged in validation.
NUM_IMPUTE_COLS = [
    "Age_of_Vehicle",        # 16 %  missing — MAR (foreign / unregistered)
    "Engine_Capacity_CC",  # 12 %  missing — MAR
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
    # a ─ drop hopelessly-empty columns + target-leakage columns
    to_drop = [
        c for c in HIGH_MISSING_COLS + LEAKAGE_COLS
        if c in df.columns
    ]
    if to_drop:
        df = df.drop(*to_drop)
    # ── Rename columns with special characters in their names ────────────
    # Dots in column names cause AnalysisException throughout Spark ML.
    # Rename once here so all downstream stages see clean names.
    RENAME_COLS = {
        "Engine_Capacity_.CC.": "Engine_Capacity_CC",
    }
    for old, new in RENAME_COLS.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)
    # Cast known numeric columns to double early — avoids type errors downstream
    NUMERIC_CAST_COLS = [
        "Speed_limit", "Number_of_Vehicles", "Latitude", "Longitude",
        "Age_of_Vehicle", "Engine_Capacity_CC", "Driver_IMD_Decile",
        "tavg", "tmin", "tmax", "prcp", "snow", "wspd", "pres",
    ]
    for c in NUMERIC_CAST_COLS:
        if c in df.columns:
            safe_ref = f"`{c}`" if any(ch in c for ch in [".", "(", ")"]) else c
            df = df.withColumn(c, F.col(safe_ref).cast("double"))

    # b ─ null out sentinel strings across every string column
    str_cols = [
        f.name for f in df.schema.fields
        if f.dataType.simpleString() == "string"
    ]
    for c in str_cols:
        safe = f"`{c}`"
        df = df.withColumn(
            c,
            F.when(F.col(safe).isin(SENTINEL_STRINGS), None).otherwise(F.col(safe)),
        )

    # c ─ numeric -1 sentinel → NULL (DfT "unknown" code for numerics)
    for c in NUM_IMPUTE_COLS:
        if c in df.columns:
            safe = f"`{c}`"
            df = df.withColumn(
                c,
                F.when(F.col(safe) < 0, None).otherwise(F.col(safe)),
            )

    # d ─ snap 36 invalid Speed_limit values to nearest UK legal limit
    if "Speed_limit" in df.columns:
        df = _snap_speed_limit(df)

    # e ─ enforce validity bounds — real data errors → NULL
    # (applied BEFORE the required-column drop so Lat/Lon garbage like 0.0
    # or 9999 gets nulled first, then the row is dropped if unrecoverable.)
    for col, (lo, hi) in VALIDITY_BOUNDS.items():
        if col in df.columns:
            safe = f"`{col}`"
            df = df.withColumn(
                col,
                F.when((F.col(safe) < lo) | (F.col(safe) > hi), None)
                .otherwise(F.col(safe)),
            )

    # e' ─ drop rows missing any REQUIRED column (target + geo)
    for c in REQUIRED_COLS:
        if c in df.columns:
            df = df.filter(F.col(c).isNotNull())


    # g ─ numeric median imputation (MAR, robust to skew)
    num_present = [c for c in NUM_IMPUTE_COLS if c in df.columns]
    if num_present:
        # Rename columns with special chars (dots/parens) to safe names
        # Imputer cannot handle column names containing dots
        rename_map = {}   # original → safe
        for c in num_present:
            if any(ch in c for ch in [".", "(", ")", " "]):
                safe = c.replace(".", "_").replace("(", "_").replace(")", "_").replace(" ", "_")
                rename_map[c] = safe
                df = df.withColumnRenamed(c, safe)

        # Use safe names for imputer
        safe_num_present = [rename_map.get(c, c) for c in num_present]

        imputer = Imputer(
            inputCols=safe_num_present,
            outputCols=safe_num_present,
            strategy="median",
        )
        df = imputer.fit(df).transform(df)

        # Rename back to original names
        for original, safe_name in rename_map.items():
            df = df.withColumnRenamed(safe_name, original)

    # h ─ group-wise mode fill (model ← mode per make)
    for target, group in GROUP_MODE_IMPUTE:
        df = _fill_mode_by_group(df, target, group)

    # i ─ remaining categoricals — give nulls their own "Unknown" level
    cat_present = [c for c in CAT_IMPUTE_COLS if c in df.columns]
    if cat_present:
        df = df.fillna("Unknown", subset=cat_present)

    return df


# ══════════════════════════════════════════════════════════════════════════
# Optional helper — class-imbalance rebalancing (training split only!)
# ══════════════════════════════════════════════════════════════════════════
def rebalance_undersample(
    df: DataFrame,
    label_col: str = "Accident_Severity",
    ratio: float = 3.0,
    seed: int = 42,
) -> DataFrame:
    """
    Undersample the majority class so that the largest class is at most
    `ratio` × the smallest class. With the default ratio=3 the target
    distribution becomes roughly Slight:Serious:Fatal = 3:3:1 instead of
    85:14:1 — still slightly imbalanced (so priors aren't totally wrong)
    but close enough for LR/RF to learn minority patterns.

    IMPORTANT: call this ONLY on the training split, never on test.
    Resampling the test set produces misleading performance metrics.

    Example:
        train, test = cleaned.randomSplit([0.8, 0.2], seed=42)
        train_balanced = rebalance_undersample(train)
        model = pipeline.fit(train_balanced)
        metrics = evaluator.evaluate(model.transform(test))   # unresampled!
    """
    counts = {r[label_col]: r["count"]
              for r in df.groupBy(label_col).count().collect()}
    if not counts:
        return df

    min_count = min(counts.values())
    target = int(min_count * ratio)

    parts = []
    for cls, n in counts.items():
        frac = min(1.0, target / n)
        parts.append(
            df.filter(F.col(label_col) == cls).sample(frac, seed=seed)
        )

    # Union all the per-class samples
    out = parts[0]
    for p in parts[1:]:
        out = out.unionByName(p)
    return out
