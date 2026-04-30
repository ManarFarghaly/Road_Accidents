"""
Data Cleaning.

Eager cleaning function that runs on the raw merged DataFrame *before*
the Spark ML Pipeline is built. Addresses every finding from the
8-dimension validation report.

Order of operations (each step is idempotent and commented with its
MCAR/MAR/MNAR justification where relevant):

    a   Drop 6 noise cols (>85 % missing OR 81 %-dominated by "Unclassified")
    a'  Drop target-leakage cols (Number_of_Casualties — STATS19 defines
        severity FROM casualty counts; Did_Police_Officer_Attend — recorded
        AFTER severity is assessed, so it leaks the target)
    a'' Rename Engine_Capacity_.CC. → Engine_Capacity_CC (dots break Spark ML)
    a'''Cast numeric columns to double (CSV inferSchema reads some as string)
    b   Null out UK DfT sentinel strings ("Data missing or out of range", etc.)
    b'  Fix Day_of_Week inconsistencies — derive correct value from Date column
        (Phase 2 Validation found ~2 rows where Day_of_Week ≠ date.day_name())
    b'' Null out invalid Age_Band_of_Driver codes (4,664 invalid per Phase 2)
    b'''Standardize Sex_of_Driver codes (M→Male, F→Female, 3→Not known)
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

CLASS IMBALANCE The target distribution is roughly Slight 85 % / Serious 14 % / Fatal 1 %. Fatal is
60× rarer than Slight. The recommended remedy is class weights passed to the
classifier (classWeightCol in LR/RF/GBT) — this uses ALL training data with
no information loss, unlike undersampling.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, Window, functions as F
from pyspark.ml.feature import Imputer


# ─── a. columns with so much missing/garbage they carry no signal ─────────
HIGH_MISSING_COLS = [
    "Carriageway_Hazards",            # 98.07 %
    "Special_Conditions_at_Site",     # 97.45 %
    "Hit_Object_in_Carriageway",      # 95.89 %
    "Hit_Object_off_Carriageway",     # 91.39 %
    "Skidding_and_Overturning",       # 87.19 %
    "2nd_Road_Class",                 # 41 % NA + 40 % "Unclassified" = 81 % noise
]

# ─── a'. target-leakage columns ───────────────────────────────────────────
# Number_of_Casualties: STATS19 defines severity directly from casualty counts
#   — Fatal ≥1 death, Serious ≥1 serious injury, Slight = only slight injuries.
#   Using it gives ~99% accuracy by reading the answer from another column.
# Did_Police_Officer_Attend: Police decide whether to attend BASED ON severity.
#   Fatal/Serious → almost always attended; Slight → often not.
#   This value is recorded after severity is assessed, so it leaks the target.
LEAKAGE_COLS = [
    "Number_of_Casualties",
    "Did_Police_Officer_Attend_Scene_of_Accident",
]

# ─── b. UK DfT sentinel string values meaning "missing" ──────────────────
# Sex_of_Driver (76 k rows) and carries real signal.
SENTINEL_STRINGS = [
    "Data missing or out of range",
    "Unknown",
    "unknown",
    "N/A",
    "",
]

# ─── b''. valid Age_Band_of_Driver values ────────────────────────────────
# Phase 2 found 4,664 records with invalid age band codes.
# UK STATS19 uses these exact strings — anything else is a data error.
VALID_AGE_BANDS = [
    "0 - 5", "6 - 10", "11 - 15", "16 - 20", "21 - 25",
    "26 - 35", "36 - 45", "46 - 55", "56 - 65", "66 - 75", "Over 75",
]

# ─── d. valid UK legal speed limits (mph) ────────────────────────────────
UK_LEGAL_SPEED_LIMITS = [20, 30, 40, 50, 60, 70]

# ─── e. required columns — rows missing any of these are dropped ──────────
REQUIRED_COLS = [
    "Accident_Severity",
    "Latitude",
    "Longitude",
]

# ─── f. validity bounds — real data errors ────────────────────────────────
VALIDITY_BOUNDS = {
    "Latitude":           (49.0, 61.0),
    "Longitude":          (-8.0, 2.0),
    "Engine_Capacity_CC": (50.0, 8000.0),
    "Age_of_Vehicle":     (0.0, 100.0),
    "Number_of_Vehicles": (1.0, 100.0),
}

# ─── g. numerics to median-impute ─────────────────────────────────────────
NUM_IMPUTE_COLS = [
    "Age_of_Vehicle",       # 16 % missing — MAR (foreign / unregistered)
    "Engine_Capacity_CC",   # 12 % missing — MAR
    "Driver_IMD_Decile",    # 34 % missing — MAR (foreign postcodes)
    # Weather features — MCAR (no observation for that station/date pair)
    "tavg", "tmin", "tmax", "prcp", "snow", "wspd", "pres",
]

# ─── h. group-wise mode imputation ───────────────────────────────────────
GROUP_MODE_IMPUTE = [
    ("model", "make"),    # fill null model with mode(model) per make
]

# ─── i. remaining categoricals — "Unknown" fill ───────────────────────────
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
    "Age_Band_of_Driver",   # added — gets nulled in step b'' for invalid codes
    "Sex_of_Driver",        # added — standardized in step b''' but may still have nulls
]

LABEL_COL = "Accident_Severity"


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════
def _snap_speed_limit(df: DataFrame) -> DataFrame:
    """
    Snap 36 invalid Speed_limit values to nearest UK legal limit.
    Midpoints: 25→20, 35→30, 45→40, 55→50, 65→60, >65→70.
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
    legal = col.isin(UK_LEGAL_SPEED_LIMITS)
    return df.withColumn(
        "Speed_limit",
        F.when(col.isNull(), col)
         .when(legal, col)
         .otherwise(snapped),
    )


def _fill_mode_by_group(df: DataFrame, target: str, group: str) -> DataFrame:
    """
    Fill null values in `target` with the mode of `target` within each
    `group`. Falls back to the global mode when a whole group is null.
    """
    if target not in df.columns or group not in df.columns:
        return df

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

    global_mode_row = (
        df.filter(F.col(target).isNotNull())
          .groupBy(target).count()
          .orderBy(F.col("count").desc())
          .select(target).head()
    )
    global_mode = global_mode_row[0] if global_mode_row else "Unknown"

    return (
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

def _clean_column_names(df):
    for col in df.columns:
        new_col = (
            col.replace(".", "_")
               .replace("-", "_")
               .replace("(", "")
               .replace(")", "")
               .replace(" ", "_")
        )
        df = df.withColumnRenamed(col, new_col)
    return df
# ══════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════
def clean(df: DataFrame) -> DataFrame:
    """
    Run the full cleaning pipeline on the raw merged DataFrame.
    Returns the cleaned DataFrame ready for the Spark ML Pipeline.
    """
    df = _clean_column_names(df)
    # a ─ drop noise + leakage columns
    to_drop = [c for c in HIGH_MISSING_COLS + LEAKAGE_COLS if c in df.columns]
    if to_drop:
        df = df.drop(*to_drop)

    # a'' ─ rename columns with dots — dots break Spark ML column resolution
    RENAME_COLS = {"Engine_Capacity_.CC.": "Engine_Capacity_CC"}
    for old, new in RENAME_COLS.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)

    # a''' ─ cast to double early — CSV inferSchema may read numerics as string
    NUMERIC_CAST_COLS = [
        "Speed_limit", "Number_of_Vehicles", "Latitude", "Longitude",
        "Age_of_Vehicle", "Engine_Capacity_CC", "Driver_IMD_Decile",
        "tavg", "tmin", "tmax", "prcp", "snow", "wspd", "pres",
    ]
    for c in NUMERIC_CAST_COLS:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast("double"))
  
   # b ─ null out DfT sentinel strings across all string columns (vectorized)
    str_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() == "string"]

    exprs = [
        F.when(F.col(f"`{c}`").isin(SENTINEL_STRINGS), None)
        .otherwise(F.col(f"`{c}`"))
        .alias(c)
        if c in str_cols else F.col(f"`{c}`")
        for c in df.columns
    ]

    df = df.select(*exprs)
    # b' ─ fix Day_of_Week inconsistencies (Phase 2: ~2 mismatches found)
    # Date is the source of truth — Day_of_Week is derivable and should match.
    if "Date" in df.columns and "Day_of_Week" in df.columns:
        df = df.withColumn(
            "Day_of_Week",
            F.date_format(F.to_date(F.col("Date"), "yyyy-MM-dd"), "EEEE"),
        )

    # b'' ─ null out invalid Age_Band_of_Driver codes (Phase 2: 4,664 invalid)
    # UK STATS19 uses a fixed set of age band strings — anything else is a
    # data entry error. Mark as null so step i fills with "Unknown".
    if "Age_Band_of_Driver" in df.columns:
        df = df.withColumn(
            "Age_Band_of_Driver",
            F.when(F.col("Age_Band_of_Driver").isin(VALID_AGE_BANDS),
                   F.col("Age_Band_of_Driver"))
             .otherwise(None),
        )

    # b''' ─ standardize Sex_of_Driver codes (Phase 2: 76,119 non-standard)
    # Different source systems use M/F/1/2/3 — normalize to full text.
    # "Not known" is a legitimate category (foreign/uninsured drivers).
    if "Sex_of_Driver" in df.columns:
        df = df.withColumn(
            "Sex_of_Driver",
            F.when(F.col("Sex_of_Driver").isin(["M", "1"]), "Male")
             .when(F.col("Sex_of_Driver").isin(["F", "2"]), "Female")
             .when(F.col("Sex_of_Driver").isin(["3"]), "Not known")
             .otherwise(F.col("Sex_of_Driver")),   # already correct text values kept
        )

    # c ─ null out numeric -1 sentinel (DfT "unknown" code for numerics)
    for c in NUM_IMPUTE_COLS:
        if c in df.columns:
            df = df.withColumn(c, F.when(F.col(c) < 0, None).otherwise(F.col(c)))

    # d ─ snap 36 invalid Speed_limit values to nearest UK legal limit
    if "Speed_limit" in df.columns:
        df = _snap_speed_limit(df)

    # e ─ enforce validity bounds — impossible values → NULL
    for col_name, (lo, hi) in VALIDITY_BOUNDS.items():
        if col_name in df.columns:
            df = df.withColumn(
                col_name,
                F.when((F.col(col_name) < lo) | (F.col(col_name) > hi), None)
                 .otherwise(F.col(col_name)),
            )

    # e' ─ drop rows missing required columns (target + geo coordinates)
    for c in REQUIRED_COLS:
        if c in df.columns:
            df = df.filter(F.col(c).isNotNull())

    # g ─ median impute numeric NULLs (MAR — median robust to skew)
    num_present = [c for c in NUM_IMPUTE_COLS if c in df.columns]
    if num_present:
        imputer = Imputer(
            inputCols=num_present,
            outputCols=num_present,
            strategy="median",
        )
        df = imputer.fit(df).transform(df)

    # h ─ group-wise mode fill (model ← mode per make)
    for target, group in GROUP_MODE_IMPUTE:
        df = _fill_mode_by_group(df, target, group)

    # i ─ remaining categoricals → "Unknown" level
    # StringIndexer will treat "Unknown" as its own level — preserves rows
    # rather than dropping them, and gives the model signal that data was absent.
    cat_present = [c for c in CAT_IMPUTE_COLS if c in df.columns]
    if cat_present:
        df = df.fillna("Unknown", subset=cat_present)

    return df


# ══════════════════════════════════════════════════════════════════════════
# Class-weight helper — RECOMMENDED over undersampling
# ══════════════════════════════════════════════════════════════════════════
def compute_class_weights(df: DataFrame, label_col: str = "Accident_Severity") -> dict:
    """
    Compute inverse-frequency class weights for use with Spark ML classifiers
    that support `weightCol` (LogisticRegression, RandomForestClassifier, GBTClassifier).

    This is the RECOMMENDED approach over undersampling because:
      - Uses ALL training data (no information loss)
      - Mathematically equivalent to oversampling without data duplication
      - No risk of overfitting to duplicated minority rows
      - Natively supported by Spark's LR, RF, and GBT

    Formula: weight(class) = total / (n_classes × count(class))
    This gives Fatal ~60×, Serious ~4×, Slight ~0.4× with this dataset.

    Usage in run.py:
        weights = compute_class_weights(train)
        train_w = add_class_weights(train, weights)
        rf = RandomForestClassifier(featuresCol="features",
                                    labelCol="label",
                                    weightCol="classWeight")

    Returns:
        dict mapping class label string → float weight
    """
    counts = {r[label_col]: r["count"] for r in df.groupBy(label_col).count().collect()}
    total = sum(counts.values())
    n_classes = len(counts)
    return {cls: total / (n_classes * cnt) for cls, cnt in counts.items()}


def add_class_weights(df: DataFrame,
                      weights: dict,
                      label_col: str = "Accident_Severity",
                      weight_col: str = "classWeight") -> DataFrame:
    """
    Add a `classWeight` column to the DataFrame based on the weights dict
    returned by `compute_class_weights()`.

    Call on the TRAINING split only — never on test.

    Example:
        weights = compute_class_weights(train)
        train_w = add_class_weights(train, weights)
        rf = RandomForestClassifier(..., weightCol="classWeight")
        model = Pipeline(stages=preprocessing_stages + [rf]).fit(train_w)
        predictions = model.transform(test)   # test has NO classWeight — correct
    """
    expr = None
    for cls, w in weights.items():
        condition = F.col(label_col) == cls
        expr = F.when(condition, w) if expr is None else expr.when(condition, w)
    expr = expr.otherwise(1.0)
    return df.withColumn(weight_col, expr)


