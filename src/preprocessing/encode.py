from __future__ import annotations

from pyspark.ml.feature import OneHotEncoder, StringIndexer, FeatureHasher
from pyspark.sql import DataFrame, functions as F

# 
# Column lists
# 

LOW_CARD_CATS = [
    "Day_of_Week",                             # 7
    "Road_Type",                               # 6
    "Weather_Conditions",                      # 9
    "Light_Conditions",                        # 5
    "Road_Surface_Conditions",                 # 5
    "Urban_or_Rural_Area",                     # 3
    "Junction_Detail",                         # 9
    "Junction_Control",                        # 5
    "Sex_of_Driver",                           # 4 incl. "Not known"
    "Pedestrian_Crossing-Human_Control",       # 4
    "Pedestrian_Crossing-Physical_Facilities", # 6
    "Driver_Home_Area_Type",                   # 4
    "Journey_Purpose_of_Driver",               # 8
    "Junction_Location",                       # 10
    "Propulsion_Code",                         # 13
    "Towing_and_Articulation",                 # 7
    "Vehicle_Leaving_Carriageway",             # 10
    "X1st_Point_of_Impact",                    # 6
]

# Ordinal — StringIndexer with alphabetAsc preserves rank order
ORDINAL_CATS = [
    "Age_Band_of_Driver",   # "0 - 5" < "11 - 15" < ... < "Over 75"
    "1st_Road_Class",       # Motorway > A > B > C > Unclassified
]

# High-card nominal — goes into FeatureHasher for LR, StringIndexer for trees
HIGH_CARD_CATS = [
    "Vehicle_Type",                # ~21
    "make",                        # ~60
    "Local_Authority_(District)",  # ~400
    "Local_Authority_(Highway)",   # ~207
    "Police_Force",                # ~50
    "Vehicle_Manoeuvre",           # ~19
]

# Extreme-cardinality — target encoding only (no hashing, no OHE)
# These are excluded from HIGH_CARD_CATS to avoid poisoning the hash space.
TARGET_ENCODE_COLS = [
    "model",                     # ~20,000 unique values
    "LSOA_of_Accident_Location", # ~35,000 unique values
]

# After target encoding these become plain floats with these names:
TARGET_ENCODE_OUT_COLS = [f"{c}_te" for c in TARGET_ENCODE_COLS]

LABEL_COL = "Accident_Severity"

# Numeric mapping for target encoding mean calculation.
# StringIndexer assigns 0=Slight (most frequent), 1=Serious, 2=Fatal.
# We use the same ordering so target-encoded values are consistent with labels.
SEVERITY_NUM_MAP = {
    "Slight":  0.0,
    "Serious": 1.0,
    "Fatal":   2.0,
}

_SEVERITY_NUM_COL = "_severity_num"   # temp internal column name


# 
# Target encoding  — must be called OUTSIDE the Pipeline
# 

def fit_target_encoding(
    train_df: DataFrame,
    col_name: str,
    smoothing: float = 10.0,
) -> DataFrame:
    out_col = f"{col_name}_te"

    # Add numeric severity so we can compute mean()
    severity_expr = F.create_map(
        *[item for pair in
          [(F.lit(k), F.lit(v)) for k, v in SEVERITY_NUM_MAP.items()]
          for item in pair]
    )
    train_with_num = train_df.withColumn(
        _SEVERITY_NUM_COL,
        severity_expr[F.col(LABEL_COL)],
    )

    global_mean = train_with_num.select(
        F.mean(_SEVERITY_NUM_COL)
    ).collect()[0][0]

    stats = (
        train_with_num
        .groupBy(col_name)
        .agg(
            F.count("*").alias("_n"),
            F.mean(_SEVERITY_NUM_COL).alias("_cat_mean"),
        )
        .withColumn(
            out_col,
            (F.col("_n") * F.col("_cat_mean") + smoothing * global_mean)
            / (F.col("_n") + smoothing),
        )
        .select(col_name, out_col)
    )

    return stats


def apply_target_encoding(
    df: DataFrame,
    stats_df: DataFrame,
    col_name: str,
    global_fallback: float = None,
) -> DataFrame:
    out_col = f"{col_name}_te"

    if global_fallback is None:
        global_fallback = float(
            stats_df.select(F.mean(out_col)).collect()[0][0]
        )

    return (
        df.join(F.broadcast(stats_df), on=col_name, how="left")
          .withColumn(
              out_col,
              F.coalesce(F.col(out_col), F.lit(global_fallback)),
          )
    )


def fit_and_apply_target_encodings(
    train_df: DataFrame,
    test_df: DataFrame,
    smoothing: float = 10.0,
) -> tuple[DataFrame, DataFrame]:
    for col_name in TARGET_ENCODE_COLS:
        stats = fit_target_encoding(train_df, col_name, smoothing=smoothing)
        train_df = apply_target_encoding(train_df, stats, col_name)
        test_df  = apply_target_encoding(test_df,  stats, col_name)

    return train_df, test_df


# 
# Pipeline encoding stages
# 

def build_encoding_stages_trees():
    """
    Trees pipeline: ALL categoricals → StringIndexer integer index.
    No OHE — trees split on integer values natively.
    TARGET_ENCODE_COLS are excluded: they will be float columns by the
    time the pipeline runs (added to NUMERIC_COLS in scale.py).
    """
    stages = []
    encoded_output_cols = []

    # Trees CAN use model and LSOA as numeric target-encoded features,
    # so we exclude the raw string columns from the tree indexer.
    all_cats = LOW_CARD_CATS + HIGH_CARD_CATS + ORDINAL_CATS

    for c in all_cats:
        idx_col = f"{c}_idx"
        order   = "alphabetAsc" if c in ORDINAL_CATS else "frequencyDesc"
        stages.append(StringIndexer(
            inputCol=c, outputCol=idx_col,
            handleInvalid="keep", stringOrderType=order,
        ))
        encoded_output_cols.append(idx_col)

    stages.append(StringIndexer(
        inputCol=LABEL_COL, outputCol="label", handleInvalid="keep"
    ))
    return stages, encoded_output_cols


def build_encoding_stages_lr(num_features: int = 4096):
    """
    LR pipeline:
      - LOW_CARD_CATS  → StringIndexer → OneHotEncoder  (sparse vectors)
      - ORDINAL_CATS   → StringIndexer only              (preserve rank)
      - HIGH_CARD_CATS → FeatureHasher                   (4096 buckets,
                         ~757 unique values → <1% collision)
      - TARGET_ENCODE_COLS are NOT processed here — they must already
        exist as float columns (model_te, LSOA_of_Accident_Location_te)
        added to NUMERIC_COLS in scale.py before the pipeline runs.
        Call fit_and_apply_target_encodings() in run.py before pipeline.fit().
    """
    stages = []
    encoded_output_cols = []

    # Nominal low-card → OHE
    for c in LOW_CARD_CATS:
        idx_col = f"{c}_idx"
        ohe_col = f"{c}_ohe"
        stages.append(StringIndexer(
            inputCol=c, outputCol=idx_col, handleInvalid="keep"
        ))
        stages.append(OneHotEncoder(
            inputCol=idx_col, outputCol=ohe_col, handleInvalid="keep"
        ))
        encoded_output_cols.append(ohe_col)

    # Ordinal → integer index only
    for c in ORDINAL_CATS:
        idx_col = f"{c}_idx"
        stages.append(StringIndexer(
            inputCol=c, outputCol=idx_col,
            stringOrderType="alphabetAsc", handleInvalid="keep",
        ))
        encoded_output_cols.append(idx_col)

    # High-card nominal → hash trick
    # Only 6 columns with ~757 total unique values → 4096 buckets is clean
    stages.append(FeatureHasher(
        inputCols=HIGH_CARD_CATS,
        outputCol="hashed_high_card",
        numFeatures=num_features,
    ))
    encoded_output_cols.append("hashed_high_card")

    # TARGET_ENCODE_COLS (model_te, LSOA_te) are float columns in NUMERIC_COLS
    # — they never enter the pipeline as strings, so nothing to add here.

    stages.append(StringIndexer(
        inputCol=LABEL_COL, outputCol="label", handleInvalid="keep"
    ))
    return stages, encoded_output_cols