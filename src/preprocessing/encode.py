"""
Categorical Encoding.

Builds the ordered list of unfit Spark ML stages for:
  - Low-cardinality categoricals   -> StringIndexer + OneHotEncoder
  - High-cardinality categoricals  -> StringIndexer only (integer code)
  - The target (Accident_Severity) -> StringIndexer → column name "label"

StringIndexer maps categories → integers based on frequency order.
It is NOT scaling , It is NOT ordinal meaning , It is just a lookup table
Mapping is learned during .fit() on training data

Rationale: OneHotEncoder on a 20k-level column (make/model/LSOA) produces
a 20k-wide sparse vector per row which blows up and crawl models.
Integer encoding is fine for tree models and an acceptable trade-off
for LR at this cardinality.
"""
from __future__ import annotations

# from pyspark.ml.pipeline import PipelineStage 
from pyspark.ml.feature import OneHotEncoder, StringIndexer


# ── Low-cardinality (< 15 levels) → one-hot ───────────────────────────────
# Chosen threshold: 15 levels. At 15 dummies per column × 12 columns we add
# ~180 sparse-vector slots, which LR handles happily and GBT tolerates.
# Anything ≥ 15 levels (Vehicle_Type ≈ 21, Age_Band_of_Driver borderline)
# goes to HIGH_CARD_CATS as an integer index instead.
LOW_CARD_CATS = [
    "Day_of_Week",                               # 7
    "Road_Type",                                 # 6
    "Weather_Conditions",                        # 9
    "Light_Conditions",                          # 5
    "Road_Surface_Conditions",                   # 5
    "Urban_or_Rural_Area",                       # 3
    "Junction_Detail",                           # 9
    "Junction_Control",                          # 5
    "Sex_of_Driver",                             # 4 incl. "Not known"
    "Age_Band_of_Driver",                        # 11
    "Pedestrian_Crossing-Human_Control",         # 4
    "Pedestrian_Crossing-Physical_Facilities",   # 6
]

# ── High-cardinality (≥ 15 levels) → integer index only ───────────────────
# OneHot would blow LR up and slow GBT. Tree models handle integer-encoded
# categoricals natively; LR loses a little expressiveness but saves
# thousands of sparse dimensions.

HIGH_CARD_CATS = [
    "Vehicle_Type",                # ~21 levels
    "make",                        # ~60
    "model",                       # ~20 000
    "LSOA_of_Accident_Location",   # ~35 000
    "Local_Authority_(District)",  # ~400
    "Police_Force",                # ~50
]

LABEL_COL = "Accident_Severity"


def build_encoding_stages() -> tuple[list, list[str]]:
    """
    Returns:
        stages:              ordered list of unfit Spark ML stages.
        encoded_output_cols: names of the columns that should be fed
                             to the final VectorAssembler.
    """
    stages: list = []
    encoded_output_cols: list[str] = []

    # Low-card: index → one-hot
    for c in LOW_CARD_CATS:
        idx_col = f"{c}_idx"
        ohe_col = f"{c}_ohe"
        stages.append(
            StringIndexer(
                inputCol=c,
                outputCol=idx_col,
                handleInvalid="keep",    # unseen category bucket
            )
        )
        stages.append(
            OneHotEncoder(
                inputCol=idx_col,
                outputCol=ohe_col,
                handleInvalid="keep",
            )
        )
        encoded_output_cols.append(ohe_col)

    # High-card: index only, feed the integer code straight to the vector
    for c in HIGH_CARD_CATS:
        idx_col = f"{c}_idx"
        stages.append(
            StringIndexer(
                inputCol=c,
                outputCol=idx_col,
                handleInvalid="keep",
            )
        )
        encoded_output_cols.append(idx_col)

    
    stages.append(
        StringIndexer(
            inputCol=LABEL_COL,
            outputCol="label",
            handleInvalid="error",       # target null is a bug, not a new class
        )
    )

    return stages, encoded_output_cols
