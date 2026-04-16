"""
Sub-task 2 — Categorical Encoding.

Builds the ordered list of unfit Spark ML stages for:
  - Low-cardinality categoricals   -> StringIndexer + OneHotEncoder
  - High-cardinality categoricals  -> StringIndexer only (integer code)
  - The target (Accident_Severity) -> StringIndexer → column name "label"

Rationale: OneHotEncoder on a 20k-level column (make/model/LSOA) produces
a 20k-wide sparse vector per row which blows up LR and makes GBT crawl.
Integer encoding is fine for tree models and an acceptable trade-off
for LR at this cardinality.
"""
from __future__ import annotations

from pyspark.ml import PipelineStage
from pyspark.ml.feature import OneHotEncoder, StringIndexer


# ── Low-cardinality (≤ 20 levels) → one-hot ───────────────────────────────
LOW_CARD_CATS = [
    "Day_of_Week",
    "Road_Type",
    "Weather_Conditions",
    "Light_Conditions",
    "Road_Surface_Conditions",
    "Urban_or_Rural_Area",
    "Junction_Detail",
    "Junction_Control",
    "Sex_of_Driver",
    "Age_Band_of_Driver",
    "Vehicle_Type",
    "Pedestrian_Crossing-Human_Control",
    "Pedestrian_Crossing-Physical_Facilities",
]

# ── High-cardinality (> 20 levels) → integer index only ───────────────────
HIGH_CARD_CATS = [
    "make",
    "model",
    "LSOA_of_Accident_Location",
    "Local_Authority_(District)",
    "Police_Force",
]

LABEL_COL = "Accident_Severity"


def build_encoding_stages() -> tuple[list[PipelineStage], list[str]]:
    """
    Returns:
        stages:              ordered list of unfit Spark ML stages.
        encoded_output_cols: names of the columns that should be fed
                             to the final VectorAssembler.
    """
    stages: list[PipelineStage] = []
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

    # Target label — Member 3's classifier expects exactly this column name
    stages.append(
        StringIndexer(
            inputCol=LABEL_COL,
            outputCol="label",
            handleInvalid="error",       # target null is a bug, not a new class
        )
    )

    return stages, encoded_output_cols
