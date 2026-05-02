
from __future__ import annotations

# from pyspark.ml.pipeline import PipelineStage  
from pyspark.ml.feature import StandardScaler, VectorAssembler

NUMERIC_COLS = [
    # Accident numerics
    "Speed_limit",
    "Number_of_Vehicles",
    "Latitude",
    "Longitude",
    # Vehicle numerics
    "Age_of_Vehicle",
    "Engine_Capacity_CC",
    "Driver_IMD_Decile",
    "InScotland",                   
    "Was_Vehicle_Left_Hand_Drive",  
    # Weather numerics (from Meteostat join in ingestion stage 3e)
    "temp", "tmin", "tmax", "wspd", "pres",
        "rhum"
]

from pyspark.ml.feature import StandardScaler, VectorAssembler

# TARGET_ENCODE_OUT_COLS imported here so scale.py stays in sync with encode.py
from src.preprocessing.encode import TARGET_ENCODE_OUT_COLS

NUMERIC_COLS = [
    # ── Accident 
    "Speed_limit",
    "Number_of_Vehicles",
    "Latitude",
    "Longitude",
    # ── Vehicle 
    "Age_of_Vehicle",
    "Engine_Capacity_CC",
    "Driver_IMD_Decile",
    # ── Binary flags
    "InScotland",
    "Was_Vehicle_Left_Hand_Drive",
    # ── Weather 
    "temp", "tmin", "tmax", "wspd", "pres", "rhum",
    # ── Target-encoded high-card cols (model_te, LSOA_of_Accident_Location_te)
    # These are float columns produced by fit_and_apply_target_encodings()
    # in run.py BEFORE the pipeline runs. They are treated as plain numerics
    # here and scaled alongside the other numeric features.
    *TARGET_ENCODE_OUT_COLS,   # expands to ["model_te", "LSOA_of_Accident_Location_te"]
]


def build_scaling_stages_for_lr():
    pre_assembler = VectorAssembler(
        inputCols=NUMERIC_COLS, outputCol="numeric_raw", handleInvalid="keep"
    )
    scaler = StandardScaler(
        inputCol="numeric_raw", outputCol="numeric_scaled",
        withMean=False,
        withStd=True,
    )
    return [pre_assembler, scaler], "numeric_scaled"


def build_scaling_stages_for_trees():
    tree_numeric_cols = [
        c for c in NUMERIC_COLS
        if c not in TARGET_ENCODE_OUT_COLS   # trees use StringIndexer, not _te floats
    ]
    pre_assembler = VectorAssembler(
        inputCols=tree_numeric_cols, outputCol="numeric_scaled",
        handleInvalid="keep"
    )
    return [pre_assembler], "numeric_scaled"