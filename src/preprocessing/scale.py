
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
    # Weather numerics (from Meteostat join in ingestion stage 3e)
    "temp", "tmin", "tmax", "wspd", "pres",
        "rhum"
]


def build_scaling_stages_for_lr():
    """withMean=False to preserve sparsity for LR."""
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
    """Trees don't need scaling at all """
    pre_assembler = VectorAssembler(
        inputCols=NUMERIC_COLS, outputCol="numeric_scaled",  # same name, no scaler
        handleInvalid="keep"
    )
    return [pre_assembler], "numeric_scaled"