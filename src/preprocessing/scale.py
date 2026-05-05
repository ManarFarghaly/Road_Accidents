
from __future__ import annotations

from pyspark.ml.feature import StandardScaler, VectorAssembler

NUMERIC_COLS = [
    "Speed_limit",
    "Number_of_Vehicles",
    "Latitude",
    "Longitude",
    "Age_of_Vehicle",
    "Engine_Capacity_CC",
    "Driver_IMD_Decile",
    "InScotland",
    "Was_Vehicle_Left_Hand_Drive",
    "temp", "tmin", "tmax", "wspd", "pres",
        "rhum"
]

from pyspark.ml.feature import StandardScaler, VectorAssembler

from src.preprocessing.encode import TARGET_ENCODE_OUT_COLS

NUMERIC_COLS = [
    "Speed_limit",
    "Number_of_Vehicles",
    "Latitude",
    "Longitude",
    "Age_of_Vehicle",
    "Engine_Capacity_CC",
    "Driver_IMD_Decile",
    "InScotland",
    "Was_Vehicle_Left_Hand_Drive",
    "temp", "tmin", "tmax", "wspd", "pres", "rhum",
    *TARGET_ENCODE_OUT_COLS,
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
    tree_numeric_cols = NUMERIC_COLS
    pre_assembler = VectorAssembler(
        inputCols=tree_numeric_cols, outputCol="numeric_scaled",
        handleInvalid="keep"
    )
    return [pre_assembler], "numeric_scaled"