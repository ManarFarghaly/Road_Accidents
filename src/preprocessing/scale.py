"""
Sub-task 3 — Numerical Scaling.

Builds an intermediate VectorAssembler + StandardScaler pair of
pipeline stages that produce the `numeric_scaled` vector column.

Why StandardScaler and not MinMaxScaler:
    Logistic Regression (trained by Member 3) benefits most from
    zero-mean / unit-variance features; tree models are indifferent.
    StandardScaler is also what the project proposal promised.

Why an intermediate VectorAssembler:
    Spark's StandardScaler operates on a vector column, not individual
    columns. The final assembler in assemble.py then concatenates this
    scaled-numeric vector with the one-hot vectors from encode.py.
"""
from __future__ import annotations

from pyspark.ml import PipelineStage
from pyspark.ml.feature import StandardScaler, VectorAssembler


NUMERIC_COLS = [
    # Accident numerics
    "Speed_limit",
    "Number_of_Vehicles",
    "Number_of_Casualties",
    "Latitude",
    "Longitude",
    # Vehicle numerics
    "Age_of_Vehicle",
    "Engine_Capacity_.CC.",
    "Driver_IMD_Decile",
    # Weather numerics (from Meteostat join in ingestion stage 3e)
    "tavg",
    "tmin",
    "tmax",
    "prcp",
    "snow",
    "wspd",
    "pres",
]


def build_scaling_stages() -> tuple[list[PipelineStage], str]:
    """
    Returns:
        stages:           [intermediate VectorAssembler, StandardScaler]
        scaled_vec_col:   name of the scaled vector column produced
                          (passed to the final assembler in assemble.py)
    """
    pre_assembler = VectorAssembler(
        inputCols=NUMERIC_COLS,
        outputCol="numeric_raw",
        handleInvalid="skip",       # any remaining null → row dropped at fit time
    )
    scaler = StandardScaler(
        inputCol="numeric_raw",
        outputCol="numeric_scaled",
        withMean=True,
        withStd=True,
    )
    return [pre_assembler, scaler], "numeric_scaled"
