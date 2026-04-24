"""
Member 1 & Member 2 — Preprocessing public API.

Member 3 imports exactly two things from this package:

    from src.preprocessing import clean, build_preprocessing_stages

Usage (what Member 3's code looks like):

    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from src.preprocessing import clean, build_preprocessing_stages

    raw         = spark.read.parquet("data/interim/merged.parquet")
    df          = clean(raw)
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    pipeline = Pipeline(stages=build_preprocessing_stages() + [LogisticRegression()])
    model    = pipeline.fit(train)

Member 2 additions:
    - DangerIndexTransformer: Adds danger_index feature (0-1 safety score)
    - VehicleVulnerabilityTransformer: Adds vehicle_vulnerable feature (boolean)
    - HotspotClusterTransformer: Adds accident_hotspot_cluster feature (0-9 region ID)
    - LocationDensityTransformer: Adds location_density feature (urban/suburban/rural)

These are integrated into build_preprocessing_stages() at the beginning,
BEFORE encoding/scaling to ensure domain features are created from raw data.
"""
# from pyspark.ml.pipeline import PipelineStage

from src.preprocessing.assemble import build_assembler_stage
from src.preprocessing.clean import clean, rebalance_undersample
from src.preprocessing.encode import build_encoding_stages
from src.preprocessing.scale import build_scaling_stages
from src.feature_engineering import (
    DangerIndexTransformer,
    VehicleVulnerabilityTransformer,
    HotspotClusterTransformer,
    LocationDensityTransformer,
)


def build_preprocessing_stages() -> list:
    """
    Returns the ordered list of unfit Spark ML stages that perform
    domain feature engineering + spatial feature engineering + encoding + scaling + final assembly.
    
    Pipeline stages order:
    1. DangerIndexTransformer - creates danger_index feature (0-1)
    2. VehicleVulnerabilityTransformer - creates vehicle_vulnerable feature (bool)
    3. HotspotClusterTransformer - creates accident_hotspot_cluster feature (0-9)
    4. LocationDensityTransformer - creates location_density feature (string)
    5. Encoding stages - StringIndexer + OneHotEncoder for categoricals
    6. Scaling stages - StandardScaler for numerics
    7. Final VectorAssembler - combines all features into 'features' vector

    Member 3 prepends their classifier inside a pyspark.ml.Pipeline.

    Note: cleaning is NOT a stage — call `clean(df)` on the raw
    DataFrame *before* constructing the Pipeline.
    """
    # Domain + spatial features (Member 2)
    domain_stages = [
        DangerIndexTransformer(),
        VehicleVulnerabilityTransformer(),
        HotspotClusterTransformer(),
        LocationDensityTransformer(),
    ]

    # Member 1 stages
    scale_stages, scaled_vec_col = build_scaling_stages()
    encode_stages, encoded_cols = build_encoding_stages()
    assembler = build_assembler_stage(scaled_vec_col, encoded_cols)

    return domain_stages + scale_stages + encode_stages + [assembler]


__all__ = [
    "clean",
    "rebalance_undersample",
    "build_preprocessing_stages",
    "DangerIndexTransformer",
    "VehicleVulnerabilityTransformer",
    "HotspotClusterTransformer",
    "LocationDensityTransformer",
]
