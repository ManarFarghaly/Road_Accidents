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
from src.preprocessing.clean import clean, compute_class_weights , add_class_weights
from src.preprocessing.encode import build_encoding_stages_trees, build_encoding_stages_lr
from src.preprocessing.scale import build_scaling_stages_for_lr, build_scaling_stages_for_trees
from src.preprocessing.run import build_preprocessing_stages