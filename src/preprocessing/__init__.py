"""
Member 1 — Preprocessing public API.

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
"""
# from pyspark.ml.pipeline import PipelineStage

from src.preprocessing.assemble import build_assembler_stage
from src.preprocessing.clean import clean, compute_class_weights , add_class_weights
from src.preprocessing.encode import build_encoding_stages_trees, build_encoding_stages_lr
from src.preprocessing.scale import build_scaling_stages_for_lr, build_scaling_stages_for_trees
