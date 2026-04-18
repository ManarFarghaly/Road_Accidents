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
from src.preprocessing.clean import clean, rebalance_undersample
from src.preprocessing.encode import build_encoding_stages
from src.preprocessing.scale import build_scaling_stages


def build_preprocessing_stages() -> list:
    """
    Returns the ordered list of unfit Spark ML stages that perform
    encoding + scaling + final assembly. Member 3 prepends these to
    their classifier inside a pyspark.ml.Pipeline.

    Note: cleaning is NOT a stage — call `clean(df)` on the raw
    DataFrame *before* constructing the Pipeline.
    """
    scale_stages, scaled_vec_col = build_scaling_stages()
    encode_stages, encoded_cols  = build_encoding_stages()
    assembler = build_assembler_stage(scaled_vec_col, encoded_cols)
    return scale_stages + encode_stages + [assembler]


__all__ = ["clean", "rebalance_undersample", "build_preprocessing_stages"]
