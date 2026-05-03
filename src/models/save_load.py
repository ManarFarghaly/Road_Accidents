from __future__ import annotations
from pathlib import Path
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession


def save_model(model, path: str | Path) -> None:
    save_path = str(path)
    actual_model = getattr(model, "bestModel", model)

    spark = SparkSession.getActiveSession()
    original_arrow = None
    original_worker_reuse = None
    if spark is not None:
        try:
            original_arrow = spark.conf.get("spark.sql.execution.arrow.pyspark.enabled")
        except Exception:
            original_arrow = None
        try:
            original_worker_reuse = spark.conf.get("spark.python.worker.reuse")
        except Exception:
            original_worker_reuse = None
        try:
            spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
        except Exception:
            pass
        try:
            spark.conf.set("spark.python.worker.reuse", "false")
        except Exception:
            pass

    try:
        actual_model.write().overwrite().save(save_path)
    finally:
        if spark is not None:
            if original_arrow is not None:
                try:
                    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", original_arrow)
                except Exception:
                    pass
            if original_worker_reuse is not None:
                try:
                    spark.conf.set("spark.python.worker.reuse", original_worker_reuse)
                except Exception:
                    pass


def load_model(path: str | Path) -> PipelineModel:
    loaded = PipelineModel.load(str(path))
    return loaded
