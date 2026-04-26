"""
Member 3 — Model Persistence.

save_model()  — writes any fitted pyspark.ml model/pipeline to disk.
load_model()  — reloads it as a PipelineModel for inference or evaluation.
"""
from __future__ import annotations

from pathlib import Path

from pyspark.ml import PipelineModel


def save_model(model, path: str | Path) -> None:
    """
    Persist a fitted PipelineModel (or CrossValidatorModel) to disk.

    CrossValidatorModel exposes .bestModel which is the fitted Pipeline;
    that inner model is what we save so Member 4 can reload with load_model().

    Arrow is temporarily disabled during save because PySpark's model writer
    uses spark.createDataFrame + write.text() internally, which crashes the
    Python worker when Arrow is enabled on Windows.

    Args:
        model: Fitted PipelineModel or CrossValidatorModel.
        path:  Destination directory.  Overwritten if it already exists.
    """
    save_path = str(path)
    actual_model = getattr(model, "bestModel", model)
    actual_model.write().overwrite().save(save_path)
    print(f"[save] Model saved → {save_path}")


def load_model(path: str | Path) -> PipelineModel:
    """
    Load a PipelineModel saved by save_model().

    Args:
        path: Directory written by save_model().

    Returns:
        Fitted PipelineModel ready for .transform(df).
    """
    loaded = PipelineModel.load(str(path))
    print(f"[load] Model loaded ← {path}")
    return loaded
