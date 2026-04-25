"""
Member 3 — Models package public API.

Quick-start for Member 4 (Evaluation):

    from src.models import load_model
    from src.config import get_spark

    spark = get_spark("evaluation")
    test  = spark.read.parquet("data/processed/test.parquet")

    lr_model  = load_model("models/lr")
    rf_model  = load_model("models/rf")
    gbt_model = load_model("models/gbt")

    lr_preds  = lr_model.transform(test)   # adds 'prediction', 'probability'
    rf_preds  = rf_model.transform(test)
    gbt_preds = gbt_model.transform(test)
"""

from src.models.pipeline import build_model_pipeline
from src.models.save_load import load_model, save_model
from src.models.train import train_gbt, train_logistic_regression, train_random_forest

__all__ = [
    "build_model_pipeline",
    "train_logistic_regression",
    "train_random_forest",
    "train_gbt",
    "save_model",
    "load_model",
]
