from __future__ import annotations
import os
import time
from pathlib import Path
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from src.config import MERGED_PARQUET, PROJECT_ROOT, get_spark
from src.preprocessing import build_preprocessing_stages, clean
from src.preprocessing.run import fit_and_apply_target_encodings
from src.models.train import train_gbt, train_logistic_regression, train_random_forest
from src.models.save_load import save_model
from src.models.evaluate import evaluate_model, save_metrics_json, make_results_container


MODELS_DIR         = PROJECT_ROOT / "models"
REPORTS_DIR        = PROJECT_ROOT / "reports"
METRICS_PATH       = MODELS_DIR   / "metrics.txt"
MODEL_METRICS_JSON = REPORTS_DIR  / "model_metrics.json"

USE_CV_LR  = os.getenv("USE_CV_LR",  "true").lower()  == "true"
USE_CV_RF  = os.getenv("USE_CV_RF",  "true").lower()  == "true"
USE_CV_GBT = os.getenv("USE_CV_GBT", "false").lower() == "true"


def _write_metrics_header(metrics_path: Path) -> None:
    metrics_path.write_text(
        "Road Accidents model evaluation results\n"
        "=======================================\n\n",
        encoding="utf-8",
    )


def _evaluate_and_log(model_name: str, model, test_df, metrics_path: Path) -> None:
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy",
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1",
    )

    predictions = model.transform(test_df)
    accuracy = evaluator_acc.evaluate(predictions)
    f1_score = evaluator_f1.evaluate(predictions)

    line = (
        f"[{model_name}] Test accuracy = {accuracy:.4f}, "
        f"weighted-F1 = {f1_score:.4f}\n"
    )

    with open(metrics_path, "a", encoding="utf-8") as out_f:
        out_f.write(line)

    print(line.strip())


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    spark = get_spark("road-accidents-training")
    wall_start = time.time()
    _write_metrics_header(METRICS_PATH)

    # 1. Load & clean raw data
    raw_df = spark.read.parquet(str(MERGED_PARQUET))
    cleaned_df = clean(raw_df)

    # 2. Train / test split
    train_df, test_df = cleaned_df.randomSplit([0.8, 0.2], seed=42)
    train_df, test_df = fit_and_apply_target_encodings(train_df, test_df, smoothing=10.0)

    # Cache training data — reused by all three trainers
    train_df.cache()

    # 3. Shared preprocessing stages (fit inside each Pipeline)
    preprocessing_stages = build_preprocessing_stages()

    all_results = make_results_container()

    # # 4. Train Logistic Regression (Baseline)
    # lr_model = train_logistic_regression(
    #     train_df, preprocessing_stages, use_cv=USE_CV_LR
    # )
    # save_model(lr_model, MODELS_DIR / "lr")
    # _evaluate_and_log("Logistic Regression", lr_model, test_df, METRICS_PATH)
    # all_results["models"]["Logistic Regression"] = evaluate_model(
    #     "Logistic Regression", lr_model, train_df, test_df
    # )

    # # 5. Train Random Forest
    # rf_model = train_random_forest(
    #     train_df, preprocessing_stages, use_cv=USE_CV_RF
    # )
    # save_model(rf_model, MODELS_DIR / "rf")
    # _evaluate_and_log("Random Forest", rf_model, test_df, METRICS_PATH)
    # all_results["models"]["Random Forest"] = evaluate_model(
    #     "Random Forest", rf_model, train_df, test_df
    # )

    # 6. Train Gradient-Boosted Trees
    gbt_model = train_gbt(
        train_df, preprocessing_stages, use_cv=USE_CV_GBT
    )
    save_model(gbt_model, MODELS_DIR / "gbt")
    _evaluate_and_log("GBT", gbt_model, test_df, METRICS_PATH)
    all_results["models"]["GBT"] = evaluate_model("GBT", gbt_model, train_df, test_df)

    # 7. Save full metrics JSON for the dashboard
    save_metrics_json(all_results, MODEL_METRICS_JSON)

    # 8. Persist test split
    test_path = Path("data/processed/test.parquet")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.write.mode("overwrite").parquet(str(test_path))
    train_df.unpersist()
    spark.stop()

if __name__ == "__main__":
    main()