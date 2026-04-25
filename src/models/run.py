"""
Member 3 — Training entry point.

Trains all three models end-to-end and saves them under models/.
Run from the project root:

    python -m src.models.run

Flags (edit at the top of the file or pass via environment variables):
    USE_CV_LR  — cross-validate Logistic Regression   (slow, ~30-60 min)
    USE_CV_RF  — cross-validate Random Forest          (very slow, ~2-4 h)
    USE_CV_GBT — cross-validate GBT                   (extremely slow, off by default)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from src.config import MERGED_PARQUET, PROJECT_ROOT, get_spark
from src.preprocessing import build_preprocessing_stages, clean
from src.models.train import train_gbt, train_logistic_regression, train_random_forest
from src.models.save_load import save_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "models"

USE_CV_LR  = os.getenv("USE_CV_LR",  "true").lower()  == "true"
USE_CV_RF  = os.getenv("USE_CV_RF",  "true").lower()  == "true"
USE_CV_GBT = os.getenv("USE_CV_GBT", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    spark = get_spark("road-accidents-training")
    wall_start = time.time()

    # ------------------------------------------------------------------
    # 1. Load & clean raw data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading merged.parquet ...")
    raw_df = spark.read.parquet(str(MERGED_PARQUET))

    print("Applying eager cleaning ...")
    cleaned_df = clean(raw_df)

    # ------------------------------------------------------------------
    # 2. Train / test split  (consistent seed with Member 1's run.py)
    # ------------------------------------------------------------------
    train_df, test_df = cleaned_df.randomSplit([0.8, 0.2], seed=42)

    # Cache training data — reused by all three trainers
    train_df.cache()
    train_count = train_df.count()
    test_count  = test_df.count()
    print(f"Split: {train_count:,} train  /  {test_count:,} test rows")

    # ------------------------------------------------------------------
    # 3. Shared preprocessing stages (fit inside each Pipeline)
    # ------------------------------------------------------------------
    preprocessing_stages = build_preprocessing_stages()

    # ------------------------------------------------------------------
    # 4. Train Logistic Regression (Baseline)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL 1/3 — Logistic Regression (Baseline)")
    print("=" * 60)
    lr_model = train_logistic_regression(
        train_df, preprocessing_stages, use_cv=USE_CV_LR
    )
    save_model(lr_model, MODELS_DIR / "lr")

    # ------------------------------------------------------------------
    # 5. Train Random Forest (Advanced)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL 2/3 — Random Forest (Advanced)")
    print("=" * 60)
    rf_model = train_random_forest(
        train_df, preprocessing_stages, use_cv=USE_CV_RF
    )
    save_model(rf_model, MODELS_DIR / "rf")

    # ------------------------------------------------------------------
    # 6. Train Gradient-Boosted Trees (Advanced)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL 3/3 — Gradient-Boosted Trees via OneVsRest (Advanced)")
    print("=" * 60)
    gbt_model = train_gbt(
        train_df, preprocessing_stages, use_cv=USE_CV_GBT
    )
    save_model(gbt_model, MODELS_DIR / "gbt")

    # ------------------------------------------------------------------
    # 7. Persist test split for Member 4
    # ------------------------------------------------------------------
    test_path = Path("data/processed/test.parquet")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.write.mode("overwrite").parquet(str(test_path))
    print(f"\n[run] Test set saved → {test_path}")

    train_df.unpersist()

    total = time.time() - wall_start
    print("\n" + "=" * 60)
    print(f"All three models trained and saved in {total / 60:.1f} minutes.")
    print(f"Model directories:")
    for name in ("lr", "rf", "gbt"):
        print(f"  models/{name}/")
    print("=" * 60)

    spark.stop()

if __name__ == "__main__":
    main()
