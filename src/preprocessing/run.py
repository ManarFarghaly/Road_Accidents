"""
Preprocessing Pipeline Runner.

Reads the full merged.parquet, cleans it, splits into train/test,
rebalances the training split only, fits the ML pipeline on training data,
transforms both splits, and writes the results to processed/.

Run:
    python -m src.preprocessing.run
"""
from __future__ import annotations

import os
from pyspark.ml import Pipeline
from pyspark.sql import functions as F

from src.config import get_spark, MERGED_PARQUET, PROCESSED_DIR
from src.preprocessing.clean import clean, rebalance_undersample
from src.preprocessing.encode import build_encoding_stages
from src.preprocessing.scale import build_scaling_stages
from src.preprocessing.assemble import build_assembler_stage


def build_preprocessing_stages():
    encode_stages, encoded_cols = build_encoding_stages()
    scale_stages, scaled_vec_col = build_scaling_stages()
    assembler = build_assembler_stage(scaled_vec_col, encoded_cols)
    return encode_stages + scale_stages + [assembler]


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    spark = get_spark("preprocessing")
    spark.sparkContext.setLogLevel("WARN")

    # ── 1. Load full merged dataset ────────────────────────────────────
    print(f"\n[1] Loading full dataset from {MERGED_PARQUET} ...")
    df = spark.read.parquet(str(MERGED_PARQUET))
    total = df.count()
    print(f"    Raw rows: {total:,}, columns: {len(df.columns)}")

    # ── 2. Clean the entire dataset ────────────────────────────────────
    print("\n[2] Cleaning full dataset ...")
    cleaned = clean(df)
    cleaned_count = cleaned.count()
    dropped = total - cleaned_count
    print(f"    Cleaned rows: {cleaned_count:,}  (dropped {dropped:,} = {100*dropped/total:.1f}%)")

    # Label distribution before split
    print("\n    Label distribution (full cleaned):")
    cleaned.groupBy("Accident_Severity").count() \
           .orderBy("Accident_Severity").show()

    # ── 3. Train / Test split — split BEFORE rebalancing ──────────────
    print("\n[3] Splitting 80/20 train/test ...")
    train_raw, test = cleaned.randomSplit([0.8, 0.2], seed=42)
    train_count = train_raw.count()
    test_count  = test.count()
    print(f"    Train: {train_count:,}  |  Test: {test_count:,}")

    # ── 4. Rebalance training split only ──────────────────────────────
    print("\n[4] Rebalancing training split (undersample majority) ...")
    train = rebalance_undersample(train_raw, ratio=3.0, seed=42)
    train_balanced_count = train.count()
    print(f"    Balanced train rows: {train_balanced_count:,}")
    print("\n    Label distribution after rebalancing (train only):")
    train.groupBy("Accident_Severity").count() \
         .orderBy("Accident_Severity").show()

    # ── 5. Fit preprocessing pipeline on training data ONLY ───────────
    print("\n[5] Fitting preprocessing pipeline on training data ...")
    stages = build_preprocessing_stages()
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(train)
    print(f"    Pipeline fitted with {len(stages)} stages.")

    # ── 6. Transform both splits ───────────────────────────────────────
    print("\n[6] Transforming train and test splits ...")
    train_out = model.transform(train)
    test_out  = model.transform(test)

    # Keep only what the model needs — avoids writing 64 raw cols to disk
    final_cols = ["features", "label"]
    train_final = train_out.select(final_cols)
    test_final  = test_out.select(final_cols)

    # ── 7. Write to processed/ ─────────────────────────────────────────
    train_path = str(PROCESSED_DIR / "train.parquet")
    test_path  = str(PROCESSED_DIR / "test.parquet")

    print(f"\n[7] Writing processed data ...")
    print(f"    Train → {train_path}")
    train_final.write.mode("overwrite").parquet(train_path)

    print(f"    Test  → {test_path}")
    test_final.write.mode("overwrite").parquet(test_path)

    # ── 8. Sanity check ───────────────────────────────────────────────
    print("\n[8] Sanity check on written data ...")
    tr = spark.read.parquet(train_path)
    te = spark.read.parquet(test_path)

    print(f"    Train parquet rows: {tr.count():,}, cols: {len(tr.columns)}")
    print(f"    Test  parquet rows: {te.count():,}, cols: {len(te.columns)}")

    null_features_train = tr.filter(F.col("features").isNull()).count()
    null_features_test  = te.filter(F.col("features").isNull()).count()
    null_label_train    = tr.filter(F.col("label").isNull()).count()
    null_label_test     = te.filter(F.col("label").isNull()).count()

    print(f"\n    Null features — train: {null_features_train}, test: {null_features_test}")
    print(f"    Null labels   — train: {null_label_train},    test: {null_label_test}")

    print("\n    Train label distribution:")
    tr.groupBy("label").count().orderBy("label").show()

    print("\n    Test label distribution:")
    te.groupBy("label").count().orderBy("label").show()

    print(f"\n[done] Preprocessing complete.")
    print(f"       Train: {PROCESSED_DIR}/train.parquet")
    print(f"       Test:  {PROCESSED_DIR}/test.parquet")

    spark.stop()


if __name__ == "__main__":
    main()