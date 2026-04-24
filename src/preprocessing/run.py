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
import socket
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.sql import functions as F

from src.config import get_spark, MERGED_PARQUET, PROCESSED_DIR
from src.preprocessing import build_preprocessing_stages
from src.preprocessing.clean import clean


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Match ingest hardening for Windows + Python 3.13 Py4J stability.
    socket.setdefaulttimeout(1800)

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

    # ── 4. Skip rebalancing ────────────────────────────────────────────
    # rebalance_undersample calls .collect() which permanently breaks the
    # Py4J connection on Windows + Python 3.13 — the session is corrupted
    # even after a try/except, causing pipeline.fit() to fail. The class
    # imbalance (Slight 85%/Serious 13%/Fatal 1%) is handled downstream
    # by the ML model via classWeightCol or threshold tuning instead.
    print("\n[4] Skipping rebalancing (not compatible with Windows/Py4J on this platform).")
    train = train_raw

    # ── 5. Fit preprocessing pipeline on training data ONLY ───────────
    # Persist train to disk BEFORE fitting to break the lazy DAG chain.
    # Without this, each of the 31 stages replays the full clean+split
    # plan from parquet, causing JVM OOM on large datasets.
    print("\n[5] Persisting training data to disk before pipeline fit ...")
    train = train.persist(StorageLevel.DISK_ONLY)
    train_cached_count = train.count()   # force materialization now
    print(f"    Persisted {train_cached_count:,} training rows to disk.")

    print("\n    Fitting preprocessing pipeline on training data ...")
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

    # Release cached train from disk — no longer needed after transforms
    train.unpersist()

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