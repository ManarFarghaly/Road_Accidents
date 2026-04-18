"""
Preprocessing smoke tests.

Run AFTER ingestion has produced data/interim/merged.parquet.

    python -m tests.test_preprocessing

Tests each of preprocessing steps individually, then end-to-end:
    1. clean()                     — no nulls in imputed columns, dropped cols gone
    2. build_encoding_stages()     — StringIndexer/OneHotEncoder produce expected cols
    3. build_scaling_stages()      — StandardScaler output has mean≈0, std≈1
    4. build_assembler_stage()     — final 'features' vector column exists, no nulls
    + end-to-end: full build_preprocessing_stages() → Pipeline.fit → transform
"""
from __future__ import annotations

import sys
from pyspark.ml import Pipeline
from pyspark.sql import functions as F

from src.config import get_spark, MERGED_PARQUET
from src.preprocessing import clean, build_preprocessing_stages
from src.preprocessing.clean import HIGH_MISSING_COLS, NUM_IMPUTE_COLS, CAT_IMPUTE_COLS
from src.preprocessing.encode import LOW_CARD_CATS, HIGH_CARD_CATS, build_encoding_stages
from src.preprocessing.scale import NUMERIC_COLS, build_scaling_stages
from src.preprocessing.assemble import build_assembler_stage


PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  — {detail}")


def main():
    global PASS, FAIL

    spark = get_spark("preprocessing-test")
    spark.sparkContext.setLogLevel("WARN")

    print(f"\nLoading 1% sample from {MERGED_PARQUET} ...")
    raw = spark.read.parquet(str(MERGED_PARQUET)).sample(0.01, seed=42)
    raw.cache()
    raw_count = raw.count()
    raw_cols  = set(raw.columns)
    print(f"  Sample: {raw_count:,} rows, {len(raw_cols)} columns\n")

    # ──────────────────────────────────────────────────────────────────────
    # TEST 1: clean()
    # ──────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("TEST 1 — clean()")
    print("=" * 60)

    cleaned = clean(raw)
    cleaned.cache()
    cleaned_count = cleaned.count()
    cleaned_cols  = set(cleaned.columns)

    # 1a — high-missing columns should be gone
    for c in HIGH_MISSING_COLS:
        check(f"dropped column '{c}'", c not in cleaned_cols, f"'{c}' still in schema")

    # 1b/1d — no nulls in imputed numeric columns
    for c in NUM_IMPUTE_COLS:
        if c in cleaned_cols:
            n_null = cleaned.filter(F.col(c).isNull()).count()
            check(f"no nulls in numeric '{c}'", n_null == 0, f"{n_null:,} nulls remain")

    # 1b/1d — no nulls in imputed categorical columns
    for c in CAT_IMPUTE_COLS:
        if c in cleaned_cols:
            n_null = cleaned.filter(F.col(c).isNull()).count()
            check(f"no nulls in categorical '{c}'", n_null == 0, f"{n_null:,} nulls remain")

    # 1c — no nulls in target
    n_null_target = cleaned.filter(F.col("Accident_Severity").isNull()).count()
    check("no nulls in Accident_Severity", n_null_target == 0, f"{n_null_target} nulls")

    # Row count should not have dropped drastically
    drop_pct = 100 * (1 - cleaned_count / max(raw_count, 1))
    check(f"row loss < 5% (lost {drop_pct:.1f}%)", drop_pct < 5, f"lost {drop_pct:.1f}%")

    print(f"\n  cleaned: {cleaned_count:,} rows, {len(cleaned_cols)} columns\n")

    # ──────────────────────────────────────────────────────────────────────
    # TEST 2: build_encoding_stages()
    # ──────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("TEST 2 — build_encoding_stages()")
    print("=" * 60)

    encode_stages, encoded_cols = build_encoding_stages()

    # Should have stages for low-card (2 per col: indexer+OHE) + high-card (1 per col) + 1 label
    expected_n_stages = len(LOW_CARD_CATS) * 2 + len(HIGH_CARD_CATS) + 1
    check(
        f"stage count = {expected_n_stages}",
        len(encode_stages) == expected_n_stages,
        f"got {len(encode_stages)}",
    )

    # Fit the encoding stages alone to check output columns
    encode_pipeline = Pipeline(stages=encode_stages)
    encode_model = encode_pipeline.fit(cleaned)
    encoded_df = encode_model.transform(cleaned)
    encoded_out_cols = set(encoded_df.columns)

    for c in LOW_CARD_CATS:
        ohe_col = f"{c}_ohe"
        check(f"OHE column '{ohe_col}' exists", ohe_col in encoded_out_cols, "missing")

    for c in HIGH_CARD_CATS:
        idx_col = f"{c}_idx"
        check(f"index column '{idx_col}' exists", idx_col in encoded_out_cols, "missing")

    check("label column exists", "label" in encoded_out_cols, "missing")

    # Label should have exactly 3 distinct values
    n_labels = encoded_df.select("label").distinct().count()
    check(f"label has 3 classes", n_labels == 3, f"got {n_labels}")

    print()

    # ──────────────────────────────────────────────────────────────────────
    # TEST 3: build_scaling_stages()
    # ──────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("TEST 3 — build_scaling_stages()")
    print("=" * 60)

    scale_stages, scaled_vec_col = build_scaling_stages()
    check("scaled vector col name", scaled_vec_col == "numeric_scaled", f"got '{scaled_vec_col}'")
    check("2 stages (assembler + scaler)", len(scale_stages) == 2, f"got {len(scale_stages)}")

    # Fit the scaling stages alone
    scale_pipeline = Pipeline(stages=scale_stages)
    scale_model = scale_pipeline.fit(cleaned)
    scaled_df = scale_model.transform(cleaned)

    check("'numeric_scaled' column exists", "numeric_scaled" in scaled_df.columns, "missing")

    # Check mean ≈ 0 and std ≈ 1
    if "numeric_scaled" in scaled_df.columns:
        from pyspark.ml.stat import Summarizer
        stats_row = scaled_df.select(
            Summarizer.metrics("mean", "std").summary(F.col("numeric_scaled"))
        ).head()

        if stats_row:
            summary = stats_row[0]
            mean_vec = summary.mean
            std_vec  = summary.std
            max_mean_dev = max(abs(v) for v in mean_vec)
            min_std = min(std_vec)
            max_std = max(std_vec)
            check(
                f"mean ≈ 0 (max deviation: {max_mean_dev:.4f})",
                max_mean_dev < 0.1,
                f"max mean deviation = {max_mean_dev:.4f}",
            )
            check(
                f"std ≈ 1 (range: {min_std:.4f} – {max_std:.4f})",
                0.5 < min_std and max_std < 1.5,
                f"std range = {min_std:.4f} – {max_std:.4f}",
            )

    print()

    # ──────────────────────────────────────────────────────────────────────
    # TEST 4: Full end-to-end pipeline
    # ──────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("TEST 4 — full build_preprocessing_stages() end-to-end")
    print("=" * 60)

    all_stages = build_preprocessing_stages()
    check(f"total stages > 0", len(all_stages) > 0, f"got {len(all_stages)}")

    full_pipeline = Pipeline(stages=all_stages)
    full_model = full_pipeline.fit(cleaned)
    out = full_model.transform(cleaned)

    check("'features' column exists", "features" in out.columns, "missing")
    check("'label' column exists", "label" in out.columns, "missing")

    if "features" in out.columns:
        n_null_features = out.filter(F.col("features").isNull()).count()
        check("no nulls in features", n_null_features == 0, f"{n_null_features} nulls")

    if "label" in out.columns:
        print("\n  Label distribution:")
        out.groupBy("label").count().orderBy("label").show()

    if "features" in out.columns:
        print("  Sample output (3 rows):")
        out.select("features", "label").show(3, truncate=80)

    # ──────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"RESULTS:  {PASS} passed,  {FAIL} failed")
    print("=" * 60)

    spark.stop()
    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    main()
