from __future__ import annotations

from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler

from src.config import get_spark, MERGED_PARQUET, PROCESSED_DIR, INTERIM_DIR
from src.preprocessing.clean import clean, compute_class_weights, add_class_weights
from src.preprocessing.encode import build_encoding_stages_lr, build_encoding_stages_trees, fit_target_encoding, apply_target_encoding, fit_and_apply_target_encodings
from src.preprocessing.scale import build_scaling_stages_for_lr, build_scaling_stages_for_trees
from src.preprocessing.assemble import build_assembler_stage

MODEL_TYPES = {"lr", "trees"}


def build_preprocessing_stages(model_type: str = "trees") -> list:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type!r}")

    if model_type == "lr":
        encode_stages, encoded_cols = build_encoding_stages_lr()
        scale_stages, scaled_vec_col = build_scaling_stages_for_lr()
    else:
        encode_stages, encoded_cols = build_encoding_stages_trees()
        scale_stages, scaled_vec_col = build_scaling_stages_for_trees()

    assembler = build_assembler_stage(scaled_vec_col, encoded_cols)
    return encode_stages + scale_stages + [assembler]


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    spark = get_spark("preprocessing")
    spark.sparkContext.setLogLevel("WARN")

    # ── 1. Load full merged dataset ────────────────────────────────────────
    print(f"\n[1] Loading full dataset from {MERGED_PARQUET} ...")
    df = spark.read.parquet(str(MERGED_PARQUET))
    total = df.count()
    print(f"    Raw rows: {total:,}, columns: {len(df.columns)}")

    # ── 2. Clean ───────────────────────────────────────────────────────────
    print("\n[2] Cleaning full dataset ...")
    cleaned = clean(df)

    # ── CRITICAL: checkpoint after clean() ────────────────────────────────
    # clean() produces a deep plan (Imputer + Window join + select chain).
    # Writing to parquet and re-reading gives downstream steps a flat scan
    # with a single Project node — no StackOverflowError on split/fit/transform.
    CLEAN_CHECKPOINT = str(INTERIM_DIR / "cleaned.parquet")
    print(f"\n    Checkpointing cleaned data → {CLEAN_CHECKPOINT}")
    cleaned.write.mode("overwrite").parquet(CLEAN_CHECKPOINT)
    cleaned = spark.read.parquet(CLEAN_CHECKPOINT)
    # ── end checkpoint ─────────────────────────────────────────────────────

    cleaned_count = cleaned.count()
    dropped = total - cleaned_count
    print(f"    Cleaned rows: {cleaned_count:,}  (dropped {dropped:,} = {100*dropped/total:.1f}%)")

    print("\n    Label distribution (full cleaned):")
    cleaned.groupBy("Accident_Severity").count().orderBy("Accident_Severity").show()

    # ── 3. Train / Test split — after checkpoint, plan is flat ────────────
    print("\n[3] Splitting 80/20 train/test ...")
    train_raw, test = cleaned.randomSplit([0.8, 0.2], seed=42)

    # Checkpoint splits too — prevents the random split from being
    # re-executed independently for fit() and transform()
    TRAIN_RAW_PATH = str(INTERIM_DIR / "train_raw.parquet")
    TEST_RAW_PATH  = str(INTERIM_DIR / "test_raw.parquet")
    train_raw.write.mode("overwrite").parquet(TRAIN_RAW_PATH)
    test.write.mode("overwrite").parquet(TEST_RAW_PATH)
    train_raw = spark.read.parquet(TRAIN_RAW_PATH)
    test      = spark.read.parquet(TEST_RAW_PATH)

    train_count = train_raw.count()
    test_count  = test.count()
    print(f"    Train: {train_count:,}  |  Test: {test_count:,}")

    # ── 4. Rebalance training split only ──────────────────────────────────
    print("\n[4] Computing class weights on training split ...")
    weights = compute_class_weights(train_raw)
    print(f"    Weights: { {k: f'{v:.2f}' for k, v in weights.items()} }")
    train = add_class_weights(train_raw, weights)

    print("\n[4b] Fitting and applying target encoding ...")
    train, test = fit_and_apply_target_encodings(train, test, smoothing=10.0)
    print(f"    Added columns: {[c + '_te' for c in ['model', 'LSOA_of_Accident_Location']]}")

    # ── 5. Fit preprocessing pipeline on training data ONLY ───────────────
    print("\n[5] Fitting preprocessing pipeline on training data ...")
    stages = build_preprocessing_stages()
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(train)
    print(f"    Pipeline fitted with {len(stages)} stages.")
    # After model = pipeline.fit(train)
    assembler = [s for s in model.stages if isinstance(s, VectorAssembler)][0]
    print(f"\n    Assembler input cols ({len(assembler.getInputCols())}):")
    for c in assembler.getInputCols():
        print(f"      {c}")

    # ── 6. Transform ──────────────────────────────────────────────────────
    print("\n[6] Transforming train and test splits ...")
    train_out = model.transform(train)
    test_out  = model.transform(test)

    final_cols = ["features", "label", "classWeight"]
    train_final = train_out.select(
        [c for c in final_cols if c in train_out.columns]
    )
    test_final = test_out.select(
        [c for c in ["features", "label"] if c in test_out.columns]
        # classWeight is NOT added to test — correct: weights are training-only
    )

    # ── 7. Write processed output ──────────────────────────────────────────
    train_path = str(PROCESSED_DIR / "train.parquet")
    test_path  = str(PROCESSED_DIR / "test.parquet")

    print(f"\n[7] Writing processed data ...")
    train_final.write.mode("overwrite").parquet(train_path)
    print(f"    Train → {train_path}")
    test_final.write.mode("overwrite").parquet(test_path)
    print(f"    Test  → {test_path}")

    # ── 8. Sanity check ───────────────────────────────────────────────────
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
    print(f"    Null labels   — train: {null_label_train},     test: {null_label_test}")

    print("\n    Train label distribution:")
    tr.groupBy("label").count().orderBy("label").show()
    print("\n    Test label distribution:")
    te.groupBy("label").count().orderBy("label").show()

    print(f"\n[done] Preprocessing complete.")
    spark.stop()


if __name__ == "__main__":
    main()