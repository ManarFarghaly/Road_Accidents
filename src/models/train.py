"""
Member 3 — Model Training.

Three trainers, each returning a fitted PipelineModel:

    train_logistic_regression() — Baseline: multinomial LR with 5-fold CV
    train_random_forest()       — Advanced: RF classifier with 5-fold CV
    train_gbt()                 — Advanced: GBT via OneVsRest (binary GBT per class)

All classifiers receive a per-sample class weight to counteract the heavy class
imbalance (Slight 85 % / Serious 14 % / Fatal 1 %).
"""
from __future__ import annotations
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression, OneVsRest,RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.models.pipeline import build_model_pipeline

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_class_weights(df: DataFrame) -> dict[str, float]:
    """
    Balanced class weights:  class_weight = total / (n_classes * class_count)

    Uses Accident_Severity (the raw string column that exists before the
    preprocessing Pipeline runs its StringIndexer to create 'label').

    A rare class (Fatal) gets a high weight so the model does not ignore it.
    A dominant class (Slight) gets a low weight to prevent it from drowning
    out the minority signal during gradient updates.
    """
    total = df.count()
    n_classes = 3
    rows = df.groupBy("Accident_Severity").count().collect()

    weights: dict[str, float] = {}
    print("[weights] Class distribution:")
    for r in rows:
        severity = r["Accident_Severity"]
        count = r["count"]
        pct = 100.0 * count / total
        class_weight = total / (n_classes * count)
        weights[severity] = class_weight
        print(f"          {severity:>8s}: {count:>9,} rows ({pct:5.1f}%)  →  weight = {class_weight:.4f}")

    return weights

def _add_weight_col(df: DataFrame, weights: dict[str, float]) -> DataFrame:
    """
    Appends a 'class_weight' column by looking up each row's Accident_Severity
    string in a Spark MapType column built from the weights dict.

    We key on Accident_Severity (not 'label') because this function is called
    before the Pipeline runs — 'label' does not exist yet at this point.

    F.create_map() expects a flat list of alternating key/value Column
    expressions:  [key0, val0, key1, val1, key2, val2, ...]
    We build that list explicitly so the structure is clear.
    """
    # Build alternating [F.lit(severity), F.lit(weight), ...] for create_map
    map_args = []
    for severity, weight in weights.items():
        map_args.append(F.lit(severity))  # key   — severity string e.g. "Fatal"
        map_args.append(F.lit(weight))    # value — the computed class weight

    # create_map turns those pairs into a Spark MapType column
    # e.g. {"Fatal": 166.23, "Serious": 11.93, "Slight": 0.34}
    severity_to_weight = F.create_map(*map_args)

    # Look up each row's severity to get its weight
    return df.withColumn("class_weight", severity_to_weight[F.col("Accident_Severity")])

def _make_evaluator() -> MulticlassClassificationEvaluator:
    return MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1",
    )

# ---------------------------------------------------------------------------
# Public trainers
# ---------------------------------------------------------------------------

def train_logistic_regression(
    train_df: DataFrame,
    preprocessing_stages: list,
    use_cv: bool = True,
    num_folds: int = 5,
) -> object:
    """
    Baseline model: Multinomial Logistic Regression.

    Hyperparameter grid (when use_cv=True):
        regParam         : [0.01, 0.1]
        elasticNetParam  : [0.0,  0.5]
        maxIter          : [50,   100]

    Returns:
        Fitted PipelineModel (or CrossValidatorModel wrapping it).
    """
    print("\n[LR] Building Logistic Regression pipeline ...")
    weights = _compute_class_weights(train_df)
    print(f"[LR] Class weights: { {k: round(v, 3) for k, v in weights.items()} }")
    weighted_train = _add_weight_col(train_df, weights)

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="class_weight",
        family="multinomial", # loss is Multinomial cross-entropy
        maxIter=100,
    )
    pipeline = build_model_pipeline(lr, preprocessing_stages)

    if not use_cv:
        print("[LR] Training without cross-validation ...")
        t0 = time.time()
        model = pipeline.fit(weighted_train)
        print(f"[LR] Done in {time.time() - t0:.1f}s")
        return model

    param_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.01, 0.1]) # regularisation to prevent overfitting: penalty to the loss function that punishes large weights
        .addGrid(lr.elasticNetParam, [0.0, 0.5])
        .addGrid(lr.maxIter, [50, 100])
        .build()
    )
    """
    L2 penalty (elasticNetParam = 0.0) — Ridge
    Penalizes the square of each weight.
    Every feature keeps some influence but gets shrunk toward zero.
    Good when many features are somewhat useful
    (which is our case — weather, speed, light conditions all matter a bit).

    L1 penalty (elasticNetParam = 1.0) — Lasso
    Penalizes the absolute value of each weight.
    Aggressively pushes useless feature weights all the way to exactly zero
    effectively removing them.
    Good for very high-dimensional sparse data.

    elasticNetParam = 0.5 — Elastic Net
    A 50/50 mix: shrinks all weights like L2,
    but also zeros out the truly irrelevant ones like L1.
    """
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=_make_evaluator(),
        numFolds=num_folds,
        seed=42,
    )
    print(f"[LR] Running {num_folds}-fold CV over {len(param_grid)} param combinations ...")
    t0 = time.time()
    model = cv.fit(weighted_train)
    print(f"[LR] Best weighted-F1 = {max(model.avgMetrics):.4f}  ({time.time() - t0:.1f}s)")
    return model

def train_random_forest(
    train_df: DataFrame,
    preprocessing_stages: list,
    use_cv: bool = True,
    num_folds: int = 5,
) -> object:
    """
    Advanced model: Random Forest Classifier.

    Hyperparameter grid (when use_cv=True):
        numTrees  : [50,  100]
        maxDepth  : [5,   10]

    Returns:
        Fitted PipelineModel (or CrossValidatorModel wrapping it).
    """
    print("\n[RF] Building Random Forest pipeline ...")
    weights = _compute_class_weights(train_df)
    print(f"[RF] Class weights: { {k: round(v, 3) for k, v in weights.items()} }")
    weighted_train = _add_weight_col(train_df, weights)

    # loss is Gini impurity (per split)
    # Gini = 1 - (p_Fatal² + p_Serious² + p_Slight²)
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="class_weight",
        numTrees=100,
        maxDepth=10,
        seed=42,
    )
    """
    Bootstrapping then Aggregating
    """
    
    pipeline = build_model_pipeline(rf, preprocessing_stages)

    if not use_cv:
        print("[RF] Training without cross-validation ...")
        t0 = time.time()
        model = pipeline.fit(weighted_train)
        print(f"[RF] Done in {time.time() - t0:.1f}s")
        return model

    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [50, 100])
        .addGrid(rf.maxDepth, [5, 10])
        .build()
    )
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=_make_evaluator(),
        numFolds=num_folds,
        seed=42,
    )
    print(f"[RF] Running {num_folds}-fold CV over {len(param_grid)} param combinations ...")
    t0 = time.time()
    model = cv.fit(weighted_train)
    print(f"[RF] Best weighted-F1 = {max(model.avgMetrics):.4f}  ({time.time() - t0:.1f}s)")
    return model

def train_gbt(
    train_df: DataFrame,
    preprocessing_stages: list,
    use_cv: bool = False,
    num_folds: int = 3,
) -> object:
    """
    Advanced model: Gradient-Boosted Trees via OneVsRest.

    PySpark's GBTClassifier is binary-only, so it is wrapped in OneVsRest
    which trains one binary GBT per class (3 models internally).

    Class imbalance is handled by downsampling the majority class inside
    each binary subproblem via sampleWeights (not supported directly in OvR),
    so we instead undersample Slight before fitting to avoid extreme imbalance.

    Hyperparameter grid (when use_cv=True, 3-fold only due to compute cost):
        maxIter  : [30, 50]
        maxDepth : [4,  6]

    Returns:
        Fitted PipelineModel (or CrossValidatorModel wrapping it).
    """
    print("\n[GBT] Building Gradient-Boosted Trees (OneVsRest) pipeline ...")

    # Gentle undersampling: keep all Fatal + Serious, sample 35 % of Slight
    fatal_serious = train_df.filter(F.col("Accident_Severity") != "Slight")
    slight = train_df.filter(F.col("Accident_Severity") == "Slight").sample(fraction=0.35, seed=42)
    balanced_train = fatal_serious.union(slight)
    print(f"[GBT] Balanced training rows: {balanced_train.count():,}")

    # Logistic loss (binary, per tree)
    # loss = -[ y × log(p) + (1-y) × log(1-p) ]

    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        maxIter=50,
        maxDepth=5,
        stepSize=0.1,
        seed=42,
    )
    ovr = OneVsRest(classifier=gbt, featuresCol="features", labelCol="label")
    pipeline = build_model_pipeline(ovr, preprocessing_stages)

    if not use_cv:
        print("[GBT] Training without cross-validation ...")
        t0 = time.time()
        model = pipeline.fit(balanced_train)
        print(f"[GBT] Done in {time.time() - t0:.1f}s")
        return model

    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxIter, [30, 50])
        .addGrid(gbt.maxDepth, [4, 6])
        .build()
    )
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=_make_evaluator(),
        numFolds=num_folds,
        seed=42,
    )
    print(f"[GBT] Running {num_folds}-fold CV over {len(param_grid)} param combinations ...")
    t0 = time.time()
    model = cv.fit(balanced_train)
    print(f"[GBT] Best weighted-F1 = {max(model.avgMetrics):.4f}  ({time.time() - t0:.1f}s)")
    return model
