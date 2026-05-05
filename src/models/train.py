from __future__ import annotations
from pyspark.ml.classification import GBTClassifier, LogisticRegression, OneVsRest,RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from src.models.pipeline import build_model_pipeline
from src.preprocessing.clean import compute_class_weights, add_class_weights

def _make_evaluator() -> MulticlassClassificationEvaluator:
    return MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1",
    )

def train_logistic_regression(
    train_df: DataFrame,
    preprocessing_stages: list,
    use_cv: bool = True,
    num_folds: int = 5,
) -> object:
    
    weights = compute_class_weights(train_df)
    weighted_train = add_class_weights(train_df, weights, weight_col="class_weight")

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="class_weight",
        family="multinomial",
        maxIter=100,
    )
    pipeline = build_model_pipeline(lr, preprocessing_stages)

    if not use_cv:
        model = pipeline.fit(weighted_train)
        return model

    param_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.01, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.5])
        .addGrid(lr.maxIter, [50, 100])
        .build()
    )
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=_make_evaluator(),
        numFolds=num_folds,
        seed=42,
    )
    model = cv.fit(weighted_train)
    return model

def train_random_forest(
    train_df: DataFrame,
    preprocessing_stages: list,
    use_cv: bool = True,
    num_folds: int = 5,
) -> object:
    weights = compute_class_weights(train_df)
    weighted_train = add_class_weights(train_df, weights, weight_col="class_weight")

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="class_weight",
        numTrees=50,
        maxDepth=7,
        maxBins=512,
        featureSubsetStrategy="sqrt",
        subsamplingRate=0.8,
        seed=42,
    )
    pipeline = build_model_pipeline(rf, preprocessing_stages)

    if not use_cv:
        model = pipeline.fit(weighted_train)
        return model

    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [25, 50])
        .addGrid(rf.maxDepth, [5, 7])
        .build()
    )
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=_make_evaluator(),
        numFolds=num_folds,
        seed=42,
    )
    model = cv.fit(weighted_train)
    return model

def train_gbt(
    train_df: DataFrame,
    preprocessing_stages: list,
    use_cv: bool = False,
    num_folds: int = 3,
) -> object:
    fatal_serious = train_df.filter(F.col("Accident_Severity") != "Slight")
    slight = train_df.filter(F.col("Accident_Severity") == "Slight").sample(fraction=0.5, seed=42)
    balanced_train = fatal_serious.union(slight)

    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        maxIter=15,
        maxDepth=3,
        maxBins=512,
        stepSize=0.1,
        seed=42,
    )
    ovr = OneVsRest(classifier=gbt, featuresCol="features", labelCol="label")
    pipeline = build_model_pipeline(ovr, preprocessing_stages)

    if not use_cv:
        model = pipeline.fit(balanced_train)
        return model

    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxIter, [10, 15])
        .addGrid(gbt.maxDepth, [3, 4])
        .build()
    )
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=_make_evaluator(),
        numFolds=num_folds,
        seed=42,
    )
    model = cv.fit(balanced_train)
    return model
