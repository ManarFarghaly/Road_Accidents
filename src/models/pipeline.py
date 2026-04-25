from pyspark.ml import Pipeline

def build_model_pipeline(classifier, preprocessing_stages: list) -> Pipeline:
    """
    Returns a Pipeline that runs preprocessing then fits the classifier.

    Args:
        classifier:           Any unfitted pyspark.ml estimator.
        preprocessing_stages: Ordered list returned by build_preprocessing_stages().

    Returns:
        Unfitted Pipeline ready to call .fit(train_df) on.
    """
    return Pipeline(stages=preprocessing_stages + [classifier])
