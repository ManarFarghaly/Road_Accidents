"""
Sub-task 4 — Feature Assembly.

Final VectorAssembler that concatenates:
    - the scaled-numeric vector (from scale.py)
    - one-hot vectors for low-cardinality categoricals (from encode.py)
    - integer indices for high-cardinality categoricals (from encode.py)

into a single `features` column. This is the exact column name the
modelling teammate (Member 3) reads from.
"""
from __future__ import annotations

from pyspark.ml.feature import VectorAssembler


def build_assembler_stage(
    scaled_vec_col: str,
    encoded_cols: list[str],
) -> VectorAssembler:
    """
    Args:
        scaled_vec_col: name of the scaled-numeric vector column
                        (from build_scaling_stages()).
        encoded_cols:   ordered list of encoded categorical columns
                        (from build_encoding_stages()).

    Returns:
        A single VectorAssembler stage producing `features`.
    """
    return VectorAssembler(
        inputCols=[scaled_vec_col] + encoded_cols,
        outputCol="features",
        handleInvalid="skip",
    )
