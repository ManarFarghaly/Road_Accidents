
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
        handleInvalid="error",    
    )
