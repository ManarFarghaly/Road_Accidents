"""
Sub-task 1 — Data Cleaning.

Eager cleaning function that runs on the raw merged DataFrame
*before* the Spark ML Pipeline is built.

Actions (in order):
    1a  Drop 5 columns with >85% missing (per validation report).
    1b  Null out UK DfT sentinel values ('Data missing or out of range',
        'Unknown', numeric -1) so StringIndexer won't treat them as real
        categories downstream.
    1c  Drop rows missing the target (Accident_Severity).
    1d  Median-impute numerics; 'Unknown' fill for categoricals.
    1e  (Documented non-action) — do not winsorize outliers; they are
        legitimate and tree models handle them.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, functions as F
from pyspark.ml.feature import Imputer


# ── Columns with >85% missing (per reports/validation_report.json) ────────
HIGH_MISSING_COLS = [
    "Carriageway_Hazards",            # 98.07 %
    "Special_Conditions_at_Site",     # 97.45 %
    "Hit_Object_in_Carriageway",      # 95.89 %
    "Hit_Object_off_Carriageway",     # 91.39 %
    "Skidding_and_Overturning",       # 87.19 %
]

# UK DfT sentinel string values meaning "missing"
SENTINEL_STRINGS = [
    "Data missing or out of range",
    "Unknown",
    "unknown",
    "N/A",
    "",
]

# Numerics to median-impute (and null-out where they hold -1 sentinels)
NUM_IMPUTE_COLS = [
    "Age_of_Vehicle",
    "Engine_Capacity_.CC.",
    "Driver_IMD_Decile",
    # Weather features added in ingestion stage 3e. Some rows will be null
    # for stations that had no data on the accident's date — median-impute.
    "tavg",
    "tmin",
    "tmax",
    "prcp",
    "snow",
    "wspd",
    "pres",
]

# Categoricals to fill with the literal string "Unknown"
CAT_IMPUTE_COLS = [
    "2nd_Road_Class",
    "LSOA_of_Accident_Location",
    "Weather_Conditions",
    "Road_Surface_Conditions",
    "Light_Conditions",
    "Junction_Detail",
    "Junction_Control",
    "Road_Type",
    "Vehicle_Type",
    "make",
]

LABEL_COL = "Accident_Severity"


def clean(df: DataFrame) -> DataFrame:
    """
    Run the full cleaning pipeline on the raw merged DataFrame and
    return the cleaned DataFrame. Called once by Member 3 before
    constructing their Spark ML Pipeline.
    """
    # 1a — drop hopelessly-empty columns
    to_drop = [c for c in HIGH_MISSING_COLS if c in df.columns]
    if to_drop:
        df = df.drop(*to_drop)

    # 1b — null out sentinel strings across every string column
    str_cols = [
        f.name for f in df.schema.fields
        if f.dataType.simpleString() == "string"
    ]
    for c in str_cols:
        df = df.withColumn(
            c,
            F.when(F.col(c).isin(SENTINEL_STRINGS), None).otherwise(F.col(c)),
        )

    # 1b(ii) — numeric -1 "unknown" sentinel → NULL
    for c in NUM_IMPUTE_COLS:
        if c in df.columns:
            df = df.withColumn(
                c,
                F.when(F.col(c) < 0, None).otherwise(F.col(c)),
            )

    # 1c — drop rows missing the target
    df = df.filter(F.col(LABEL_COL).isNotNull())

    # 1d(i) — numeric median imputation
    num_present = [c for c in NUM_IMPUTE_COLS if c in df.columns]
    if num_present:
        imputer = Imputer(
            inputCols=num_present,
            outputCols=num_present,      # overwrite in place
            strategy="median",
        )
        df = imputer.fit(df).transform(df)

    # 1d(ii) — categorical "Unknown" fill
    cat_present = [c for c in CAT_IMPUTE_COLS if c in df.columns]
    if cat_present:
        df = df.fillna("Unknown", subset=cat_present)

    return df
