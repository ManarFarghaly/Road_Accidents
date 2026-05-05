from __future__ import annotations

from functools import reduce

from pyspark.sql import DataFrame, Window, functions as F
from pyspark.ml.feature import Imputer

HIGH_MISSING_COLS = [
    "Carriageway_Hazards",
    "Special_Conditions_at_Site",
    "Hit_Object_in_Carriageway",
    "Hit_Object_off_Carriageway",
    "Skidding_and_Overturning",
    "2nd_Road_Class",
    "wpgt", "prcp", "snwd", "tsun", "cldc",
]

LEAKAGE_COLS = [
    "Number_of_Casualties",
    "Did_Police_Officer_Attend_Scene_of_Accident",
]

SENTINEL_STRINGS = [
    "Data missing or out of range",
    "Unknown", "unknown",
    "Null", "NULL", "[null]", "null",
]

VALID_AGE_BANDS = [
    "0 - 5", "6 - 10", "11 - 15", "16 - 20", "21 - 25",
    "26 - 35", "36 - 45", "46 - 55", "56 - 65", "66 - 75", "Over 75",
]

UK_LEGAL_SPEED_LIMITS = [20, 30, 40, 50, 60, 70]

REQUIRED_COLS = [
    "Accident_Severity", "Latitude", "Longitude",
    "Accident_Index", "Speed_limit","Time", "Date",
]

VALIDITY_BOUNDS = {
    "Latitude":           (49.0, 61.0),
    "Longitude":          (-8.0, 2.0),
    "Engine_Capacity_CC": (50.0, 8000.0),
    "Age_of_Vehicle":     (0.0, 100.0),
    "Number_of_Vehicles": (1.0, 100.0),
}

# Columns to cast to double (done in the single select pass)
NUMERIC_CAST_COLS = [
    "Speed_limit", "Number_of_Vehicles", "Latitude", "Longitude",
    "Age_of_Vehicle", "Engine_Capacity_CC", "Driver_IMD_Decile",
    "temp", "tmin", "tmax", "prcp", "snwd", "wspd", "pres",
    "rhum", "wpgt", "tsun", "cldc",
]

NUM_IMPUTE_COLS = [
    "Age_of_Vehicle",
    "Engine_Capacity_CC",
    "Driver_IMD_Decile",
    "Location_Easting_OSGR",
    "Location_Northing_OSGR",
    "Was_Vehicle_Left_Hand_Drive",
    "InScotland" ,
    "tmin", "tmax", "pres", "temp", "rhum", "wspd",
]

MODE_IMPUTE_COLS = [
    "Road_Type",
    "Weather_Conditions",
    "Light_Conditions",
    "Road_Surface_Conditions",
    "Sex_of_Driver",
    "Propulsion_Code",
    "Towing_and_Articulation",
    "Age_Band_of_Driver",
]

GROUP_MODE_IMPUTE = [
    ("model", "make"),
]

CAT_IMPUTE_COLS = [
    "1st_Road_Class", "Urban_or_Rural_Area",
    "Pedestrian_Crossing-Human_Control",
    "LSOA_of_Accident_Location",
    "Junction_Detail", "Junction_Control",
    "Road_Type", "Vehicle_Type", "make",
    "Junction_Location", "Vehicle_Leaving_Carriageway",
    "X1st_Point_of_Impact", "Journey_Purpose_of_Driver",
    "Driver_Home_Area_Type", "Vehicle_Manoeuvre",
    "2nd_Road_Number", "1st_Road_Number",
    "Local_Authority_(District)", "Local_Authority_(Highway)", "Police_Force","Vehicle_Reference"
]

LABEL_COL = "Accident_Severity"


def _fill_mode_by_group(df: DataFrame, target: str, group: str) -> DataFrame:
    if target not in df.columns or group not in df.columns:
        return df

    grp_mode = (
        df.filter(F.col(target).isNotNull())
          .groupBy(group, target)
          .count()
          .withColumn(
              "rk",
              F.row_number().over(
                  Window.partitionBy(group).orderBy(F.col("count").desc())
              ),
          )
          .filter(F.col("rk") == 1)
          .select(group, F.col(target).alias(f"{target}_grp_mode"))
    )

    global_mode_row = (
        df.filter(F.col(target).isNotNull())
          .groupBy(target).count()
          .orderBy(F.col("count").desc())
          .select(target).head()
    )
    global_mode = global_mode_row[0] if global_mode_row else "Unknown"

    return (
        df.join(F.broadcast(grp_mode), on=group, how="left")
          .withColumn(
              target,
              F.coalesce(F.col(target), F.col(f"{target}_grp_mode"), F.lit(global_mode)),
          )
          .drop(f"{target}_grp_mode")
    )


def _build_select_exprs(df: DataFrame) -> list:
    schema_map = {f.name: f.dataType.simpleString() for f in df.schema.fields}
    exprs = []

    skip = set(HIGH_MISSING_COLS + LEAKAGE_COLS)

    rename_map = {"Engine_Capacity_.CC.": "Engine_Capacity_CC"
    , "Vehicle_Location.Restricted_Lane": "Vehicle_Location_Restricted_Lane"
    }
   

    for raw_name in df.columns:
        if raw_name in skip:
            continue

        out_name = rename_map.get(raw_name, raw_name)
        dtype    = schema_map.get(raw_name, "string")
        c        = F.col(f"`{raw_name}`")

        if out_name in NUMERIC_CAST_COLS:
            c = c.cast("double")
            dtype = "double"

        if out_name == "InScotland":
            c = (F.when(c == "Yes", 1.0)
                  .when(c == "No",  0.0)
                  .otherwise(None))
            exprs.append(c.alias(out_name))
            continue
        if out_name == "Was_Vehicle_Left_Hand_Drive":
            c = (F.when(c.isin("2", "Yes", "yes"), 1.0)
                  .when(c.isin("1", "No",  "no"),  0.0)
                  .otherwise(None))
            exprs.append(c.alias(out_name))
            continue

        if dtype == "string":
            c = F.when(c.isin(SENTINEL_STRINGS), None).otherwise(c)

        if out_name == "Age_Band_of_Driver":
            c = F.when(c.isin(VALID_AGE_BANDS), c).otherwise(None)

        if out_name == "Sex_of_Driver":
            c = (F.when(c.isin(["M", "1"]), "Male")
                  .when(c.isin(["F", "2"]), "Female")
                  .when(c == "3",           "Not known")
                  .otherwise(c))

        if out_name == "Day_of_Week":
            c = F.date_format(F.col("`Date`"), "EEEE")

        if out_name in NUM_IMPUTE_COLS and dtype in ("double", "float"):
            c = F.when(c < 0, None).otherwise(c)

        if out_name == "Speed_limit":
            legal = c.isin(UK_LEGAL_SPEED_LIMITS)
            snapped = (
                F.when(c <= 25, 20).when(c <= 35, 30)
                 .when(c <= 45, 40).when(c <= 55, 50)
                 .when(c <= 65, 60).otherwise(70)
            )
            c = F.when(c.isNull(), c).when(legal, c).otherwise(snapped)

        if out_name in VALIDITY_BOUNDS:
            lo, hi = VALIDITY_BOUNDS[out_name]
            c = F.when((c < lo) | (c > hi), None).otherwise(c)

        if out_name == "Location_Easting_OSGR":
            c = F.when(c.isNull() | F.isnan(c), F.lit(443666.0)).otherwise(c)
        if out_name == "Location_Northing_OSGR":
            c = F.when(c.isNull() | F.isnan(c), F.lit(259090.0)).otherwise(c)

        exprs.append(c.alias(out_name))
    exprs.append(((F.col("`Latitude`") * 2).cast("int") / 2).alias("lat_bin"))
    exprs.append(((F.col("`Longitude`") * 2).cast("int") / 2).alias("lon_bin"))

    return exprs


def clean(df: DataFrame) -> DataFrame:
    """
    Full cleaning pipeline. Query-plan safe:
      - All scalar column transforms → ONE .select() call (one Project node)
      - All required-col null checks  → ONE .filter() call (one Filter node)
      - Imputer + group-mode fill run after a lineage-breaking cache()
    """

    to_drop = [c for c in HIGH_MISSING_COLS + LEAKAGE_COLS if c in df.columns]
    if to_drop:
        df = df.drop(*to_drop)

    exprs = _build_select_exprs(df)
    df = df.select(exprs)

    present_required = [c for c in REQUIRED_COLS if c in df.columns]
    if present_required:
        combined = reduce(lambda a, b: a & b,
                          [F.col(c).isNotNull() for c in present_required])
        df = df.filter(combined)

    mode_present = [c for c in MODE_IMPUTE_COLS if c in df.columns]
    for c in mode_present:
        mode_row = (
            df.filter(F.col(c).isNotNull())
              .groupBy(c).count()
              .orderBy(F.col("count").desc())
              .select(c).head()
        )
        if mode_row:
            df = df.fillna({c: mode_row[0]})

    num_present = [
        c for c in NUM_IMPUTE_COLS
        if c in df.columns
        and df.filter(F.col(c).isNotNull()).limit(1).count() > 0
    ]
    if num_present:
        imputer = Imputer(
            inputCols=num_present,
            outputCols=num_present,
            strategy="median",
        )
        df = imputer.fit(df).transform(df)

    df.cache()
    df.count()

    for target, group in GROUP_MODE_IMPUTE:
        df = _fill_mode_by_group(df, target, group)

    cat_present = [c for c in CAT_IMPUTE_COLS if c in df.columns]
    if cat_present:
        df = df.fillna("Unknown", subset=cat_present)

    DOT_CAT_COLS = ["Vehicle_Location.Restricted_Lane"]
    for c in DOT_CAT_COLS:
        if c in df.columns:
            df = df.withColumn(
                c,
                F.when(F.col(f"`{c}`").isNull(), F.lit("Unknown"))
                 .otherwise(F.col(f"`{c}`"))
            )

    return df


# 
# Class weighting
# 

def compute_class_weights(df: DataFrame, label_col: str = "Accident_Severity") -> dict:
    counts = {r[label_col]: r["count"] for r in df.groupBy(label_col).count().collect()}
    total = sum(counts.values())
    n_classes = len(counts)
    return {cls: total / (n_classes * cnt) for cls, cnt in counts.items()}


def add_class_weights(
    df: DataFrame,
    weights: dict,
    label_col: str = "Accident_Severity",
    weight_col: str = "classWeight",
) -> DataFrame:
    expr = None
    for cls, w in weights.items():
        condition = F.col(label_col) == cls
        expr = F.when(condition, w) if expr is None else expr.when(condition, w)
    return df.withColumn(weight_col, expr.otherwise(1.0))