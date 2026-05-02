"""
Data Ingestion Pipeline.

Weather Stations (API)         ← small reference
        ↓
Weather Data per station       ← bulk download + normalise per-file
        ↓
Accidents CSV + Vehicles CSV   ← Spark ingestion with explicit schema
        ↓
Nearest station per accident   ← haversine on driver, joined via Spark
        ↓
Join weather                   ← distributed join on (station_id, date)
        ↓
Join vehicles                  ← final merge
        ↓
Write Parquet (partitioned)    ← final dataset partitioned by Year

Notes:
- Weather data is DAILY — joining on date is the correct "nearest time" join.
  Sub-daily granularity does not exist in the Meteostat bulk data.
- A `datetime` column (Date + Time concatenated) is created for downstream
  feature engineering (hour_of_day, rush_hour, etc.) — not used for joining.
- Nearest-station lookup runs on the driver with vectorised NumPy haversine,
  then joins the 2-column result back into Spark. This avoids the pandas_udf
  Python worker OOM crashes seen on Windows with large Arrow batches.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import meteostat as ms

socket.setdefaulttimeout(30)
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType,
    StructField, StructType, TimestampType,
)

from src.config import (
    ACCIDENTS_CSV,
    ACCIDENTS_WITH_STATION_PARQUET,
    INTERIM_DIR,
    MERGED_PARQUET,
    STATIONS_PARQUET,
    STATION_WEATHER_PARQUET,
    VEHICLES_CSV,
    WEATHER_END,
    WEATHER_START,
    WEATHER_RAW_DIR,
    get_spark,
)
from src.data.acquire import download_dataset


# ══════════════════════════════════════════════════════════════════════════
# Schemas — explicit typing avoids inferSchema overhead and mis-typed cols
# ══════════════════════════════════════════════════════════════════════════

# Date and Time arrive as plain strings in the CSV ("2005-01-04", "05:15").
# We read them as StringType and convert to a proper timestamp in stage [c].
# Reading Date as TimestampType would require Spark to parse the raw string
# during CSV reading — unreliable with mixed locale formats.
ACCIDENTS_SCHEMA = StructType([
    StructField("Accident_Index",                                    StringType(),  True),
    StructField("1st_Road_Class",                                    StringType(),  True),
    StructField("1st_Road_Number",                                   DoubleType(),  True),
    StructField("2nd_Road_Class",                                    StringType(),  True),
    StructField("2nd_Road_Number",                                   DoubleType(),  True),
    StructField("Accident_Severity",                                 StringType(),  True),
    StructField("Carriageway_Hazards",                               StringType(),  True),
    StructField("Date",                                              StringType(),  True),  # "2005-01-04"
    StructField("Day_of_Week",                                       StringType(),  True),
    StructField("Did_Police_Officer_Attend_Scene_of_Accident",       StringType(),  True),
    StructField("Junction_Control",                                  StringType(),  True),
    StructField("Junction_Detail",                                   StringType(),  True),
    StructField("Latitude",                                          DoubleType(),  True),
    StructField("Light_Conditions",                                  StringType(),  True),
    StructField("Local_Authority_(District)",                        StringType(),  True),
    StructField("Local_Authority_(Highway)",                         StringType(),  True),
    StructField("Location_Easting_OSGR",                             DoubleType(),  True),
    StructField("Location_Northing_OSGR",                            DoubleType(),  True),
    StructField("Longitude",                                         DoubleType(),  True),
    StructField("LSOA_of_Accident_Location",                         StringType(),  True),
    StructField("Number_of_Casualties",                              IntegerType(), True),
    StructField("Number_of_Vehicles",                                IntegerType(), True),
    StructField("Pedestrian_Crossing-Human_Control",                 StringType(),  True),
    StructField("Pedestrian_Crossing-Physical_Facilities",           StringType(),  True),
    StructField("Police_Force",                                      StringType(),  True),
    StructField("Road_Surface_Conditions",                           StringType(),  True),
    StructField("Road_Type",                                         StringType(),  True),
    StructField("Special_Conditions_at_Site",                        StringType(),  True),
    StructField("Speed_limit",                                       DoubleType(),  True),
    StructField("Time",                                              StringType(),  True),  # "05:15"
    StructField("Urban_or_Rural_Area",                               StringType(),  True),
    StructField("Weather_Conditions",                                StringType(),  True),
    StructField("Year",                                              IntegerType(), True),
    StructField("InScotland",                                        StringType(),  True),
])

VEHICLES_SCHEMA = StructType([
    StructField("Accident_Index",                StringType(),  True),
    StructField("Age_Band_of_Driver",            StringType(),  True),
    StructField("Age_of_Vehicle",                DoubleType(),  True),
    StructField("Driver_Home_Area_Type",         StringType(),  True),
    StructField("Driver_IMD_Decile",             DoubleType(),  True),
    StructField("Engine_Capacity_.CC.",          DoubleType(),  True),
    StructField("Hit_Object_in_Carriageway",     StringType(),  True),
    StructField("Hit_Object_off_Carriageway",    StringType(),  True),
    StructField("Journey_Purpose_of_Driver",     StringType(),  True),
    StructField("Junction_Location",             StringType(),  True),
    StructField("make",                          StringType(),  True),
    StructField("model",                         StringType(),  True),
    StructField("Propulsion_Code",               StringType(),  True),
    StructField("Sex_of_Driver",                 StringType(),  True),
    StructField("Skidding_and_Overturning",      StringType(),  True),
    StructField("Towing_and_Articulation",       StringType(),  True),
    StructField("Vehicle_Leaving_Carriageway",   StringType(),  True),
    StructField("Vehicle_Location.Restricted_Lane", DoubleType(), True),
    StructField("Vehicle_Manoeuvre",             StringType(),  True),
    StructField("Vehicle_Reference",             IntegerType(), True),
    StructField("Vehicle_Type",                  StringType(),  True),
    StructField("Was_Vehicle_Left_Hand_Drive",   StringType(),  True),
    StructField("X1st_Point_of_Impact",          StringType(),  True),
    StructField("Year",                          IntegerType(), True),
])


# ══════════════════════════════════════════════════════════════════════════
# Stage a — Stations reference table
# ══════════════════════════════════════════════════════════════════════════
def build_stations_parquet(spark: SparkSession) -> None:
    if STATIONS_PARQUET.exists():
        print(f"[a] {STATIONS_PARQUET} already exists — skipping")
        return

    print("[a] fetching GB stations from Meteostat ...")
    try:
        UK_CENTER = ms.Point(52.83, -1.83)
        stations_pdf = ms.stations.nearby(UK_CENTER, radius=500_000, limit=500)
        stations_pdf = stations_pdf[stations_pdf["country"] == "GB"].copy()
    except AttributeError:
        stations_pdf = ms.Stations().region("GB").fetch()

    stations_pdf = stations_pdf.reset_index()
    stations_pdf = stations_pdf[["id", "latitude", "longitude"]].dropna()
    stations_pdf["id"] = stations_pdf["id"].astype(str)

    (spark.createDataFrame(stations_pdf)
         .write.mode("overwrite")
         .parquet(str(STATIONS_PARQUET)))

    print(f"[a] wrote {len(stations_pdf)} stations → {STATIONS_PARQUET}")


# ══════════════════════════════════════════════════════════════════════════
# Stage b — Per-station daily weather (bulk download + per-file normalise)
# ══════════════════════════════════════════════════════════════════════════
def download_bulk_weather(station_ids: list[str]) -> None:
    """
    Download one .csv.gz per station-year from the Meteostat bulk endpoint.
    Skips already-downloaded files — safe to re-run after interruption.
    Caches confirmed 404s in _not_found.json to make re-runs instant.
    """
    WEATHER_RAW_DIR.mkdir(parents=True, exist_ok=True)
    base_url = "https://data.meteostat.net/daily"
    years = range(2005, 2018)

    import json
    not_found_cache = WEATHER_RAW_DIR / "_not_found.json"
    known_missing: set[str] = set()
    if not_found_cache.exists():
        known_missing = set(json.loads(not_found_cache.read_text()))

    saved, not_found, failed = 0, 0, 0
    newly_missing: set[str] = set()
    total = len(station_ids) * len(list(years))
    print(f"[b] downloading {len(station_ids)} stations × {len(list(years))} years = {total} files ...")

    with tqdm(total=total, desc="bulk-download") as pbar:
        for sid in station_ids:
            for year in years:
                out_path  = WEATHER_RAW_DIR / f"{sid}_{year}.csv.gz"
                cache_key = f"{sid}_{year}"

                if out_path.exists():
                    saved += 1
                    pbar.update(1)
                    continue

                if cache_key in known_missing:
                    not_found += 1
                    pbar.update(1)
                    continue

                url = f"{base_url}/{year}/{sid}.csv.gz"
                try:
                    r = requests.get(url, timeout=30)
                    if r.status_code == 200:
                        out_path.write_bytes(r.content)
                        saved += 1
                    elif r.status_code == 404:
                        newly_missing.add(cache_key)
                        not_found += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
                pbar.update(1)

    all_missing = known_missing | newly_missing
    not_found_cache.write_text(json.dumps(list(all_missing)))
    print(f"[b] done — saved: {saved}, not found: {not_found}, errors: {failed}")


def _normalise_gz(f: Path) -> pd.DataFrame | None:
    """
    Read one .csv.gz and return a normalised DataFrame with fixed columns.

    Meteostat bulk files have three different column layouts depending on the
    station and year. Reading each file individually and mapping by column NAME
    (not position) is the only reliable approach — Spark's multi-CSV reader
    assigns columns by position and silently misaligns data across layouts.
    """
    sid = f.stem.rsplit("_", 1)[0]
    try:
        raw = pd.read_csv(f, compression="gzip")
        raw.columns = raw.columns.str.strip()

        out = pd.DataFrame()
        out["station_id"] = sid
        out["time"] = pd.to_datetime(
            raw["year"].astype(str) + "-" +
            raw["month"].astype(str).str.zfill(2) + "-" +
            raw["day"].astype(str).str.zfill(2),
            errors="coerce",
        ).dt.date

        out["temp"] = pd.to_numeric(raw.get("temp"),  errors="coerce").astype("float32")
        out["tmin"] = pd.to_numeric(raw.get("tmin"),  errors="coerce").astype("float32")
        out["tmax"] = pd.to_numeric(raw.get("tmax"),  errors="coerce").astype("float32")
        out["prcp"] = pd.to_numeric(raw.get("prcp"),  errors="coerce").astype("float32")
        out["snwd"] = pd.to_numeric(raw.get("snwd"),  errors="coerce").astype("float32")
        out["wspd"] = pd.to_numeric(raw.get("wspd"),  errors="coerce").astype("float32")
        out["pres"] = pd.to_numeric(raw.get("pres"),  errors="coerce").astype("float32")
        out["rhum"] = pd.to_numeric(raw.get("rhum"),  errors="coerce").astype("float32")
        out["wpgt"] = pd.to_numeric(raw.get("wpgt"),  errors="coerce").astype("float32")
        out["tsun"] = pd.to_numeric(raw.get("tsun"),  errors="coerce").astype("float32")
        out["cldc"] = pd.to_numeric(raw.get("cldc"),  errors="coerce").astype("float32")

        return out.dropna(subset=["time"])
    except Exception:
        return None


def build_station_weather_parquet(spark: SparkSession) -> None:
    """
    Normalise all downloaded .csv.gz files with pandas (one file at a time),
    write normalised data as parquet chunks, then let Spark consolidate them.

    Why not Spark CSV reader directly?
    Meteostat files have 3 different column layouts — Spark assigns columns by
    position when reading multiple CSVs together, silently corrupting data when
    a file has fewer columns than the schema expects.
    Reading each file individually with pandas (column-by-name) avoids this.
    """
    if STATION_WEATHER_PARQUET.exists():
        print(f"[b] {STATION_WEATHER_PARQUET} already exists — skipping")
        return

    stations_pdf = spark.read.parquet(str(STATIONS_PARQUET)).toPandas()
    station_ids  = stations_pdf["id"].tolist()
    download_bulk_weather(station_ids)

    gz_files = list(WEATHER_RAW_DIR.glob("*.csv.gz"))
    if not gz_files:
        raise RuntimeError("No weather files downloaded.")

    # Write normalised data in chunks — keeps peak driver RAM bounded.
    # 500 files × ~365 rows × 8 float32 cols ≈ 6 MB per chunk.
    CHUNK_SIZE = 500
    chunk_dir  = WEATHER_RAW_DIR / "_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    start_date = pd.Timestamp(WEATHER_START)
    end_date   = pd.Timestamp(WEATHER_END)

    print(f"[b] normalising {len(gz_files)} files in chunks of {CHUNK_SIZE} ...")
    errors = 0
    for chunk_idx, start in enumerate(range(0, len(gz_files), CHUNK_SIZE)):
        chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.parquet"
        if chunk_path.exists():
            continue
        frames = [
            r for f in gz_files[start: start + CHUNK_SIZE]
            if (r := _normalise_gz(f)) is not None
        ]
        if not frames:
            continue
        chunk_pd = pd.concat(frames, ignore_index=True)
        chunk_pd["time"] = pd.to_datetime(chunk_pd["time"])
        chunk_pd = chunk_pd[
            (chunk_pd["time"] >= start_date) & (chunk_pd["time"] <= end_date)
        ]
        chunk_pd["time"] = chunk_pd["time"].dt.date
        chunk_pd.to_parquet(str(chunk_path), index=False, engine="pyarrow")

    chunk_files = list(chunk_dir.glob("chunk_*.parquet"))
    if not chunk_files:
        raise RuntimeError("No weather data could be normalised.")

    print(f"[b] reading {len(chunk_files)} chunks into Spark ...")
    (spark.read.parquet(*[str(f) for f in chunk_files])
          .write.mode("overwrite")
          .parquet(str(STATION_WEATHER_PARQUET)))

    print(f"[b] weather parquet written → {STATION_WEATHER_PARQUET}")


# ══════════════════════════════════════════════════════════════════════════
# Stage c — Load accidents + vehicles CSVs
# ══════════════════════════════════════════════════════════════════════════
def load_accidents_and_vehicles(
    spark: SparkSession, raw_dir: Path
) -> tuple[DataFrame, DataFrame]:
    """
    Read both CSVs with explicit schemas (no inferSchema overhead).

    The `datetime` column concatenates Date + Time strings into a proper
    timestamp — used by Member 2 for temporal feature engineering
    (hour_of_day, is_rush_hour, etc.). It is NOT used for weather joining
    because weather data is daily — the join key is just the date part.
    """
    print(f"[c] reading CSVs from {raw_dir}")

    accidents = (
        spark.read
        .schema(ACCIDENTS_SCHEMA)
        .option("header", True)
        .option("mode", "PERMISSIVE")
        .option("encoding", "iso-8859-1")
        .csv(str(raw_dir / ACCIDENTS_CSV))
        # # Build datetime from the two string columns.
        # # "2005-01-04" + " " + "05:15" → timestamp 2005-01-04 05:15:00
        # .withColumn(
        #     "datetime",
        #     F.to_timestamp(
        #         F.concat_ws(" ", F.col("Date"), F.col("Time")),
        #         "yyyy-MM-dd HH:mm",
        #     ),
        # )
        # Keep a proper Date column as DateType for the weather join key
        .withColumn("Date", F.to_date(F.col("Date"), "yyyy-MM-dd"))
        
    )

    vehicles = (
        spark.read
        .schema(VEHICLES_SCHEMA)
        .option("header", True)
        .option("mode", "PERMISSIVE")
        .option("encoding", "iso-8859-1")
        .csv(str(raw_dir / VEHICLES_CSV))
    )

    print(f"[c] accidents partitions: {accidents.rdd.getNumPartitions()}, "
          f"vehicles partitions: {vehicles.rdd.getNumPartitions()}")
    return accidents, vehicles


# ══════════════════════════════════════════════════════════════════════════
# Stage d — Nearest-station lookup (vectorised NumPy on driver)
# ══════════════════════════════════════════════════════════════════════════
# def attach_nearest_station(spark: SparkSession, accidents: DataFrame) -> DataFrame:
#     """
#     Compute the nearest weather station for each accident using a vectorised
#     NumPy haversine on the driver, then join the 2-column result back into Spark.

#     Why not pandas_udf?
#     pandas_udf spawns one Python worker process per Spark task. On Windows with
#     Python 3.14 + Spark 4.0, the Arrow IPC serialisation causes worker OOM crashes
#     with this dataset (2M rows × 156 stations). The driver-side approach pulls only
#     3 columns (Accident_Index, Lat, Lon) to the driver, runs the NumPy haversine
#     in ~2 seconds, then joins a tiny 2-column DataFrame back — no worker crashes.

#     Weather is DAILY — there is no sub-daily granularity in Meteostat bulk data.
#     The nearest station is therefore purely spatial (closest by haversine distance),
#     not spatio-temporal. The `datetime` column is irrelevant for this lookup.
#     """
#     print("[d] attaching nearest_station via vectorised NumPy on driver ...")

#     stations_pd = spark.read.parquet(str(STATIONS_PARQUET)).toPandas()
#     s_lat = np.radians(stations_pd["latitude"].values).astype(np.float32)
#     s_lon = np.radians(stations_pd["longitude"].values).astype(np.float32)
#     s_id  = stations_pd["id"].values

#     # Pull only the 3 columns needed — avoids sending 64 cols to the driver
#     acc_pd = (
#         accidents
#         .select("Accident_Index", "Latitude", "Longitude")
#         .toPandas()
#     )

#     # Vectorised haversine: (N_accidents × N_stations) matrix in one shot
#     a_lat = np.radians(
#         pd.to_numeric(acc_pd["Latitude"],  errors="coerce").fillna(0).values
#     ).astype(np.float32)[:, None]
#     a_lon = np.radians(
#         pd.to_numeric(acc_pd["Longitude"], errors="coerce").fillna(0).values
#     ).astype(np.float32)[:, None]

#     dlat = s_lat - a_lat
#     dlon = s_lon - a_lon
#     h    = np.sin(dlat / 2) ** 2 + np.cos(a_lat) * np.cos(s_lat) * np.sin(dlon / 2) ** 2
#     idx  = np.argmin(h, axis=1)

#     acc_pd["station_id"] = s_id[idx]

#     # Null out rows where lat/lon was genuinely missing
#     missing = (
#         pd.to_numeric(acc_pd["Latitude"],  errors="coerce").isna() |
#         pd.to_numeric(acc_pd["Longitude"], errors="coerce").isna()
#     )
#     acc_pd.loc[missing, "station_id"] = None

#     # Join the 2-column result back — Spark distributes the join
#     station_map = spark.createDataFrame(acc_pd[["Accident_Index", "station_id"]])
#     return accidents.join(F.broadcast(station_map), on="Accident_Index", how="left")

# def attach_nearest_station(spark, accidents_df):
#     # 1. Load stations and broadcast them (Small data -> Send to all workers)
#     stations_pd = spark.read.parquet(str(STATIONS_PARQUET)).toPandas()
#     s_id = stations_pd["id"].values
#     s_lat = np.radians(stations_pd["latitude"].values).astype(np.float32)
#     s_lon = np.radians(stations_pd["longitude"].values).astype(np.float32)

#     b_s_id = spark.sparkContext.broadcast(s_id)
#     b_s_lat = spark.sparkContext.broadcast(s_lat)
#     b_s_lon = spark.sparkContext.broadcast(s_lon)

#     # 2. Define the Vectorized Pandas UDF
#     @pandas_udf("string")
#     def find_nearest_udf(lat_series: pd.Series, lon_series: pd.Series) -> pd.Series:
#         # This code runs on WORKERS in parallel
#         # It only receives a "chunk" of the 2M rows at a time
#         mask = lat_series.notna() & lon_series.notna()
#         result = pd.Series([None] * len(lat_series))
        
#         if not mask.any():
#             return result

#         # Vectorized NumPy Math
#         a_lat = np.radians(lat_series[mask].values).astype(np.float32)[:, None]
#         a_lon = np.radians(lon_series[mask].values).astype(np.float32)[:, None]

#         dlat = b_s_lat.value - a_lat
#         dlon = b_s_lon.value - a_lon
#         h = np.sin(dlat/2)**2 + np.cos(a_lat) * np.cos(b_s_lat.value) * np.sin(dlon/2)**2
        
#         idx = np.argmin(h, axis=1)
#         result.loc[mask] = b_s_id.value[idx]
#         return result

#     # 3. Apply the UDF to the distributed DataFrame
#     return accidents_df.withColumn(
#         "station_id", 
#         find_nearest_udf(col("Latitude"), col("Longitude"))
#     )

from pyspark.sql.functions import pandas_udf, col
import pandas as pd
import numpy as np

def attach_nearest_station(spark: SparkSession, accidents: DataFrame) -> DataFrame:
    """
    Nearest-station lookup via a vectorised pandas_udf + broadcast.

    Each Arrow batch arriving at a worker is a 2D NumPy matrix operation:
      batch_size × n_stations — with maxRecordsPerBatch=2000 and ~156 stations
      that's a 2000×156 float32 matrix ≈ 1.9 MB peak RAM per batch.
    The row-by-row loop in the previous version negated all vectorisation.
    """
    stations_pd = spark.read.parquet(str(STATIONS_PARQUET)).toPandas()
    s_id  = stations_pd["id"].values
    s_lat = np.radians(stations_pd["latitude"].values).astype(np.float32)   # (S,)
    s_lon = np.radians(stations_pd["longitude"].values).astype(np.float32)  # (S,)

    b_sid  = spark.sparkContext.broadcast(s_id)
    b_slat = spark.sparkContext.broadcast(s_lat)
    b_slon = spark.sparkContext.broadcast(s_lon)

    @pandas_udf("string")
    def find_nearest_udf(lat_ser: pd.Series, lon_ser: pd.Series) -> pd.Series:
        slat = b_slat.value          # (S,)
        slon = b_slon.value          # (S,)
        sid  = b_sid.value           # (S,)

        valid  = lat_ser.notna() & lon_ser.notna()
        result = pd.array([None] * len(lat_ser), dtype=object)

        if not valid.any():
            return pd.Series(result)

        # (N_valid, 1) − (S,) broadcasts to (N_valid, S) — pure NumPy, no loop
        a_lat = np.radians(lat_ser[valid].to_numpy()).astype(np.float32)[:, None]
        a_lon = np.radians(lon_ser[valid].to_numpy()).astype(np.float32)[:, None]

        dlat = slat - a_lat                                          # (N, S)
        dlon = slon - a_lon                                          # (N, S)
        h    = (np.sin(dlat / 2) ** 2
                + np.cos(a_lat) * np.cos(slat) * np.sin(dlon / 2) ** 2)

        result[valid.to_numpy()] = sid[np.argmin(h, axis=1)]
        return pd.Series(result)

    return accidents.withColumn(
        "station_id",
        find_nearest_udf(F.col("Latitude"), F.col("Longitude")),
    )
# ══════════════════════════════════════════════════════════════════════════
# Stage e — Weather join (date-based distributed join)
# ══════════════════════════════════════════════════════════════════════════
def join_weather(spark: SparkSession, accidents: DataFrame) -> DataFrame:
    if not STATION_WEATHER_PARQUET.exists():
        print("[e] no weather parquet — attaching NULL weather columns")
        for col, dtype in [
            ("temp", "double"), ("tmin", "double"), ("tmax", "double"),
            ("prcp", "double"), ("snwd", "double"), ("wspd", "double"),
            ("pres", "double"),("rhum", "double"), ("wpgt", "double"), ("tsun", "double"), ("cldc", "double")
        ]:
            accidents = accidents.withColumn(col, F.lit(None).cast(dtype))
        return accidents

    weather = spark.read.parquet(str(STATION_WEATHER_PARQUET))
    print("[e] joining weather on (station_id, time) — broadcast join")

    # F.broadcast() tells the Spark planner to send the weather table to every
    # executor rather than shuffling 2M accident rows over the network.
    # Weather is ~740 k rows × 9 columns — well within broadcast threshold.
    accidents = accidents.drop("join_date") if "join_date" in accidents.columns else accidents
    weather = weather.drop("join_date") if "join_date" in weather.columns else weather
    weather = weather.withColumn("time", F.col("time").cast("date"))

    return (  
        accidents.alias("a")
        .join(
            F.broadcast(weather).alias("w"),
            (F.col("a.station_id") == F.col("w.station_id"))
            & (F.col("a.Date") == F.col("w.time")),
            how="left",
        )
        .drop(F.col("w.station_id"))
        .drop(F.col("w.time"))
    )
    

# ══════════════════════════════════════════════════════════════════════════
# Stage f — Join vehicles + write partitioned Parquet
# ══════════════════════════════════════════════════════════════════════════
def join_vehicles_and_write(acc_weather: DataFrame, vehicles: DataFrame) -> None:
    """
    Left-join vehicle records onto the accident+weather DataFrame, then write
    the result as Parquet partitioned by Year.

    Repartition strategy:
    repartition(n_files, "Year") creates n_files partitions with rows shuffled
    by Year — each output file contains one year's data without the skew risk
    of repartition(F.col("Year")) which produces exactly 1 partition per year
    (13 tiny files, bad for parallel reads downstream).
    """
    vehicles = vehicles.drop("Year")
    merged = acc_weather.join(vehicles, on="Accident_Index", how="left")

    n_files = 26
    print(f"[f] writing {MERGED_PARQUET} (repartition={n_files}, partitionBy=Year)")
    (merged
        .repartition(n_files, F.col("Year"))   # shuffle by Year, n_files output partitions
        .write.mode("overwrite")
        .partitionBy("Year")
        .parquet(str(MERGED_PARQUET)))


# ══════════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    spark = get_spark("ingest")
    spark.sparkContext.setLogLevel("WARN")

    # Stage a — stations reference
    build_stations_parquet(spark)

    # Stage b — weather (failure here is non-fatal; join_weather falls back to NULLs)
    try:
        build_station_weather_parquet(spark)
    except Exception as e:
        print(f"[b] WARNING: weather fetch failed ({e!r}); continuing without weather")

    # Stage c — load source CSVs
    raw_dir = download_dataset()
    accidents, vehicles = load_accidents_and_vehicles(spark, raw_dir)

    # Stage d — nearest-station lookup with checkpoint
    # Checkpoint isolates the expensive haversine stage: if stages [e]/[f]
    # crash later, we re-read from this checkpoint instead of re-running the lookup.
    if not ACCIDENTS_WITH_STATION_PARQUET.exists():
        print("[d] running nearest-station lookup and checkpointing ...")
        temp_accidents = attach_nearest_station(spark, accidents)
        (temp_accidents
            .write.mode("overwrite")
            .parquet(str(ACCIDENTS_WITH_STATION_PARQUET)))
        print(f"[d] checkpoint written → {ACCIDENTS_WITH_STATION_PARQUET}")
        accidents = spark.read.parquet(str(ACCIDENTS_WITH_STATION_PARQUET))
    else:
        print(f"[d] checkpoint exists — reading {ACCIDENTS_WITH_STATION_PARQUET}")

        accidents = spark.read.parquet(str(ACCIDENTS_WITH_STATION_PARQUET))

    # Stage e — weather join
    acc_weather = join_weather(spark, accidents)

     # ── NEW: checkpoint to disk so stage [f] doesn't re-run the entire DAG ──
    ACC_WEATHER_PARQUET = INTERIM_DIR / "acc_weather.parquet"
    if not ACC_WEATHER_PARQUET.exists():
        print("[e] writing acc_weather checkpoint ...")
        (acc_weather
            .write.mode("overwrite")
            .parquet(str(ACC_WEATHER_PARQUET)))
    acc_weather = spark.read.parquet(str(ACC_WEATHER_PARQUET))
    # ────────────────────────────────────────────────────────────────────────

    # Stage f — vehicle join + write
    join_vehicles_and_write(acc_weather, vehicles)
    #acc_weather.unpersist()

    # Sanity check
    out = spark.read.parquet(str(MERGED_PARQUET))
    print(f"[done] merged rows: {out.count():,}, columns: {len(out.columns)}")

    spark.stop()


if __name__ == "__main__":
    main()