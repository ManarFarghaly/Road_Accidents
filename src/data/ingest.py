from __future__ import annotations

import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import meteostat as ms
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



# Schemas — explicit typing avoids inferSchema overhead and mis-typed cols


ACCIDENTS_SCHEMA = StructType([
    StructField("Accident_Index",                                    StringType(),  True),
    StructField("1st_Road_Class",                                    StringType(),  True),
    StructField("1st_Road_Number",                                   DoubleType(),  True),
    StructField("2nd_Road_Class",                                    StringType(),  True),
    StructField("2nd_Road_Number",                                   DoubleType(),  True),
    StructField("Accident_Severity",                                 StringType(),  True),
    StructField("Carriageway_Hazards",                               StringType(),  True),
    StructField("Date",                                              StringType(),  True),
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
    StructField("Time",                                              StringType(),  True),
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



# Stage a — Stations reference table

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



# Stage b — Per-station daily weather (bulk download + per-file normalise)

def download_bulk_weather(station_ids: list[str]) -> None:
    """
    Download one .csv.gz per station-year from the Meteostat bulk endpoint.
    Skips already-downloaded files — safe to re-run after interruption.
    Caches confirmed 404s in _not_found.json to make re-runs instant.

    NOTE: timeout=30 is passed directly to requests.get(), NOT via
    socket.setdefaulttimeout() which would affect the Py4J socket too.
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
                    # ← timeout= here, NOT socket.setdefaulttimeout()
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
    sid = f.stem.rsplit("_", 1)[0]
    try:
        raw = pd.read_csv(f, compression="gzip")
        raw.columns = raw.columns.str.strip()
        n = len(raw)
        if n == 0:
            return None

        out = pd.DataFrame(index=range(n))
        out["station_id"] = str(sid)           # plain object dtype — reliable broadcast

        # ── Handle BOTH Meteostat bulk date formats ────────────────────────
        if "date" in raw.columns:
            # Modern format: single ISO-8601 date column
            out["time"] = pd.to_datetime(raw["date"], errors="coerce").dt.date
        elif "year" in raw.columns:
            # Legacy format: separate year / month / day columns
            out["time"] = pd.to_datetime(
                raw["year"].astype(str) + "-" +
                raw["month"].astype(str).str.zfill(2) + "-" +
                raw["day"].astype(str).str.zfill(2),
                errors="coerce",
            ).dt.date
        else:
            # Unrecognised layout — skip silently
            return None

        # Map Meteostat column aliases (tavg → temp in some files)
        col_map = {"tavg": "temp", "snow": "snwd"}
        raw = raw.rename(columns=col_map)

        for weather_col in ("temp", "tmin", "tmax", "prcp", "snwd",
                            "wspd", "pres", "rhum", "wpgt", "tsun", "cldc"):
            out[weather_col] = pd.to_numeric(
                raw.get(weather_col), errors="coerce"
            ).astype("float32")

        result = out.dropna(subset=["time"])
        return result if len(result) > 0 else None

    except Exception as e:
        print(f"[b] WARN: skipping {f.name} — {e}")
        return None          # ← return None, don't raise; let other files proceed


def build_station_weather_parquet(spark: SparkSession) -> None:
    stations_pdf = spark.read.parquet(str(STATIONS_PARQUET)).toPandas()
    station_ids  = stations_pdf["id"].tolist()
    download_bulk_weather(station_ids)
    gz_files = list(WEATHER_RAW_DIR.glob("*.csv.gz"))
    if not gz_files:
        raise RuntimeError("No weather files downloaded.")

    CHUNK_SIZE = 500
    chunk_dir  = WEATHER_RAW_DIR / "_chunks"

    # ── Always wipe stale chunks before re-normalising ─────────────────────
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
        print("[b] cleared stale _chunks dir")
    chunk_dir.mkdir(parents=True)

    start_date = pd.Timestamp(WEATHER_START)
    end_date   = pd.Timestamp(WEATHER_END)

    print(f"[b] normalising {len(gz_files)} files in chunks of {CHUNK_SIZE} ...")
    written = 0
    for chunk_idx, start in enumerate(range(0, len(gz_files), CHUNK_SIZE)):
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

        # ── Sanity-check before writing ────────────────────────────────────
        null_ids = chunk_pd["station_id"].isna().sum()
        if null_ids > 0:
            raise RuntimeError(
                f"[b] chunk {chunk_idx}: {null_ids} null station_ids — "
                f"fix _normalise_gz before proceeding"
            )

        chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.parquet"
        chunk_pd.to_parquet(str(chunk_path), index=False, engine="pyarrow")
        written += 1
        print(f"[b] chunk {chunk_idx}: {len(chunk_pd):,} rows, "
              f"stations: {chunk_pd['station_id'].nunique()}")

    if written == 0:
        raise RuntimeError("No weather data could be normalised.")

    chunk_files = list(chunk_dir.glob("chunk_*.parquet"))
    print(f"[b] consolidating {len(chunk_files)} chunks with Spark ...")
    (spark.read
          .option("mergeSchema", "true")
          .parquet(*[str(f) for f in chunk_files])
          .write.mode("overwrite")
          .parquet(str(STATION_WEATHER_PARQUET)))

    all_weather = pd.concat(frames, ignore_index=True)
    # Filter to accident date range
    all_weather = all_weather[
        (all_weather["time"] >= pd.Timestamp(WEATHER_START)) &
        (all_weather["time"] <= pd.Timestamp(WEATHER_END))
    ]
    # Convert datetime to date for Spark compatibility
    all_weather["time"] = all_weather["time"].dt.date

    print(f"[b] total weather rows: {len(all_weather):,} — creating Spark DataFrame ...")
    # Write directly with pandas/pyarrow — avoids spark.createDataFrame() which crashes
    # on Windows/Python 3.13 due to Python worker socket errors for large DataFrames.
    # Both pd.read_parquet() and spark.read.parquet() handle a single-file parquet fine.
    print(f"[b] total weather rows: {len(all_weather):,} — writing parquet with pandas ...")
    all_weather.to_parquet(str(STATION_WEATHER_PARQUET), index=False, engine="pyarrow")
    print(f"[b] weather parquet written → {STATION_WEATHER_PARQUET}")

# Stage c — Load accidents + vehicles CSVs

def load_accidents_and_vehicles(
    spark: SparkSession, raw_dir: Path
) -> tuple[DataFrame, DataFrame]:
    print(f"[c] reading CSVs from {raw_dir}")

    accidents = (
        spark.read
        .schema(ACCIDENTS_SCHEMA)
        .option("header", True)
        .option("mode", "PERMISSIVE")
        .option("encoding", "iso-8859-1")
        .csv(str(raw_dir / ACCIDENTS_CSV))
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


# 
# Stage d — Nearest-station lookup (vectorised pandas_udf + broadcast)
# 
def attach_nearest_station(spark: SparkSession, accidents: DataFrame) -> DataFrame:
    """
    Nearest-station lookup via a vectorised pandas_udf + broadcast.

    Each Arrow batch is a (batch_size × n_stations) NumPy matrix op.
    With maxRecordsPerBatch=2000 and ~156 stations that's ≈ 1.9 MB RAM
    per batch — safe on any machine.
    """
    stations_pd = spark.read.parquet(str(STATIONS_PARQUET)).toPandas()
    s_id  = stations_pd["id"].values
    s_lat = np.radians(stations_pd["latitude"].values).astype(np.float32)
    s_lon = np.radians(stations_pd["longitude"].values).astype(np.float32)

    b_sid  = spark.sparkContext.broadcast(s_id)
    b_slat = spark.sparkContext.broadcast(s_lat)
    b_slon = spark.sparkContext.broadcast(s_lon)

    @pandas_udf("string")
    def find_nearest_udf(lat_ser: pd.Series, lon_ser: pd.Series) -> pd.Series:
        slat = b_slat.value   # (S,)
        slon = b_slon.value   # (S,)
        sid  = b_sid.value    # (S,)

        valid  = lat_ser.notna() & lon_ser.notna()
        result = pd.array([None] * len(lat_ser), dtype=object)

        if not valid.any():
            return pd.Series(result)

        a_lat = np.radians(lat_ser[valid].to_numpy()).astype(np.float32)[:, None]
        a_lon = np.radians(lon_ser[valid].to_numpy()).astype(np.float32)[:, None]

        dlat = slat - a_lat
        dlon = slon - a_lon
        h    = (np.sin(dlat / 2) ** 2
                + np.cos(a_lat) * np.cos(slat) * np.sin(dlon / 2) ** 2)

        result[valid.to_numpy()] = sid[np.argmin(h, axis=1)]
        return pd.Series(result)

    return accidents.withColumn(
        "station_id",
        find_nearest_udf(F.col("Latitude"), F.col("Longitude")),
    )


# 
# Stage e — Weather join (date-based, broadcast join)
# 
def join_weather(spark: SparkSession, accidents: DataFrame) -> DataFrame:
    if not STATION_WEATHER_PARQUET.exists():
        print("[e] no weather parquet — attaching NULL weather columns")
        null_cols = [
            ("temp", "double"), ("tmin", "double"), ("tmax", "double"),
            ("prcp", "double"), ("snwd", "double"), ("wspd", "double"),
            ("pres", "double"), ("rhum", "double"), ("wpgt", "double"),
            ("tsun", "double"), ("cldc", "double"),
        ]
        for c, dtype in null_cols:
            accidents = accidents.withColumn(c, F.lit(None).cast(dtype))
        return accidents

    weather = spark.read.parquet(str(STATION_WEATHER_PARQUET))
    print("[e] joining weather on (station_id, time) — broadcast join")

    # Clean up any stale join columns before aliasing
    accidents = accidents.drop("join_date") if "join_date" in accidents.columns else accidents
    weather   = weather.drop("join_date")   if "join_date" in weather.columns   else weather
    weather   = weather.withColumn("time", F.col("time").cast("date"))

    result = (
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
     # ── Sanity check: flag if join produced mostly nulls ─────────────────
    total   = result.count()
    matched = result.filter(F.col("temp").isNotNull()).count()
    rate    = matched / total * 100 if total else 0
    print(f"[e] weather join: {matched:,}/{total:,} rows matched ({rate:.1f}%)")

    if rate < 10:
        # Diagnose: how many accidents have a station_id at all?
        with_station = accidents.filter(F.col("station_id").isNotNull()).count()
        print(f"[e] WARN: only {rate:.1f}% weather match rate!")
        print(f"[e]   accidents with station_id: {with_station:,}/{total:,}")
        print(f"[e]   check that stage [d] ran and ACCIDENTS_WITH_STATION_PARQUET is not stale")

    return result

# 
# Stage f — Join vehicles + write partitioned Parquet
# 
def join_vehicles_and_write(acc_weather: DataFrame, vehicles: DataFrame) -> None:
    vehicles = vehicles.drop("Year")
    merged = acc_weather.join(vehicles, on="Accident_Index", how="left")

    n_files = 26   # 2 per year × 13 years — tune up if files are >256 MB
    print(f"[f] writing {MERGED_PARQUET} (repartition={n_files}, partitionBy=Year)")
    (merged
        .repartition(n_files, F.col("Year"))
        .write.mode("overwrite")
        .partitionBy("Year")
        .parquet(str(MERGED_PARQUET)))



# Orchestration

def main() -> None:
    # Increase socket timeout to 30 minutes before any Spark operations to prevent py4j timeouts
    import socket
    socket.setdefaulttimeout(1800)
    
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    # Force stop any existing cached session to apply new config
    from pyspark.sql import SparkSession
    import pandas as pd
    existing = SparkSession.getActiveSession()
    if existing:
        existing.stop()

    spark = get_spark("ingest")
    spark.sparkContext.setLogLevel("WARN")

    # Stage a — stations reference
    build_stations_parquet(spark)

    # Stage b — weather (non-fatal; join_weather falls back to NULLs)
    try:
        build_station_weather_parquet(spark)
    except Exception as e:
        print(f"[b] WARNING: weather fetch failed ({e!r}); continuing without weather")

    # Stage c — load source CSVs
    raw_dir = download_dataset()
    accidents, vehicles = load_accidents_and_vehicles(spark, raw_dir)

    # Stage d — nearest-station lookup with checkpoint
    # Checkpointing isolates the expensive haversine stage; if [e]/[f] crash
    # later, we re-read from this checkpoint instead of re-running the UDF.
    if not ACCIDENTS_WITH_STATION_PARQUET.exists():
        print("[d] running nearest-station lookup and checkpointing ...")
        temp_accidents = attach_nearest_station(spark, accidents)
        (temp_accidents
            .write.mode("overwrite")
            .parquet(str(ACCIDENTS_WITH_STATION_PARQUET)))
        print(f"[d] checkpoint written → {ACCIDENTS_WITH_STATION_PARQUET}")
    else:
        print(f"[d] checkpoint exists — reading {ACCIDENTS_WITH_STATION_PARQUET}")

    accidents = spark.read.parquet(str(ACCIDENTS_WITH_STATION_PARQUET))

    # Stage e — weather join
    acc_weather = join_weather(spark, accidents)

    # Checkpoint acc_weather so stage [f] does not re-execute the entire DAG
    # on retry.
    ACC_WEATHER_PARQUET = INTERIM_DIR / "acc_weather.parquet"
    

    def _checkpoint_has_weather(spark: SparkSession, path: Path) -> bool:
        """Return False if the checkpoint exists but weather cols are all null."""
        if not path.exists():
            return False
        sample = spark.read.parquet(str(path)).select("temp").limit(1000)
        non_null = sample.filter(F.col("temp").isNotNull()).count()
        return non_null > 0

    if not _checkpoint_has_weather(spark, ACC_WEATHER_PARQUET):
        print("[e] writing acc_weather checkpoint ...")
        acc_weather = join_weather(spark, accidents)   # re-run join
        (acc_weather
            .write.mode("overwrite")
            .parquet(str(ACC_WEATHER_PARQUET)))
        print(f"[e] checkpoint written → {ACC_WEATHER_PARQUET}")

    acc_weather = spark.read.parquet(str(ACC_WEATHER_PARQUET))

    if not ACC_WEATHER_PARQUET.exists():
        print("[e] writing acc_weather checkpoint ...")
        (acc_weather
            .write.mode("overwrite")
            .parquet(str(ACC_WEATHER_PARQUET)))
    acc_weather = spark.read.parquet(str(ACC_WEATHER_PARQUET))

    # Stage f — vehicle join + write
    join_vehicles_and_write(acc_weather, vehicles)

    # Sanity check
    out = spark.read.parquet(str(MERGED_PARQUET))
    print(f"[done] merged rows: {out.count():,}, columns: {len(out.columns)}")

    spark.stop()


if __name__ == "__main__":
    main()