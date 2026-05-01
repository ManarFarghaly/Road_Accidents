"""
Data Ingestion Pipeline.

Weather Stations (API)         ← small reference
        ↓
Weather Data per station       ← heavy fetch (cached)
        ↓
Accidents CSV + Vehicles CSV   ← Spark ingestion
        ↓
Nearest station per accident   ← geo computation
        ↓
Join weather                   ← distributed join
        ↓
Join vehicles                  ← final merge
        ↓
Write Parquet (partitioned)    ← final dataset

"""

from __future__ import annotations

import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import meteostat as ms
import requests
import gzip
import shutil


# Global socket timeout — prevents urllib from hanging for minutes on bad connections
socket.setdefaulttimeout(30)

from pyspark.sql import DataFrame, SparkSession, functions as F, Window
from pyspark.sql.functions import pandas_udf



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


def download_bulk_weather(station_ids: list[str]) -> None:
    """
    Download yearly .csv.gz files per station from Meteostat bulk endpoint.
    URL format: https://data.meteostat.net/daily/{year}/{station}.csv.gz
    """
    WEATHER_RAW_DIR.mkdir(parents=True, exist_ok=True)
    base_url = "https://data.meteostat.net/daily"
    years = range(2005, 2018)  # 2005–2017 inclusive
    saved, not_found, failed = 0, 0, 0

    total = len(station_ids) * len(list(years))
    print(f"[b] downloading {len(station_ids)} stations × {len(list(years))} years = {total} files ...")

    with tqdm(total=total, desc="bulk-download") as pbar:
        for sid in station_ids:
            for year in years:
                out_path = WEATHER_RAW_DIR / f"{sid}_{year}.csv.gz"
                if out_path.exists():
                    saved += 1
                    pbar.update(1)
                    continue

                url = f"{base_url}/{year}/{sid}.csv.gz"
                try:
                    r = requests.get(url, timeout=30)
                    if r.status_code == 200:
                        out_path.write_bytes(r.content)
                        saved += 1
                    elif r.status_code == 404:
                        not_found += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
                pbar.update(1)


    print(f"[b] done — saved: {saved}, not found: {not_found}, errors: {failed}")


def build_station_weather_parquet(spark: SparkSession) -> None:
    if STATION_WEATHER_PARQUET.exists():
        print(f"[b] {STATION_WEATHER_PARQUET} already exists — skipping")
        return

    stations_pdf = spark.read.parquet(str(STATIONS_PARQUET)).toPandas()
    station_ids  = stations_pdf["id"].tolist()
    download_bulk_weather(station_ids)

    gz_files = list(WEATHER_RAW_DIR.glob("*.csv.gz"))
    if not gz_files:
        raise RuntimeError("No weather files downloaded.")

    print(f"[b] ingesting {len(gz_files)} station files via Spark ...")

    # Read all columns as strings — actual columns are year/month/day + data + _source cols
    weather = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)  
        .option("enforceSchema", False)
        .csv([str(f) for f in gz_files])
        # Build a proper date from the 3 separate columns
        .withColumn(
            "time",
            F.to_date(
                F.concat_ws("-", F.col("year"), F.col("month"), F.col("day")),
                "yyyy-M-d"
            )
        )
        # Extract station_id from filename
        .withColumn(
            "station_id",
            F.regexp_extract(F.input_file_name(), r"([^/\\]+)_\d{4}\.csv\.gz$", 1)
        )
        # Cast only the weather columns we need to float
        .withColumn("temp", F.col("temp").cast("float"))
        .withColumn("tmin", F.col("tmin").cast("float"))
        .withColumn("tmax", F.col("tmax").cast("float"))
        .withColumn("prcp", F.col("prcp").cast("float"))
        .withColumn("snow", F.col("snwd").cast("float"))
        .withColumn("wspd", F.col("wspd").cast("float"))
        .withColumn("pres", F.col("pres").cast("float"))
        .withColumn("rhum", F.col("rhum").cast("float"))  
        
        # Keep only what we need
        .select("station_id", "time", "temp", "tmin", "tmax", "prcp", "snow", "wspd", "pres")
        .filter(F.col("time").isNotNull())
        .filter(F.col("time").between(WEATHER_START, WEATHER_END))
    )

    (weather
        .write
        .mode("overwrite")
        .parquet(str(STATION_WEATHER_PARQUET)))

    print(f"[b] weather parquet written → {STATION_WEATHER_PARQUET}")

# ════════════════════════════════════════════════════════════════════════════
# Stage a — Stations reference table
# ════════════════════════════════════════════════════════════════════════════
def build_stations_parquet(spark: SparkSession) -> None:
    """
    Fetch all Meteostat stations in Great Britain and write them as Parquet.

    We use the `meteostat` Python library (v2.x API).

    meteostat v2 (Dec 2024) removed the old Stations().region("GB") API.
    The new approach: query stations near a central UK coordinate with a
    large radius (500 km covers all of GB), then filter by country == 'GB'.
    """
    if STATIONS_PARQUET.exists():
        print(f"[a] {STATIONS_PARQUET} already exists — skipping")
        return

    print("[a] fetching GB stations from Meteostat ...")
    # Meteostat had a breaking API change in Dec 2024 (v1 → v2). Try v2 first,
    # fall back to v1 so the code works on both road_env (Py3.10 / v1) and
    # newer Python envs that pulled v2.
    try:
        # v2 API: ms.stations.nearby(point, radius, limit)
        UK_CENTER = ms.Point(52.83, -1.83)
        stations_pdf = ms.stations.nearby(UK_CENTER, radius=500_000, limit=500)
        stations_pdf = stations_pdf[stations_pdf["country"] == "GB"].copy()
    except AttributeError:
        # v1 API: Stations().region("GB").fetch()
        stations_pdf = ms.Stations().region("GB").fetch()

    stations_pdf = stations_pdf.reset_index()                  # station id becomes a column
    stations_pdf = stations_pdf[["id", "latitude", "longitude"]].dropna()
    stations_pdf["id"] = stations_pdf["id"].astype(str)

    (spark.createDataFrame(stations_pdf)
        .write.mode("overwrite")
        .parquet(str(STATIONS_PARQUET)))

    print(f"[a] wrote {len(stations_pdf)} stations → {STATIONS_PARQUET}")

def load_accidents_and_vehicles(
    spark: SparkSession, raw_dir: Path
) -> tuple[DataFrame, DataFrame]:
    """
    Read both source CSVs with Spark (distributed, parallel splits).
    """
    print(f"[c] reading CSVs from {raw_dir}")

    # converts string → date will be needed for the weather join later, so parse dates on read
    accidents = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("encoding", "iso-8859-1")      
        .csv(str(raw_dir / ACCIDENTS_CSV))
        .withColumn("Date", F.to_date("Date", "yyyy-MM-dd"))
    )

    vehicles = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("encoding", "iso-8859-1")      # cp1252 → iso-8859-1 (compatible for this data)
        .csv(str(raw_dir / VEHICLES_CSV))
    )

    print(f"[c] accidents partitions: {accidents.rdd.getNumPartitions()}, "
        f"vehicles partitions: {vehicles.rdd.getNumPartitions()}")
    return accidents, vehicles

def attach_nearest_station(spark: SparkSession, accidents: DataFrame) -> DataFrame:
    """
    Compute nearest station entirely on the driver using pandas + numpy,
    then join the result back as a Spark column. No cross join, no OOM.
    
    Why this works: the haversine matrix is (2M accidents × 156 stations)
    but we only need argmin per row — pandas does this in ~2 seconds.
    The result is a tiny 2-column DataFrame (Accident_Index → station_id)
    that we join back to accidents.
    """
    import numpy as np
    import pandas as pd

    print("[d] attaching nearest_station via pandas haversine on driver ...")

    # Load both as pandas — stations is tiny, accidents index is just 2 columns
    stations_pd = spark.read.parquet(str(STATIONS_PARQUET)).toPandas()
    acc_pd = (
        accidents
        .select("Accident_Index", "Latitude", "Longitude")
        .toPandas()
    )

    # Vectorized haversine: (N_accidents, N_stations) matrix
    s_lat = np.radians(stations_pd["latitude"].values)
    s_lon = np.radians(stations_pd["longitude"].values)
    a_lat = np.radians(pd.to_numeric(acc_pd["Latitude"], errors="coerce").fillna(0).values)[:, None]
    a_lon = np.radians(pd.to_numeric(acc_pd["Longitude"], errors="coerce").fillna(0).values)[:, None]

    dlat = s_lat - a_lat
    dlon = s_lon - a_lon
    h = (np.sin(dlat / 2) ** 2
         + np.cos(a_lat) * np.cos(s_lat) * np.sin(dlon / 2) ** 2)
    idx = np.argmin(h, axis=1)

    acc_pd["station_id"] = stations_pd["id"].values[idx]
    # Null out rows where lat/lon was missing
    missing = pd.to_numeric(acc_pd["Latitude"], errors="coerce").isna() | \
          pd.to_numeric(acc_pd["Longitude"], errors="coerce").isna()
    acc_pd.loc[missing, "station_id"] = None

    # Join the station_id column back to the full accidents DataFrame
    station_map = spark.createDataFrame(
        acc_pd[["Accident_Index", "station_id"]]
    )
    return accidents.join(station_map, on="Accident_Index", how="left")

def join_weather(spark: SparkSession, accidents: DataFrame) -> DataFrame:
    # If weather fetch was skipped (Meteostat unreliable), attach NULL columns
    # so downstream code keeps the same schema and Imputer fills them later.
    if not STATION_WEATHER_PARQUET.exists():
        print("[e] no weather parquet — attaching NULL weather columns (median-imputed later)")
        for col, dtype in [
            ("tavg", "double"), ("tmin", "double"), ("tmax", "double"),
            ("prcp", "double"), ("snow", "double"), ("wspd", "double"),
            ("pres", "double"),
        ]:
            accidents = accidents.withColumn(col, F.lit(None).cast(dtype))
        return accidents

    weather = spark.read.parquet(str(STATION_WEATHER_PARQUET))
    print("[e] joining weather on (station_id, Date)")
    # Left join: 
    # keep all accidents
    # attach weather if exists
    return (
        accidents.alias("a")
        .join(
            weather.alias("w"),
            (F.col("a.station_id") == F.col("w.station_id"))
            & (F.col("a.Date") == F.col("w.time")),
            how="left",
        )
        .drop(F.col("w.station_id"))
        .drop(F.col("w.time"))
    )


def join_vehicles_and_write(acc_weather: DataFrame, vehicles: DataFrame) -> None:
    merged = (acc_weather.join(vehicles, on="Accident_Index", how="left").drop(vehicles["Year"]))

    n_files_per_year = os.cpu_count() or 8
    print(f"[f] writing {MERGED_PARQUET} "
          f"(repartition={n_files_per_year}, partitionBy=Year)")

    (merged
        .repartition(n_files_per_year)
        .write.mode("overwrite")
        .partitionBy("Year")
        .parquet(str(MERGED_PARQUET)))


# Orchastration
def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    spark = get_spark("ingest")
    spark.sparkContext.setLogLevel("WARN")

    build_stations_parquet(spark)
    try:
        build_station_weather_parquet(spark)
    except Exception as e:
        # Meteostat's CDN is unreliable. If fetching weather fails
        # catastrophically, don't kill the whole pipeline — join_weather()
        # below will fall back to NULL weather columns.
        print(f"[b] WARNING: weather fetch failed ({e!r}); continuing without weather")

    # Main distributed pipeline
    raw_dir = download_dataset()
    accidents, vehicles = load_accidents_and_vehicles(spark, raw_dir)

    # ── Checkpoint after the pandas_udf ────────────────────────────────
    # Writing accidents+station_id to parquet here isolates the expensive
    # (and Windows-fragile) nearest-station UDF stage from the later
    # joins. If stage [e]/[f] crash, we don't re-run the 30-minute UDF;
    # we just re-read this checkpoint.
    if not ACCIDENTS_WITH_STATION_PARQUET.exists():
        print("[d] running nearest-station UDF and checkpointing ...")
        accidents = attach_nearest_station(spark, accidents)
        (accidents
            .write.mode("overwrite")
            .parquet(str(ACCIDENTS_WITH_STATION_PARQUET)))
        print(f"[d] checkpoint written → {ACCIDENTS_WITH_STATION_PARQUET}")
    else:
        print(f"[d] checkpoint exists — skipping UDF, reading {ACCIDENTS_WITH_STATION_PARQUET}")

    accidents   = spark.read.parquet(str(ACCIDENTS_WITH_STATION_PARQUET))
    acc_weather = join_weather(spark, accidents)
    join_vehicles_and_write(acc_weather, vehicles)

    # Quick sanity check on the written output to ensure everything is as expected 
    out = spark.read.parquet(str(MERGED_PARQUET))
    print(f"[done] merged rows: {out.count():,}, columns: {len(out.columns)}")

    spark.stop()


if __name__ == "__main__":
    main()