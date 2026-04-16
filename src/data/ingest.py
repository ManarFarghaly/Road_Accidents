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

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import meteostat as ms

from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.functions import pandas_udf

from src.config import (
    ACCIDENTS_CSV,
    INTERIM_DIR,
    MERGED_PARQUET,
    STATIONS_PARQUET,
    STATION_WEATHER_PARQUET,
    VEHICLES_CSV,
    WEATHER_END,
    WEATHER_START,
    get_spark,
)
from src.data.acquire import download_dataset


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
    # Central UK point (roughly Stoke-on-Trent) — 500 km radius covers
    # all of England, Wales, Scotland
    UK_CENTER = ms.Point(52.83, -1.83)
    stations_pdf = ms.stations.nearby(UK_CENTER, radius=500_000, limit=500)

    # Filter to GB-only stations (the radius may pick up some Irish/French ones)
    stations_pdf = stations_pdf[stations_pdf["country"] == "GB"].copy()
    stations_pdf = stations_pdf.reset_index()                  # station id becomes a column
    stations_pdf = stations_pdf.rename(columns={"id": "id"})   # already named 'id' after reset
    stations_pdf = stations_pdf[["id", "latitude", "longitude"]].dropna()
    stations_pdf["id"] = stations_pdf["id"].astype(str)

    (spark.createDataFrame(stations_pdf)
        .write.mode("overwrite")
        .parquet(str(STATIONS_PARQUET)))

    print(f"[a] wrote {len(stations_pdf)} stations → {STATIONS_PARQUET}")


# ════════════════════════════════════════════════════════════════════════════
# Stage b — Per-station daily weather (one-time fetch from Meteostat API)
# ════════════════════════════════════════════════════════════════════════════
def build_station_weather_parquet(spark: SparkSession) -> None:
    """
    For each GB station, fetch daily weather over the accidents date range
    (2005-01-01 to 2017-12-31), flatten into one long DataFrame, and write
    as Parquet.

    This is a pandas loop because the Meteostat API is I/O-bound (HTTP calls),
    not CPU-bound — Spark would not make it faster. It runs ONCE; the result
    is cached as Parquet and every subsequent run reads that parquet directly.
    """
    if STATION_WEATHER_PARQUET.exists():
        print(f"[b] {STATION_WEATHER_PARQUET} already exists — skipping")
        return

    stations_pdf = spark.read.parquet(str(STATIONS_PARQUET)).toPandas()
    start = datetime.fromisoformat(WEATHER_START)
    end   = datetime.fromisoformat(WEATHER_END)

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    print(f"[b] fetching daily weather for {len(stations_pdf)} stations "
          f"({WEATHER_START} → {WEATHER_END}) ...")
    for _, row in tqdm(stations_pdf.iterrows(), total=len(stations_pdf), desc="meteostat"):
        station_id = row["id"]
        try:
            # meteostat v2 API: ms.daily(ms.Station(id=...), start, end)
            ts = ms.daily(ms.Station(id=station_id), start, end)
            df = ts.fetch().reset_index()
            if df is None or df.empty:
                continue
            df["station_id"] = station_id
            frames.append(df)
        except Exception:
            failed.append(station_id)
            continue

    print(f"[b] fetched {len(frames)} stations ({len(failed)} failed)")

    if not frames:
        raise RuntimeError(
            "No weather data fetched for any station. "
            "Check internet connection / Meteostat API availability."
        )

    weather_pdf = pd.concat(frames, ignore_index=True)
    weather_pdf["time"] = pd.to_datetime(weather_pdf["time"]).dt.date

    # Cast any Int64/Float64 NaN-bearing cols to float (Spark can't handle pandas NA)
    for col in weather_pdf.columns:
        if col not in ("station_id", "time"):
            weather_pdf[col] = pd.to_numeric(weather_pdf[col], errors="coerce").astype("float64")

    (spark.createDataFrame(weather_pdf)
        .write.mode("overwrite")
        .parquet(str(STATION_WEATHER_PARQUET)))

    print(f"[b] wrote {len(weather_pdf):,} weather rows → {STATION_WEATHER_PARQUET}")
    del weather_pdf, frames


# ════════════════════════════════════════════════════════════════════════════
# Stage c — Parallel CSV reads
# ════════════════════════════════════════════════════════════════════════════
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
        .option("encoding", "latin1")
        .csv(str(raw_dir / ACCIDENTS_CSV))
        .withColumn("Date", F.to_date("Date", "yyyy-MM-dd"))
    )

    vehicles = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("encoding", "windows-1252")
        .csv(str(raw_dir / VEHICLES_CSV))
    )

    print(f"[c] accidents partitions: {accidents.rdd.getNumPartitions()}, "
          f"vehicles partitions: {vehicles.rdd.getNumPartitions()}")
    return accidents, vehicles


# ════════════════════════════════════════════════════════════════════════════
# Stage d — Nearest-station lookup (broadcast + vectorized pandas_udf)
# ════════════════════════════════════════════════════════════════════════════
def attach_nearest_station(spark: SparkSession, accidents: DataFrame) -> DataFrame:
    """
    For every accident row, compute the nearest Meteostat station using a
    vectorized pandas_udf that does the haversine math as a single NumPy
    (batch_size, n_stations) matrix operation per partition.

    pandas UDF
    A pandas UDF (user-defined function) is a vectorized function feature in Apache Spark that uses pandas and 
    Apache Arrow to efficiently apply custom Python logic to distributed data. It enables high-performance data
    transformations in PySpark by processing batches of rows as pandas objects instead of individual records.
    How it works
    A pandas UDF operates by converting Spark’s columnar data into pandas Series or DataFrames using Apache Arrow,
    executing the user’s Python function on these batches, and then converting the results back to Spark’s 
    internal format.This batch-based design minimizes serialization overhead and improves performance compared 
    to traditional row-wise Python UDFs.

    Logic:
    1- Broadcast the small stations reference DataFrame to all executors (happens once per job).
    2- Define a pandas UDF that takes batches of accident lat/lon as input, computes the haversine distance to all
       stations in a vectorized manner, and returns the nearest station ID for each accident.
    3- Apply this UDF to the accidents DataFrame, creating a new "station_id" column with the nearest station for each accident.

    Note: This approach is efficient because the stations data is small enough to fit in memory and be broadcasted,
    and the haversine calculation is done in a vectorized way using NumPy, which is much faster than row-wise UDFs.

    - The haversine formula is a mathematical equation used to calculate the great-circle distance between two points on 
    the surface of a sphere, given their latitudes and longitudes. It is commonly used in navigation and geospatial
    applications to determine the shortest distance between two locations on Earth. The formula accounts for the curvature
    of the Earth, providing an accurate distance measurement in kilometers or miles.

    """
    stations_pdf = spark.read.parquet(str(STATIONS_PARQUET)).toPandas()

    # Broadcast tiny stations arrays to every executor — happens ONCE per job
    stations_bc = spark.sparkContext.broadcast({
        "id":  stations_pdf["id"].to_numpy(),
        "lat": np.radians(stations_pdf["latitude"].to_numpy(dtype=float)),
        "lon": np.radians(stations_pdf["longitude"].to_numpy(dtype=float)),
    })

    @pandas_udf("string")
    def nearest_station(lat: pd.Series, lon: pd.Series) -> pd.Series:
        s = stations_bc.value
        s_lat = s["lat"][None, :]           # shape (1, S)
        s_lon = s["lon"][None, :]
        ids   = s["id"]

        valid = lat.notna() & lon.notna()
        a_lat = np.radians(lat.to_numpy(dtype=float))[:, None]   # shape (B, 1)
        a_lon = np.radians(lon.to_numpy(dtype=float))[:, None]

        # Haversine across the full (B, S) matrix in one NumPy shot
        dlat = s_lat - a_lat
        dlon = s_lon - a_lon
        h    = np.sin(dlat / 2) ** 2 + np.cos(a_lat) * np.cos(s_lat) * np.sin(dlon / 2) ** 2
        idx  = np.argmin(h, axis=1)         # nearest station index per accident

        result = pd.Series(ids[idx], index=lat.index)
        result[~valid] = None
        return result

    print("[d] attaching nearest_station via pandas_udf ...")
    return accidents.withColumn(
        "station_id", nearest_station(F.col("Latitude"), F.col("Longitude"))
    )


# ════════════════════════════════════════════════════════════════════════════
# Stage e — Weather join (plain two-column distributed join)
# ════════════════════════════════════════════════════════════════════════════
def join_weather(spark: SparkSession, accidents: DataFrame) -> DataFrame:
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


# ════════════════════════════════════════════════════════════════════════════
# Stage f — Join vehicles, write partitioned Parquet
# ════════════════════════════════════════════════════════════════════════════
def join_vehicles_and_write(acc_weather: DataFrame, vehicles: DataFrame) -> None:
    merged = acc_weather.join(vehicles, on="Accident_Index", how="left")

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
    build_station_weather_parquet(spark)

    # Main distributed pipeline
    raw_dir = download_dataset()
    accidents, vehicles = load_accidents_and_vehicles(spark, raw_dir)
    accidents   = attach_nearest_station(spark, accidents)
    acc_weather = join_weather(spark, accidents)
    join_vehicles_and_write(acc_weather, vehicles)

    # Quick sanity check on the written output to ensure everything is as expected 
    out = spark.read.parquet(str(MERGED_PARQUET))
    print(f"[done] merged rows: {out.count():,}, columns: {len(out.columns)}")

    spark.stop()


if __name__ == "__main__":
    main()
