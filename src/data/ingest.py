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
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from datetime import datetime
from pathlib import Path

# Set Python executable path for Spark workers BEFORE importing pyspark
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

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

    # Meteostat bulk files have INCONSISTENT schemas across stations and years:
    #   some have: year,month,day,temp,tmin,tmax,prcp,snwd,wspd,pres
    #   some have: year,month,day,temp,tmin,tmax,prcp,wspd,wpgt,tsun   (no snwd/pres)
    #   some have: year,month,day,temp,tmin,tmax,prcp,wspd,pres        (no snwd)
    # Spark's CSV reader fails with CSVHeaderChecker when files have different headers.
    # Fix: read each file in pandas to normalize columns, then build one Spark DataFrame.
    print(f"[b] reading {len(gz_files)} station files via pandas (schema normalization) ...")

    frames = []
    errors = 0
    for f in gz_files:
        # Extract station_id from filename pattern: {station_id}_{year}.csv.gz
        sid = f.stem.rsplit("_", 1)[0]
        try:
            df = pd.read_csv(f, compression="gzip")
            df.columns = df.columns.str.strip()
            # Normalize to fixed column set — missing columns become NaN
            df["station_id"] = sid
            df["tavg"] = pd.to_numeric(df.get("temp"), errors="coerce").astype("float32")
            df["tmin"] = pd.to_numeric(df.get("tmin"), errors="coerce").astype("float32")
            df["tmax"] = pd.to_numeric(df.get("tmax"), errors="coerce").astype("float32")
            df["prcp"] = pd.to_numeric(df.get("prcp"), errors="coerce").astype("float32")
            df["snow"] = pd.to_numeric(df.get("snwd"), errors="coerce").astype("float32")
            df["wspd"] = pd.to_numeric(df.get("wspd"), errors="coerce").astype("float32")
            df["pres"] = pd.to_numeric(df.get("pres"), errors="coerce").astype("float32")
            df["time"] = pd.to_datetime(
                df["year"].astype(str) + "-" +
                df["month"].astype(str).str.zfill(2) + "-" +
                df["day"].astype(str).str.zfill(2),
                errors="coerce"
            )
            frames.append(
                df[["station_id", "time", "tavg", "tmin", "tmax", "prcp", "snow", "wspd", "pres"]]
                .dropna(subset=["time"])
            )
        except Exception:
            errors += 1

    print(f"[b] parsed {len(frames)} files ({errors} skipped with errors)")
    if not frames:
        raise RuntimeError("No weather data could be parsed from any downloaded file.")

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


# ════════════════════════════════════════════════════════════════════════════
# Stage b — Per-station daily weather (one-time fetch from Meteostat API)
# ════════════════════════════════════════════════════════════════════════════
# def _fetch_one_station(
#     station_id: str,
#     start: datetime,
#     end: datetime,
#     max_retries: int = 3,
# ) -> pd.DataFrame | None:
#     """
#     Fetch daily weather for a single station with retry + exponential backoff.
#     Returns a DataFrame on success, or None on permanent failure.
#     """
#     for attempt in range(1, max_retries + 1):
#         try:
#             # v2 API first, fall back to v1 on AttributeError
#             try:
#                 ts = ms.daily(ms.Station(id=station_id), start, end)
#             except AttributeError:
#                 ts = ms.Daily(station_id, start, end)     # v1
#             df = ts.fetch().reset_index()
#             if df is None or df.empty:
#                 return None
#             df["station_id"] = station_id
#             return df
#         except Exception:
#             if attempt < max_retries:
#                 # 5s, 10s, 20s — longer than before; Meteostat is rate-limiting
#                 time.sleep(5 * (2 ** (attempt - 1)))
#             else:
#                 return None


# # Path to a JSON file tracking which stations have been fetched already
# # so that re-runs continue from where they stopped instead of restarting.
# _PROGRESS_FILE = INTERIM_DIR / "_weather_progress.json"
# _PARTIAL_FILE  = INTERIM_DIR / "_weather_partial.pkl"


# def build_station_weather_parquet(spark: SparkSession) -> None:
#     """
#     For each GB station, fetch daily weather over the accidents date range
#     (2005-01-01 to 2017-12-31), flatten into one long DataFrame, and write
#     as Parquet.

#     Resilience features (because Meteostat can be flaky):
#       - 30-second socket timeout (set globally above).
#       - 3 retries with exponential backoff per station.
#       - Progress is saved to disk after every batch of 10 stations.
#         If the script is killed or crashes, re-running it continues
#         from the last saved checkpoint instead of re-fetching everything.
#       - 4 parallel threads for the I/O-bound HTTP fetches (~4x speedup).
#     """
#     if STATION_WEATHER_PARQUET.exists():
#         print(f"[b] {STATION_WEATHER_PARQUET} already exists — skipping")
#         return

#     stations_pdf = spark.read.parquet(str(STATIONS_PARQUET)).toPandas()
#     start = datetime.fromisoformat(WEATHER_START)
#     end   = datetime.fromisoformat(WEATHER_END)

#     # ── Resume from partial progress if available ──────────────────────
#     done_ids: set[str] = set()
#     frames: list[pd.DataFrame] = []

#     if _PARTIAL_FILE.exists() and _PROGRESS_FILE.exists():
#         with open(_PROGRESS_FILE) as f:
#             done_ids = set(json.load(f))
#         frames = [pd.read_pickle(_PARTIAL_FILE)]
#         print(f"[b] resuming — {len(done_ids)} stations already cached")

#     remaining = stations_pdf[~stations_pdf["id"].isin(done_ids)]
#     failed: list[str] = []

#     print(f"[b] fetching daily weather for {len(remaining)} stations "
#           f"({WEATHER_START} → {WEATHER_END})  [2 threads, 3 retries each] ...")

#     # ── Parallel fetch with ThreadPoolExecutor ─────────────────────────
#     # Reduced to 2 workers — Meteostat's CDN aggressively rate-limits and
#     # 4+ concurrent connections trigger ConnectionReset bursts.
#     batch_count = 0
#     with ThreadPoolExecutor(max_workers=2) as pool:
#         futures = {
#             pool.submit(_fetch_one_station, row["id"], start, end): row["id"]
#             for _, row in remaining.iterrows()
#         }
#         with tqdm(total=len(futures), desc="meteostat") as pbar:
#             for future in as_completed(futures):
#                 station_id = futures[future]
#                 try:
#                     df = future.result()
#                     if df is not None:
#                         frames.append(df)
#                     else:
#                         failed.append(station_id)
#                 except Exception:
#                     failed.append(station_id)

#                 done_ids.add(station_id)
#                 batch_count += 1
#                 pbar.update(1)

#                 # Save progress every 10 stations
#                 if batch_count % 10 == 0:
#                     _save_partial(frames, done_ids)

#     # ── Final save ─────────────────────────────────────────────────────
#     print(f"[b] fetched {len(frames)} station chunks "
#           f"({len(failed)} permanently failed)")

#     if not frames:
#         raise RuntimeError(
#             "No weather data fetched for any station. "
#             "Check internet connection / Meteostat API availability."
#         )

#     weather_pdf = pd.concat(frames, ignore_index=True)
#     weather_pdf["time"] = pd.to_datetime(weather_pdf["time"]).dt.date

#     # Cast NaN-bearing cols to float64 (Spark can't handle pandas nullable Int64)
#     for col in weather_pdf.columns:
#         if col not in ("station_id", "time"):
#             weather_pdf[col] = (
#                 pd.to_numeric(weather_pdf[col], errors="coerce")
#                 .astype("float64")
#             )

#     (spark.createDataFrame(weather_pdf)
#         .write.mode("overwrite")
#         .parquet(str(STATION_WEATHER_PARQUET)))

#     print(f"[b] wrote {len(weather_pdf):,} weather rows → {STATION_WEATHER_PARQUET}")

#     # Clean up progress files — they're no longer needed
#     _PROGRESS_FILE.unlink(missing_ok=True)
#     _PARTIAL_FILE.unlink(missing_ok=True)

#     del weather_pdf, frames


# def _save_partial(frames: list[pd.DataFrame], done_ids: set[str]) -> None:
#     """Save partial weather data + list of done station IDs to disk."""
#     if frames:
#         pd.concat(frames, ignore_index=True).to_pickle(str(_PARTIAL_FILE))
#     with open(_PROGRESS_FILE, "w") as f:
#         json.dump(list(done_ids), f)


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


# ════════════════════════════════════════════════════════════════════════════
# Pure pandas versions to avoid Spark distributed shuffle crashes
# ════════════════════════════════════════════════════════════════════════════
def attach_nearest_station_pandas(acc_pd):
    """Attach nearest station ID to accidents using pure pandas (on driver)."""
    import numpy as np
    import pandas as pd
    
    # Read stations once
    stations_pd = pd.read_parquet(str(STATIONS_PARQUET))
    
    print("[d] computing nearest stations via haversine...")
    # Vectorized haversine with batching
    s_lat = np.radians(stations_pd["latitude"].values)
    s_lon = np.radians(stations_pd["longitude"].values)
    a_lat_vals = np.radians(pd.to_numeric(acc_pd["Latitude"], errors="coerce").fillna(0).values)
    a_lon_vals = np.radians(pd.to_numeric(acc_pd["Longitude"], errors="coerce").fillna(0).values)

    batch_size = 50000
    idx_list = []
    
    for i, batch_start in enumerate(range(0, len(a_lat_vals), batch_size)):
        if i % 20 == 0:
            print(f"    batch {i+1}/{(len(a_lat_vals)-1)//batch_size + 1}")
        batch_end = min(batch_start + batch_size, len(a_lat_vals))
        a_lat = a_lat_vals[batch_start:batch_end][:, None]
        a_lon = a_lon_vals[batch_start:batch_end][:, None]

        dlat = s_lat - a_lat
        dlon = s_lon - a_lon
        h = (np.sin(dlat / 2) ** 2
             + np.cos(a_lat) * np.cos(s_lat) * np.sin(dlon / 2) ** 2)
        idx_list.extend(np.argmin(h, axis=1))

    acc_pd = acc_pd.copy()
    acc_pd["station_id"] = stations_pd["id"].values[idx_list]
    
    # Null out rows where lat/lon was missing
    missing = pd.to_numeric(acc_pd["Latitude"], errors="coerce").isna() | \
              pd.to_numeric(acc_pd["Longitude"], errors="coerce").isna()
    acc_pd.loc[missing, "station_id"] = None
    
    return acc_pd


def join_weather_pandas(spark, acc_pd):
    """Join weather data using pure pandas."""
    import pandas as pd
    
    if not STATION_WEATHER_PARQUET.exists():
        print("[e] no weather parquet — attaching NULL weather columns")
        for col in ["tavg", "tmin", "tmax", "prcp", "snow", "wspd", "pres"]:
            acc_pd[col] = None
        return acc_pd

    print("[e] joining weather on (station_id, Date) via pandas...")
    weather_pd = pd.read_parquet(str(STATION_WEATHER_PARQUET))
    
    # Merge on station_id and date
    acc_weather_pd = acc_pd.merge(
        weather_pd,
        left_on=["station_id", "Date"],
        right_on=["station_id", "time"],
        how="left"
    ).drop(columns=["time"], errors="ignore")
    
    return acc_weather_pd


def join_vehicles_and_write_pandas(spark, acc_weather_pd, vehicles_pd):
    """Join vehicles and write using pandas first, then Spark for parquet."""
    import pandas as pd
    
    print("[f] merging accidents+weather+vehicles via pandas...")
    
    # Merge
    merged_pd = acc_weather_pd.merge(vehicles_pd, on="Accident_Index", how="left")
    
    # Handle duplicate Year columns
    if "Year_y" in merged_pd.columns:
        merged_pd = merged_pd.drop(columns=["Year_y"])
    if "Year_x" in merged_pd.columns:
        merged_pd = merged_pd.rename(columns={"Year_x": "Year"})
    
    print(f"[f] writing {MERGED_PARQUET} (partitionBy=Year)")
    merged_spark = spark.createDataFrame(merged_pd)
    (merged_spark
         .write.mode("overwrite")
         .partitionBy("Year")
         .parquet(str(MERGED_PARQUET)))


# ════════════════════════════════════════════════════════════════════════════
# Stage d — Nearest-station lookup (broadcast + vectorized pandas_udf)
# ════════════════════════════════════════════════════════════════════════════
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

    print("[d] attaching nearest_station via pandas_udf (distributed) ...")

    stations_pd = spark.read.parquet(str(STATIONS_PARQUET)).toPandas()

    # Broadcast tiny stations arrays to every executor once.
    stations_bc = spark.sparkContext.broadcast({
        "id": stations_pd["id"].to_numpy(),
        "lat": np.radians(stations_pd["latitude"].to_numpy(dtype=float)),
        "lon": np.radians(stations_pd["longitude"].to_numpy(dtype=float)),
    })

    @pandas_udf("string")
    def nearest_station(lat: pd.Series, lon: pd.Series) -> pd.Series:
        s = stations_bc.value
        s_lat = s["lat"][None, :]
        s_lon = s["lon"][None, :]
        ids = s["id"]

        valid = lat.notna() & lon.notna()
        a_lat = np.radians(lat.fillna(0).to_numpy(dtype=float))[:, None]
        a_lon = np.radians(lon.fillna(0).to_numpy(dtype=float))[:, None]

        dlat = s_lat - a_lat
        dlon = s_lon - a_lon
        h = np.sin(dlat / 2) ** 2 + np.cos(a_lat) * np.cos(s_lat) * np.sin(dlon / 2) ** 2
        idx = np.argmin(h, axis=1)

        out = pd.Series(ids[idx], index=lat.index)
        out[~valid] = None
        return out

    return accidents.withColumn("station_id", nearest_station(F.col("Latitude"), F.col("Longitude")))

# def attach_nearest_station(spark: SparkSession, accidents: DataFrame) -> DataFrame:
#     """
#     For every accident row, compute the nearest Meteostat station using a
#     vectorized pandas_udf that does the haversine math as a single NumPy
#     (batch_size, n_stations) matrix operation per partition.

#     pandas UDF
#     A pandas UDF (user-defined function) is a vectorized function feature in Apache Spark that uses pandas and 
#     Apache Arrow to efficiently apply custom Python logic to distributed data. It enables high-performance data
#     transformations in PySpark by processing batches of rows as pandas objects instead of individual records.
#     How it works
#     A pandas UDF operates by converting Spark’s columnar data into pandas Series or DataFrames using Apache Arrow,
#     executing the user’s Python function on these batches, and then converting the results back to Spark’s 
#     internal format.This batch-based design minimizes serialization overhead and improves performance compared 
#     to traditional row-wise Python UDFs.

#     Logic:
#     1- Broadcast the small stations reference DataFrame to all executors (happens once per job).
#     2- Define a pandas UDF that takes batches of accident lat/lon as input, computes the haversine distance to all
#        stations in a vectorized manner, and returns the nearest station ID for each accident.
#     3- Apply this UDF to the accidents DataFrame, creating a new "station_id" column with the nearest station for each accident.

#     Note: This approach is efficient because the stations data is small enough to fit in memory and be broadcasted,
#     and the haversine calculation is done in a vectorized way using NumPy, which is much faster than row-wise UDFs.

#     - The haversine formula is a mathematical equation used to calculate the great-circle distance between two points on 
#     the surface of a sphere, given their latitudes and longitudes. It is commonly used in navigation and geospatial
#     applications to determine the shortest distance between two locations on Earth. The formula accounts for the curvature
#     of the Earth, providing an accurate distance measurement in kilometers or miles.

#     """
#     stations_pdf = spark.read.parquet(str(STATIONS_PARQUET)).toPandas()

#     # Broadcast tiny stations arrays to every executor — happens ONCE per job
#     stations_bc = spark.sparkContext.broadcast({
#         "id":  stations_pdf["id"].to_numpy(),
#         "lat": np.radians(stations_pdf["latitude"].to_numpy(dtype=float)),
#         "lon": np.radians(stations_pdf["longitude"].to_numpy(dtype=float)),
#     })

#     @pandas_udf("string")
#     def nearest_station(lat: pd.Series, lon: pd.Series) -> pd.Series:
#         s = stations_bc.value
#         s_lat = s["lat"][None, :]           # shape (1, S)
#         s_lon = s["lon"][None, :]
#         ids   = s["id"]

#         valid = lat.notna() & lon.notna()
#         a_lat = np.radians(lat.to_numpy(dtype=float))[:, None]   # shape (B, 1)
#         a_lon = np.radians(lon.to_numpy(dtype=float))[:, None]

#         # Haversine across the full (B, S) matrix in one NumPy shot
#         dlat = s_lat - a_lat
#         dlon = s_lon - a_lon
#         h    = np.sin(dlat / 2) ** 2 + np.cos(a_lat) * np.cos(s_lat) * np.sin(dlon / 2) ** 2
#         idx  = np.argmin(h, axis=1)         # nearest station index per accident

#         result = pd.Series(ids[idx], index=lat.index)
#         result[~valid] = None
#         return result

#     print("[d] attaching nearest_station via pandas_udf ...")
#     return accidents.withColumn(
#         "station_id", nearest_station(F.col("Latitude"), F.col("Longitude"))
#     )


# ════════════════════════════════════════════════════════════════════════════
# Stage e — Weather join (pure pandas to avoid Spark shuffle)
# ════════════════════════════════════════════════════════════════════════════
def join_weather(spark: SparkSession, accidents: DataFrame) -> DataFrame:
    if not STATION_WEATHER_PARQUET.exists():
        print("[e] no weather parquet — attaching NULL weather columns (median-imputed later)")
        for col, dtype in [
            ("tavg", "double"), ("tmin", "double"), ("tmax", "double"),
            ("prcp", "double"), ("snow", "double"), ("wspd", "double"),
            ("pres", "double"),
        ]:
            accidents = accidents.withColumn(col, F.lit(None).cast(dtype))
        return accidents

    # Try to read weather, but if it's corrupted/empty, attach NULLs instead
    try:
        weather = spark.read.parquet(str(STATION_WEATHER_PARQUET))
        print("[e] joining weather on (station_id, Date)")
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
    except Exception as e:
        print(f"[e] weather parquet corrupted ({e!r}) — attaching NULL weather columns")
        for col, dtype in [
            ("tavg", "double"), ("tmin", "double"), ("tmax", "double"),
            ("prcp", "double"), ("snow", "double"), ("wspd", "double"),
            ("pres", "double"),
        ]:
            accidents = accidents.withColumn(col, F.lit(None).cast(dtype))
        return accidents


# ════════════════════════════════════════════════════════════════════════════
# Stage f — Join vehicles, write partitioned Parquet
# ════════════════════════════════════════════════════════════════════════════
def join_vehicles_and_write(acc_weather: DataFrame, vehicles: DataFrame) -> None:
    print("[f] joining accidents+weather+vehicles fully in Spark ...")

    vehicles_for_join = vehicles.drop("Year") if "Year" in vehicles.columns else vehicles
    merged = acc_weather.join(vehicles_for_join, on="Accident_Index", how="left")

    # Pre-shuffle by Year to spread the load evenly across 32 partitions BEFORE write.
    # Write directly without additional partitionBy to avoid extra overhead.
    print(f"[f] writing {MERGED_PARQUET} (repartition to 32 by Year, then direct write) ...")
    (merged
         .repartition(32, F.col("Year"))
         .write.mode("overwrite")
         .parquet(str(MERGED_PARQUET)))


# Orchastration
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

    build_stations_parquet(spark)
    try:
        build_station_weather_parquet(spark)
    except Exception as e:
        print(f"[b] WARNING: weather fetch failed ({e!r}); continuing without weather")

    # Original distributed Spark pipeline
    raw_dir = download_dataset()
    accidents, vehicles = load_accidents_and_vehicles(spark, raw_dir)

    # ── Checkpoint after the nearest-station stage ────────────────────────
    if not ACCIDENTS_WITH_STATION_PARQUET.exists():
        print("[d] running nearest-station UDF and checkpointing ...")
        accidents = attach_nearest_station(spark, accidents)
        (accidents
            .write.mode("overwrite")
            .parquet(str(ACCIDENTS_WITH_STATION_PARQUET)))
        print(f"[d] checkpoint written → {ACCIDENTS_WITH_STATION_PARQUET}")
    else:
        print(f"[d] checkpoint exists — reading {ACCIDENTS_WITH_STATION_PARQUET}")

    accidents   = spark.read.parquet(str(ACCIDENTS_WITH_STATION_PARQUET))
    acc_weather = join_weather(spark, accidents)
    join_vehicles_and_write(acc_weather, vehicles)

    # Quick sanity check on the written output to ensure everything is as expected 
    out = spark.read.parquet(str(MERGED_PARQUET))
    print(f"[done] merged rows: {out.count():,}, columns: {len(out.columns)}")

    spark.stop()


if __name__ == "__main__":
    main()