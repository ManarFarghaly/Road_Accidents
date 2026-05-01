from __future__ import annotations

import os
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from pyspark.sql.functions import pandas_udf
import meteostat as ms

socket.setdefaulttimeout(30)

from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, StructField, StructType
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


# ── Explicit schema for the accidents CSV ─────────────────────────────────
# inferSchema=True forces a full extra scan of the 2M-row file before the
# real read. Defining the schema here avoids that scan and prevents the
# occasional mis-typing of Latitude/Longitude as strings when the first
# rows happen to contain nulls.
# Columns not listed here default to StringType via PERMISSIVE mode.
ACCIDENTS_SCHEMA = StructType([
    StructField("Accident_Index",       StringType(),  nullable=True),
    StructField("Latitude",             DoubleType(),  nullable=True),
    StructField("Longitude",            DoubleType(),  nullable=True),
    StructField("Date",                 StringType(),  nullable=True),
    StructField("Day_of_Week",          StringType(),  nullable=True),
    StructField("Speed_limit",          IntegerType(), nullable=True),
    StructField("Number_of_Vehicles",   IntegerType(), nullable=True),
    StructField("Number_of_Casualties", IntegerType(), nullable=True),
    StructField("Year",                 IntegerType(), nullable=True),
    StructField("Accident_Severity",    StringType(),  nullable=True),
    StructField("Urban_or_Rural_Area",  StringType(),  nullable=True),
    StructField("Road_Type",            StringType(),  nullable=True),
    StructField("Junction_Detail",      StringType(),  nullable=True),
    StructField("Junction_Control",     StringType(),  nullable=True),
    StructField("Light_Conditions",     StringType(),  nullable=True),
    StructField("Weather_Conditions",   StringType(),  nullable=True),
    StructField("Road_Surface_Conditions", StringType(), nullable=True),
    StructField("LSOA_of_Accident_Location", StringType(), nullable=True),
])

# Columns not critical to type — left as StringType by inferSchema
VEHICLES_SCHEMA = StructType([
    StructField("Accident_Index",       StringType(),  nullable=True),
    StructField("Vehicle_Reference",    IntegerType(), nullable=True),
    StructField("Year",                 IntegerType(), nullable=True),
    StructField("Age_of_Vehicle",       DoubleType(),  nullable=True),
    StructField("Driver_IMD_Decile",    DoubleType(),  nullable=True),
])

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

def download_bulk_weather(station_ids: list[str]) -> None:
    """Download yearly .csv.gz files per station from the Meteostat bulk endpoint."""
    WEATHER_RAW_DIR.mkdir(parents=True, exist_ok=True)
    base_url = "https://data.meteostat.net/daily"
    years = range(2005, 2018)
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


def _normalise_gz(f: Path) -> pd.DataFrame | None:
    """
    Read one .csv.gz file and return a normalised DataFrame with fixed columns.

    Meteostat bulk files have three different layouts depending on the station
    and year. Spark's CSV reader assigns columns by position when ingesting
    multiple files together, silently misaligning data when layouts differ.
    Reading each file individually and mapping columns by name avoids this.
    Missing columns (e.g. 'snwd' in layout B) become NaN.
    """
    sid = f.stem.rsplit("_", 1)[0]
    try:
        df = pd.read_csv(f, compression="gzip")
        df.columns = df.columns.str.strip()
        out = pd.DataFrame()
        out["station_id"] = sid
        out["time"] = pd.to_datetime(
            df["year"].astype(str) + "-" +
            df["month"].astype(str).str.zfill(2) + "-" +
            df["day"].astype(str).str.zfill(2),
            errors="coerce",
        )
        out["tavg"] = pd.to_numeric(df.get("temp"), errors="coerce").astype("float32")
        out["tmin"] = pd.to_numeric(df.get("tmin"), errors="coerce").astype("float32")
        out["tmax"] = pd.to_numeric(df.get("tmax"), errors="coerce").astype("float32")
        out["prcp"] = pd.to_numeric(df.get("prcp"), errors="coerce").astype("float32")
        out["snow"] = pd.to_numeric(df.get("snwd"), errors="coerce").astype("float32")
        out["wspd"] = pd.to_numeric(df.get("wspd"), errors="coerce").astype("float32")
        out["pres"] = pd.to_numeric(df.get("pres"), errors="coerce").astype("float32")
        return out.dropna(subset=["time"])
    except Exception:
        return None


def build_station_weather_parquet(spark: SparkSession) -> None:
    """
    Normalise all downloaded .csv.gz files with pandas (one file at a time,
    so column names are read from each file's own header), write the result
    as intermediate parquet chunks, then let Spark read those chunks into the
    final weather parquet. This avoids the position-based column misalignment
    that occurs when Spark reads multiple CSVs with different headers together.
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

    # Write normalised data in chunks of 500 files to keep peak RAM bounded.
    # 500 files × ~365 rows × ~9 columns at float32 ≈ 6 MB per chunk — safe.
    CHUNK_SIZE = 500
    chunk_dir  = WEATHER_RAW_DIR / "_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    print(f"[b] normalising {len(gz_files)} files in chunks of {CHUNK_SIZE} ...")
    errors = 0
    for chunk_idx, start in enumerate(range(0, len(gz_files), CHUNK_SIZE)):
        chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.parquet"
        if chunk_path.exists():
            continue
        frames = []
        for f in gz_files[start: start + CHUNK_SIZE]:
            result = _normalise_gz(f)
            if result is not None:
                frames.append(result)
            else:
                errors += 1
        if not frames:
            continue
        chunk_pd = pd.concat(frames, ignore_index=True)
        chunk_pd = chunk_pd[
            (chunk_pd["time"] >= pd.Timestamp(WEATHER_START)) &
            (chunk_pd["time"] <= pd.Timestamp(WEATHER_END))
        ]
        chunk_pd["time"] = chunk_pd["time"].dt.date
        chunk_pd.to_parquet(str(chunk_path), index=False, engine="pyarrow")

    print(f"[b] chunks written ({errors} files skipped with errors) — "
          f"reading into Spark ...")

    chunk_files = list(chunk_dir.glob("chunk_*.parquet"))
    if not chunk_files:
        raise RuntimeError("No weather data could be normalised from any downloaded file.")

    # Spark reads parquet chunks — schema is consistent across all files
    # because pyarrow wrote them all with the same column set.
    (spark.read.parquet(*[str(f) for f in chunk_files])
         .write.mode("overwrite")
         .parquet(str(STATION_WEATHER_PARQUET)))

    print(f"[b] weather parquet written → {STATION_WEATHER_PARQUET}")


def load_accidents_and_vehicles(
    spark: SparkSession, raw_dir: Path
) -> tuple[DataFrame, DataFrame]:
    print(f"[c] reading CSVs from {raw_dir}")

    accidents = (
        spark.read
        .schema(ACCIDENTS_SCHEMA)
        .option("header", True)
        .option("mode", "PERMISSIVE")       # extra columns → null, not error
        .option("encoding", "iso-8859-1")
        .csv(str(raw_dir / ACCIDENTS_CSV))
        .withColumn("Date", F.to_date(F.col("Date"), "yyyy-MM-dd"))
    )

    # Vehicles schema covers only the typed columns we rely on; the rest
    # remain as strings, which is fine — clean.py casts them as needed.
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



def attach_nearest_station_vectorized(spark, accidents):

    stations_pd = spark.read.parquet(str(STATIONS_PARQUET)).toPandas()

    s_id = stations_pd["id"].values
    s_lat = np.radians(stations_pd["latitude"].values).astype(np.float32)
    s_lon = np.radians(stations_pd["longitude"].values).astype(np.float32)

    b_s_id = spark.sparkContext.broadcast(s_id)
    b_s_lat = spark.sparkContext.broadcast(s_lat)
    b_s_lon = spark.sparkContext.broadcast(s_lon)

    @pandas_udf("string")
    def find_nearest(lat_series: pd.Series, lon_series: pd.Series) -> pd.Series:

        result = pd.Series([None] * len(lat_series))

        mask = lat_series.notna() & lon_series.notna()
        if mask.sum() == 0:
            return result

        a_lat = np.radians(lat_series[mask].values).astype(np.float32)[:, None]
        a_lon = np.radians(lon_series[mask].values).astype(np.float32)[:, None]

        slat = b_s_lat.value
        slon = b_s_lon.value

        dlat = slat - a_lat
        dlon = slon - a_lon

        h = np.sin(dlat/2)**2 + np.cos(a_lat) * np.cos(slat) * np.sin(dlon/2)**2

        idx = np.argmin(h, axis=1)

        result.loc[mask] = b_s_id.value[idx]
        return result

    return accidents.withColumn(
        "station_id",
        find_nearest(accidents["Latitude"], accidents["Longitude"])
    )


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

    weather = spark.read.parquet(str(STATION_WEATHER_PARQUET))
    print("[e] joining weather on (station_id, Date) — broadcast join")

    # F.broadcast() tells the Spark planner to send the weather table to every
    # executor rather than shuffling 2M accident rows over the network.
    # Weather is ~740 k rows × 9 columns — well within broadcast threshold.
    return (
        accidents.alias("a")
        .join(
            F.broadcast(weather).alias("w"),
            (F.col("a.station_id") == F.col("w.station_id"))
            & (F.col("a.Date")       == F.col("w.time")),
            how="left",
        )
        .drop(F.col("w.station_id"))
        .drop(F.col("w.time"))
    )
    
def join_vehicles_and_write(acc_weather: DataFrame, vehicles: DataFrame) -> None:
    merged = acc_weather.join(vehicles, on="Accident_Index", how="left").drop(vehicles["Year"])

    # repartition(F.col("Year")) redistributes by Year so each output
    # partition contains only one year's data, giving clean directory splits.
    # Avoid the anti-pattern of repartition(N) followed by partitionBy("Year"),
    # which creates N × n_years small files instead of n_years files.
    print(f"[f] writing {MERGED_PARQUET} (repartition by Year, partitionBy=Year)")
    (merged
        .repartition(F.col("Year"))
        .write.mode("overwrite")
        .partitionBy("Year")
        .parquet(str(MERGED_PARQUET)))


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    spark = get_spark("ingest")
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10000)
    

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
    if not ACCIDENTS_WITH_STATION_PARQUET.exists():
        print("[d] running nearest-station lookup and checkpointing ...")
        accidents = attach_nearest_station_vectorized(spark, accidents)
        (accidents
            .write.mode("overwrite")
            .parquet(str(ACCIDENTS_WITH_STATION_PARQUET)))
        print(f"[d] checkpoint written → {ACCIDENTS_WITH_STATION_PARQUET}")
    else:
        print(f"[d] checkpoint exists — reading {ACCIDENTS_WITH_STATION_PARQUET}")

    accidents   = spark.read.parquet(str(ACCIDENTS_WITH_STATION_PARQUET))

    # Stage e — weather join
    acc_weather = join_weather(spark, accidents)

    # Cache here so the weather join DAG is not replayed when stage [f]
    # triggers the write action. Unpersist after write to free executor memory.
    acc_weather.cache()

    # Stage f — vehicle join + write
    join_vehicles_and_write(acc_weather, vehicles)
    acc_weather.unpersist()

    # Sanity check
    out = spark.read.parquet(str(MERGED_PARQUET))
    print(f"[done] merged rows: {out.count():,}, columns: {len(out.columns)}")

    spark.stop()


if __name__ == "__main__":
    main()