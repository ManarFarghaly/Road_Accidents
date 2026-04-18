"""
Shared configuration for the UK Road Accidents project.

All modules import `get_spark()` from here so every script uses
identical Spark settings (pseudo-distributed local[*] mode).
"""
from pathlib import Path

from pyspark.sql import SparkSession


# Project paths 
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
INTERIM_DIR   = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR   = PROJECT_ROOT / "reports"
WEATHER_RAW_DIR = INTERIM_DIR / "weather_raw"

# Ingestion artifacts 
STATIONS_PARQUET        = INTERIM_DIR / "stations.parquet"
STATION_WEATHER_PARQUET = INTERIM_DIR / "station_weather.parquet"
# Checkpoint after the pandas_udf stage — avoids re-running the expensive
# 2M-row nearest-station UDF if the weather/vehicles join later crashes.
ACCIDENTS_WITH_STATION_PARQUET = INTERIM_DIR / "accidents_with_station.parquet"
MERGED_PARQUET          = INTERIM_DIR / "merged.parquet"

ACCIDENTS_CSV = "Accident_Information.csv"
VEHICLES_CSV  = "Vehicle_Information.csv"

# Weather date range (matches the DfT accidents dataset) 
WEATHER_START = "2005-01-01"
WEATHER_END   = "2017-12-31"

# It is added here as it is required by multiple modules, and it is important to keep 
# the Spark settings consistent across all scripts.

def get_spark(app_name: str = "road-accidents") -> SparkSession:
    """
    Build (or return the existing) Spark session in pseudo-distributed mode.

    local[*]  — use every available core as a separate JVM worker thread.
    4 g driver memory — leaves ~4 GB for the OS + Python on an 8 GB machine.
    Apache Arrow — faster pandas UDF and Spark-to-pandas conversions.
    shuffle_partitions = 32 — ~4x cores on an i5 (avoids overhead of 200 default).
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        # Smaller Arrow batches — pandas_udf workers OOM/crash less often on
        # Windows + Py3.10 with the default 10 000-row batch. 2 000 keeps
        # per-batch peak memory small while still being vectorized.
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "2000")
        # When a Python worker crashes, surface the real traceback instead
        # of just "Connection reset by peer". Essential for debugging UDFs.
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
        .config("spark.hadoop.fs.file.impl.disable.cache", "true")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .config("spark.executor.heartbeatInterval", "60s")   # was 10s default
        .config("spark.network.timeout", "600s")             # was 120s default
        .config("spark.sql.broadcastTimeout", "600")
        .getOrCreate()
    )
