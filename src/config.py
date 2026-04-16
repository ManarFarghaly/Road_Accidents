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

# Ingestion artifacts 
STATIONS_PARQUET        = INTERIM_DIR / "stations.parquet"
STATION_WEATHER_PARQUET = INTERIM_DIR / "station_weather.parquet"
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

    local[*]  — * use every available core as a separate JVM worker thread.
    6 g driver memory — enough to hold the broadcast stations table and
                        materialize small collect()s without OOM.
    Aphache Arrow — faster Pandas UDFs and collect()s , Spark ↔ pandas conversions data transfer
    shuffle_partitions ≈ 4–8 × number of cores — avoid too many small tasks and too much overhead, but still allow parallelism.
    default middle ground ram is 8g so we set it to 6g to leave some headroom for the OS and other processes.
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "6g")
        .config("spark.hadoop.io.native.lib.available", "false")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
        .config("spark.hadoop.fs.file.impl.disable.cache", "true")
    .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .getOrCreate()
    )
