"""
Shared configuration for the UK Road Accidents project.

All modules import `get_spark()` from here so every script uses
identical Spark settings (pseudo-distributed local[*] mode).
"""
import os
import platform
import sys
from pathlib import Path

os.environ['PYSPARK_GATEWAY_SOCKET_TIMEOUT'] = '1800'
os.environ.setdefault('SPARK_LOCAL_HOSTNAME', 'localhost')

from pyspark.sql import SparkSession


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _configure_hadoop() -> None:
    """
    Auto-detect and configure HADOOP_HOME so PySpark can find winutils on Windows.
    """
    if platform.system() != "Windows":
        return

    if os.environ.get("HADOOP_HOME"):
        return

    bundled = PROJECT_ROOT / "winutils"
    if (bundled / "bin" / "winutils.exe").exists():
        hadoop_home = str(bundled)
        os.environ["HADOOP_HOME"] = hadoop_home
        os.environ["hadoop.home.dir"] = hadoop_home
        os.environ["PATH"] = str(bundled / "bin") + os.pathsep + os.environ.get("PATH", "")
        return

    legacy = Path(r"C:\hadoop")
    if legacy.exists():
        os.environ["HADOOP_HOME"] = str(legacy)
        os.environ["hadoop.home.dir"] = str(legacy)
        os.environ["PATH"] += os.pathsep + str(legacy / "bin")
        return

    print(
        "\n[ERROR] winutils not found. PySpark on Windows needs winutils.exe to run.\n"
        "Fix: run the setup script once, then re-run your pipeline:\n\n"
        "    python scripts/get_winutils.py\n\n"
        "This downloads winutils.exe into the project's winutils/ folder.\n"
        "Alternatively set HADOOP_HOME to an existing Hadoop installation.\n",
        file=sys.stderr,
    )
    sys.exit(1)


_configure_hadoop()

DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
INTERIM_DIR   = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR   = PROJECT_ROOT / "reports"
WEATHER_RAW_DIR = INTERIM_DIR / "weather_raw"

STATIONS_PARQUET        = INTERIM_DIR / "stations.parquet"
STATION_WEATHER_PARQUET = INTERIM_DIR / "station_weather.parquet"
ACCIDENTS_WITH_STATION_PARQUET = INTERIM_DIR / "accidents_with_station.parquet"
MERGED_PARQUET          = INTERIM_DIR / "merged.parquet"

ACCIDENTS_CSV = "Accident_Information.csv"
VEHICLES_CSV  = "Vehicle_Information.csv"

WEATHER_START = "2005-01-01"
WEATHER_END   = "2017-12-31"

def get_spark(app_name: str = "road-accidents") -> SparkSession:
    """
    Build (or return the existing) Spark session in pseudo-distributed mode.

    local[*]  — use every available core as a separate JVM worker thread.
    4 g driver memory — leaves ~4 GB for the OS + Python on an 8 GB machine.
    Apache Arrow — faster pandas UDF and Spark-to-pandas conversions.
    shuffle_partitions = 32 — ~4x cores on an i5 (avoids overhead of 200 default).
    """
    python_exec = sys.executable
    
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory",        "8g")
        .config("spark.driver.maxResultSize",  "4g")
        .config("spark.memory.fraction",       "0.8")
        .config("spark.memory.storageFraction","0.3")
        .config("spark.serializer",
                "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")

        .config("spark.sql.shuffle.partitions", "50")

        .config("spark.sql.adaptive.enabled",                     "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled",  "true")
        .config("spark.sql.adaptive.skewJoin.enabled",            "true")
        .config("spark.sql.autoBroadcastJoinThreshold", str(100 * 1024 * 1024))

        .config("spark.sql.execution.arrow.pyspark.enabled",      "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch",   "2000")

        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        .config("spark.python.worker.faulthandler.enabled",              "true")

        .config("spark.executor.heartbeatInterval", "60s")
        .config("spark.network.timeout",            "800s")
        .config("spark.sql.broadcastTimeout",       "1200")
        .config("spark.driver.extraJavaOptions",
            "-Dpy4j.gateway.server.connection_timeout=0")

        .config("spark.sql.parquet.compression.codec",               "snappy")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .config("spark.hadoop.fs.file.impl",
                "org.apache.hadoop.fs.LocalFileSystem")
        .config("spark.hadoop.fs.file.impl.disable.cache", "true")

        .config("spark.sql.ansi.enabled", "false")
        .config("spark.python.worker.reuse", "true")
        .config("spark.rpc.message.maxSize", "512")
        .getOrCreate()
    )
