"""
Shared configuration for the UK Road Accidents project.

All modules import `get_spark()` from here so every script uses
identical Spark settings (pseudo-distributed local[*] mode).
"""
import os
import platform
import sys
from pathlib import Path

# Increase Py4j gateway socket timeout to 30 minutes (prevent premature timeouts during slow writes)
os.environ['PYSPARK_GATEWAY_SOCKET_TIMEOUT'] = '1800'
# Avoid invalid Spark URLs on Windows hosts with underscores in machine name.
os.environ.setdefault('SPARK_LOCAL_HOSTNAME', 'localhost')

from pyspark.sql import SparkSession


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _configure_hadoop() -> None:
    """
    Auto-detect and configure HADOOP_HOME so PySpark can find winutils on Windows.

    Resolution order (first match wins):
      1. HADOOP_HOME already set in the environment → nothing to do.
      2. Bundled winutils inside the project: <project_root>/winutils/
         (run  python scripts/get_winutils.py  to populate it).
      3. The legacy default location C:\\hadoop (keeps existing setups working).
      4. Not found → print a clear, actionable error and exit.

    On Linux / macOS winutils is not required, so this function returns immediately.
    """
    if platform.system() != "Windows":
        return

    if os.environ.get("HADOOP_HOME"):
        # Already configured (e.g. set in the system environment or a .env file)
        return

    # Bundled copy inside the repo — works on any teammate's machine after
    # running `python scripts/get_winutils.py`
    bundled = PROJECT_ROOT / "winutils"
    if (bundled / "bin" / "winutils.exe").exists():
        hadoop_home = str(bundled)
        os.environ["HADOOP_HOME"] = hadoop_home
        os.environ["hadoop.home.dir"] = hadoop_home
        os.environ["PATH"] = str(bundled / "bin") + os.pathsep + os.environ.get("PATH", "")
        return

    # Legacy / manual install at the documented default location
    legacy = Path(r"C:\hadoop")
    if legacy.exists():
        os.environ["HADOOP_HOME"] = str(legacy)
        os.environ["hadoop.home.dir"] = str(legacy)
        os.environ["PATH"] += os.pathsep + str(legacy / "bin")
        return

    # Nothing found — give a clear, actionable error instead of a cryptic Spark crash
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
    # Set Python executable for Spark workers
    python_exec = sys.executable
    
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        # .config("spark.driver.memory", "4g")
        # .config("spark.sql.shuffle.partitions", "100")
        # .config("spark.sql.parquet.compression.codec", "snappy")
        # .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        # # Smaller Arrow batches — pandas_udf workers OOM/crash less often on
        # # Windows + Py3.10 with the default 10 000-row batch. 2 000 keeps
        # # per-batch peak memory small while still being vectorized.
        # .config("spark.sql.execution.arrow.maxRecordsPerBatch", "2000")
        # # When a Python worker crashes, surface the real traceback instead
        # # of just "Connection reset by peer". Essential for debugging UDFs.
        # .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        # .config("spark.python.worker.faulthandler.enabled", "true")
        # .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
        # .config("spark.hadoop.fs.file.impl.disable.cache", "true")
        # .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        # .config("spark.executor.heartbeatInterval", "60s")   # was 10s default
        # .config("spark.network.timeout", "800s")             # was 120s default
        # .config("spark.sql.broadcastTimeout", "1200")
        # .config("spark.sql.ansi.enabled", "false")
        # .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000") # Reduce overhea
        # .config("spark.driver.maxResultSize", "2g")
        # .getOrCreate()
        
        # ── Memory ────────────────────────────────────────────────────────
        # Give the driver enough headroom for toPandas() calls (stations,
        # the station-map join) without spilling. 8 g is safe on a 16 g
        # machine; drop to 6 g if you have <12 g free.
        .config("spark.driver.memory",        "8g")
        .config("spark.driver.maxResultSize",  "4g")
        # Executor = driver in local mode, but these still govern the
        # JVM heap split between execution and storage.
        .config("spark.memory.fraction",       "0.8")   # default 0.6
        .config("spark.memory.storageFraction","0.3")   # of the above

        # ── Serialisation ─────────────────────────────────────────────────
        # Kryo is ~10× faster than Java serialisation for NumPy-like data.
        .config("spark.serializer",
                "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")

        # ── Shuffle / partitioning ────────────────────────────────────────
        # 200 (default) creates tiny files for 2M rows; 50–100 is right here.
        # AQE will coalesce automatically if you enable it (see below).
        .config("spark.sql.shuffle.partitions", "50")

        # ── Adaptive Query Execution (Spark 3+) ───────────────────────────
        # Automatically coalesces shuffle partitions, picks broadcast joins,
        # and handles skewed joins — free wins with no code changes.
        .config("spark.sql.adaptive.enabled",                     "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled",  "true")
        .config("spark.sql.adaptive.skewJoin.enabled",            "true")
        # Raise the broadcast threshold — weather table is ~740k rows × 9
        # cols ≈ 50 MB; default threshold is 10 MB, so raise it.
        .config("spark.sql.autoBroadcastJoinThreshold", str(100 * 1024 * 1024))  # 100 MB

        # ── Arrow / pandas_udf ────────────────────────────────────────────
        .config("spark.sql.execution.arrow.pyspark.enabled",      "true")
        # 2 000 rows × 156 stations × float32 ≈ 1.9 MB per batch — safe on
        # any machine. Larger batches are faster but risk worker OOM.
        .config("spark.sql.execution.arrow.maxRecordsPerBatch",   "2000")

        # ── Fault visibility ──────────────────────────────────────────────
        # Surface real Python tracebacks instead of "connection reset".
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        .config("spark.python.worker.faulthandler.enabled",              "true")

        # ── Timeouts (critical on Windows with local Spark) ───────────────
        .config("spark.executor.heartbeatInterval", "60s")
        .config("spark.network.timeout",            "800s")
        .config("spark.sql.broadcastTimeout",       "1200")
        .config("spark.driver.extraJavaOptions",
            "-Dpy4j.gateway.server.connection_timeout=0")

        # ── I/O ───────────────────────────────────────────────────────────
        .config("spark.sql.parquet.compression.codec",               "snappy")
        # Version 2 avoids the rename step that causes issues on Windows NTFS.
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .config("spark.hadoop.fs.file.impl",
                "org.apache.hadoop.fs.LocalFileSystem")
        .config("spark.hadoop.fs.file.impl.disable.cache", "true")

        # ── Misc ──────────────────────────────────────────────────────────
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.python.worker.reuse", "true")         # Reuse workers to avoid startup overhead
        .config("spark.rpc.message.maxSize", "512")          # Increase RPC message size for large payloads
        .getOrCreate()
    )
