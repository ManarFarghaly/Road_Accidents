# UK Road Accidents — Big Data Analytics Pipeline

Predicts road accident severity (Fatal / Serious / Slight) from UK Department for Transport STATS19 data enriched with Meteostat weather observations. Built with **Apache Spark 4.0** in pseudo-distributed local mode.

---

## Project Structure

```
Road_Accidents/
├── data/
│   ├── raw/                        ← downloaded CSVs (auto-populated)
│   ├── interim/                    ← Spark intermediate artefacts
│   │   ├── stations.parquet        ← 156 GB weather stations
│   │   ├── station_weather.parquet ← daily weather per station
│   │   ├── accidents_with_station.parquet  ← checkpoint after geo join
│   │   └── merged.parquet          ← final merged dataset (partitioned by Year)
│   └── processed/                  ← model-ready features (after preprocessing)
├── src/
│   ├── config.py                   ← Spark session + shared paths
│   ├── data/
│   │   ├── ingest.py               ← Stage 1: full ingestion pipeline
│   │   ├── validate.py             ← Stage 2: data quality report
│   │   └── acquire.py              ← Kaggle download helper
│   └── preprocessing/
│       ├── __init__.py             ← exports clean() + build_preprocessing_stages()
│       ├── clean.py                ← Stage 3a: cleaning + imputation
│       ├── encode.py               ← Stage 3b: categorical encoding
│       ├── scale.py                ← Stage 3c: numeric scaling
│       └── assemble.py             ← Stage 3d: final feature vector
├── tests/
│   └── test_preprocessing.py       ← smoke tests for preprocessing stages
├── main.py                         ← end-to-end pipeline runner (see below)
├── requirements.txt
└── README.md
```

---

## Prerequisites

### 1. Java 17
Download and install [Java 17 JDK](https://www.oracle.com/java/technologies/downloads/#java17). Set `JAVA_HOME`:
```powershell
$env:JAVA_HOME = "C:\Program Files\Java\jdk-17"
```

### 2. Apache Spark 4.0.2
Download **Spark 4.0.2 Pre-built for Hadoop 3.4** from [spark.apache.org/downloads](https://spark.apache.org/downloads.html). Extract to `C:\spark`.

```powershell
$env:SPARK_HOME = "C:\spark"
$env:PATH += ";C:\spark\bin"
```

### 3. Hadoop WinUtils (Windows only)
Place `winutils.exe` and `hadoop.dll` in `C:\hadoop\bin`. Set:
```powershell
$env:HADOOP_HOME = "C:\hadoop"
```

### 4. Python
Use **Python 3.11 or 3.12**. Python 3.14 has known compatibility issues with PySpark 4.0.

### 5. Allow PowerShell scripts
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 6. Install dependencies
```powershell
pip install -r requirements.txt
```

---

## Pipeline Overview

```
[a] Meteostat API          → stations.parquet          (156 GB stations)
[b] Meteostat Bulk HTTP    → station_weather.parquet   (1006 station-year CSVs via Spark)
[c] Kaggle CSV             → accidents + vehicles      (Spark, 12 partitions each)
[d] Haversine geo-join     → accidents_with_station    (pandas on driver, joined via Spark)
[e] Weather join           → acc + weather             (distributed Spark join)
[f] Vehicles join + write  → merged.parquet            (2.7M rows × 64 cols, by Year)
         ↓
[3a] clean()               → drop noise cols, impute nulls, fix outliers
[3b] encode()              → StringIndexer + OneHotEncoder
[3c] scale()               → StandardScaler on numeric columns
[3d] assemble()            → final `features` vector + `label` column
         ↓
[4]  Model training        → Logistic Regression / Random Forest / GBT
```

All stages are **idempotent** — re-running skips already-built artefacts automatically.

---

## Running the Pipeline

For full pipeline scroll down 

### Stage 1 — Data Ingestion
Downloads weather data, reads CSVs, joins everything, writes partitioned Parquet.
```powershell
python -m src.data.ingest
```
Expected output: `[done] merged rows: 2,715,940, columns: 64`

This takes ~30 minutes on first run (weather download). Re-runs take ~5 minutes.

### Stage 2 — Data Validation
Prints a quality report: null rates, value distributions, schema summary.
```powershell
python -m src.data.validate
```

### Stage 3 — Preprocessing Tests
Runs smoke tests on a 1% sample verifying every preprocessing stage works correctly.
```powershell
python -m tests.test_preprocessing
```
Expected: all `[PASS]` — verifies clean, encode, scale, and assemble stages individually and end-to-end.

### Full Pipeline (all stages)
```powershell
python -m src.preprocessing.run
```
Runs ingestion → validation → preprocessing → model training in sequence.

---

## Preprocessing Details (`src/preprocessing/`)

The preprocessing module is a pure Spark ML Pipeline. Call it like this:

```python
from src.config import get_spark, MERGED_PARQUET
from src.preprocessing import clean, build_preprocessing_stages
from pyspark.ml import Pipeline

spark = get_spark("my-app")

# Load merged data
df = spark.read.parquet(str(MERGED_PARQUET))

# Step 1: eager cleaning (not a Pipeline stage — runs immediately)
cleaned = clean(df)

# Step 2: split BEFORE fitting to avoid data leakage
train, test = cleaned.randomSplit([0.8, 0.2], seed=42)

# Optional: rebalance training set (Fatal is 60x rarer than Slight)
# from src.preprocessing.clean import rebalance_undersample
# train = rebalance_undersample(train)

# Step 3: build and fit the preprocessing pipeline on training data only
stages = build_preprocessing_stages()
pipeline = Pipeline(stages=stages)
model = pipeline.fit(train)

# Step 4: transform both splits
train_features = model.transform(train)   # has 'features' and 'label' columns
test_features  = model.transform(test)    # use for evaluation only
```

### What `clean()` does

| Step | What | Why |
|------|------|-----|
| a | Drop 6 high-missing columns | >85% missing — no signal |
| a' | Drop `Number_of_Casualties` | Target leakage (severity defined FROM casualty counts) |
| b | Null sentinel strings | "Data missing or out of range", "Unknown", "" → NULL |
| c | Null numeric `-1` codes | DfT "unknown" code for numerics |
| d | Snap `Speed_limit` to legal values | {20,30,40,50,60,70} mph only |
| e | Enforce validity bounds | Null out impossible values (Lat/Lon out of UK, etc.) |
| e' | Drop rows missing target or location | Cannot train without Severity, Lat, Lon |
| g | Median-impute numeric nulls | MAR — robust to skew |
| h | Mode-by-group impute `model` | Fill unknown model with most common model per make |
| i | "Unknown" fill for categoricals | Preserves rows, gives nulls their own StringIndexer level |

### What `build_preprocessing_stages()` does

Returns an ordered list of unfit Spark ML stages:

1. **StringIndexer + OneHotEncoder** for 12 low-cardinality categoricals (< 15 levels)
2. **StringIndexer only** for 6 high-cardinality categoricals (make, model, LSOA, etc.)
3. **StringIndexer** on `Accident_Severity` → `label` column (0=Fatal, 1=Serious, 2=Slight)
4. **VectorAssembler + StandardScaler** on 15 numeric columns → `numeric_scaled`
5. **Final VectorAssembler** combining all encoded + scaled columns → `features`

Output columns used by the model: `features` (dense vector) and `label` (double).

---

## Class Imbalance

The target is heavily skewed: Slight 85% / Serious 14% / Fatal 1%. Fatal is 60× rarer than Slight. An optional undersampler is provided — use it on training data only:

```python
from src.preprocessing.clean import rebalance_undersample
train_balanced = rebalance_undersample(train, ratio=3.0)
# Now roughly Slight:Serious:Fatal = 3:3:1
model = pipeline.fit(train_balanced)
# Evaluate on the original unresampled test split
metrics = evaluator.evaluate(model.transform(test_features))
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Constructor SparkSession does not exist` | PySpark version ≠ Spark version. Run `pip install pyspark==4.0.0` |
| `Python worker exited unexpectedly` | Out of memory. Increase `spark.driver.memory` in `config.py` |
| `No weather files downloaded` | Meteostat bulk URL changed. Check `https://data.meteostat.net/daily/{year}/{station}.csv.gz` |
| `UNABLE_TO_INFER_SCHEMA` | Corrupted partial parquet from a crash. Delete `data/interim/accidents_with_station.parquet` and re-run |
| `WinError 10054 Connection reset` | Python worker OOM on Windows. Reduce `spark.sql.execution.arrow.maxRecordsPerBatch` |

### Safe cleanup (keeps downloaded files, removes broken artefacts)
```powershell
Remove-Item -Recurse -Force data\interim\accidents_with_station.parquet
Remove-Item -Recurse -Force data\interim\station_weather.parquet
Remove-Item -Recurse -Force data\interim\merged.parquet
# Keep: data\interim\stations.parquet, data\interim\weather_raw\
# Keep: data\raw\*.csv
```

---

## Dataset

- **Accidents + Vehicles**: [UK Road Safety — Accidents and Vehicles](https://www.kaggle.com/datasets/tsiaras/uk-road-safety-accidents-and-vehicles) (Kaggle, auto-downloaded)
- **Weather**: [Meteostat Bulk Data](https://dev.meteostat.net/bulk/) — daily observations per station, 2005–2017
- **Size**: ~2.7M accident records × 64 features after joining