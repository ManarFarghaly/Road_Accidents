"""Quick check of preprocessed data — displays 100 rows"""
from src.config import get_spark

spark = get_spark("check_preprocessed")
spark.sparkContext.setLogLevel("ERROR")

train_path = "data/processed/train.parquet"

print(f"\n{'='*80}")
print("PREPROCESSED TRAINING DATA — FIRST 100 ROWS")
print(f"{'='*80}\n")

df = spark.read.parquet(train_path)
print(f"Total rows: {df.count():,}")
print(f"Columns: {df.columns}\n")

# Show 100 rows with all columns
df.limit(100).show(100, truncate=False, vertical=False)

spark.stop()
