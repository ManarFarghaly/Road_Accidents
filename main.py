# from pyspark.sql import SparkSession
# from data  import raw
# # from src.data.processed import STATION_WEATHER_PARQUET

# import pandas as pd

# # Load CSV file
# df = pd.read_csv(r"E:\Data\BigData\Project\Road_Accidents\data\raw\Accident_Information.csv")

# # Show first few rows
# print(df.head())

# # Show schema (column names and data types)
# print(df.info())

# # Show summary statistics for numeric columns
# print(df.describe())

# # If you want data types only
# print(df.dtypes)

# # Load CSV file
# df = pd.read_csv(r"E:\Data\BigData\Project\Road_Accidents\data\raw\Vehicle_Information.csv", encoding="latin1")

# # Show first few rows
# print(df.head())

# # Show schema (column names and data types)
# print(df.info())

# # Show summary statistics for numeric columns
# print(df.describe())

# # If you want data types only
# print(df.dtypes)

# # weather = spark.read.parquet(str(STATION_WEATHER_PARQUET))
# # weather.printSchema()
# import pandas as pd

# # Load it back
# df_loaded = pd.read_pickle(r"E:\Data\BigData\Project\Road_Accidents\data\interim\station_weather.parquet")

# # Schema (column names and dtypes)
# print(df_loaded.dtypes)
# print(df_loaded.info())

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Read the parquet file
df = spark.read.parquet(r"E:\Data\BigData\Project\Road_Accidents\data\processed\train.parquet")

# Show first 10 columns
cols10 = df.columns[:10]
df.select(cols10).show(10, truncate=False)   # show first 10 rows of those columns

# Get summary statistics for those columns
df.select(cols10).describe().show()
