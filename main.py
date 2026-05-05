from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet(r"E:\Data\BigData\Project\Road_Accidents\data\processed\train.parquet")
cols10 = df.columns[:10]
df.select(cols10).show(10, truncate=False)  
df.select(cols10).describe().show()
