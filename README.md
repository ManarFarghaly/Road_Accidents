# Road_Accidents

- src/data/ingest

Pipeline (run `python -m src.data.ingest`):

    a) Build data/interim/stations.parquet          (Meteostat GB stations)

    b) Build data/interim/station_weather.parquet   (daily weather per station)

    c) Read the raw CSVs in Spark (parallel splits)

    d) Attach nearest-station via broadcast + vectorized pandas_udf

    e) Join weather on (station_id, Date) — plain distributed Spark join
    
    f) Join vehicles on Accident_Index, write partitioned Parquet

Stages a and b are idempotent , so re-runs only redo the expensive main merge. 


For installing dependencies 

bash

```
    $ pip install -r requirements.txt
```

for ingesting 

```
   $  python -m src.data.ingest
``` 
Please make sure you installed spark version 4.0.2 with Pre-built for Hadoop 3.4 or later and the bin folder of hadoop has the .dll and .exe

You should have pthon 3.10 not upper as this causes conflicts with spark
For your system to accept runing scripts 

bash
```
 $ Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
