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
