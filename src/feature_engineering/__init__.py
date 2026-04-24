"""
Feature Engineering Module

Implements custom domain-specific features for road accident severity prediction:
    - danger_index: Continuous risk score (0-1) combining 5 environmental factors
    - vehicle_vulnerable: Boolean flag for high-casualty-risk vehicles

These features are engineered BEFORE train/test split to avoid data leakage.

Usage:
    from src.feature_engineering import DangerIndexTransformer, VehicleVulnerabilityTransformer
    
    stages = [
        DangerIndexTransformer(),
        VehicleVulnerabilityTransformer(),
        # ... other preprocessing stages
    ]
    pipeline = Pipeline(stages=stages)
"""

from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import functions as F
from pyspark.sql.types import StructField, DoubleType, BooleanType

from .domain_features import danger_index_spark_udf, vulnerability_spark_udf
from .spatial_features import add_spatial_features

__all__ = [
    "DangerIndexTransformer",
    "VehicleVulnerabilityTransformer",
    "HotspotClusterTransformer",
    "LocationDensityTransformer",
    "danger_index_spark_udf",
    "vulnerability_spark_udf",
    "add_spatial_features",
]


class DangerIndexTransformer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    Spark ML Transformer: Adds 'danger_index' column to DataFrame.

    Combines 5 environmental/road factors into a single 0-1 safety risk score.

    Components:
      1. Light_Conditions (0.25 weight): Darkness = highest risk
      2. Weather_Conditions (0.25 weight): Fog/snow = highest risk
      3. Road_Surface_Conditions (0.20 weight): Icy/flooded = highest risk
      4. Speed_limit (0.20 weight): Normalized 20-70 mph to 0-1
      5. Road_Type (0.10 weight): Minor roads = highest risk

    Output Column:
      - danger_index (DoubleType): Float in [0.0, 1.0]
        0.0 = safest (daylight + clear + dry + low speed + motorway)
        1.0 = most dangerous (darkness + fog + icy + high speed + minor road)
        0.5 = neutral (missing data)

    Example:
        >>> transformer = DangerIndexTransformer()
        >>> df = transformer.transform(df)
        >>> df.select('Light_Conditions', 'Weather_Conditions', 'danger_index').show()
    """

    def _transform(self, dataset):
        """
        Transform: Add danger_index column using UDF.

        Args:
            dataset: Input DataFrame with required columns:
              - Light_Conditions
              - Weather_Conditions
              - Road_Surface_Conditions
              - Speed_limit
              - Road_Type

        Returns:
            DataFrame with added 'danger_index' column (DoubleType).
        """
        return dataset.withColumn(
            "danger_index",
            danger_index_spark_udf(
                F.col("Light_Conditions"),
                F.col("Weather_Conditions"),
                F.col("Road_Surface_Conditions"),
                F.col("Speed_limit"),
                F.col("Road_Type"),
            ),
        )


class VehicleVulnerabilityTransformer(
    Transformer, DefaultParamsReadable, DefaultParamsWritable
):
    """
    Spark ML Transformer: Adds 'vehicle_vulnerable' column to DataFrame.

    Flags vehicles with minimal cabin protection or weak structure.

    Vulnerable vehicles:
      - All motorcycles (no cabin protection)
      - Pedal cycles, ridden horses
      - Small cars: engine < 1200cc (weak frame structure)

    Output Column:
      - vehicle_vulnerable (BooleanType): True if vulnerable, False if safe

    Example:
        >>> transformer = VehicleVulnerabilityTransformer()
        >>> df = transformer.transform(df)
        >>> df.select('Vehicle_Type', 'Engine_Capacity_CC', 'vehicle_vulnerable').show()
    """

    def _transform(self, dataset):
        """
        Transform: Add vehicle_vulnerable column using UDF.

        Args:
            dataset: Input DataFrame with required columns:
              - Vehicle_Type
              - Engine_Capacity_CC

        Returns:
            DataFrame with added 'vehicle_vulnerable' column (BooleanType).
        """
        return dataset.withColumn(
            "vehicle_vulnerable",
            vulnerability_spark_udf(
                F.col("Vehicle_Type"),
                F.col("Engine_Capacity_CC"),
            ),
        )


class HotspotClusterTransformer(
    Transformer, DefaultParamsReadable, DefaultParamsWritable
):
    """
    Spark ML Transformer: Adds 'accident_hotspot_cluster' column to DataFrame.

    Assigns each accident to nearest KMeans cluster (0-9) representing UK regions:
      - Cluster 0: Manchester/North-West (12.8%)
      - Cluster 1: Newcastle/North-East (4.4%)
      - Cluster 2: Glasgow/Scotland (2.7%)
      - Cluster 3: London/South-East (24.7% - largest)
      - Cluster 4: Plymouth/South-West (4.4%)
      - Cluster 5: Leeds/Yorkshire (12.5%)
      - Cluster 6: Edinburgh/Scotland (3.8%)
      - Cluster 7: Birmingham/Midlands (11.1%)
      - Cluster 8: Greater London (16.3%)
      - Cluster 9: Bristol/South (7.4%)

    Output Column:
      - accident_hotspot_cluster (int): Cluster ID 0-9 or -1 if missing GPS

    Example:
        >>> transformer = HotspotClusterTransformer()
        >>> df = transformer.transform(df)
        >>> df.select('Latitude', 'Longitude', 'accident_hotspot_cluster').show()
    """

    def _transform(self, dataset):
        """
        Transform: Add hotspot cluster assignment using Euclidean distance.

        Args:
            dataset: Input DataFrame with required columns:
              - Latitude (double)
              - Longitude (double)

        Returns:
            DataFrame with added 'accident_hotspot_cluster' column (int).
        """
        from .spatial_features import assign_cluster_udf
        
        return dataset.withColumn(
            "accident_hotspot_cluster",
            assign_cluster_udf()(F.col("Latitude"), F.col("Longitude")),
        )


class LocationDensityTransformer(
    Transformer, DefaultParamsReadable, DefaultParamsWritable
):
    """
    Spark ML Transformer: Adds 'location_density' column to DataFrame.

    Classifies accident locations as urban/suburban/rural based on cluster density.

    Classification:
      - Urban: High accident concentration (> 11% of total)
        Examples: London, Manchester, Leeds, Birmingham, Greater London
      - Suburban: Medium concentration (4-11% of total)
        Examples: Newcastle, Plymouth, Bristol
      - Rural: Low concentration (< 4% of total)
        Examples: Glasgow, Edinburgh, remote regions

    Output Column:
      - location_density (string): 'urban', 'suburban', 'rural', or 'unknown'

    Example:
        >>> transformer = LocationDensityTransformer()
        >>> df = transformer.transform(df)
        >>> df.select('accident_hotspot_cluster', 'location_density').show()
    """

    def _transform(self, dataset):
        """
        Transform: Add location density classification.

        Args:
            dataset: Input DataFrame with column:
              - accident_hotspot_cluster (int)

        Returns:
            DataFrame with added 'location_density' column (string).
        """
        from .spatial_features import classify_location_density_udf
        
        return dataset.withColumn(
            "location_density",
            classify_location_density_udf()(F.col("accident_hotspot_cluster")),
        )
