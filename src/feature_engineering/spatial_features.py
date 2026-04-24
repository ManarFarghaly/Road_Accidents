"""
Spatial feature engineering module for GPS-based analysis.

This module provides spatial features derived from GPS coordinates:
1. accident_hotspot_cluster: KMeans cluster ID (0-9) representing geographic regions
2. location_density: Urban/suburban/rural classification based on accident density

Features are research-backed:
- Clusters identified using KMeans on 2.7M accidents across UK
- Density thresholds derived from empirical cluster distributions
"""

from pyspark.sql import functions as F
from pyspark.ml.clustering import KMeansModel
from typing import Optional


# GPS cluster centers (from KMeans k=10 model trained on full dataset)
CLUSTER_CENTERS = {
    0: (53.4950, -2.6402),  # Manchester/North-West (12.8% of accidents)
    1: (54.8242, -1.6078),  # Newcastle/North-East (4.4%)
    2: (56.4883, -3.0013),  # Glasgow/Scotland (2.7%)
    3: (51.6349, 0.2161),   # London/South-East (24.7% - largest cluster)
    4: (50.9687, -4.1277),  # Plymouth/South-West (4.4%)
    5: (53.4927, -1.1366),  # Leeds/Yorkshire (12.5%)
    6: (55.8328, -4.2245),  # Edinburgh/Central Scotland (3.8%)
    7: (52.5489, -1.7536),  # Birmingham/Midlands (11.1%)
    8: (51.4128, -0.6775),  # Surrey/Greater London (16.3%)
    9: (51.3509, -2.5248),  # Bristol/South (7.4%)
}

# Accident density per cluster (accidents per cluster size)
# Used to classify locations as urban/suburban/rural
CLUSTER_DENSITIES = {
    3: 24.7,   # London/SE: Very high (24.7%)
    8: 16.3,   # Greater London: High (16.3%)
    0: 12.8,   # Manchester: High (12.8%)
    5: 12.5,   # Leeds: High (12.5%)
    7: 11.1,   # Birmingham: High (11.1%)
    9: 7.4,    # Bristol: Medium (7.4%)
    1: 4.4,    # Newcastle: Medium (4.4%)
    4: 4.4,    # Plymouth: Medium (4.4%)
    2: 2.7,    # Glasgow: Low (2.7%)
    6: 3.8,    # Edinburgh: Low (3.8%)
}

# Urban/suburban/rural classification thresholds
# Based on cluster accident density distribution
URBAN_THRESHOLD = 11.0        # Clusters with > 11% of total accidents
SUBURBAN_THRESHOLD = 4.0      # Clusters with > 4% of total accidents


def assign_cluster_udf():
    """
    Assigns GPS coordinates to nearest KMeans cluster (0-9).
    
    Returns:
        pyspark.sql.functions.UserDefinedFunction
        Input: (Latitude: double, Longitude: double)
        Output: int (cluster ID 0-9)
    
    Logic:
        - Calculates Euclidean distance from coordinates to all cluster centers
        - Returns ID of nearest cluster
        - Returns -1 if latitude/longitude are NULL
    """
    def assign_cluster(lat, lon):
        if lat is None or lon is None:
            return -1
        
        # Ensure numeric types
        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            return -1
        
        min_distance = float('inf')
        nearest_cluster = -1
        
        for cluster_id, (center_lat, center_lon) in CLUSTER_CENTERS.items():
            # Euclidean distance (simplified, doesn't account for Earth curvature)
            distance = ((lat - center_lat) ** 2 + (lon - center_lon) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_cluster = cluster_id
        
        return int(nearest_cluster)
    
    return F.udf(assign_cluster, 'int')


def classify_location_density_udf():
    """
    Classifies locations as urban/suburban/rural based on cluster density.
    
    Returns:
        pyspark.sql.functions.UserDefinedFunction
        Input: cluster_id (int)
        Output: str ('urban', 'suburban', or 'rural')
    
    Classification:
        - Urban: Clusters with high accident density (> 11% of total)
          Examples: London, Greater London, Manchester, Leeds, Birmingham
        - Suburban: Medium density clusters (4-11% of total)
          Examples: Newcastle, Plymouth, Bristol
        - Rural: Low density clusters (< 4% of total)
          Examples: Glasgow, Edinburgh, smaller regions
    """
    def classify_density(cluster_id):
        if cluster_id is None or cluster_id < 0:
            return 'unknown'
        
        density = CLUSTER_DENSITIES.get(cluster_id, 0.0)
        
        if density > URBAN_THRESHOLD:
            return 'urban'
        elif density > SUBURBAN_THRESHOLD:
            return 'suburban'
        else:
            return 'rural'
    
    return F.udf(classify_density, 'string')


def add_spatial_features(df):
    """
    Adds spatial features to a DataFrame.
    
    Args:
        df: PySpark DataFrame with Latitude and Longitude columns
    
    Returns:
        DataFrame with two new columns:
        - accident_hotspot_cluster (int): Cluster ID 0-9 or -1 if no GPS data
        - location_density (string): 'urban', 'suburban', 'rural', or 'unknown'
    
    Example:
        df_with_spatial = add_spatial_features(df)
        # Shows which region (cluster) and urbanization level each accident is in
    """
    # Add cluster assignment
    df_with_clusters = df.withColumn(
        'accident_hotspot_cluster',
        assign_cluster_udf()(F.col('Latitude'), F.col('Longitude'))
    )
    
    # Add location density classification
    df_with_spatial = df_with_clusters.withColumn(
        'location_density',
        classify_location_density_udf()(F.col('accident_hotspot_cluster'))
    )
    
    return df_with_spatial
