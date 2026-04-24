"""
EDA (Exploratory Data Analysis) Module

Provides statistical analysis functions for road accident data:
- Aggregations: Group-by summaries by severity, weather, time, location
- Distributions: Univariate analysis of features and target variable
- Correlations: Feature correlation matrix
- Summary reports: Comprehensive text-based EDA summaries
"""

from pyspark.sql import functions as F
from pyspark.sql import Window
import json
from typing import Dict, List, Tuple, Optional


def generate_severity_aggregations(df) -> Dict:
    """
    Aggregate accident statistics by severity level.
    
    Args:
        df: PySpark DataFrame with Accident_Severity column
    
    Returns:
        Dict with severity counts, percentages, and casualty rates
    """
    severity_dist = df.groupby('Accident_Severity').agg(
        F.count('*').alias('count'),
        F.mean('Number_of_Casualties').alias('avg_casualties'),
        F.mean('Number_of_Vehicles').alias('avg_vehicles')
    ).collect()
    
    result = {}
    total = sum([row['count'] for row in severity_dist])
    
    for row in severity_dist:
        severity = row['Accident_Severity']
        result[str(severity)] = {
            'count': int(row['count']),
            'percentage': round((row['count'] / total) * 100, 2),
            'avg_casualties': round(float(row['avg_casualties']), 2),
            'avg_vehicles': round(float(row['avg_vehicles']), 2)
        }
    
    return result


def generate_weather_aggregations(df) -> Dict:
    """
    Aggregate accident statistics by weather condition.
    
    Args:
        df: PySpark DataFrame with Weather_Conditions column
    
    Returns:
        Dict with counts and casualty statistics per weather type
    """
    weather_stats = df.groupby('Weather_Conditions').agg(
        F.count('*').alias('count'),
        F.mean('Number_of_Casualties').alias('avg_casualties'),
        F.countDistinct('Accident_Index').alias('unique_accidents')
    ).orderBy(F.col('count').desc()).collect()
    
    result = {}
    for row in weather_stats:
        weather = row['Weather_Conditions']
        result[str(weather)] = {
            'count': int(row['count']),
            'avg_casualties': round(float(row['avg_casualties']), 2),
            'total_accidents': int(row['unique_accidents'])
        }
    
    return result


def generate_temporal_aggregations(df) -> Dict:
    """
    Aggregate accidents by time patterns.
    
    Args:
        df: PySpark DataFrame with Date and Time columns
    
    Returns:
        Dict with hourly, day-of-week, and monthly distributions
    """
    # Extract hour from time
    df_temporal = df.withColumn('Hour', F.hour(F.col('Time')))
    df_temporal = df_temporal.withColumn('DayOfWeek', F.dayofweek(F.col('Date')))
    df_temporal = df_temporal.withColumn('Month', F.month(F.col('Date')))
    
    # Hourly distribution
    hourly = df_temporal.groupby('Hour').agg(
        F.count('*').alias('count')
    ).orderBy('Hour').collect()
    
    hourly_dict = {int(row['Hour']): int(row['count']) for row in hourly}
    
    # Day of week distribution
    dow_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    dow_dist = df_temporal.groupby('DayOfWeek').agg(
        F.count('*').alias('count')
    ).collect()
    
    dow_dict = {}
    for row in dow_dist:
        dow = int(row['DayOfWeek']) - 1  # Convert to 0-6
        dow_name = dow_names[dow] if 0 <= dow < len(dow_names) else 'Unknown'
        dow_dict[dow_name] = int(row['count'])
    
    return {
        'hourly': hourly_dict,
        'day_of_week': dow_dict
    }


def generate_location_aggregations(df) -> Dict:
    """
    Aggregate accidents by location density and clusters.
    
    Args:
        df: PySpark DataFrame with location_density and accident_hotspot_cluster
    
    Returns:
        Dict with location-based statistics
    """
    # Location density
    density_dist = df.groupby('location_density').agg(
        F.count('*').alias('count'),
        F.mean('Number_of_Casualties').alias('avg_casualties')
    ).collect()
    
    density_dict = {}
    for row in density_dist:
        location = row['location_density']
        density_dict[str(location)] = {
            'count': int(row['count']),
            'avg_casualties': round(float(row['avg_casualties']), 2)
        }
    
    # Cluster distribution
    cluster_dist = df.groupby('accident_hotspot_cluster').agg(
        F.count('*').alias('count'),
        F.mean('Number_of_Casualties').alias('avg_casualties')
    ).orderBy('accident_hotspot_cluster').collect()
    
    cluster_dict = {}
    for row in cluster_dist:
        cluster_id = int(row['accident_hotspot_cluster'])
        cluster_dict[f'Cluster_{cluster_id}'] = {
            'count': int(row['count']),
            'avg_casualties': round(float(row['avg_casualties']), 2)
        }
    
    return {
        'by_density': density_dict,
        'by_cluster': cluster_dict
    }


def get_null_statistics(df) -> Dict:
    """
    Calculate percentage of null values per column.
    
    Args:
        df: PySpark DataFrame
    
    Returns:
        Dict with null percentages for each column
    """
    total_rows = df.count()
    null_stats = {}
    
    for col in df.columns:
        null_count = df.filter(F.col(col).isNull()).count()
        null_pct = (null_count / total_rows) * 100 if total_rows > 0 else 0
        if null_pct > 0:
            null_stats[col] = {
                'null_count': int(null_count),
                'null_percentage': round(null_pct, 2)
            }
    
    return null_stats


def get_numeric_statistics(df) -> Dict:
    """
    Get min, max, mean, stddev for numeric columns.
    
    Args:
        df: PySpark DataFrame
    
    Returns:
        Dict with statistics per numeric column
    """
    numeric_cols = [f.name for f in df.schema.fields 
                   if f.dataType.typeName() in ['integer', 'long', 'float', 'double']]
    
    stats_dict = {}
    
    for col in numeric_cols:
        stats = df.agg(
            F.min(col).alias('min'),
            F.max(col).alias('max'),
            F.mean(col).alias('mean'),
            F.stddev(col).alias('stddev')
        ).collect()[0]
        
        stats_dict[col] = {
            'min': float(stats['min']) if stats['min'] else None,
            'max': float(stats['max']) if stats['max'] else None,
            'mean': round(float(stats['mean']), 2) if stats['mean'] else None,
            'stddev': round(float(stats['stddev']), 2) if stats['stddev'] else None
        }
    
    return stats_dict


def generate_eda_summary(df) -> Dict:
    """
    Generate comprehensive EDA summary report.
    
    Args:
        df: PySpark DataFrame (should be cleaned)
    
    Returns:
        Dict with complete EDA report
    """
    report = {
        'dataset_shape': {
            'total_rows': df.count(),
            'total_columns': len(df.columns)
        },
        'null_statistics': get_null_statistics(df),
        'numeric_statistics': get_numeric_statistics(df),
        'severity_distribution': generate_severity_aggregations(df),
        'weather_analysis': generate_weather_aggregations(df),
        'temporal_patterns': generate_temporal_aggregations(df),
        'location_analysis': generate_location_aggregations(df),
    }
    
    return report


def save_eda_report(report: Dict, output_path: str = 'reports/eda_summary.json'):
    """
    Save EDA report to JSON file.
    
    Args:
        report: Dict from generate_eda_summary()
        output_path: Path to save JSON report
    """
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"EDA report saved to {output_path}")


__all__ = [
    'generate_severity_aggregations',
    'generate_weather_aggregations',
    'generate_temporal_aggregations',
    'generate_location_aggregations',
    'get_null_statistics',
    'get_numeric_statistics',
    'generate_eda_summary',
    'save_eda_report',
]
