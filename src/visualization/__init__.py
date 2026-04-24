"""
Visualization Module for Road Accident Analysis

Generates core analytical plots and heatmaps:
- Severity distribution (pie + bar charts)
- Accident hotspot heatmap (2D density)
- Feature correlation matrix
- Weather-severity comparisons
- Temporal patterns (hourly, daily, monthly)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_severity_distribution(severity_stats: Dict, output_path: str = 'artifacts/severity_distribution.png'):
    """
    Create pie and bar charts of accident severity distribution.
    
    Args:
        severity_stats: Dict from EDA with severity counts/percentages
        output_path: Path to save PNG file
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prepare data
    severities = list(severity_stats.keys())
    counts = [severity_stats[s]['count'] for s in severities]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    
    # Pie chart
    ax1.pie(counts, labels=severities, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Accident Severity Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart
    bars = ax2.bar(severities, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Accidents', fontsize=11)
    ax2.set_title('Accident Count by Severity', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Severity distribution saved to {output_path}")
    plt.close()


def plot_hotspot_heatmap(df, output_path: str = 'artifacts/hotspot_heatmap.png'):
    """
    Create 2D density heatmap of accident locations (Latitude vs Longitude).
    
    Args:
        df: PySpark DataFrame with Latitude, Longitude columns
        output_path: Path to save PNG file
    """
    # Convert to pandas for plotting
    data = df.select('Latitude', 'Longitude').toPandas()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create 2D histogram (heatmap)
    heatmap, xedges, yedges = np.histogram2d(
        data['Longitude'].dropna(), 
        data['Latitude'].dropna(),
        bins=40
    )
    
    # Plot as image
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(heatmap.T, origin='lower', extent=extent, cmap='YlOrRd', aspect='auto')
    
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('UK Accident Hotspot Heatmap', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accident Density', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Hotspot heatmap saved to {output_path}")
    plt.close()


def plot_correlation_matrix(correlation_matrix: pd.DataFrame, output_path: str = 'artifacts/correlation_matrix.png'):
    """
    Create heatmap of feature correlation matrix.
    
    Args:
        correlation_matrix: Pandas DataFrame with correlation coefficients
        output_path: Path to save PNG file
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Correlation matrix saved to {output_path}")
    plt.close()


def plot_weather_severity_comparison(weather_stats: Dict, severity_stats: Dict, 
                                    output_path: str = 'artifacts/weather_severity.png'):
    """
    Compare accident severity distribution across weather conditions.
    
    Args:
        weather_stats: Dict from EDA with weather counts
        severity_stats: Dict from EDA with severity counts
        output_path: Path to save PNG file
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare top weather conditions
    top_weather = sorted(weather_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:8]
    weather_types = [w[0] for w in top_weather]
    casualties = [w[1]['avg_casualties'] for w in top_weather]
    counts = [w[1]['count'] for w in top_weather]
    
    # Create bars (size proportional to count)
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(weather_types)))
    bars = ax.barh(weather_types, casualties, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Average Casualties per Accident', fontsize=11)
    ax.set_title('Average Severity by Weather Condition', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
               f' n={count:,}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Weather-severity comparison saved to {output_path}")
    plt.close()


def plot_temporal_patterns(temporal_stats: Dict, output_path: str = 'artifacts/temporal_patterns.png'):
    """
    Create temporal pattern visualizations (hourly and daily).
    
    Args:
        temporal_stats: Dict from EDA with hourly and day-of-week distributions
        output_path: Path to save PNG file
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Hourly pattern
    hours = sorted(temporal_stats['hourly'].keys())
    hourly_counts = [temporal_stats['hourly'][h] for h in hours]
    
    ax1.plot(hours, hourly_counts, marker='o', linewidth=2, markersize=6, color='#3498db')
    ax1.fill_between(hours, hourly_counts, alpha=0.3, color='#3498db')
    ax1.set_xlabel('Hour of Day', fontsize=11)
    ax1.set_ylabel('Number of Accidents', fontsize=11)
    ax1.set_title('Accidents by Hour of Day', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24, 2))
    
    # Day of week pattern
    dow_names = list(temporal_stats['day_of_week'].keys())
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_counts = [temporal_stats['day_of_week'].get(d, 0) for d in dow_order]
    colors_dow = ['#e74c3c' if d in ['Saturday', 'Sunday'] else '#2ecc71' for d in dow_order]
    
    bars = ax2.bar(dow_order, dow_counts, color=colors_dow, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Accidents', fontsize=11)
    ax2.set_title('Accidents by Day of Week', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Temporal patterns saved to {output_path}")
    plt.close()


def create_summary_page(report_dict: Dict, output_path: str = 'artifacts/analysis_summary.txt'):
    """
    Create a text-based summary report of all analyses.
    
    Args:
        report_dict: Dict from EDA with all statistics
        output_path: Path to save text file
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ROAD ACCIDENT ANALYSIS - EXPLORATORY DATA ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Dataset shape
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Records: {report_dict['dataset_shape']['total_rows']:,}\n")
        f.write(f"Total Features: {report_dict['dataset_shape']['total_columns']}\n\n")
        
        # Severity distribution
        f.write("SEVERITY DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        for severity, stats in report_dict['severity_distribution'].items():
            f.write(f"  {severity:20}: {stats['count']:7,} ({stats['percentage']:5.2f}%) | " +
                   f"Avg Casualties: {stats['avg_casualties']:.2f}\n")
        f.write("\n")
        
        # Weather analysis
        f.write("TOP WEATHER CONDITIONS\n")
        f.write("-" * 80 + "\n")
        weather_sorted = sorted(report_dict['weather_analysis'].items(), 
                               key=lambda x: x[1]['count'], reverse=True)[:5]
        for weather, stats in weather_sorted:
            f.write(f"  {weather:30}: {stats['count']:7,} | Avg Casualties: {stats['avg_casualties']:.2f}\n")
        f.write("\n")
        
        # Temporal patterns
        f.write("TEMPORAL PATTERNS\n")
        f.write("-" * 80 + "\n")
        f.write("Day of Week Distribution:\n")
        for day, count in report_dict['temporal_patterns']['day_of_week'].items():
            f.write(f"  {day:15}: {count:7,} accidents\n")
        f.write("\n")
        
        # Location analysis
        f.write("LOCATION DENSITY DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        for location, stats in report_dict['location_analysis']['by_density'].items():
            f.write(f"  {location:20}: {stats['count']:7,} ({(stats['count']/report_dict['dataset_shape']['total_rows']*100):5.1f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"✓ Summary report saved to {output_path}")


__all__ = [
    'plot_severity_distribution',
    'plot_hotspot_heatmap',
    'plot_correlation_matrix',
    'plot_weather_severity_comparison',
    'plot_temporal_patterns',
    'create_summary_page',
]
