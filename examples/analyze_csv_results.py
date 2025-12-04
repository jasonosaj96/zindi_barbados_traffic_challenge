#!/usr/bin/env python3
"""
Example: Analyze Consolidated CSV Results

This script demonstrates how to analyze the consolidated CSV output
from the traffic analysis pipeline.

Usage:
    python examples/analyze_csv_results.py consolidated_results.csv
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path


def load_data(csv_file):
    """Load the consolidated CSV file"""
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} videos\n")
    return df


def basic_statistics(df):
    """Print basic statistics"""
    print("=" * 70)
    print("BASIC STATISTICS")
    print("=" * 70)

    print(f"\nTotal videos processed: {len(df)}")

    if 'camera' in df.columns:
        print(f"\nVideos per camera:")
        for camera, count in df['camera'].value_counts().sort_index().items():
            print(f"  {camera}: {count}")

    if 'date' in df.columns:
        dates = df['date'].dropna()
        if len(dates) > 0:
            print(f"\nDate range: {dates.min()} to {dates.max()}")

    if 'total_vehicles_tracked' in df.columns:
        total = df['total_vehicles_tracked'].sum()
        mean = df['total_vehicles_tracked'].mean()
        median = df['total_vehicles_tracked'].median()
        print(f"\nVehicles tracked:")
        print(f"  Total: {int(total)}")
        print(f"  Mean per video: {mean:.1f}")
        print(f"  Median per video: {median:.1f}")
        print(f"  Min: {int(df['total_vehicles_tracked'].min())}")
        print(f"  Max: {int(df['total_vehicles_tracked'].max())}")


def vehicle_class_analysis(df):
    """Analyze vehicle class distribution"""
    print("\n" + "=" * 70)
    print("VEHICLE CLASS DISTRIBUTION")
    print("=" * 70)

    vehicle_cols = [col for col in df.columns if col.startswith('vehicle_class_')]

    if not vehicle_cols:
        print("\nNo vehicle class data found")
        return

    print("\nTotal vehicles by class (across all videos):")
    class_totals = {}
    for col in vehicle_cols:
        class_name = col.replace('vehicle_class_', '')
        total = df[col].sum()
        class_totals[class_name] = int(total)

    # Sort by count
    for class_name, count in sorted(class_totals.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / sum(class_totals.values()) * 100) if sum(class_totals.values()) > 0 else 0
        print(f"  {class_name}: {count} ({percentage:.1f}%)")


def zone_analysis(df):
    """Analyze zone statistics"""
    print("\n" + "=" * 70)
    print("ZONE STATISTICS")
    print("=" * 70)

    zone_count_cols = [col for col in df.columns if col.startswith('zone_count_')]

    if not zone_count_cols:
        print("\nNo zone count data found")
        return

    print("\nTotal vehicles per zone (across all videos):")
    zone_totals = {}
    for col in zone_count_cols:
        zone_name = col.replace('zone_count_', '')
        total = df[col].sum()
        zone_totals[zone_name] = int(total)

    # Sort by count
    for zone_name, count in sorted(zone_totals.items(), key=lambda x: x[1], reverse=True):
        avg = count / len(df) if len(df) > 0 else 0
        print(f"  {zone_name}: {count} (avg {avg:.1f} per video)")


def temporal_analysis(df):
    """Analyze traffic patterns by time"""
    print("\n" + "=" * 70)
    print("TEMPORAL ANALYSIS")
    print("=" * 70)

    if 'hour' not in df.columns or 'total_vehicles_tracked' not in df.columns:
        print("\nTemporal data not available")
        return

    print("\nAverage vehicles by hour of day:")
    hourly = df.groupby('hour')['total_vehicles_tracked'].agg(['mean', 'count']).sort_index()

    for hour, row in hourly.iterrows():
        if pd.notna(hour):
            print(f"  {int(hour):02d}:00 - {row['mean']:6.1f} vehicles (from {int(row['count'])} videos)")

    # Peak hours
    if len(hourly) > 0:
        peak_hour = hourly['mean'].idxmax()
        peak_value = hourly['mean'].max()
        print(f"\nPeak hour: {int(peak_hour):02d}:00 with {peak_value:.1f} vehicles on average")


def camera_comparison(df):
    """Compare statistics across cameras"""
    print("\n" + "=" * 70)
    print("CAMERA COMPARISON")
    print("=" * 70)

    if 'camera' not in df.columns or 'total_vehicles_tracked' not in df.columns:
        print("\nCamera comparison not available")
        return

    print("\nAverage vehicles per camera:")
    camera_stats = df.groupby('camera')['total_vehicles_tracked'].agg(['mean', 'median', 'sum', 'count'])

    for camera, row in camera_stats.iterrows():
        print(f"\n  {camera}:")
        print(f"    Videos: {int(row['count'])}")
        print(f"    Total vehicles: {int(row['sum'])}")
        print(f"    Mean: {row['mean']:.1f}")
        print(f"    Median: {row['median']:.1f}")


def dwell_time_analysis(df):
    """Analyze dwell time statistics"""
    print("\n" + "=" * 70)
    print("DWELL TIME ANALYSIS")
    print("=" * 70)

    dwell_mean_cols = [col for col in df.columns if col.startswith('dwell_mean_')]

    if not dwell_mean_cols:
        print("\nNo dwell time data found")
        return

    print("\nAverage dwell times by zone (seconds):")
    for col in sorted(dwell_mean_cols):
        zone_name = col.replace('dwell_mean_', '')
        mean_dwell = df[col].mean()
        if pd.notna(mean_dwell):
            print(f"  {zone_name}: {mean_dwell:.2f}s")


def export_summary(df, output_file='analysis_summary.csv'):
    """Export a simplified summary"""
    print("\n" + "=" * 70)
    print("EXPORTING SUMMARY")
    print("=" * 70)

    # Select key columns for summary
    summary_cols = [
        'filename', 'camera', 'date', 'time', 'hour',
        'total_vehicles_tracked', 'duration_seconds'
    ]

    # Add zone counts if available
    zone_count_cols = [col for col in df.columns if col.startswith('zone_count_')]
    summary_cols.extend(zone_count_cols[:6])  # Include first 6 zone counts

    # Filter to columns that exist
    existing_cols = [col for col in summary_cols if col in df.columns]

    summary_df = df[existing_cols]
    summary_df.to_csv(output_file, index=False)

    print(f"\n✓ Summary exported to: {output_file}")
    print(f"  Rows: {len(summary_df)}")
    print(f"  Columns: {len(existing_cols)}")


def find_interesting_videos(df):
    """Identify interesting videos for further analysis"""
    print("\n" + "=" * 70)
    print("INTERESTING VIDEOS")
    print("=" * 70)

    if 'total_vehicles_tracked' not in df.columns:
        print("\nCannot identify interesting videos without vehicle data")
        return

    # Highest traffic
    print("\nTop 5 videos by vehicle count:")
    top_videos = df.nlargest(5, 'total_vehicles_tracked')
    for idx, row in top_videos.iterrows():
        filename = row.get('filename', 'unknown')
        camera = row.get('camera', 'unknown')
        vehicles = int(row.get('total_vehicles_tracked', 0))
        date = row.get('date', '')
        time = row.get('time', '')
        print(f"  {filename} ({camera}) - {vehicles} vehicles [{date} {time}]")

    # Lowest traffic
    print("\nBottom 5 videos by vehicle count:")
    bottom_videos = df.nsmallest(5, 'total_vehicles_tracked')
    for idx, row in bottom_videos.iterrows():
        filename = row.get('filename', 'unknown')
        camera = row.get('camera', 'unknown')
        vehicles = int(row.get('total_vehicles_tracked', 0))
        date = row.get('date', '')
        time = row.get('time', '')
        print(f"  {filename} ({camera}) - {vehicles} vehicles [{date} {time}]")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_csv_results.py <consolidated_results.csv>")
        print("\nExample:")
        print("  python examples/analyze_csv_results.py consolidated_results.csv")
        sys.exit(1)

    csv_file = Path(sys.argv[1])

    if not csv_file.exists():
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)

    # Load data
    df = load_data(csv_file)

    if df.empty:
        print("Error: CSV file is empty")
        sys.exit(1)

    # Run analyses
    basic_statistics(df)
    vehicle_class_analysis(df)
    zone_analysis(df)
    temporal_analysis(df)
    camera_comparison(df)
    dwell_time_analysis(df)
    find_interesting_videos(df)

    # Export summary
    export_summary(df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
