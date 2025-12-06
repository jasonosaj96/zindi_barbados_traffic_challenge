#!/usr/bin/env python3
"""
Feature Analysis Utility

Analyze merged multi-camera features:
- Summary statistics
- Feature correlations
- Missing value analysis
- Feature importance (if targets provided)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def analyze_features(csv_path: str, show_top_n: int = 20):
    """Analyze merged features CSV"""

    print("="*80)
    print("MULTI-CAMERA FEATURE ANALYSIS")
    print("="*80)
    print()

    # Load data
    df = pd.read_csv(csv_path)
    print(f"File: {csv_path}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print()

    # Identify feature categories
    temporal_cols = [col for col in df.columns if col in [
        'window_idx', 'start_time', 'end_time', 'window_duration',
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_rush_hour', 'time_since_midnight'
    ]]

    camera_cols = {
        'north': [col for col in df.columns if col.startswith('north_')],
        'east': [col for col in df.columns if col.startswith('east_')],
        'south': [col for col in df.columns if col.startswith('south_')],
        'west': [col for col in df.columns if col.startswith('west_')],
    }

    roundabout_cols = [col for col in df.columns if col.startswith(('total_', 'avg_', 'system_'))]
    cross_camera_cols = [col for col in df.columns if 'imbalance' in col or 'balance' in col]
    directional_cols = [col for col in df.columns if col.startswith(('NS_', 'EW_'))]
    target_cols = [col for col in df.columns if col.endswith('_congestion')]

    # Summary by category
    print("FEATURE CATEGORIES:")
    print("-" * 80)
    print(f"  Temporal features:     {len(temporal_cols)}")
    print(f"  North camera features: {len(camera_cols['north'])}")
    print(f"  East camera features:  {len(camera_cols['east'])}")
    print(f"  South camera features: {len(camera_cols['south'])}")
    print(f"  West camera features:  {len(camera_cols['west'])}")
    print(f"  Roundabout-wide:       {len(roundabout_cols)}")
    print(f"  Cross-camera:          {len(cross_camera_cols)}")
    print(f"  Directional:           {len(directional_cols)}")
    print(f"  Target columns:        {len(target_cols)}")
    print()

    # Missing values analysis
    print("MISSING VALUES:")
    print("-" * 80)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percent': missing_pct
    }).sort_values('Missing', ascending=False)

    if missing_df['Missing'].sum() > 0:
        print(f"Columns with missing values: {(missing_df['Missing'] > 0).sum()}")
        print()
        print(missing_df[missing_df['Missing'] > 0].head(show_top_n))
    else:
        print("No missing values found!")
    print()

    # Summary statistics for key features
    print("SUMMARY STATISTICS (Key Features):")
    print("-" * 80)

    key_features = [
        'total_entry_count', 'total_exit_count',
        'total_entry_flow_rate', 'total_exit_flow_rate',
        'total_circulating_occupancy', 'avg_circulating_occupancy',
        'avg_speed_km_h', 'entry_flow_imbalance',
        'north_entry_count', 'east_entry_count',
        'south_entry_count', 'west_entry_count',
    ]

    available_keys = [col for col in key_features if col in df.columns]
    if available_keys:
        print(df[available_keys].describe().round(2))
    print()

    # Temporal distribution
    if 'hour_of_day' in df.columns and df['hour_of_day'].notna().any():
        print("TEMPORAL DISTRIBUTION:")
        print("-" * 80)
        print("Hour distribution:")
        print(df['hour_of_day'].value_counts().sort_index())
        print()

        if 'is_rush_hour' in df.columns:
            rush_hour_pct = (df['is_rush_hour'].sum() / len(df)) * 100
            print(f"Rush hour windows: {df['is_rush_hour'].sum()} ({rush_hour_pct:.1f}%)")
        print()

    # Flow balance analysis
    if 'total_entry_count' in df.columns and 'total_exit_count' in df.columns:
        print("FLOW BALANCE ANALYSIS:")
        print("-" * 80)
        print(f"Total entries:  {df['total_entry_count'].sum():.0f}")
        print(f"Total exits:    {df['total_exit_count'].sum():.0f}")
        print(f"Net flow:       {(df['total_entry_count'].sum() - df['total_exit_count'].sum()):.0f}")
        print(f"Avg entry rate: {df['total_entry_flow_rate'].mean():.2f} vehicles/min")
        print(f"Avg exit rate:  {df['total_exit_flow_rate'].mean():.2f} vehicles/min")
        print()

    # Camera-level comparison
    print("CAMERA-LEVEL COMPARISON:")
    print("-" * 80)

    comparison_metrics = ['entry_count', 'exit_count', 'entry_flow_rate', 'circulating_occupancy_avg']
    for metric in comparison_metrics:
        cols = [f"{direction}_{metric}" for direction in ['north', 'east', 'south', 'west']]
        if all(col in df.columns for col in cols):
            values = [df[col].mean() for col in cols]
            print(f"\n{metric}:")
            for direction, val in zip(['North', 'East', 'South', 'West'], values):
                print(f"  {direction:6s}: {val:8.2f}")

    print()

    # Directional asymmetry
    if 'NS_total_entry' in df.columns and 'EW_total_entry' in df.columns:
        print("DIRECTIONAL ASYMMETRY:")
        print("-" * 80)
        print(f"North-South avg entry: {df['NS_total_entry'].mean():.2f}")
        print(f"East-West avg entry:   {df['EW_total_entry'].mean():.2f}")
        if 'NS_EW_imbalance' in df.columns:
            print(f"NS-EW imbalance:       {df['NS_EW_imbalance'].mean():.3f}")
        print()

    # Target variable analysis (if present)
    if target_cols and df[target_cols].notna().any().any():
        print("TARGET VARIABLE DISTRIBUTION:")
        print("-" * 80)
        for col in target_cols:
            if df[col].notna().any():
                print(f"\n{col}:")
                print(df[col].value_counts().sort_index())
        print()

    # Feature type breakdown
    print("FEATURE TYPE BREAKDOWN:")
    print("-" * 80)

    lag_features = [col for col in df.columns if '_lag_' in col]
    rolling_features = [col for col in df.columns if '_rolling_' in col]
    speed_features = [col for col in df.columns if 'speed' in col.lower()]
    flow_features = [col for col in df.columns if 'flow' in col.lower()]
    occupancy_features = [col for col in df.columns if 'occupancy' in col.lower()]

    print(f"  Lag features:       {len(lag_features)}")
    print(f"  Rolling features:   {len(rolling_features)}")
    print(f"  Speed features:     {len(speed_features)}")
    print(f"  Flow features:      {len(flow_features)}")
    print(f"  Occupancy features: {len(occupancy_features)}")
    print()

    # Correlations (top correlated features with targets if available)
    if target_cols and df[target_cols].notna().any().any():
        print("TOP CORRELATED FEATURES WITH TARGETS:")
        print("-" * 80)

        # Get numeric columns only (exclude targets)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in target_cols]

        for target in target_cols:
            if df[target].notna().any():
                correlations = df[feature_cols].corrwith(df[target]).abs().sort_values(ascending=False)
                print(f"\n{target} (top 10):")
                print(correlations.head(10).round(3))
        print()

    print("="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze multi-camera features")
    parser.add_argument("csv_path", type=str, help="Path to merged features CSV")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top items to show")

    args = parser.parse_args()

    analyze_features(args.csv_path, show_top_n=args.top_n)
