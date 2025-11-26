#!/usr/bin/env python3
"""
Analyze vehicle counts and their relationship with congestion ratings.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the enriched CSV with vehicle counts."""
    df = pd.read_csv(csv_path)

    # Convert timestamps
    df['datetimestamp_start'] = pd.to_datetime(df['datetimestamp_start'])
    df['datetimestamp_end'] = pd.to_datetime(df['datetimestamp_end'])
    df['date'] = pd.to_datetime(df['date'])

    # Extract hour
    df['hour'] = df['datetimestamp_start'].dt.hour

    return df


def analyze_congestion_correlation(df: pd.DataFrame):
    """Analyze correlation between vehicle counts and congestion ratings."""
    print("\n" + "="*60)
    print("Congestion Analysis")
    print("="*60)

    # Group by congestion rating
    print("\nEntering Traffic:")
    enter_stats = df.groupby('congestion_enter_rating').agg({
        'count_incoming': ['mean', 'std', 'min', 'max'],
        'count_total': ['mean', 'std', 'min', 'max']
    }).round(2)
    print(enter_stats)

    print("\nExiting Traffic:")
    exit_stats = df.groupby('congestion_exit_rating').agg({
        'count_exiting': ['mean', 'std', 'min', 'max'],
        'count_total': ['mean', 'std', 'min', 'max']
    }).round(2)
    print(exit_stats)


def analyze_by_camera(df: pd.DataFrame):
    """Analyze counts by camera/view."""
    print("\n" + "="*60)
    print("Analysis by Camera")
    print("="*60)

    camera_stats = df.groupby('view_label').agg({
        'count_incoming': 'mean',
        'count_exiting': 'mean',
        'count_curve': 'mean',
        'count_inside_roundabout': 'mean',
        'count_total': 'mean'
    }).round(2)

    print(camera_stats)


def analyze_temporal_patterns(df: pd.DataFrame):
    """Analyze temporal patterns in traffic."""
    print("\n" + "="*60)
    print("Temporal Patterns")
    print("="*60)

    # By hour
    print("\nAverage vehicle counts by hour:")
    hourly = df.groupby('hour')['count_total'].mean().round(2)
    print(hourly)

    # By date
    print("\nDaily traffic volume:")
    daily = df.groupby('date')['count_total'].sum()
    print(daily)


def create_visualizations(df: pd.DataFrame, output_dir: str = "analysis_plots"):
    """Create visualization plots."""
    Path(output_dir).mkdir(exist_ok=True)

    sns.set_style("whitegrid")

    # 1. Congestion vs Counts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(data=df, x='congestion_enter_rating', y='count_total', ax=axes[0])
    axes[0].set_title('Vehicle Counts by Enter Congestion Rating')
    axes[0].set_xlabel('Congestion Rating (Enter)')
    axes[0].set_ylabel('Total Vehicle Count')
    axes[0].tick_params(axis='x', rotation=45)

    sns.boxplot(data=df, x='congestion_exit_rating', y='count_total', ax=axes[1])
    axes[1].set_title('Vehicle Counts by Exit Congestion Rating')
    axes[1].set_xlabel('Congestion Rating (Exit)')
    axes[1].set_ylabel('Total Vehicle Count')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/congestion_vs_counts.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/congestion_vs_counts.png")

    # 2. Zone breakdown by camera
    fig, ax = plt.subplots(figsize=(12, 6))

    zone_cols = ['count_incoming', 'count_exiting', 'count_curve', 'count_inside_roundabout']
    zone_data = df.groupby('view_label')[zone_cols].mean()

    zone_data.plot(kind='bar', ax=ax)
    ax.set_title('Average Vehicle Counts by Zone and Camera')
    ax.set_xlabel('Camera')
    ax.set_ylabel('Average Count')
    ax.legend(title='Zone', labels=['Incoming', 'Exiting', 'Curve', 'Inside Roundabout'])
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/zones_by_camera.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/zones_by_camera.png")

    # 3. Temporal patterns
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Hourly
    hourly = df.groupby('hour')['count_total'].mean()
    axes[0].plot(hourly.index, hourly.values, marker='o', linewidth=2)
    axes[0].set_title('Average Vehicle Counts by Hour of Day')
    axes[0].set_xlabel('Hour')
    axes[0].set_ylabel('Average Total Count')
    axes[0].grid(True, alpha=0.3)

    # Daily
    daily = df.groupby('date')['count_total'].sum()
    axes[1].plot(daily.index, daily.values, marker='o', linewidth=2)
    axes[1].set_title('Total Daily Vehicle Counts')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Total Count')
    axes[1].grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/temporal_patterns.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/temporal_patterns.png")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze vehicle counts from enriched dataframe"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data_challenge/Train_with_counts.csv",
        help="Path to enriched CSV with vehicle counts"
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate visualization plots"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_plots",
        help="Directory to save plots"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from: {args.csv}")
    df = load_data(args.csv)
    print(f"Loaded {len(df)} rows")

    # Run analyses
    analyze_congestion_correlation(df)
    analyze_by_camera(df)
    analyze_temporal_patterns(df)

    # Create visualizations
    if args.plots:
        print("\nGenerating visualizations...")
        create_visualizations(df, args.output_dir)

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
