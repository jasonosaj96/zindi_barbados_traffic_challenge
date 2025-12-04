#!/usr/bin/env python3
"""
Consolidate Video Processing Results to CSV

This script reads all .counts.json files and consolidates them into a CSV file
with one row per video and summary statistics as columns.

Output includes:
- Video metadata (filename, camera, date, time, duration)
- Zone counts (vehicles per zone)
- Dwell time statistics (mean, median, min, max per zone)
- Vehicle class distribution
- Polygon statistics (throughput, unique vehicles, etc.)
- Roundabout metrics (if available)

Usage:
    python consolidate_to_csv.py --dir video_processed_files/ --output results.csv
    python consolidate_to_csv.py --dir video_processed_files/ --cameras 1,2
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict


class DataConsolidator:
    """Consolidate multiple .counts.json files into a single CSV"""

    def __init__(self):
        self.all_rows = []
        self.all_columns = set()

    def process_counts_file(self, counts_file: Path) -> Dict:
        """Extract summary statistics from a single counts.json file"""

        with open(counts_file, 'r') as f:
            data = json.load(f)

        # Start with a flat row dictionary
        row = {}

        # Add metadata
        row['video_path'] = data.get('video_path', '')
        row['filename'] = data.get('filename', '')
        row['camera'] = data.get('camera', '')
        row['date'] = data.get('date', '')
        row['time'] = data.get('time', '')
        row['datetime'] = data.get('datetime', '')
        row['hour'] = data.get('hour', '')
        row['minute'] = data.get('minute', '')
        row['total_frames'] = data.get('total_frames', 0)
        row['fps'] = data.get('fps', 0)
        row['duration_seconds'] = data.get('duration_seconds', 0)
        row['video_width'] = data.get('video_width', 0)
        row['video_height'] = data.get('video_height', 0)
        row['total_vehicles_tracked'] = data.get('total_vehicles_tracked', 0)

        # Add all flattened fields that are already in the JSON
        # (zone counts, dwell stats, vehicle classes, polygon stats, roundabout metrics)
        for key, value in data.items():
            # Skip nested structures - only take flat values
            if not isinstance(value, (dict, list)):
                if key not in row:  # Don't overwrite metadata
                    row[key] = value

        # Extract vehicle journey statistics
        vehicle_journeys = data.get('vehicle_journeys', [])
        if vehicle_journeys:
            # Count vehicles by class
            class_counts = defaultdict(int)
            for journey in vehicle_journeys:
                class_name = journey.get('class_name', 'unknown')
                class_counts[class_name] += 1

            # Add class counts
            for class_name, count in class_counts.items():
                row[f'vehicle_class_{class_name}'] = count

            # Calculate average journey duration
            journey_durations = [
                j.get('total_time_visible', 0)
                for j in vehicle_journeys
                if j.get('total_time_visible')
            ]
            if journey_durations:
                row['avg_journey_duration'] = sum(journey_durations) / len(journey_durations)
                row['min_journey_duration'] = min(journey_durations)
                row['max_journey_duration'] = max(journey_durations)

            # Calculate average zones visited per vehicle
            zones_visited_counts = [
                j.get('num_zones_visited', 0)
                for j in vehicle_journeys
            ]
            if zones_visited_counts:
                row['avg_zones_visited'] = sum(zones_visited_counts) / len(zones_visited_counts)
                row['max_zones_visited'] = max(zones_visited_counts)

        return row

    def consolidate_directory(self, data_dir: Path, cameras: Optional[List[int]] = None) -> pd.DataFrame:
        """Consolidate all .counts.json files in a directory"""

        # Find all counts.json files
        counts_files = list(data_dir.glob('**/*.counts.json'))

        if cameras:
            # Filter by camera
            camera_names = [f'normanniles{c}' for c in cameras]
            counts_files = [
                f for f in counts_files
                if any(cam in str(f) for cam in camera_names)
            ]

        if not counts_files:
            print(f"⚠️  No .counts.json files found in {data_dir}")
            return pd.DataFrame()

        print(f"Found {len(counts_files)} .counts.json files")

        # Process each file
        for i, counts_file in enumerate(sorted(counts_files), 1):
            print(f"Processing [{i}/{len(counts_files)}]: {counts_file.name}")
            try:
                row = self.process_counts_file(counts_file)
                row['source_file'] = str(counts_file)
                self.all_rows.append(row)

                # Track all columns
                self.all_columns.update(row.keys())

            except Exception as e:
                print(f"  ✗ Error: {e}")

        # Create DataFrame
        if not self.all_rows:
            print("⚠️  No data extracted")
            return pd.DataFrame()

        # Ensure all rows have all columns (fill missing with None/NaN)
        for row in self.all_rows:
            for col in self.all_columns:
                if col not in row:
                    row[col] = None

        df = pd.DataFrame(self.all_rows)

        # Sort columns for better readability
        # Order: metadata, zone counts, dwell stats, vehicle classes, polygon stats, roundabout metrics
        metadata_cols = [
            'filename', 'camera', 'date', 'time', 'datetime', 'hour', 'minute',
            'duration_seconds', 'fps', 'total_frames', 'video_width', 'video_height',
            'total_vehicles_tracked', 'source_file', 'video_path'
        ]

        # Get all columns and sort them logically
        all_cols = list(df.columns)

        # Metadata first
        ordered_cols = [col for col in metadata_cols if col in all_cols]

        # Zone counts
        zone_count_cols = sorted([col for col in all_cols if col.startswith('zone_count_')])
        ordered_cols.extend(zone_count_cols)

        # Dwell time stats
        dwell_cols = sorted([col for col in all_cols if col.startswith('dwell_')])
        ordered_cols.extend(dwell_cols)

        # Vehicle classes
        vehicle_class_cols = sorted([col for col in all_cols if col.startswith('vehicle_class_')])
        ordered_cols.extend(vehicle_class_cols)

        # Journey stats
        journey_cols = sorted([col for col in all_cols if 'journey' in col])
        ordered_cols.extend(journey_cols)

        # Zones visited
        zones_visited_cols = sorted([col for col in all_cols if 'zones_visited' in col])
        ordered_cols.extend(zones_visited_cols)

        # Polygon stats
        poly_cols = sorted([col for col in all_cols if col.startswith('poly_')])
        ordered_cols.extend(poly_cols)

        # Roundabout metrics
        roundabout_cols = sorted([col for col in all_cols if col.startswith('roundabout_')])
        ordered_cols.extend(roundabout_cols)

        # Add any remaining columns
        remaining_cols = [col for col in all_cols if col not in ordered_cols]
        ordered_cols.extend(sorted(remaining_cols))

        df = df[ordered_cols]

        return df

    def save_csv(self, df: pd.DataFrame, output_file: Path):
        """Save DataFrame to CSV"""
        df.to_csv(output_file, index=False)
        print(f"\n✓ CSV saved to: {output_file}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")

    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics"""
        print(f"\n{'=' * 70}")
        print("Data Summary")
        print(f"{'=' * 70}")

        if df.empty:
            print("No data to summarize")
            return

        print(f"\nTotal videos: {len(df)}")

        # Videos per camera
        if 'camera' in df.columns:
            print(f"\nVideos per camera:")
            camera_counts = df['camera'].value_counts().sort_index()
            for camera, count in camera_counts.items():
                print(f"  {camera}: {count}")

        # Date range
        if 'date' in df.columns:
            dates = df['date'].dropna()
            if len(dates) > 0:
                print(f"\nDate range:")
                print(f"  Earliest: {dates.min()}")
                print(f"  Latest: {dates.max()}")

        # Total vehicles tracked
        if 'total_vehicles_tracked' in df.columns:
            total_vehicles = df['total_vehicles_tracked'].sum()
            avg_vehicles = df['total_vehicles_tracked'].mean()
            print(f"\nVehicles tracked:")
            print(f"  Total: {int(total_vehicles)}")
            print(f"  Average per video: {avg_vehicles:.1f}")
            print(f"  Min per video: {int(df['total_vehicles_tracked'].min())}")
            print(f"  Max per video: {int(df['total_vehicles_tracked'].max())}")

        # Most common vehicle classes
        vehicle_class_cols = [col for col in df.columns if col.startswith('vehicle_class_')]
        if vehicle_class_cols:
            print(f"\nVehicle classes (total across all videos):")
            for col in sorted(vehicle_class_cols):
                class_name = col.replace('vehicle_class_', '')
                total = df[col].sum()
                print(f"  {class_name}: {int(total)}")

        print(f"\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate .counts.json files into a CSV with summary statistics"
    )

    parser.add_argument(
        '--dir',
        type=str,
        default='video_processed_files',
        help='Directory containing .counts.json files (default: video_processed_files)'
    )

    parser.add_argument(
        '--cameras',
        type=str,
        help='Comma-separated list of cameras to process (e.g., "1,2,3")'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='consolidated_results.csv',
        help='Output CSV file (default: consolidated_results.csv)'
    )

    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary statistics'
    )

    args = parser.parse_args()

    # Validate input directory
    data_dir = Path(args.dir)
    if not data_dir.exists():
        print(f"✗ Error: Directory not found: {data_dir}")
        return

    # Parse cameras
    cameras = None
    if args.cameras:
        try:
            cameras = [int(c.strip()) for c in args.cameras.split(',')]
        except ValueError:
            print("✗ Error: Invalid camera list. Use comma-separated numbers (e.g., '1,2,3')")
            return

    # Create consolidator
    consolidator = DataConsolidator()

    # Process directory
    print(f"Processing directory: {data_dir}")
    if cameras:
        print(f"Filtering cameras: {cameras}")
    print()

    df = consolidator.consolidate_directory(data_dir, cameras)

    if df.empty:
        print("✗ No data to save")
        return

    # Save CSV
    output_file = Path(args.output)
    consolidator.save_csv(df, output_file)

    # Print summary if requested
    if args.summary:
        consolidator.print_summary(df)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
