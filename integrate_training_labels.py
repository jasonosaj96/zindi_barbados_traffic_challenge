#!/usr/bin/env python3
"""
Integrate Training Labels with Extracted Features

Merges congestion labels from Train.csv with extracted multi-camera features.

The Train.csv has per-minute video records with congestion labels.
Our extracted features are aggregated in 5-minute time windows.

This script:
1. Loads Train.csv with labels
2. Loads merged multi-camera features
3. Aligns labels with feature time windows
4. Creates training-ready dataset

Usage:
    python integrate_training_labels.py \
        --features features_output/merged_features_*.csv \
        --train dataset/Train.csv \
        --output training_data.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingDataIntegrator:
    """Integrate training labels with extracted features"""

    CAMERA_MAPPING = {
        'Norman Niles #1': {'camera_id': 1, 'direction': 'north'},
        'Norman Niles #2': {'camera_id': 2, 'direction': 'east'},
        'Norman Niles #3': {'camera_id': 3, 'direction': 'south'},
        'Norman Niles #4': {'camera_id': 4, 'direction': 'west'},
    }

    def __init__(self, train_csv_path: str):
        """
        Initialize integrator

        Args:
            train_csv_path: Path to Train.csv
        """
        self.train_csv_path = Path(train_csv_path)
        self.train_df = None
        self.load_training_data()

    def load_training_data(self):
        """Load and parse training data"""
        logger.info(f"Loading training data from: {self.train_csv_path}")

        self.train_df = pd.read_csv(self.train_csv_path)
        logger.info(f"Loaded {len(self.train_df)} training records")

        # Parse timestamps
        self.train_df['datetime_start'] = pd.to_datetime(self.train_df['datetimestamp_start'])
        self.train_df['datetime_end'] = pd.to_datetime(self.train_df['datetimestamp_end'])
        self.train_df['date'] = pd.to_datetime(self.train_df['date'])

        # Extract video timestamp from video filename
        self.train_df['video_timestamp'] = self.train_df['videos'].apply(
            self._extract_video_timestamp
        )

        # Map camera names to directions
        self.train_df['camera_direction'] = self.train_df['view_label'].map(
            lambda x: self.CAMERA_MAPPING.get(x, {}).get('direction')
        )
        self.train_df['camera_id'] = self.train_df['view_label'].map(
            lambda x: self.CAMERA_MAPPING.get(x, {}).get('camera_id')
        )

        # Check for unique congestion classes
        enter_classes = self.train_df['congestion_enter_rating'].unique()
        exit_classes = self.train_df['congestion_exit_rating'].unique()

        logger.info(f"Congestion classes (entry): {sorted(enter_classes)}")
        logger.info(f"Congestion classes (exit): {sorted(exit_classes)}")
        logger.info(f"Date range: {self.train_df['date'].min()} to {self.train_df['date'].max()}")
        logger.info(f"Cameras: {sorted(self.train_df['view_label'].unique())}")

    def _extract_video_timestamp(self, video_path: str) -> Optional[str]:
        """Extract timestamp from video filename"""
        import re
        # Pattern: YYYY-MM-DD-HH-MM-SS
        match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', video_path)
        if match:
            return match.group(1)
        return None

    def get_labels_by_video_timestamp(
        self,
        camera_direction: str,
        video_timestamp: str,
        window_start_seconds: float,
        window_end_seconds: float,
        aggregation: str = 'mode'
    ) -> Dict[str, str]:
        """
        Get congestion labels by matching video timestamp and window time

        More reliable than datetime matching - uses the actual video filename.

        Args:
            camera_direction: 'north', 'east', 'south', 'west'
            video_timestamp: Video timestamp (YYYY-MM-DD-HH-MM-SS)
            window_start_seconds: Window start offset in seconds from video start
            window_end_seconds: Window end offset in seconds from video start
            aggregation: How to aggregate multiple labels ('mode', 'max', 'last')

        Returns:
            Dict with entry and exit congestion labels
        """
        # Parse video timestamp
        try:
            video_dt = datetime.strptime(video_timestamp, "%Y-%m-%d-%H-%M-%S")
        except:
            logger.warning(f"Could not parse video timestamp: {video_timestamp}")
            return {
                'entry_label': None,
                'exit_label': None,
                'entry_count_in_window': 0,
                'signaling': None,
                'video_filenames': []
            }

        # Calculate absolute window times
        window_start_dt = video_dt + timedelta(seconds=window_start_seconds)
        window_end_dt = video_dt + timedelta(seconds=window_end_seconds)

        # Filter by camera direction using time range overlap
        # A record overlaps if its time range [datetime_start, datetime_end) intersects with [window_start_dt, window_end_dt)
        mask = (
            (self.train_df['camera_direction'] == camera_direction) &
            (self.train_df['datetime_start'] < window_end_dt) &
            (self.train_df['datetime_end'] > window_start_dt)
        )

        window_records = self.train_df[mask]

        if len(window_records) == 0:
            return {
                'entry_label': None,
                'exit_label': None,
                'entry_count_in_window': 0,
                'signaling': None,
                'video_filenames': [],
                'train_records': pd.DataFrame()  # Empty DataFrame
            }

        # Aggregate labels
        if aggregation == 'mode':
            entry_label = window_records['congestion_enter_rating'].mode()
            entry_label = entry_label.iloc[0] if len(entry_label) > 0 else None

            exit_label = window_records['congestion_exit_rating'].mode()
            exit_label = exit_label.iloc[0] if len(exit_label) > 0 else None

        elif aggregation == 'max':
            severity_order = {
                'free flowing': 0,
                'light delay': 1,
                'moderate delay': 2,
                'heavy delay': 3
            }

            entry_severities = window_records['congestion_enter_rating'].map(severity_order)
            if len(entry_severities) > 0:
                max_idx = entry_severities.idxmax()
                entry_label = window_records.loc[max_idx, 'congestion_enter_rating']
            else:
                entry_label = None

            exit_severities = window_records['congestion_exit_rating'].map(severity_order)
            if len(exit_severities) > 0:
                max_idx = exit_severities.idxmax()
                exit_label = window_records.loc[max_idx, 'congestion_exit_rating']
            else:
                exit_label = None

        elif aggregation == 'last':
            # Sort by datetime_start and take the last one
            window_records_sorted = window_records.sort_values('datetime_start')
            entry_label = window_records_sorted.iloc[-1]['congestion_enter_rating']
            exit_label = window_records_sorted.iloc[-1]['congestion_exit_rating']

        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        return {
            'entry_label': entry_label,
            'exit_label': exit_label,
            'entry_count_in_window': len(window_records),
            'signaling': window_records['signaling'].mode().iloc[0] if len(window_records) > 0 else None,
            'video_filenames': window_records['videos'].tolist(),
            'train_records': window_records.copy()  # Return actual Train.csv records
        }

    def get_labels_for_timewindow(
        self,
        camera_direction: str,
        window_start: datetime,
        window_end: datetime,
        aggregation: str = 'mode'
    ) -> Dict[str, str]:
        """
        Get congestion labels for a time window

        Args:
            camera_direction: 'north', 'east', 'south', 'west'
            window_start: Window start datetime
            window_end: Window end datetime
            aggregation: How to aggregate multiple labels ('mode', 'max', 'last')

        Returns:
            Dict with entry and exit congestion labels
        """
        # Filter training records using time range overlap
        # A record overlaps if its time range [datetime_start, datetime_end) intersects with [window_start, window_end)
        mask = (
            (self.train_df['camera_direction'] == camera_direction) &
            (self.train_df['datetime_start'] < window_end) &
            (self.train_df['datetime_end'] > window_start)
        )

        window_records = self.train_df[mask]

        if len(window_records) == 0:
            return {
                'entry_label': None,
                'exit_label': None,
                'entry_count_in_window': 0,
                'signaling': None,
                'video_filenames': [],
                'train_records': pd.DataFrame()  # Empty DataFrame
            }

        # Aggregate labels
        if aggregation == 'mode':
            # Most frequent label
            entry_label = window_records['congestion_enter_rating'].mode()
            entry_label = entry_label.iloc[0] if len(entry_label) > 0 else None

            exit_label = window_records['congestion_exit_rating'].mode()
            exit_label = exit_label.iloc[0] if len(exit_label) > 0 else None

        elif aggregation == 'max':
            # Worst congestion level
            severity_order = {
                'free flowing': 0,
                'light delay': 1,
                'moderate delay': 2,
                'heavy delay': 3
            }

            entry_severities = window_records['congestion_enter_rating'].map(severity_order)
            if len(entry_severities) > 0:
                max_idx = entry_severities.idxmax()
                entry_label = window_records.loc[max_idx, 'congestion_enter_rating']
            else:
                entry_label = None

            exit_severities = window_records['congestion_exit_rating'].map(severity_order)
            if len(exit_severities) > 0:
                max_idx = exit_severities.idxmax()
                exit_label = window_records.loc[max_idx, 'congestion_exit_rating']
            else:
                exit_label = None

        elif aggregation == 'last':
            # Last label in window (sorted by datetime_start)
            window_records_sorted = window_records.sort_values('datetime_start')
            entry_label = window_records_sorted.iloc[-1]['congestion_enter_rating']
            exit_label = window_records_sorted.iloc[-1]['congestion_exit_rating']

        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        return {
            'entry_label': entry_label,
            'exit_label': exit_label,
            'entry_count_in_window': len(window_records),
            'signaling': window_records['signaling'].mode().iloc[0] if len(window_records) > 0 else None,
            'video_filenames': window_records['videos'].tolist(),
            'train_records': window_records.copy()  # Return actual Train.csv records
        }

    def integrate_features_with_labels(
        self,
        features_csv: str,
        aggregation: str = 'mode',
        output_csv: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Integrate extracted features with training labels

        Args:
            features_csv: Path to merged features CSV
            aggregation: Label aggregation method ('mode', 'max', 'last')
            output_csv: Optional path to save output

        Returns:
            DataFrame with features and labels
        """
        logger.info(f"Loading features from: {features_csv}")
        features_df = pd.read_csv(features_csv)

        logger.info(f"Features shape: {features_df.shape}")

        # Extract video timestamp from features (if available)
        # Or parse from start_time if absolute timestamps available
        if 'hour_of_day' in features_df.columns and features_df['hour_of_day'].notna().any():
            # We have absolute time information
            logger.info("Using absolute timestamps from features")
            has_absolute_time = True
        else:
            logger.warning("No absolute timestamps in features - using relative time only")
            has_absolute_time = False

        # Add labels for each camera-direction
        directions = ['north', 'east', 'south', 'west']

        for direction in directions:
            logger.info(f"Adding labels for {direction}...")

            entry_labels = []
            exit_labels = []
            label_counts = []
            signaling_values = []

            for idx, row in features_df.iterrows():
                if has_absolute_time and 'hour_of_day' in row and pd.notna(row['hour_of_day']):
                    # Reconstruct datetime from features
                    # This requires knowing the base date (from filename or metadata)
                    # For now, we'll use a placeholder approach

                    # Check if we have day/time info
                    hour = int(row['hour_of_day']) if pd.notna(row['hour_of_day']) else 0
                    start_minutes = row['start_time'] / 60.0  # Convert seconds to minutes

                    # Try to find matching records in training data
                    # Match by hour and approximate time
                    window_duration = row.get('window_duration', 300)  # seconds

                    # Simple matching: find training records with same hour and similar time
                    time_mask = (
                        (self.train_df['camera_direction'] == direction) &
                        (self.train_df['datetime_start'].dt.hour == hour)
                    )

                    matching_records = self.train_df[time_mask]

                    if len(matching_records) > 0:
                        # Take mode of matched records
                        entry_label = matching_records['congestion_enter_rating'].mode()
                        entry_label = entry_label.iloc[0] if len(entry_label) > 0 else None

                        exit_label = matching_records['congestion_exit_rating'].mode()
                        exit_label = exit_label.iloc[0] if len(exit_label) > 0 else None

                        signaling = matching_records['signaling'].mode()
                        signaling = signaling.iloc[0] if len(signaling) > 0 else None

                        entry_labels.append(entry_label)
                        exit_labels.append(exit_label)
                        label_counts.append(len(matching_records))
                        signaling_values.append(signaling)
                    else:
                        entry_labels.append(None)
                        exit_labels.append(None)
                        label_counts.append(0)
                        signaling_values.append(None)
                else:
                    # No absolute time - cannot match labels
                    entry_labels.append(None)
                    exit_labels.append(None)
                    label_counts.append(0)
                    signaling_values.append(None)

            # Add to features dataframe
            features_df[f'{direction}_entry_congestion'] = entry_labels
            features_df[f'{direction}_exit_congestion'] = exit_labels
            features_df[f'{direction}_label_count'] = label_counts
            features_df[f'{direction}_signaling'] = signaling_values

        # Count how many labels were matched
        label_cols = [f'{d}_entry_congestion' for d in directions]
        labels_matched = features_df[label_cols].notna().sum().sum()
        total_possible = len(features_df) * len(directions) * 2  # entry + exit

        logger.info(f"Matched {labels_matched}/{total_possible} labels ({labels_matched/total_possible*100:.1f}%)")

        # Save if requested
        if output_csv:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            features_df.to_csv(output_path, index=False)
            logger.info(f"Saved integrated data to: {output_path}")

        return features_df

    def create_training_dataset_from_videos(
        self,
        video_timestamps: List[str],
        features_dir: str = 'features_output',
        output_csv: str = 'training_dataset.csv',
        aggregation: str = 'mode'
    ) -> pd.DataFrame:
        """
        Create training dataset from multiple video timestamps

        Args:
            video_timestamps: List of timestamps that have been processed
            features_dir: Directory containing merged feature CSVs
            output_csv: Output path for combined dataset
            aggregation: Label aggregation method

        Returns:
            Combined DataFrame with all features and labels
        """
        features_dir = Path(features_dir)
        all_datasets = []

        for timestamp in video_timestamps:
            features_csv = features_dir / f"merged_features_{timestamp}.csv"

            if not features_csv.exists():
                logger.warning(f"Features not found for {timestamp}: {features_csv}")
                continue

            logger.info(f"\nProcessing {timestamp}...")

            # Load features
            df = pd.read_csv(features_csv)

            # Add timestamp column
            df['video_timestamp'] = timestamp

            # Extract date/time from timestamp
            try:
                dt = datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S")
                df['video_date'] = dt.date()
                df['video_hour'] = dt.hour
                df['video_minute'] = dt.minute
            except:
                logger.warning(f"Could not parse timestamp: {timestamp}")

            # Add labels by matching with Train.csv
            directions = ['north', 'east', 'south', 'west']

            for direction in directions:
                # Get labels for each time window
                entry_labels = []
                exit_labels = []
                signaling_values = []

                for idx, row in df.iterrows():
                    # Calculate absolute window time
                    window_start_dt = dt + timedelta(seconds=row['start_time'])
                    window_end_dt = dt + timedelta(seconds=row['end_time'])

                    labels = self.get_labels_for_timewindow(
                        direction,
                        window_start_dt,
                        window_end_dt,
                        aggregation
                    )

                    entry_labels.append(labels['entry_label'])
                    exit_labels.append(labels['exit_label'])
                    signaling_values.append(labels['signaling'])

                df[f'{direction}_entry_congestion'] = entry_labels
                df[f'{direction}_exit_congestion'] = exit_labels
                df[f'{direction}_signaling'] = signaling_values

            all_datasets.append(df)

        if not all_datasets:
            logger.error("No datasets created")
            return None

        # Combine all
        combined = pd.concat(all_datasets, ignore_index=True)
        logger.info(f"\nCombined dataset: {combined.shape}")

        # Check label coverage
        label_cols = [f'{d}_{t}_congestion' for d in directions for t in ['entry', 'exit']]
        for col in label_cols:
            if col in combined.columns:
                non_null = combined[col].notna().sum()
                pct = (non_null / len(combined)) * 100
                logger.info(f"  {col}: {non_null}/{len(combined)} ({pct:.1f}%) labeled")

        # Save
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        logger.info(f"\nSaved training dataset to: {output_path}")

        return combined


def main():
    parser = argparse.ArgumentParser(
        description="Integrate training labels with extracted features"
    )

    parser.add_argument(
        '--train',
        type=str,
        default='dataset/Train.csv',
        help='Path to Train.csv'
    )

    parser.add_argument(
        '--features',
        type=str,
        help='Path to merged features CSV (single file)'
    )

    parser.add_argument(
        '--features-dir',
        type=str,
        default='features_output',
        help='Directory containing feature CSVs (batch mode)'
    )

    parser.add_argument(
        '--timestamps',
        type=str,
        nargs='+',
        help='Video timestamps to process (batch mode)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='training_dataset.csv',
        help='Output CSV path'
    )

    parser.add_argument(
        '--aggregation',
        type=str,
        choices=['mode', 'max', 'last'],
        default='mode',
        help='Label aggregation method (default: mode)'
    )

    args = parser.parse_args()

    # Initialize integrator
    integrator = TrainingDataIntegrator(args.train)

    # Single file mode
    if args.features:
        logger.info("Single file mode")
        df = integrator.integrate_features_with_labels(
            args.features,
            aggregation=args.aggregation,
            output_csv=args.output
        )

        print(f"\nOutput shape: {df.shape}")
        print(f"Saved to: {args.output}")

    # Batch mode
    elif args.timestamps:
        logger.info(f"Batch mode: {len(args.timestamps)} timestamps")
        df = integrator.create_training_dataset_from_videos(
            video_timestamps=args.timestamps,
            features_dir=args.features_dir,
            output_csv=args.output,
            aggregation=args.aggregation
        )

        if df is not None:
            print(f"\nOutput shape: {df.shape}")
            print(f"Saved to: {args.output}")

    else:
        parser.error("Must provide either --features or --timestamps")


if __name__ == "__main__":
    main()
