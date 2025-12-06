#!/usr/bin/env python3
"""
Merge Generated Features with Original Train.csv

Preserves all original Train.csv columns and adds computed features.

The relationship:
- Train.csv has per-minute labeled video segments
- Generated features are per 5-minute time windows
- This script creates a dataset where each Train.csv record gets the corresponding
  5-minute window features added to it

Usage:
    python merge_with_train.py \
        --train dataset/Train.csv \
        --features training_dataset.csv \
        --output final_training_dataset.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Optional
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_timestamp_from_video(video_path: str) -> Optional[str]:
    """Extract timestamp from video filename"""
    match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', video_path)
    if match:
        return match.group(1)
    return None


def merge_features_with_train(
    train_csv: str,
    features_csv: str,
    output_csv: str,
    camera_mapping: Optional[dict] = None
) -> pd.DataFrame:
    """
    Merge generated features with original Train.csv

    Preserves all Train.csv columns and adds computed features.

    Args:
        train_csv: Path to original Train.csv
        features_csv: Path to generated features (from parallel processing)
        output_csv: Output path for merged dataset
        camera_mapping: Camera name to direction mapping

    Returns:
        Merged DataFrame
    """
    if camera_mapping is None:
        camera_mapping = {
            'Norman Niles #1': 'north',
            'Norman Niles #2': 'east',
            'Norman Niles #3': 'south',
            'Norman Niles #4': 'west',
        }

    logger.info(f"Loading Train.csv: {train_csv}")
    train_df = pd.read_csv(train_csv)
    logger.info(f"  Train.csv: {train_df.shape}")

    logger.info(f"Loading features: {features_csv}")
    features_df = pd.read_csv(features_csv)
    logger.info(f"  Features: {features_df.shape}")

    # Parse timestamps
    logger.info("Parsing timestamps...")
    train_df['video_timestamp'] = train_df['videos'].apply(extract_timestamp_from_video)
    train_df['datetime_start'] = pd.to_datetime(train_df['datetimestamp_start'])
    train_df['datetime_end'] = pd.to_datetime(train_df['datetimestamp_end'])

    # Map camera names to directions
    train_df['camera_direction'] = train_df['view_label'].map(camera_mapping)

    # Parse feature timestamps
    if 'video_timestamp' not in features_df.columns:
        logger.error("Features CSV missing 'video_timestamp' column")
        return None

    # For each Train.csv record, find the matching feature window
    logger.info("Matching Train records to feature windows...")

    merged_records = []

    for idx, train_row in train_df.iterrows():
        if idx % 1000 == 0:
            logger.info(f"  Processing record {idx}/{len(train_df)}...")

        video_ts = train_row['video_timestamp']
        camera_dir = train_row['camera_direction']
        record_start = train_row['datetime_start']

        if pd.isna(video_ts) or pd.isna(camera_dir):
            # Can't match - add record with no features
            merged_records.append(train_row.to_dict())
            continue

        # Find matching feature window
        # Filter features by timestamp
        matching_features = features_df[features_df['video_timestamp'] == video_ts]

        if len(matching_features) == 0:
            # No features for this timestamp
            merged_records.append(train_row.to_dict())
            continue

        # Find the specific window this record falls into
        # Each feature row has start_time and end_time (seconds from video start)
        try:
            video_dt = datetime.strptime(video_ts, "%Y-%m-%d-%H-%M-%S")
        except:
            merged_records.append(train_row.to_dict())
            continue

        # Calculate offset in seconds from video start
        offset_seconds = (record_start - video_dt).total_seconds()

        # Find feature window that contains this offset
        matching_window = matching_features[
            (matching_features['start_time'] <= offset_seconds) &
            (matching_features['end_time'] > offset_seconds)
        ]

        if len(matching_window) == 0:
            # Record falls outside feature windows
            merged_records.append(train_row.to_dict())
            continue

        if len(matching_window) > 1:
            # Multiple matches - take first
            matching_window = matching_window.iloc[0:1]

        # Merge: keep all Train columns + add feature columns
        merged_row = train_row.to_dict()

        # Add features (exclude columns that might conflict)
        feature_row = matching_window.iloc[0]
        exclude_cols = ['video_timestamp']  # Already in Train

        for col in matching_window.columns:
            if col not in exclude_cols:
                # Prefix feature columns to avoid conflicts
                merged_row[f'feature_{col}'] = feature_row[col]

        merged_records.append(merged_row)

    logger.info("Creating merged DataFrame...")
    merged_df = pd.DataFrame(merged_records)

    logger.info(f"Merged dataset: {merged_df.shape}")

    # Check how many records got features
    feature_cols = [c for c in merged_df.columns if c.startswith('feature_')]
    if feature_cols:
        has_features = merged_df[feature_cols[0]].notna().sum()
        logger.info(f"  Records with features: {has_features}/{len(merged_df)} ({has_features/len(merged_df)*100:.1f}%)")

    # Save
    logger.info(f"Saving to: {output_csv}")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    # Print column summary
    logger.info("\nColumn summary:")
    train_cols = [c for c in merged_df.columns if not c.startswith('feature_')]
    feature_cols = [c for c in merged_df.columns if c.startswith('feature_')]
    logger.info(f"  Original Train columns: {len(train_cols)}")
    logger.info(f"  Feature columns: {len(feature_cols)}")
    logger.info(f"  Total columns: {len(merged_df.columns)}")

    return merged_df


def main():
    parser = argparse.ArgumentParser(
        description="Merge generated features with original Train.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python merge_with_train.py \
    --train dataset/Train.csv \
    --features training_dataset.csv \
    --output final_training_dataset.csv

This preserves ALL Train.csv columns and adds computed features.
        """
    )

    parser.add_argument(
        '--train',
        type=str,
        default='dataset/Train.csv',
        help='Path to original Train.csv'
    )

    parser.add_argument(
        '--features',
        type=str,
        required=True,
        help='Path to generated features CSV'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='final_training_dataset.csv',
        help='Output CSV path'
    )

    args = parser.parse_args()

    merged_df = merge_features_with_train(
        train_csv=args.train,
        features_csv=args.features,
        output_csv=args.output
    )

    if merged_df is not None:
        print(f"\n✓ SUCCESS")
        print(f"Merged dataset: {merged_df.shape}")
        print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
