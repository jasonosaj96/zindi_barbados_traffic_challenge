#!/usr/bin/env python3
"""
Complete Pipeline with Training Labels

Downloads videos, extracts features, and integrates training labels from Train.csv.

Usage from command line:
    python process_with_labels.py --timestamp 2025-10-20-06-00-45

Usage from Jupyter notebook:
    from process_with_labels import process_video_set_with_labels

    df = process_video_set_with_labels("2025-10-20-06-00-45")
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import logging

# Import our existing pipelines
from download_and_extract_features import process_video_set
from integrate_training_labels import TrainingDataIntegrator


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_video_set_with_labels(
    timestamp: str,
    dataset: str = 'small',
    download_dir: str = 'videos',
    output_dir: str = 'features_output',
    train_csv: str = 'dataset/Train.csv',
    zone_distances_json: Optional[str] = None,
    time_window_seconds: float = 300.0,
    skip_download: bool = False,
    keep_videos: bool = False,
    label_aggregation: str = 'mode'
) -> Optional[pd.DataFrame]:
    """
    Complete pipeline: download → extract features → add labels

    Args:
        timestamp: Video timestamp (YYYY-MM-DD-HH-MM-SS)
        dataset: 'small' or 'full' GCS bucket
        download_dir: Directory to download videos
        output_dir: Directory for output features
        train_csv: Path to Train.csv with labels
        zone_distances_json: Path to zone distances JSON
        time_window_seconds: Time window for aggregation (default 300s = 5min)
        skip_download: Skip download if videos exist
        keep_videos: Keep videos after processing
        label_aggregation: Label aggregation method ('mode', 'max', 'last')

    Returns:
        DataFrame with features and labels, or None if failed

    Example:
        >>> from process_with_labels import process_video_set_with_labels
        >>> df = process_video_set_with_labels("2025-10-20-06-00-45")
        >>> print(df[['north_entry_congestion', 'north_exit_congestion']].head())
    """
    logger.info("="*80)
    logger.info(f"COMPLETE PIPELINE WITH LABELS: {timestamp}")
    logger.info("="*80)

    # Step 1-5: Download and extract features
    logger.info("\nPhase 1: Download videos and extract features...")
    logger.info("-"*80)

    df_features = process_video_set(
        timestamp=timestamp,
        dataset=dataset,
        download_dir=download_dir,
        output_dir=output_dir,
        zone_distances_json=zone_distances_json,
        time_window_seconds=time_window_seconds,
        skip_download=skip_download,
        keep_videos=keep_videos
    )

    if df_features is None:
        logger.error("Feature extraction failed")
        return None

    logger.info(f"✓ Features extracted: {df_features.shape}")

    # Step 6: Add training labels
    logger.info("\nPhase 2: Adding training labels from Train.csv...")
    logger.info("-"*80)

    integrator = TrainingDataIntegrator(train_csv)

    # Parse timestamp to datetime
    try:
        video_dt = datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S")
    except:
        logger.error(f"Could not parse timestamp: {timestamp}")
        return df_features

    # Add labels for each camera
    directions = ['north', 'east', 'south', 'west']

    for direction in directions:
        logger.info(f"  Adding labels for {direction}...")

        entry_labels = []
        exit_labels = []
        signaling_values = []
        label_counts = []

        for idx, row in df_features.iterrows():
            # Use video timestamp-based matching (more reliable)
            labels = integrator.get_labels_by_video_timestamp(
                direction,
                timestamp,  # Video timestamp string (YYYY-MM-DD-HH-MM-SS)
                row['start_time'],  # Window start in seconds
                row['end_time'],  # Window end in seconds
                aggregation=label_aggregation
            )

            entry_labels.append(labels['entry_label'])
            exit_labels.append(labels['exit_label'])
            signaling_values.append(labels['signaling'])
            label_counts.append(labels['entry_count_in_window'])

        df_features[f'{direction}_entry_congestion'] = entry_labels
        df_features[f'{direction}_exit_congestion'] = exit_labels
        df_features[f'{direction}_signaling'] = signaling_values
        df_features[f'{direction}_label_count'] = label_counts

        # Log coverage
        entry_coverage = pd.Series(entry_labels).notna().sum() / len(entry_labels) * 100
        exit_coverage = pd.Series(exit_labels).notna().sum() / len(exit_labels) * 100
        logger.info(f"    Entry labels: {entry_coverage:.1f}% coverage")
        logger.info(f"    Exit labels: {exit_coverage:.1f}% coverage")

    # Save labeled features
    labeled_csv = Path(output_dir) / f"labeled_features_{timestamp}.csv"
    df_features.to_csv(labeled_csv, index=False)
    logger.info(f"\n✓ Saved labeled features to: {labeled_csv}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Features shape: {df_features.shape}")
    logger.info(f"Output: {labeled_csv}")

    # Check overall label coverage
    label_cols = [f'{d}_{t}_congestion' for d in directions for t in ['entry', 'exit']]
    total_labels = sum(df_features[col].notna().sum() for col in label_cols if col in df_features.columns)
    total_possible = len(df_features) * len(label_cols)
    coverage = total_labels / total_possible * 100

    logger.info(f"Label coverage: {coverage:.1f}% ({total_labels}/{total_possible})")
    logger.info("="*80)

    return df_features


def batch_process_with_labels(
    timestamps: List[str],
    output_combined: str = 'training_dataset.csv',
    **kwargs
) -> pd.DataFrame:
    """
    Process multiple timestamps and combine into single training dataset

    Args:
        timestamps: List of timestamps to process
        output_combined: Output path for combined dataset
        **kwargs: Arguments passed to process_video_set_with_labels

    Returns:
        Combined DataFrame with all features and labels

    Example:
        >>> timestamps = [
        ...     "2025-10-20-06-00-45",
        ...     "2025-10-20-07-00-45",
        ...     "2025-10-20-08-00-45"
        ... ]
        >>> df = batch_process_with_labels(timestamps)
    """
    logger.info(f"Batch processing {len(timestamps)} timestamps...")

    all_dfs = []

    for i, timestamp in enumerate(timestamps, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {i}/{len(timestamps)}: {timestamp}")
        logger.info(f"{'='*80}\n")

        df = process_video_set_with_labels(timestamp, **kwargs)

        if df is not None:
            df['video_timestamp'] = timestamp
            all_dfs.append(df)
        else:
            logger.warning(f"Failed to process {timestamp}")

    if not all_dfs:
        logger.error("No datasets created")
        return None

    # Combine all
    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"\n{'='*80}")
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Processed: {len(all_dfs)}/{len(timestamps)} timestamps")
    logger.info(f"Combined shape: {combined.shape}")

    # Save combined dataset
    output_path = Path(output_combined)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info(f"Saved to: {output_path}")
    logger.info(f"{'='*80}\n")

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline: download + extract + add labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video set with labels
  python process_with_labels.py --timestamp 2025-10-20-06-00-45

  # Batch process multiple timestamps
  python process_with_labels.py --batch \
    --timestamps 2025-10-20-06-00-45 2025-10-20-07-00-45 2025-10-20-08-00-45 \
    --output training_dataset.csv

  # Use full dataset
  python process_with_labels.py --timestamp 2025-10-20-06-00-45 --dataset full

  # Skip download if videos exist
  python process_with_labels.py --timestamp 2025-10-20-06-00-45 --skip-download
        """
    )

    parser.add_argument(
        '--timestamp',
        type=str,
        help='Single video timestamp (YYYY-MM-DD-HH-MM-SS)'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch mode: process multiple timestamps'
    )

    parser.add_argument(
        '--timestamps',
        type=str,
        nargs='+',
        help='Multiple timestamps for batch mode'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['small', 'full'],
        default='small',
        help='GCS dataset (default: small)'
    )

    parser.add_argument(
        '--download-dir',
        type=str,
        default='videos',
        help='Download directory (default: videos)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='features_output',
        help='Output directory (default: features_output)'
    )

    parser.add_argument(
        '--train-csv',
        type=str,
        default='dataset/Train.csv',
        help='Path to Train.csv (default: dataset/Train.csv)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='training_dataset.csv',
        help='Output file for batch mode (default: training_dataset.csv)'
    )

    parser.add_argument(
        '--zone-distances',
        type=str,
        help='Path to zone distances JSON'
    )

    parser.add_argument(
        '--window',
        type=float,
        default=300.0,
        help='Time window in seconds (default: 300)'
    )

    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download if videos exist'
    )

    parser.add_argument(
        '--keep-videos',
        action='store_true',
        help='Keep videos after processing'
    )

    parser.add_argument(
        '--label-aggregation',
        type=str,
        choices=['mode', 'max', 'last'],
        default='mode',
        help='Label aggregation method (default: mode)'
    )

    args = parser.parse_args()

    # Batch mode
    if args.batch:
        if not args.timestamps:
            parser.error("--batch requires --timestamps")

        df = batch_process_with_labels(
            timestamps=args.timestamps,
            output_combined=args.output,
            dataset=args.dataset,
            download_dir=args.download_dir,
            output_dir=args.output_dir,
            train_csv=args.train_csv,
            zone_distances_json=args.zone_distances,
            time_window_seconds=args.window,
            skip_download=args.skip_download,
            keep_videos=args.keep_videos,
            label_aggregation=args.label_aggregation
        )

        if df is not None:
            print(f"\n✓ SUCCESS")
            print(f"Combined dataset: {df.shape}")
            print(f"Output: {args.output}")

    # Single timestamp mode
    else:
        if not args.timestamp:
            parser.error("Must provide --timestamp or use --batch mode")

        df = process_video_set_with_labels(
            timestamp=args.timestamp,
            dataset=args.dataset,
            download_dir=args.download_dir,
            output_dir=args.output_dir,
            train_csv=args.train_csv,
            zone_distances_json=args.zone_distances,
            time_window_seconds=args.window,
            skip_download=args.skip_download,
            keep_videos=args.keep_videos,
            label_aggregation=args.label_aggregation
        )

        if df is not None:
            print(f"\n✓ SUCCESS")
            print(f"Features shape: {df.shape}")
            output_file = Path(args.output_dir) / f"labeled_features_{args.timestamp}.csv"
            print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
