#!/usr/bin/env python3
"""
Parallel Processing for Test Dataset (TestInputSegments.csv)

Processes test video timestamps in parallel using multiprocessing.
Prepares features for model inference without labels.

Features:
- Parallel processing with configurable workers
- Progress tracking with tqdm
- Robust error handling per timestamp
- Resume capability (skip already processed)
- Memory-efficient batch combining

Usage:
    # Process entire test dataset with 4 workers
    python parallel_process_test_dataset.py --workers 4

    # Process subset (first 20 timestamps)
    python parallel_process_test_dataset.py --workers 4 --limit 20

    # Process specific date range
    python parallel_process_test_dataset.py --workers 4 \
        --start-date 2025-10-20 --end-date 2025-10-21

    # Resume (skip already processed)
    python parallel_process_test_dataset.py --workers 4 --resume

From Jupyter notebook:
    from parallel_process_test_dataset import parallel_process_test_all

    df = parallel_process_test_all(
        workers=4,
        limit=20,
        output_dir='test_features_output',
        output_combined='test_dataset.csv'
    )
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Optional, List, Dict, Tuple
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import traceback
from tqdm import tqdm

from download_and_extract_features import process_video_set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParallelTestDatasetProcessor:
    """Process entire test dataset in parallel"""

    def __init__(
        self,
        test_csv: str = 'TestInputSegments.csv',
        output_dir: str = 'test_features_output',
        dataset: str = 'small',
        skip_download: bool = False,
        keep_videos: bool = False,
        time_window_seconds: float = 300.0
    ):
        """
        Initialize parallel test processor

        Args:
            test_csv: Path to TestInputSegments.csv
            output_dir: Output directory for features
            dataset: 'small' or 'full' GCS bucket
            skip_download: Skip download if videos exist
            keep_videos: Keep videos after processing
            time_window_seconds: Time window for features
        """
        self.test_csv = Path(test_csv)
        self.output_dir = Path(output_dir)
        self.dataset = dataset
        self.skip_download = skip_download
        self.keep_videos = keep_videos
        self.time_window_seconds = time_window_seconds

        # Load TestInputSegments.csv to get available timestamps
        self.test_df = None
        self.load_test_csv()

    def load_test_csv(self):
        """Load TestInputSegments.csv and extract unique timestamps"""
        logger.info(f"Loading {self.test_csv}...")
        self.test_df = pd.read_csv(self.test_csv)
        logger.info(f"Loaded {len(self.test_df)} test records")

        # Extract timestamps from videos column
        self.test_df['video_timestamp'] = self.test_df['videos'].apply(
            self._extract_timestamp
        )

        # Parse datetime
        self.test_df['datetime'] = pd.to_datetime(self.test_df['video_time'])

    def _extract_timestamp(self, video_path: str) -> Optional[str]:
        """Extract timestamp from video filename"""
        import re
        match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', video_path)
        if match:
            return match.group(1)
        return None

    def get_unique_timestamps(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Get unique timestamps from TestInputSegments.csv

        Args:
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            limit: Maximum number of timestamps

        Returns:
            List of unique timestamps
        """
        timestamps = self.test_df['video_timestamp'].dropna().unique()

        # Filter by date if specified
        if start_date or end_date:
            filtered = []
            for ts in timestamps:
                try:
                    ts_date = datetime.strptime(ts, "%Y-%m-%d-%H-%M-%S").date()

                    if start_date:
                        start = datetime.strptime(start_date, "%Y-%m-%d").date()
                        if ts_date < start:
                            continue

                    if end_date:
                        end = datetime.strptime(end_date, "%Y-%m-%d").date()
                        if ts_date > end:
                            continue

                    filtered.append(ts)
                except:
                    continue

            timestamps = filtered

        # Sort chronologically
        timestamps = sorted(timestamps)

        # Apply limit
        if limit:
            timestamps = timestamps[:limit]

        return list(timestamps)

    def get_processed_timestamps(self) -> set:
        """Get set of already processed timestamps"""
        processed = set()

        if not self.output_dir.exists():
            return processed

        # Look for test_features_*.csv files
        for csv_file in self.output_dir.glob("test_features_*.csv"):
            # Extract timestamp from filename
            import re
            match = re.search(r'test_features_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.csv', csv_file.name)
            if match:
                processed.add(match.group(1))

        return processed

    def process_single_timestamp(
        self,
        timestamp: str,
        progress_dict: Optional[Dict] = None
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Process single timestamp (worker function)

        Args:
            timestamp: Video timestamp
            progress_dict: Shared dict for progress tracking

        Returns:
            Tuple of (timestamp, success, error_message)
        """
        try:
            # Extract features only (no labels for test data)
            df = process_video_set(
                timestamp=timestamp,
                dataset=self.dataset,
                download_dir='videos',
                output_dir=str(self.output_dir),
                time_window_seconds=self.time_window_seconds,
                skip_download=self.skip_download,
                keep_videos=self.keep_videos
            )

            if df is not None:
                # Save with test-specific naming
                output_file = Path(self.output_dir) / f"test_features_{timestamp}.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"✓ Saved test features: {output_file}")

                if progress_dict is not None:
                    progress_dict['completed'] += 1
                return (timestamp, True, None)
            else:
                if progress_dict is not None:
                    progress_dict['failed'] += 1
                return (timestamp, False, "Processing returned None")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Failed to process {timestamp}: {error_msg}")
            logger.debug(traceback.format_exc())

            if progress_dict is not None:
                progress_dict['failed'] += 1

            return (timestamp, False, error_msg)

    def parallel_process(
        self,
        timestamps: List[str],
        workers: int = 4,
        resume: bool = False
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Process multiple timestamps in parallel

        Args:
            timestamps: List of timestamps to process
            workers: Number of parallel workers
            resume: Skip already processed timestamps

        Returns:
            Tuple of (successful_timestamps, failed_timestamps_with_errors)
        """
        if resume:
            processed = self.get_processed_timestamps()
            original_count = len(timestamps)
            timestamps = [ts for ts in timestamps if ts not in processed]
            logger.info(f"Resume mode: {len(processed)} already processed, {len(timestamps)} remaining")
            logger.info(f"Skipped: {original_count - len(timestamps)}")

        if not timestamps:
            logger.info("No timestamps to process")
            return [], []

        logger.info(f"Processing {len(timestamps)} test timestamps with {workers} workers...")

        # Create shared progress dict
        manager = Manager()
        progress_dict = manager.dict()
        progress_dict['completed'] = 0
        progress_dict['failed'] = 0

        # Create worker function with shared dict
        worker_func = partial(
            self.process_single_timestamp,
            progress_dict=progress_dict
        )

        successful = []
        failed = []

        # Process with pool
        with Pool(processes=workers) as pool:
            results = []
            with tqdm(total=len(timestamps), desc="Processing Test Data") as pbar:
                for result in pool.imap_unordered(worker_func, timestamps):
                    results.append(result)
                    pbar.update(1)

                    timestamp, success, error = result
                    if success:
                        successful.append(timestamp)
                        pbar.set_postfix({'✓': len(successful), '✗': len(failed)})
                    else:
                        failed.append((timestamp, error))
                        pbar.set_postfix({'✓': len(successful), '✗': len(failed)})

        return successful, failed

    def combine_results(
        self,
        timestamps: List[str],
        output_csv: str = 'test_dataset.csv',
        batch_size: int = 50
    ) -> Optional[pd.DataFrame]:
        """
        Combine processed test results into single dataset

        Args:
            timestamps: Timestamps to combine
            output_csv: Output CSV path
            batch_size: Number of files to load at once (memory management)

        Returns:
            Combined DataFrame
        """
        logger.info(f"Combining {len(timestamps)} test results...")

        all_dfs = []

        # Process in batches to manage memory
        for i in range(0, len(timestamps), batch_size):
            batch_timestamps = timestamps[i:i+batch_size]
            logger.info(f"Loading batch {i//batch_size + 1}/{(len(timestamps)-1)//batch_size + 1}...")

            batch_dfs = []
            for timestamp in batch_timestamps:
                csv_path = self.output_dir / f"test_features_{timestamp}.csv"

                if not csv_path.exists():
                    logger.warning(f"File not found: {csv_path}")
                    continue

                try:
                    df = pd.read_csv(csv_path)
                    df['video_timestamp'] = timestamp
                    batch_dfs.append(df)
                except Exception as e:
                    logger.error(f"Failed to load {csv_path}: {e}")

            if batch_dfs:
                batch_combined = pd.concat(batch_dfs, ignore_index=True)
                all_dfs.append(batch_combined)
                logger.info(f"  Loaded {len(batch_dfs)} files, {batch_combined.shape[0]} rows")

        if not all_dfs:
            logger.error("No data loaded")
            return None

        # Combine all batches
        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Combined test dataset: {combined.shape}")

        # Add metadata from TestInputSegments.csv
        combined = self._add_test_metadata(combined)

        # Save
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        logger.info(f"Saved to: {output_path}")

        # Print statistics
        self._print_dataset_stats(combined)

        return combined

    def _add_test_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add metadata from TestInputSegments.csv to features"""
        logger.info("Adding test metadata...")

        # Create lookup dictionary
        metadata = self.test_df[['video_timestamp', 'responseId', 'view_label', 
                                 'ID_enter', 'ID_exit', 'time_segment_id', 
                                 'cycle_phase']].drop_duplicates()

        # Merge
        df = df.merge(metadata, on='video_timestamp', how='left')

        return df

    def _print_dataset_stats(self, df: pd.DataFrame):
        """Print dataset statistics"""
        logger.info("\n" + "="*80)
        logger.info("TEST DATASET STATISTICS")
        logger.info("="*80)

        logger.info(f"Total rows: {len(df):,}")
        logger.info(f"Total columns: {len(df.columns)}")

        if 'video_timestamp' in df.columns:
            logger.info(f"Unique timestamps: {df['video_timestamp'].nunique()}")

        if 'view_label' in df.columns:
            logger.info(f"Unique cameras: {df['view_label'].nunique()}")
            logger.info(f"Cameras: {sorted(df['view_label'].unique())}")

        if 'cycle_phase' in df.columns:
            logger.info(f"Unique test phases: {df['cycle_phase'].nunique()}")

        # Feature columns
        feature_cols = [col for col in df.columns if not col.startswith(('video_', 'responseId', 'ID_', 'time_segment', 'cycle'))]
        logger.info(f"\nFeature columns: {len(feature_cols)}")

        logger.info("="*80 + "\n")


def parallel_process_test_all(
    test_csv: str = 'TestInputSegments.csv',
    output_dir: str = 'test_features_output',
    output_combined: str = 'test_dataset.csv',
    workers: int = 4,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
    resume: bool = False,
    dataset: str = 'small',
    skip_download: bool = False,
    keep_videos: bool = False
) -> Optional[pd.DataFrame]:
    """
    Process entire test dataset in parallel (notebook-friendly function)

    Args:
        test_csv: Path to TestInputSegments.csv
        output_dir: Output directory for features
        output_combined: Output path for combined dataset
        workers: Number of parallel workers (default: 4)
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
        limit: Maximum number of timestamps to process
        resume: Skip already processed timestamps
        dataset: 'small' or 'full' GCS bucket
        skip_download: Skip download if videos exist
        keep_videos: Keep videos after processing

    Returns:
        Combined DataFrame with all features (no labels)

    Example:
        >>> from parallel_process_test_dataset import parallel_process_test_all
        >>> df = parallel_process_test_all(workers=4, limit=10)
        >>> print(df.shape)
    """
    processor = ParallelTestDatasetProcessor(
        test_csv=test_csv,
        output_dir=output_dir,
        dataset=dataset,
        skip_download=skip_download,
        keep_videos=keep_videos
    )

    # Get timestamps to process
    timestamps = processor.get_unique_timestamps(
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )

    logger.info(f"Found {len(timestamps)} unique timestamps in TestInputSegments.csv")
    if limit:
        logger.info(f"Limiting to first {limit} timestamps")

    if not timestamps:
        logger.error("No timestamps found")
        return None

    # Process in parallel
    successful, failed = processor.parallel_process(
        timestamps=timestamps,
        workers=workers,
        resume=resume
    )

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TEST PROCESSING SUMMARY")
    logger.info("="*80)
    logger.info(f"Total timestamps: {len(timestamps)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")

    if failed:
        logger.info("\nFailed timestamps:")
        for timestamp, error in failed[:10]:  # Show first 10
            logger.info(f"  {timestamp}: {error}")
        if len(failed) > 10:
            logger.info(f"  ... and {len(failed) - 10} more")

    logger.info("="*80 + "\n")

    if not successful:
        logger.error("No successful processing - cannot combine results")
        return None

    # Combine results
    combined = processor.combine_results(
        timestamps=successful,
        output_csv=output_combined
    )

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Parallel processing for test dataset (TestInputSegments.csv)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire test dataset with 4 workers
  python parallel_process_test_dataset.py --workers 4

  # Process first 20 timestamps (for testing)
  python parallel_process_test_dataset.py --workers 4 --limit 20

  # Process specific date range
  python parallel_process_test_dataset.py --workers 4 \
    --start-date 2025-10-20 --end-date 2025-10-21

  # Resume processing (skip already processed)
  python parallel_process_test_dataset.py --workers 4 --resume

  # Use full dataset bucket
  python parallel_process_test_dataset.py --workers 8 --dataset full
        """
    )

    parser.add_argument(
        '--test-csv',
        type=str,
        default='TestInputSegments.csv',
        help='Path to TestInputSegments.csv (default: TestInputSegments.csv)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='test_features_output',
        help='Output directory (default: test_features_output)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='test_dataset.csv',
        help='Output CSV path (default: test_dataset.csv)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date filter (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        help='End date filter (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of timestamps to process'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip already processed timestamps'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['small', 'full'],
        default='small',
        help='GCS dataset (default: small)'
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
        '--auto-workers',
        action='store_true',
        help='Automatically use CPU count - 1 workers'
    )

    args = parser.parse_args()

    # Auto workers
    if args.auto_workers:
        args.workers = max(1, cpu_count() - 1)
        logger.info(f"Auto workers: using {args.workers} workers")

    # Run parallel processing
    df = parallel_process_test_all(
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        output_combined=args.output,
        workers=args.workers,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
        resume=args.resume,
        dataset=args.dataset,
        skip_download=args.skip_download,
        keep_videos=args.keep_videos
    )

    if df is not None:
        print(f"\n✓ SUCCESS")
        print(f"Combined test dataset: {df.shape}")
        print(f"Output: {args.output}")
    else:
        print(f"\n✗ FAILED")
        exit(1)


if __name__ == "__main__":
    main()
