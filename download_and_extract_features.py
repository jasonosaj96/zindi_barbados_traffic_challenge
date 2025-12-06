#!/usr/bin/env python3
"""
Download and Feature Extraction Pipeline for 4 Synchronized Cameras

This script:
1. Downloads 4 synchronized camera videos from GCS bucket (based on timestamp)
2. Runs object tracking with speed calculation (step 2)
3. Extracts features per camera (step 3)
4. Merges features from all 4 cameras (step 4)
5. Returns ML-ready features DataFrame

Usage from command line:
    python download_and_extract_features.py --timestamp 2025-10-20-06-00-45
    python download_and_extract_features.py --timestamp 2025-10-20-06-00-45 --dataset full

Usage from Jupyter notebook:
    from download_and_extract_features import process_video_set

    df = process_video_set(
        timestamp="2025-10-20-06-00-45",
        dataset="small",
        download_dir="videos",
        output_dir="features"
    )
"""

import argparse
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re
import time
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MultiCameraFeaturePipeline:
    """Complete pipeline for downloading and processing 4 synchronized cameras"""

    GCS_BUCKETS = {
        'small': 'gs://brb-traffic/',
        'full': 'gs://brb-traffic-full/'
    }

    CAMERA_MAPPING = {
        1: {'name': 'normanniles1', 'direction': 'north', 'abbrev': 'N'},
        2: {'name': 'normanniles2', 'direction': 'east', 'abbrev': 'E'},
        3: {'name': 'normanniles3', 'direction': 'south', 'abbrev': 'S'},
        4: {'name': 'normanniles4', 'direction': 'west', 'abbrev': 'W'}
    }

    def __init__(
        self,
        dataset: str = 'small',
        download_dir: str = 'videos',
        output_dir: str = 'features_output',
        zone_distances_json: Optional[str] = None,
        time_window_seconds: float = 300.0,
        skip_download: bool = False,
        keep_videos: bool = False
    ):
        """
        Initialize pipeline

        Args:
            dataset: 'small' or 'full' GCS bucket
            download_dir: Directory to download videos to
            output_dir: Directory for output features
            zone_distances_json: Path to zone distances JSON for speed calculation
            time_window_seconds: Time window for feature aggregation (default 300s = 5min)
            skip_download: Skip download if videos already exist
            keep_videos: Keep downloaded videos after processing
        """
        self.dataset = dataset
        self.bucket = self.GCS_BUCKETS[dataset]
        self.download_dir = Path(download_dir)
        self.output_dir = Path(output_dir)
        self.zone_distances_json = zone_distances_json
        self.time_window_seconds = time_window_seconds
        self.skip_download = skip_download
        self.keep_videos = keep_videos

        # Create directories
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default zone distances
        if self.zone_distances_json is None:
            self.zone_distances_json = "camera_configs/zone_distances_example.json"

        logger.info(f"Pipeline initialized: {dataset} dataset")
        logger.info(f"Download dir: {self.download_dir}")
        logger.info(f"Output dir: {self.output_dir}")

    def check_gsutil(self) -> bool:
        """Check if gsutil is installed"""
        try:
            result = subprocess.run(
                ['gsutil', 'version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"✓ gsutil found: {result.stdout.split()[2]}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        logger.error("✗ gsutil not found!")
        logger.error("Install: brew install --cask google-cloud-sdk (macOS)")
        logger.error("Or visit: https://cloud.google.com/sdk/docs/install")
        return False

    def find_videos_by_timestamp(self, timestamp: str) -> List[str]:
        """
        Find 4 synchronized video files in GCS bucket by timestamp

        Args:
            timestamp: Timestamp in format YYYY-MM-DD-HH-MM-SS

        Returns:
            List of GCS paths for 4 camera videos
        """
        logger.info(f"Searching for videos with timestamp: {timestamp}")

        cmd = ['gsutil', 'ls', '-r', self.bucket]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            logger.error(f"Error listing bucket: {result.stderr}")
            return []

        # Find matching videos
        found_videos = {}
        for line in result.stdout.split('\n'):
            if timestamp in line and any(ext in line.lower() for ext in ['.mp4', '.avi', '.mov']):
                # Determine camera
                for cam_id, cam_info in self.CAMERA_MAPPING.items():
                    if cam_info['name'] in line:
                        found_videos[cam_id] = line.strip()
                        break

        if len(found_videos) != 4:
            logger.warning(f"Found {len(found_videos)}/4 cameras: {list(found_videos.keys())}")

        return found_videos

    def download_video(self, gcs_path: str, local_path: Path) -> bool:
        """Download single video from GCS"""
        if local_path.exists() and self.skip_download:
            logger.info(f"  ⏭️  Skipping download (exists): {local_path.name}")
            return True

        local_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"  ⬇️  Downloading: {local_path.name}")
        cmd = ['gsutil', '-m', 'cp', gcs_path, str(local_path)]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"  ✓ Downloaded: {local_path}")
            return True
        else:
            logger.error(f"  ✗ Download failed: {result.stderr}")
            return False

    def download_video_set(self, video_paths: Dict[int, str]) -> Dict[int, Path]:
        """
        Download all 4 synchronized videos

        Args:
            video_paths: Dict mapping camera_id to GCS path

        Returns:
            Dict mapping camera_id to local path
        """
        logger.info(f"Downloading {len(video_paths)} videos...")

        downloaded = {}
        for cam_id, gcs_path in video_paths.items():
            filename = gcs_path.split('/')[-1]
            local_path = self.download_dir / filename

            if self.download_video(gcs_path, local_path):
                downloaded[cam_id] = local_path
            else:
                logger.error(f"Failed to download camera {cam_id}")

        return downloaded

    def run_tracking(self, video_path: Path) -> Optional[Path]:
        """
        Run object tracking on video (step 2)

        Args:
            video_path: Path to video file

        Returns:
            Path to tracking JSON file, or None if failed
        """
        logger.info(f"  📹 Tracking: {video_path.name}")

        # Check if tracking JSON already exists
        tracking_json = video_path.parent / f"{video_path.stem}_tracking.json"
        if tracking_json.exists():
            logger.info(f"  ⏭️  Tracking JSON exists: {tracking_json.name}")
            return tracking_json

        # Build command
        cmd = [
            'python', 'step_2_object_tracking.py',
            str(video_path),
            '--zone-distances', self.zone_distances_json
        ]

        logger.info(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"  ✓ Tracking complete: {tracking_json.name}")
            return tracking_json
        else:
            logger.error(f"  ✗ Tracking failed: {result.stderr}")
            return None

    def extract_features(self, tracking_json: Path) -> Optional[Path]:
        """
        Extract features from tracking data (step 3)

        Args:
            tracking_json: Path to tracking JSON

        Returns:
            Path to features CSV, or None if failed
        """
        logger.info(f"  📊 Extracting features: {tracking_json.name}")

        # Check if features CSV already exists
        features_csv = Path(str(tracking_json).replace('_tracking.json', '_tracking_features.csv'))
        if features_csv.exists():
            logger.info(f"  ⏭️  Features CSV exists: {features_csv.name}")
            return features_csv

        # Build command
        cmd = [
            'python', 'step_3_feature_extraction.py',
            str(tracking_json),
            '--output', str(features_csv),
            '--window', str(self.time_window_seconds),
            '--zone-distances', self.zone_distances_json
        ]

        logger.info(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"  ✓ Features extracted: {features_csv.name}")
            return features_csv
        else:
            logger.error(f"  ✗ Feature extraction failed: {result.stderr}")
            return None

    def merge_features(
        self,
        feature_csvs: Dict[int, Path],
        timestamp: str
    ) -> Optional[Path]:
        """
        Merge features from all 4 cameras (step 4)

        Args:
            feature_csvs: Dict mapping camera_id to features CSV path
            timestamp: Timestamp for output filename

        Returns:
            Path to merged features CSV, or None if failed
        """
        logger.info(f"  🔗 Merging features from {len(feature_csvs)} cameras...")

        output_csv = self.output_dir / f"merged_features_{timestamp}.csv"

        # Check if already exists
        if output_csv.exists():
            logger.info(f"  ⏭️  Merged features exist: {output_csv.name}")
            return output_csv

        # Build command
        cmd = [
            'python', 'step_4_multicamera_features.py',
            '--cam1', str(feature_csvs[1]),
            '--cam2', str(feature_csvs[2]),
            '--cam3', str(feature_csvs[3]),
            '--cam4', str(feature_csvs[4]),
            '--output', str(output_csv)
        ]

        logger.info(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"  ✓ Features merged: {output_csv}")
            return output_csv
        else:
            logger.error(f"  ✗ Feature merging failed: {result.stderr}")
            return None

    def cleanup_videos(self, video_paths: Dict[int, Path]):
        """Delete downloaded videos to save space"""
        if self.keep_videos:
            logger.info("Keeping videos (--keep-videos flag)")
            return

        logger.info("Cleaning up downloaded videos...")
        for cam_id, video_path in video_paths.items():
            try:
                video_path.unlink()
                logger.info(f"  🗑️  Deleted: {video_path.name}")
            except Exception as e:
                logger.warning(f"  ⚠️  Could not delete {video_path.name}: {e}")

    def process_timestamp(self, timestamp: str) -> Optional[pd.DataFrame]:
        """
        Complete pipeline: download → track → extract → merge

        Args:
            timestamp: Video timestamp (YYYY-MM-DD-HH-MM-SS)

        Returns:
            DataFrame with merged features, or None if failed
        """
        logger.info("="*80)
        logger.info(f"Processing video set: {timestamp}")
        logger.info("="*80)

        # Step 1: Find videos in GCS
        logger.info("Step 1: Finding videos in GCS bucket...")
        video_paths = self.find_videos_by_timestamp(timestamp)

        if len(video_paths) != 4:
            logger.error(f"Could not find all 4 cameras. Found: {list(video_paths.keys())}")
            return None

        logger.info(f"✓ Found all 4 camera videos")

        # Step 2: Download videos
        logger.info("\nStep 2: Downloading videos...")
        downloaded = self.download_video_set(video_paths)

        if len(downloaded) != 4:
            logger.error(f"Failed to download all videos. Downloaded: {list(downloaded.keys())}")
            return None

        logger.info(f"✓ Downloaded all 4 videos")

        # Step 3: Run tracking for each camera
        logger.info("\nStep 3: Running object tracking...")
        tracking_jsons = {}
        for cam_id, video_path in downloaded.items():
            tracking_json = self.run_tracking(video_path)
            if tracking_json:
                tracking_jsons[cam_id] = tracking_json
            else:
                logger.error(f"Tracking failed for camera {cam_id}")

        if len(tracking_jsons) != 4:
            logger.error(f"Tracking incomplete. Completed: {list(tracking_jsons.keys())}")
            return None

        logger.info(f"✓ Tracking complete for all 4 cameras")

        # Step 4: Extract features for each camera
        logger.info("\nStep 4: Extracting features per camera...")
        feature_csvs = {}
        for cam_id, tracking_json in tracking_jsons.items():
            features_csv = self.extract_features(tracking_json)
            if features_csv:
                feature_csvs[cam_id] = features_csv
            else:
                logger.error(f"Feature extraction failed for camera {cam_id}")

        if len(feature_csvs) != 4:
            logger.error(f"Feature extraction incomplete. Completed: {list(feature_csvs.keys())}")
            return None

        logger.info(f"✓ Features extracted for all 4 cameras")

        # Step 5: Merge features
        logger.info("\nStep 5: Merging features from all cameras...")
        merged_csv = self.merge_features(feature_csvs, timestamp)

        if not merged_csv:
            logger.error("Feature merging failed")
            return None

        logger.info(f"✓ Features merged successfully")

        # Step 6: Load merged features
        logger.info("\nStep 6: Loading merged features...")
        df = pd.read_csv(merged_csv)
        logger.info(f"✓ Loaded DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")

        # Step 7: Cleanup
        self.cleanup_videos(downloaded)

        # Summary
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Timestamp: {timestamp}")
        logger.info(f"Output: {merged_csv}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Time windows: {len(df)}")
        logger.info(f"Features: {len(df.columns)}")
        logger.info("="*80)

        return df


# Convenience function for notebook usage
def process_video_set(
    timestamp: str,
    dataset: str = 'small',
    download_dir: str = 'videos',
    output_dir: str = 'features_output',
    zone_distances_json: Optional[str] = None,
    time_window_seconds: float = 300.0,
    skip_download: bool = False,
    keep_videos: bool = False
) -> Optional[pd.DataFrame]:
    """
    Process a set of 4 synchronized videos and return merged features

    This is a convenience function for Jupyter notebook usage.

    Args:
        timestamp: Video timestamp (YYYY-MM-DD-HH-MM-SS)
        dataset: 'small' or 'full' GCS bucket
        download_dir: Directory to download videos
        output_dir: Directory for output features
        zone_distances_json: Path to zone distances JSON
        time_window_seconds: Time window for aggregation (default 300s = 5min)
        skip_download: Skip download if videos exist
        keep_videos: Keep videos after processing

    Returns:
        DataFrame with merged features, or None if failed

    Example:
        >>> from download_and_extract_features import process_video_set
        >>> df = process_video_set("2025-10-20-06-00-45")
        >>> print(df.head())
    """
    pipeline = MultiCameraFeaturePipeline(
        dataset=dataset,
        download_dir=download_dir,
        output_dir=output_dir,
        zone_distances_json=zone_distances_json,
        time_window_seconds=time_window_seconds,
        skip_download=skip_download,
        keep_videos=keep_videos
    )

    return pipeline.process_timestamp(timestamp)


def batch_process_timestamps(
    timestamps: List[str],
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Process multiple timestamp sets in batch

    Args:
        timestamps: List of timestamps to process
        **kwargs: Arguments passed to process_video_set

    Returns:
        Dict mapping timestamp to DataFrame

    Example:
        >>> timestamps = [
        ...     "2025-10-20-06-00-45",
        ...     "2025-10-20-07-00-45",
        ...     "2025-10-20-08-00-45"
        ... ]
        >>> results = batch_process_timestamps(timestamps)
        >>> combined = pd.concat(results.values(), ignore_index=True)
    """
    results = {}

    for timestamp in timestamps:
        logger.info(f"\n{'='*80}")
        logger.info(f"Batch processing {timestamp} ({len(results)+1}/{len(timestamps)})")
        logger.info(f"{'='*80}\n")

        df = process_video_set(timestamp, **kwargs)
        if df is not None:
            results[timestamp] = df
        else:
            logger.warning(f"Failed to process {timestamp}")

    logger.info(f"\nBatch processing complete: {len(results)}/{len(timestamps)} successful")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract features from 4 synchronized camera videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video set
  python download_and_extract_features.py --timestamp 2025-10-20-06-00-45

  # Use full dataset
  python download_and_extract_features.py --timestamp 2025-10-20-06-00-45 --dataset full

  # Keep videos after processing
  python download_and_extract_features.py --timestamp 2025-10-20-06-00-45 --keep-videos

  # Custom time window (10 minutes)
  python download_and_extract_features.py --timestamp 2025-10-20-06-00-45 --window 600

  # Skip download if videos exist
  python download_and_extract_features.py --timestamp 2025-10-20-06-00-45 --skip-download
        """
    )

    parser.add_argument(
        '--timestamp',
        type=str,
        required=True,
        help='Video timestamp (YYYY-MM-DD-HH-MM-SS)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['small', 'full'],
        default='small',
        help='GCS dataset: small (re-encoded) or full (>500GB)'
    )

    parser.add_argument(
        '--download-dir',
        type=str,
        default='videos',
        help='Directory to download videos (default: videos)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='features_output',
        help='Output directory for features (default: features_output)'
    )

    parser.add_argument(
        '--zone-distances',
        type=str,
        help='Path to zone distances JSON (default: camera_configs/zone_distances_example.json)'
    )

    parser.add_argument(
        '--window',
        type=float,
        default=300.0,
        help='Time window in seconds (default: 300 = 5min)'
    )

    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download if videos already exist'
    )

    parser.add_argument(
        '--keep-videos',
        action='store_true',
        help='Keep downloaded videos after processing'
    )

    args = parser.parse_args()

    # Run pipeline
    df = process_video_set(
        timestamp=args.timestamp,
        dataset=args.dataset,
        download_dir=args.download_dir,
        output_dir=args.output_dir,
        zone_distances_json=args.zone_distances,
        time_window_seconds=args.window,
        skip_download=args.skip_download,
        keep_videos=args.keep_videos
    )

    if df is not None:
        print("\n" + "="*80)
        print("SUCCESS")
        print("="*80)
        print(f"Features shape: {df.shape}")
        print(f"Output: {args.output_dir}/merged_features_{args.timestamp}.csv")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("FAILED")
        print("="*80)
        exit(1)


if __name__ == "__main__":
    main()
