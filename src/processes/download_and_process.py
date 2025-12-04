#!/usr/bin/env python3
"""
Automated data download and processing pipeline from Google Cloud Storage

This script:
1. Downloads traffic videos from GCS buckets
2. Processes them with YOLOv8n for vehicle detection
3. Generates traffic analysis data (counts, movements, congestion)
4. Saves results in a structured format

Usage:
    python download_and_process.py --dataset small --cameras all
    python download_and_process.py --dataset full --cameras 1,2 --start-date 2025-10-01
"""

import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
import sys
import time
from typing import List, Optional
import re


class GCSDataPipeline:
    """Pipeline for downloading and processing traffic data from GCS"""

    GCS_BUCKETS = {
        'small': 'gs://brb-traffic/',
        'full': 'gs://brb-traffic-full/'
    }

    CAMERA_NAMES = {
        1: 'normanniles1',
        2: 'normanniles2',
        3: 'normanniles3',
        4: 'normanniles4'
    }

    def __init__(self, dataset='small', output_dir='video_processed_files', download_only=False, save_video=False):
        self.dataset = dataset
        self.bucket = self.GCS_BUCKETS[dataset]
        self.output_dir = Path(output_dir)
        self.download_only = download_only
        self.save_video = save_video
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def check_gsutil(self):
        """Check if gsutil is installed"""
        try:
            result = subprocess.run(['gsutil', 'version'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ gsutil found: {result.stdout.split()[2]}")
                return True
        except FileNotFoundError:
            pass

        print("✗ gsutil not found!")
        print("\nInstall Google Cloud SDK:")
        print("  macOS: brew install --cask google-cloud-sdk")
        print("  Linux: curl https://sdk.cloud.google.com | bash")
        print("  Or visit: https://cloud.google.com/sdk/docs/install")
        return False

    def list_available_files(self, camera: Optional[int] = None,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> List[str]:
        """List available video files in the GCS bucket"""
        print(f"\n📋 Listing files from {self.bucket}")

        cmd = ['gsutil', 'ls', '-r', self.bucket]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"✗ Error listing files: {result.stderr}")
            return []

        # Filter video files
        files = []
        for line in result.stdout.split('\n'):
            if any(ext in line.lower() for ext in ['.mp4', '.avi', '.mov']):
                files.append(line.strip())

        # Apply filters
        if camera:
            camera_name = self.CAMERA_NAMES.get(camera)
            if camera_name:
                files = [f for f in files if camera_name in f]

        if start_date:
            files = [f for f in files if self._filter_by_date(f, start_date, end_date)]

        print(f"✓ Found {len(files)} video files")
        return files

    def _filter_by_date(self, filepath: str, start_date: str,
                       end_date: Optional[str] = None) -> bool:
        """Filter files by date range"""
        # Extract date from filename (format: YYYY-MM-DD)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filepath)
        if not date_match:
            return True

        file_date = date_match.group(1)
        if file_date < start_date:
            return False

        if end_date and file_date > end_date:
            return False

        return True

    def download_file(self, gcs_path: str, local_path: Path) -> bool:
        """Download a single file from GCS"""
        local_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = ['gsutil', '-m', 'cp', gcs_path, str(local_path)]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  ✓ Downloaded to: {local_path}")
            return True
        else:
            print(f"  ✗ Download failed: {result.stderr}")
            return False

    def download_batch(self, gcs_files: List[str],
                      max_files: Optional[int] = None) -> List[Path]:
        """Download multiple files from GCS"""
        downloaded = []

        files_to_download = gcs_files[:max_files] if max_files else gcs_files

        print(f"\n⬇️  Downloading {len(files_to_download)} files...")

        for gcs_path in files_to_download:
            # Determine local path
            filename = gcs_path.split('/')[-1]

            # Extract camera name from path
            camera_dir = None
            for cam_num, cam_name in self.CAMERA_NAMES.items():
                if cam_name in gcs_path:
                    camera_dir = cam_name
                    break

            if camera_dir:
                local_path = self.output_dir / camera_dir / filename
            else:
                local_path = self.output_dir / filename

            # Skip if already exists
            if local_path.exists():
                print(f"  ⏭️  Skipping (already exists): {filename}")
                downloaded.append(local_path)
                continue

            if self.download_file(gcs_path, local_path):
                downloaded.append(local_path)
                time.sleep(0.5)  # Rate limiting

        return downloaded

    def process_video(self, video_path: Path, config_path: Optional[Path] = None):
        """Process a video file with YOLOv8n vehicle detection"""
        print(f"\n🎥 Processing: {video_path.name}")

        # Determine which camera config to use
        camera_num = None
        for num, name in self.CAMERA_NAMES.items():
            if name in str(video_path):
                camera_num = num
                break

        if camera_num and not config_path:
            config_path = Path(f"camera_configs/camera_{camera_num}_zones.json")

        if config_path and not config_path.exists():
            print(f"  ⚠️  Warning: Config not found: {config_path}")
            print(f"  ⚠️  Run setup first: ./setup_all_cameras.sh")
            return False

        # Build command
        cmd = ['python', 'vehicle_counting.py', str(video_path)]

        if config_path and config_path.exists():
            cmd.extend(['--config', str(config_path)])

        # Add save-video flag if requested
        if self.save_video:
            cmd.append('--save-video')
            print(f"  📹 Will save annotated video")

        # Run processing
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  ✓ Processing complete")
            return True
        else:
            print(f"  ✗ Processing failed: {result.stderr}")
            return False

    def run_pipeline(self, cameras: Optional[List[int]] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    max_files: Optional[int] = None):
        """Run the complete download and processing pipeline"""
        print("=" * 70)
        print("GCS Traffic Data Pipeline")
        print("=" * 70)
        print(f"Dataset: {self.dataset}")
        print(f"Bucket: {self.bucket}")
        print(f"Output: {self.output_dir}")
        print("=" * 70)

        # Check prerequisites
        if not self.check_gsutil():
            return

        # List available files once
        all_files = self.list_available_files(None, start_date, end_date)
        
        # Filter by cameras if specified
        if cameras:
            filtered_files = []
            for camera in cameras:
                camera_name = self.CAMERA_NAMES.get(camera)
                if camera_name:
                    camera_files = [f for f in all_files if camera_name in f]
                    filtered_files.extend(camera_files)
            all_files = filtered_files

        if not all_files:
            print("✗ No files found matching criteria")
            return

        # Download files
        downloaded = self.download_batch(all_files, max_files)

        if not downloaded:
            print("✗ No files downloaded")
            return

        print(f"\n✓ Downloaded {len(downloaded)} files")

        # Process files if not download-only mode
        if not self.download_only:
            print("\n" + "=" * 70)
            print("Processing Videos")
            print("=" * 70)

            processed = 0
            failed = 0

            for video_path in downloaded:
                if self.process_video(video_path):
                    processed += 1
                else:
                    failed += 1

            print("\n" + "=" * 70)
            print("Pipeline Complete")
            print("=" * 70)
            print(f"✓ Processed: {processed}")
            if failed > 0:
                print(f"✗ Failed: {failed}")

        print(f"\n✓ Pipeline finished successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Download and process traffic videos from Google Cloud Storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and process from small dataset, all cameras
  python download_and_process.py --dataset small --cameras all

  # Download only from full dataset, specific cameras
  python download_and_process.py --dataset full --cameras 1,2 --download-only

  # Process specific date range
  python download_and_process.py --cameras 1 --start-date 2025-10-01 --end-date 2025-10-31

  # Limit number of files
  python download_and_process.py --cameras all --max-files 10
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['small', 'full'],
        default='small',
        help='Dataset to use: small (re-encoded) or full (>500GB)'
    )

    parser.add_argument(
        '--cameras',
        type=str,
        default='all',
        help='Cameras to process: "all" or comma-separated list (e.g., "1,2,3")'
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
        '--max-files',
        type=int,
        help='Maximum number of files to download per camera'
    )

    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download files, do not process them'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='video_processed_files',
        help='Output directory for downloaded files (default: video_processed_files)'
    )

    parser.add_argument(
        '--save-video',
        action='store_true',
        help='Save annotated videos with detections and zones drawn'
    )

    args = parser.parse_args()

    # Parse cameras
    if args.cameras.lower() == 'all':
        cameras = [1, 2, 3, 4]
    else:
        try:
            cameras = [int(c.strip()) for c in args.cameras.split(',')]
        except ValueError:
            print("Error: Invalid camera list. Use 'all' or comma-separated numbers (e.g., '1,2,3')")
            return

    # Create and run pipeline
    pipeline = GCSDataPipeline(
        dataset=args.dataset,
        output_dir=args.output_dir,
        download_only=args.download_only,
        save_video=args.save_video
    )

    pipeline.run_pipeline(
        cameras=cameras,
        start_date=args.start_date,
        end_date=args.end_date,
        max_files=args.max_files
    )


if __name__ == "__main__":
    main()
