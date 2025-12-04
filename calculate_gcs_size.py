#!/usr/bin/env python3
"""
Calculate the total size of files in Google Cloud Storage buckets.
Designed for the Barbados Traffic Challenge project.
Uses gsutil commands (same approach as download_and_process.py).
"""
import subprocess
import sys
import argparse
import re
from typing import Dict, Tuple, Optional


# Project-specific bucket configurations
GCS_BUCKETS = {
    'small': 'gs://brb-traffic/',
    'full': 'gs://brb-traffic-full/'
}

CAMERA_NAMES = ['normanniles1', 'normanniles2', 'normanniles3', 'normanniles4']


def check_gsutil() -> bool:
    """Check if gsutil is installed"""
    try:
        result = subprocess.run(['gsutil', 'version'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    
    print("✗ gsutil not found!", file=sys.stderr)
    print("\nInstall Google Cloud SDK:", file=sys.stderr)
    print("  macOS: brew install --cask google-cloud-sdk", file=sys.stderr)
    print("  Linux: curl https://sdk.cloud.google.com | bash", file=sys.stderr)
    print("  Or visit: https://cloud.google.com/sdk/docs/install", file=sys.stderr)
    return False


def get_bucket_size(bucket_url: str, prefix: Optional[str] = None, camera: Optional[str] = None) -> Tuple[int, int, Dict]:
    """
    Calculate the total size of all files in a GCS bucket using gsutil.
    
    Args:
        bucket_url (str): Full bucket URL (e.g., 'gs://brb-traffic/')
        prefix (str): Optional prefix to filter files
        camera (str): Optional camera name to filter (e.g., 'normanniles1')
        
    Returns:
        tuple: (total_size_bytes, file_count, stats_by_camera)
    """
    # Build the path to scan
    if prefix:
        scan_path = f"{bucket_url.rstrip('/')}/{prefix}"
    else:
        scan_path = bucket_url
    
    print(f"Scanning: {scan_path}")
    if camera:
        print(f"Camera filter: {camera}")
    print("-" * 60)
    
    # Use gsutil ls -L to get detailed file information including sizes
    cmd = ['gsutil', 'ls', '-L', '-r', scan_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ Error listing files: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    
    total_size = 0
    file_count = 0
    stats_by_camera = {cam: {'size': 0, 'count': 0} for cam in CAMERA_NAMES}
    
    # Parse the output
    current_file = None
    current_size = 0
    
    for line in result.stdout.split('\n'):
        line = line.strip()
        
        # File path line (starts with gs://)
        if line.startswith('gs://'):
            # Save previous file stats if it was valid
            if current_file and current_size > 0:
                # Apply camera filter
                if camera and camera not in current_file:
                    current_file = None
                    current_size = 0
                    continue
                    
                total_size += current_size
                file_count += 1
                
                # Track per-camera statistics
                for cam in CAMERA_NAMES:
                    if cam in current_file:
                        stats_by_camera[cam]['size'] += current_size
                        stats_by_camera[cam]['count'] += 1
                        break
            
            current_file = line.rstrip(':')
            current_size = 0
            
        # Size line (contains "Content-Length:")
        elif 'Content-Length:' in line:
            match = re.search(r'Content-Length:\s+(\d+)', line)
            if match:
                current_size = int(match.group(1))
    
    # Don't forget the last file
    if current_file and current_size > 0:
        if not camera or camera in current_file:
            total_size += current_size
            file_count += 1
            
            for cam in CAMERA_NAMES:
                if cam in current_file:
                    stats_by_camera[cam]['size'] += current_size
                    stats_by_camera[cam]['count'] += 1
                    break
    
    return total_size, file_count, stats_by_camera


def format_size(size_bytes):
    """
    Convert bytes to human-readable format.
    
    Args:
        size_bytes (int): Size in bytes
        
    Returns:
        str: Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def main():
    parser = argparse.ArgumentParser(
        description='Calculate total size of files in Google Cloud Storage buckets for Barbados Traffic Challenge',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze both buckets with per-camera breakdown
  python calculate_gcs_size.py --breakdown

  # Analyze only the small dataset
  python calculate_gcs_size.py --dataset small

  # Check size for camera 1 only
  python calculate_gcs_size.py --camera 1 --breakdown

  # Filter by prefix (e.g., specific camera folder)
  python calculate_gcs_size.py --prefix normanniles1 --dataset small
        """
    )
    parser.add_argument(
        '--dataset',
        choices=['small', 'full', 'both'],
        default='both',
        help='Which dataset to analyze (default: both)'
    )
    parser.add_argument(
        '--bucket',
        help='Custom bucket URL (e.g., gs://my-bucket/), overrides --dataset'
    )
    parser.add_argument(
        '--camera',
        choices=['1', '2', '3', '4'],
        help='Filter by specific camera (1-4)'
    )
    parser.add_argument(
        '--prefix',
        help='Filter files by prefix/folder path'
    )
    parser.add_argument(
        '--breakdown',
        action='store_true',
        help='Show per-camera breakdown'
    )
    
    args = parser.parse_args()
    
    # Check if gsutil is available
    if not check_gsutil():
        sys.exit(1)
    
    # Determine which buckets to analyze
    if args.bucket:
        buckets_to_check = [args.bucket]
    elif args.dataset == 'both':
        buckets_to_check = [GCS_BUCKETS['small'], GCS_BUCKETS['full']]
    else:
        buckets_to_check = [GCS_BUCKETS[args.dataset]]
    
    # Convert camera number to name
    camera_filter = None
    if args.camera:
        camera_filter = f'normanniles{args.camera}'
    
    # Process each bucket
    grand_total_size = 0
    grand_total_files = 0
    
    for bucket_url in buckets_to_check:
        print(f"\n{'='*60}")
        total_size, file_count, stats_by_camera = get_bucket_size(
            bucket_url, 
            prefix=args.prefix,
            camera=camera_filter
        )
        
        print(f"\n✓ Results for {bucket_url}:")
        print(f"  Total files: {file_count:,}")
        print(f"  Total size: {format_size(total_size)} ({total_size:,} bytes)")
        
        if args.breakdown and file_count > 0:
            print(f"\n  Per-camera breakdown:")
            for cam in CAMERA_NAMES:
                if stats_by_camera[cam]['count'] > 0:
                    cam_size = stats_by_camera[cam]['size']
                    cam_count = stats_by_camera[cam]['count']
                    print(f"    {cam}: {cam_count:,} files, {format_size(cam_size)}")
        
        grand_total_size += total_size
        grand_total_files += file_count
    
    # Show grand total if multiple buckets
    if len(buckets_to_check) > 1:
        print(f"\n{'='*60}")
        print(f"\n📊 GRAND TOTAL (all buckets):")
        print(f"  Total files: {grand_total_files:,}")
        print(f"  Total size: {format_size(grand_total_size)} ({grand_total_size:,} bytes)")


if __name__ == "__main__":
    main()
