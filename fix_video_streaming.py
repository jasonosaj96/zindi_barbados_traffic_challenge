#!/usr/bin/env python3
"""
Fix Video Streaming Issues

This script re-encodes annotated videos to fix moov atom placement issues
that prevent streaming in web browsers. The moov atom needs to be at the
beginning of the file for progressive streaming.

Usage:
    python fix_video_streaming.py --video path/to/video.mp4
    python fix_video_streaming.py --dir video_processed_files --pattern "*.annotated.mp4"
"""

import argparse
import subprocess
from pathlib import Path
from typing import List
import sys


def check_ffmpeg():
    """Check if ffmpeg is installed."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def fix_video_moov_atom(input_path: Path, output_path: Path = None) -> bool:
    """
    Re-encode video with moov atom at the beginning for web streaming.

    Args:
        input_path: Path to input video
        output_path: Path to output video (defaults to input_path with .fixed.mp4)

    Returns:
        True if successful, False otherwise
    """
    if output_path is None:
        output_path = input_path.with_suffix('').with_suffix('.fixed.mp4')

    print(f"Processing: {input_path.name}")
    print(f"Output: {output_path.name}")

    try:
        # Use ffmpeg to re-encode with moov atom at start
        # -movflags +faststart moves moov atom to beginning
        # -c copy would be faster but might not fix the issue
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-c:v', 'libx264',  # Re-encode with H264
            '-preset', 'fast',  # Encoding speed
            '-crf', '23',  # Quality (lower = better, 23 is good)
            '-movflags', '+faststart',  # Move moov atom to start
            '-y',  # Overwrite output
            str(output_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"✓ Successfully fixed: {output_path.name}")

            # Replace original with fixed version if requested
            if output_path.name.endswith('.fixed.mp4'):
                original_backup = input_path.with_suffix('.backup.mp4')
                input_path.rename(original_backup)
                output_path.rename(input_path)
                print(f"✓ Replaced original (backup: {original_backup.name})")

            return True
        else:
            print(f"✗ Error processing video:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"✗ Exception: {e}")
        return False


def find_videos(directory: Path, pattern: str = "*.annotated.mp4") -> List[Path]:
    """Find videos matching pattern in directory."""
    videos = []

    if directory.is_file():
        return [directory]

    for video_path in directory.rglob(pattern):
        if video_path.is_file() and not video_path.name.endswith('.backup.mp4'):
            videos.append(video_path)

    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(
        description='Fix video streaming issues by moving moov atom to beginning'
    )
    parser.add_argument(
        '--video',
        type=str,
        help='Path to single video file to fix'
    )
    parser.add_argument(
        '--dir',
        type=str,
        default='video_processed_files',
        help='Directory to search for videos (default: video_processed_files)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.annotated.mp4',
        help='Glob pattern for videos (default: *.annotated.mp4)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path (only for --video mode)'
    )
    parser.add_argument(
        '--keep-original',
        action='store_true',
        help='Keep original file (create .fixed.mp4 instead of replacing)'
    )

    args = parser.parse_args()

    # Check for ffmpeg
    if not check_ffmpeg():
        print("✗ Error: ffmpeg is not installed or not in PATH")
        print("\nInstall ffmpeg:")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu: sudo apt install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        sys.exit(1)

    print("=" * 70)
    print("Video Streaming Fix - Moov Atom Repositioning")
    print("=" * 70)
    print()

    # Find videos to process
    if args.video:
        videos = [Path(args.video)]
    else:
        videos = find_videos(Path(args.dir), args.pattern)

    if not videos:
        print(f"No videos found matching pattern: {args.pattern}")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) to process\n")

    # Process videos
    success_count = 0
    for video in videos:
        output_path = None
        if args.video and args.output:
            output_path = Path(args.output)
        elif args.keep_original:
            output_path = video.with_suffix('.fixed.mp4')

        if fix_video_moov_atom(video, output_path):
            success_count += 1

        print()

    print("=" * 70)
    print(f"Processed {success_count}/{len(videos)} videos successfully")
    print("=" * 70)

    if success_count < len(videos):
        sys.exit(1)


if __name__ == '__main__':
    main()
