#!/usr/bin/env python3
"""
Parallel video processing with YOLOv8n for traffic analysis

This script processes multiple videos in parallel using multiprocessing,
significantly speeding up large-scale data generation.

Usage:
    python parallel_process.py --workers 4 --data-dir _data
    python parallel_process.py --cameras 1,2 --workers 2
"""

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional
import subprocess
import json
import time
from datetime import datetime


class ParallelProcessor:
    """Parallel video processor for traffic analysis"""

    CAMERA_NAMES = {
        1: 'normanniles1',
        2: 'normanniles2',
        3: 'normanniles3',
        4: 'normanniles4'
    }

    def __init__(self, data_dir='video_processed_files', workers=None, save_video=False):
        self.data_dir = Path(data_dir)
        self.workers = workers or max(1, mp.cpu_count() - 1)
        self.save_video = save_video

    def find_videos(self, cameras: Optional[List[int]] = None) -> List[Path]:
        """Find all video files in data directory"""
        videos = []

        if cameras:
            # Search specific camera directories
            for camera_num in cameras:
                camera_name = self.CAMERA_NAMES.get(camera_num)
                if camera_name:
                    camera_dir = self.data_dir / camera_name
                    if camera_dir.exists():
                        videos.extend(camera_dir.glob('*.mp4'))
                        videos.extend(camera_dir.glob('*.avi'))
                        videos.extend(camera_dir.glob('*.mov'))
        else:
            # Search all directories
            for pattern in ['**/*.mp4', '**/*.avi', '**/*.mov']:
                videos.extend(self.data_dir.glob(pattern))

        # Filter out already processed videos
        unprocessed = []
        for video in videos:
            # Check if output files already exist
            counts_file = video.with_suffix('.counts.json')
            if not counts_file.exists():
                unprocessed.append(video)

        return sorted(unprocessed)

    def get_camera_config(self, video_path: Path) -> Optional[Path]:
        """Determine which camera config to use for a video"""
        for camera_num, camera_name in self.CAMERA_NAMES.items():
            if camera_name in str(video_path):
                config_path = Path(f"camera_configs/camera_{camera_num}_zones.json")
                if config_path.exists():
                    return config_path
        return None

    @staticmethod
    def process_single_video(video_path: Path, config_path: Optional[Path], save_video: bool = False) -> dict:
        """Process a single video (worker function)"""
        start_time = time.time()
        worker_name = mp.current_process().name

        result = {
            'video': str(video_path),
            'worker': worker_name,
            'success': False,
            'duration': 0,
            'error': None
        }

        try:
            print(f"[{worker_name}] Processing: {video_path.name}")

            cmd = ['python', 'vehicle_counting.py', str(video_path)]

            if config_path:
                cmd.extend(['--config', str(config_path)])

            if save_video:
                cmd.append('--save-video')

            # Run processing
            proc_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per video
            )

            if proc_result.returncode == 0:
                result['success'] = True
                print(f"[{worker_name}] ✓ Completed: {video_path.name}")
            else:
                result['error'] = proc_result.stderr[:200]
                print(f"[{worker_name}] ✗ Failed: {video_path.name}")

        except subprocess.TimeoutExpired:
            result['error'] = "Processing timeout (>1 hour)"
            print(f"[{worker_name}] ✗ Timeout: {video_path.name}")
        except Exception as e:
            result['error'] = str(e)
            print(f"[{worker_name}] ✗ Error: {video_path.name} - {e}")

        result['duration'] = time.time() - start_time
        return result

    def process_videos(self, videos: List[Path]) -> List[dict]:
        """Process multiple videos in parallel"""
        if not videos:
            print("No videos to process")
            return []

        print(f"\n{'=' * 70}")
        print(f"Parallel Video Processing")
        print(f"{'=' * 70}")
        print(f"Videos to process: {len(videos)}")
        print(f"Workers: {self.workers}")
        print(f"{'=' * 70}\n")

        # Prepare tasks
        tasks = []
        for video in videos:
            config = self.get_camera_config(video)
            tasks.append((video, config, self.save_video))

        # Process in parallel
        with mp.Pool(processes=self.workers) as pool:
            results = pool.starmap(self.process_single_video, tasks)

        return results

    def print_summary(self, results: List[dict]):
        """Print processing summary"""
        print(f"\n{'=' * 70}")
        print("Processing Summary")
        print(f"{'=' * 70}")

        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        print(f"Total videos: {len(results)}")
        print(f"✓ Successful: {len(successful)}")
        print(f"✗ Failed: {len(failed)}")

        if successful:
            total_time = sum(r['duration'] for r in successful)
            avg_time = total_time / len(successful)
            print(f"\nAverage processing time: {avg_time:.1f}s")
            print(f"Total processing time: {total_time:.1f}s")

        if failed:
            print(f"\n❌ Failed videos:")
            for r in failed:
                video_name = Path(r['video']).name
                error = r['error'][:100] if r['error'] else 'Unknown error'
                print(f"  - {video_name}: {error}")

        print(f"{'=' * 70}\n")

        # Save summary to file
        summary_file = Path('processing_summary.json')
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'results': results
            }, f, indent=2)

        print(f"📄 Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Process multiple traffic videos in parallel with YOLOv8n"
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='video_processed_files',
        help='Directory containing video files (default: video_processed_files)'
    )

    parser.add_argument(
        '--cameras',
        type=str,
        help='Comma-separated list of cameras to process (e.g., "1,2,3")'
    )

    parser.add_argument(
        '--workers',
        type=int,
        help=f'Number of parallel workers (default: CPU count - 1)'
    )

    parser.add_argument(
        '--max-videos',
        type=int,
        help='Maximum number of videos to process'
    )

    parser.add_argument(
        '--save-video',
        action='store_true',
        help='Save annotated videos with detections and zones drawn'
    )

    args = parser.parse_args()

    # Parse cameras
    cameras = None
    if args.cameras:
        try:
            cameras = [int(c.strip()) for c in args.cameras.split(',')]
        except ValueError:
            print("Error: Invalid camera list. Use comma-separated numbers (e.g., '1,2,3')")
            return

    # Create processor
    processor = ParallelProcessor(
        data_dir=args.data_dir,
        workers=args.workers,
        save_video=args.save_video
    )

    # Find videos
    videos = processor.find_videos(cameras)

    if not videos:
        print("✓ No unprocessed videos found")
        return

    if args.max_videos:
        videos = videos[:args.max_videos]

    print(f"Found {len(videos)} unprocessed videos")

    # Confirm processing
    print("\nVideos to process:")
    for i, video in enumerate(videos[:10], 1):
        print(f"  {i}. {video.name}")
    if len(videos) > 10:
        print(f"  ... and {len(videos) - 10} more")

    response = input(f"\nProcess {len(videos)} videos with {processor.workers} workers? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled")
        return

    # Process videos
    start_time = time.time()
    results = processor.process_videos(videos)
    total_time = time.time() - start_time

    # Print summary
    processor.print_summary(results)

    print(f"⏱️  Total wall time: {total_time:.1f}s ({total_time/60:.1f}m)")


if __name__ == "__main__":
    main()
