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
from typing import List, Optional, Tuple
import subprocess
import json
import time
from datetime import datetime
import sys
import threading


class ParallelProcessor:
    """Parallel video processor for traffic analysis"""

    CAMERA_NAMES = {
        1: 'normanniles1',
        2: 'normanniles2',
        3: 'normanniles3',
        4: 'normanniles4'
    }

    def __init__(self, data_dir='video_processed_files', workers=None, save_video=False, 
                 auto_metrics=True, resume=False):
        self.data_dir = Path(data_dir)
        self.workers = workers or max(1, mp.cpu_count() - 1)
        self.save_video = save_video
        self.auto_metrics = auto_metrics
        self.resume = resume
        self.progress_lock = threading.Lock()
        self.completed_count = 0
        self.total_count = 0

    def find_videos(self, cameras: Optional[List[int]] = None) -> Tuple[List[Path], List[Path]]:
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

        # Filter based on processing status
        unprocessed = []
        needs_metrics = []
        
        for video in videos:
            counts_file = video.with_suffix('.counts.json')
            metrics_file = video.with_suffix('.metrics.json')
            
            if not counts_file.exists():
                # Needs full processing
                unprocessed.append(video)
            elif self.auto_metrics and not metrics_file.exists():
                # Has counts but needs metrics
                needs_metrics.append(video)

        return sorted(unprocessed), sorted(needs_metrics)

    def get_camera_config(self, video_path: Path) -> Optional[Path]:
        """Determine which camera config to use for a video"""
        for camera_num, camera_name in self.CAMERA_NAMES.items():
            if camera_name in str(video_path):
                config_path = Path(f"camera_configs/camera_{camera_num}_zones.json")
                if config_path.exists():
                    return config_path
        return None

    @staticmethod
    def process_single_video(video_path: Path, config_path: Optional[Path], 
                           save_video: bool = False, progress_callback=None) -> dict:
        """Process a single video (worker function)"""
        start_time = time.time()
        worker_name = mp.current_process().name

        result = {
            'video': str(video_path),
            'worker': worker_name,
            'success': False,
            'duration': 0,
            'error': None,
            'has_metrics': False
        }

        try:
            print(f"[{worker_name}] Processing: {video_path.name}")

            cmd = [sys.executable, 'src/classes/vehicle_counting.py', str(video_path)]

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
                # Check if metrics were generated
                metrics_file = video_path.with_suffix('.metrics.json')
                result['has_metrics'] = metrics_file.exists()
                print(f"[{worker_name}] ✓ Completed: {video_path.name}")
                if result['has_metrics']:
                    print(f"[{worker_name}]   • Metrics generated")
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
    
    @staticmethod
    def generate_metrics_for_video(video_path: Path) -> dict:
        """Generate roundabout metrics for an already processed video"""
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
            counts_file = video_path.with_suffix('.counts.json')
            if not counts_file.exists():
                result['error'] = "Counts file not found"
                return result
            
            print(f"[{worker_name}] Generating metrics: {video_path.name}")
            
            cmd = [sys.executable, 'src/classes/roundabout_metrics.py', str(counts_file)]
            
            proc_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout for metrics
            )
            
            if proc_result.returncode == 0:
                result['success'] = True
                print(f"[{worker_name}] ✓ Metrics generated: {video_path.name}")
            else:
                result['error'] = proc_result.stderr[:200]
                print(f"[{worker_name}] ✗ Metrics failed: {video_path.name}")
        
        except subprocess.TimeoutExpired:
            result['error'] = "Metrics timeout"
        except Exception as e:
            result['error'] = str(e)
        
        result['duration'] = time.time() - start_time
        return result

    def process_videos(self, videos: List[Path]) -> List[dict]:
        """Process multiple videos in parallel"""
        if not videos:
            print("No videos to process")
            return []

        self.total_count = len(videos)
        self.completed_count = 0

        print(f"\n{'=' * 70}")
        print(f"Parallel Video Processing")
        print(f"{'=' * 70}")
        print(f"Videos to process: {len(videos)}")
        print(f"Workers: {self.workers}")
        print(f"Auto-generate metrics: {self.auto_metrics}")
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
    
    def generate_metrics_batch(self, videos: List[Path]) -> List[dict]:
        """Generate metrics for multiple videos in parallel"""
        if not videos:
            return []
        
        print(f"\n{'=' * 70}")
        print(f"Generating Roundabout Metrics")
        print(f"{'=' * 70}")
        print(f"Videos: {len(videos)}")
        print(f"Workers: {self.workers}")
        print(f"{'=' * 70}\n")
        
        with mp.Pool(processes=self.workers) as pool:
            results = pool.map(self.generate_metrics_for_video, videos)
        
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
    
    def run_batch_analysis(self, camera: Optional[int] = None):
        """Run batch roundabout metrics analysis"""
        print(f"\n{'=' * 70}")
        print("Running Batch Roundabout Analysis")
        print(f"{'=' * 70}\n")
        
        try:
            cmd = [sys.executable, 'src/classes/analyze_roundabout_metrics.py', 
                   '--data-dir', str(self.data_dir),
                   '--output', 'batch_roundabout_analysis.json',
                   '--print-summary']
            
            if camera:
                cmd.extend(['--camera', str(camera)])
            
            result = subprocess.run(cmd, timeout=300)
            
            if result.returncode == 0:
                print(f"\n✓ Batch analysis complete")
            else:
                print(f"\n⚠️  Batch analysis encountered issues")
        
        except Exception as e:
            print(f"\n⚠️  Could not run batch analysis: {e}")


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
    
    parser.add_argument(
        '--no-metrics',
        action='store_true',
        help='Skip automatic roundabout metrics generation'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume processing (generate metrics for videos with counts but no metrics)'
    )
    
    parser.add_argument(
        '--batch-analysis',
        action='store_true',
        help='Run batch analysis after processing'
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
        save_video=args.save_video,
        auto_metrics=not args.no_metrics,
        resume=args.resume
    )

    # Find videos
    videos, needs_metrics = processor.find_videos(cameras)
    
    # Handle resume mode
    if args.resume and needs_metrics:
        print(f"\n📊 Resume Mode: Found {len(needs_metrics)} videos needing metrics")
        
        if needs_metrics:
            print("\nVideos needing metrics:")
            for i, video in enumerate(needs_metrics[:10], 1):
                print(f"  {i}. {video.name}")
            if len(needs_metrics) > 10:
                print(f"  ... and {len(needs_metrics) - 10} more")
            
            response = input(f"\nGenerate metrics for {len(needs_metrics)} videos? (y/n): ")
            if response.lower() == 'y':
                start_time = time.time()
                metrics_results = processor.generate_metrics_batch(needs_metrics)
                metrics_time = time.time() - start_time
                
                successful = sum(1 for r in metrics_results if r['success'])
                print(f"\n✓ Generated metrics for {successful}/{len(needs_metrics)} videos")
                print(f"⏱️  Time: {metrics_time:.1f}s")

    if not videos:
        print("✓ No unprocessed videos found")
        
        # Run batch analysis if requested
        if args.batch_analysis:
            camera_num = cameras[0] if cameras and len(cameras) == 1 else None
            processor.run_batch_analysis(camera_num)
        return

    if args.max_videos:
        videos = videos[:args.max_videos]

    print(f"\nFound {len(videos)} unprocessed videos")
    if needs_metrics:
        print(f"Found {len(needs_metrics)} videos needing metrics (use --resume to generate)")

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
    
    # Run batch analysis if requested
    if args.batch_analysis:
        camera_num = cameras[0] if cameras and len(cameras) == 1 else None
        processor.run_batch_analysis(camera_num)


if __name__ == "__main__":
    main()
