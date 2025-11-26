#!/usr/bin/env python3
"""
Example script demonstrating vehicle tracking and dwell time analysis.
"""

import json
from pathlib import Path
from vehicle_counting import VehicleCounter

def main():
    # Example video path (update with your actual video)
    video_path = "_data/normanniles1/normanniles1_2025-10-25-08-41-45.mp4"
    config_path = "camera_configs/camera_1.json"

    print("=" * 70)
    print("Vehicle Tracking Example")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Config: {config_path}")
    print()

    # Check if video exists
    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        print("Please update the video_path in this script to point to a valid video.")
        return

    # Load camera configuration
    with open(config_path, 'r') as f:
        camera_config = json.load(f)

    print("Processing with vehicle tracking enabled...")
    print()

    # Initialize counter with tracking enabled
    counter = VehicleCounter(
        model_path="yolov8n.pt",
        camera_config=camera_config,
        confidence_threshold=0.3,
        enable_tracking=True  # Enable tracking
    )

    # Process video
    counts = counter.process_video(
        video_path=video_path,
        output_path=None,  # Don't save annotated video
        display=False,
        save_counts=True
    )

    print()
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)

    # Load the saved JSON to show full results
    counts_path = Path(video_path).with_suffix('.counts.json')
    with open(counts_path, 'r') as f:
        data = json.load(f)

    print("\nVehicle Counts:")
    for zone_name, count in data['zones'].items():
        print(f"  {zone_name}: {count}")

    if 'dwell_times' in data:
        print("\nDwell Time Statistics (seconds):")
        for zone_name, stats in data['dwell_times'].items():
            print(f"\n  {zone_name}:")
            print(f"    Mean: {stats['mean']:.2f}s")
            print(f"    Median: {stats['median']:.2f}s")
            print(f"    Range: {stats['min']:.2f}s - {stats['max']:.2f}s")
            print(f"    Vehicles tracked: {stats['count']}")

    print()
    print(f"Full results saved to: {counts_path}")
    print()

    # Example analysis
    print("=" * 70)
    print("Analysis Example")
    print("=" * 70)

    if 'dwell_times' in data and 'enter_in' in data['dwell_times']:
        enter_stats = data['dwell_times']['enter_in']

        print("\nCongestion Indicators:")

        # Analyze enter lane
        if enter_stats['mean'] > 5.0:
            print("  ⚠️  High dwell time in entry lane suggests congestion")
            print(f"     Average wait time: {enter_stats['mean']:.2f}s")
        elif enter_stats['mean'] > 3.0:
            print("  ⚡ Moderate dwell time - traffic flowing with some delays")
            print(f"     Average wait time: {enter_stats['mean']:.2f}s")
        else:
            print("  ✓  Low dwell time - good traffic flow")
            print(f"     Average wait time: {enter_stats['mean']:.2f}s")

        # Check for outliers
        if enter_stats['max'] > enter_stats['mean'] * 2:
            print(f"  📊 Some vehicles waited much longer (max: {enter_stats['max']:.2f}s)")
            print("     This suggests intermittent queue buildup")

    print()

if __name__ == "__main__":
    main()
