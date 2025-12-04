#!/usr/bin/env python3
"""
Extract Vehicle Zone Duration Data

This script processes the .counts.json files from vehicle counting
and extracts a consolidated JSON with all detected vehicle IDs and
their entry/exit times for each zone.

Output format:
{
    "video_path": "path/to/video.mp4",
    "metadata": {...},
    "vehicles": {
        "1": {
            "tracker_id": 1,
            "class_name": "car",
            "first_seen": 0.5,
            "last_seen": 15.3,
            "zones": {
                "entry": [
                    {
                        "time_entered": 0.5,
                        "time_exited": 3.2,
                        "duration": 2.7,
                        "frame_entered": 15,
                        "frame_exited": 96
                    }
                ],
                "exit_north": [...]
            }
        },
        "2": {...}
    },
    "summary": {
        "total_vehicles": 25,
        "vehicles_per_zone": {...}
    }
}

Usage:
    python extract_vehicle_durations.py path/to/video.counts.json
    python extract_vehicle_durations.py --dir video_processed_files/
    python extract_vehicle_durations.py --dir video_processed_files/ --output vehicle_durations.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


class VehicleDurationExtractor:
    """Extract vehicle zone duration data from counts.json files"""

    def __init__(self):
        self.all_vehicles = {}
        self.file_count = 0
        self.total_vehicles = 0

    def extract_from_counts_file(self, counts_file: Path) -> Dict:
        """Extract vehicle duration data from a single counts.json file"""

        with open(counts_file, 'r') as f:
            data = json.load(f)

        # Extract metadata
        metadata = {
            'video_path': data.get('video_path'),
            'filename': data.get('filename'),
            'camera': data.get('camera'),
            'date': data.get('date'),
            'time': data.get('time'),
            'datetime': data.get('datetime'),
            'duration_seconds': data.get('duration_seconds'),
            'fps': data.get('fps'),
            'total_frames': data.get('total_frames')
        }

        # Extract vehicle journeys
        vehicle_journeys = data.get('vehicle_journeys', [])

        # Convert to the new format: vehicle ID -> zone duration mapping
        vehicles = {}

        for journey in vehicle_journeys:
            vehicle_id = str(journey.get('vehicle_id'))

            vehicle_data = {
                'tracker_id': journey.get('vehicle_id'),
                'class_id': journey.get('class_id'),
                'class_name': journey.get('class_name'),
                'first_seen_time': journey.get('first_seen_time'),
                'last_seen_time': journey.get('last_seen_time'),
                'total_time_visible': journey.get('total_time_visible'),
                'total_frames_tracked': journey.get('total_frames_tracked'),
                'zones': {}
            }

            # Extract zone dwell times
            zones_data = journey.get('zones', {})

            # The zones data structure varies, handle both formats
            if isinstance(zones_data, dict):
                # Format 1: zone_name -> list of visits
                for zone_name, visits in zones_data.items():
                    if isinstance(visits, list):
                        vehicle_data['zones'][zone_name] = []
                        for visit in visits:
                            if isinstance(visit, dict):
                                entry = {
                                    'time_entered': visit.get('time_entered'),
                                    'time_exited': visit.get('time_exited'),
                                    'duration': visit.get('dwell_time'),
                                    'frame_entered': visit.get('frame_entered'),
                                    'frame_exited': visit.get('frame_exited')
                                }
                                vehicle_data['zones'][zone_name].append(entry)

            # Also check zone_dwell_times for alternative format
            zone_dwell_times = journey.get('zone_dwell_times', {})
            if isinstance(zone_dwell_times, dict):
                for zone_name, dwell_times in zone_dwell_times.items():
                    if zone_name not in vehicle_data['zones']:
                        vehicle_data['zones'][zone_name] = []

                    # If we have dwell times but no detailed visit data, create simplified entries
                    if isinstance(dwell_times, list) and dwell_times:
                        if not vehicle_data['zones'][zone_name]:  # Only if we don't have detailed data
                            for dwell_time in dwell_times:
                                entry = {
                                    'time_entered': None,
                                    'time_exited': None,
                                    'duration': dwell_time,
                                    'frame_entered': None,
                                    'frame_exited': None
                                }
                                vehicle_data['zones'][zone_name].append(entry)

            vehicles[vehicle_id] = vehicle_data

        # Calculate summary statistics
        vehicles_per_zone = defaultdict(set)
        for vehicle_id, vehicle_data in vehicles.items():
            for zone_name in vehicle_data['zones'].keys():
                vehicles_per_zone[zone_name].add(vehicle_id)

        summary = {
            'total_vehicles': len(vehicles),
            'vehicles_per_zone': {
                zone: len(vehicle_ids) for zone, vehicle_ids in vehicles_per_zone.items()
            }
        }

        return {
            'metadata': metadata,
            'vehicles': vehicles,
            'summary': summary
        }

    def extract_from_directory(self, data_dir: Path, cameras: Optional[List[int]] = None) -> List[Dict]:
        """Extract vehicle duration data from all counts.json files in a directory"""

        # Find all counts.json files
        counts_files = list(data_dir.glob('**/*.counts.json'))

        if cameras:
            # Filter by camera
            camera_names = [f'normanniles{c}' for c in cameras]
            counts_files = [
                f for f in counts_files
                if any(cam in str(f) for cam in camera_names)
            ]

        results = []

        print(f"Found {len(counts_files)} counts.json files")

        total_files = len(counts_files)
        for index, counts_file in enumerate(sorted(counts_files), start=1):
            print(f"Processing ({index}/{total_files}): {counts_file.name}")
            try:
                result = self.extract_from_counts_file(counts_file)
                result['source_file'] = str(counts_file)
                results.append(result)

                self.file_count += 1
                self.total_vehicles += result['summary']['total_vehicles']

            except Exception as e:
                print(f"  ✗ Error processing {counts_file.name}: {e}")

        return results

    def save_consolidated_output(self, results: List[Dict], output_file: Path):
        """Save all results to a single consolidated JSON file"""

        consolidated = {
            'files_processed': self.file_count,
            'total_vehicles_tracked': self.total_vehicles,
            'videos': results
        }

        with open(output_file, 'w') as f:
            json.dump(consolidated, f, indent=2)

        print(f"\n✓ Consolidated output saved to: {output_file}")
        print(f"  Files processed: {self.file_count}")
        print(f"  Total vehicles tracked: {self.total_vehicles}")

    def save_individual_outputs(self, results: List[Dict], output_dir: Path):
        """Save each video's vehicle duration data to a separate file"""

        output_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            # Determine output filename
            source_file = Path(result['source_file'])
            output_file = output_dir / source_file.name.replace('.counts.json', '.durations.json')

            # Remove source_file from the output
            output_data = {k: v for k, v in result.items() if k != 'source_file'}

            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"  Saved: {output_file.name}")

        print(f"\n✓ Saved {len(results)} individual duration files to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract vehicle zone duration data from counts.json files"
    )

    parser.add_argument(
        'input',
        nargs='?',
        type=str,
        help='Path to a single counts.json file or directory'
    )

    parser.add_argument(
        '--dir',
        type=str,
        help='Directory containing counts.json files (alternative to positional input)'
    )

    parser.add_argument(
        '--cameras',
        type=str,
        help='Comma-separated list of cameras to process (e.g., "1,2,3")'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='vehicle_durations.json',
        help='Output file for consolidated results (default: vehicle_durations.json)'
    )

    parser.add_argument(
        '--individual',
        action='store_true',
        help='Save individual .durations.json files alongside each .counts.json file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save individual duration files (used with --individual)'
    )

    args = parser.parse_args()

    # Determine input path
    input_path = args.input or args.dir
    if not input_path:
        parser.error("Please provide an input file or directory")

    input_path = Path(input_path)

    if not input_path.exists():
        print(f"✗ Error: {input_path} does not exist")
        return

    # Parse cameras
    cameras = None
    if args.cameras:
        try:
            cameras = [int(c.strip()) for c in args.cameras.split(',')]
        except ValueError:
            print("✗ Error: Invalid camera list. Use comma-separated numbers (e.g., '1,2,3')")
            return

    # Create extractor
    extractor = VehicleDurationExtractor()

    # Extract data
    if input_path.is_file():
        print(f"Processing single file: {input_path}")
        result = extractor.extract_from_counts_file(input_path)
        result['source_file'] = str(input_path)
        results = [result]
        extractor.file_count = 1
        extractor.total_vehicles = result['summary']['total_vehicles']
    else:
        print(f"Processing directory: {input_path}")
        results = extractor.extract_from_directory(input_path, cameras)

    if not results:
        print("✗ No data extracted")
        return

    # Save consolidated output
    output_file = Path(args.output)
    extractor.save_consolidated_output(results, output_file)

    # Save individual outputs if requested
    if args.individual:
        output_dir = Path(args.output_dir) if args.output_dir else input_path
        print(f"\nSaving individual duration files...")
        extractor.save_individual_outputs(results, output_dir)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
