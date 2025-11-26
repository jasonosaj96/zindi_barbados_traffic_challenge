#!/usr/bin/env python3
"""
Batch Roundabout Metrics Analysis

Analyzes multiple traffic videos and generates comprehensive roundabout metrics:
- Processes all .counts.json files in a directory
- Aggregates metrics across time periods
- Identifies peak hours and congestion patterns
- Generates summary reports and visualizations
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import re

try:
    from src.classes.roundabout_metrics import RoundaboutMetricsCalculator, RoundaboutGeometry
except ImportError:
    print("Error: roundabout_metrics.py not found!")
    exit(1)


class BatchRoundaboutAnalyzer:
    """Batch analyzer for roundabout traffic metrics"""
    
    def __init__(self, data_dir: str = "video_processed_files"):
        self.data_dir = Path(data_dir)
        self.results: List[Dict] = []
        self.time_series: Dict[str, List] = defaultdict(list)
        
    def find_counts_files(self, camera: int = None) -> List[Path]:
        """Find all counts.json files"""
        pattern = "**/*.counts.json"
        files = list(self.data_dir.glob(pattern))
        
        if camera:
            camera_name = f"normanniles{camera}"
            files = [f for f in files if camera_name in str(f)]
        
        return sorted(files)
    
    def extract_timestamp_from_filename(self, filepath: Path) -> datetime:
        """Extract timestamp from video filename"""
        # Pattern: normanniles1_2025-10-20-08-59-45.mp4
        match = re.search(r'(\d{4}-\d{2}-\d{2})-(\d{2})-(\d{2})-(\d{2})', filepath.name)
        if match:
            date_str = match.group(1)
            time_str = f"{match.group(2)}:{match.group(3)}:{match.group(4)}"
            return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        return None
    
    def analyze_single_file(self, counts_file: Path) -> Dict:
        """Analyze a single counts file and calculate metrics"""
        with open(counts_file, 'r') as f:
            data = json.load(f)
        
        # Check if metrics file exists
        metrics_file = counts_file.with_suffix('.metrics.json')
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            return {
                'file': str(counts_file),
                'timestamp': self.extract_timestamp_from_filename(counts_file),
                'counts': data,
                'metrics': metrics_data
            }
        
        # Calculate metrics if not already done
        calculator = RoundaboutMetricsCalculator()
        
        zones = data.get('zones', {})
        dwell_times = data.get('dwell_times', {})
        duration = data.get('duration_seconds', 
                          data.get('total_frames', 0) / data.get('fps', 1))
        
        # Group zones by approach
        approaches = set()
        for zone_name in zones.keys():
            parts = zone_name.split('_')
            if len(parts) >= 2:
                approaches.add(parts[0])
        
        for approach in approaches:
            entry_count = zones.get(f'{approach}_entry', 0)
            circ_count = zones.get(f'{approach}_circulating', 0)
            exit_count = zones.get(f'{approach}_exit', 0)
            
            entry_dwell = dwell_times.get(f'{approach}_entry', {})
            entry_times = []
            if 'mean' in entry_dwell and 'count' in entry_dwell:
                entry_times = [entry_dwell['mean']] * entry_dwell['count']
            
            calculator.add_approach_from_zone_data(
                approach_name=approach,
                entry_count=entry_count,
                circulating_count=circ_count,
                exit_count=exit_count,
                duration_seconds=duration,
                entry_dwell_times=entry_times
            )
        
        calculator.calculate_overall_performance()
        metrics_data = calculator.generate_report(str(metrics_file))
        
        return {
            'file': str(counts_file),
            'timestamp': self.extract_timestamp_from_filename(counts_file),
            'counts': data,
            'metrics': metrics_data
        }
    
    def analyze_batch(self, camera: int = None) -> List[Dict]:
        """Analyze all files in batch"""
        files = self.find_counts_files(camera)
        
        print(f"\n{'=' * 70}")
        print(f"Batch Roundabout Metrics Analysis")
        print(f"{'=' * 70}")
        print(f"Data directory: {self.data_dir}")
        print(f"Files to analyze: {len(files)}")
        if camera:
            print(f"Camera: {camera}")
        print(f"{'=' * 70}\n")
        
        results = []
        for i, counts_file in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Analyzing: {counts_file.name}")
            try:
                result = self.analyze_single_file(counts_file)
                results.append(result)
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        self.results = results
        return results
    
    def aggregate_by_time_period(self, period: str = 'hour') -> Dict:
        """
        Aggregate metrics by time period
        
        Args:
            period: 'hour', 'day', or 'all'
        """
        aggregated = defaultdict(lambda: {
            'entering_flow': [],
            'circulating_flow': [],
            'capacity_index': [],
            'level_of_service': [],
            'files': []
        })
        
        for result in self.results:
            timestamp = result['timestamp']
            if not timestamp:
                continue
            
            if period == 'hour':
                key = timestamp.strftime('%Y-%m-%d %H:00')
            elif period == 'day':
                key = timestamp.strftime('%Y-%m-%d')
            else:
                key = 'all'
            
            metrics = result['metrics']
            
            # Aggregate approach metrics
            for approach_name, approach_data in metrics.get('approaches', {}).items():
                aggregated[key]['entering_flow'].append(approach_data['entering_flow'])
                aggregated[key]['circulating_flow'].append(approach_data['circulating_flow'])
                aggregated[key]['capacity_index'].append(approach_data['capacity_index'])
                aggregated[key]['level_of_service'].append(approach_data['level_of_service'])
            
            aggregated[key]['files'].append(result['file'])
        
        # Calculate statistics
        summary = {}
        for key, data in aggregated.items():
            summary[key] = {
                'period': key,
                'num_observations': len(data['files']),
                'avg_entering_flow': np.mean(data['entering_flow']) if data['entering_flow'] else 0,
                'max_entering_flow': np.max(data['entering_flow']) if data['entering_flow'] else 0,
                'avg_circulating_flow': np.mean(data['circulating_flow']) if data['circulating_flow'] else 0,
                'max_circulating_flow': np.max(data['circulating_flow']) if data['circulating_flow'] else 0,
                'avg_capacity_index': np.mean(data['capacity_index']) if data['capacity_index'] else 0,
                'max_capacity_index': np.max(data['capacity_index']) if data['capacity_index'] else 0,
                'dominant_los': max(set(data['level_of_service']), 
                                   key=data['level_of_service'].count) if data['level_of_service'] else 'N/A',
                'files': data['files']
            }
        
        return summary
    
    def identify_peak_hours(self, threshold_percentile: float = 75) -> List[Dict]:
        """Identify peak traffic hours"""
        hourly = self.aggregate_by_time_period('hour')
        
        all_flows = [data['avg_entering_flow'] for data in hourly.values()]
        if not all_flows:
            return []
        
        threshold = np.percentile(all_flows, threshold_percentile)
        
        peaks = []
        for period, data in hourly.items():
            if data['avg_entering_flow'] >= threshold:
                peaks.append({
                    'period': period,
                    'entering_flow': data['avg_entering_flow'],
                    'capacity_index': data['avg_capacity_index'],
                    'level_of_service': data['dominant_los']
                })
        
        return sorted(peaks, key=lambda x: x['entering_flow'], reverse=True)
    
    def generate_summary_report(self, output_path: str = None) -> Dict:
        """Generate comprehensive summary report"""
        if not self.results:
            print("No results to analyze. Run analyze_batch() first.")
            return {}
        
        # Overall statistics
        all_capacity_indices = []
        all_los = []
        all_entering_flows = []
        feasible_count = 0
        
        for result in self.results:
            metrics = result['metrics']
            perf = metrics.get('overall_performance', {})
            
            all_capacity_indices.append(perf.get('overall_capacity_index', 0))
            all_los.append(perf.get('overall_los', 'N/A'))
            all_entering_flows.append(perf.get('total_entering_flow', 0))
            
            if perf.get('is_feasible', True):
                feasible_count += 1
        
        # Time-based aggregations
        hourly_summary = self.aggregate_by_time_period('hour')
        daily_summary = self.aggregate_by_time_period('day')
        peak_hours = self.identify_peak_hours()
        
        report = {
            'analysis_summary': {
                'total_videos_analyzed': len(self.results),
                'date_range': {
                    'start': min(r['timestamp'] for r in self.results if r['timestamp']).isoformat() if self.results else None,
                    'end': max(r['timestamp'] for r in self.results if r['timestamp']).isoformat() if self.results else None
                }
            },
            'overall_statistics': {
                'avg_capacity_index': float(np.mean(all_capacity_indices)) if all_capacity_indices else 0,
                'max_capacity_index': float(np.max(all_capacity_indices)) if all_capacity_indices else 0,
                'min_capacity_index': float(np.min(all_capacity_indices)) if all_capacity_indices else 0,
                'avg_entering_flow': float(np.mean(all_entering_flows)) if all_entering_flows else 0,
                'max_entering_flow': float(np.max(all_entering_flows)) if all_entering_flows else 0,
                'dominant_los': max(set(all_los), key=all_los.count) if all_los else 'N/A',
                'feasibility_rate': (feasible_count / len(self.results) * 100) if self.results else 0
            },
            'temporal_analysis': {
                'hourly_summary': hourly_summary,
                'daily_summary': daily_summary,
                'peak_hours': peak_hours
            },
            'detailed_results': [
                {
                    'file': r['file'],
                    'timestamp': r['timestamp'].isoformat() if r['timestamp'] else None,
                    'overall_los': r['metrics']['overall_performance']['overall_los'],
                    'capacity_index': r['metrics']['overall_performance']['overall_capacity_index'],
                    'entering_flow': r['metrics']['overall_performance']['total_entering_flow']
                }
                for r in self.results
            ]
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nSummary report saved to: {output_path}")
        
        return report
    
    def print_summary(self):
        """Print formatted summary"""
        if not self.results:
            print("No results to display.")
            return
        
        report = self.generate_summary_report()
        
        print("\n" + "=" * 80)
        print("BATCH ROUNDABOUT METRICS ANALYSIS SUMMARY")
        print("=" * 80)
        
        stats = report['overall_statistics']
        print(f"\nOverall Statistics:")
        print(f"  Videos Analyzed:        {report['analysis_summary']['total_videos_analyzed']}")
        print(f"  Average Capacity Index: {stats['avg_capacity_index']:.1f}%")
        print(f"  Max Capacity Index:     {stats['max_capacity_index']:.1f}%")
        print(f"  Average Entering Flow:  {stats['avg_entering_flow']:.1f} veh/h")
        print(f"  Max Entering Flow:      {stats['max_entering_flow']:.1f} veh/h")
        print(f"  Dominant LOS:           {stats['dominant_los']}")
        print(f"  Feasibility Rate:       {stats['feasibility_rate']:.1f}%")
        
        peak_hours = report['temporal_analysis']['peak_hours']
        if peak_hours:
            print(f"\nTop 5 Peak Hours:")
            for i, peak in enumerate(peak_hours[:5], 1):
                print(f"  {i}. {peak['period']}")
                print(f"     Flow: {peak['entering_flow']:.1f} veh/h, "
                      f"CI: {peak['capacity_index']:.1f}%, LOS: {peak['level_of_service']}")
        
        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Batch analysis of roundabout traffic metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all cameras
  python analyze_roundabout_metrics.py
  
  # Analyze specific camera
  python analyze_roundabout_metrics.py --camera 1
  
  # Save detailed report
  python analyze_roundabout_metrics.py --output roundabout_analysis.json
  
  # Print summary to console
  python analyze_roundabout_metrics.py --print-summary
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='video_processed_files',
        help='Directory containing processed video files'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        choices=[1, 2, 3, 4],
        help='Analyze specific camera only'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save summary report JSON'
    )
    
    parser.add_argument(
        '--print-summary',
        action='store_true',
        help='Print formatted summary to console'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = BatchRoundaboutAnalyzer(data_dir=args.data_dir)
    
    # Run batch analysis
    analyzer.analyze_batch(camera=args.camera)
    
    # Generate report
    if args.output or args.print_summary:
        report = analyzer.generate_summary_report(args.output)
    
    # Print summary
    if args.print_summary:
        analyzer.print_summary()


if __name__ == "__main__":
    main()
