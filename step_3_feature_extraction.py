#!/usr/bin/env python3
"""
Feature Extraction for Roundabout Traffic Analysis

Extracts features from tracking data for traffic congestion prediction:
- Entry/Exit flow metrics
- Circulating flow metrics
- Temporal features
- Origin-Destination matrices
- Speed and density calculations
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from vehicle tracking data"""

    def __init__(
        self,
        tracking_json_path: str,
        time_window_seconds: float = 300.0,  # 5 minutes
        zone_distances: Optional[Dict] = None
    ):
        """
        Initialize feature extractor

        Args:
            tracking_json_path: Path to tracking JSON file
            time_window_seconds: Time window for aggregation (default 300s = 5min)
            zone_distances: Optional dict with zone distances for speed calculation
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.time_window = time_window_seconds
        self.zone_distances = zone_distances or {}

        # Load tracking data
        with open(tracking_json_path, 'r') as f:
            self.tracking_data = json.load(f)

        self.video_name = self.tracking_data['video_name']
        self.vehicles = self.tracking_data['vehicles']
        self.zones = self.tracking_data['zones']

        # Extract video timestamp from filename
        self.video_timestamp = self._extract_video_timestamp()

        self.logger.info(f"Loaded tracking data: {len(self.vehicles)} vehicles")
        self.logger.info(f"Time window: {time_window_seconds}s ({time_window_seconds/60:.1f} min)")

    def _extract_video_timestamp(self) -> Optional[datetime]:
        """Extract timestamp from video filename (e.g., normanniles1_2025-10-20-06-00-45)"""
        import re

        # Pattern: YYYY-MM-DD-HH-MM-SS
        pattern = r'(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})'
        match = re.search(pattern, self.video_name)

        if match:
            year, month, day, hour, minute, second = map(int, match.groups())
            dt = datetime(year, month, day, hour, minute, second)
            self.logger.info(f"Video timestamp: {dt}")
            return dt
        else:
            self.logger.warning(f"Could not extract timestamp from: {self.video_name}")
            return None

    def extract_all_features(self) -> pd.DataFrame:
        """
        Extract all features and return as DataFrame

        Returns:
            DataFrame with features per time window
        """
        # Get video duration
        max_time = max(
            v['last_seen_time']
            for v in self.vehicles
            if v['last_seen_time'] is not None
        )

        # Create time windows
        num_windows = int(np.ceil(max_time / self.time_window))
        time_windows = [
            (i * self.time_window, (i + 1) * self.time_window)
            for i in range(num_windows)
        ]

        self.logger.info(f"Video duration: {max_time:.2f}s ({max_time/60:.1f} min)")
        self.logger.info(f"Number of time windows: {num_windows}")

        # Extract features for each window
        features_list = []
        for window_idx, (start_time, end_time) in enumerate(time_windows):
            features = self._extract_window_features(window_idx, start_time, end_time)
            features_list.append(features)

        # Convert to DataFrame
        df = pd.DataFrame(features_list)

        self.logger.info(f"Extracted {len(df)} feature rows with {len(df.columns)} columns")

        return df

    def _extract_window_features(
        self,
        window_idx: int,
        start_time: float,
        end_time: float
    ) -> Dict:
        """
        Extract features for a specific time window

        Args:
            window_idx: Window index
            start_time: Window start time (seconds)
            end_time: Window end time (seconds)

        Returns:
            Dictionary of features
        """
        features = {
            'window_idx': window_idx,
            'start_time': start_time,
            'end_time': end_time,
            'window_duration': end_time - start_time,
        }

        # Add temporal features
        features.update(self._extract_temporal_features(start_time))

        # Add entry flow metrics
        features.update(self._extract_entry_flow_metrics(start_time, end_time))

        # Add exit flow metrics
        features.update(self._extract_exit_flow_metrics(start_time, end_time))

        # Add circulating flow metrics
        features.update(self._extract_circulating_flow_metrics(start_time, end_time))

        # Add origin-destination metrics
        features.update(self._extract_od_metrics(start_time, end_time))

        # Add speed metrics
        features.update(self._extract_speed_metrics(start_time, end_time))

        # Add derived features
        features.update(self._extract_derived_features(features))

        return features

    def _extract_temporal_features(self, start_time: float) -> Dict:
        """Extract temporal features"""
        features = {}

        if self.video_timestamp:
            # Calculate absolute timestamp for this window
            window_dt = self.video_timestamp + timedelta(seconds=start_time)

            features['hour_of_day'] = window_dt.hour
            features['day_of_week'] = window_dt.weekday()  # 0=Monday, 6=Sunday
            features['is_weekend'] = 1 if window_dt.weekday() >= 5 else 0
            features['is_rush_hour'] = 1 if window_dt.hour in [7, 8, 9, 16, 17, 18] else 0

            # Time since midnight (in hours)
            features['time_since_midnight'] = window_dt.hour + window_dt.minute / 60.0
        else:
            # Use relative time from video start
            features['hour_of_day'] = None
            features['day_of_week'] = None
            features['is_weekend'] = None
            features['is_rush_hour'] = None
            features['time_since_midnight'] = start_time / 3600.0

        return features

    def _extract_entry_flow_metrics(self, start_time: float, end_time: float) -> Dict:
        """
        Extract entry flow metrics

        Counts vehicles crossing entry line during window
        """
        features = {}

        # Count vehicles entering the 'enter' zone
        entry_events = []
        for vehicle in self.vehicles:
            for event in vehicle['events']:
                if (event['zone_name'] == 'enter' and
                    event['event_type'] == 'enter' and
                    start_time <= event['timestamp'] < end_time):
                    entry_events.append({
                        'tracker_id': vehicle['tracker_id'],
                        'class_name': vehicle['class_name'],
                        'timestamp': event['timestamp']
                    })

        # Total entry count
        features['entry_count'] = len(entry_events)

        # Entry count by vehicle type
        entry_by_type = defaultdict(int)
        for event in entry_events:
            entry_by_type[event['class_name']] += 1

        features['entry_count_car'] = entry_by_type.get('car', 0)
        features['entry_count_truck'] = entry_by_type.get('truck', 0)
        features['entry_count_bus'] = entry_by_type.get('bus', 0)
        features['entry_count_motorcycle'] = entry_by_type.get('motorcycle', 0)

        # Entry flow rate (vehicles per minute)
        window_minutes = (end_time - start_time) / 60.0
        features['entry_flow_rate'] = features['entry_count'] / window_minutes if window_minutes > 0 else 0

        return features

    def _extract_exit_flow_metrics(self, start_time: float, end_time: float) -> Dict:
        """Extract exit flow metrics"""
        features = {}

        # Count vehicles entering the 'exit' zone
        exit_events = []
        for vehicle in self.vehicles:
            for event in vehicle['events']:
                if (event['zone_name'] == 'exit' and
                    event['event_type'] == 'enter' and
                    start_time <= event['timestamp'] < end_time):
                    exit_events.append({
                        'tracker_id': vehicle['tracker_id'],
                        'class_name': vehicle['class_name'],
                        'timestamp': event['timestamp']
                    })

        # Total exit count
        features['exit_count'] = len(exit_events)

        # Exit count by vehicle type
        exit_by_type = defaultdict(int)
        for event in exit_events:
            exit_by_type[event['class_name']] += 1

        features['exit_count_car'] = exit_by_type.get('car', 0)
        features['exit_count_truck'] = exit_by_type.get('truck', 0)
        features['exit_count_bus'] = exit_by_type.get('bus', 0)
        features['exit_count_motorcycle'] = exit_by_type.get('motorcycle', 0)

        # Exit flow rate
        window_minutes = (end_time - start_time) / 60.0
        features['exit_flow_rate'] = features['exit_count'] / window_minutes if window_minutes > 0 else 0

        return features

    def _extract_circulating_flow_metrics(self, start_time: float, end_time: float) -> Dict:
        """
        Extract circulating flow metrics

        Measures vehicles in circulation zones
        """
        features = {}

        # Count vehicles in circulating zones at each time point
        # We'll sample at start, middle, and end of window
        sample_times = [start_time, (start_time + end_time) / 2, end_time]

        occupancy_samples = []
        for sample_time in sample_times:
            occupancy = self._count_vehicles_in_zone_at_time(
                'circulating_left',
                sample_time
            ) + self._count_vehicles_in_zone_at_time(
                'circulating_right',
                sample_time
            )
            occupancy_samples.append(occupancy)

        features['circulating_occupancy_avg'] = np.mean(occupancy_samples)
        features['circulating_occupancy_max'] = np.max(occupancy_samples)
        features['circulating_occupancy_min'] = np.min(occupancy_samples)

        # Count vehicles passing through circulating zones
        circulating_events = []
        for vehicle in self.vehicles:
            for event in vehicle['events']:
                if (event['zone_name'] in ['circulating_left', 'circulating_right'] and
                    event['event_type'] == 'enter' and
                    start_time <= event['timestamp'] < end_time):
                    circulating_events.append({
                        'tracker_id': vehicle['tracker_id'],
                        'zone_name': event['zone_name'],
                        'timestamp': event['timestamp']
                    })

        features['circulating_flow_count'] = len(circulating_events)

        # Flow rate
        window_minutes = (end_time - start_time) / 60.0
        features['circulating_flow_rate'] = features['circulating_flow_count'] / window_minutes if window_minutes > 0 else 0

        return features

    def _count_vehicles_in_zone_at_time(self, zone_name: str, timestamp: float) -> int:
        """
        Count how many vehicles are in a zone at a specific timestamp

        A vehicle is in a zone if:
        - It has entered the zone before timestamp
        - It has not yet exited the zone by timestamp
        """
        count = 0
        for vehicle in self.vehicles:
            # Find all events for this zone
            zone_events = [e for e in vehicle['events'] if e['zone_name'] == zone_name]

            # Sort by timestamp
            zone_events = sorted(zone_events, key=lambda e: e['timestamp'])

            # Check if vehicle is in zone at timestamp
            is_in_zone = False
            for event in zone_events:
                if event['timestamp'] > timestamp:
                    break
                if event['event_type'] == 'enter':
                    is_in_zone = True
                elif event['event_type'] == 'exit':
                    is_in_zone = False

            if is_in_zone:
                count += 1

        return count

    def _extract_od_metrics(self, start_time: float, end_time: float) -> Dict:
        """
        Extract Origin-Destination metrics

        Tracks which entry -> which exit patterns
        """
        features = {}

        # Define OD pairs based on valid journey patterns
        od_pairs = [
            ('enter', 'circulating_right'),
            ('enter', 'exit'),
            ('circulating_left', 'circulating_right'),
            ('circulating_left', 'exit'),
        ]

        # Count journeys for each OD pair
        od_counts = defaultdict(int)

        for vehicle in self.vehicles:
            # Check each validation
            if 'validations' in vehicle:
                for validation in vehicle['validations']:
                    if not validation['is_valid']:
                        continue

                    journey_times = validation.get('journey_times')
                    if not journey_times:
                        continue

                    # Check if journey started in this window
                    t1 = journey_times['t1_enter_start_zone']
                    if start_time <= t1 < end_time:
                        origin = validation['start_zone']
                        destination = validation['end_zone']
                        od_counts[(origin, destination)] += 1

        # Create features for each OD pair
        for origin, destination in od_pairs:
            key = f"od_{origin}_to_{destination}"
            features[key] = od_counts[(origin, destination)]

        # Total OD flow
        features['od_total_flow'] = sum(od_counts.values())

        return features

    def _extract_speed_metrics(self, start_time: float, end_time: float) -> Dict:
        """
        Extract speed metrics from journey times

        Requires zone_distances to be provided
        """
        features = {}

        speeds = []
        entry_speeds = []
        circulating_speeds = []

        for vehicle in self.vehicles:
            if 'validations' not in vehicle:
                continue

            for validation in vehicle['validations']:
                if not validation['is_valid']:
                    continue

                journey_times = validation.get('journey_times')
                if not journey_times:
                    continue

                # Check if journey started in this window
                t1 = journey_times['t1_enter_start_zone']
                if not (start_time <= t1 < end_time):
                    continue

                # Extract speed if available
                if 'speed_km_h' in journey_times:
                    speed = journey_times['speed_km_h']
                    speeds.append(speed)

                    # Categorize by zone
                    start_zone = validation['start_zone']
                    if start_zone == 'enter':
                        entry_speeds.append(speed)
                    elif 'circulating' in start_zone:
                        circulating_speeds.append(speed)

        # Overall speed statistics
        if speeds:
            features['speed_avg_km_h'] = np.mean(speeds)
            features['speed_std_km_h'] = np.std(speeds)
            features['speed_min_km_h'] = np.min(speeds)
            features['speed_max_km_h'] = np.max(speeds)
            features['speed_median_km_h'] = np.median(speeds)
        else:
            features['speed_avg_km_h'] = None
            features['speed_std_km_h'] = None
            features['speed_min_km_h'] = None
            features['speed_max_km_h'] = None
            features['speed_median_km_h'] = None

        # Entry zone speeds
        if entry_speeds:
            features['entry_speed_avg_km_h'] = np.mean(entry_speeds)
        else:
            features['entry_speed_avg_km_h'] = None

        # Circulating zone speeds
        if circulating_speeds:
            features['circulating_speed_avg_km_h'] = np.mean(circulating_speeds)
        else:
            features['circulating_speed_avg_km_h'] = None

        return features

    def _extract_derived_features(self, features: Dict) -> Dict:
        """
        Extract derived features from basic features

        These are calculated ratios and relationships
        """
        derived = {}

        # Entry/Exit balance
        entry_count = features.get('entry_count', 0)
        exit_count = features.get('exit_count', 0)

        if entry_count + exit_count > 0:
            derived['entry_exit_balance'] = (entry_count - exit_count) / (entry_count + exit_count)
        else:
            derived['entry_exit_balance'] = 0

        # Entry/Circulating flow ratio (conflict indicator)
        entry_flow = features.get('entry_flow_rate', 0)
        circulating_flow = features.get('circulating_flow_rate', 0)

        if circulating_flow > 0:
            derived['entry_circulating_ratio'] = entry_flow / circulating_flow
        else:
            derived['entry_circulating_ratio'] = entry_flow if entry_flow > 0 else 0

        # Flow imbalance index (variance across entries)
        # For now, we only have one entry zone, so this is placeholder
        derived['flow_imbalance_index'] = 0.0

        # Occupancy to flow ratio
        occupancy = features.get('circulating_occupancy_avg', 0)
        total_flow = features.get('entry_flow_rate', 0) + features.get('exit_flow_rate', 0)

        if total_flow > 0:
            derived['occupancy_flow_ratio'] = occupancy / total_flow
        else:
            derived['occupancy_flow_ratio'] = occupancy

        # Density indicator (vehicles per unit flow)
        if features.get('circulating_flow_rate', 0) > 0:
            derived['circulating_density'] = occupancy / features['circulating_flow_rate']
        else:
            derived['circulating_density'] = occupancy

        return derived

    def add_lag_features(self, df: pd.DataFrame, lag_windows: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        Add lag features (previous window values)

        Args:
            df: DataFrame with features
            lag_windows: List of lag periods (in windows)

        Returns:
            DataFrame with lag features added
        """
        df = df.copy()

        # Key features to lag
        lag_columns = [
            'entry_count', 'entry_flow_rate',
            'exit_count', 'exit_flow_rate',
            'circulating_occupancy_avg',
            'circulating_flow_rate',
            'speed_avg_km_h'
        ]

        for col in lag_columns:
            if col not in df.columns:
                continue

            for lag in lag_windows:
                lag_col_name = f"{col}_lag_{lag}"
                df[lag_col_name] = df[col].shift(lag)

        return df

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [3, 6, 12]
    ) -> pd.DataFrame:
        """
        Add rolling average features

        Args:
            df: DataFrame with features
            windows: List of window sizes (in time windows)
                    e.g., for 5min windows: [3, 6, 12] = [15min, 30min, 60min]

        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()

        # Key features for rolling averages
        rolling_columns = [
            'entry_count', 'entry_flow_rate',
            'exit_count', 'exit_flow_rate',
            'circulating_occupancy_avg',
            'speed_avg_km_h'
        ]

        for col in rolling_columns:
            if col not in df.columns:
                continue

            for window in windows:
                # Rolling mean
                roll_mean_name = f"{col}_rolling_mean_{window}"
                df[roll_mean_name] = df[col].rolling(window=window, min_periods=1).mean()

                # Rolling std
                roll_std_name = f"{col}_rolling_std_{window}"
                df[roll_std_name] = df[col].rolling(window=window, min_periods=1).std()

        return df


def extract_features_from_tracking(
    tracking_json_path: str,
    output_csv_path: Optional[str] = None,
    time_window_seconds: float = 300.0,
    zone_distances_json: Optional[str] = None,
    add_lags: bool = True,
    add_rolling: bool = True
) -> pd.DataFrame:
    """
    Main function to extract features from tracking data

    Args:
        tracking_json_path: Path to tracking JSON file
        output_csv_path: Optional path to save features CSV
        time_window_seconds: Time window for aggregation (default 300s = 5min)
        zone_distances_json: Optional path to zone distances JSON
        add_lags: Whether to add lag features
        add_rolling: Whether to add rolling features

    Returns:
        DataFrame with extracted features
    """
    # Load zone distances if provided
    zone_distances = None
    if zone_distances_json:
        zone_distances_path = Path(zone_distances_json)
        if zone_distances_path.exists():
            with open(zone_distances_path, 'r') as f:
                zone_distances = json.load(f)
            logger.info(f"Loaded zone distances from {zone_distances_json}")

    # Extract features
    extractor = FeatureExtractor(
        tracking_json_path=tracking_json_path,
        time_window_seconds=time_window_seconds,
        zone_distances=zone_distances
    )

    df = extractor.extract_all_features()

    # Add lag features
    if add_lags:
        logger.info("Adding lag features...")
        df = extractor.add_lag_features(df, lag_windows=[1, 2, 3])

    # Add rolling features
    if add_rolling:
        logger.info("Adding rolling features...")
        # For 5min windows: [3, 6, 12] = [15min, 30min, 60min]
        df = extractor.add_rolling_features(df, windows=[3, 6, 12])

    # Save to CSV if requested
    if output_csv_path:
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved features to: {output_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features from tracking data")
    parser.add_argument("tracking_json", type=str, help="Path to tracking JSON file")
    parser.add_argument("--output", type=str, help="Output CSV path")
    parser.add_argument("--window", type=float, default=300.0,
                       help="Time window in seconds (default: 300 = 5min)")
    parser.add_argument("--zone-distances", type=str,
                       help="Path to zone distances JSON file")
    parser.add_argument("--no-lags", action="store_true",
                       help="Disable lag features")
    parser.add_argument("--no-rolling", action="store_true",
                       help="Disable rolling features")

    args = parser.parse_args()

    # Auto-generate output path if not provided
    if not args.output:
        tracking_path = Path(args.tracking_json)
        args.output = str(tracking_path.parent / f"{tracking_path.stem}_features.csv")

    df = extract_features_from_tracking(
        tracking_json_path=args.tracking_json,
        output_csv_path=args.output,
        time_window_seconds=args.window,
        zone_distances_json=args.zone_distances,
        add_lags=not args.no_lags,
        add_rolling=not args.no_rolling
    )

    print("\n" + "="*80)
    print("FEATURE EXTRACTION SUMMARY")
    print("="*80)
    print(f"Input: {args.tracking_json}")
    print(f"Output: {args.output}")
    print(f"Time window: {args.window}s ({args.window/60:.1f} min)")
    print(f"Number of windows: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    print("\nFeature categories:")
    print("  - Temporal features (hour, day, etc.)")
    print("  - Entry flow metrics (count, rate, by vehicle type)")
    print("  - Exit flow metrics (count, rate, by vehicle type)")
    print("  - Circulating flow metrics (occupancy, flow rate)")
    print("  - Origin-Destination matrices")
    print("  - Speed metrics (avg, std, min, max)")
    print("  - Derived features (ratios, density)")
    if not args.no_lags:
        print("  - Lag features (t-1, t-2, t-3)")
    if not args.no_rolling:
        print("  - Rolling features (15min, 30min, 60min)")
    print("\nFirst few rows:")
    print(df.head())
    print("="*80)
