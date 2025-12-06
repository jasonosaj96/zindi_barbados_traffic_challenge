#!/usr/bin/env python3
"""
Multi-Camera Feature Extraction for Roundabout Traffic Analysis

Merges features from 4 synchronized cameras (North, East, South, West) to create:
- Camera-specific features (per entrance/exit)
- Roundabout-wide features (total flow, occupancy)
- Cross-camera features (directional imbalance, OD patterns)

For ML prediction of 8 outputs: 4 cameras × 2 directions (enter/exit)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MultiCameraFeatureExtractor:
    """Extract and merge features from 4 synchronized roundabout cameras"""

    # Camera mapping: ID -> Direction -> Name
    CAMERA_MAPPING = {
        1: {'direction': 'north', 'abbrev': 'N'},
        2: {'direction': 'east', 'abbrev': 'E'},
        3: {'direction': 'south', 'abbrev': 'S'},
        4: {'direction': 'west', 'abbrev': 'W'}
    }

    # Opposite camera pairs (for traffic balance analysis)
    OPPOSITE_PAIRS = [(1, 3), (2, 4)]  # (North-South), (East-West)

    def __init__(self, feature_csv_paths: Dict[int, str]):
        """
        Initialize multi-camera feature extractor

        Args:
            feature_csv_paths: Dict mapping camera_id to feature CSV path
                              e.g., {1: 'cam1_features.csv', 2: 'cam2_features.csv', ...}
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.feature_csv_paths = feature_csv_paths

        # Validate all 4 cameras provided
        if set(feature_csv_paths.keys()) != {1, 2, 3, 4}:
            raise ValueError(f"Must provide features for all 4 cameras. Got: {list(feature_csv_paths.keys())}")

        # Load camera features
        self.camera_features = {}
        for cam_id, csv_path in feature_csv_paths.items():
            if not Path(csv_path).exists():
                raise FileNotFoundError(f"Camera {cam_id} features not found: {csv_path}")

            df = pd.read_csv(csv_path)
            direction = self.CAMERA_MAPPING[cam_id]['direction']
            self.camera_features[cam_id] = df
            self.logger.info(f"Loaded camera {cam_id} ({direction}): {len(df)} time windows")

        # Validate time alignment
        self._validate_time_alignment()

    def _validate_time_alignment(self):
        """Validate that all cameras have aligned time windows"""
        num_windows = [len(df) for df in self.camera_features.values()]

        if len(set(num_windows)) > 1:
            self.logger.warning(
                f"Cameras have different number of windows: {dict(zip([1,2,3,4], num_windows))}. "
                f"Will use intersection."
            )

        # Check time window alignment
        ref_cam = self.camera_features[1]
        for cam_id in [2, 3, 4]:
            cam_df = self.camera_features[cam_id]
            if not np.allclose(ref_cam['start_time'].values, cam_df['start_time'].values, rtol=1e-3):
                self.logger.warning(f"Camera {cam_id} time windows may not be aligned with camera 1")

    def extract_merged_features(self) -> pd.DataFrame:
        """
        Extract and merge features from all cameras

        Returns:
            DataFrame with:
            - Camera-specific features (prefixed with cam{id}_ or {direction}_)
            - Roundabout-wide features (total_, avg_, etc.)
            - Cross-camera features (imbalance, asymmetry, etc.)
        """
        self.logger.info("Extracting merged multi-camera features...")

        # Start with base temporal features from camera 1
        merged_df = self.camera_features[1][['window_idx', 'start_time', 'end_time', 'window_duration']].copy()

        # Add temporal features if available
        temporal_cols = ['hour_of_day', 'day_of_week', 'is_weekend', 'is_rush_hour', 'time_since_midnight']
        for col in temporal_cols:
            if col in self.camera_features[1].columns:
                merged_df[col] = self.camera_features[1][col]

        # Add camera-specific features (renamed with direction)
        for cam_id in [1, 2, 3, 4]:
            direction = self.CAMERA_MAPPING[cam_id]['direction']
            abbrev = self.CAMERA_MAPPING[cam_id]['abbrev']

            cam_df = self.camera_features[cam_id]

            # Select key features to include
            feature_cols = [
                'entry_count', 'exit_count',
                'entry_flow_rate', 'exit_flow_rate',
                'entry_count_car', 'entry_count_truck', 'entry_count_bus',
                'exit_count_car', 'exit_count_truck', 'exit_count_bus',
                'circulating_occupancy_avg', 'circulating_occupancy_max',
                'circulating_flow_rate',
                'speed_avg_km_h', 'speed_std_km_h',
                'entry_speed_avg_km_h', 'circulating_speed_avg_km_h',
                'entry_exit_balance', 'entry_circulating_ratio',
                'circulating_density'
            ]

            # Add lag features if present
            lag_cols = [col for col in cam_df.columns if '_lag_' in col]
            feature_cols.extend(lag_cols)

            # Add rolling features if present
            rolling_cols = [col for col in cam_df.columns if '_rolling_' in col]
            feature_cols.extend(rolling_cols)

            # Rename and add to merged dataframe
            for col in feature_cols:
                if col in cam_df.columns:
                    # Use direction name for primary features
                    new_col = f"{direction}_{col}"
                    merged_df[new_col] = cam_df[col].values

        # Add roundabout-wide features
        merged_df = self._add_roundabout_features(merged_df)

        # Add cross-camera features
        merged_df = self._add_cross_camera_features(merged_df)

        # Add directional features
        merged_df = self._add_directional_features(merged_df)

        self.logger.info(f"Created merged features: {len(merged_df)} rows × {len(merged_df.columns)} columns")

        return merged_df

    def _add_roundabout_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add roundabout-wide aggregated features"""
        df = df.copy()

        # Total entry/exit counts across all cameras
        entry_cols = [f"{self.CAMERA_MAPPING[i]['direction']}_entry_count" for i in [1,2,3,4]]
        exit_cols = [f"{self.CAMERA_MAPPING[i]['direction']}_exit_count" for i in [1,2,3,4]]

        df['total_entry_count'] = sum(df[col] for col in entry_cols if col in df.columns)
        df['total_exit_count'] = sum(df[col] for col in exit_cols if col in df.columns)

        # Total flow rates
        entry_flow_cols = [f"{self.CAMERA_MAPPING[i]['direction']}_entry_flow_rate" for i in [1,2,3,4]]
        exit_flow_cols = [f"{self.CAMERA_MAPPING[i]['direction']}_exit_flow_rate" for i in [1,2,3,4]]

        df['total_entry_flow_rate'] = sum(df[col] for col in entry_flow_cols if col in df.columns)
        df['total_exit_flow_rate'] = sum(df[col] for col in exit_flow_cols if col in df.columns)

        # Total circulating occupancy
        occupancy_cols = [f"{self.CAMERA_MAPPING[i]['direction']}_circulating_occupancy_avg" for i in [1,2,3,4]]
        df['total_circulating_occupancy'] = sum(df[col] for col in occupancy_cols if col in df.columns)
        df['avg_circulating_occupancy'] = df['total_circulating_occupancy'] / 4.0

        # Average speeds across roundabout
        speed_cols = [f"{self.CAMERA_MAPPING[i]['direction']}_speed_avg_km_h" for i in [1,2,3,4]]
        valid_speeds = [df[col] for col in speed_cols if col in df.columns]
        if valid_speeds:
            # Use nanmean to handle None/NaN values
            df['avg_speed_km_h'] = pd.concat(valid_speeds, axis=1).mean(axis=1, skipna=True)
            df['min_speed_km_h'] = pd.concat(valid_speeds, axis=1).min(axis=1, skipna=True)
            df['max_speed_km_h'] = pd.concat(valid_speeds, axis=1).max(axis=1, skipna=True)

        # System-wide balance
        if 'total_entry_count' in df.columns and 'total_exit_count' in df.columns:
            total_flow = df['total_entry_count'] + df['total_exit_count']
            df['system_flow_balance'] = np.where(
                total_flow > 0,
                (df['total_entry_count'] - df['total_exit_count']) / total_flow,
                0
            )

        return df

    def _add_cross_camera_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features capturing relationships between cameras"""
        df = df.copy()

        # Flow imbalance index (variance across all entry flows)
        entry_flow_cols = [f"{self.CAMERA_MAPPING[i]['direction']}_entry_flow_rate" for i in [1,2,3,4]]
        if all(col in df.columns for col in entry_flow_cols):
            entry_flows = pd.concat([df[col] for col in entry_flow_cols], axis=1)
            df['entry_flow_imbalance'] = entry_flows.std(axis=1, skipna=True)
            df['entry_flow_variance'] = entry_flows.var(axis=1, skipna=True)

        # Exit flow imbalance
        exit_flow_cols = [f"{self.CAMERA_MAPPING[i]['direction']}_exit_flow_rate" for i in [1,2,3,4]]
        if all(col in df.columns for col in exit_flow_cols):
            exit_flows = pd.concat([df[col] for col in exit_flow_cols], axis=1)
            df['exit_flow_imbalance'] = exit_flows.std(axis=1, skipna=True)

        # Occupancy imbalance
        occupancy_cols = [f"{self.CAMERA_MAPPING[i]['direction']}_circulating_occupancy_avg" for i in [1,2,3,4]]
        if all(col in df.columns for col in occupancy_cols):
            occupancies = pd.concat([df[col] for col in occupancy_cols], axis=1)
            df['occupancy_imbalance'] = occupancies.std(axis=1, skipna=True)

        # Opposite entrance balance (North-South, East-West)
        for cam1_id, cam2_id in self.OPPOSITE_PAIRS:
            dir1 = self.CAMERA_MAPPING[cam1_id]['direction']
            dir2 = self.CAMERA_MAPPING[cam2_id]['direction']
            abbrev1 = self.CAMERA_MAPPING[cam1_id]['abbrev']
            abbrev2 = self.CAMERA_MAPPING[cam2_id]['abbrev']

            entry1_col = f"{dir1}_entry_flow_rate"
            entry2_col = f"{dir2}_entry_flow_rate"

            if entry1_col in df.columns and entry2_col in df.columns:
                # Absolute difference
                df[f'{abbrev1}{abbrev2}_entry_diff'] = abs(df[entry1_col] - df[entry2_col])

                # Ratio (handle division by zero)
                total = df[entry1_col] + df[entry2_col]
                df[f'{abbrev1}{abbrev2}_entry_balance'] = np.where(
                    total > 0,
                    (df[entry1_col] - df[entry2_col]) / total,
                    0
                )

        return df

    def _add_directional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for directional traffic patterns"""
        df = df.copy()

        # North-South axis vs East-West axis
        # North-South entry flow
        if 'north_entry_flow_rate' in df.columns and 'south_entry_flow_rate' in df.columns:
            df['NS_total_entry'] = df['north_entry_flow_rate'] + df['south_entry_flow_rate']

        # East-West entry flow
        if 'east_entry_flow_rate' in df.columns and 'west_entry_flow_rate' in df.columns:
            df['EW_total_entry'] = df['east_entry_flow_rate'] + df['west_entry_flow_rate']

        # Dominant axis
        if 'NS_total_entry' in df.columns and 'EW_total_entry' in df.columns:
            total = df['NS_total_entry'] + df['EW_total_entry']
            df['NS_EW_imbalance'] = np.where(
                total > 0,
                (df['NS_total_entry'] - df['EW_total_entry']) / total,
                0
            )

        # Diagonal flows (North to South vs East to West)
        # This captures through-traffic patterns
        if all(f"{self.CAMERA_MAPPING[i]['direction']}_entry_count" in df.columns for i in [1,2,3,4]):
            # Major diagonal: North entry -> South exit + South entry -> North exit
            if 'north_entry_count' in df.columns and 'south_exit_count' in df.columns:
                df['NS_through_flow'] = (df['north_entry_count'] + df['south_entry_count']) / 2

            # Minor diagonal: East entry -> West exit + West entry -> East exit
            if 'east_entry_count' in df.columns and 'west_exit_count' in df.columns:
                df['EW_through_flow'] = (df['east_entry_count'] + df['west_entry_count']) / 2

        # Speed differentials between directions
        speed_cols = {i: f"{self.CAMERA_MAPPING[i]['direction']}_speed_avg_km_h" for i in [1,2,3,4]}
        if all(col in df.columns for col in speed_cols.values()):
            # Max speed difference across all cameras
            speeds = pd.concat([df[col] for col in speed_cols.values()], axis=1)
            df['speed_range'] = speeds.max(axis=1, skipna=True) - speeds.min(axis=1, skipna=True)

        return df

    def create_prediction_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add placeholder columns for the 8 prediction targets

        8 targets = 4 cameras × 2 directions (enter/exit)

        Args:
            df: Merged features DataFrame

        Returns:
            DataFrame with target columns added (initialized to NaN)
        """
        df = df.copy()

        # Create target columns for each camera-direction pair
        for cam_id in [1, 2, 3, 4]:
            direction = self.CAMERA_MAPPING[cam_id]['direction']

            # Target: Congestion level for entry
            df[f'{direction}_entry_congestion'] = np.nan

            # Target: Congestion level for exit
            df[f'{direction}_exit_congestion'] = np.nan

        self.logger.info("Added 8 prediction target columns (initialized to NaN)")
        return df


def extract_multicamera_features(
    feature_csv_paths: Dict[int, str],
    output_csv_path: Optional[str] = None,
    add_target_columns: bool = True
) -> pd.DataFrame:
    """
    Main function to extract multi-camera features

    Args:
        feature_csv_paths: Dict mapping camera_id to feature CSV path
                          e.g., {1: 'cam1_features.csv', 2: 'cam2_features.csv', ...}
        output_csv_path: Optional path to save merged features CSV
        add_target_columns: Whether to add placeholder target columns

    Returns:
        DataFrame with merged features
    """
    extractor = MultiCameraFeatureExtractor(feature_csv_paths)
    df = extractor.extract_merged_features()

    # Add prediction target columns if requested
    if add_target_columns:
        df = extractor.create_prediction_targets(df)

    # Save to CSV if requested
    if output_csv_path:
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved merged features to: {output_path}")

    return df


def batch_process_video_set(
    video_dir: Path,
    tracking_pattern: str = "*_tracking.json",
    features_pattern: str = "*_features.csv",
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Process a complete set of 4 synchronized videos

    Args:
        video_dir: Directory containing tracking JSONs and feature CSVs
        tracking_pattern: Glob pattern for tracking JSON files
        features_pattern: Glob pattern for feature CSV files
        output_dir: Optional directory to save output

    Returns:
        DataFrame with merged features from all 4 cameras
    """
    video_dir = Path(video_dir)

    # Find feature CSV files for each camera
    feature_files = sorted(video_dir.glob(features_pattern))

    if len(feature_files) != 4:
        raise ValueError(
            f"Expected 4 feature CSV files, found {len(feature_files)}: {[f.name for f in feature_files]}"
        )

    # Map camera IDs to feature files based on filename
    # Assumes filenames like: normanniles1_..._features.csv, normanniles2_..._features.csv
    feature_csv_paths = {}
    for feature_file in feature_files:
        # Extract camera number from filename
        import re
        match = re.search(r'normanniles(\d+)', feature_file.name)
        if match:
            cam_id = int(match.group(1))
            feature_csv_paths[cam_id] = str(feature_file)
            logger.info(f"Camera {cam_id}: {feature_file.name}")
        else:
            logger.warning(f"Could not extract camera ID from: {feature_file.name}")

    if len(feature_csv_paths) != 4:
        raise ValueError(f"Could not map all 4 cameras. Found: {list(feature_csv_paths.keys())}")

    # Extract merged features
    output_path = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use timestamp from first video for output filename
        first_file = feature_files[0]
        timestamp_match = re.search(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', first_file.name)
        if timestamp_match:
            timestamp = timestamp_match.group(0)
            output_path = output_dir / f"merged_features_{timestamp}.csv"
        else:
            output_path = output_dir / "merged_features.csv"

    df = extract_multicamera_features(
        feature_csv_paths=feature_csv_paths,
        output_csv_path=str(output_path) if output_path else None,
        add_target_columns=True
    )

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract multi-camera features from 4 synchronized cameras")

    # Two modes: explicit file paths OR directory with auto-discovery
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=str, help="Directory containing all 4 camera feature CSVs")
    group.add_argument("--cam1", type=str, help="Camera 1 (North) features CSV")

    parser.add_argument("--cam2", type=str, help="Camera 2 (East) features CSV")
    parser.add_argument("--cam3", type=str, help="Camera 3 (South) features CSV")
    parser.add_argument("--cam4", type=str, help="Camera 4 (West) features CSV")
    parser.add_argument("--output", type=str, help="Output CSV path")
    parser.add_argument("--no-targets", action="store_true", help="Don't add target columns")

    args = parser.parse_args()

    # Directory mode
    if args.dir:
        df = batch_process_video_set(
            video_dir=Path(args.dir),
            output_dir=Path(args.output).parent if args.output else None
        )

    # Explicit file mode
    else:
        if not all([args.cam1, args.cam2, args.cam3, args.cam4]):
            parser.error("When using --cam1, must also provide --cam2, --cam3, and --cam4")

        feature_csv_paths = {
            1: args.cam1,
            2: args.cam2,
            3: args.cam3,
            4: args.cam4
        }

        df = extract_multicamera_features(
            feature_csv_paths=feature_csv_paths,
            output_csv_path=args.output,
            add_target_columns=not args.no_targets
        )

    print("\n" + "="*80)
    print("MULTI-CAMERA FEATURE EXTRACTION SUMMARY")
    print("="*80)
    print(f"Number of time windows: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    print("\nFeature categories:")
    print("  - Camera-specific features (north_, east_, south_, west_)")
    print("  - Roundabout-wide features (total_, avg_, system_)")
    print("  - Cross-camera features (imbalance, balance)")
    print("  - Directional features (NS_, EW_, through_flow)")
    print("  - Temporal features (hour, day, weekend, rush_hour)")
    if not args.no_targets:
        print("  - Target columns (8): {direction}_{entry|exit}_congestion")
    print(f"\nShape: {df.shape}")
    print("\nFirst few columns:")
    print(df.columns.tolist()[:20])
    print("\nSample data:")
    print(df.head())
    print("="*80)
