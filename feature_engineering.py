#!/usr/bin/env python3
"""
Feature Engineering Module for Barbados Traffic Challenge

This module handles temporal feature extraction from traffic data:
- Creates 15-minute sliding windows
- Extracts temporal patterns (trends, rolling stats)
- Handles multi-camera feature fusion
- Saves engineered features for model training

Usage:
    python feature_engineering.py --input dataset/Train.csv --output features/train_features.csv
    python feature_engineering.py --input dataset/Test.csv --output features/test_features.csv --test-mode
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TemporalFeatureEngineer:
    """
    Creates temporal features from traffic data for congestion prediction.

    Features include:
    - Current state (last minute in window)
    - Temporal trends and acceleration
    - Rolling statistics (mean, max, std)
    - Persistence (how long current state has lasted)
    - Direction imbalances
    - Time-based features (hour, day of week, rush hour)
    """

    def __init__(
        self,
        window_size: int = 15,
        prediction_horizon: int = 5,
        embargo_period: int = 2
    ):
        """
        Initialize feature engineer.

        Args:
            window_size: Number of minutes in input window (default: 15)
            prediction_horizon: Minutes ahead to predict (default: 5)
            embargo_period: Processing embargo in minutes (default: 2)
        """
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.embargo_period = embargo_period

        # Congestion level encoding
        self.congestion_levels = ['free flowing', 'light delay', 'moderate delay', 'heavy delay']
        self.congestion_map = {level: i for i, level in enumerate(self.congestion_levels)}

        logger.info(f"Initialized TemporalFeatureEngineer")
        logger.info(f"  Window size: {window_size} minutes")
        logger.info(f"  Prediction horizon: {prediction_horizon} minutes")
        logger.info(f"  Embargo period: {embargo_period} minutes")

    def create_windows(self, df: pd.DataFrame, test_mode: bool = False) -> List[Dict]:
        """
        Create sliding temporal windows from data.

        Args:
            df: DataFrame with traffic data
            test_mode: If True, don't require labels in data

        Returns:
            List of window dictionaries
        """
        logger.info("Creating temporal windows...")

        # Validate data
        required_cols = ['view_label', 'time_segment_id', 'datetimestamp_start']
        if not test_mode:
            required_cols.extend(['congestion_enter_rating', 'congestion_exit_rating'])

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Sort by camera and time
        df = df.sort_values(['view_label', 'time_segment_id']).reset_index(drop=True)

        windows = []
        cameras = df['view_label'].unique()
        logger.info(f"  Processing {len(cameras)} cameras: {cameras}")

        for camera in cameras:
            camera_df = df[df['view_label'] == camera].copy()
            camera_df = camera_df.sort_values('time_segment_id').reset_index(drop=True)

            # Calculate required data points
            required_points = self.window_size + self.embargo_period + self.prediction_horizon

            if len(camera_df) < required_points:
                logger.warning(
                    f"  {camera}: Insufficient data ({len(camera_df)} < {required_points}), skipping"
                )
                continue

            # Create sliding windows
            n_windows = len(camera_df) - required_points + 1

            for i in range(n_windows):
                # Input window (e.g., minutes 0-14)
                window_data = camera_df.iloc[i:i + self.window_size].copy()

                # Target (e.g., minute 17 after 2-minute embargo)
                target_idx = i + self.window_size + self.embargo_period + self.prediction_horizon - 1

                window = {
                    'camera': camera,
                    'window_start_id': window_data.iloc[0]['time_segment_id'],
                    'window_end_id': window_data.iloc[-1]['time_segment_id'],
                    'window_data': window_data,
                }

                if not test_mode:
                    target_row = camera_df.iloc[target_idx]
                    window['target_id'] = target_row['time_segment_id']
                    window['target_enter'] = target_row['congestion_enter_rating']
                    window['target_exit'] = target_row['congestion_exit_rating']

                windows.append(window)

            logger.info(f"  {camera}: Created {n_windows} windows")

        logger.info(f"Total windows created: {len(windows)}")
        return windows

    def extract_features(self, window_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features from a single temporal window.

        Args:
            window_data: DataFrame with window data (15 minutes)

        Returns:
            Dictionary of features
        """
        features = {}

        # 1. TIME-BASED FEATURES
        window_data['datetime'] = pd.to_datetime(window_data['datetimestamp_start'])
        last_datetime = window_data['datetime'].iloc[-1]

        features['hour'] = last_datetime.hour
        features['minute'] = last_datetime.minute
        features['day_of_week'] = last_datetime.dayofweek
        features['is_weekend'] = int(last_datetime.dayofweek >= 5)
        features['is_rush_hour_morning'] = int(7 <= last_datetime.hour <= 9)
        features['is_rush_hour_evening'] = int(16 <= last_datetime.hour <= 18)
        features['is_rush_hour'] = max(features['is_rush_hour_morning'], features['is_rush_hour_evening'])

        # 2. CONGESTION LEVEL FEATURES (encode as numeric)
        enter_encoded = window_data['congestion_enter_rating'].map(self.congestion_map)
        exit_encoded = window_data['congestion_exit_rating'].map(self.congestion_map)

        # Current state (last minute)
        features['current_enter'] = enter_encoded.iloc[-1]
        features['current_exit'] = exit_encoded.iloc[-1]

        # 3. TEMPORAL TRENDS
        features['enter_trend'] = self._compute_trend(enter_encoded.values)
        features['exit_trend'] = self._compute_trend(exit_encoded.values)
        features['enter_acceleration'] = self._compute_acceleration(enter_encoded.values)
        features['exit_acceleration'] = self._compute_acceleration(exit_encoded.values)

        # 4. ROLLING STATISTICS
        # Last 5 minutes
        if len(enter_encoded) >= 5:
            features['enter_mean_5min'] = enter_encoded.iloc[-5:].mean()
            features['exit_mean_5min'] = exit_encoded.iloc[-5:].mean()
            features['enter_max_5min'] = enter_encoded.iloc[-5:].max()
            features['exit_max_5min'] = exit_encoded.iloc[-5:].max()
            features['enter_min_5min'] = enter_encoded.iloc[-5:].min()
            features['exit_min_5min'] = exit_encoded.iloc[-5:].min()
        else:
            features['enter_mean_5min'] = enter_encoded.mean()
            features['exit_mean_5min'] = exit_encoded.mean()
            features['enter_max_5min'] = enter_encoded.max()
            features['exit_max_5min'] = exit_encoded.max()
            features['enter_min_5min'] = enter_encoded.min()
            features['exit_min_5min'] = exit_encoded.min()

        # Full window statistics
        features['enter_mean_full'] = enter_encoded.mean()
        features['exit_mean_full'] = exit_encoded.mean()
        features['enter_max_full'] = enter_encoded.max()
        features['exit_max_full'] = exit_encoded.max()
        features['enter_std_full'] = enter_encoded.std()
        features['exit_std_full'] = exit_encoded.std()

        # 5. VOLATILITY
        features['enter_volatility'] = enter_encoded.std()
        features['exit_volatility'] = exit_encoded.std()

        # 6. DIRECTION IMBALANCE
        features['enter_exit_imbalance'] = features['current_enter'] - features['current_exit']
        features['enter_exit_imbalance_mean'] = features['enter_mean_full'] - features['exit_mean_full']

        # 7. PERSISTENCE
        features['enter_persistence'] = self._compute_persistence(enter_encoded.values)
        features['exit_persistence'] = self._compute_persistence(exit_encoded.values)

        # 8. CHANGE DETECTION
        features['enter_changed_recently'] = int(enter_encoded.iloc[-1] != enter_encoded.iloc[-2] if len(enter_encoded) >= 2 else 0)
        features['exit_changed_recently'] = int(exit_encoded.iloc[-1] != exit_encoded.iloc[-2] if len(exit_encoded) >= 2 else 0)

        # 9. CONGESTION PROGRESSION
        features['enter_worsening'] = int(features['enter_trend'] > 0.1)
        features['exit_worsening'] = int(features['exit_trend'] > 0.1)
        features['enter_improving'] = int(features['enter_trend'] < -0.1)
        features['exit_improving'] = int(features['exit_trend'] < -0.1)

        return features

    def _compute_trend(self, values: np.ndarray) -> float:
        """Compute linear trend (slope)."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return float(coeffs[0])

    def _compute_acceleration(self, values: np.ndarray) -> float:
        """Compute second derivative (acceleration)."""
        if len(values) < 3:
            return 0.0
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 2)
        return float(coeffs[0])  # a in ax^2 + bx + c

    def _compute_persistence(self, values: np.ndarray) -> int:
        """Count consecutive periods with current value."""
        if len(values) == 0:
            return 0
        current = values[-1]
        count = 1
        for i in range(len(values) - 2, -1, -1):
            if values[i] == current:
                count += 1
            else:
                break
        return count

    def process_windows(
        self,
        windows: List[Dict],
        test_mode: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Process all windows and create feature matrix.

        Args:
            windows: List of window dictionaries
            test_mode: If True, don't create labels

        Returns:
            (features_df, labels_df) where labels_df is None in test mode
        """
        logger.info("Extracting features from windows...")

        feature_list = []
        label_list = []

        for i, window in enumerate(windows):
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(windows)} windows")

            # Extract features
            features = self.extract_features(window['window_data'])
            features['camera'] = window['camera']
            features['window_start_id'] = window['window_start_id']
            features['window_end_id'] = window['window_end_id']

            feature_list.append(features)

            # Extract labels if not test mode
            if not test_mode:
                labels = {
                    'target_id': window['target_id'],
                    'target_enter': window['target_enter'],
                    'target_exit': window['target_exit'],
                    'camera': window['camera']
                }
                label_list.append(labels)

        # Create DataFrames
        features_df = pd.DataFrame(feature_list)
        logger.info(f"Feature matrix shape: {features_df.shape}")
        logger.info(f"Features: {[col for col in features_df.columns if col not in ['camera', 'window_start_id', 'window_end_id']]}")

        if test_mode:
            return features_df, None
        else:
            labels_df = pd.DataFrame(label_list)
            logger.info(f"\nTarget distribution (enter):")
            logger.info(f"\n{labels_df['target_enter'].value_counts().sort_index()}")
            logger.info(f"\nTarget distribution (exit):")
            logger.info(f"\n{labels_df['target_exit'].value_counts().sort_index()}")
            return features_df, labels_df

    def save_features(
        self,
        features_df: pd.DataFrame,
        labels_df: Optional[pd.DataFrame],
        output_path: Path
    ):
        """Save engineered features to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save features
        features_df.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")

        # Save labels if available
        if labels_df is not None:
            labels_path = output_path.parent / output_path.name.replace('features', 'labels')
            labels_df.to_csv(labels_path, index=False)
            logger.info(f"Saved labels to {labels_path}")

        # Save feature info
        info = {
            'n_samples': len(features_df),
            'n_features': len(features_df.columns) - 3,  # excluding metadata columns
            'features': list(features_df.columns),
            'window_size': self.window_size,
            'prediction_horizon': self.prediction_horizon,
            'embargo_period': self.embargo_period
        }

        info_path = output_path.parent / 'feature_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Saved feature info to {info_path}")


def main():
    parser = argparse.ArgumentParser(description='Feature engineering for traffic congestion prediction')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file (Train.csv or Test.csv)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output features CSV file')
    parser.add_argument('--window-size', type=int, default=15,
                        help='Temporal window size in minutes (default: 15)')
    parser.add_argument('--prediction-horizon', type=int, default=5,
                        help='Prediction horizon in minutes (default: 5)')
    parser.add_argument('--embargo-period', type=int, default=2,
                        help='Processing embargo in minutes (default: 2)')
    parser.add_argument('--test-mode', action='store_true',
                        help='Test mode (no labels required)')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("FEATURE ENGINEERING - BARBADOS TRAFFIC CHALLENGE")
    logger.info("=" * 80)

    # Load data
    logger.info(f"\nLoading data from {args.input}...")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} records")
    logger.info(f"Cameras: {df['view_label'].unique()}")
    logger.info(f"Time segments: {df['time_segment_id'].min()} - {df['time_segment_id'].max()}")

    # Initialize feature engineer
    engineer = TemporalFeatureEngineer(
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon,
        embargo_period=args.embargo_period
    )

    # Create windows
    windows = engineer.create_windows(df, test_mode=args.test_mode)

    # Extract features
    features_df, labels_df = engineer.process_windows(windows, test_mode=args.test_mode)

    # Save features
    engineer.save_features(features_df, labels_df, Path(args.output))

    logger.info("\n" + "=" * 80)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
