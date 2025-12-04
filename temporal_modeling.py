#!/usr/bin/env python3
"""
Temporal Modeling Script for Barbados Traffic Challenge

This script implements Stage 2 of the ML pipeline:
- Loads training data with labels
- Creates 15-minute temporal windows
- Engineers time-series features (trends, rolling stats, flow imbalances)
- Trains XGBoost classifier for congestion prediction
- Evaluates with Macro-F1 (70%) + Accuracy (30%)

Usage:
    python temporal_modeling.py --train dataset/Train.csv --output models/
    python temporal_modeling.py --train dataset/Train.csv --predict --test dataset/Test.csv
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

warnings.filterwarnings('ignore')


class TemporalFeatureEngine:
    """
    Builds temporal features from 15-minute video sequences.

    Features:
    - Current state (minute 15)
    - Trends and acceleration
    - Rolling statistics
    - Multi-camera fusion
    """

    def __init__(self, window_size: int = 15, prediction_horizon: int = 5):
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.label_encoder = LabelEncoder()
        # Congestion order: free flowing < light delay < moderate delay < heavy delay
        self.label_encoder.fit(['free flowing', 'light delay', 'moderate delay', 'heavy delay'])

    def create_temporal_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sliding windows of 15 minutes to predict 5 minutes ahead.

        Structure:
        - Minutes 0-14 (input window)
        - Minutes 15-16 (embargo)
        - Minutes 17-21 (prediction targets)
        """
        print(f"\n📊 Creating temporal windows...")
        print(f"   Window size: {self.window_size} minutes")
        print(f"   Prediction horizon: {self.prediction_horizon} minutes")

        # Sort by camera and time
        df = df.sort_values(['view_label', 'time_segment_id']).reset_index(drop=True)

        # Group by camera view
        windows = []

        for camera in df['view_label'].unique():
            camera_data = df[df['view_label'] == camera].copy()
            camera_data = camera_data.sort_values('time_segment_id').reset_index(drop=True)

            # Create windows
            for i in range(len(camera_data) - self.window_size - self.prediction_horizon):
                window = {
                    'camera': camera,
                    'window_start_id': camera_data.iloc[i]['time_segment_id'],
                    'window_end_id': camera_data.iloc[i + self.window_size - 1]['time_segment_id'],
                    'target_id': camera_data.iloc[i + self.window_size + self.prediction_horizon - 1]['time_segment_id'],
                }

                # Input features (minutes 0-14)
                window_data = camera_data.iloc[i:i + self.window_size]

                # Target labels (minute 17-21, we'll use minute 17 for now)
                target_idx = i + self.window_size + self.prediction_horizon - 1
                target_row = camera_data.iloc[target_idx]

                window['target_enter'] = target_row['congestion_enter_rating']
                window['target_exit'] = target_row['congestion_exit_rating']
                window['window_data'] = window_data

                windows.append(window)

        print(f"   Created {len(windows)} temporal windows")
        return windows

    def extract_baseline_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract simple baseline features from the raw data.

        Note: This assumes we don't have video features yet.
        We'll use metadata and temporal patterns.
        """
        features = {}

        # Time-based features
        df['datetime'] = pd.to_datetime(df['datetimestamp_start'])
        features['hour'] = df['datetime'].dt.hour.iloc[-1]
        features['minute'] = df['datetime'].dt.minute.iloc[-1]
        features['day_of_week'] = df['datetime'].dt.dayofweek.iloc[-1]
        features['is_weekend'] = int(features['day_of_week'] >= 5)
        features['is_rush_hour'] = int((7 <= features['hour'] <= 9) or (16 <= features['hour'] <= 18))

        # Historical congestion encoding (numeric)
        enter_ratings = df['congestion_enter_rating'].map({
            'free flowing': 0,
            'light delay': 1,
            'moderate delay': 2,
            'heavy delay': 3
        })
        exit_ratings = df['congestion_exit_rating'].map({
            'free flowing': 0,
            'light delay': 1,
            'moderate delay': 2,
            'heavy delay': 3
        })

        # Current state (last minute in window)
        features['current_enter_rating'] = enter_ratings.iloc[-1]
        features['current_exit_rating'] = exit_ratings.iloc[-1]

        # Trends over 15 minutes
        features['enter_trend'] = self._compute_trend(enter_ratings.values)
        features['exit_trend'] = self._compute_trend(exit_ratings.values)

        # Rolling statistics (last 5 minutes)
        if len(enter_ratings) >= 5:
            features['enter_mean_5min'] = enter_ratings.iloc[-5:].mean()
            features['exit_mean_5min'] = exit_ratings.iloc[-5:].mean()
            features['enter_max_5min'] = enter_ratings.iloc[-5:].max()
            features['exit_max_5min'] = exit_ratings.iloc[-5:].max()
        else:
            features['enter_mean_5min'] = enter_ratings.mean()
            features['exit_mean_5min'] = exit_ratings.mean()
            features['enter_max_5min'] = enter_ratings.max()
            features['exit_max_5min'] = exit_ratings.max()

        # Volatility
        features['enter_volatility'] = enter_ratings.std()
        features['exit_volatility'] = exit_ratings.std()

        # Direction imbalance (proxy for flow issues)
        features['enter_exit_imbalance'] = features['current_enter_rating'] - features['current_exit_rating']

        # Persistence (how long current state has lasted)
        features['enter_persistence'] = self._compute_persistence(enter_ratings.values)
        features['exit_persistence'] = self._compute_persistence(exit_ratings.values)

        return features

    def _compute_trend(self, values: np.ndarray) -> float:
        """Compute linear trend (slope) of values."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]  # slope

    def _compute_persistence(self, values: np.ndarray) -> int:
        """Count how many consecutive periods the current value has lasted."""
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

    def build_feature_matrix(self, windows: List[Dict]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Build feature matrix X and target vectors y_enter, y_exit.
        """
        print(f"\n🔧 Building feature matrix...")

        X_list = []
        y_enter_list = []
        y_exit_list = []

        for window in windows:
            features = self.extract_baseline_features(window['window_data'])
            X_list.append(features)
            y_enter_list.append(window['target_enter'])
            y_exit_list.append(window['target_exit'])

        X = pd.DataFrame(X_list)
        y_enter = pd.Series(y_enter_list)
        y_exit = pd.Series(y_exit_list)

        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Features: {list(X.columns)}")
        print(f"\n   Target distribution (enter):")
        print(y_enter.value_counts().sort_index())
        print(f"\n   Target distribution (exit):")
        print(y_exit.value_counts().sort_index())

        return X, y_enter, y_exit


class CongestionPredictor:
    """
    XGBoost-based congestion predictor with dual-metric evaluation.
    """

    def __init__(self):
        self.model_enter = None
        self.model_exit = None
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['free flowing', 'light delay', 'moderate delay', 'heavy delay'])

    def train(self, X: pd.DataFrame, y_enter: pd.Series, y_exit: pd.Series):
        """Train separate models for enter and exit predictions."""
        print(f"\n🚀 Training XGBoost models...")

        # Encode labels
        y_enter_encoded = self.label_encoder.transform(y_enter)
        y_exit_encoded = self.label_encoder.transform(y_exit)

        # XGBoost parameters optimized for multi-class
        params = {
            'objective': 'multi:softmax',
            'num_class': 4,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist'
        }

        # Train entrance model
        print(f"   Training entrance congestion model...")
        self.model_enter = xgb.XGBClassifier(**params)
        self.model_enter.fit(X, y_enter_encoded)

        # Train exit model
        print(f"   Training exit congestion model...")
        self.model_exit = xgb.XGBClassifier(**params)
        self.model_exit.fit(X, y_exit_encoded)

        print(f"   ✓ Training complete")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict congestion for both entrance and exit."""
        y_enter_pred = self.model_enter.predict(X)
        y_exit_pred = self.model_exit.predict(X)

        # Decode labels
        y_enter_pred = self.label_encoder.inverse_transform(y_enter_pred)
        y_exit_pred = self.label_encoder.inverse_transform(y_exit_pred)

        return y_enter_pred, y_exit_pred

    def evaluate(self, X: pd.DataFrame, y_enter_true: pd.Series, y_exit_true: pd.Series) -> Dict:
        """
        Evaluate with dual metrics: Macro-F1 (70%) + Accuracy (30%).
        """
        print(f"\n📈 Evaluating models...")

        # Predictions
        y_enter_pred, y_exit_pred = self.predict(X)

        # Entrance metrics
        enter_f1 = f1_score(y_enter_true, y_enter_pred, average='macro')
        enter_acc = accuracy_score(y_enter_true, y_enter_pred)
        enter_score = 0.7 * enter_f1 + 0.3 * enter_acc

        # Exit metrics
        exit_f1 = f1_score(y_exit_true, y_exit_pred, average='macro')
        exit_acc = accuracy_score(y_exit_true, y_exit_pred)
        exit_score = 0.7 * exit_f1 + 0.3 * exit_acc

        # Overall score
        overall_score = (enter_score + exit_score) / 2

        results = {
            'enter': {
                'macro_f1': enter_f1,
                'accuracy': enter_acc,
                'weighted_score': enter_score
            },
            'exit': {
                'macro_f1': exit_f1,
                'accuracy': exit_acc,
                'weighted_score': exit_score
            },
            'overall_score': overall_score
        }

        # Print results
        print(f"\n   ENTRANCE CONGESTION:")
        print(f"      Macro-F1:        {enter_f1:.4f}")
        print(f"      Accuracy:        {enter_acc:.4f}")
        print(f"      Weighted Score:  {enter_score:.4f} (70% F1 + 30% Acc)")

        print(f"\n   EXIT CONGESTION:")
        print(f"      Macro-F1:        {exit_f1:.4f}")
        print(f"      Accuracy:        {exit_acc:.4f}")
        print(f"      Weighted Score:  {exit_score:.4f} (70% F1 + 30% Acc)")

        print(f"\n   OVERALL SCORE: {overall_score:.4f}")

        # Detailed reports
        print(f"\n   Classification Report (Entrance):")
        print(classification_report(y_enter_true, y_enter_pred))

        print(f"\n   Classification Report (Exit):")
        print(classification_report(y_exit_true, y_exit_pred))

        return results

    def get_feature_importance(self, X: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance for interpretability prize."""
        print(f"\n🔍 Feature Importance Analysis...")

        importance_enter = pd.DataFrame({
            'feature': X.columns,
            'importance_enter': self.model_enter.feature_importances_
        }).sort_values('importance_enter', ascending=False)

        importance_exit = pd.DataFrame({
            'feature': X.columns,
            'importance_exit': self.model_exit.feature_importances_
        }).sort_values('importance_exit', ascending=False)

        # Merge
        importance = importance_enter.merge(importance_exit, on='feature')
        importance['importance_avg'] = (importance['importance_enter'] + importance['importance_exit']) / 2
        importance = importance.sort_values('importance_avg', ascending=False)

        print(f"\n   Top {top_n} features:")
        print(importance.head(top_n).to_string(index=False))

        return importance

    def save_model(self, output_dir: Path):
        """Save trained models."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model_enter.save_model(output_dir / 'model_enter.json')
        self.model_exit.save_model(output_dir / 'model_exit.json')

        print(f"\n💾 Models saved to {output_dir}/")

    def load_model(self, output_dir: Path):
        """Load trained models."""
        output_dir = Path(output_dir)

        self.model_enter = xgb.XGBClassifier()
        self.model_enter.load_model(output_dir / 'model_enter.json')

        self.model_exit = xgb.XGBClassifier()
        self.model_exit.load_model(output_dir / 'model_exit.json')

        print(f"\n📂 Models loaded from {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Temporal modeling for traffic congestion prediction')
    parser.add_argument('--train', type=str, default='dataset/Train.csv',
                        help='Path to training CSV')
    parser.add_argument('--test', type=str, default=None,
                        help='Path to test CSV (optional)')
    parser.add_argument('--output', type=str, default='models/',
                        help='Output directory for models')
    parser.add_argument('--window-size', type=int, default=15,
                        help='Temporal window size in minutes (default: 15)')
    parser.add_argument('--prediction-horizon', type=int, default=5,
                        help='Prediction horizon in minutes (default: 5)')
    parser.add_argument('--test-split', type=float, default=0.2,
                        help='Test split ratio (default: 0.2)')
    parser.add_argument('--predict', action='store_true',
                        help='Run prediction on test set')

    args = parser.parse_args()

    print("=" * 80)
    print("  BARBADOS TRAFFIC CHALLENGE - TEMPORAL MODELING")
    print("=" * 80)

    # Load training data
    print(f"\n📂 Loading training data from {args.train}...")
    df_train = pd.read_csv(args.train)
    print(f"   Loaded {len(df_train)} records")
    print(f"   Cameras: {df_train['view_label'].unique()}")
    print(f"   Time segments: {df_train['time_segment_id'].min()} - {df_train['time_segment_id'].max()}")

    # Initialize feature engine
    feature_engine = TemporalFeatureEngine(
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon
    )

    # Create temporal windows
    windows = feature_engine.create_temporal_windows(df_train)

    # Build feature matrix
    X, y_enter, y_exit = feature_engine.build_feature_matrix(windows)

    # Train/validation split
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        indices, test_size=args.test_split, random_state=42, stratify=y_enter
    )

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_enter_train, y_enter_val = y_enter.iloc[train_idx], y_enter.iloc[val_idx]
    y_exit_train, y_exit_val = y_exit.iloc[train_idx], y_exit.iloc[val_idx]

    print(f"\n✂️  Train/Val Split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")

    # Train model
    predictor = CongestionPredictor()
    predictor.train(X_train, y_enter_train, y_exit_train)

    # Evaluate on validation set
    results = predictor.evaluate(X_val, y_enter_val, y_exit_val)

    # Feature importance
    importance = predictor.get_feature_importance(X_train)

    # Save model
    predictor.save_model(args.output)

    # Save feature importance for interpretability prize
    importance_path = Path(args.output) / 'feature_importance.csv'
    importance.to_csv(importance_path, index=False)
    print(f"   Feature importance saved to {importance_path}")

    # Save results
    results_path = Path(args.output) / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Evaluation results saved to {results_path}")

    print("\n" + "=" * 80)
    print(f"  OVERALL SCORE: {results['overall_score']:.4f}")
    print("=" * 80)

    # Prediction on test set (if provided)
    if args.predict and args.test:
        print(f"\n🔮 Running predictions on test set...")
        df_test = pd.read_csv(args.test)

        # TODO: Implement test prediction pipeline
        print(f"   Test prediction not yet implemented (requires test data structure)")


if __name__ == '__main__':
    main()
