#!/usr/bin/env python3
"""
Fixed Label Integration

Properly matches Train.csv records (1-minute videos) with feature windows (5-minute aggregations)
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FixedTrainingDataIntegrator:
    """Fixed version that properly matches 1-minute Train.csv records with 5-minute feature windows"""
    
    CAMERA_MAPPING = {
        'Norman Niles #1': 'north',
        'Norman Niles #2': 'east',
        'Norman Niles #3': 'south',
        'Norman Niles #4': 'west',
    }
    
    def __init__(self, train_csv_path: str):
        self.train_csv_path = Path(train_csv_path)
        self.train_df = None
        self.load_training_data()
    
    def load_training_data(self):
        """Load and prepare Train.csv"""
        logger.info(f"Loading {self.train_csv_path}...")
        self.train_df = pd.read_csv(self.train_csv_path)
        logger.info(f"Loaded {len(self.train_df)} records")
        
        # Extract video timestamp from videos column
        # Format: normanniles1/normanniles1_2025-10-20-06-00-45.mp4
        self.train_df['video_filename'] = self.train_df['videos'].apply(
            lambda x: x.split('/')[1].replace('.mp4', '') if '/' in x else x.replace('.mp4', '')
        )
        
        # Parse datetimes
        self.train_df['datetime_start'] = pd.to_datetime(self.train_df['datetimestamp_start'])
        self.train_df['datetime_end'] = pd.to_datetime(self.train_df['datetimestamp_end'])
        
        # Map camera names to directions
        self.train_df['camera_direction'] = self.train_df['view_label'].map(self.CAMERA_MAPPING)
        
        logger.info(f"Unique cameras: {self.train_df['view_label'].unique()}")
        logger.info(f"Records per camera: {self.train_df['view_label'].value_counts().to_dict()}")
    
    def get_labels_for_window(
        self,
        camera_direction: str,
        window_start_dt: datetime,
        window_end_dt: datetime,
        aggregation: str = 'mode'
    ) -> Dict:
        """
        Get congestion labels for a specific time window
        
        Args:
            camera_direction: 'north', 'east', 'south', 'west'
            window_start_dt: Window start datetime
            window_end_dt: Window end datetime
            aggregation: 'mode', 'max', or 'last'
        
        Returns:
            Dict with entry/exit labels and metadata
        """
        # Find all Train.csv records that overlap with this window
        # A record overlaps if its time range [datetime_start, datetime_end) intersects with [window_start_dt, window_end_dt)
        mask = (
            (self.train_df['camera_direction'] == camera_direction) &
            (self.train_df['datetime_start'] < window_end_dt) &
            (self.train_df['datetime_end'] > window_start_dt)
        )
        
        matching_records = self.train_df[mask]
        
        if len(matching_records) == 0:
            return {
                'entry_label': None,
                'exit_label': None,
                'label_count': 0,
                'signaling': None
            }
        
        # Aggregate labels
        if aggregation == 'mode':
            entry_label = matching_records['congestion_enter_rating'].mode()
            entry_label = entry_label.iloc[0] if len(entry_label) > 0 else None
            
            exit_label = matching_records['congestion_exit_rating'].mode()
            exit_label = exit_label.iloc[0] if len(exit_label) > 0 else None
        
        elif aggregation == 'max':
            severity_order = {
                'free flowing': 0,
                'light delay': 1,
                'moderate delay': 2,
                'heavy delay': 3
            }
            
            entry_severities = matching_records['congestion_enter_rating'].map(severity_order)
            if len(entry_severities) > 0:
                max_idx = entry_severities.idxmax()
                entry_label = matching_records.loc[max_idx, 'congestion_enter_rating']
            else:
                entry_label = None
            
            exit_severities = matching_records['congestion_exit_rating'].map(severity_order)
            if len(exit_severities) > 0:
                max_idx = exit_severities.idxmax()
                exit_label = matching_records.loc[max_idx, 'congestion_exit_rating']
            else:
                exit_label = None
        
        elif aggregation == 'last':
            # Sort by datetime_start and take the last one
            matching_records_sorted = matching_records.sort_values('datetime_start')
            entry_label = matching_records_sorted.iloc[-1]['congestion_enter_rating']
            exit_label = matching_records_sorted.iloc[-1]['congestion_exit_rating']
        
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Get signaling (mode)
        signaling = matching_records['signaling'].mode()
        signaling = signaling.iloc[0] if len(signaling) > 0 else None
        
        return {
            'entry_label': entry_label,
            'exit_label': exit_label,
            'label_count': len(matching_records),
            'signaling': signaling
        }
    
    def add_labels_to_features(
        self,
        features_df: pd.DataFrame,
        video_timestamp: str,
        aggregation: str = 'mode'
    ) -> pd.DataFrame:
        """
        Add labels to a features DataFrame
        
        Args:
            features_df: DataFrame with features (must have start_time, end_time columns in seconds)
            video_timestamp: Video timestamp string (YYYY-MM-DD-HH-MM-SS)
            aggregation: Label aggregation method
        
        Returns:
            DataFrame with added label columns
        """
        # Parse video timestamp
        try:
            video_dt = datetime.strptime(video_timestamp, "%Y-%m-%d-%H-%M-%S")
        except Exception as e:
            logger.error(f"Could not parse timestamp {video_timestamp}: {e}")
            return features_df
        
        directions = ['north', 'east', 'south', 'west']
        
        for direction in directions:
            entry_labels = []
            exit_labels = []
            label_counts = []
            signaling_values = []
            
            for idx, row in features_df.iterrows():
                # Calculate absolute window times
                window_start_dt = video_dt + timedelta(seconds=row['start_time'])
                window_end_dt = video_dt + timedelta(seconds=row['end_time'])
                
                # Get labels
                labels = self.get_labels_for_window(
                    direction,
                    window_start_dt,
                    window_end_dt,
                    aggregation
                )
                
                entry_labels.append(labels['entry_label'])
                exit_labels.append(labels['exit_label'])
                label_counts.append(labels['label_count'])
                signaling_values.append(labels['signaling'])
            
            # Add to DataFrame
            features_df[f'{direction}_entry_congestion'] = entry_labels
            features_df[f'{direction}_exit_congestion'] = exit_labels
            features_df[f'{direction}_label_count'] = label_counts
            features_df[f'{direction}_signaling'] = signaling_values
            
            # Log coverage
            coverage = pd.Series(entry_labels).notna().sum() / len(entry_labels) * 100
            logger.info(f"  {direction}: {coverage:.1f}% coverage ({label_counts[0] if label_counts else 0} avg records/window)")
        
        return features_df


def fix_training_dataset(
    features_dir: str = 'features_output',
    train_csv: str = 'dataset/Train.csv',
    output_csv: str = 'training_dataset_fixed.csv',
    aggregation: str = 'mode'
):
    """
    Rebuild training dataset with fixed label matching
    
    Args:
        features_dir: Directory with labeled_features_*.csv files
        train_csv: Path to Train.csv
        output_csv: Output path for fixed dataset
        aggregation: Label aggregation method
    """
    features_dir = Path(features_dir)
    
    # Initialize integrator
    integrator = FixedTrainingDataIntegrator(train_csv)
    
    # Find all labeled features files
    feature_files = sorted(features_dir.glob("labeled_features_*.csv"))
    
    if not feature_files:
        logger.error(f"No feature files found in {features_dir}")
        return None
    
    logger.info(f"Found {len(feature_files)} feature files")
    
    all_datasets = []
    
    for feature_file in feature_files:
        # Extract timestamp from filename
        import re
        match = re.search(r'labeled_features_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.csv', feature_file.name)
        if not match:
            logger.warning(f"Could not extract timestamp from {feature_file.name}")
            continue
        
        timestamp = match.group(1)
        logger.info(f"\nProcessing {timestamp}...")
        
        # Load features
        df = pd.read_csv(feature_file)
        
        # Drop existing label columns if present
        label_cols_to_drop = [col for col in df.columns if 
                             any(x in col for x in ['_congestion', '_label_count', '_signaling'])]
        if label_cols_to_drop:
            df = df.drop(columns=label_cols_to_drop)
        
        # Add fixed labels
        df = integrator.add_labels_to_features(df, timestamp, aggregation)
        
        # Add timestamp
        df['video_timestamp'] = timestamp
        
        all_datasets.append(df)
    
    # Combine all
    combined = pd.concat(all_datasets, ignore_index=True)
    logger.info(f"\nCombined dataset: {combined.shape}")
    
    # Check label coverage
    directions = ['north', 'east', 'south', 'west']
    label_cols = [f'{d}_{t}_congestion' for d in directions for t in ['entry', 'exit']]
    
    logger.info("\nLabel coverage:")
    for col in label_cols:
        if col in combined.columns:
            non_null = combined[col].notna().sum()
            pct = (non_null / len(combined)) * 100
            logger.info(f"  {col}: {non_null}/{len(combined)} ({pct:.1f}%)")
    
    # Save
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info(f"\nSaved to: {output_path}")
    
    return combined


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix training dataset label matching")
    parser.add_argument('--features-dir', default='features_output', help='Features directory')
    parser.add_argument('--train-csv', default='dataset/Train.csv', help='Train.csv path')
    parser.add_argument('--output', default='training_dataset_fixed.csv', help='Output CSV')
    parser.add_argument('--aggregation', choices=['mode', 'max', 'last'], default='mode')
    
    args = parser.parse_args()
    
    fix_training_dataset(
        features_dir=args.features_dir,
        train_csv=args.train_csv,
        output_csv=args.output,
        aggregation=args.aggregation
    )
