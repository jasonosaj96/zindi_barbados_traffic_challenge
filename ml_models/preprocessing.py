"""
Preprocessing module for Barbados Traffic Congestion Challenge

This module contains all preprocessing logic for loading, transforming,
and preparing the traffic data for modeling.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional, List


def extract_timestamp(video_path: str) -> Optional[str]:
    """
    Extract timestamp from video filename
    
    Args:
        video_path: Path to video file containing timestamp pattern
        
    Returns:
        Timestamp string in format YYYY-MM-DD-HH-MM-SS or None if not found
    """
    # Pattern: YYYY-MM-DD-HH-MM-SS
    match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', video_path)
    if match:
        return match.group(1)
    return None


def load_all_features(train_features_dir: str, test_features_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load all feature CSVs from both train and test features directories
    
    Args:
        train_features_dir: Directory with merged_features_*.csv files
        test_features_dir: Directory with test_features_*.csv files
    
    Returns:
        Tuple of (train_features_df, test_features_df)
    """
    train_features = None
    test_features = None
    
    # Load TRAIN features
    train_path = Path(train_features_dir)
    if train_path.exists():
        # Find all merged feature files
        feature_files = list(train_path.glob("merged_features_*.csv"))
        
        if feature_files:
            print(f"Found {len(feature_files)} train feature files")
            
            # Load all features
            dfs = []
            for file in tqdm(feature_files, desc="Loading train features"):
                try:
                    df = pd.read_csv(file)
                    # Extract timestamp from filename: merged_features_2025-10-20-06-00-45.csv
                    timestamp = file.stem.replace('merged_features_', '')
                    df['video_timestamp'] = timestamp
                    dfs.append(df)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
            
            if dfs:
                train_features = pd.concat(dfs, ignore_index=True)
                print(f"✓ Loaded train features: {train_features.shape}")
        else:
            print(f"❌ No train feature files found in {train_features_dir}")
            print("Expected files like: merged_features_2025-10-20-06-00-45.csv")
    else:
        print(f"❌ Train features directory not found: {train_features_dir}")
    
    # Load TEST features
    test_path = Path(test_features_dir)
    if test_path.exists():
        # Find all test feature files
        feature_files = list(test_path.glob("test_features_*.csv"))
        
        if feature_files:
            print(f"Found {len(feature_files)} test feature files")
            
            # Load all features
            dfs = []
            for file in tqdm(feature_files, desc="Loading test features"):
                try:
                    df = pd.read_csv(file)
                    # Extract timestamp from filename: test_features_2025-10-20-06-00-45.csv
                    timestamp = file.stem.replace('test_features_', '')
                    df['video_timestamp'] = timestamp
                    dfs.append(df)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
            
            if dfs:
                test_features = pd.concat(dfs, ignore_index=True)
                print(f"✓ Loaded test features: {test_features.shape}")
        else:
            print(f"❌ No test feature files found in {test_features_dir}")
            print("Expected files like: test_features_2025-10-20-06-00-45.csv")
    else:
        print(f"❌ Test features directory not found: {test_features_dir}")
    
    return train_features, test_features


def add_time_features(df: pd.DataFrame, time_col: str = "video_time") -> pd.DataFrame:
    """
    Add temporal features from datetime column
    
    Args:
        df: DataFrame with datetime column
        time_col: Name of the datetime column
        
    Returns:
        DataFrame with added temporal features
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df["day_week"] = df[time_col].dt.day_of_week   # 0=Mon, 6=Sun
    df["hour"] = df[time_col].dt.hour              # 0–23
    df["Date"] = df[time_col].dt.date
    df["minute"] = df[time_col].dt.minute
    df["is_weekend"] = (df["day_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].apply(lambda x: 1 if x in [7, 8, 9, 16, 17, 18] else 0)
    return df


def encode_signaling(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode signaling categorical feature
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        
    Returns:
        Tuple of (encoded_train_df, encoded_test_df)
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    categories = pd.concat([train_df['signaling'], test_df['signaling']]).unique()
    train_df['signaling'] = pd.Categorical(train_df['signaling'], categories=categories).codes
    test_df['signaling'] = pd.Categorical(test_df['signaling'], categories=categories).codes
    
    return train_df, test_df


def create_long_format(df: pd.DataFrame, label_cols: List[str] = ['congestion_enter_rating', 'congestion_exit_rating']) -> pd.DataFrame:
    """
    Melt ID_enter and ID_exit into a single 'ID' column and create unified congestion_rating
    
    Args:
        df: DataFrame with ID_enter and ID_exit columns
        label_cols: List of label column names
        
    Returns:
        DataFrame in long format with single ID and congestion_rating columns
    """
    # Melt ID_enter and ID_exit into a single 'ID' column
    long_df = df.melt(
        id_vars=[col for col in df.columns if col not in ['ID_enter', 'ID_exit']],
        value_vars=['ID_enter', 'ID_exit'],
        var_name='ID_type',
        value_name='ID'
    )
    
    # Create single congestion_rating column
    long_df['congestion_rating'] = long_df.apply(
        lambda row: row['congestion_enter_rating'] 
                    if row['ID_type'] == 'ID_enter' 
                    else row['congestion_exit_rating'],
        axis=1
    )
    long_df = long_df.drop(columns=['congestion_enter_rating', 'congestion_exit_rating'])
    
    return long_df


def parse_submission_template(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Parse submission file to add test_output_5 rows
    
    Args:
        sub: Sample submission DataFrame
        
    Returns:
        Parsed submission DataFrame with extracted metadata
    """
    ss = sub.copy()
    ss['view_label'] = ss['ID'].apply(lambda x: x.split('_')[3])
    ss['time_segment_id'] = ss['ID'].apply(lambda x: int(x.split('_')[2]))
    ss['ID_type'] = ss['ID'].apply(lambda x: 'ID_' + x.split('#')[1].split('_')[2])
    ss['cycle_phase'] = "test_output_5"
    ss = ss.drop(columns=['Target', 'Target_Accuracy'])
    
    return ss


def select_features(merged_df: pd.DataFrame, 
                   train_features: Optional[pd.DataFrame] = None, 
                   test_features: Optional[pd.DataFrame] = None) -> Tuple[List[str], List[str]]:
    """
    Select features for modeling based on available columns
    
    Args:
        merged_df: Merged DataFrame with all data
        train_features: Train features DataFrame (optional)
        test_features: Test features DataFrame (optional)
        
    Returns:
        Tuple of (all_features, complete_features)
    """
    # Base features (always available)
    base_features = [
        'signaling', 'day_week', 'hour', 'minute',
        'is_weekend', 'is_rush_hour'
    ]
    
    # Advanced features (if available from tracking pipeline)
    advanced_features = []
    if train_features is not None or test_features is not None:
        # Entry/Exit flow features
        flow_features = [
            col for col in merged_df.columns 
            if any(x in col for x in ['entry_count', 'exit_count', 'entry_flow', 'exit_flow'])
        ]
        
        # Circulating features
        circulating_features = [
            col for col in merged_df.columns
            if 'circulating' in col
        ]
        
        # Speed features
        speed_features = [
            col for col in merged_df.columns
            if 'speed' in col
        ]
        
        # OD features
        od_features = [
            col for col in merged_df.columns
            if 'od_' in col
        ]
        
        # Multi-camera features (with camera prefixes)
        camera_features = [
            col for col in merged_df.columns
            if any(cam in col for cam in ['cam1_', 'cam2_', 'cam3_', 'cam4_'])
        ]
        
        # Derived features
        derived_features = [
            col for col in merged_df.columns
            if any(x in col for x in ['balance', 'ratio', 'density', 'imbalance'])
        ]
        
        advanced_features = (
            flow_features + circulating_features + speed_features + 
            od_features + camera_features + derived_features
        )
        
        # Remove duplicates
        advanced_features = list(set(advanced_features))
    
    # Combine all features
    all_features = base_features + advanced_features
    
    # Filter to only existing columns
    features_cols = [col for col in all_features if col in merged_df.columns]
    
    # Check feature completeness
    complete_features = []
    for col in features_cols:
        nan_count = merged_df[col].isna().sum()
        nan_pct = 100 * nan_count / len(merged_df)
        if nan_pct < 100:  # Feature has at least some values
            complete_features.append(col)
        if nan_pct > 50:  # Warn about high NaN percentage
            print(f"  ⚠️ {col}: {nan_pct:.1f}% NaN")
    
    if len(complete_features) < len(features_cols):
        print(f"\n⚠️ {len(features_cols) - len(complete_features)} features are completely empty (100% NaN)")
        print(f"Using {len(complete_features)} features with actual values")
    
    return features_cols, complete_features


def shift_features(merged_df: pd.DataFrame, features_cols: List[str], shift_periods: int = 7) -> pd.DataFrame:
    """
    Apply time-shift to features for forecasting
    
    Args:
        merged_df: Merged DataFrame with all data
        features_cols: List of feature column names to shift
        shift_periods: Number of periods to shift (default: 7)
        
    Returns:
        Shifted DataFrame
    """
    # Only shift the features that actually exist and have values
    available_features = [
        col for col in features_cols 
        if col in merged_df.columns and merged_df[col].notna().any()
    ]
    
    print(f"Available features for shifting: {len(available_features)}/{len(features_cols)}")
    if len(available_features) < len(features_cols):
        missing = set(features_cols) - set(available_features)
        print(f"Features with all NaN values: {missing}")
    
    # Apply shift only on available features
    shifted_df = merged_df.groupby(['view_label', 'ID_type'], as_index=False)[available_features].shift(shift_periods)
    shifted_df['ID'] = merged_df['ID'].values
    shifted_df['cycle_phase'] = merged_df['cycle_phase'].values
    shifted_df['congestion_rating'] = merged_df['congestion_rating'].values
    shifted_df['view_label'] = merged_df['view_label'].values
    shifted_df['ID_type'] = merged_df['ID_type'].values
    
    # IMPORTANT: Only drop NaN for train and validation, NOT for test_output_5
    # For test_output_5, we need to keep all rows (even with NaN) and fill them
    initial_count = len(shifted_df)
    
    # Split before dropping NaN
    train_and_val_mask = shifted_df['cycle_phase'].isin(['train', 'test_input_15'])
    test_mask = shifted_df['cycle_phase'] == 'test_output_5'
    
    # Drop NaN only from train/validation
    train_and_val_df = shifted_df[train_and_val_mask].dropna(subset=available_features)
    test_df_shifted = shifted_df[test_mask].copy()  # Keep all test rows
    
    print(f"\nAfter shifting:")
    print(f"  Train+Validation (after dropna): {len(train_and_val_df)} rows")
    print(f"  Test (kept all rows): {len(test_df_shifted)} rows")
    
    # Combine back
    shifted_df = pd.concat([train_and_val_df, test_df_shifted], ignore_index=True)
    
    print(f"\nCombined shifted dataset: {shifted_df.shape}")
    print(f"Dropped {initial_count - len(shifted_df)} rows from train/val (expected: ~{shift_periods} per view/ID_type group)")
    print(f"Remaining rows by cycle_phase:")
    print(shifted_df['cycle_phase'].value_counts())
    
    if len(shifted_df[shifted_df['cycle_phase'] == 'test_output_5']) != 880:
        print(f"\n⚠️ WARNING: Expected 880 test rows, got {len(shifted_df[shifted_df['cycle_phase'] == 'test_output_5'])}")
    
    return shifted_df


def split_train_val_test(shifted_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into training, validation, and test sets based on cycle_phase
    
    Args:
        shifted_df: Shifted DataFrame with cycle_phase column
        
    Returns:
        Tuple of (training_df, validation_df, testing_df)
    """
    training_df = shifted_df[shifted_df['cycle_phase'] == 'train']
    validation_df = shifted_df[shifted_df['cycle_phase'] == 'test_input_15']
    testing_df = shifted_df[shifted_df['cycle_phase'] == 'test_output_5']
    
    print(f"Training samples: {len(training_df):,}")
    print(f"Validation samples: {len(validation_df):,}")
    print(f"Testing samples: {len(testing_df):,}")
    
    return training_df, validation_df, testing_df


def fill_test_nan(testing_df: pd.DataFrame, training_df: pd.DataFrame, features_cols: List[str]) -> pd.DataFrame:
    """
    Fill NaN values in test set with training medians
    
    Args:
        testing_df: Test DataFrame with potential NaN values
        training_df: Training DataFrame to calculate medians from
        features_cols: List of feature column names
        
    Returns:
        Test DataFrame with filled NaN values
    """
    testing_df = testing_df.copy()
    
    print("Checking for NaN values:")
    print(f"Training NaN count: {training_df[features_cols].isna().sum().sum()}")
    print(f"Testing NaN count: {testing_df[features_cols].isna().sum().sum()}")
    
    if testing_df[features_cols].isna().sum().sum() > 0:
        print("\n⚠️ Test set has NaN values. Filling with training medians...")
        
        # Calculate medians from training data
        medians = training_df[features_cols].median()
        
        # Fill NaN in test set
        testing_df[features_cols] = testing_df[features_cols].fillna(medians)
        
        print(f"✓ Filled NaN values")
        print(f"Testing NaN count after filling: {testing_df[features_cols].isna().sum().sum()}")
    
    return testing_df


def preprocess_data(train_csv: str,
                   test_csv: str,
                   sample_submission_csv: str,
                   train_features_dir: str,
                   test_features_dir: str,
                   shift_periods: int = 7,
                   verbose: bool = True) -> dict:
    """
    Complete preprocessing pipeline
    
    Args:
        train_csv: Path to training CSV file
        test_csv: Path to test CSV file
        sample_submission_csv: Path to sample submission CSV file
        train_features_dir: Directory with train feature files
        test_features_dir: Directory with test feature files
        shift_periods: Number of periods to shift features (default: 7)
        verbose: Whether to print progress messages
        
    Returns:
        Dictionary containing:
        - training_df: Training DataFrame
        - validation_df: Validation DataFrame
        - testing_df: Test DataFrame
        - features_cols: List of feature column names
        - sub: Original submission template
    """
    if verbose:
        print("=" * 80)
        print("PREPROCESSING PIPELINE")
        print("=" * 80)
    
    # 1. Load CSV files
    if verbose:
        print("\n1. Loading CSV files...")
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    sub = pd.read_csv(sample_submission_csv)
    
    if verbose:
        print(f"   Train shape: {train.shape}")
        print(f"   Test shape: {test.shape}")
        print(f"   Submission shape: {sub.shape}")
    
    # 2. Extract timestamps
    if verbose:
        print("\n2. Extracting timestamps...")
    train['video_timestamp'] = train['videos'].apply(extract_timestamp)
    test['video_timestamp'] = test['videos'].apply(extract_timestamp)
    train['video_time'] = pd.to_datetime(train['video_time'])
    test['video_time'] = pd.to_datetime(test['video_time'])
    
    if verbose:
        print(f"   Train timestamps: {train['video_timestamp'].nunique()} unique")
        print(f"   Test timestamps: {test['video_timestamp'].nunique()} unique")
    
    # 3. Load features
    if verbose:
        print("\n3. Loading advanced features...")
    train_features, test_features = load_all_features(train_features_dir, test_features_dir)
    
    # 4. Merge features
    if verbose:
        print("\n4. Merging features with labels...")
    if train_features is not None:
        train_df = train.merge(train_features, on='video_timestamp', how='left')
        if verbose:
            print(f"   Train with features: {train_df.shape}")
    else:
        if verbose:
            print("   Using original train data without advanced features")
        train_df = train.copy()
    
    if test_features is not None:
        test_df = test.merge(test_features, on='video_timestamp', how='left')
        if verbose:
            print(f"   Test with features: {test_df.shape}")
    else:
        if verbose:
            print("   Using original test data without advanced features")
        test_df = test.copy()
    
    # 5. Add temporal features
    if verbose:
        print("\n5. Adding temporal features...")
    train_df = add_time_features(train_df)
    test_df = add_time_features(test_df)
    
    # 6. Encode categorical features
    if verbose:
        print("\n6. Encoding categorical features...")
    train_df, test_df = encode_signaling(train_df, test_df)
    
    # 7. Create long format
    if verbose:
        print("\n7. Creating long format dataset...")
    final_df = pd.concat([train_df, test_df], ignore_index=True)
    long_df = create_long_format(final_df)
    
    # 8. Add submission template rows
    if verbose:
        print("\n8. Adding submission template rows...")
    ss = parse_submission_template(sub)
    merged_df = pd.concat([long_df, ss], ignore_index=True)
    merged_df = merged_df.sort_values(by=['view_label', 'ID_type', 'time_segment_id'])
    merged_df = merged_df.reset_index(drop=True)
    
    if verbose:
        print(f"   Merged dataset: {merged_df.shape}")
        print(f"   Cycle phases: {merged_df['cycle_phase'].unique()}")
    
    # 9. Select features
    if verbose:
        print("\n9. Selecting features for modeling...")
    features_cols, complete_features = select_features(merged_df, train_features, test_features)
    features_cols = complete_features  # Use only complete features
    
    if verbose:
        print(f"   Selected {len(features_cols)} features")
    
    # 10. Shift features
    if verbose:
        print(f"\n10. Shifting features by {shift_periods} periods...")
    shifted_df = shift_features(merged_df, features_cols, shift_periods)
    
    # 11. Split into train/val/test
    if verbose:
        print("\n11. Splitting into train/validation/test...")
    training_df, validation_df, testing_df = split_train_val_test(shifted_df)
    
    # 12. Fill NaN in test set
    if verbose:
        print("\n12. Filling NaN values in test set...")
    testing_df = fill_test_nan(testing_df, training_df, features_cols)
    
    if verbose:
        print("\n" + "=" * 80)
        print("✓ PREPROCESSING COMPLETE")
        print("=" * 80)
    
    return {
        'training_df': training_df,
        'validation_df': validation_df,
        'testing_df': testing_df,
        'features_cols': features_cols,
        'sub': sub
    }
