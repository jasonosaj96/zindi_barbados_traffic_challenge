"""
Create rolling window sequences for traffic congestion prediction.

This script transforms the existing training dataset (one row per 5-minute window)
into sequences of windows suitable for temporal models (LSTM, GRU, Transformers).

Target columns:
- north_entry_congestion
- east_entry_congestion
- south_entry_congestion
- west_entry_congestion
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from pathlib import Path


def load_and_prepare_data(input_csv: str) -> pd.DataFrame:
    """Load training dataset and prepare datetime index."""
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Convert video_timestamp to datetime
    df['video_timestamp'] = pd.to_datetime(df['video_timestamp'], format='%Y-%m-%d-%H-%M-%S')

    # Sort by timestamp to ensure temporal order
    df = df.sort_values('video_timestamp').reset_index(drop=True)

    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['video_timestamp'].min()} to {df['video_timestamp'].max()}")
    print(f"Unique timestamps: {df['video_timestamp'].nunique()}")

    return df


def get_target_columns(df: pd.DataFrame) -> list:
    """Identify target columns for congestion prediction."""
    target_cols = [
        'north_entry_congestion',
        'east_entry_congestion',
        'south_entry_congestion',
        'west_entry_congestion'
    ]

    # Verify all target columns exist
    available_targets = [col for col in target_cols if col in df.columns]

    if len(available_targets) < len(target_cols):
        missing = set(target_cols) - set(available_targets)
        print(f"Warning: Missing target columns: {missing}")

    print(f"Target columns: {available_targets}")
    return available_targets


def get_feature_columns(df: pd.DataFrame, target_cols: list) -> list:
    """Get feature columns (excluding targets, timestamps, and metadata)."""
    exclude_patterns = [
        'video_timestamp',
        'window_idx',
        '_congestion',
        '_signaling',
        '_label_count',
        'responseId',
        'view_label',
        'ID_enter',
        'ID_exit'
    ]

    feature_cols = []
    for col in df.columns:
        # Skip if matches any exclude pattern
        if any(pattern in col for pattern in exclude_patterns):
            continue
        feature_cols.append(col)

    print(f"Found {len(feature_cols)} feature columns")
    return feature_cols


def create_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    target_cols: list,
    window_size: int = 6,
    stride: int = 1,
    min_valid_targets: int = 3
) -> tuple:
    """
    Create rolling window sequences from the dataset.

    Args:
        df: Input dataframe with temporal data
        feature_cols: List of feature column names
        target_cols: List of target column names
        window_size: Number of consecutive time windows to include in each sequence
        stride: Step size for sliding window (1 = no overlap, 2 = 50% overlap, etc.)
        min_valid_targets: Minimum number of valid (non-null) targets required

    Returns:
        X: Feature sequences (n_samples, window_size, n_features)
        y: Target values (n_samples, n_targets)
        metadata: DataFrame with sequence metadata
    """
    print(f"\nCreating sequences with window_size={window_size}, stride={stride}...")

    sequences_X = []
    sequences_y = []
    metadata_list = []

    # Group by date to avoid creating sequences across different days
    df['date'] = df['video_timestamp'].dt.date

    for date, date_df in df.groupby('date'):
        date_df = date_df.sort_values('video_timestamp').reset_index(drop=True)

        # Create sequences for this date
        for i in range(0, len(date_df) - window_size + 1, stride):
            window_df = date_df.iloc[i:i+window_size]

            # Check temporal continuity (timestamps should be roughly consecutive)
            time_diffs = window_df['video_timestamp'].diff().iloc[1:].dt.total_seconds()
            max_gap = time_diffs.max() if len(time_diffs) > 0 else 0

            # Skip if there's a large gap (> 10 minutes) indicating missing data
            if max_gap > 600:  # 10 minutes
                continue

            # Extract features (sequence of windows)
            X_window = window_df[feature_cols].values  # Shape: (window_size, n_features)

            # Use the LAST window's targets as the prediction target
            y_targets = window_df.iloc[-1][target_cols]

            # Count valid (non-null) targets
            valid_count = y_targets.notna().sum()

            # Skip if not enough valid targets
            if valid_count < min_valid_targets:
                continue

            # Store sequence
            sequences_X.append(X_window)
            sequences_y.append(y_targets.values)

            # Store metadata
            metadata_list.append({
                'sequence_id': len(sequences_X) - 1,
                'date': date,
                'start_timestamp': window_df.iloc[0]['video_timestamp'],
                'end_timestamp': window_df.iloc[-1]['video_timestamp'],
                'n_windows': window_size,
                'valid_targets': valid_count,
                'max_time_gap_sec': max_gap
            })

    # Convert to numpy arrays
    X = np.array(sequences_X)  # Shape: (n_samples, window_size, n_features)
    y = np.array(sequences_y)  # Shape: (n_samples, n_targets)
    metadata_df = pd.DataFrame(metadata_list)

    print(f"\nCreated {len(X)} sequences")
    print(f"X shape: {X.shape} (samples, windows, features)")
    print(f"y shape: {y.shape} (samples, targets)")
    print(f"Average valid targets per sequence: {metadata_df['valid_targets'].mean():.1f}")

    return X, y, metadata_df


def encode_categorical_targets(y: np.ndarray, target_cols: list) -> tuple:
    """
    Encode categorical congestion levels to integers.

    Congestion levels (ordinal encoding):
    - free flowing: 0
    - light delay: 1
    - moderate delay: 2
    - heavy delay: 3
    - NaN: -1 (to be handled separately)
    """
    print("\nEncoding categorical targets...")

    congestion_mapping = {
        'free flowing': 0,
        'light delay': 1,
        'moderate delay': 2,
        'heavy delay': 3,
        np.nan: -1,
        'nan': -1,
        None: -1
    }

    # Create copy to avoid modifying original
    y_encoded = y.copy()

    # Apply encoding
    for i in range(y_encoded.shape[1]):
        # Get unique values safely
        col_values = y_encoded[:, i]
        unique_str_vals = set()
        for val in col_values:
            if pd.notna(val):
                unique_str_vals.add(str(val))
        print(f"  {target_cols[i]}: {len(unique_str_vals)} unique values: {unique_str_vals}")

        # Vectorized encoding (handle both strings and NaN)
        for old_val, new_val in congestion_mapping.items():
            if pd.isna(old_val):
                mask = pd.isna(y_encoded[:, i])
            else:
                mask = y_encoded[:, i] == old_val
            y_encoded[mask, i] = new_val

    # Convert to integer type
    y_encoded = y_encoded.astype(np.float32)  # Use float to preserve -1 for missing

    print(f"Encoded y shape: {y_encoded.shape}")
    print(f"Value distribution:")
    for i, col in enumerate(target_cols):
        values, counts = np.unique(y_encoded[:, i], return_counts=True)
        print(f"  {col}:")
        for val, count in zip(values, counts):
            label = {0: 'free flowing', 1: 'light delay', 2: 'moderate delay',
                    3: 'heavy delay', -1: 'missing'}.get(int(val), str(val))
            print(f"    {label}: {count} ({count/len(y_encoded)*100:.1f}%)")

    return y_encoded, congestion_mapping


def temporal_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    metadata_df: pd.DataFrame,
    test_size: float = 0.2
) -> tuple:
    """
    Split data temporally (no shuffling - preserve time order).

    Args:
        X: Feature sequences
        y: Targets
        metadata_df: Metadata with timestamps
        test_size: Fraction of data for test set

    Returns:
        X_train, X_test, y_train, y_test, train_meta, test_meta
    """
    print(f"\nSplitting data temporally (test_size={test_size})...")

    # Sort by end timestamp to ensure temporal order
    sort_idx = metadata_df['end_timestamp'].argsort()
    X = X[sort_idx]
    y = y[sort_idx]
    metadata_df = metadata_df.iloc[sort_idx].reset_index(drop=True)

    # Calculate split point
    split_idx = int(len(X) * (1 - test_size))

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    train_meta = metadata_df.iloc[:split_idx].reset_index(drop=True)
    test_meta = metadata_df.iloc[split_idx:].reset_index(drop=True)

    print(f"Train: {len(X_train)} sequences ({train_meta['start_timestamp'].min()} to {train_meta['end_timestamp'].max()})")
    print(f"Test:  {len(X_test)} sequences ({test_meta['start_timestamp'].min()} to {test_meta['end_timestamp'].max()})")

    return X_train, X_test, y_train, y_test, train_meta, test_meta


def save_sequences(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    feature_cols: list,
    target_cols: list,
    output_dir: str,
    encoding_map: dict = None
):
    """Save sequences and metadata to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\nSaving sequences to {output_path}...")

    # Save numpy arrays
    np.save(output_path / 'X_train.npy', X_train)
    np.save(output_path / 'X_test.npy', X_test)
    np.save(output_path / 'y_train.npy', y_train)
    np.save(output_path / 'y_test.npy', y_test)

    # Save metadata
    train_meta.to_csv(output_path / 'train_metadata.csv', index=False)
    test_meta.to_csv(output_path / 'test_metadata.csv', index=False)

    # Save column names
    pd.DataFrame({'feature': feature_cols}).to_csv(output_path / 'feature_columns.csv', index=False)
    pd.DataFrame({'target': target_cols}).to_csv(output_path / 'target_columns.csv', index=False)

    # Save encoding map if provided
    if encoding_map:
        pd.DataFrame([
            {'congestion_level': k, 'encoded_value': v}
            for k, v in encoding_map.items() if not pd.isna(k)
        ]).to_csv(output_path / 'encoding_map.csv', index=False)

    # Save dataset info
    info = {
        'creation_date': datetime.now().isoformat(),
        'n_train_sequences': len(X_train),
        'n_test_sequences': len(X_test),
        'window_size': X_train.shape[1],
        'n_features': X_train.shape[2],
        'n_targets': y_train.shape[1],
        'train_date_range': f"{train_meta['start_timestamp'].min()} to {train_meta['end_timestamp'].max()}",
        'test_date_range': f"{test_meta['start_timestamp'].min()} to {test_meta['end_timestamp'].max()}"
    }
    pd.DataFrame([info]).to_csv(output_path / 'dataset_info.csv', index=False)

    print(f"✓ Saved all files to {output_path}")
    print(f"\nFiles created:")
    for f in output_path.glob('*'):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Create rolling window sequences for traffic prediction')
    parser.add_argument('--input', type=str, default='training_dataset.csv',
                       help='Input CSV file with training data')
    parser.add_argument('--output-dir', type=str, default='sequences',
                       help='Output directory for sequences')
    parser.add_argument('--window-size', type=int, default=6,
                       help='Number of consecutive windows per sequence (default: 6 = 30 minutes)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride for sliding window (default: 1 = no overlap)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data for test set (default: 0.2)')
    parser.add_argument('--min-valid-targets', type=int, default=3,
                       help='Minimum valid targets required per sequence (default: 3)')
    parser.add_argument('--no-encode', action='store_true',
                       help='Skip categorical encoding (keep original labels)')

    args = parser.parse_args()

    print("=" * 80)
    print("ROLLING WINDOW SEQUENCE CREATION")
    print("=" * 80)

    # Load data
    df = load_and_prepare_data(args.input)

    # Get columns
    target_cols = get_target_columns(df)
    feature_cols = get_feature_columns(df, target_cols)

    if len(target_cols) == 0:
        print("ERROR: No target columns found!")
        return

    if len(feature_cols) == 0:
        print("ERROR: No feature columns found!")
        return

    # Create sequences
    X, y, metadata = create_sequences(
        df=df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        window_size=args.window_size,
        stride=args.stride,
        min_valid_targets=args.min_valid_targets
    )

    if len(X) == 0:
        print("ERROR: No sequences created! Check your data and parameters.")
        return

    # Encode targets
    encoding_map = None
    if not args.no_encode:
        y, encoding_map = encode_categorical_targets(y, target_cols)

    # Temporal split
    X_train, X_test, y_train, y_test, train_meta, test_meta = temporal_train_test_split(
        X, y, metadata, test_size=args.test_size
    )

    # Save everything
    save_sequences(
        X_train, X_test, y_train, y_test,
        train_meta, test_meta,
        feature_cols, target_cols,
        args.output_dir,
        encoding_map
    )

    print("\n" + "=" * 80)
    print("✓ SUCCESS - Sequences created and saved!")
    print("=" * 80)
    print(f"\nTo load the sequences in your model:")
    print(f"""
import numpy as np
import pandas as pd

# Load training data
X_train = np.load('{args.output_dir}/X_train.npy')
y_train = np.load('{args.output_dir}/y_train.npy')

# Load test data
X_test = np.load('{args.output_dir}/X_test.npy')
y_test = np.load('{args.output_dir}/y_test.npy')

# Load metadata
train_meta = pd.read_csv('{args.output_dir}/train_metadata.csv')
test_meta = pd.read_csv('{args.output_dir}/test_metadata.csv')

print(f"X_train shape: {{X_train.shape}}")  # (samples, windows, features)
print(f"y_train shape: {{y_train.shape}}")  # (samples, targets)
""")


if __name__ == '__main__':
    main()
