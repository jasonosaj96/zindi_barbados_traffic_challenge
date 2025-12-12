#!/usr/bin/env python3
"""
Diagnose Label Matching Issues

Checks why labels have too many None values compared to original Train.csv
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Load datasets
print("Loading data...")
train_df = pd.read_csv('dataset/Train.csv')
training_dataset = pd.read_csv('training_dataset.csv')

print(f"\nOriginal Train.csv shape: {train_df.shape}")
print(f"Training dataset shape: {training_dataset.shape}")

# Check NA percentages in original Train.csv
print("\n" + "="*80)
print("ORIGINAL TRAIN.CSV")
print("="*80)
for col in ['congestion_enter_rating', 'congestion_exit_rating']:
    total = len(train_df)
    na_count = train_df[col].isna().sum()
    print(f"{col}:")
    print(f"  NA: {na_count}/{total} ({na_count/total*100:.2f}%)")
    print(f"  Non-NA: {total-na_count}/{total} ({(total-na_count)/total*100:.2f}%)")

# Check NA percentages in processed training dataset
print("\n" + "="*80)
print("PROCESSED TRAINING DATASET")
print("="*80)
target_cols = [
    'north_entry_congestion', 
    'east_entry_congestion', 
    'south_entry_congestion', 
    'west_entry_congestion'
]

for col in target_cols:
    if col in training_dataset.columns:
        total = len(training_dataset)
        na_count = training_dataset[col].isna().sum()
        print(f"{col}:")
        print(f"  NA: {na_count}/{total} ({na_count/total*100:.2f}%)")
        print(f"  Non-NA: {total-na_count}/{total} ({(total-na_count)/total*100:.2f}%)")

# Check if label_count columns exist
print("\n" + "="*80)
print("LABEL MATCH COUNTS (from label_count columns)")
print("="*80)
for direction in ['north', 'east', 'south', 'west']:
    col = f'{direction}_label_count'
    if col in training_dataset.columns:
        avg_count = training_dataset[col].mean()
        zero_count = (training_dataset[col] == 0).sum()
        print(f"{direction}:")
        print(f"  Average labels per window: {avg_count:.2f}")
        print(f"  Windows with 0 labels: {zero_count}/{len(training_dataset)} ({zero_count/len(training_dataset)*100:.2f}%)")

# Examine a specific example
print("\n" + "="*80)
print("EXAMPLE DIAGNOSIS")
print("="*80)

if 'video_timestamp' in training_dataset.columns:
    # Get first timestamp
    first_timestamp = training_dataset['video_timestamp'].iloc[0]
    print(f"\nAnalyzing timestamp: {first_timestamp}")
    
    # Extract video timestamp from Train.csv videos column
    train_df['video_timestamp'] = train_df['videos'].str.extract(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})')
    
    # Parse datetime
    train_df['datetime_start'] = pd.to_datetime(train_df['datetimestamp_start'])
    
    # Filter Train.csv for this timestamp
    train_subset = train_df[train_df['video_timestamp'] == first_timestamp]
    
    print(f"\nTrain.csv records for this timestamp: {len(train_subset)}")
    print(f"Unique cameras: {train_subset['view_label'].unique()}")
    print(f"Time range: {train_subset['datetime_start'].min()} to {train_subset['datetime_start'].max()}")
    
    # Check one specific window
    feature_subset = training_dataset[training_dataset['video_timestamp'] == first_timestamp]
    print(f"\nFeature rows for this timestamp: {len(feature_subset)}")
    
    if len(feature_subset) > 0:
        first_row = feature_subset.iloc[0]
        print(f"\nFirst feature window:")
        print(f"  start_time: {first_row.get('start_time', 'N/A')} seconds")
        print(f"  end_time: {first_row.get('end_time', 'N/A')} seconds")
        
        # Calculate absolute times
        if 'start_time' in first_row:
            video_dt = datetime.strptime(first_timestamp, "%Y-%m-%d-%H-%M-%S")
            window_start = video_dt + timedelta(seconds=first_row['start_time'])
            window_end = video_dt + timedelta(seconds=first_row['end_time'])
            
            print(f"  Absolute window: {window_start} to {window_end}")
            
            # Find matching Train.csv records
            matching = train_subset[
                (train_subset['view_label'] == 'Norman Niles #1') &
                (train_subset['datetime_start'] >= window_start) &
                (train_subset['datetime_start'] < window_end)
            ]
            
            print(f"\n  Matching Train.csv records (Norman Niles #1): {len(matching)}")
            if len(matching) > 0:
                print(f"  Labels found: {matching['congestion_enter_rating'].tolist()}")
            
            print(f"\n  Feature labels:")
            print(f"    north_entry_congestion: {first_row.get('north_entry_congestion', 'N/A')}")
            print(f"    north_label_count: {first_row.get('north_label_count', 'N/A')}")

# Check camera mapping
print("\n" + "="*80)
print("CAMERA MAPPING CHECK")
print("="*80)
print("\nUnique cameras in Train.csv:")
print(train_df['view_label'].unique())

print("\nExpected mapping:")
print("  Norman Niles #1 → north")
print("  Norman Niles #2 → east")
print("  Norman Niles #3 → south")
print("  Norman Niles #4 → west")

# Count records per camera
print("\nRecords per camera in Train.csv:")
print(train_df['view_label'].value_counts())
