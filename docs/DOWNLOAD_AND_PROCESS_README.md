# Download and Process - Quick Reference

Complete pipeline for downloading 4 synchronized camera videos from GCS and extracting ML-ready features.

## Overview

The `download_and_extract_features.py` script automatically:
1. 🔍 Finds 4 camera videos in GCS bucket by timestamp
2. ⬇️ Downloads them to local directory
3. 🎥 Runs object tracking with speed calculation
4. 📊 Extracts features per camera
5. 🔗 Merges features from all 4 cameras
6. 📈 Returns DataFrame ready for ML

## Quick Start

### Command Line Usage

```bash
# Process single video set
python download_and_extract_features.py --timestamp 2025-10-20-06-00-45

# Use full dataset (instead of small)
python download_and_extract_features.py --timestamp 2025-10-20-06-00-45 --dataset full

# Keep videos after processing (don't delete)
python download_and_extract_features.py --timestamp 2025-10-20-06-00-45 --keep-videos

# Use custom time window (10 minutes instead of 5)
python download_and_extract_features.py --timestamp 2025-10-20-06-00-45 --window 600

# Skip download if videos already exist
python download_and_extract_features.py --timestamp 2025-10-20-06-00-45 --skip-download
```

### Jupyter Notebook Usage

```python
from download_and_extract_features import process_video_set

# Process single video set
df = process_video_set(
    timestamp="2025-10-20-06-00-45",
    dataset="small",
    download_dir="videos",
    output_dir="features_output"
)

print(f"Shape: {df.shape}")
df.head()
```

### Batch Processing

```python
from download_and_extract_features import batch_process_timestamps
import pandas as pd

# Process multiple video sets
timestamps = [
    "2025-10-20-06-00-45",
    "2025-10-20-07-00-45",
    "2025-10-20-08-00-45",
]

results = batch_process_timestamps(
    timestamps=timestamps,
    dataset="small",
    skip_download=True  # Skip if already processed
)

# Combine all into single DataFrame
all_dfs = []
for timestamp, df in results.items():
    df['video_timestamp'] = timestamp
    all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)
print(f"Combined dataset: {combined.shape}")

# Save to CSV
combined.to_csv('complete_training_data.csv', index=False)
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--timestamp` | Video timestamp (YYYY-MM-DD-HH-MM-SS) | **Required** |
| `--dataset` | GCS bucket: 'small' or 'full' | `small` |
| `--download-dir` | Directory to download videos | `videos` |
| `--output-dir` | Output directory for features | `features_output` |
| `--zone-distances` | Path to zone distances JSON | `camera_configs/zone_distances_example.json` |
| `--window` | Time window in seconds | `300` (5 min) |
| `--skip-download` | Skip download if videos exist | `False` |
| `--keep-videos` | Keep videos after processing | `False` |

## Function Parameters

### `process_video_set()`

Main function for processing a single video set.

```python
def process_video_set(
    timestamp: str,                      # Video timestamp (required)
    dataset: str = 'small',              # 'small' or 'full'
    download_dir: str = 'videos',
    output_dir: str = 'features_output',
    zone_distances_json: Optional[str] = None,
    time_window_seconds: float = 300.0,  # 5 minutes
    skip_download: bool = False,
    keep_videos: bool = False
) -> Optional[pd.DataFrame]
```

**Returns**: DataFrame with merged features, or None if failed

### `batch_process_timestamps()`

Process multiple timestamps in batch.

```python
def batch_process_timestamps(
    timestamps: List[str],
    **kwargs  # Same parameters as process_video_set
) -> Dict[str, pd.DataFrame]
```

**Returns**: Dict mapping timestamp to DataFrame

## Pipeline Steps

```
Input: Timestamp "2025-10-20-06-00-45"
  ↓
Step 1: Find 4 videos in GCS
  • normanniles1_2025-10-20-06-00-45.mp4 (North)
  • normanniles2_2025-10-20-06-00-45.mp4 (East)
  • normanniles3_2025-10-20-06-00-45.mp4 (South)
  • normanniles4_2025-10-20-06-00-45.mp4 (West)
  ↓
Step 2: Download to local directory
  ↓
Step 3: Object tracking (step_2_object_tracking.py)
  • YOLO + ByteTrack detection
  • Zone validation
  • Speed calculation
  • Output: 4 tracking JSON files
  ↓
Step 4: Feature extraction (step_3_feature_extraction.py)
  • Entry/exit flows
  • Circulating metrics
  • Speed metrics
  • Temporal features
  • Lag & rolling features
  • Output: 4 feature CSV files
  ↓
Step 5: Merge features (step_4_multicamera_features.py)
  • Camera-specific features
  • Roundabout-wide features
  • Cross-camera features
  • Directional features
  • Output: 1 merged CSV file
  ↓
Step 6: Cleanup (optional)
  • Delete downloaded videos to save space
  • Keep tracking JSONs and feature CSVs
  ↓
Output: DataFrame with 100+ features × N time windows
```

## Output Files

After running, you'll have:

```
videos/
├── normanniles1_2025-10-20-06-00-45_tracking.json
├── normanniles1_2025-10-20-06-00-45_tracking_features.csv
├── normanniles2_2025-10-20-06-00-45_tracking.json
├── normanniles2_2025-10-20-06-00-45_tracking_features.csv
├── normanniles3_2025-10-20-06-00-45_tracking.json
├── normanniles3_2025-10-20-06-00-45_tracking_features.csv
├── normanniles4_2025-10-20-06-00-45_tracking.json
└── normanniles4_2025-10-20-06-00-45_tracking_features.csv

features_output/
└── merged_features_2025-10-20-06-00-45.csv  ← ML-ready features
```

## Features in Output

The merged CSV contains:

- **Temporal**: hour_of_day, day_of_week, is_weekend, is_rush_hour
- **Camera-specific** (×4 cameras): north_*, east_*, south_*, west_*
  - Entry/exit counts and flow rates
  - Vehicle types (car, truck, bus)
  - Circulating occupancy
  - Speed metrics
  - Lag features (t-1, t-2, t-3)
  - Rolling averages (15min, 30min, 60min)
- **Roundabout-wide**: total_*, avg_*
- **Cross-camera**: imbalance indices, balance ratios
- **Directional**: NS_*, EW_*
- **Targets** (8): {direction}_{entry|exit}_congestion

## Examples

### Example 1: Simple Processing

```python
# In Jupyter notebook
from download_and_extract_features import process_video_set

df = process_video_set("2025-10-20-06-00-45")
print(df.shape)
```

### Example 2: Process Morning Rush Hour

```python
# Process 6-9 AM (3 video sets)
timestamps = [
    "2025-10-20-06-00-45",
    "2025-10-20-07-00-45",
    "2025-10-20-08-00-45",
]

results = batch_process_timestamps(timestamps, dataset='small')
morning_data = pd.concat(results.values(), ignore_index=True)
```

### Example 3: Full Day Processing

```python
# Generate timestamps for full day
from datetime import datetime, timedelta

base_date = datetime(2025, 10, 20, 6, 0, 45)
timestamps = []

for hour in range(6, 18):  # 6 AM to 6 PM
    dt = base_date.replace(hour=hour)
    timestamps.append(dt.strftime("%Y-%m-%d-%H-%M-%S"))

# Process all
results = batch_process_timestamps(
    timestamps=timestamps,
    dataset='small',
    skip_download=True  # Assumes already downloaded
)

# Combine
full_day = pd.concat(results.values(), ignore_index=True)
full_day.to_csv('full_day_features.csv', index=False)
```

### Example 4: Reprocess Existing Videos

```python
# Videos already downloaded, just reprocess
df = process_video_set(
    timestamp="2025-10-20-06-00-45",
    skip_download=True,   # Don't download again
    keep_videos=True      # Keep videos
)
```

## Troubleshooting

### "gsutil not found"

**Problem**: Google Cloud SDK not installed

**Solution**:
```bash
# macOS
brew install --cask google-cloud-sdk

# Linux
curl https://sdk.cloud.google.com | bash

# Then authenticate
gcloud auth login
```

### "Could not find all 4 cameras"

**Problem**: Videos don't exist in bucket for this timestamp

**Solution**:
- Check timestamp format: YYYY-MM-DD-HH-MM-SS
- List available videos: `gsutil ls gs://brb-traffic/`
- Verify all 4 cameras exist for that timestamp

### "Tracking failed for camera X"

**Problem**: Zone configuration missing

**Solution**:
```bash
# Setup zones for missing camera
python step_1_setup_zones.py path/to/sample_video.mp4
```

### Memory Issues

**Problem**: Not enough RAM for processing

**Solution**:
- Process videos one at a time
- Use smaller time windows (--window 120 for 2-min)
- Delete videos after processing (default behavior)

## Best Practices

1. **Start with small dataset**: Use `dataset='small'` for testing
2. **Skip re-downloads**: Use `skip_download=True` to save bandwidth
3. **Batch processing**: Process multiple timestamps together for efficiency
4. **Clean up videos**: Default deletes videos after processing to save space
5. **Check outputs**: Use `analyze_features.py` to inspect results

## Integration with ML Pipeline

```python
from download_and_extract_features import batch_process_timestamps
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# 1. Collect training data
timestamps = ["2025-10-20-06-00-45", "2025-10-20-07-00-45", ...]
results = batch_process_timestamps(timestamps)

# 2. Combine and label
df = pd.concat(results.values(), ignore_index=True)
# Add your congestion labels here

# 3. Prepare features and targets
feature_cols = [col for col in df.columns if not col.endswith('_congestion')]
target_cols = [col for col in df.columns if col.endswith('_congestion')]

X = df[feature_cols]
y = df[target_cols]

# 4. Train model
model = MultiOutputClassifier(RandomForestClassifier())
model.fit(X, y)

# 5. Predict future congestion
# Given 15 minutes history → Predict 6 future windows
```

## See Also

- **[QUICK_START.md](QUICK_START.md)** - Overall pipeline guide
- **[MULTICAMERA_PIPELINE.md](MULTICAMERA_PIPELINE.md)** - Detailed documentation
- **[example_feature_extraction.ipynb](example_feature_extraction.ipynb)** - Jupyter notebook examples
- **[analyze_features.py](analyze_features.py)** - Feature analysis utility
