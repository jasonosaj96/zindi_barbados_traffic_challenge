# Parallel Dataset Processing Guide

Process the entire training dataset efficiently using parallel workers.

## Quick Start

### Command Line - Process Entire Dataset

```bash
# Use 4 workers (recommended for most systems)
python parallel_process_dataset.py --workers 4

# Use all available CPUs (auto-detect)
python parallel_process_dataset.py --auto-workers

# Test with first 10 timestamps
python parallel_process_dataset.py --workers 12 --limit 10
```

### Jupyter Notebook

```python
from parallel_process_dataset import parallel_process_all

# Process entire dataset
df = parallel_process_all(
    workers=4,
    output_combined='training_dataset.csv'
)

print(f"Dataset shape: {df.shape}")
```

## Features

### 1. Parallel Processing
- **Multiple workers** process timestamps simultaneously
- **Configurable worker count** (default: 4)
- **Auto-detect CPUs** with `--auto-workers`

### 2. Progress Tracking
- **tqdm progress bar** shows real-time progress
- **Success/failure counts** updated live
- **Detailed error reporting** for failed timestamps

### 3. Resume Capability
- **Skip already processed** timestamps automatically
- **Useful for interruptions** or adding more data
- **No reprocessing** of completed work

### 4. Memory Efficient
- **Batch combining** loads results in chunks
- **Handles large datasets** without memory issues
- **Configurable batch size**

### 5. Flexible Filtering
- **Date range filtering** (start-date, end-date)
- **Limit processing** (test with subset)
- **Full dataset support** from Train.csv

## Command Line Options

```bash
python parallel_process_dataset.py [OPTIONS]
```

### Basic Options

| Option | Description | Default |
|--------|-------------|---------|
| `--workers N` | Number of parallel workers | 4 |
| `--auto-workers` | Use CPU count - 1 workers | - |
| `--limit N` | Process only first N timestamps | All |
| `--output FILE` | Output CSV path | training_dataset.csv |
| `--output-dir DIR` | Features output directory | features_output |

### Filtering Options

| Option | Description | Example |
|--------|-------------|---------|
| `--start-date DATE` | Start date (YYYY-MM-DD) | 2025-10-20 |
| `--end-date DATE` | End date (YYYY-MM-DD) | 2025-10-21 |
| `--limit N` | Max timestamps to process | 20 |

### Processing Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset small\|full` | GCS bucket to use | small |
| `--skip-download` | Skip download if videos exist | False |
| `--keep-videos` | Keep videos after processing | False |
| `--label-aggregation mode\|max\|last` | Label aggregation method | mode |
| `--resume` | Skip already processed timestamps | False |

## Usage Examples

### Example 1: Test Run (10 timestamps)

```bash
python parallel_process_dataset.py --workers 4 --limit 10
```

**Output:**
```
Processing 10 timestamps with 4 workers...
Processing: 100%|████████████████| 10/10 [05:23<00:00, ✓: 10, ✗: 0]

PROCESSING SUMMARY
Total timestamps: 10
Successful: 10
Failed: 0

Combined dataset: (180, 142)
Saved to: training_dataset.csv

✓ SUCCESS
```

### Example 2: Process Specific Date

```bash
python parallel_process_dataset.py \
    --workers 4 \
    --start-date 2025-10-20 \
    --end-date 2025-10-20
```

Processes only videos from October 20, 2025.

### Example 3: Resume Interrupted Processing

```bash
# First run (interrupted after 50 timestamps)
python parallel_process_dataset.py --workers 4

# Resume without reprocessing
python parallel_process_dataset.py --workers 4 --resume
```

**Output:**
```
Resume mode: 50 already processed, 150 remaining
Skipped: 50
Processing 150 timestamps with 4 workers...
```

### Example 4: Full Dataset with All CPUs

```bash
python parallel_process_dataset.py --auto-workers --resume
```

Uses `CPU count - 1` workers to maximize throughput while keeping system responsive.

### Example 5: Use Full GCS Bucket

```bash
python parallel_process_dataset.py \
    --workers 8 \
    --dataset full \
    --skip-download
```

Processes from the full dataset bucket with 8 workers.

## Jupyter Notebook Examples

### Example 1: Basic Usage

```python
from parallel_process_dataset import parallel_process_all

# Process first 20 timestamps
df = parallel_process_all(
    workers=4,
    limit=20,
    output_combined='training_data_subset.csv'
)

print(f"Shape: {df.shape}")
print(f"Timestamps: {df['video_timestamp'].nunique()}")
```

### Example 2: Date Range

```python
# Process morning rush hour (6-9 AM) for one day
df = parallel_process_all(
    workers=4,
    start_date='2025-10-20',
    end_date='2025-10-20',
    output_combined='morning_rush_dataset.csv'
)

# Filter by hour
df['hour'] = pd.to_datetime(df['video_timestamp'], format='%Y-%m-%d-%H-%M-%S').dt.hour
morning_df = df[(df['hour'] >= 6) & (df['hour'] <= 9)]
```

### Example 3: Resume Processing

```python
# First run (test with 10)
df_test = parallel_process_all(workers=4, limit=10)

# If successful, process rest
df_full = parallel_process_all(
    workers=4,
    resume=True,  # Skip the 10 already processed
    output_combined='full_training_dataset.csv'
)
```

### Example 4: Custom Processing Settings

```python
df = parallel_process_all(
    train_csv='dataset/Train.csv',
    output_dir='features_output',
    output_combined='training_dataset.csv',
    workers=8,
    limit=100,
    resume=False,
    dataset='small',
    skip_download=True,  # Videos already downloaded
    keep_videos=False,   # Delete after processing
    label_aggregation='max'  # Use worst congestion
)
```

## How It Works

### 1. Discover Timestamps

Reads `dataset/Train.csv` and extracts unique video timestamps from the `videos` column:

```python
# Train.csv videos column format:
'normanniles1/normanniles1_2025-10-20-06-00-45.mp4'
                          ^^^^^^^^^^^^^^^^^^^^
                          Extracted timestamp
```

### 2. Filter and Sort

- Apply date range filters if specified
- Apply limit if specified
- Sort chronologically
- Check for already processed (if resume mode)

### 3. Parallel Processing

Spawns `N` worker processes that each:
1. Download 4 synchronized videos from GCS
2. Run object tracking with ByteTrack
3. Extract features per camera
4. Merge multi-camera features
5. Add training labels from Train.csv
6. Save as `labeled_features_{timestamp}.csv`

Workers run independently on different timestamps.

### 4. Combine Results

After all processing:
1. Load individual CSVs in batches (memory-efficient)
2. Concatenate into single DataFrame
3. Add `video_timestamp` column
4. Save as `training_dataset.csv`
5. Print statistics (label coverage, class distribution)

## Performance Tips

### Worker Count Selection

| System | Recommended Workers | Rationale |
|--------|---------------------|-----------|
| 4 CPU cores | 2-3 workers | Leave 1-2 cores for system |
| 8 CPU cores | 4-6 workers | Balance parallelism and overhead |
| 16+ CPU cores | 8-12 workers | Diminishing returns beyond this |

**Note**: Video processing is I/O-bound (download, disk writes), so more workers ≠ proportionally faster.

### Download Optimization

```bash
# Skip download if videos already exist
python parallel_process_dataset.py --workers 4 --skip-download
```

Speeds up processing significantly if you've already downloaded videos.

### Memory Management

The script processes in batches to avoid loading all results into memory:

```python
# Default batch size: 50 timestamps at a time
processor.combine_results(timestamps, batch_size=50)
```

For systems with limited RAM, reduce batch size in code or process in smaller chunks with `--limit`.

## Troubleshooting

### Error: "No timestamps found"

**Cause**: Train.csv not found or empty

**Fix**:
```bash
# Check Train.csv exists
ls -lh dataset/Train.csv

# Verify it's not empty
wc -l dataset/Train.csv
```

### Error: Worker crashes

**Cause**: Out of memory or download issues

**Fix**:
```bash
# Reduce workers
python parallel_process_dataset.py --workers 2

# Or process in smaller batches
python parallel_process_dataset.py --workers 4 --limit 50
python parallel_process_dataset.py --workers 4 --limit 50 --resume
```

### Some timestamps fail

**Expected behavior** - some videos may be missing from GCS bucket.

Check failed timestamps:
```
PROCESSING SUMMARY
Failed: 5
  2025-10-20-12-00-45: FileNotFoundError: Video not found in GCS
  2025-10-20-13-15-45: ValueError: Tracking failed
  ...
```

Use `--resume` to skip successful ones on retry.

### Low label coverage

**Cause**: Timestamps in GCS but not in Train.csv (or vice versa)

**Solution**: Only process timestamps that exist in both:
```python
# This is done automatically - the script filters to timestamps in Train.csv
```

## Output Files

### Individual Feature Files

```
features_output/
├── labeled_features_2025-10-20-06-00-45.csv
├── labeled_features_2025-10-20-06-15-45.csv
├── labeled_features_2025-10-20-06-30-45.csv
└── ...
```

Each file contains:
- Multi-camera features (~100+ columns)
- Congestion labels (8 columns: 4 cameras × 2 directions)
- Temporal features (hour, day, rush_hour, etc.)
- Metadata (window times, label counts)

### Combined Dataset

```
training_dataset.csv
```

All individual files concatenated with added `video_timestamp` column.

**Use this for ML training**.

## Next Steps

After processing:

### 1. Explore Dataset

```python
import pandas as pd

df = pd.read_csv('training_dataset.csv')

print(f"Shape: {df.shape}")
print(f"Timestamps: {df['video_timestamp'].nunique()}")
print(f"Columns: {df.columns.tolist()}")

# Check label coverage
label_cols = [c for c in df.columns if c.endswith('_congestion')]
for col in label_cols:
    coverage = df[col].notna().mean() * 100
    print(f"{col}: {coverage:.1f}% coverage")
```

### 2. Train Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Prepare features and targets
feature_cols = [c for c in df.columns if not any([
    c.endswith('_congestion'),
    c.endswith('_signaling'),
    c in ['video_timestamp', 'window_idx']
])]

target_cols = [c for c in df.columns if c.endswith('_congestion')]

X = df[feature_cols]
y = df[target_cols]

# Temporal split (no shuffling!)
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train
model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Accuracy: {score:.3f}")
```

### 3. Feature Importance

```python
import matplotlib.pyplot as plt

# Get feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(10, 6))
plt.title("Top 20 Most Important Features")
plt.barh(range(20), importances[indices])
plt.yticks(range(20), [feature_cols[i] for i in indices])
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
```

## See Also

- [TRAINING_STRATEGY.md](TRAINING_STRATEGY.md) - ML training best practices
- [TRAIN_INTEGRATION_README.md](TRAIN_INTEGRATION_README.md) - Label integration details
- [QUICK_START.md](QUICK_START.md) - Overall pipeline overview
