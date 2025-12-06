# Training Data Integration - Complete Guide

How to integrate congestion labels from `dataset/Train.csv` with extracted features for ML model training.

## Overview

The **Train.csv** dataset contains:
- **16,076 records** (per-camera, per-minute video segments)
- **Congestion labels**: `congestion_enter_rating` and `congestion_exit_rating`
- **4 cameras**: Norman Niles #1-4 (North, East, South, West)
- **Additional features**: `signaling` (traffic light state), timestamps

Our extracted features are in **5-minute time windows**, so we need to:
1. Match 1-minute labeled videos to 5-minute feature windows
2. Aggregate multiple labels per window (mode, max, or last)
3. Create training-ready dataset

## Quick Start

### Option 1: Complete Pipeline (Recommended)

Use `process_with_labels.py` for end-to-end processing:

```bash
# Download videos + extract features + add labels (one command!)
python process_with_labels.py --timestamp 2025-10-20-06-00-45
```

### Option 2: Add Labels to Existing Features

If you already have extracted features:

```bash
python integrate_training_labels.py \
    --features features_output/merged_features_2025-10-20-06-00-45.csv \
    --train dataset/Train.csv \
    --output labeled_features.csv
```

## File Structure

```
dataset/
└── Train.csv          ← Training labels (provided)

features_output/
├── merged_features_2025-10-20-06-00-45.csv    ← From step 4
└── labeled_features_2025-10-20-06-00-45.csv   ← Output (with labels)

training_dataset.csv   ← Final combined dataset for ML
```

## Train.csv Structure

| Column | Description | Example |
|--------|-------------|---------|
| `view_label` | Camera name | "Norman Niles #1" |
| `videos` | Video filename | "normanniles1/normanniles1_2025-10-20-06-00-45.mp4" |
| `video_time` | Video timestamp | "2025-10-20 06:00:45" |
| `datetimestamp_start` | Segment start | "2025-10-20 06:00:45" |
| `datetimestamp_end` | Segment end | "2025-10-20 06:01:44" |
| `congestion_enter_rating` | Entry congestion | "free flowing" |
| `congestion_exit_rating` | Exit congestion | "light delay" |
| `signaling` | Traffic light state | "none", "low", "high" |
| `time_segment_id` | Segment ID | 0-4976 |

### Congestion Classes

4 classes (ordered by severity):
1. **free flowing** - No congestion
2. **light delay** - Minor slowdown
3. **moderate delay** - Moderate congestion
4. **heavy delay** - Severe congestion

### Camera Mapping

| view_label | Camera ID | Direction | Our Features Prefix |
|------------|-----------|-----------|---------------------|
| Norman Niles #1 | 1 | North | `north_*` |
| Norman Niles #2 | 2 | East | `east_*` |
| Norman Niles #3 | 3 | South | `south_*` |
| Norman Niles #4 | 4 | West | `west_*` |

## Label Aggregation Methods

Since 5 min windows contain ~5 one-minute labeled videos, we aggregate:

### Mode (Default)
Most frequent label in the window.
```python
--label-aggregation mode
```
**Use when**: Want majority opinion, robust to outliers

### Max
Worst congestion level in the window.
```python
--label-aggregation max
```
**Use when**: Conservative (predict worst-case scenario)

### Last
Last label chronologically in the window.
```python
--label-aggregation last
```
**Use when**: Want most recent state

## Complete Examples

### Example 1: Single Video Set with Labels

```python
from process_with_labels import process_video_set_with_labels

# Process one 15-minute video set
df = process_video_set_with_labels(
    timestamp="2025-10-20-06-00-45",
    train_csv="dataset/Train.csv",
    label_aggregation='mode'
)

# Check labels
print(df[['north_entry_congestion', 'north_exit_congestion']].value_counts())
```

Output columns added:
- `north_entry_congestion`, `north_exit_congestion`
- `east_entry_congestion`, `east_exit_congestion`
- `south_entry_congestion`, `south_exit_congestion`
- `west_entry_congestion`, `west_exit_congestion`
- `north_signaling`, `east_signaling`, etc.

### Example 2: Batch Processing for Training Dataset

```python
from process_with_labels import batch_process_with_labels

# Process morning rush hour (6-9 AM)
timestamps = [
    "2025-10-20-06-00-45",
    "2025-10-20-06-15-45",
    "2025-10-20-06-30-45",
    "2025-10-20-06-45-45",
    "2025-10-20-07-00-45",
    # ... more timestamps
]

df = batch_process_with_labels(
    timestamps=timestamps,
    output_combined='training_dataset.csv'
)

print(f"Training data: {df.shape}")
```

### Example 3: Command Line Batch Mode

```bash
# Process multiple timestamps
python process_with_labels.py --batch \
    --timestamps \
        2025-10-20-06-00-45 \
        2025-10-20-07-00-45 \
        2025-10-20-08-00-45 \
    --output training_dataset.csv \
    --label-aggregation max
```

## Label Coverage Analysis

Not all time windows will have labels (if videos missing from Train.csv):

```python
import pandas as pd

df = pd.read_csv('labeled_features_2025-10-20-06-00-45.csv')

# Check coverage per camera
directions = ['north', 'east', 'south', 'west']
for direction in directions:
    entry_col = f'{direction}_entry_congestion'
    exit_col = f'{direction}_exit_congestion'

    entry_coverage = df[entry_col].notna().sum() / len(df) * 100
    exit_coverage = df[exit_col].notna().sum() / len(df) * 100

    print(f"{direction.capitalize():6s} - Entry: {entry_coverage:.1f}%, Exit: {exit_coverage:.1f}%")
```

## Feature + Label Structure

After integration, your DataFrame has:

**Original Features** (~100+ columns):
- Temporal: `hour_of_day`, `is_rush_hour`, etc.
- Camera-specific: `north_entry_flow_rate`, `east_circulating_occupancy_avg`, etc.
- Roundabout-wide: `total_entry_count`, `avg_speed_km_h`, etc.
- Cross-camera: `entry_flow_imbalance`, `NS_entry_balance`, etc.

**Added Labels** (8 columns):
- `north_entry_congestion`, `north_exit_congestion`
- `east_entry_congestion`, `east_exit_congestion`
- `south_entry_congestion`, `south_exit_congestion`
- `west_entry_congestion`, `west_exit_congestion`

**Added Metadata**:
- `north_signaling`, `east_signaling`, etc. (traffic light state)
- `north_label_count`, `east_label_count`, etc. (# labels in window)

## Machine Learning Workflow

### 1. Collect Training Data

```python
from process_with_labels import batch_process_with_labels

# Get all timestamps from Train.csv
import pandas as pd
train = pd.read_csv('dataset/Train.csv')
train['video_time'] = pd.to_datetime(train['video_time'])

# Extract unique timestamps (rounded to nearest 15-min for efficiency)
timestamps = train['video_time'].dt.floor('15min').unique()
timestamps = [t.strftime("%Y-%m-%d-%H-%M-%S") for t in timestamps]

print(f"Processing {len(timestamps)} timestamps...")

# Process all
df = batch_process_with_labels(timestamps[:10])  # Start with subset
```

### 2. Prepare Features and Targets

```python
# Load labeled data
df = pd.read_csv('training_dataset.csv')

# Drop rows with no labels
df = df.dropna(subset=['north_entry_congestion'])  # Adjust as needed

# Feature columns (exclude labels and metadata)
feature_cols = [col for col in df.columns if not any([
    col.endswith('_congestion'),
    col.endswith('_signaling'),
    col.endswith('_label_count'),
    col in ['window_idx', 'video_timestamp']
])]

# Target columns
target_cols = [col for col in df.columns if col.endswith('_congestion')]

X = df[feature_cols]
y = df[target_cols]

print(f"Features: {X.shape}")
print(f"Targets: {y.shape}")
```

### 3. Train Multi-Output Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode labels
y_encoded = y.copy()
for col in y.columns:
    le = LabelEncoder()
    y_encoded[col] = le.fit_transform(y[col].astype(str))

# Split data (preserve temporal order)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, shuffle=False
)

# Train model (8 outputs: 4 cameras × 2 directions)
model = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
)

model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Accuracy: {score:.3f}")
```

### 4. Per-Camera Evaluation

```python
from sklearn.metrics import classification_report

# Predict
y_pred = model.predict(X_test)

# Evaluate each camera-direction
for i, target_col in enumerate(y.columns):
    print(f"\n{target_col}:")
    print(classification_report(
        y_test.iloc[:, i],
        y_pred[:, i],
        target_names=['free flowing', 'light delay', 'moderate delay', 'heavy delay']
    ))
```

## Useful Scripts

| Script | Purpose |
|--------|---------|
| `process_with_labels.py` | Complete pipeline: download + features + labels |
| `integrate_training_labels.py` | Add labels to existing features |
| `download_and_extract_features.py` | Download + features only |
| `analyze_features.py` | Analyze feature distributions |

## Troubleshooting

### "No labels matched"

**Problem**: Time windows don't align with Train.csv records

**Solution**:
- Check that timestamps exist in Train.csv
- Verify camera mapping is correct
- Try different aggregation method

### "Label coverage is low"

**Problem**: Missing videos in Train.csv for some time periods

**Solution**:
- Filter to only timestamps with good coverage
- Use `df.dropna(subset=target_cols)` to remove unlabeled rows
- Check Train.csv for available date range

### "Imbalanced classes"

**Problem**: Most labels are "free flowing"

**Solution**:
```python
# Check class distribution
for col in target_cols:
    print(f"\n{col}:")
    print(df[col].value_counts(normalize=True))

# Use class weighting in model
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
```

## Best Practices

1. **Start small**: Process 10-20 timestamps first to verify pipeline
2. **Check coverage**: Ensure good label coverage before training
3. **Use mode aggregation**: Most robust for multi-label windows
4. **Validate timestamps**: Ensure they exist in Train.csv
5. **Handle missing labels**: Drop or impute missing values
6. **Class imbalance**: Use class weights or sampling techniques

## Next Steps

1. **Process all available timestamps** from Train.csv
2. **Explore label distribution** across cameras and time
3. **Feature selection**: Identify most important features
4. **Hyperparameter tuning**: Optimize model performance
5. **Cross-validation**: Use time-based splits
6. **Deploy**: Create prediction pipeline for new videos

## See Also

- [QUICK_START.md](QUICK_START.md) - Overall pipeline
- [MULTICAMERA_PIPELINE.md](MULTICAMERA_PIPELINE.md) - Feature details
- [DOWNLOAD_AND_PROCESS_README.md](DOWNLOAD_AND_PROCESS_README.md) - Download pipeline
