# Traffic Prediction with Advanced Features Notebook

## Overview

This notebook (`traffic_prediction_with_advanced_features.ipynb`) combines the **submission file preparation structure** from the original starter notebook with **advanced object tracking and motion features** from the parallel processing pipeline.

## Key Features

### 1. Maintains Original Submission Structure
- ✅ Proper time-series shifting (7 segments = 2min embargo + 5min prediction)
- ✅ Correct ID formatting for entry/exit predictions
- ✅ Compatible with Zindi submission format
- ✅ Validates submission file before saving

### 2. Uses Advanced Features from Object Tracking
The notebook integrates rich features extracted from YOLO + ByteTrack object detection:

#### **Flow Metrics**
- Entry/exit vehicle counts per time window
- Flow rates (vehicles per minute)
- Vehicle type breakdown (car, truck, bus, motorcycle)

#### **Speed Analysis**
- Average vehicle speeds across zones
- Speed statistics (min, max, median, std)
- Zone-specific speeds (entry vs circulating)

#### **Occupancy Measurements**
- Circulating zone occupancy levels
- Occupancy sampling at multiple time points
- Density indicators

#### **Origin-Destination Patterns**
- Traffic flow matrices between zones
- Journey pattern analysis
- OD pair counts

#### **Multi-Camera Fusion**
- Combined features from all 4 synchronized cameras
- Cross-camera traffic patterns
- Comprehensive roundabout coverage

#### **Temporal Features**
- Hour of day, day of week
- Weekend/weekday indicators
- Rush hour detection
- Time since midnight

#### **Derived Features**
- Entry/exit balance ratios
- Entry/circulating flow ratios
- Occupancy to flow ratios
- Density indicators

## Prerequisites

### 1. Feature Extraction Pipeline

Before using this notebook, you must run **TWO** separate feature extraction pipelines - one for training data and one for test data:

#### A. Training Features
```bash
# Process all training videos (recommended: use parallel processing)
python parallel_process_dataset.py --workers 4

# Or process a subset for testing
python parallel_process_dataset.py --workers 4 --limit 20

# Or process specific date range
python parallel_process_dataset.py --workers 4 \
    --start-date 2025-10-20 --end-date 2025-10-21
```

This will create a `features_output/` directory containing files like:
```
features_output/
  ├── merged_features_2025-10-20-06-00-45.csv
  ├── merged_features_2025-10-20-06-01-45.csv
  └── ...
```

#### B. Test Features
```bash
# Process all test videos
python parallel_process_test_dataset.py --workers 4

# Or process a subset for testing
python parallel_process_test_dataset.py --workers 4 --limit 20

# Or process specific date range
python parallel_process_test_dataset.py --workers 4 \
    --start-date 2025-10-20 --end-date 2025-10-21
```

This will create a `test_features_output/` directory containing files like:
```
test_features_output/
  ├── test_features_2025-10-20-07-52-45.csv
  ├── test_features_2025-10-20-07-53-45.csv
  └── ...
```

**Important**: The test features are stored in a **separate directory** with a **different filename pattern** than training features!

### 2. Required Data Files

Place these files in the same directory as the notebook:
- `Train.csv` - Training labels
- `TestInputSegments.csv` - Test video segments
- `SampleSubmission.csv` - Submission template

## Usage

### Step 1: Run Feature Extraction (One-time Setup)

**You need to run BOTH scripts:**

```bash
# TRAINING features
python parallel_process_dataset.py --workers 8 --dataset small

# TEST features
python parallel_process_test_dataset.py --workers 8 --dataset small

# Quick test with subset (both)
python parallel_process_dataset.py --workers 4 --limit 50
python parallel_process_test_dataset.py --workers 4 --limit 20
```

**Note**: This step can take several hours for the full dataset. Use `--resume` flag to continue interrupted processing.

### Step 2: Run the Notebook

Open `traffic_prediction_with_advanced_features.ipynb` in Jupyter:

```bash
jupyter notebook traffic_prediction_with_advanced_features.ipynb
```

Or run in JupyterLab, VS Code, or Kaggle.

### Step 3: Configure Paths

In the notebook's configuration cell, set your paths:

```python
TRAIN_CSV = "Train.csv"
TEST_CSV = "TestInputSegments.csv"
SAMPLE_SUBMISSION_CSV = "SampleSubmission.csv"

# Two separate feature directories!
TRAIN_FEATURES_DIR = "features_output"       # Training features
TEST_FEATURES_DIR = "test_features_output"   # Test features
```

### Step 4: Run All Cells

Execute all cells in order. The notebook will:
1. Load training labels and test segments
2. Load pre-computed advanced features
3. Merge features with labels
4. Add temporal features
5. Apply proper time-shifting (7 segments)
6. Train Random Forest model
7. Evaluate on validation set
8. Generate predictions
9. Save submission file

### Step 5: Submit to Zindi

Upload the generated `submission_advanced_features.csv` to the competition.

## Notebook Structure

### Data Loading & Preparation
1. **Load CSV files**: Train, test, submission template
2. **Extract timestamps**: Parse video filenames
3. **Load advanced features**: From feature extraction pipeline
4. **Merge datasets**: Combine features with labels

### Feature Engineering
1. **Temporal features**: Hour, day, rush hour indicators
2. **Categorical encoding**: Signaling categories
3. **Multi-camera fusion**: Features from all 4 cameras

### Submission Format Preparation
1. **Long format conversion**: Melt entry/exit into single column
2. **Add test_output_5 rows**: From submission template
3. **Sort and organize**: By view, ID type, time segment

### Time-Series Modeling
1. **Apply 7-segment shift**: Implements 2min embargo + 5min prediction
2. **Split datasets**: Train, validation (test_input_15), test (test_output_5)
3. **Handle NaN values**: Drop first 7 segments per sequence

### Model Training & Evaluation
1. **Train Random Forest**: Class-weighted to handle imbalance
2. **Validation metrics**: Classification report, confusion matrix, F1 scores
3. **Feature importance**: Identify key predictive features

### Prediction & Submission
1. **Generate predictions**: For test_output_5 segments
2. **Format submission**: Proper ID matching
3. **Validate format**: Check against sample submission
4. **Save to CSV**: Ready for Zindi upload

## Feature Availability Modes

The notebook adapts based on feature availability:

### Mode 1: With Advanced Features (Recommended)
When `features_output/` directory exists with merged features:
- Uses all advanced tracking features
- Best performance expected
- Rich feature set (50+ features)

### Mode 2: Fallback to Base Features
If advanced features are not available:
- Falls back to basic features only
- Temporal features (hour, day, signaling)
- Reduced performance but still functional

## Expected Performance

With advanced features, you should see:
- **Validation F1 (Macro)**: 0.65 - 0.75 (estimated)
- **Key predictive features**:
  - Entry/exit flow rates
  - Circulating occupancy
  - Speed metrics
  - Hour of day
  - Multi-camera patterns

## Customization Options

### 1. Model Hyperparameters

In the training cell, adjust:
```python
rf = RandomForestClassifier(
    n_estimators=300,      # Increase for better performance (slower)
    max_depth=None,        # Control tree depth
    min_samples_split=5,   # Minimum samples to split
    class_weight="balanced" # Handle class imbalance
)
```

### 2. Feature Selection

Modify the feature selection cell to:
- Include/exclude specific feature groups
- Add custom derived features
- Filter by feature importance

### 3. Time Window Shift

Change the shift amount in the time-shift cell:
```python
# Current: 7 segments (2min embargo + 5min prediction)
shifted_df = merged_df.groupby(['view_label', 'ID_type'])[features_cols].shift(7)

# Try different shifts for experimentation
# shifted_df = merged_df.groupby(['view_label', 'ID_type'])[features_cols].shift(5)
```

### 4. Model Selection

Replace Random Forest with other models:
```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# XGBoost
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=10  # Handle imbalance
)

# LightGBM
model = LGBMClassifier(
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.05,
    class_weight='balanced'
)
```

## Troubleshooting

### Issue: "Features directory not found"

**Solution**: Run BOTH feature extraction pipelines:
```bash
# Training features
python parallel_process_dataset.py --workers 4

# Test features
python parallel_process_test_dataset.py --workers 4
```

### Issue: "No feature files found"

**Solution**: Check that feature files exist in both directories:
```bash
# Check train features
ls features_output/merged_features_*.csv

# Check test features
ls test_features_output/test_features_*.csv
```

If either is empty, re-run the corresponding feature extraction script.

### Issue: "Merge quality: 100% missing features"

**Cause**: Timestamp mismatch between labels and features.

**Solution**:
1. Check video filename format in Train.csv
2. Ensure feature extraction used same timestamp format
3. Verify timestamp extraction regex in notebook

### Issue: "Submission shape mismatch"

**Cause**: Missing or extra predictions.

**Solution**:
1. Verify all test_output_5 rows are present
2. Check that submission template was loaded correctly
3. Ensure no duplicate IDs in submission

### Issue: Low validation F1 score

**Solutions**:
1. Check feature quality (many NaN values?)
2. Increase n_estimators in Random Forest
3. Try different models (XGBoost, LightGBM)
4. Add lag features (looking back more segments)
5. Tune class_weight parameter for imbalance

## Next Steps for Improvement

### 1. Ensemble Methods
Combine multiple models:
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier([
    ('rf', RandomForestClassifier(...)),
    ('xgb', XGBClassifier(...)),
    ('lgbm', LGBMClassifier(...))
], voting='soft')
```

### 2. Add Lag Features
Look back further in time:
```python
# Create lag features for 1, 2, 3, 5, 10 minutes back
for lag in [1, 2, 3, 5, 10]:
    for col in key_features:
        merged_df[f'{col}_lag_{lag}'] = merged_df.groupby(['view_label', 'ID_type'])[col].shift(lag)
```

### 3. Rolling Window Features
Aggregate over time windows:
```python
# Rolling 5-minute averages
for col in flow_features:
    merged_df[f'{col}_rolling_5min'] = merged_df.groupby(['view_label', 'ID_type'])[col].rolling(5).mean()
```

### 4. Camera Interaction Features
Create cross-camera patterns:
```python
# Entry-exit matching across cameras
merged_df['cam1_cam3_entry_diff'] = merged_df['cam1_entry_count'] - merged_df['cam3_entry_count']
merged_df['cam2_cam4_flow_ratio'] = merged_df['cam2_exit_flow'] / (merged_df['cam4_exit_flow'] + 1)
```

### 5. Deep Learning
Try neural networks with sequence modeling:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# LSTM for time-series prediction
model = Sequential([
    LSTM(64, input_shape=(15, n_features)),  # 15 segments
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax')
])
```

## File Structure

```
.
├── traffic_prediction_with_advanced_features.ipynb  # This notebook
├── parallel_process_dataset.py                      # TRAIN feature extraction
├── parallel_process_test_dataset.py                 # TEST feature extraction
├── download_and_extract_features.py                 # Single video processing
├── step_2_object_tracking.py                        # YOLO + ByteTrack
├── step_3_feature_extraction.py                     # Feature computation
├── step_4_multicamera_features.py                   # Multi-camera fusion
├── Train.csv                                        # Training labels
├── TestInputSegments.csv                            # Test segments
├── SampleSubmission.csv                             # Submission template
├── features_output/                                 # TRAIN features
│   ├── merged_features_2025-10-20-06-00-45.csv
│   └── ...
├── test_features_output/                            # TEST features
│   ├── test_features_2025-10-20-07-52-45.csv
│   └── ...
└── submission_advanced_features.csv                 # Output submission
```

## Performance Tips

1. **Use parallel processing**: Run feature extraction with `--workers 8` on multi-core machines
2. **Cache features**: Features are computed once and reused
3. **Resume interrupted runs**: Use `--resume` flag to skip already processed videos
4. **Monitor memory**: For large datasets, process in batches
5. **GPU acceleration**: YOLO tracking uses GPU if available (significant speedup)

## References

- Original starter notebook: `what-causes-traffic-congestion-in-barbadoes.ipynb`
- Feature extraction: `parallel_process_dataset.py`
- Competition page: https://zindi.africa/competitions/barbados-traffic-analysis-challenge

## Support

For issues or questions:
1. Check this README
2. Review error messages in notebook cells
3. Verify feature extraction completed successfully
4. Check that all required files are present

Good luck with the competition! 🚦🚗
