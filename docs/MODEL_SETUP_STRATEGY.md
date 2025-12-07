# Model Setup Strategy for Traffic Congestion Prediction

Complete guide for setting up the prediction model based on the competition requirements.

## Problem Statement

**Competition Requirement:**
- **Input**: 15 minutes of video data (t=0 to t=15)
- **Embargo**: 2-minute operational lag (t=15 to t=17) - simulates download/processing time
- **Output**: Predict congestion for minutes 18-23 (5 minutes ahead)
- **Constraint**: No backpropagation during inference (pre-trained model only)

## Timeline Structure

```
┌─────────────────┬───────────┬──────────────────┐
│   Input Data    │  Embargo  │  Prediction      │
│   (15 minutes)  │ (2 min)   │  (6 minutes)     │
├─────────────────┼───────────┼──────────────────┤
│   t=0 → t=15    │ t=15→t=17 │  t=18 → t=23     │
│                 │ NO DATA   │                  │
│   3 windows     │  SKIP     │  Predict this    │
│   (5min each)   │           │                  │
└─────────────────┴───────────┴──────────────────┘

Windows:
- Window 1: t=0-5 min    ]
- Window 2: t=5-10 min   ] → INPUT (use these)
- Window 3: t=10-15 min  ]
- [Embargo: t=15-17 min]   → SKIP (no data available)
- Window 4: t=15-20 min    → Partially in embargo
- Window 5: t=20-25 min    → TARGET (predict this, covers t=18-23)
```

## Prediction Task

**Model learns**: `f(features[0:15 min]) → congestion[18:23 min]`

**Total predictions per timestamp:**
- 4 cameras (North, East, South, West)
- 2 directions per camera (Entry, Exit)
- 1 time window (covering minutes 18-23)
- **Total: 8 predictions** (4 cameras × 2 directions)

## Recommended Approach

### Option 1: Direct Multi-Step Forecasting (Recommended)

Train a single multi-output model that learns the direct mapping from 15-minute features to future congestion.

#### Architecture

```python
Input: 15-minute aggregated features
  ↓
Multi-Output Model
  ↓
Output: 8 congestion predictions (4 cameras × 2 directions)
```

#### Implementation

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load dataset
df = pd.read_csv('training_dataset.csv')
df = df.sort_values(['video_timestamp', 'start_time']).reset_index(drop=True)

# 2. Create sequences with embargo
def create_sequences_with_embargo(df, input_steps=3, embargo_steps=1, output_steps=1):
    """
    Create input-output sequences respecting embargo period

    Args:
        input_steps: Number of 5-min windows for input (3 = 15 minutes)
        embargo_steps: Gap between input and output (1 = 5 minutes, includes 2-min embargo)
        output_steps: Number of 5-min windows to predict (1 = 5 minutes, covers t=18-23)

    Returns:
        X: Features from t=0 to t=15 (flattened from 3 windows)
        y: Labels for t=20-25 (covers target period t=18-23)
    """
    X_sequences = []
    y_sequences = []

    # Group by video_timestamp to ensure temporal continuity
    for timestamp, group in df.groupby('video_timestamp'):
        group = group.sort_values('start_time').reset_index(drop=True)

        # Need at least: input_steps + embargo_steps + output_steps windows
        min_windows = input_steps + embargo_steps + output_steps
        if len(group) < min_windows:
            continue

        # Create sequences with rolling window
        for i in range(len(group) - min_windows + 1):
            # Input: windows [i : i+input_steps]
            # Example: [0, 1, 2] = t=0-15 min
            input_windows = group.iloc[i:i+input_steps]

            # Output: window [i+input_steps+embargo_steps]
            # Example: [3+1] = [4] = t=20-25 min (covers t=18-23)
            output_idx = i + input_steps + embargo_steps
            output_window = group.iloc[output_idx]

            # Extract features (flatten 3 windows into 1 row)
            X_row = input_windows[feature_cols].values.flatten()
            X_sequences.append(X_row)

            # Extract labels (8 values: 4 cameras × 2 directions)
            y_row = output_window[target_cols].values
            y_sequences.append(y_row)

    return np.array(X_sequences), np.array(y_sequences)


# 3. Define feature and target columns
feature_cols = [c for c in df.columns if not any([
    c.endswith('_congestion'),
    c.endswith('_signaling'),
    c.endswith('_label_count'),
    c in ['video_timestamp', 'window_idx', 'start_time', 'end_time']
])]

target_cols = [
    'north_entry_congestion', 'north_exit_congestion',
    'east_entry_congestion', 'east_exit_congestion',
    'south_entry_congestion', 'south_exit_congestion',
    'west_entry_congestion', 'west_exit_congestion'
]

print(f"Feature columns: {len(feature_cols)}")
print(f"Target columns: {len(target_cols)}")

# 4. Create sequences
X, y = create_sequences_with_embargo(df, input_steps=3, embargo_steps=1, output_steps=1)

print(f"X shape: {X.shape}")  # (N, 3*~150) = (N, ~450) features
print(f"y shape: {y.shape}")  # (N, 8) predictions

# 5. Encode labels
label_encoders = {}
y_encoded = np.zeros_like(y, dtype=int)

for i, col in enumerate(target_cols):
    le = LabelEncoder()
    y_encoded[:, i] = le.fit_transform(y[:, i])
    label_encoders[col] = le

# 6. Temporal train/test split (NO SHUFFLING!)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]

print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# 7. Train multi-output model
model = MultiOutputClassifier(
    GradientBoostingClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
)

print("Training model...")
model.fit(X_train, y_train)

# 8. Evaluate
score = model.score(X_test, y_test)
print(f"\nOverall Accuracy: {score:.3f}")

# 9. Per-output evaluation
from sklearn.metrics import classification_report, confusion_matrix

predictions = model.predict(X_test)

for i, target in enumerate(target_cols):
    print(f"\n{'='*80}")
    print(f"Target: {target}")
    print('='*80)

    # Decode predictions
    y_true_decoded = label_encoders[target].inverse_transform(y_test[:, i])
    y_pred_decoded = label_encoders[target].inverse_transform(predictions[:, i])

    # Classification report
    print(classification_report(y_true_decoded, y_pred_decoded))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true_decoded, y_pred_decoded))
```

#### Feature Engineering for 15-Minute Input

```python
def create_15min_aggregated_features(df):
    """
    Create rich features from 3 consecutive 5-minute windows (15 minutes total)

    This gives the model a comprehensive view of traffic patterns
    """
    # For each 15-minute sequence
    aggregated_features = []

    for timestamp, group in df.groupby('video_timestamp'):
        group = group.sort_values('start_time')

        for i in range(len(group) - 2):  # Need 3 consecutive windows
            windows = group.iloc[i:i+3]

            features = {}

            # 1. Current state (last window = t=10-15)
            last_window = windows.iloc[-1]
            for col in feature_cols:
                features[f'current_{col}'] = last_window[col]

            # 2. 15-minute averages
            for col in feature_cols:
                features[f'avg_15min_{col}'] = windows[col].mean()

            # 3. 15-minute trends (std deviation)
            for col in feature_cols:
                features[f'std_15min_{col}'] = windows[col].std()

            # 4. Momentum (change from start to end)
            for col in feature_cols:
                features[f'momentum_{col}'] = (
                    windows.iloc[-1][col] - windows.iloc[0][col]
                )

            # 5. Acceleration (second derivative)
            for col in feature_cols:
                if len(windows) >= 3:
                    diff1 = windows.iloc[1][col] - windows.iloc[0][col]
                    diff2 = windows.iloc[2][col] - windows.iloc[1][col]
                    features[f'accel_{col}'] = diff2 - diff1

            # 6. Min/Max over 15 minutes
            for col in feature_cols:
                features[f'min_15min_{col}'] = windows[col].min()
                features[f'max_15min_{col}'] = windows[col].max()

            # 7. Temporal context (from last window)
            features['hour_of_day'] = last_window['hour_of_day']
            features['is_rush_hour'] = last_window['is_rush_hour']
            features['day_of_week'] = last_window['day_of_week']
            features['is_weekend'] = last_window['is_weekend']

            aggregated_features.append(features)

    return pd.DataFrame(aggregated_features)
```

### Option 2: LSTM Sequence-to-Sequence Model

For better temporal pattern learning.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Reshape for LSTM: (samples, timesteps, features)
X_lstm = X.reshape(X.shape[0], 3, len(feature_cols))  # (N, 3 timesteps, 150 features)

print(f"LSTM input shape: {X_lstm.shape}")

# 2. Split
X_train_lstm = X_lstm[:split_idx]
X_test_lstm = X_lstm[split_idx:]

# 3. Build LSTM model
model = Sequential([
    # First LSTM layer
    LSTM(128, return_sequences=True, input_shape=(3, len(feature_cols))),
    Dropout(0.3),
    BatchNormalization(),

    # Second LSTM layer
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    BatchNormalization(),

    # Dense layers
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),

    # Output layer (8 outputs, 4 classes each)
    Dense(8 * 4, activation='softmax')  # 8 outputs × 4 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

# 5. Train
history = model.fit(
    X_train_lstm, y_train,
    validation_data=(X_test_lstm, y_test),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 6. Evaluate
loss, accuracy = model.evaluate(X_test_lstm, y_test)
print(f"Test Accuracy: {accuracy:.3f}")
```

### Option 3: Per-Camera Models (Simpler, More Interpretable)

Train 4 separate models (one per camera), each predicting entry + exit.

```python
from sklearn.ensemble import RandomForestClassifier

models = {}

for direction in ['north', 'east', 'south', 'west']:
    print(f"\nTraining model for {direction} camera...")

    # Targets for this camera
    targets = [
        f'{direction}_entry_congestion',
        f'{direction}_exit_congestion'
    ]

    # Use all features from 15-minute input
    X = df[feature_cols]

    # Target: future window (skip embargo)
    # Shift by 4 windows = 20 minutes (15 input + 5 buffer with embargo)
    y = df[targets].shift(-4)

    # Remove NaN rows
    mask = y[targets[0]].notna()
    X_clean = X[mask]
    y_clean = y[mask]

    # Encode labels
    y_encoded = np.zeros((len(y_clean), 2), dtype=int)
    les = {}
    for i, target in enumerate(targets):
        le = LabelEncoder()
        y_encoded[:, i] = le.fit_transform(y_clean[target])
        les[target] = le

    # Train
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_clean, y_encoded)

    models[direction] = {
        'model': model,
        'encoders': les
    }

    # Evaluate
    score = model.score(X_clean[-len(X_clean)//5:], y_encoded[-len(y_encoded)//5:])
    print(f"  Accuracy: {score:.3f}")

# At inference
def predict_all_cameras(X_new):
    """Predict congestion for all cameras"""
    predictions = {}

    for direction, model_dict in models.items():
        model = model_dict['model']
        encoders = model_dict['encoders']

        # Predict
        y_pred_encoded = model.predict(X_new)

        # Decode
        for i, flow in enumerate(['entry', 'exit']):
            target = f'{direction}_{flow}_congestion'
            le = encoders[target]
            predictions[target] = le.inverse_transform(y_pred_encoded[:, i])

    return predictions
```

## Critical Implementation Details

### 1. Handling the Embargo Period

**MUST NOT** use any data from t=15 to t=17:

```python
# ❌ WRONG: Includes embargo period
X_input = df[df['end_time'] <= 17*60]  # Includes t=15-17

# ✅ CORRECT: Only use data before embargo
X_input = df[df['end_time'] <= 15*60]  # Only t=0-15
```

### 2. Window Alignment

Our features are in 5-minute windows:

| Window | Time Range | Usage |
|--------|------------|-------|
| 0 | t=0-5 min | INPUT ✓ |
| 1 | t=5-10 min | INPUT ✓ |
| 2 | t=10-15 min | INPUT ✓ |
| 3 | t=15-20 min | SKIP (embargo) |
| 4 | t=20-25 min | TARGET (covers t=18-23) |

**Target window is window 4** (t=20-25), which covers the required prediction period (t=18-23).

### 3. No Data Leakage

Ensure temporal integrity:

```python
# ✅ CORRECT: Temporal split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ❌ WRONG: Random shuffling breaks temporal order
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)  # NO!
```

### 4. Feature Selection for 15-Minute Input

Key features to include:

**Per-Camera Features:**
- Entry flow rate, count
- Exit flow rate, count
- Circulating occupancy, density
- Average speed
- Queue length
- Vehicle type distribution

**Roundabout-Wide Features:**
- Total entry/exit counts
- Average speed across all cameras
- Entry/exit flow imbalance
- NS vs EW balance

**Temporal Features:**
- Hour of day
- Day of week
- Is rush hour
- Is weekend

**Derived Features:**
- Lag features (t-1, t-2, t-3)
- Rolling averages (from past 15 min)
- Momentum (trend direction)
- Acceleration (trend change)

## Model Training Workflow

### Complete End-to-End Example

```python
# ========================================
# 1. LOAD AND PREPARE DATA
# ========================================
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('training_dataset.csv')
df = df.sort_values(['video_timestamp', 'start_time']).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Unique timestamps: {df['video_timestamp'].nunique()}")

# ========================================
# 2. DEFINE FEATURES AND TARGETS
# ========================================
feature_cols = [c for c in df.columns if not any([
    c.endswith('_congestion'),
    c.endswith('_signaling'),
    c.endswith('_label_count'),
    c in ['video_timestamp', 'window_idx', 'start_time', 'end_time']
])]

target_cols = [
    'north_entry_congestion', 'north_exit_congestion',
    'east_entry_congestion', 'east_exit_congestion',
    'south_entry_congestion', 'south_exit_congestion',
    'west_entry_congestion', 'west_exit_congestion'
]

print(f"\nFeature columns: {len(feature_cols)}")
print(f"Target columns: {len(target_cols)}")

# ========================================
# 3. CREATE SEQUENCES WITH EMBARGO
# ========================================
def create_sequences_with_embargo(df, input_steps=3, embargo_steps=1, output_steps=1):
    X_sequences = []
    y_sequences = []
    metadata = []

    for timestamp, group in df.groupby('video_timestamp'):
        group = group.sort_values('start_time').reset_index(drop=True)

        min_windows = input_steps + embargo_steps + output_steps
        if len(group) < min_windows:
            continue

        for i in range(len(group) - min_windows + 1):
            # Input windows
            input_windows = group.iloc[i:i+input_steps]

            # Output window (skip embargo)
            output_idx = i + input_steps + embargo_steps
            output_window = group.iloc[output_idx]

            # Extract features
            X_row = input_windows[feature_cols].values.flatten()
            y_row = output_window[target_cols].values

            X_sequences.append(X_row)
            y_sequences.append(y_row)
            metadata.append({
                'video_timestamp': timestamp,
                'input_start_time': input_windows.iloc[0]['start_time'],
                'input_end_time': input_windows.iloc[-1]['end_time'],
                'output_start_time': output_window['start_time'],
                'output_end_time': output_window['end_time']
            })

    return np.array(X_sequences), np.array(y_sequences), pd.DataFrame(metadata)

X, y, metadata = create_sequences_with_embargo(df)

print(f"\nSequences created:")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  Metadata shape: {metadata.shape}")

# ========================================
# 4. ENCODE LABELS
# ========================================
label_encoders = {}
y_encoded = np.zeros(y.shape, dtype=int)

for i, col in enumerate(target_cols):
    le = LabelEncoder()
    # Handle NaN values
    mask = pd.notna(y[:, i])
    y_encoded[mask, i] = le.fit_transform(y[mask, i])
    y_encoded[~mask, i] = -1  # Mark missing labels
    label_encoders[col] = le

print("\nLabel classes:")
for col in target_cols:
    classes = label_encoders[col].classes_
    print(f"  {col}: {classes}")

# ========================================
# 5. REMOVE ROWS WITH MISSING LABELS
# ========================================
# Remove sequences where any target is missing
valid_mask = (y_encoded >= 0).all(axis=1)
X_clean = X[valid_mask]
y_clean = y_encoded[valid_mask]
metadata_clean = metadata[valid_mask]

print(f"\nAfter removing missing labels:")
print(f"  X shape: {X_clean.shape}")
print(f"  y shape: {y_clean.shape}")

# ========================================
# 6. TEMPORAL TRAIN/TEST SPLIT
# ========================================
split_idx = int(len(X_clean) * 0.8)

X_train = X_clean[:split_idx]
X_test = X_clean[split_idx:]
y_train = y_clean[:split_idx]
y_test = y_clean[split_idx:]

print(f"\nTrain/test split:")
print(f"  Train samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")

# ========================================
# 7. TRAIN MODEL
# ========================================
print("\nTraining multi-output model...")

model = MultiOutputClassifier(
    GradientBoostingClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        verbose=1
    ),
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training complete!")

# ========================================
# 8. EVALUATE
# ========================================
print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

# Overall accuracy
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"\nOverall Accuracy:")
print(f"  Train: {train_score:.3f}")
print(f"  Test: {test_score:.3f}")

# Per-output evaluation
predictions = model.predict(X_test)

for i, target in enumerate(target_cols):
    print(f"\n{'='*80}")
    print(f"{target}")
    print('='*80)

    # Decode
    y_true = label_encoders[target].inverse_transform(y_test[:, i])
    y_pred = label_encoders[target].inverse_transform(predictions[:, i])

    # Metrics
    print(classification_report(y_true, y_pred))

# ========================================
# 9. SAVE MODEL
# ========================================
import joblib

model_data = {
    'model': model,
    'label_encoders': label_encoders,
    'feature_cols': feature_cols,
    'target_cols': target_cols,
    'input_steps': 3,
    'embargo_steps': 1
}

joblib.dump(model_data, 'traffic_congestion_model.pkl')
print("\n✓ Model saved to: traffic_congestion_model.pkl")
```

## Inference Pipeline

### Loading and Using the Trained Model

```python
import joblib
import pandas as pd
import numpy as np

# ========================================
# LOAD MODEL
# ========================================
model_data = joblib.load('traffic_congestion_model.pkl')

model = model_data['model']
label_encoders = model_data['label_encoders']
feature_cols = model_data['feature_cols']
target_cols = model_data['target_cols']

print("Model loaded successfully!")

# ========================================
# PREPARE NEW DATA (15 minutes of video)
# ========================================
# Assume you've processed 15 minutes of new video data
# through the pipeline and have 3 feature windows

new_df = pd.read_csv('new_video_features.csv')  # Your 3 windows (0-15 min)

# Ensure correct order
new_df = new_df.sort_values('start_time')

# Take first 3 windows
if len(new_df) < 3:
    raise ValueError("Need at least 3 windows (15 minutes) of data")

input_windows = new_df.iloc[:3]

# Extract and flatten features
X_new = input_windows[feature_cols].values.flatten().reshape(1, -1)

print(f"Input shape: {X_new.shape}")

# ========================================
# PREDICT
# ========================================
predictions_encoded = model.predict(X_new)[0]

# Decode predictions
predictions = {}
for i, target in enumerate(target_cols):
    le = label_encoders[target]
    pred_class = le.inverse_transform([predictions_encoded[i]])[0]
    predictions[target] = pred_class

# ========================================
# OUTPUT RESULTS
# ========================================
print("\nPredictions for minutes 18-23:")
print("="*60)

for camera in ['north', 'east', 'south', 'west']:
    entry_pred = predictions[f'{camera}_entry_congestion']
    exit_pred = predictions[f'{camera}_exit_congestion']

    print(f"{camera.capitalize():6s} - Entry: {entry_pred:20s} Exit: {exit_pred}")

# ========================================
# FORMAT FOR SUBMISSION
# ========================================
submission = pd.DataFrame([predictions])
submission.to_csv('submission.csv', index=False)
print("\n✓ Submission saved to: submission.csv")
```

## Best Practices

### 1. **Data Preparation**
- ✅ Sort by timestamp before splitting
- ✅ Use temporal split (no shuffling)
- ✅ Handle missing labels (drop or impute)
- ✅ Verify embargo period is respected

### 2. **Feature Engineering**
- ✅ Include current state features
- ✅ Add 15-minute aggregations (mean, std, min, max)
- ✅ Create momentum/trend features
- ✅ Include temporal context (hour, day, rush hour)

### 3. **Model Selection**
- ✅ Start with Gradient Boosting (good baseline)
- ✅ Try LSTM for temporal patterns
- ✅ Consider ensemble of both
- ✅ Use class weights for imbalanced data

### 4. **Validation**
- ✅ Use temporal cross-validation
- ✅ Evaluate per camera/direction separately
- ✅ Check confusion matrices for misclassification patterns
- ✅ Monitor class-specific F1 scores

### 5. **Production Deployment**
- ✅ Save model with label encoders
- ✅ Document feature order (critical!)
- ✅ Include feature extraction pipeline
- ✅ Add input validation
- ✅ Handle edge cases (missing data, corrupted video)

## Common Pitfalls to Avoid

### ❌ DON'T: Use data from embargo period
```python
# WRONG
X = df[df['end_time'] <= 17*60]  # Includes t=15-17
```

### ❌ DON'T: Shuffle temporal data
```python
# WRONG
X_train, X_test = train_test_split(X, y, shuffle=True)
```

### ❌ DON'T: Predict wrong time window
```python
# WRONG: Predicting t=15-20 instead of t=20-25
y = df[target_cols].shift(-3)  # Should be -4
```

### ❌ DON'T: Forget to encode labels
```python
# WRONG: String labels won't work with sklearn
model.fit(X_train, y_train)  # y_train has strings
```

### ✅ DO: Verify predictions align with requirements
```python
# CORRECT: Validate prediction window
assert output_window['start_time'] >= 18 * 60  # t >= 18 min
assert output_window['end_time'] <= 25 * 60    # t <= 25 min (covers up to 23)
```

## Expected Performance

Based on the problem structure:

- **Baseline (majority class)**: ~60-70% accuracy
- **Good model**: 75-85% accuracy
- **Excellent model**: 85-90% accuracy

Performance varies by:
- **Camera**: Some cameras may be easier to predict
- **Direction**: Entry vs exit may have different patterns
- **Congestion level**: "Free flowing" easier than "heavy delay"
- **Time of day**: Rush hour more variable

## Next Steps

1. **Run parallel processing** to create full training dataset
2. **Implement sequence creation** with embargo logic
3. **Train baseline model** (Gradient Boosting)
4. **Evaluate per camera** to identify weak spots
5. **Engineer better features** based on error analysis
6. **Try LSTM model** for temporal patterns
7. **Ensemble models** for final submission

## See Also

- [TRAINING_STRATEGY.md](TRAINING_STRATEGY.md) - General ML training strategy
- [PARALLEL_PROCESSING_README.md](PARALLEL_PROCESSING_README.md) - Dataset creation
- [TRAIN_INTEGRATION_README.md](TRAIN_INTEGRATION_README.md) - Label integration
