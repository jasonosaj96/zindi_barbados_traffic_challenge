# Training Strategy for Traffic Congestion Prediction

## Overview

This document explains the high-level strategy for training machine learning models on the extracted traffic features, with special emphasis on temporal data handling and the **critical importance of not shuffling**.

---

## Table of Contents

- [The Core Challenge](#the-core-challenge)
- [Why You Should NOT Shuffle](#why-you-should-not-shuffle)
- [Feature Logic Explained](#feature-logic-explained)
- [Train/Test Split Strategies](#traintest-split-strategies)
- [Training Workflow](#training-workflow)
- [Prediction Strategy](#prediction-strategy)
- [Key Recommendations](#key-recommendations)
- [Expected Challenges](#expected-challenges)

---

## The Core Challenge

### Data Structure

You have **temporal traffic data** where:
- Each row = one 5-minute time window
- Multiple consecutive windows from the same video set (15 minutes = 3 windows)
- Features include:
  - **Lag values**: `entry_count_lag_1`, `entry_count_lag_2`, `entry_count_lag_3`
  - **Rolling averages**: `rolling_mean_3` (15min), `rolling_mean_6` (30min), `rolling_mean_12` (60min)
  - **Current state**: Flow rates, occupancy, speeds
  - **Temporal context**: Hour of day, day of week, rush hour flag

### Prediction Task

**Input**: 15 minutes of historical video data (3 windows)
**Embargo**: 2 minutes processing time
**Output**: Predict congestion for 6 future time windows (minutes 18-23)

```
Timeline:
[t-2][t-1][t0]  [embargo]  [t+3][t+4][t+5][t+6][t+7][t+8]
├─ History ─┤   ├─2min─┤   ├────── Predictions (6 × 8) ──────┤
15 minutes                  18min 21min 24min 27min 30min 33min
```

**Total predictions**: 6 time windows × 8 outputs (4 cameras × 2 directions) = **48 predictions**

---

## Why You Should NOT Shuffle

### ⚠️ Critical Rule: DO NOT SHUFFLE YOUR DATA

This is a **time-series prediction problem**, not standard classification.

### Reason 1: Temporal Dependencies

Your data has explicit temporal dependencies:

```python
# Window at t=10 depends on previous windows
window_10 = {
    'entry_count': 45,                    # Current state
    'entry_count_lag_1': 42,              # Value at t=5 (depends on past)
    'entry_count_lag_2': 38,              # Value at t=0 (depends on past)
    'entry_flow_rate_rolling_mean_3': 41.7  # Average of t=0,5,10 (depends on past)
}
```

**If you shuffle**:
- Window at t=10 might be in training set
- Window at t=5 (which t=10 depends on) might be in test set
- Your model learns relationships that don't exist in deployment

### Reason 2: Data Leakage

**Example of leakage from shuffling**:

```
Original timeline:
[t=0: congestion=free] → [t=5: congestion=light] → [t=10: congestion=heavy]

After shuffling:
Training:   [t=10: congestion=heavy]
Test:       [t=5: congestion=light]

Problem: The model at t=10 has rolling_mean that includes t=5's value.
         If t=10 is in training, you're using "future" information to predict t=5!
```

### Reason 3: Real-World Deployment

In production:
- You only have **historical data** (past)
- You must predict the **future**
- Model should learn: `past → future`
- Shuffling teaches: `random time → random time` ❌

**Correct approach**: Train on early data, test on later data ✅

---

## Feature Logic Explained

Understanding your features is critical for proper time-series handling.

### 1. Window-Level Features (Independent)

**Calculated fresh for each window** - no temporal dependency:

| Feature | Description | Time Dependency |
|---------|-------------|-----------------|
| `north_entry_count` | Vehicle count entering THIS window | None - current |
| `total_circulating_occupancy` | Vehicles in roundabout NOW | None - current |
| `avg_speed_km_h` | Average speed RIGHT NOW | None - current |
| `hour_of_day` | Time of THIS window | None - current |

✅ **Safe to use**: Always available, no leakage risk

### 2. Lag Features (Backward-Looking)

**Explicitly look at PREVIOUS windows**:

| Feature | Description | Looks Back |
|---------|-------------|------------|
| `north_entry_count_lag_1` | Entry count 1 window ago | 5 minutes |
| `north_entry_count_lag_2` | Entry count 2 windows ago | 10 minutes |
| `north_entry_count_lag_3` | Entry count 3 windows ago | 15 minutes |

✅ **Safe to use**: Uses only historical data
⚠️ **Important**: First 1-3 windows will have NaN values

**Example**:
```
Window 0: lag_1=NaN, lag_2=NaN, lag_3=NaN (no history)
Window 1: lag_1=value[0], lag_2=NaN, lag_3=NaN
Window 2: lag_1=value[1], lag_2=value[0], lag_3=NaN
Window 3: lag_1=value[2], lag_2=value[1], lag_3=value[0] ✓ All lags available
```

### 3. Rolling Features (Backward Window Aggregation)

**Aggregate RECENT history**:

| Feature | Description | Window Size |
|---------|-------------|-------------|
| `entry_flow_rate_rolling_mean_3` | Average of last 3 windows | 15 minutes |
| `entry_flow_rate_rolling_mean_6` | Average of last 6 windows | 30 minutes |
| `entry_flow_rate_rolling_mean_12` | Average of last 12 windows | 60 minutes |

✅ **Safe to use**: Uses only past data
⚠️ **Important**: Early windows have fewer samples

**Example**:
```
Window 0: rolling_mean_3 = mean([value[0]]) = value[0]
Window 1: rolling_mean_3 = mean([value[0], value[1]])
Window 2: rolling_mean_3 = mean([value[0], value[1], value[2]])
Window 3: rolling_mean_3 = mean([value[1], value[2], value[3]]) ✓ Full window
```

### 4. Cross-Camera Features (Spatial Dependencies)

**Capture relationships between cameras AT THE SAME TIME**:

| Feature | Description | Spatial |
|---------|-------------|---------|
| `entry_flow_imbalance` | Std dev across 4 cameras | Current moment |
| `NS_entry_balance` | North vs South difference | Current moment |
| `EW_total_entry` | East + West combined flow | Current moment |

✅ **Safe to use**: Calculated from current window data across cameras

### 5. Target Labels

**What you're predicting**:

| Target | Description |
|--------|-------------|
| `north_entry_congestion` | Entry congestion at North entrance |
| `north_exit_congestion` | Exit congestion at North exit |
| (×4 cameras × 2 directions = 8 targets) | |

Classes: `["free flowing", "light delay", "moderate delay", "heavy delay"]`

---

## Train/Test Split Strategies

### Strategy 1: Temporal Split (Recommended)

**Concept**: Split data chronologically - train on early, test on later.

```
Timeline: ══════════════════════════════════════════════
         [    Training Data (60%)    ]│[Val 20%]│[Test 20%]
         Oct 20 ────────────────────→ Oct 24 ──→ Oct 26
```

**How to implement**:
1. Sort data by timestamp
2. Take first 60% for training
3. Next 20% for validation
4. Last 20% for test

**Advantages**:
- ✅ No data leakage
- ✅ Mimics real deployment (predict future from past)
- ✅ Tests on truly unseen future
- ✅ Simple to implement

**Disadvantages**:
- ❌ Test set may have different conditions (time of day, weather)
- ❌ Doesn't test generalization across different days

**When to use**:
- Default choice for time series
- When you have enough continuous data
- When you want to predict the immediate future

### Strategy 2: Video Set Grouping

**Concept**: Keep entire 15-minute video sets together, never split them.

```
Video Sets: [A][B][C][D][E][F][G][H][I][J]
            ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
Training:   [A][B][C][D][E][F]          (60%)
Validation:                   [G][H]    (20%)
Test:                             [I][J] (20%)

Each set = 15 minutes = 3 time windows
```

**How to implement**:
1. Group windows by `video_timestamp`
2. Sort groups chronologically
3. Split groups (not individual windows)

**Advantages**:
- ✅ Prevents leakage between consecutive windows
- ✅ More realistic (complete sessions)
- ✅ Respects temporal coherence within sessions
- ✅ Better for lag/rolling features

**Disadvantages**:
- ❌ Slightly less data per split (must keep groups intact)

**When to use**:
- When you have many distinct video sets
- When video sets come from different times/days
- When preserving session integrity matters

### Strategy 3: Time-Period Cross-Validation (Advanced)

**Concept**: Test generalization across different time periods.

```
Data spans: Oct 20-26 (7 days)

Fold 1: Train [Oct 21-26], Test [Oct 20]
Fold 2: Train [Oct 20, 22-26], Test [Oct 21]
Fold 3: Train [Oct 20-21, 23-26], Test [Oct 22]
...
Fold 7: Train [Oct 20-25], Test [Oct 26]
```

**Advantages**:
- ✅ Uses all data efficiently
- ✅ Tests across different conditions (morning/evening, weekday/weekend)
- ✅ More robust performance estimate

**Disadvantages**:
- ❌ More complex to implement
- ❌ 7× training time
- ❌ Still must respect temporal order within each fold

**When to use**:
- Limited data across multiple distinct periods
- Want to test generalization across different days/conditions
- Have computational resources for multiple training runs

### Strategy 4: Day-of-Week Stratification (Alternative)

**Concept**: Separate by day type (weekday vs weekend).

```
Weekdays (Oct 20-24):    [Train 80%][Test 20%]
Weekend (Oct 25-26):     [Train 80%][Test 20%]

Combine test sets for final evaluation
```

**When to use**:
- Traffic patterns differ significantly on weekends
- Want to ensure test set covers both conditions

---

## Training Workflow

### Phase 1: Data Preparation

#### Step 1: Load and Inspect

```python
# Conceptual steps:
1. Load labeled_features CSV
2. Check shape, dtypes, missing values
3. Examine label distribution
4. Check temporal coverage
```

#### Step 2: Handle Missing Values

**Lag features** (first few windows):
- **Option A**: Forward-fill from first available value
- **Option B**: Drop first 3 windows (lose lag history)
- **Option C**: Fill with 0 or mean (less ideal)

**Speed features** (if distances not provided):
- **Option A**: Drop rows without speeds
- **Option B**: Impute with median
- **Option C**: Create "speed_available" indicator feature

#### Step 3: Sort Chronologically

```python
# Critical step - ALWAYS sort before splitting
df.sort_values(['video_timestamp', 'window_idx'], inplace=True)
```

#### Step 4: Create Train/Val/Test Split

```python
# Using temporal split (Strategy 1)
n = len(df)
train_idx = int(0.6 * n)
val_idx = int(0.8 * n)

train = df[:train_idx]
val = df[train_idx:val_idx]
test = df[val_idx:]

# Verify no overlap in timestamps
assert train['video_timestamp'].max() < val['video_timestamp'].min()
assert val['video_timestamp'].max() < test['video_timestamp'].min()
```

### Phase 2: Feature Engineering

#### Separate Features from Targets

```python
# Define what NOT to use as features
exclude_cols = [
    'window_idx',           # Just an index
    'video_timestamp',      # Identifier (use hour_of_day instead)
    '*_congestion',         # Target labels
    '*_label_count',        # Metadata
    'responseId',           # Identifier from Train.csv
]

# Feature columns
feature_cols = [col for col in df.columns
                if not any(pattern in col for pattern in exclude_cols)]

# Target columns (8 outputs)
target_cols = [
    'north_entry_congestion', 'north_exit_congestion',
    'east_entry_congestion', 'east_exit_congestion',
    'south_entry_congestion', 'south_exit_congestion',
    'west_entry_congestion', 'west_exit_congestion'
]
```

#### Feature Importance Categories

**Most Important** (temporal patterns):
1. Lag features (`*_lag_1`, `*_lag_2`, `*_lag_3`)
2. Rolling averages (`*_rolling_mean_*`)
3. Hour of day / rush hour indicators

**Important** (traffic state):
4. Flow rates (`*_entry_flow_rate`, `*_exit_flow_rate`)
5. Occupancy (`*_circulating_occupancy_avg`)
6. Speed metrics (`*_speed_avg_km_h`)

**Useful** (spatial patterns):
7. Cross-camera features (`entry_flow_imbalance`, `NS_entry_balance`)
8. Directional features (`NS_total_entry`, `EW_total_entry`)

**Context** (environmental):
9. Temporal features (`hour_of_day`, `day_of_week`, `is_weekend`)
10. Signaling (`*_signaling`)

### Phase 3: Model Selection

#### Option A: Single Multi-Output Model (Simplest)

**Approach**: One model predicts all 8 outputs simultaneously.

```
Input: All features
Output: 8 predictions (4 cameras × 2 directions)
```

**Algorithms**:
- Random Forest with `MultiOutputClassifier`
- XGBoost/LightGBM with multi-output support
- Neural network with 8 output nodes

**Advantages**:
- ✅ Simple to implement
- ✅ Learns cross-camera dependencies
- ✅ One model to maintain

**Disadvantages**:
- ❌ Less flexibility per camera
- ❌ All cameras share same feature importance

#### Option B: Per-Camera Models (More Control)

**Approach**: Train 4 separate models (one per camera), each with 2 outputs.

```
North model: north_entry_congestion, north_exit_congestion
East model: east_entry_congestion, east_exit_congestion
South model: south_entry_congestion, south_exit_congestion
West model: west_entry_congestion, west_exit_congestion
```

**Advantages**:
- ✅ Can use camera-specific features
- ✅ Tune hyperparameters per camera
- ✅ Handle camera-specific patterns better

**Disadvantages**:
- ❌ 4× models to train and maintain
- ❌ Doesn't learn cross-camera dependencies as well

#### Option C: Per-Direction Models (Alternative)

**Approach**: Train 2 models - one for entry, one for exit.

```
Entry model: Predicts all 4 camera entries
Exit model: Predicts all 4 camera exits
```

**When to use**: If entry/exit have very different patterns

#### Option D: Ensemble (Advanced)

**Approach**: Combine multiple models for better predictions.

```
Ensemble = avg([RandomForest, XGBoost, LightGBM])
```

**When to use**: Squeezing out last few % accuracy

---

## Prediction Strategy

### The Task: Multi-Step Ahead Forecasting

You need to predict **6 future time windows** (18-33 minutes ahead).

```
Given:    [t-2][t-1][t0]  (15 min history)
Predict:  [t+3][t+4][t+5][t+6][t+7][t+8]  (6 windows, 3 min each)

For each window, predict 8 values:
- 4 cameras × 2 directions = 8 congestion labels
```

### Approach A: Direct Multi-Horizon (Recommended)

**Concept**: Train model to directly predict all 6 future windows.

```
Input:  Features from t-2, t-1, t0
Output: 48 predictions (6 windows × 8 cameras/directions)
```

**Advantages**:
- ✅ One prediction pass
- ✅ No error accumulation
- ✅ Simpler to implement

**Disadvantages**:
- ❌ 48 outputs (large model)
- ❌ Assumes independence between future windows

**Implementation**:
```python
# Create 48 output columns:
targets = [
    f'{direction}_{type}_congestion_t{t}'
    for direction in ['north', 'east', 'south', 'west']
    for type in ['entry', 'exit']
    for t in [3, 4, 5, 6, 7, 8]
]
```

### Approach B: Iterative Forecasting

**Concept**: Predict t+1, use it to predict t+2, etc.

```
Step 1: Predict t+1 using [t-2, t-1, t0]
Step 2: Predict t+2 using [t-1, t0, t+1_predicted]
Step 3: Predict t+3 using [t0, t+1_predicted, t+2_predicted]
...
```

**Advantages**:
- ✅ More flexible
- ✅ Can adapt to predictions
- ✅ Models temporal evolution

**Disadvantages**:
- ❌ Error compounds over steps
- ❌ More complex implementation
- ❌ Slower at inference

### Approach C: Separate Model Per Horizon

**Concept**: Train 6 separate models for t+3, t+4, ..., t+8.

**When to use**:
- Different horizons have different difficulty
- Want to optimize each horizon separately

---

## Key Recommendations

### 1. ❌ DO NOT SHUFFLE

**Never use**:
```python
# WRONG - breaks temporal dependencies
train_test_split(X, y, shuffle=True)  ❌
KFold(n_splits=5, shuffle=True)  ❌
```

**Always use**:
```python
# CORRECT - respects time order
data.sort_values('timestamp')
train_test_split(X, y, shuffle=False)  ✅
TimeSeriesSplit(n_splits=5)  ✅
```

### 2. ✅ Preserve Window Order

Keep consecutive windows together:
- Same video set should stay in same split
- Lag features depend on previous windows
- Rolling features depend on recent history

### 3. ⚠️ Handle Missing Lags Carefully

First few windows lack full history:

```python
# Option 1: Forward-fill
df['entry_count_lag_1'].fillna(method='ffill', inplace=True)

# Option 2: Drop incomplete windows
df = df[df['window_idx'] >= 3]  # Keep only windows with full lag history

# Option 3: Fill with neutral value
df['entry_count_lag_1'].fillna(0, inplace=True)
```

### 4. ⏰ Validate Temporally

Test set must be chronologically AFTER training:

```python
# Verify temporal split
print(f"Train: {train['video_timestamp'].min()} to {train['video_timestamp'].max()}")
print(f"Test: {test['video_timestamp'].min()} to {test['video_timestamp'].max()}")

# Should be: train.max() < test.min()
```

### 5. 📊 Class Imbalance

Traffic is usually "free flowing":

```python
# Check distribution
df['north_entry_congestion'].value_counts(normalize=True)
# Output:
# free flowing      0.70
# light delay       0.20
# moderate delay    0.08
# heavy delay       0.02
```

**Solutions**:
- Use class weights: `class_weight='balanced'`
- Oversample minority classes (SMOTE)
- Focus on precision/recall for delay classes
- Use F1-score instead of accuracy

### 6. 📈 Feature Engineering Priorities

**Most impactful features**:

1. **Lag features** - Recent history is strongest predictor
2. **Hour of day** - Rush hour vs off-peak patterns
3. **Rolling averages** - Smooth out noise, capture trends
4. **Flow rates** - Current traffic intensity
5. **Occupancy** - Congestion proxy
6. **Cross-camera patterns** - System-wide state

**Less critical** (but still useful):
- Speed metrics (if available)
- Directional imbalance
- Day of week

### 7. 🎯 Evaluation Metrics

**Per-camera metrics**:
```python
# Some cameras may be easier to predict
for camera in ['north', 'east', 'south', 'west']:
    accuracy_entry = accuracy_score(y_test[f'{camera}_entry_congestion'], y_pred)
    print(f"{camera}: {accuracy_entry:.3f}")
```

**Per-time-horizon** (if using direct multi-horizon):
```python
# Is t+3 easier than t+8?
for t in [3, 4, 5, 6, 7, 8]:
    accuracy = accuracy_score(y_test[f'congestion_t{t}'], y_pred[f't{t}'])
    print(f"t+{t}: {accuracy:.3f}")
```

**Per-congestion-class**:
```python
# Confusion matrix shows if you're just predicting "free flowing" always
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred,
      target_names=['free flowing', 'light delay', 'moderate delay', 'heavy delay']))
```

---

## Expected Challenges

### Challenge 1: Class Imbalance

**Problem**: 70%+ of windows are "free flowing".

**Symptoms**:
- Model achieves 70% accuracy by always predicting "free flowing"
- Poor recall for "moderate delay" and "heavy delay"
- Confusion matrix shows all predictions in one class

**Solutions**:
```python
# Class weights
model = RandomForestClassifier(class_weight='balanced')

# Or manually compute weights
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced',
                               classes=np.unique(y_train),
                               y=y_train)

# Oversample minority classes (use with caution - respect time order!)
from imblearn.over_sampling import SMOTE
# Only apply to training set, preserve temporal order
```

### Challenge 2: Temporal Autocorrelation

**Problem**: Consecutive windows are very similar.

**Symptoms**:
- Model achieves high accuracy on validation
- But accuracy is actually just "predict same as last window"
- Fails when traffic conditions change

**Solutions**:
- Test on different time periods (different days/hours)
- Add "change detection" features (delta from previous window)
- Ensure test set has variety of transitions
- Use metrics that penalize "lazy" predictions

### Challenge 3: Rush Hour vs Off-Peak

**Problem**: Different patterns at different times.

**Symptoms**:
- Model works well on morning rush hour
- Fails on midday or evening data
- High variance across test periods

**Solutions**:
```python
# Strategy 1: Hour-based features (already included)
df['is_rush_hour']  # Model learns different patterns

# Strategy 2: Separate models
morning_model = train_on(df[df['hour_of_day'].between(6, 9)])
evening_model = train_on(df[df['hour_of_day'].between(16, 19)])

# Strategy 3: Time-based features
df['time_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['time_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
```

### Challenge 4: Camera Differences

**Problem**: Some entrances are busier than others.

**Symptoms**:
- North camera: 85% accuracy
- East camera: 60% accuracy
- Model struggles with low-traffic cameras

**Solutions**:
```python
# Strategy 1: Per-camera models (different complexity)
north_model = RandomForestClassifier(n_estimators=100)  # More complex
east_model = RandomForestClassifier(n_estimators=50)    # Simpler

# Strategy 2: Camera-specific feature importance
# Focus on local features for low-traffic cameras
# Focus on cross-camera features for high-traffic cameras

# Strategy 3: Ensemble with camera-specific weights
predictions = 0.4*north_model + 0.3*east_model + ...
```

### Challenge 5: Missing Labels

**Problem**: Not all time windows have labels in Train.csv.

**Symptoms**:
- Dataset has NaN values in target columns
- Training fails or performance suffers

**Solutions**:
```python
# Strategy 1: Drop rows without labels
df_clean = df.dropna(subset=target_cols)

# Strategy 2: Semi-supervised learning
# Train on labeled data, predict on unlabeled, retrain

# Strategy 3: Per-camera filtering
# Keep rows where at least one camera has labels
df_partial = df[df[target_cols].notna().any(axis=1)]
```

### Challenge 6: Limited Training Data

**Problem**: Only a few days of labeled data.

**Solutions**:
- Use simpler models (less overfitting)
- Regularization (L1/L2, max_depth limits)
- Cross-validation to use all data efficiently
- Feature selection (fewer features = less overfitting)
- Transfer learning (if you have unlabeled videos)

---

## Complete Training Checklist

### Before Training

- [ ] Sort data chronologically
- [ ] Check for missing values (lags, speeds, labels)
- [ ] Verify temporal coverage (continuous or gaps?)
- [ ] Examine label distribution (class balance)
- [ ] Verify no data leakage (train dates < test dates)
- [ ] Group video sets together (don't split 15-min sessions)

### During Training

- [ ] Use temporal split (no shuffling!)
- [ ] Handle class imbalance (weights or sampling)
- [ ] Monitor per-camera performance
- [ ] Track feature importance
- [ ] Validate on held-out time period
- [ ] Check for "lazy" predictions (same as last window)

### After Training

- [ ] Evaluate on test set (chronologically later data)
- [ ] Confusion matrices per camera
- [ ] Per-class metrics (precision/recall for each congestion level)
- [ ] Check predictions make sense (no wild transitions)
- [ ] Validate temporal consistency (smooth predictions over time)
- [ ] Error analysis: Which windows/cameras are hardest?

---

## Summary

### The Golden Rules

1. **NEVER SHUFFLE** - This is time series, order matters
2. **TEMPORAL SPLIT** - Train on past, test on future
3. **KEEP SESSIONS TOGETHER** - Don't split 15-minute video sets
4. **LAG FEATURES ARE KEY** - They encode recent history
5. **VALIDATE FORWARD** - Test must be chronologically after training

### Quick Decision Tree

```
Q: Should I shuffle my data?
A: NO ❌

Q: How should I split train/test?
A: Chronologically - first 60% train, last 20% test ✅

Q: What if I have multiple separate days?
A: Keep video sets together, split by date ranges ✅

Q: Can I use KFold cross-validation?
A: Only TimeSeriesSplit, never shuffled KFold ⚠️

Q: What features are most important?
A: Lag features > rolling averages > flow rates > hour of day ✅

Q: How do I handle class imbalance?
A: Class weights or SMOTE (carefully, preserving time order) ✅
```

### Expected Performance

**Realistic targets** (based on task difficulty):

- **Easy cameras** (high traffic, stable patterns): 75-85% accuracy
- **Hard cameras** (low traffic, variable patterns): 60-70% accuracy
- **Overall system**: 70-80% accuracy across all 8 outputs
- **Per-class recall**:
  - Free flowing: 85%+ (majority class)
  - Light delay: 60-70%
  - Moderate delay: 40-60%
  - Heavy delay: 20-40% (rare class, hardest)

**Signs of good model**:
- Better than "predict most common class" baseline
- Reasonable predictions across all congestion levels
- Smooth transitions over time (no wild jumps)
- Works across different time periods (morning/evening)

**Signs of problems**:
- Always predicts "free flowing" (not learning)
- Perfect accuracy on validation (data leakage!)
- Random-looking predictions (poor features or shuffled data)
- Works on validation, fails on test (overfitting to time period)

---

## References

For implementation details, see:
- [TRAIN_INTEGRATION_README.md](TRAIN_INTEGRATION_README.md) - Label integration
- [MULTICAMERA_PIPELINE.md](MULTICAMERA_PIPELINE.md) - Feature descriptions
- [example_feature_extraction.ipynb](example_feature_extraction.ipynb) - Code examples

Good luck with your model training! 🚦📊
