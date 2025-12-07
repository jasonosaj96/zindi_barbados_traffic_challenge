# Multi-Camera Traffic Analysis Pipeline

Complete pipeline for processing 4 synchronized roundabout cameras (North, East, South, West) and extracting ML-ready features for congestion prediction.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│ Input: 4 Synchronized Videos (N, E, S, W)              │
│ - 15 minutes of footage per camera                      │
│ - 1 camera per entrance/exit                            │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Step 1: Zone Setup (one-time per camera)                │
│ - Define entry/exit/circulating zones                   │
│ - Output: camera_configs/camera_{1-4}_zones.json        │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: Object Tracking (per camera)                    │
│ - YOLO + ByteTrack vehicle detection                    │
│ - Zone-based journey validation                         │
│ - Speed calculation (if distances provided)             │
│ - Output: *_tracking.json (4 files)                     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: Feature Extraction (per camera)                 │
│ - Entry/exit flow metrics                               │
│ - Circulating metrics                                   │
│ - Speed metrics                                          │
│ - Temporal features                                      │
│ - Lag & rolling features                                │
│ - Output: *_features.csv (4 files)                      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Step 4: Multi-Camera Feature Merging                    │
│ - Camera-specific features (north_, east_, etc.)        │
│ - Roundabout-wide features (total_, avg_)               │
│ - Cross-camera features (imbalance, balance)            │
│ - Directional features (NS_, EW_)                       │
│ - Output: merged_features.csv (1 file)                  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ Output: ML-Ready Features                               │
│ - Rows: Time windows (e.g., 5-min intervals)            │
│ - Columns: 100+ features                                │
│ - Targets: 8 predictions (4 cameras × 2 directions)     │
│   • north_entry_congestion, north_exit_congestion       │
│   • east_entry_congestion, east_exit_congestion         │
│   • south_entry_congestion, south_exit_congestion       │
│   • west_entry_congestion, west_exit_congestion         │
└─────────────────────────────────────────────────────────┘
```

## Camera Mapping

| Camera ID | Direction | Zone Names |
|-----------|-----------|------------|
| 1 | North (N) | enter, exit, circulating_left, circulating_right |
| 2 | East (E) | enter, exit, circulating_left, circulating_right |
| 3 | South (S) | enter, exit, circulating_left, circulating_right |
| 4 | West (W) | enter, exit, circulating_left, circulating_right |

## Quick Start

### Option 1: Run Complete Pipeline (Automated)

```bash
# Edit pipeline_example.sh to set your video paths
./pipeline_example.sh
```

### Option 2: Run Steps Manually

```bash
# Step 2: Track all 4 cameras
for cam in 1 2 3 4; do
    python step_2_object_tracking.py "normanniles${cam}_2025-10-20-06-00-45.mp4" \
        --zone-distances camera_configs/zone_distances_example.json
done

# Step 3: Extract features for each camera
for cam in 1 2 3 4; do
    python step_3_feature_extraction.py "normanniles${cam}_2025-10-20-06-00-45_tracking.json" \
        --zone-distances camera_configs/zone_distances_example.json
done

# Step 4: Merge features from all cameras
python step_4_multicamera_features.py \
    --cam1 normanniles1_2025-10-20-06-00-45_tracking_features.csv \
    --cam2 normanniles2_2025-10-20-06-00-45_tracking_features.csv \
    --cam3 normanniles3_2025-10-20-06-00-45_tracking_features.csv \
    --cam4 normanniles4_2025-10-20-06-00-45_tracking_features.csv \
    --output merged_features.csv
```

### Option 3: Auto-Discovery (Directory Mode)

```bash
# If all feature CSVs are in the same directory
python step_4_multicamera_features.py \
    --dir videos/ \
    --output features_output/merged_features.csv
```

## Step 4: Multi-Camera Features

### Features Created

#### 1. Camera-Specific Features (Prefixed by Direction)

For each direction (north, east, south, west):

**Flow Metrics:**
- `{direction}_entry_count` - Vehicle count entering
- `{direction}_exit_count` - Vehicle count exiting
- `{direction}_entry_flow_rate` - Entry rate (vehicles/min)
- `{direction}_exit_flow_rate` - Exit rate (vehicles/min)

**Vehicle Types:**
- `{direction}_entry_count_car`
- `{direction}_entry_count_truck`
- `{direction}_entry_count_bus`
- `{direction}_exit_count_car` (etc.)

**Circulating Metrics:**
- `{direction}_circulating_occupancy_avg` - Avg vehicles in circulation
- `{direction}_circulating_occupancy_max` - Max occupancy
- `{direction}_circulating_flow_rate` - Circulating flow rate

**Speed Metrics:**
- `{direction}_speed_avg_km_h` - Average speed
- `{direction}_speed_std_km_h` - Speed variability
- `{direction}_entry_speed_avg_km_h` - Entry zone speed
- `{direction}_circulating_speed_avg_km_h` - Circulating speed

**Derived Metrics:**
- `{direction}_entry_exit_balance` - Entry vs exit balance
- `{direction}_entry_circulating_ratio` - Conflict indicator
- `{direction}_circulating_density` - Congestion proxy

**Lag Features:**
- `{direction}_entry_count_lag_1` - Previous window value
- `{direction}_entry_count_lag_2` - 2 windows ago
- `{direction}_entry_count_lag_3` - 3 windows ago

**Rolling Features:**
- `{direction}_entry_flow_rate_rolling_mean_3` - 15min average
- `{direction}_entry_flow_rate_rolling_mean_6` - 30min average
- `{direction}_entry_flow_rate_rolling_mean_12` - 60min average

#### 2. Roundabout-Wide Features

**Total Flows:**
- `total_entry_count` - Total vehicles entering from all directions
- `total_exit_count` - Total vehicles exiting
- `total_entry_flow_rate` - Total entry rate
- `total_exit_flow_rate` - Total exit rate

**Circulating:**
- `total_circulating_occupancy` - Sum of all camera occupancies
- `avg_circulating_occupancy` - Average occupancy

**Speeds:**
- `avg_speed_km_h` - Average speed across roundabout
- `min_speed_km_h` - Slowest camera
- `max_speed_km_h` - Fastest camera
- `speed_range` - Speed differential

**System Balance:**
- `system_flow_balance` - Overall entry/exit balance

#### 3. Cross-Camera Features

**Flow Imbalance:**
- `entry_flow_imbalance` - Std dev of entry flows across cameras
- `entry_flow_variance` - Variance of entry flows
- `exit_flow_imbalance` - Std dev of exit flows
- `occupancy_imbalance` - Std dev of occupancies

**Opposite Entrance Balance (N-S, E-W):**
- `NS_entry_diff` - Absolute difference North vs South
- `NS_entry_balance` - Normalized balance [-1, 1]
- `EW_entry_diff` - Absolute difference East vs West
- `EW_entry_balance` - Normalized balance [-1, 1]

#### 4. Directional Features

**Axis Totals:**
- `NS_total_entry` - North + South entry flow
- `EW_total_entry` - East + West entry flow
- `NS_EW_imbalance` - N-S vs E-W imbalance

**Through Traffic:**
- `NS_through_flow` - North-South through traffic
- `EW_through_flow` - East-West through traffic

#### 5. Temporal Features

- `hour_of_day` - Hour (0-23)
- `day_of_week` - Day (0=Monday, 6=Sunday)
- `is_weekend` - Weekend flag
- `is_rush_hour` - Rush hour flag (7-9, 16-18)
- `time_since_midnight` - Hours since midnight

#### 6. Prediction Targets (8 columns)

- `north_entry_congestion` - Target for North entrance
- `north_exit_congestion` - Target for North exit
- `east_entry_congestion` - Target for East entrance
- `east_exit_congestion` - Target for East exit
- `south_entry_congestion` - Target for South entrance
- `south_exit_congestion` - Target for South exit
- `west_entry_congestion` - Target for West entrance
- `west_exit_congestion` - Target for West exit

Classes: `["free flowing", "light delay", "moderate delay", "heavy delay"]`

## Feature Engineering Tips

### 1. Camera-Specific Prediction

Each camera's congestion can be predicted using:
- Its own local features (e.g., `north_entry_flow_rate`)
- Upstream camera features (e.g., opposite entrance flow)
- Roundabout-wide context (e.g., `total_circulating_occupancy`)

Example for predicting `north_entry_congestion`:
```python
features = [
    'north_entry_flow_rate',
    'north_circulating_occupancy_avg',
    'north_entry_circulating_ratio',
    'south_entry_flow_rate',  # Opposite entrance (upstream)
    'total_circulating_occupancy',  # System-wide context
    'NS_entry_balance',  # Directional balance
    'entry_flow_imbalance',  # Cross-camera pattern
    'hour_of_day', 'is_rush_hour'  # Temporal
]
```

### 2. Temporal Dependencies

Use lag and rolling features for time-series prediction:
- `north_entry_count_lag_1` - What happened 5 min ago
- `north_entry_flow_rate_rolling_mean_6` - 30min trend
- Predict congestion 3-8 windows ahead (15-40 min future)

### 3. Cross-Camera Patterns

Traffic patterns often propagate:
- High North entry → High South circulating → South exit congestion
- Use opposite pair features (`NS_entry_balance`)
- Use directional features (`NS_EW_imbalance`)

### 4. Feature Importance

Key features for congestion prediction:
1. **Local flow rates**: Camera's own entry/exit rates
2. **Occupancy**: Circulating vehicle count
3. **Imbalance indicators**: Flow variance, directional imbalance
4. **Lag features**: Recent historical values
5. **Temporal context**: Hour of day, rush hour
6. **Speed metrics**: Slowing vehicles indicate congestion

## Machine Learning Pipeline

### 1. Prepare Training Data

```python
import pandas as pd

# Load merged features
df = pd.read_csv('merged_features.csv')

# Add congestion labels (example - replace with actual labels)
df['north_entry_congestion'] = label_function(df['north_entry_flow_rate'])
df['north_exit_congestion'] = label_function(df['north_exit_flow_rate'])
# ... repeat for all 8 targets

# Handle missing values
df = df.fillna(method='ffill')  # Forward fill for lag features

# Split features and targets
feature_cols = [col for col in df.columns if not col.endswith('_congestion')]
target_cols = [col for col in df.columns if col.endswith('_congestion')]

X = df[feature_cols]
y = df[target_cols]
```

### 2. Train Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

# Split data (preserve temporal order)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train multi-output model (8 predictions)
model = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=100, random_state=42)
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate per-camera performance
from sklearn.metrics import classification_report
for i, target_col in enumerate(target_cols):
    print(f"\n{target_col}:")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))
```

### 3. Future Prediction (15 min history → 6 future windows)

```python
# Given 15 minutes of historical data (3 windows of 5 min each)
historical_features = extract_features(last_15_minutes)

# Predict next 6 windows (minutes 18-23)
predictions = []
for window in range(6):
    # Predict next window
    pred = model.predict(historical_features[-1:])
    predictions.append(pred)

    # Update features for next prediction (use predicted values + lag features)
    historical_features = update_features(historical_features, pred)

# Output: 6 time windows × 8 predictions = 48 total predictions
```

## Batch Processing Multiple Video Sets

```python
import pandas as pd
from pathlib import Path

# Process multiple synchronized video sets
video_sets = [
    'normanniles*_2025-10-20-06-00-45.mp4',
    'normanniles*_2025-10-20-07-00-45.mp4',
    'normanniles*_2025-10-20-08-00-45.mp4',
]

all_features = []

for video_pattern in video_sets:
    # Run pipeline for this set
    # ... (steps 2-4)

    # Load merged features
    df = pd.read_csv(f'merged_features_{timestamp}.csv')
    all_features.append(df)

# Combine all into training dataset
training_data = pd.concat(all_features, ignore_index=True)
training_data.to_csv('complete_training_set.csv', index=False)

print(f"Combined {len(all_features)} video sets")
print(f"Total samples: {len(training_data)}")
```

## Troubleshooting

### Time Alignment Issues

If cameras have different numbers of windows:
```python
# The script uses intersection - only aligned windows are kept
# Check for warnings in output
```

### Missing Speed Data

If speeds are None/NaN:
```python
# Ensure zone_distances.json is provided in steps 2 and 3
# Check that distance keys match journey patterns:
#   - "enter_to_circulating_right"
#   - "circulating_left_to_circulating_right"
#   - "circulating_left_to_exit"
```

### Feature Name Conflicts

All camera features are prefixed with direction name to avoid conflicts:
- Camera 1 → `north_entry_count`
- Camera 2 → `east_entry_count`
- etc.

## Performance Tips

1. **Parallel Processing**: Run steps 2-3 for all cameras in parallel
   ```bash
   python step_2_object_tracking.py video1.mp4 & \
   python step_2_object_tracking.py video2.mp4 & \
   python step_2_object_tracking.py video3.mp4 & \
   python step_2_object_tracking.py video4.mp4 & \
   wait
   ```

2. **GPU Acceleration**: Use GPU for YOLO inference in step 2

3. **Time Window Selection**:
   - Smaller windows (2-3 min) → More data points, noisier
   - Larger windows (10 min) → Fewer points, smoother
   - Default 5 min is a good balance

## Next Steps

1. **Collect Labels**: Manually label congestion for training set
2. **Feature Selection**: Use feature importance to reduce dimensionality
3. **Hyperparameter Tuning**: Optimize model parameters
4. **Ensemble Models**: Combine multiple models for better accuracy
5. **Real-time Deployment**: Stream processing for live predictions

## References

- See `FEATURE_EXTRACTION_README.md` for detailed feature descriptions
- See `pipeline_example.sh` for complete working example
- Camera configs: `camera_configs/camera_{1-4}_zones.json`
- Zone distances: `camera_configs/zone_distances_example.json`
