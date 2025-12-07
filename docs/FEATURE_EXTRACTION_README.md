# Traffic Feature Extraction Pipeline

This pipeline extracts comprehensive features from roundabout traffic videos for congestion prediction.

## Pipeline Overview

1. **Step 1**: Setup zones (`step_1_setup_zones.py`)
2. **Step 2**: Object tracking (`step_2_object_tracking.py`) - **Now with speed calculation**
3. **Step 3**: Feature extraction (`step_3_feature_extraction.py`) - **NEW**

## Step 2: Object Tracking with Speed Calculation

The tracking script now calculates vehicle speeds based on zone distances.

### Usage

```bash
# Basic tracking (without speed calculation)
python step_2_object_tracking.py video.mp4

# With speed calculation (recommended)
python step_2_object_tracking.py video.mp4 \
  --zone-distances camera_configs/zone_distances_example.json

# With all options
python step_2_object_tracking.py video.mp4 \
  --model yolov8n.pt \
  --confidence 0.3 \
  --zone-distances camera_configs/zone_distances_example.json \
  --output annotated_video.mp4
```

### Zone Distances Configuration

Create a JSON file with distances between zones in meters:

```json
{
  "description": "Distances between zone pairs in meters",
  "circulating_left_to_circulating_right": 18.5,
  "circulating_left_to_exit": 22.0,
  "enter_to_circulating_right": 20.0
}
```

**How to measure distances:**
1. Use Google Maps satellite view
2. Measure distance between zone centroids
3. Or use GPS coordinates if available
4. Distances should represent typical vehicle paths

### Output

Tracking produces JSON files like `video_name_tracking.json` with:
- Vehicle journeys with zone events
- Timestamps for each zone entry/exit
- **Speed calculations** (km/h and m/s) if distances provided
- Journey validation status

Example output with speed:
```json
{
  "journey_times": {
    "t1_enter_start_zone": 12.5,
    "t2_exit_start_zone": 14.2,
    "t3_enter_end_zone": 16.8,
    "t4_exit_end_zone": 18.5,
    "total_journey_time": 6.0,
    "distance_meters": 18.5,
    "speed_m_s": 3.08,
    "speed_km_h": 11.1
  }
}
```

## Step 3: Feature Extraction

Extracts comprehensive features from tracking data for ML models.

### Features Extracted

#### 1. Entry Flow Metrics (per 5-min window)
- Total vehicle count crossing entry line
- Count by vehicle type (car, truck, bus, motorcycle)
- Entry flow rate (vehicles/min)

#### 2. Exit Flow Metrics
- Exit counts per leg
- Exit count by vehicle type
- Exit flow rate

#### 3. Circulating Flow Metrics
- Average/min/max occupancy (vehicles in circulation)
- Circulating flow count and rate
- Density indicators

#### 4. Origin-Destination Matrix
- Tracks entry → exit patterns
- OD flows for each valid pattern:
  - `enter → circulating_right`
  - `enter → exit`
  - `circulating_left → circulating_right`
  - `circulating_left → exit`

#### 5. Speed Metrics (requires zone distances)
- Average, std, min, max, median speeds
- Entry zone speeds
- Circulating zone speeds

#### 6. Temporal Features
- Hour of day
- Day of week
- Is weekend
- Is rush hour
- Time since midnight

#### 7. Derived Features
- Entry/Exit balance
- Entry/Circulating flow ratio (conflict indicator)
- Occupancy/Flow ratio
- Circulating density

#### 8. Lag Features (previous windows)
- Values from t-1, t-2, t-3 windows
- For key metrics: entry count, exit count, occupancy, speeds

#### 9. Rolling Features (moving averages)
- 15-minute rolling averages
- 30-minute rolling averages
- 60-minute rolling averages
- Rolling standard deviations

### Usage

```bash
# Basic feature extraction (5-min windows)
python step_3_feature_extraction.py video_tracking.json

# With custom output path
python step_3_feature_extraction.py video_tracking.json \
  --output features.csv

# With speed calculation (requires zone distances used in Step 2)
python step_3_feature_extraction.py video_tracking.json \
  --zone-distances camera_configs/zone_distances_example.json

# Custom time window (10 minutes = 600 seconds)
python step_3_feature_extraction.py video_tracking.json \
  --window 600

# Disable lag and rolling features
python step_3_feature_extraction.py video_tracking.json \
  --no-lags \
  --no-rolling
```

### Output

Produces CSV files with features per time window:

| window_idx | start_time | end_time | entry_count | exit_count | circulating_occupancy_avg | speed_avg_km_h | ... |
|------------|------------|----------|-------------|------------|---------------------------|----------------|-----|
| 0          | 0.0        | 300.0    | 12          | 10         | 3.5                       | 15.2           | ... |
| 1          | 300.0      | 600.0    | 15          | 13         | 4.2                       | 14.8           | ... |

## Complete Pipeline Example

```bash
# Step 1: Setup zones (if not already done)
python step_1_setup_zones.py video.mp4

# Step 2: Track objects and calculate speeds
python step_2_object_tracking.py video.mp4 \
  --zone-distances camera_configs/zone_distances_example.json

# Step 3: Extract features
python step_3_feature_extraction.py video_tracking.json \
  --zone-distances camera_configs/zone_distances_example.json \
  --output video_features.csv
```

## Feature Notes

### Implemented Features ✅

All core features are implemented:
- ✅ Entry flow metrics (count, rate, by type)
- ✅ Exit flow metrics (count, rate, by type)
- ✅ Circulating flow metrics (occupancy, flow rate)
- ✅ Origin-Destination matrices
- ✅ Speed metrics (avg, std, min, max, median)
- ✅ Temporal features (hour, day, weekend, rush hour)
- ✅ Derived features (ratios, density)
- ✅ Lag features (t-1, t-2, t-3)
- ✅ Rolling features (15min, 30min, 60min)

### Not Implemented (Too Complex) ❌

The following features were skipped as they're too complex for the current tracking setup:

- ❌ **Queue length**: Requires detecting stopped vehicles upstream of entry (needs advanced trajectory analysis)
- ❌ **Queue delay time**: Requires tracking individual vehicle waiting times in queue
- ❌ **Entry speeds from trajectory**: Current speeds are calculated between zones, not from frame-by-frame trajectory

### Workarounds

**Queue metrics alternative:**
- Use `circulating_occupancy_avg` as a proxy for congestion
- Use `entry_circulating_ratio` to detect entry conflicts
- Use `circulating_density` for congestion indication

**Speed alternative:**
- Zone-based speeds (current implementation) provide good average speed estimates
- More accurate than no speed data
- Sufficient for most ML models

## Tips for Best Results

1. **Measure zone distances accurately**
   - Use Google Maps for precise measurements
   - Measure along typical vehicle paths
   - Update `zone_distances_example.json` with real values

2. **Time windows**
   - Default 5 minutes works well for most traffic
   - Use shorter windows (2-3 min) for high-frequency data
   - Use longer windows (10 min) for sparse traffic

3. **Feature engineering**
   - Start with all features, then use feature selection
   - Lag features are important for time-series prediction
   - Rolling features smooth out noise

4. **Missing data**
   - First window will have NaN for lag features (expected)
   - Speed features will be None if distances not provided
   - Use pandas `.fillna()` or imputation for ML models

## Next Steps

After feature extraction:
1. Combine features from all videos into single dataset
2. Add target variable (congestion level)
3. Handle missing values
4. Feature selection/engineering
5. Train ML models

## Example: Batch Processing

```bash
# Process multiple videos
for video in videos/*.mp4; do
    echo "Processing $video..."

    # Track with speeds
    python step_2_object_tracking.py "$video" \
        --zone-distances camera_configs/zone_distances.json

    # Extract features
    tracking_json="${video%.mp4}_tracking.json"
    python step_3_feature_extraction.py "$tracking_json" \
        --zone-distances camera_configs/zone_distances.json
done

# Combine all features into one file
python -c "
import pandas as pd
from pathlib import Path

dfs = []
for csv_file in Path('videos').glob('*_features.csv'):
    df = pd.read_csv(csv_file)
    df['video_source'] = csv_file.stem
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined.to_csv('all_features.csv', index=False)
print(f'Combined {len(dfs)} files with {len(combined)} total rows')
"
```