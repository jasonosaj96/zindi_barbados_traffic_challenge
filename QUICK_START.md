# Quick Start Guide

Get started with the multi-camera traffic analysis pipeline in 5 minutes.

## Prerequisites

```bash
# Install dependencies
pip install ultralytics supervision opencv-python numpy pandas

# Download YOLO model (if needed)
# yolov8n.pt will be downloaded automatically on first run
```

## File Structure

```
zindi_barbados_traffic_challenge/
├── step_1_setup_zones.py          # Zone configuration (run once)
├── step_2_object_tracking.py      # Object tracking with speeds
├── step_3_feature_extraction.py   # Per-camera features
├── step_4_multicamera_features.py # Multi-camera feature merging
├── analyze_features.py             # Feature analysis utility
├── pipeline_example.sh             # Complete pipeline script
│
├── camera_configs/
│   ├── camera_1_zones.json        # North camera zones
│   ├── camera_2_zones.json        # East camera zones
│   ├── camera_3_zones.json        # South camera zones
│   ├── camera_4_zones.json        # West camera zones
│   └── zone_distances_example.json # Zone distances for speed calc
│
├── FEATURE_EXTRACTION_README.md   # Detailed feature docs
├── MULTICAMERA_PIPELINE.md        # Complete pipeline docs
└── QUICK_START.md                 # This file
```

## Camera Mapping

- **Camera 1** = North entrance/exit
- **Camera 2** = East entrance/exit
- **Camera 3** = South entrance/exit
- **Camera 4** = West entrance/exit

## 3-Step Quick Start

### 1. Setup Zones (One-time)

```bash
# For each camera, draw zones on first frame
python step_1_setup_zones.py path/to/camera1_video.mp4
python step_1_setup_zones.py path/to/camera2_video.mp4
python step_1_setup_zones.py path/to/camera3_video.mp4
python step_1_setup_zones.py path/to/camera4_video.mp4

# This creates camera_configs/camera_{1-4}_zones.json
# Define 4 zones per camera: enter, exit, circulating_left, circulating_right
```

### 2. Process 4 Synchronized Videos

```bash
# Example: Process one time period (15 minutes of 4 cameras)
TIMESTAMP="2025-10-20-06-00-45"

# Track + Extract features for each camera
for cam in 1 2 3 4; do
    # Step 2: Object tracking
    python step_2_object_tracking.py "normanniles${cam}_${TIMESTAMP}.mp4" \
        --zone-distances camera_configs/zone_distances_example.json

    # Step 3: Feature extraction
    python step_3_feature_extraction.py "normanniles${cam}_${TIMESTAMP}_tracking.json" \
        --zone-distances camera_configs/zone_distances_example.json
done

# Step 4: Merge all 4 cameras
python step_4_multicamera_features.py \
    --cam1 "normanniles1_${TIMESTAMP}_tracking_features.csv" \
    --cam2 "normanniles2_${TIMESTAMP}_tracking_features.csv" \
    --cam3 "normanniles3_${TIMESTAMP}_tracking_features.csv" \
    --cam4 "normanniles4_${TIMESTAMP}_tracking_features.csv" \
    --output "merged_features_${TIMESTAMP}.csv"
```

### 3. Analyze Results

```bash
# Analyze the merged features
python analyze_features.py "merged_features_${TIMESTAMP}.csv"
```

## Output Files

After running the pipeline, you'll have:

1. **Tracking JSONs** (4 files): `normanniles{1-4}_*_tracking.json`
   - Contains vehicle journeys, zone events, timestamps, speeds

2. **Feature CSVs** (4 files): `normanniles{1-4}_*_features.csv`
   - Per-camera features in 5-minute time windows

3. **Merged Features** (1 file): `merged_features_*.csv`
   - Combined features from all 4 cameras
   - Ready for ML model training
   - Contains 8 target columns for prediction

## Understanding the Output

### Merged Features CSV Structure

| Column Type | Example | Description |
|-------------|---------|-------------|
| Temporal | `hour_of_day`, `is_rush_hour` | Time context |
| Camera-specific | `north_entry_count`, `east_exit_flow_rate` | Per-camera metrics |
| Roundabout-wide | `total_entry_count`, `avg_speed_km_h` | System-level |
| Cross-camera | `entry_flow_imbalance`, `NS_entry_balance` | Relationships |
| Directional | `NS_total_entry`, `EW_through_flow` | Axis patterns |
| Targets | `north_entry_congestion`, `east_exit_congestion` | Predictions (8 total) |

### Example Row

```csv
window_idx,start_time,end_time,north_entry_count,north_exit_count,total_entry_count,...
0,0.0,300.0,12,10,45,42,3.5,15.2,...
1,300.0,600.0,15,13,52,48,4.2,14.8,...
```

Each row = one 5-minute time window
Each window has 100+ features describing traffic state

## ML Prediction Task

**Goal**: Predict congestion for 6 future time windows (minutes 18-23)

**Input**: 15 minutes of historical features (3 windows)

**Output**: 8 predictions per future window
- 4 cameras (N, E, S, W)
- 2 directions per camera (entry, exit)
- 4 congestion classes: [free flowing, light delay, moderate delay, heavy delay]

**Total predictions**: 6 windows × 8 outputs = 48 predictions

## Common Commands

### Process Single Video Set

```bash
./pipeline_example.sh
```

### Process Multiple Video Sets (Batch)

```bash
for timestamp in 2025-10-20-06-00-45 2025-10-20-07-00-45 2025-10-20-08-00-45; do
    echo "Processing $timestamp..."
    # Run pipeline for each set
done
```

### Auto-Discovery Mode

```bash
# If all feature CSVs are in same directory
python step_4_multicamera_features.py --dir videos/ --output merged.csv
```

### Analyze Features

```bash
# Quick analysis
python analyze_features.py merged_features.csv

# Show more details
python analyze_features.py merged_features.csv --top-n 50
```

## Configuration Files

### Zone Distances (`camera_configs/zone_distances_example.json`)

Measure distances between zones for speed calculation:

```json
{
  "circulating_left_to_circulating_right": 18.5,
  "circulating_left_to_exit": 22.0,
  "enter_to_circulating_right": 20.0
}
```

**How to measure**:
1. Open Google Maps satellite view
2. Find your roundabout
3. Measure distance between zone center points
4. Update JSON with real values (in meters)

### Time Window Configuration

Default is 5 minutes (300 seconds). Change with `--window`:

```bash
# 2-minute windows
python step_3_feature_extraction.py tracking.json --window 120

# 10-minute windows
python step_3_feature_extraction.py tracking.json --window 600
```

## Troubleshooting

### "Cannot extract camera number from filename"

**Problem**: Video filename doesn't contain `normanniles[1-4]`

**Solution**: Rename videos to include camera number:
```bash
mv video.mp4 normanniles1_2025-10-20-06-00-45.mp4
```

### "Must provide features for all 4 cameras"

**Problem**: Missing feature CSV for one or more cameras

**Solution**: Check all 4 cameras were processed:
```bash
ls -l *_features.csv
# Should see 4 files
```

### Missing Speed Values

**Problem**: Speed columns are NaN/None

**Solution**: Provide zone distances:
```bash
python step_2_object_tracking.py video.mp4 \
    --zone-distances camera_configs/zone_distances_example.json
```

### Time Windows Don't Align

**Problem**: Cameras have different number of windows

**Solution**: Check videos are same duration and synchronized:
```bash
# Check video lengths
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 video1.mp4
```

## Next Steps

1. **Collect Training Data**: Process multiple video sets
2. **Add Labels**: Label congestion levels for training
3. **Feature Engineering**: Select important features
4. **Train Model**: Use scikit-learn, XGBoost, or deep learning
5. **Evaluate**: Test on held-out time periods

## Documentation

- **Detailed Feature Docs**: See [FEATURE_EXTRACTION_README.md](FEATURE_EXTRACTION_README.md)
- **Complete Pipeline**: See [MULTICAMERA_PIPELINE.md](MULTICAMERA_PIPELINE.md)
- **Example Scripts**: See [pipeline_example.sh](pipeline_example.sh)

## Support

For issues or questions:
1. Check documentation files
2. Run `python analyze_features.py` to inspect output
3. Check log messages for warnings/errors

## Key Points

✅ **Keep cameras separate** until step 4 (feature merging)

✅ **Provide zone distances** for speed calculation

✅ **Synchronize videos** - all 4 should cover same time period

✅ **Use consistent naming** - include camera number in filename

✅ **Check alignment** - use `analyze_features.py` to verify

✅ **Temporal order matters** - don't shuffle for time-series prediction
