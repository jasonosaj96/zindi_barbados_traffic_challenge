# Zindi Barbados Traffic Challenge

Vehicle counting and congestion analysis for roundabout traffic using YOLO and Supervision.

## Overview

This project processes roundabout traffic videos from 4 cameras (Norman Niles #1-4), detects vehicles using YOLOv8, and counts them in defined zones to analyze congestion patterns.

**New Features:**
- ✨ **Individual vehicle tracking** with unique IDs
- ⏱️ **Dwell time measurement** - track how long vehicles spend in each zone
- 📊 **Enhanced statistics** - mean, median, min, max dwell times per zone
- 🔄 **Directional flow analysis** - understand traffic movement patterns


## Challenge: AI-Powered Traffic Congestion Prediction

### Problem Statement

This is a **computer vision + time series forecasting** challenge focused on predicting traffic congestion at the Norman Niles roundabout in Barbados. The problem combines automated feature extraction from video data with real-time congestion prediction.

**Business Context**: Barbados relies heavily on cars, buses, and taxis for transportation. The Ministry of Transport and Works seeks machine learning solutions to predict and mitigate traffic congestion, improving mobility for all citizens.

### The Machine Learning Task

This is a **multi-class classification problem** with temporal constraints:

**Input**:
- 4 synchronized video streams (1 per roundabout entrance/exit)
- 15 minutes of historical video data
- Each video segment: ~1 minute duration

**Output**:
- Predict congestion levels for 6 future time windows (minutes 18-23)
- 8 predictions per time window (4 cameras × 2 directions: enter/exit)
- 4 congestion classes: `["free flowing", "light delay", "moderate delay", "heavy delay"]`

**Temporal Structure**:
```
[Minutes 1-15: Input Videos] → [Minutes 16-17: Processing Embargo] → [Minutes 18-23: Prediction Target]
```

### Technical Constraints

1. **No Future Data Leakage**: Solutions must operate in real-time
   - Cannot use data from minute N+1 to predict minute N
   - Sequential prediction only (no look-ahead)
   - No back-propagation during inference

2. **Feature Extraction Pipeline**:
   - Computer vision models (e.g., YOLO, RCNN) for object detection
   - Automated feature engineering from video (no manual annotation)
   - Extract temporal dynamics: flow rates, queue lengths, vehicle trajectories

3. **Deployment Constraints**:
   - 2-minute processing embargo simulates real-world latency
   - Model must be reproducible and deployable
   - Back-propagation allowed during offline training, not during inference

### AI/ML Approach

This challenge requires a **two-stage pipeline**:

#### Stage 1: Computer Vision Feature Extraction
Transform unstructured video into structured time-series features:
- **Object Detection**: YOLOv8/Faster-RCNN to detect vehicles (cars, buses, trucks, motorcycles)
- **Object Tracking**: ByteTrack/DeepSORT for vehicle trajectories and dwell time
- **Zone Analysis**: Define entry/exit/queue/circulating zones per camera
- **Feature Engineering**:
  - Vehicle counts per zone
  - Flow rates (vehicles/minute)
  - Average dwell time in roundabout
  - Queue lengths
  - Entry-to-exit timing patterns
  - Directional flow imbalances
  - Vehicle type distributions

#### Stage 2: Time Series Classification
Convert features to congestion predictions:
- **Model Options**:
  - Traditional ML: XGBoost, Random Forest, LightGBM with temporal features
  - Sequential models: LSTM, GRU, Temporal Convolutional Networks
  - Hybrid: Vision features → Gradient boosting ensemble
- **Feature Aggregation**: Combine multi-camera views into joint representation
- **Temporal Context**: Use 15-minute history to predict 5-minute future

### Evaluation Metrics

**Dual-Metric Weighted Score**:
- **70% Macro-F1**: Class-balanced performance (handles imbalanced congestion classes)
- **30% Accuracy**: Overall prediction correctness

**Submission Format**:
```csv
ID,Target,Target_Accuracy
time_segment_181_Norman Niles #1_congestion_enter_rating,heavy delay,heavy delay
time_segment_181_Norman Niles #1_congestion_exit_rating,heavy delay,heavy delay
...
```

Both `Target` (for F1) and `Target_Accuracy` (for Accuracy) columns required.

### Interpretability Prize

Top 20 solutions must submit **feature importance analysis**:
- Feature name (e.g., "avg_dwell_time_camera_1")
- Feature contribution (SHAP values, permutation importance, etc.)
- Notes on congestion causality

**Domain Insight**: Barbadian drivers often don't signal when entering/exiting roundabouts—this behavioral pattern may be detectable in trajectory data and correlate with congestion.

### Data Characteristics

**Training Data**:
- 4 camera views: North, East, South, West (Norman Niles roundabout)
- ~1-minute video segments synchronized across cameras
- Labels: 4-class congestion ratings for enter/exit per camera
- Augmentation allowed (must be reproducible)

**Test Data**:
- 15 minutes of unlabeled video
- Predict minutes 18-23 (6 time steps ahead)
- Real-time constraint: no access to future segments

### Key Machine Learning Challenges

1. **Unstructured to Structured**: Convert raw pixels to predictive features
2. **Multi-View Fusion**: Combine 4 camera perspectives coherently
3. **Temporal Dynamics**: Capture congestion propagation patterns
4. **Class Imbalance**: Handle imbalanced congestion class distribution
5. **Real-Time Constraints**: Respect causality (no look-ahead)
6. **Generalization**: Train on limited labeled data, predict unseen patterns

### Success Criteria

**Technical**:
- High Macro-F1 and Accuracy on private test set
- Reproducible automated pipeline (no manual labels)
- Real-time compatibility (2-minute embargo respected)

**Business Impact**:
- Identify actionable congestion drivers (e.g., flow imbalances, signal usage)
- Enable Ministry of Transport to design evidence-based interventions
- Scalable solution for other Barbados roundabouts

---

**About the Partners**:
- **Keleya Labs**: Innovation for social impact and citizen experience improvement
- **GovTech Barbados Ltd.**: State-owned enterprise driving government digital transformation








See [VEHICLE_TRACKING.md](VEHICLE_TRACKING.md) for detailed documentation on tracking and dwell time analysis.

## Project Structure

```
├── vehicle_counting.py              # Core vehicle counting script
├── 1_zone_designer.py               # Interactive zone designer (16 zones)
├── example_tracking.py              # Example for vehicle tracking
│
├── Automation Scripts
├── automated_pipeline.sh            # Complete automated pipeline
├── quick_test.sh                    # Quick test with sample data
├── download_and_process.py          # Download from GCS + process
├── parallel_process.py              # Parallel video processing
├── batch_process_videos.py          # Batch process from dataframe
├── analyze_counts.py                # Analysis and visualization
│
├── Setup Scripts
├── setup_all_cameras.sh             # Interactive zone setup (all cameras)
├── process_all_cameras.sh           # Process all cameras
│
├── Configuration
├── camera_configs/                  # Zone configurations
│   ├── camera_1_zones.json         # Norman Niles #1 (North)
│   ├── camera_2_zones.json         # Norman Niles #2 (East)
│   ├── camera_3_zones.json         # Norman Niles #3 (South)
│   └── camera_4_zones.json         # Norman Niles #4 (West)
│
├── Data Directories
├── video_processed_files/           # Downloaded and processed videos
│   ├── normanniles1/
│   ├── normanniles2/
│   ├── normanniles3/
│   └── normanniles4/
├── data_challenge/
│   ├── Train.csv                    # Original metadata
│   └── Train_with_counts.csv        # Enriched with vehicle counts
│
└── Documentation
    ├── README.md                    # This file
    ├── AUTOMATION_GUIDE.md          # Automation system guide
    ├── ZONE_DESIGNER_GUIDE.md       # Zone designer usage
    ├── VEHICLE_TRACKING.md          # Vehicle tracking features
    └── DIRECTIONAL_ZONES.md         # Directional zone specifications
```

## Installation

### 1. Python Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Core Dependencies:**
- opencv-python>=4.8.0
- ultralytics>=8.0.0 (YOLOv8)
- supervision>=0.16.0
- numpy>=1.24.0
- pandas>=2.0.0
- matplotlib, seaborn (visualizations)
- tqdm (progress bars)

### 2. Google Cloud SDK (for automated downloads)

**macOS:**
```bash
brew install --cask google-cloud-sdk
```

**Linux:**
```bash
curl https://sdk.cloud.google.com | bash
```

**Windows:** Download from [cloud.google.com/sdk](https://cloud.google.com/sdk/docs/install)

### 3. Make Scripts Executable

```bash
chmod +x *.sh
```

## Quick Start

### Automated Pipeline (Recommended)

**Test with sample data:**
```bash
./quick_test.sh
```

**Full automated pipeline:**
```bash
./automation_scripts/automated_pipeline.sh --max-files 10 --save-video
```

This will:
1. Download videos from Google Cloud Storage
2. Setup zone configurations (interactive)
3. Process videos with YOLOv8n in parallel
4. Generate annotated videos and analysis data

See [AUTOMATION_GUIDE.md](AUTOMATION_GUIDE.md) for detailed documentation.

## Manual Workflow

### Step 1: Setup Polygon Zones (One-time)

**Automated setup for all 4 cameras (16 zones):**

```bash
./setup_all_cameras.sh
```

This creates 4 zones per camera direction:
- **Entry** (Green): Vehicles entering from approach
- **Exit** (Red): Vehicles leaving to approach
- **Queue** (Yellow): Waiting area before entry
- **Circulating** (Orange): Vehicles in roundabout

**Manual setup for individual cameras:**
```bash
python 1_zone_designer.py _data/normanniles1/video.mp4 \
    --config camera_configs/camera_1_zones.json \
    --direction north
```

See [ZONE_DESIGNER_GUIDE.md](ZONE_DESIGNER_GUIDE.md) and [DIRECTIONAL_ZONES.md](DIRECTIONAL_ZONES.md) for detailed instructions.

**Interactive Controls:**
- **Left click**: Add point to polygon
- **Right click**: Complete current zone, start next
- **'s' key**: Save configuration
- **'r' key**: Reset current zone
- **'q' key**: Quit without saving

The tool guides you through 4 zones:
1. **incoming** (Red): Vehicles approaching roundabout
2. **exiting** (Green): Vehicles leaving roundabout
3. **curve** (Blue): Vehicles in the roundabout curve
4. **inside_roundabout** (Yellow): Vehicles at merge point

### Step 2: Download Videos from Google Cloud Storage

**Data Sources:**
- Small dataset (re-encoded): `gs://brb-traffic/`
- Full dataset (>500GB): `gs://brb-traffic-full/`

**Download with limits:**
```bash
python download_and_process.py \
    --dataset small \
    --cameras all \
    --max-files 10 \
    --output-dir video_processed_files
```

**Options:**
- `--dataset small|full`: Choose dataset size
- `--cameras all|1,2,3,4`: Select cameras
- `--max-files N`: Limit downloads per camera
- `--output-dir DIR`: Output directory
- `--download-only`: Skip processing

### Step 3: Process Videos

#### Option A: Parallel Processing (Recommended)

Process multiple videos simultaneously:

```bash
python parallel_process.py \
    --data-dir video_processed_files \
    --workers 4 \
    --save-video
```

**Options:**
- `--data-dir DIR`: Video directory (default: video_processed_files)
- `--workers N`: Parallel workers (default: CPU count - 1)
- `--cameras 1,2,3,4`: Process specific cameras
- `--save-video`: Generate annotated videos
- `--max-videos N`: Limit number of videos

#### Option B: Batch Process from Dataframe

```bash
python batch_process_videos.py \
    --csv data_challenge/Train.csv \
    --output data_challenge/Train_with_counts.csv
```

#### Option C: Individual Videos

```bash
python vehicle_counting.py video_processed_files/normanniles1/video.mp4 \
    --config camera_configs/camera_1_zones.json \
    --save-video \
    --display
```

**Options:**
- `--config`: Zone configuration JSON
- `--save-video`: Save annotated video
- `--display`: Show video during processing
- `--conf`: Confidence threshold (default: 0.3)
- `--iou`: IOU threshold (default: 0.7)
- `--model`: YOLO model (default: yolov8n.pt)

### Step 4: Analyze Results

Generate statistics and visualizations:

```bash
python analyze_counts.py \
    --csv data_challenge/Train_with_counts.csv \
    --plots \
    --output-dir analysis_plots
```

**Analyses:**
- Congestion correlation (counts vs. ratings)
- Camera-by-camera breakdown
- Temporal patterns (hourly, daily)
- Statistical summaries
- Zone-specific analysis

**Output Plots:**
- `analysis_plots/congestion_vs_counts.png`: Boxplots by congestion rating
- `analysis_plots/zones_by_camera.png`: Zone breakdown per camera
- `analysis_plots/temporal_patterns.png`: Time series analysis

## Zone Definitions (16-Zone System)

Each camera captures 4 directional zones (North, South, East, West):

### Entry Zone (Green)
- Vehicles entering the roundabout from the approach
- Right lane for single-lane approaches
- Captures inbound traffic demand

### Exit Zone (Red)
- Vehicles exiting the roundabout to the departure road
- Left lane for single-lane exits
- Captures outbound traffic flow

### Queue Zone (Yellow)
- Waiting area before entry point
- Aligns with entry lanes
- Captures congestion and queuing

### Circulating Zone (Orange)
- Vehicles actively circulating in the roundabout
- Follows the arc/curve of the roundabout
- Captures throughput and saturation

**Key Rules:**
- Zones are direction-specific (not overlapping)
- Lanes should be side-by-side, not overlapping
- Queue zones align with entry lanes
- Entry flow > Exit flow indicates backup
- Queue length > 5 suggests congestion
- Circulating > 80% indicates saturation

See [DIRECTIONAL_ZONES.md](DIRECTIONAL_ZONES.md) for detailed specifications.

## Camera Layout

```
              Camera 3 (South)
                    ↓

Camera 2 (East) ← [ROUNDABOUT] → Camera 4 (West)

                    ↑
              Camera 1 (North)
```

## Data Format

### Input CSV (Train.csv)
```csv
responseId,view_label,videos,video_time,datetimestamp_start,datetimestamp_end,
congestion_enter_rating,congestion_exit_rating,...
```

### Output CSV (Train_with_counts.csv)
Same as input plus:
```csv
...,count_incoming,count_exiting,count_curve,count_inside_roundabout,count_total
```

### Counts JSON (*.counts.json)
Saved alongside each video:
```json
{
  "video": "normanniles1/video.mp4",
  "total_frames": 1800,
  "zones": {
    "incoming": 45,
    "exiting": 38,
    "curve": 12,
    "inside_roundabout": 8
  }
}
```

## Vehicle Detection

**YOLO Classes Detected:**
- Class 2: Car
- Class 3: Motorcycle
- Class 5: Bus
- Class 7: Truck

**Detection Settings:**
- Default model: YOLOv8n (fastest)
- Confidence threshold: 0.3 (adjustable)
- IOU threshold: 0.7 (adjustable)
- Triggering anchor: BOTTOM_CENTER (counts when vehicle's bottom-center enters zone)

## Tips & Best Practices

### Polygon Definition
1. **Exclude obstructions**: Draw around plants, signs, or other blocking objects
2. **Cover full lanes**: Ensure polygons cover entire width of road
3. **Avoid overlaps**: Keep zones separate to prevent double-counting
4. **Test first**: Process one short video to validate zones before batch processing

### Performance
- Use `yolov8n.pt` for speed (default)
- Use `yolov8x.pt` for accuracy (slower)
- Process without `--display` for faster batch processing
- Skip existing: Script automatically skips videos with `.counts.json` files

### Troubleshooting
- **No detections**: Lower `--conf` threshold
- **Too many false positives**: Increase `--conf` threshold
- **Zones not counting**: Verify polygons cover vehicle paths
- **Video not found**: Check `_data/` directory structure matches dataframe paths

## Example Workflows

### Automated Pipeline (Recommended)

```bash
# Quick test with 2 files per camera + annotated videos
./quick_test.sh

# Full pipeline with custom settings
./automation_scripts/automated_pipeline.sh \
    --dataset small \
    --cameras all \
    --max-files 50 \
    --workers 4 \
    --save-video

# View results
ls -lh video_processed_files/normanniles1/
open video_processed_files/normanniles1/*.annotated.mp4
```

### Manual Pipeline

```bash
# 1. Setup zones for all cameras (one-time)
./setup_all_cameras.sh

# 2. Download videos from GCS
python download_and_process.py \
    --dataset small \
    --cameras all \
    --max-files 20 \
    --download-only

# 3. Process in parallel with annotations
python parallel_process.py \
    --workers 4 \
    --save-video

# 4. Analyze results with visualizations
python analyze_counts.py --plots

# 5. View enriched data
head data_challenge/Train_with_counts.csv
```

### Single Camera Test

```bash
# Download for one camera
python download_and_process.py \
    --cameras 1 \
    --max-files 5

# Setup zones
python 1_zone_designer.py \
    video_processed_files/normanniles1/video.mp4 \
    --config camera_configs/camera_1_zones.json \
    --direction north

# Process with annotations
python vehicle_counting.py \
    video_processed_files/normanniles1/video.mp4 \
    --config camera_configs/camera_1_zones.json \
    --save-video \
    --display
```

## Advanced Usage

### Python API

```python
from vehicle_counting import VehicleCounter
import json

# Load configuration
with open('camera_configs/camera_1.json', 'r') as f:
    config = json.load(f)

# Initialize counter
counter = VehicleCounter(
    model_path="yolov8n.pt",
    camera_config=config,
    confidence_threshold=0.3
)

# Process video
counts = counter.process_video(
    video_path="_data/normanniles1/video.mp4",
    display=False,
    save_counts=True
)

print(f"Total vehicles: {counts}")
```

### Custom Analysis

```python
import pandas as pd

# Load enriched data
df = pd.read_csv('data_challenge/Train_with_counts.csv')

# Analyze by congestion rating
print(df.groupby('congestion_enter_rating')['count_total'].mean())

# Find peak hours
df['hour'] = pd.to_datetime(df['datetimestamp_start']).dt.hour
print(df.groupby('hour')['count_total'].mean())
```

## Project Context

**Challenge**: Predict traffic congestion at roundabouts
**Data**: 1-minute video segments with congestion ratings
**Data Sources**:
- Small dataset (re-encoded): `gs://brb-traffic/`
- Full dataset (>500GB): `gs://brb-traffic-full/`
**Approach**: Extract vehicle count and flow features using YOLOv8n + 16-zone system
**Cameras**: 4 views (N, E, S, W) of Norman Niles roundabout
**Features**: Entry/Exit counts, queue lengths, circulating vehicles, dwell times, movement patterns

## Output Files

```
video_processed_files/
├── normanniles1/
│   ├── video.mp4                      # Original video
│   ├── video.counts.json              # Vehicle counts by zone
│   ├── video.movements.json           # Movement patterns
│   ├── video.detections.json          # Raw detections
│   └── video.annotated.mp4            # Annotated (if --save-video)
├── normanniles2/
├── normanniles3/
└── normanniles4/

camera_configs/
├── camera_1_zones.json                # North camera zones
├── camera_2_zones.json                # East camera zones
├── camera_3_zones.json                # South camera zones
└── camera_4_zones.json                # West camera zones

data_challenge/
├── Train.csv                          # Original metadata
└── Train_with_counts.csv              # Enriched with counts

analysis_plots/
├── congestion_vs_counts.png
├── zones_by_camera.png
└── temporal_patterns.png

processing_summary.json                # Batch processing summary
```

## License

This project is for the Zindi Barbados Traffic Challenge.

## Contributing

Adjust polygon zones in `camera_configs/*.json` to match your specific camera angles and road layouts.


# Process videos with automatic metrics generation
python parallel_process.py --cameras 1 --workers 4

# Resume and generate missing metrics
python parallel_process.py --resume --cameras 1

# Process with batch analysis at the end
python parallel_process.py --cameras 1 --batch-analysis

# Skip automatic metrics (faster, manual metrics later)
python parallel_process.py --cameras 1 --no-metrics