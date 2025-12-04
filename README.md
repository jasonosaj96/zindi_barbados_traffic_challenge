# Zindi Barbados Traffic Challenge
## AI-Powered Traffic Congestion Prediction & Analysis

## Table of Contents
- [Challenge Overview](#challenge-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Computer Vision Pipeline](#computer-vision-pipeline)
- [ML Pipeline](#ml-pipeline)
- [CSV Consolidation & Analysis](#csv-consolidation--analysis)
- [Vehicle Duration Tracking](#vehicle-duration-tracking)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Challenge Overview

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
./automation_scripts/automated_pipeline.sh --max-files 20 --save-video
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

---

## ML Pipeline

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ML PIPELINE ARCHITECTURE                  │
└─────────────────────────────────────────────────────────────┘

Raw Data (Train.csv)
        ↓
┌───────────────────────┐
│ Feature Engineering   │  → feature_engineering.py
│ - 15-min windows      │
│ - Temporal features   │
│ - Rolling stats       │
└───────────────────────┘
        ↓
Features + Labels (CSV)
        ↓
┌───────────────────────┐
│ Model Training        │  → train_model.py
│ - XGBoost classifier  │
│ - Cross-validation    │
│ - Hyperparameter tune │
└───────────────────────┘
        ↓
Trained Model (JSON)
        ↓
┌───────────────────────┐
│ Prediction            │  → predict.py
│ - Load test data      │
│ - Generate forecasts  │
│ - Create submission   │
└───────────────────────┘
        ↓
Submission File (CSV)
```

### ML Pipeline Quick Start

#### Option 1: Full Pipeline (Recommended)

Run everything in one command:

```bash
python pipeline.py --mode full
```

This will:
1. Generate features from training data
2. Train models with cross-validation
3. Generate features from test data
4. Create predictions
5. Save submission file

#### Option 2: Step-by-Step

**Step 1: Feature Engineering**

```bash
# Training data
python feature_engineering.py \
    --input dataset/Train.csv \
    --output features/train_features.csv

# Test data
python feature_engineering.py \
    --input dataset/Test.csv \
    --output features/test_features.csv \
    --test-mode
```

**Step 2: Model Training**

```bash
python train_model.py \
    --features features/train_features.csv \
    --labels features/train_labels.csv \
    --output models/ \
    --cv 5
```

**Step 3: Prediction**

```bash
python predict.py \
    --features features/test_features.csv \
    --model models/ \
    --output submissions/submission.csv
```

### ML Configuration

Edit [config.yaml](config.yaml) to customize:

```yaml
feature_engineering:
  window_size: 15              # Input window (minutes)
  prediction_horizon: 5        # Forecast horizon (minutes)
  embargo_period: 2            # Processing delay (minutes)

model:
  xgboost:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 200
    # ... more hyperparameters

training:
  test_split: 0.2
  cross_validation_folds: 5
```

### Features Engineered

#### Temporal Features
- **Current state**: Last minute congestion levels
- **Trends**: Linear slopes over 15-minute window
- **Acceleration**: Second-order trends
- **Persistence**: Duration of current congestion state

#### Statistical Features
- **Rolling means**: 5-minute and full window averages
- **Rolling max/min**: Peak congestion levels
- **Volatility**: Standard deviation of congestion

#### Derived Features
- **Direction imbalance**: Enter vs. exit congestion difference
- **Change detection**: Recent state changes
- **Progression indicators**: Worsening/improving flags

#### Time-Based Features
- **Hour of day**: 0-23
- **Day of week**: 0-6
- **Rush hour flags**: Morning (7-9) and evening (16-18)
- **Weekend indicator**: Binary flag

### Model Details

#### Architecture
- **Algorithm**: XGBoost Classifier
- **Objective**: Multi-class softmax (4 classes)
- **Models**: Separate models for entrance and exit congestion

#### Hyperparameters
```python
{
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1
}
```

#### Evaluation Metrics
- **Primary**: 70% Macro-F1 + 30% Accuracy (weighted)
- **Macro-F1**: Treats all congestion classes equally
- **Accuracy**: Overall prediction correctness

#### Validation Strategy
- **Stratified split**: 80% train, 20% validation
- **Cross-validation**: 5-fold stratified CV
- **Monitoring**: Confusion matrices, per-class F1 scores

### ML Output Files

#### Models
- `model_enter.json`: Entrance congestion predictor
- `model_exit.json`: Exit congestion predictor
- `model_metadata.json`: Configuration and feature names

#### Evaluation
- `evaluation_results.json`: Metrics on validation set
- `feature_importance.csv`: Top contributing features
- `confusion_matrices.png`: Visualization of predictions

#### Submission
- `submission.csv`: Competition submission format

---

## CSV Consolidation & Analysis

### Overview

The CSV consolidation script consolidates all `.counts.json` files into a single CSV file with one row per video and summary statistics as columns. This makes it easy to analyze results across multiple videos using spreadsheet software or data analysis tools.

### Automatic Generation

The consolidation script runs automatically as part of the pipeline:

```bash
# Quick test (2 files per camera)
./automation_scripts/quick_test.sh

# Full pipeline
./automation_scripts/automated_pipeline.sh --dataset small --max-files 10
```

Both scripts will generate `consolidated_results.csv` in the project root directory.

### Manual CSV Consolidation

You can also run the consolidation script manually:

```bash
# Basic usage
python src/processes/consolidate_to_csv.py --dir video_processed_files/

# With custom output filename
python src/processes/consolidate_to_csv.py --dir video_processed_files/ --output my_results.csv

# Filter specific cameras
python src/processes/consolidate_to_csv.py --dir video_processed_files/ --cameras 1,2

# Show summary statistics
python src/processes/consolidate_to_csv.py --dir video_processed_files/ --summary
```

### CSV Structure

The CSV file contains one row per processed video with the following column groups:

#### 1. Metadata Columns
- `filename` - Video filename (without extension)
- `camera` - Camera name (normanniles1, normanniles2, etc.)
- `date` - Recording date (YYYY-MM-DD)
- `time` - Recording time (HH:MM:SS)
- `datetime` - Combined date and time
- `hour` - Hour of recording (0-23)
- `minute` - Minute of recording (0-59)
- `duration_seconds` - Video duration in seconds
- `fps` - Frames per second
- `total_frames` - Total number of frames
- `video_width` - Video width in pixels
- `video_height` - Video height in pixels
- `total_vehicles_tracked` - Total unique vehicles tracked

#### 2. Zone Count Columns
- `zone_count_north_entry` - Vehicles counted in north entry zone
- `zone_count_north_exit` - Vehicles counted in north exit zone
- `zone_count_north_circulating` - Vehicles counted in north circulating zone
- _(Similar for east, south, west)_

#### 3. Dwell Time Statistics
- `dwell_mean_north_entry` - Mean dwell time in north entry zone (seconds)
- `dwell_median_north_entry` - Median dwell time
- `dwell_min_north_entry` - Minimum dwell time
- `dwell_max_north_entry` - Maximum dwell time
- `dwell_std_north_entry` - Standard deviation of dwell time
- `dwell_count_north_entry` - Number of vehicles with dwell time recorded
- _(Similar for all other zones)_

#### 4. Vehicle Class Distribution
- `vehicle_class_car` - Number of cars detected
- `vehicle_class_truck` - Number of trucks detected
- `vehicle_class_bus` - Number of buses detected
- `vehicle_class_motorcycle` - Number of motorcycles detected
- _(Other vehicle classes as detected)_

#### 5. Journey Statistics
- `avg_journey_duration` - Average time vehicles were visible (seconds)
- `min_journey_duration` - Minimum journey duration
- `max_journey_duration` - Maximum journey duration
- `avg_zones_visited` - Average number of zones visited per vehicle
- `max_zones_visited` - Maximum zones visited by any vehicle

#### 6. Polygon Statistics (per zone)
- `poly_total_visits_[zone]` - Total number of zone visits
- `poly_unique_vehicles_[zone]` - Unique vehicles in zone
- `poly_throughput_[zone]` - Throughput (vehicles per minute)
- `poly_vtype_[class]_[zone]` - Vehicle type counts per zone
- `poly_dur_mean_[zone]` - Mean duration in zone
- `poly_dur_median_[zone]` - Median duration in zone
- `poly_dur_std_[zone]` - Std deviation of duration
- `poly_speed_mean_[zone]` - Mean speed in zone (pixels/sec)
- `poly_speed_median_[zone]` - Median speed in zone
- `poly_speed_std_[zone]` - Std deviation of speed
- `poly_size_mean_[zone]` - Mean bounding box area
- `poly_size_median_[zone]` - Median bounding box area
- `poly_size_std_[zone]` - Std deviation of bounding box area
- `poly_conf_mean_[zone]` - Mean detection confidence
- `poly_conf_median_[zone]` - Median detection confidence

#### 7. Roundabout Metrics (if available)
- `roundabout_total_entering_flow` - Total entering flow
- `roundabout_total_circulating_flow` - Total circulating flow
- `roundabout_overall_capacity_index` - Capacity utilization index
- `roundabout_is_feasible` - Whether roundabout is operating feasibly
- `roundabout_overall_los` - Overall level of service
- `roundabout_[direction]_entering_flow` - Per-direction entering flow
- `roundabout_[direction]_circulating_flow` - Per-direction circulating flow
- `roundabout_[direction]_entry_capacity` - Per-direction entry capacity
- `roundabout_[direction]_capacity_index` - Per-direction capacity index
- `roundabout_[direction]_level_of_service` - Per-direction LOS
- `roundabout_[direction]_average_delay` - Per-direction average delay
- `roundabout_[direction]_queue_length` - Per-direction queue length

#### 8. Other Columns
- `source_file` - Path to the source .counts.json file
- `video_path` - Path to the source video file

### Example CSV Analysis

#### Using Pandas (Python)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv('consolidated_results.csv')

# Basic statistics
print(f"Total videos: {len(df)}")
print(f"Total vehicles tracked: {df['total_vehicles_tracked'].sum()}")
print(f"Average vehicles per video: {df['total_vehicles_tracked'].mean():.1f}")

# Videos per camera
print("\nVideos per camera:")
print(df['camera'].value_counts())

# Average vehicles by camera
print("\nAverage vehicles per camera:")
print(df.groupby('camera')['total_vehicles_tracked'].mean())

# Vehicle class distribution (total)
vehicle_cols = [col for col in df.columns if col.startswith('vehicle_class_')]
print("\nTotal vehicle class distribution:")
for col in vehicle_cols:
    class_name = col.replace('vehicle_class_', '')
    total = df[col].sum()
    print(f"  {class_name}: {int(total)}")

# Traffic by hour of day
print("\nAverage vehicles by hour:")
print(df.groupby('hour')['total_vehicles_tracked'].mean().sort_index())

# Plot traffic by hour
df.groupby('hour')['total_vehicles_tracked'].mean().plot(kind='bar')
plt.title('Average Vehicles by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Average Vehicles')
plt.tight_layout()
plt.savefig('traffic_by_hour.png')

# Export filtered data
filtered = df[(df['camera'] == 'normanniles1') & (df['total_vehicles_tracked'] > 50)]
filtered.to_csv('camera1_high_traffic.csv', index=False)
```

#### Using Excel/Google Sheets

1. Open `consolidated_results.csv` in Excel or Google Sheets
2. Use filters to analyze specific cameras or time periods
3. Create pivot tables to summarize data by camera, hour, date, etc.
4. Create charts to visualize trends

#### Using Command Line

```bash
# View first 10 rows
head -n 10 consolidated_results.csv

# Count total rows (videos)
wc -l consolidated_results.csv

# Get specific columns
cut -d',' -f1,2,3,13 consolidated_results.csv | head

# Sort by total vehicles (descending)
(head -n 1 consolidated_results.csv && tail -n +2 consolidated_results.csv | sort -t',' -k13 -nr) | head -n 20

# Filter specific camera
grep "normanniles1" consolidated_results.csv > camera1_only.csv
```

### CSV Column Selection Tips

The CSV can have 100+ columns depending on your zone configuration. Here are some tips for working with it:

#### Essential Columns for Basic Analysis
```python
essential_cols = [
    'filename', 'camera', 'date', 'time', 'hour',
    'total_vehicles_tracked', 'duration_seconds'
]
df_basic = df[essential_cols]
```

#### Zone Count Analysis
```python
zone_count_cols = [col for col in df.columns if col.startswith('zone_count_')]
df_zones = df[['filename', 'camera'] + zone_count_cols]
```

#### Vehicle Class Analysis
```python
vehicle_class_cols = [col for col in df.columns if col.startswith('vehicle_class_')]
df_classes = df[['filename', 'camera'] + vehicle_class_cols]
```

#### Time-based Analysis
```python
time_cols = ['filename', 'camera', 'date', 'time', 'hour', 'minute', 'total_vehicles_tracked']
df_time = df[time_cols].sort_values(['date', 'time'])
```

---

## Vehicle Duration Tracking

### Overview

The automation pipeline generates `.durations.json` files that contain detailed information about each detected vehicle and the time they spent in each zone.

### Duration Output Files

After running the pipeline, you'll get:

1. **Individual Duration Files**: `*.durations.json` - One per video, saved alongside the corresponding `.counts.json` file
2. **Consolidated File**: `vehicle_durations_all.json` - All videos combined into a single file

### Duration Data Structure

#### Individual Duration File (*.durations.json)

```json
{
  "metadata": {
    "video_path": "path/to/video.mp4",
    "filename": "normanniles1-2025-10-15-14-30-00",
    "camera": "normanniles1",
    "date": "2025-10-15",
    "time": "14:30:00",
    "datetime": "2025-10-15 14:30:00",
    "duration_seconds": 60.0,
    "fps": 30.0,
    "total_frames": 1800
  },
  "vehicles": {
    "1": {
      "tracker_id": 1,
      "class_id": 2,
      "class_name": "car",
      "first_seen_time": 0.5,
      "last_seen_time": 15.3,
      "total_time_visible": 14.8,
      "total_frames_tracked": 444,
      "zones": {
        "entry": [
          {
            "time_entered": 0.5,
            "time_exited": 3.2,
            "duration": 2.7,
            "frame_entered": 15,
            "frame_exited": 96
          }
        ],
        "exit_north": [
          {
            "time_entered": 12.1,
            "time_exited": 15.3,
            "duration": 3.2,
            "frame_entered": 363,
            "frame_exited": 459
          }
        ]
      }
    }
  },
  "summary": {
    "total_vehicles": 1,
    "vehicles_per_zone": {
      "entry": 1,
      "exit_north": 1
    }
  }
}
```

#### Consolidated File (vehicle_durations_all.json)

```json
{
  "files_processed": 4,
  "total_vehicles_tracked": 25,
  "videos": [
    {
      "source_file": "path/to/video1.counts.json",
      "metadata": { ... },
      "vehicles": { ... },
      "summary": { ... }
    }
  ]
}
```

### Duration Field Descriptions

#### Vehicle Fields

| Field | Type | Description |
|-------|------|-------------|
| `tracker_id` | integer | Unique ID assigned to this vehicle by the tracking system |
| `class_id` | integer | YOLO class ID (e.g., 2 = car, 7 = truck) |
| `class_name` | string | Human-readable vehicle class name |
| `first_seen_time` | float | Time (seconds) when vehicle was first detected in video |
| `last_seen_time` | float | Time (seconds) when vehicle was last detected in video |
| `total_time_visible` | float | Total time (seconds) vehicle was visible in video |
| `total_frames_tracked` | integer | Number of frames where vehicle was detected |

#### Zone Entry Fields

| Field | Type | Description |
|-------|------|-------------|
| `time_entered` | float or null | Time (seconds) when vehicle entered the zone |
| `time_exited` | float or null | Time (seconds) when vehicle exited the zone |
| `duration` | float or null | Time (seconds) spent in the zone (dwell time) |
| `frame_entered` | integer or null | Frame number when vehicle entered the zone |
| `frame_exited` | integer or null | Frame number when vehicle exited the zone |

**Note**: A vehicle can visit the same zone multiple times, so `zones` contains a list of visits per zone.

### Duration Usage Examples

#### View all vehicles in the first video

```bash
cat vehicle_durations_all.json | jq '.videos[0].vehicles'
```

#### Get vehicle IDs and their zones

```bash
cat vehicle_durations_all.json | jq '.videos[0].vehicles | to_entries[] | {id: .key, zones: .value.zones | keys}'
```

#### Find vehicles that visited a specific zone

```bash
cat video.durations.json | jq '.vehicles | to_entries[] | select(.value.zones.entry != null) | {id: .key, class: .value.class_name}'
```

#### Calculate total time all vehicles spent in a zone

```bash
cat video.durations.json | jq '[.vehicles[].zones.entry[]?.duration] | add'
```

#### Get summary statistics

```bash
cat vehicle_durations_all.json | jq '{total_files: .files_processed, total_vehicles: .total_vehicles_tracked}'
```

### Duration Integration with Pipeline

#### Quick Test

The `quick_test.sh` script automatically generates duration files:

```bash
./automation_scripts/quick_test.sh
```

This will create:
- `video_processed_files/normanniles1/*.durations.json`
- `vehicle_durations_all.json`

#### Full Pipeline

The `automated_pipeline.sh` script also includes duration extraction:

```bash
./automation_scripts/automated_pipeline.sh --dataset small --max-files 5
```

#### Manual Extraction

You can also run the extraction script manually:

```bash
# Extract from a single counts.json file
python src/processes/extract_vehicle_durations.py video.counts.json

# Extract from all files in a directory
python src/processes/extract_vehicle_durations.py --dir video_processed_files/

# Extract for specific cameras only
python src/processes/extract_vehicle_durations.py --dir video_processed_files/ --cameras 1,2

# Save individual files alongside counts.json
python src/processes/extract_vehicle_durations.py --dir video_processed_files/ --individual

# Specify custom output filename
python src/processes/extract_vehicle_durations.py --dir video_processed_files/ --output my_durations.json
```

### Duration Use Cases

1. **Traffic Flow Analysis**: Track how long vehicles spend in each zone to identify bottlenecks
2. **Route Pattern Detection**: Analyze which zones vehicles visit in sequence
3. **Vehicle Classification**: Correlate dwell times with vehicle types
4. **Time Series Analysis**: Combine with video metadata to analyze patterns by time of day
5. **Machine Learning Features**: Use zone durations as input features for predictive models

### Duration Notes

- Vehicle IDs are unique **within each video** but not across videos
- A vehicle may visit the same zone multiple times (e.g., circling a roundabout)
- `null` values for entry/exit times indicate incomplete tracking (vehicle entered/exited the frame while in the zone)
- All times are in seconds from the start of the video
- Frame numbers are 0-indexed

---

## Files Utilized by automated_pipeline.sh

When you run `./automation_scripts/automated_pipeline.sh`, the following files are utilized:

### 🔧 **Scripts Executed (in execution order)**

#### Step 1: Prerequisites Check
- **External Tools**: `python`, `gsutil`, `yolov8n.pt` (YOLO model - auto-downloaded if missing)

#### Step 2: Download Videos
1. **`src/processes/download_and_process.py`** - Downloads videos from Google Cloud Storage

#### Step 3: Zone Configuration Validation
2. **`camera_configs/camera_1_zones.json`** - Camera 1 (North) zone definitions
3. **`camera_configs/camera_2_zones.json`** - Camera 2 (East) zone definitions
4. **`camera_configs/camera_3_zones.json`** - Camera 3 (South) zone definitions
5. **`camera_configs/camera_4_zones.json`** - Camera 4 (West) zone definitions

   *(If zones are missing, optionally runs:)*
6. **`1_setup_all_cameras.sh`** - Interactive zone setup script
   - Which calls **`1_zone_designer.py`** for each camera

#### Step 4: Video Processing
7. **`src/processes/parallel_process.py`** - Main parallel processing orchestrator
   - Which spawns **`src/classes/vehicle_counting.py`** as subprocess for each video
     - Which imports **`src/classes/roundabout_metrics.py`** (optional, for advanced metrics)

#### Step 5: Duration Extraction
8. **`src/processes/extract_vehicle_durations.py`** - Extracts vehicle zone entry/exit times

#### Step 6: CSV Consolidation
9. **`src/processes/consolidate_to_csv.py`** - Consolidates all results into CSV format

### 📦 **Input/Output Files**

#### Input Files (Created/Used):
- **Camera zone configs**: `camera_configs/camera_[1-4]_zones.json`
- **Downloaded videos**: From GCS → `$OUTPUT_DIR/normanniles[1-4]/*.mp4`
- **YOLO model**: `yolov8n.pt` (auto-downloaded on first run)

#### Output Files Per Video (in `$OUTPUT_DIR/normanniles[1-4]/`):
- **`video_name.counts.json`** - Complete vehicle tracking data including:
  - Vehicle counts per zone
  - Vehicle journeys with zone visits
  - Dwell time statistics
  - Zone statistics (throughput, speeds, sizes)
  - Roundabout metrics (if available)

- **`video_name.durations.json`** - Simplified vehicle zone duration data:
  - Each vehicle ID with tracker info
  - Entry/exit times for each zone visited
  - Duration spent in each zone

- **`video_name.annotated.mp4`** - Annotated video (if `--save-video` flag used):
  - Zones drawn on frames
  - Bounding boxes around vehicles
  - Vehicle IDs and classes

#### Consolidated Output Files (in project root):
- **`vehicle_durations_all.json`** - All vehicle durations from all videos combined
- **`consolidated_results.csv`** - Summary statistics for all videos (100+ columns)
- **`processing_summary.json`** - Processing metadata and statistics

### 📊 **Complete File Dependency Tree**

```
automated_pipeline.sh
│
├── Step 1: Prerequisites
│   ├── python (required)
│   ├── gsutil (required)
│   └── yolov8n.pt (auto-downloaded)
│
├── Step 2: Download
│   └── src/processes/download_and_process.py
│       └── Downloads → $OUTPUT_DIR/normanniles[1-4]/*.mp4
│
├── Step 3: Zone Configuration
│   ├── camera_configs/camera_1_zones.json
│   ├── camera_configs/camera_2_zones.json
│   ├── camera_configs/camera_3_zones.json
│   ├── camera_configs/camera_4_zones.json
│   └── (optional) 1_setup_all_cameras.sh
│       └── 1_zone_designer.py
│
├── Step 4: Process Videos
│   └── src/processes/parallel_process.py
│       └── spawns → src/classes/vehicle_counting.py (per video)
│           ├── uses → yolov8n.pt
│           ├── uses → camera_configs/camera_[1-4]_zones.json
│           ├── imports → src/classes/roundabout_metrics.py (optional)
│           └── generates → *.counts.json
│
├── Step 5: Extract Durations
│   └── src/processes/extract_vehicle_durations.py
│       ├── reads → *.counts.json
│       └── generates → *.durations.json + vehicle_durations_all.json
│
└── Step 6: Consolidate CSV
    └── src/processes/consolidate_to_csv.py
        ├── reads → *.counts.json
        └── generates → consolidated_results.csv
```

### 🎯 **File Count Summary**

| Category | Count | Files |
|----------|-------|-------|
| **Python Scripts** | 6 | download_and_process.py, parallel_process.py, vehicle_counting.py, extract_vehicle_durations.py, consolidate_to_csv.py, roundabout_metrics.py (optional) |
| **Shell Scripts** | 2 | automated_pipeline.sh, 1_setup_all_cameras.sh (optional) |
| **Zone Configs** | 4 | camera_1_zones.json, camera_2_zones.json, camera_3_zones.json, camera_4_zones.json |
| **External Tools** | 3 | python, gsutil, yolov8n.pt |
| **Output per Video** | 2-3 | *.counts.json, *.durations.json, *.annotated.mp4 (optional) |
| **Consolidated Outputs** | 3 | vehicle_durations_all.json, consolidated_results.csv, processing_summary.json |

### 💡 **Key Notes**

1. **Modular Architecture**: Each step can be run independently
2. **Subprocess Spawning**: `parallel_process.py` spawns `vehicle_counting.py` as subprocesses for parallel processing
3. **Automatic Skip**: Videos with existing `.counts.json` files are automatically skipped
4. **Optional Components**:
   - Zone setup (`1_setup_all_cameras.sh`) only runs if configs are missing
   - Roundabout metrics only if `roundabout_metrics.py` is available
   - Annotated videos only if `--save-video` flag is used
5. **Real-time Processing**: Pipeline processes videos in parallel using multiple CPU cores

### 📁 **File Location Convention**

```
project_root/
├── automation_scripts/
│   └── automated_pipeline.sh              # Main pipeline script
├── src/
│   ├── processes/
│   │   ├── download_and_process.py        # Step 2
│   │   ├── parallel_process.py            # Step 4 orchestrator
│   │   ├── extract_vehicle_durations.py   # Step 5
│   │   └── consolidate_to_csv.py          # Step 6
│   └── classes/
│       ├── vehicle_counting.py            # Step 4 worker
│       └── roundabout_metrics.py          # Optional
├── camera_configs/
│   ├── camera_1_zones.json                # Step 3
│   ├── camera_2_zones.json                # Step 3
│   ├── camera_3_zones.json                # Step 3
│   └── camera_4_zones.json                # Step 3
├── 1_setup_all_cameras.sh                 # Optional (Step 3)
├── 1_zone_designer.py                     # Optional (Step 3)
├── yolov8n.pt                             # Auto-downloaded
├── video_processed_files/                 # Output directory
│   ├── normanniles1/
│   │   ├── video1.mp4
│   │   ├── video1.counts.json
│   │   ├── video1.durations.json
│   │   └── video1.annotated.mp4
│   ├── normanniles2/
│   ├── normanniles3/
│   └── normanniles4/
├── vehicle_durations_all.json             # Consolidated output
├── consolidated_results.csv               # Consolidated output
└── processing_summary.json                # Consolidated output
```

---

## Troubleshooting

### Computer Vision Issues

#### No detections
- Lower `--conf` threshold
- Verify polygons cover vehicle paths
- Check video file is valid

#### Too many false positives
- Increase `--conf` threshold
- Refine zone polygons to exclude problem areas

#### Zones not counting
- Verify polygons cover vehicle paths using zone designer
- Check zone configuration JSON file is valid
- Test with `--display` flag to visualize

#### Video not found
- Check `_data/` directory structure matches dataframe paths
- Verify GCS download completed successfully

### ML Pipeline Issues

#### Missing required columns
**Solution**: Check that your CSV has all required columns:
- `view_label`, `time_segment_id`, `datetimestamp_start`
- `congestion_enter_rating`, `congestion_exit_rating` (for training)

#### Insufficient data
**Solution**: Ensure you have at least 22 consecutive time segments per camera:
- 15 for input window
- 2 for embargo
- 5 for prediction

#### Model files not found
**Solution**: Run training before prediction:
```bash
python pipeline.py --mode train
```

### CSV Consolidation Issues

#### Empty CSV
- Check that `.counts.json` files exist in the data directory
- Verify the directory path is correct
- Use `--summary` flag to see processing details

#### Missing Columns
- Some columns only appear if tracking is enabled
- Roundabout metrics require the roundabout_metrics module
- Vehicle classes depend on what was detected in the videos

#### Performance
- Large datasets (1000+ videos) may take a few minutes
- Consider filtering by camera to process subsets
- Use `--cameras` flag to process specific cameras only

### General Performance Tips
- Use `yolov8n.pt` for speed (default)
- Use `yolov8x.pt` for accuracy (slower)
- Process without `--display` for faster batch processing
- Skip existing: Script automatically skips videos with `.counts.json` files

---

## ML Best Practices Implemented

### ✅ Modular Design
- Separate scripts for each pipeline stage
- Reusable components
- Clean interfaces between modules

### ✅ Configuration Management
- Centralized config file (YAML)
- No hardcoded parameters
- Easy experimentation

### ✅ Reproducibility
- Fixed random seeds
- Versioned data paths
- Logging at every step

### ✅ Validation
- Input data validation
- Missing column checks
- Stratified splits

### ✅ Logging
- Structured logging throughout
- Progress tracking
- Error messages

### ✅ Evaluation
- Multiple metrics (F1, Accuracy, weighted)
- Cross-validation
- Confusion matrices
- Feature importance analysis

### ✅ Scalability
- Efficient data processing
- XGBoost's built-in parallelism
- Modular architecture for easy improvements

---

## Next Steps & Enhancements

### Immediate Improvements
1. **Add video features**: Integrate vehicle counts, dwell times, flow rates
2. **Multi-camera fusion**: Combine all 4 camera views
3. **Hyperparameter tuning**: Grid search or Bayesian optimization
4. **Feature selection**: Remove low-importance features

### Advanced Enhancements
1. **Ensemble methods**: Stack multiple models
2. **Deep learning**: LSTM/GRU for temporal modeling
3. **Data augmentation**: Generate synthetic training samples
4. **Online learning**: Update model with new data

---

## License

This project is for the Zindi Barbados Traffic Challenge.

## Contributing

Adjust polygon zones in `camera_configs/*.json` to match your specific camera angles and road layouts.


