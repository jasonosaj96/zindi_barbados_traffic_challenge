#!/bin/bash
# Automated pipeline for downloading and processing traffic data from GCS

set -e  # Exit on error

echo "========================================================================"
echo "Automated Traffic Data Pipeline"
echo "========================================================================"
echo ""

# Configuration - can be overridden with environment variables or command line args
DATASET="${DATASET:-small}"              # small or full
CAMERAS="${CAMERAS:-all}"                # all or comma-separated (e.g., 1,2)
MAX_FILES="${MAX_FILES:-}"               # Leave empty for no limit (e.g., 10)
WORKERS="${WORKERS:-4}"                  # Number of parallel workers
START_DATE="${START_DATE:-}"             # Optional date filter (YYYY-MM-DD)
END_DATE="${END_DATE:-}"                 # Optional date filter (YYYY-MM-DD)
OUTPUT_DIR="${OUTPUT_DIR:-video_processed_files}"  # Output directory
SAVE_VIDEO="${SAVE_VIDEO:-false}"        # Save annotated videos

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --cameras)
            CAMERAS="$2"
            shift 2
            ;;
        --max-files)
            MAX_FILES="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --start-date)
            START_DATE="$2"
            shift 2
            ;;
        --end-date)
            END_DATE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --save-video)
            SAVE_VIDEO="true"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET        Dataset to use: small or full (default: small)"
            echo "  --cameras CAMERAS        Cameras to process: all or comma-separated (default: all)"
            echo "  --max-files N            Maximum files to download per camera (default: no limit)"
            echo "  --workers N              Number of parallel workers (default: 4)"
            echo "  --start-date DATE        Start date filter YYYY-MM-DD (optional)"
            echo "  --end-date DATE          End date filter YYYY-MM-DD (optional)"
            echo "  --output-dir DIR         Output directory (default: video_processed_files)"
            echo "  --save-video             Save annotated videos with detections"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --dataset small --cameras all --max-files 5"
            echo "  $0 --cameras 1,2 --workers 2 --max-files 10"
            echo "  $0 --start-date 2025-10-01 --end-date 2025-10-31"
            echo ""
            echo "Or use environment variables:"
            echo "  DATASET=full CAMERAS=1,2 MAX_FILES=10 $0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Cameras: $CAMERAS"
echo "  Max files per camera: ${MAX_FILES:-unlimited}"
echo "  Parallel workers: $WORKERS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Save annotated videos: $SAVE_VIDEO"
if [ -n "$START_DATE" ]; then
    echo "  Date range: $START_DATE to ${END_DATE:-present}"
fi
echo ""
echo "========================================================================"
echo ""

# Step 1: Check prerequisites
echo "Step 1: Checking prerequisites..."
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "✗ Python not found"
    exit 1
fi
echo "✓ Python found"

# Check gsutil
if ! command -v gsutil &> /dev/null; then
    echo "✗ gsutil not found"
    echo ""
    echo "Install Google Cloud SDK:"
    echo "  macOS: brew install --cask google-cloud-sdk"
    echo "  Linux: curl https://sdk.cloud.google.com | bash"
    exit 1
fi
echo "✓ gsutil found"

# Check if YOLOv8 model exists
if [ ! -f "yolov8n.pt" ]; then
    echo "⚠️  YOLOv8n model not found, it will be downloaded automatically"
fi

echo ""

# Step 2: Download data from GCS
echo "========================================================================"
echo "Step 2: Downloading data from Google Cloud Storage"
echo "========================================================================"
echo ""

DOWNLOAD_CMD="python download_and_process.py --dataset $DATASET --cameras $CAMERAS --output-dir $OUTPUT_DIR --download-only"

if [ -n "$MAX_FILES" ]; then
    DOWNLOAD_CMD="$DOWNLOAD_CMD --max-files $MAX_FILES"
fi

if [ -n "$START_DATE" ]; then
    DOWNLOAD_CMD="$DOWNLOAD_CMD --start-date $START_DATE"
fi

if [ -n "$END_DATE" ]; then
    DOWNLOAD_CMD="$DOWNLOAD_CMD --end-date $END_DATE"
fi

echo "Running: $DOWNLOAD_CMD"
echo ""

$DOWNLOAD_CMD

if [ $? -ne 0 ]; then
    echo "✗ Download failed"
    exit 1
fi

echo ""
echo "✓ Download complete"
echo ""

# Step 3: Check if zone configurations exist
echo "========================================================================"
echo "Step 3: Checking zone configurations"
echo "========================================================================"
echo ""

CONFIGS_MISSING=false

for cam in 1 2 3 4; do
    CONFIG="camera_configs/camera_${cam}_zones.json"
    if [ ! -f "$CONFIG" ]; then
        echo "✗ Missing: $CONFIG"
        CONFIGS_MISSING=true
    else
        echo "✓ Found: $CONFIG"
    fi
done

echo ""

if [ "$CONFIGS_MISSING" = true ]; then
    echo "⚠️  Some zone configurations are missing!"
    echo ""
    read -p "Run zone designer setup now? (y/n): " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Starting zone designer setup..."
        ./setup_all_cameras.sh
    else
        echo ""
        echo "⚠️  Processing will continue but results may be incomplete"
        echo "   without zone configurations"
        echo ""
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

echo ""

# Step 4: Process videos with YOLOv8n
echo "========================================================================"
echo "Step 4: Processing videos with YOLOv8n"
echo "========================================================================"
echo ""

PROCESS_CMD="python parallel_process.py --workers $WORKERS --data-dir $OUTPUT_DIR"

if [ "$CAMERAS" != "all" ]; then
    PROCESS_CMD="$PROCESS_CMD --cameras $CAMERAS"
fi

if [ "$SAVE_VIDEO" = "true" ]; then
    PROCESS_CMD="$PROCESS_CMD --save-video"
fi

echo "Running: $PROCESS_CMD"
echo ""

$PROCESS_CMD

if [ $? -ne 0 ]; then
    echo "✗ Processing failed"
    exit 1
fi

echo ""

# Step 5: Summary
echo "========================================================================"
echo "Pipeline Complete!"
echo "========================================================================"
echo ""
echo "Results are saved in:"
echo "  - $OUTPUT_DIR/normanniles[1-4]/*.counts.json (vehicle counts)"
echo "  - $OUTPUT_DIR/normanniles[1-4]/*.movements.json (movement patterns)"
echo "  - $OUTPUT_DIR/normanniles[1-4]/*.detections.json (raw detections)"
if [ "$SAVE_VIDEO" = "true" ]; then
    echo "  - $OUTPUT_DIR/normanniles[1-4]/*.annotated.mp4 (annotated videos)"
fi
echo ""
echo "Processing summary saved in:"
echo "  - processing_summary.json"
echo ""
echo "Next steps:"
echo "  - Review results: ls -lh $OUTPUT_DIR/*/*.json"
echo "  - Analyze data: python analyze_counts.py"
echo "  - Generate visualizations: python batch_process_videos.py"
echo ""
