#!/bin/bash
# Quick test script - Downloads and processes a small sample for testing

echo "========================================================================"
echo "Quick Test - Download & Process Sample Data"
echo "========================================================================"
echo ""
echo "This will download and process a small sample (2 files per camera)"
echo "from the small dataset to test the pipeline."
echo ""
echo "Features enabled:"
echo "  - Download 2 files per camera"
echo "  - Process with 2 parallel workers"
echo "  - Generate annotated videos with detections"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

echo ""
echo "Starting quick test..."
echo ""

# Run pipeline with limited files and video annotation enabled
./automation_scripts/automated_pipeline.sh \
    --dataset small \
    --cameras all \
    --max-files 2 \
    --workers 2 \
    --output-dir video_processed_files \
    --save-video

echo ""
echo "========================================================================"
echo "Extracting Vehicle Zone Duration Data"
echo "========================================================================"
echo ""

# Extract vehicle durations from counts.json files
python src/processes/extract_vehicle_durations.py \
    --dir video_processed_files \
    --output vehicle_durations_all.json \
    --individual

echo ""
echo "========================================================================"
echo "Consolidating Results to CSV"
echo "========================================================================"
echo ""

# Consolidate to CSV
python src/processes/consolidate_to_csv.py \
    --dir video_processed_files \
    --output consolidated_results.csv \
    --summary

echo ""
echo "========================================================================"
echo "Quick Test Complete!"
echo "========================================================================"
echo ""
echo "Check results in video_processed_files/ directory:"
echo "  - JSON data files (*.counts.json - vehicle counts and zone statistics)"
echo "  - Vehicle duration data (*.durations.json - entry/exit times per vehicle)"
echo "  - Annotated videos (*.annotated.mp4 - with zones and detections drawn)"
echo ""
echo "Vehicle duration files contain:"
echo "  - All detected vehicle IDs"
echo "  - Entry/exit times for each zone"
echo "  - Duration spent in each zone"
echo ""
echo "Consolidated outputs:"
echo "  - vehicle_durations_all.json (all videos - vehicle durations)"
echo "  - consolidated_results.csv (all videos - summary statistics)"
echo ""
echo "Quick commands:"
echo "  View CSV: head consolidated_results.csv"
echo "  View annotated video: open video_processed_files/normanniles1/*.annotated.mp4"
echo "  View durations: cat vehicle_durations_all.json | jq '.videos[0].vehicles'"
echo ""
