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
echo "Quick Test Complete!"
echo "========================================================================"
echo ""
echo "Check results in video_processed_files/ directory:"
echo "  - JSON data files (*.counts.json, *.movements.json)"
echo "  - Annotated videos (*.annotated.mp4) with zones and detections"
echo ""
echo "View an annotated video:"
echo "  open video_processed_files/normanniles1/*.annotated.mp4"
echo ""
