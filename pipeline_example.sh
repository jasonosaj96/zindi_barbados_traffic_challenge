#!/bin/bash
#
# Complete Pipeline Example: From Videos to ML-Ready Features
# Processes 4 synchronized roundabout videos (North, East, South, West)
#

set -e  # Exit on error

echo "=========================================="
echo "Roundabout Traffic Feature Pipeline"
echo "=========================================="
echo ""

# Configuration
VIDEO_DIR="videos"
OUTPUT_DIR="features_output"
ZONE_DISTANCES="camera_configs/zone_distances_example.json"
TIME_WINDOW=300  # 5 minutes in seconds

# Example video set (synchronized at same timestamp)
TIMESTAMP="2025-10-20-06-00-45"
VIDEO_1="${VIDEO_DIR}/normanniles1_${TIMESTAMP}.mp4"  # North
VIDEO_2="${VIDEO_DIR}/normanniles2_${TIMESTAMP}.mp4"  # East
VIDEO_3="${VIDEO_DIR}/normanniles3_${TIMESTAMP}.mp4"  # South
VIDEO_4="${VIDEO_DIR}/normanniles4_${TIMESTAMP}.mp4"  # West

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Processing video set: $TIMESTAMP"
echo "Time window: ${TIME_WINDOW}s ($(echo "$TIME_WINDOW / 60" | bc) minutes)"
echo ""

# ==========================================
# STEP 1: Setup Zones (Run once per camera)
# ==========================================
echo "Step 1: Zone setup (skipping - assumed already done)"
echo "  If needed, run: python step_1_setup_zones.py <video_path>"
echo ""

# ==========================================
# STEP 2: Object Tracking (per camera)
# ==========================================
echo "Step 2: Object tracking with speed calculation..."
echo ""

for video in "$VIDEO_1" "$VIDEO_2" "$VIDEO_3" "$VIDEO_4"; do
    if [ -f "$video" ]; then
        echo "  Tracking: $(basename "$video")"
        python step_2_object_tracking.py "$video" \
            --zone-distances "$ZONE_DISTANCES" \
            --confidence 0.3
    else
        echo "  WARNING: Video not found: $video"
    fi
done

echo ""
echo "Step 2 complete. Tracking JSONs created."
echo ""

# ==========================================
# STEP 3: Feature Extraction (per camera)
# ==========================================
echo "Step 3: Feature extraction per camera..."
echo ""

TRACKING_1="${VIDEO_DIR}/normanniles1_${TIMESTAMP}_tracking.json"
TRACKING_2="${VIDEO_DIR}/normanniles2_${TIMESTAMP}_tracking.json"
TRACKING_3="${VIDEO_DIR}/normanniles3_${TIMESTAMP}_tracking.json"
TRACKING_4="${VIDEO_DIR}/normanniles4_${TIMESTAMP}_tracking.json"

for tracking_json in "$TRACKING_1" "$TRACKING_2" "$TRACKING_3" "$TRACKING_4"; do
    if [ -f "$tracking_json" ]; then
        output_csv="${tracking_json%.json}_features.csv"
        echo "  Extracting features: $(basename "$tracking_json")"
        python step_3_feature_extraction.py "$tracking_json" \
            --output "$output_csv" \
            --window "$TIME_WINDOW" \
            --zone-distances "$ZONE_DISTANCES"
    else
        echo "  WARNING: Tracking JSON not found: $tracking_json"
    fi
done

echo ""
echo "Step 3 complete. Feature CSVs created."
echo ""

# ==========================================
# STEP 4: Multi-Camera Feature Merging
# ==========================================
echo "Step 4: Merging features from all 4 cameras..."
echo ""

FEATURES_1="${VIDEO_DIR}/normanniles1_${TIMESTAMP}_tracking_features.csv"
FEATURES_2="${VIDEO_DIR}/normanniles2_${TIMESTAMP}_tracking_features.csv"
FEATURES_3="${VIDEO_DIR}/normanniles3_${TIMESTAMP}_tracking_features.csv"
FEATURES_4="${VIDEO_DIR}/normanniles4_${TIMESTAMP}_tracking_features.csv"

MERGED_OUTPUT="${OUTPUT_DIR}/merged_features_${TIMESTAMP}.csv"

python step_4_multicamera_features.py \
    --cam1 "$FEATURES_1" \
    --cam2 "$FEATURES_2" \
    --cam3 "$FEATURES_3" \
    --cam4 "$FEATURES_4" \
    --output "$MERGED_OUTPUT"

echo ""
echo "Step 4 complete. Merged features saved to: $MERGED_OUTPUT"
echo ""

# ==========================================
# Summary
# ==========================================
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - Tracking JSONs: ${VIDEO_DIR}/*_tracking.json"
echo "  - Camera features: ${VIDEO_DIR}/*_features.csv"
echo "  - Merged features: $MERGED_OUTPUT"
echo ""
echo "Next steps:"
echo "  1. Add congestion labels to the 8 target columns"
echo "  2. Combine multiple time periods into training dataset"
echo "  3. Train ML model"
echo ""
echo "Target columns (8 predictions per time window):"
echo "  - north_entry_congestion, north_exit_congestion"
echo "  - east_entry_congestion, east_exit_congestion"
echo "  - south_entry_congestion, south_exit_congestion"
echo "  - west_entry_congestion, west_exit_congestion"
echo ""
echo "Congestion levels: [free flowing, light delay, moderate delay, heavy delay]"
echo ""
