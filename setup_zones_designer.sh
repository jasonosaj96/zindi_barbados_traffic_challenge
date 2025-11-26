#!/bin/bash
# Setup comprehensive zones for all 4 cameras using the zone designer
# This creates 13-zone configurations for detailed traffic analysis

echo "=========================================="
echo "Zone Designer - Comprehensive Zone Setup"
echo "=========================================="
echo ""
echo "This will guide you through creating comprehensive zone configurations"
echo "for all 4 roundabout cameras using the advanced zone designer."
echo ""
echo "Each camera will have up to 13 zones:"
echo "  - North: entry, exit, queue (3 zones)"
echo "  - South: entry, exit, queue (3 zones)"
echo "  - East:  entry, exit, queue (3 zones)"
echo "  - West:  entry, exit, queue (3 zones)"
echo "  - Intersection center (1 zone)"
echo ""
echo "You only need to define zones visible in each camera view."
echo "Press 'N' to skip zones not visible from that camera."
echo ""
echo "=========================================="
echo ""

# Function to setup zones for a camera
setup_camera_zones() {
    local camera_num=$1
    local camera_name=$2
    local direction=$3

    echo ""
    echo "=========================================="
    echo "Camera $camera_num: $camera_name ($direction View)"
    echo "=========================================="

    if [ -d "_data/$camera_name" ]; then
        # Find first video file
        VIDEO=$(find "_data/$camera_name" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) ! -name "*_annotated*" | head -n 1)

        if [ -n "$VIDEO" ]; then
            echo "Found video: $VIDEO"
            echo ""
            echo "Recommended zones for $direction view camera:"
            case $direction in
                "North")
                    echo "  ✓ north_entry, north_exit, north_queue (most visible)"
                    echo "  ✓ intersection_center"
                    echo "  ⚠ east/west entry (partial view)"
                    echo "  ✗ south zones (likely not visible)"
                    ;;
                "East")
                    echo "  ✓ east_entry, east_exit, east_queue (most visible)"
                    echo "  ✓ intersection_center"
                    echo "  ⚠ north/south entry (partial view)"
                    echo "  ✗ west zones (likely not visible)"
                    ;;
                "South")
                    echo "  ✓ south_entry, south_exit, south_queue (most visible)"
                    echo "  ✓ intersection_center"
                    echo "  ⚠ east/west entry (partial view)"
                    echo "  ✗ north zones (likely not visible)"
                    ;;
                "West")
                    echo "  ✓ west_entry, west_exit, west_queue (most visible)"
                    echo "  ✓ intersection_center"
                    echo "  ⚠ north/south entry (partial view)"
                    echo "  ✗ east zones (likely not visible)"
                    ;;
            esac
            echo ""
            echo "Controls:"
            echo "  Left Click: Add point"
            echo "  Right Click: Finish zone"
            echo "  S: Save zones"
            echo "  N: Skip to next zone"
            echo "  C: Clear current zone"
            echo "  D: Delete last zone"
            echo "  Q: Quit"
            echo ""
            read -p "Press Enter to start zone designer..."

            # Run zone designer
            python 1_zone_designer.py "$VIDEO" --config "camera_configs/camera_${camera_num}_full.json"

            if [ $? -eq 0 ]; then
                echo ""
                echo "✓ Camera $camera_num zones saved"

                # Check if config was created
                if [ -f "camera_configs/camera_${camera_num}_full.json" ]; then
                    echo "✓ Configuration file: camera_configs/camera_${camera_num}_full.json"

                    # Count zones in config
                    ZONE_COUNT=$(python -c "import json; data=json.load(open('camera_configs/camera_${camera_num}_full.json')); print(len(data.get('zones', {})))" 2>/dev/null)
                    if [ -n "$ZONE_COUNT" ]; then
                        echo "  Zones defined: $ZONE_COUNT"
                    fi
                fi
            else
                echo "✗ Camera $camera_num setup failed or cancelled"
            fi
        else
            echo "✗ No video found in _data/$camera_name/"
        fi
    else
        echo "✗ Directory _data/$camera_name not found"
    fi
}

# Camera 1 (North)
setup_camera_zones 1 "normanniles1" "North"

echo ""
read -p "Continue to Camera 2? (y/n): " continue
if [ "$continue" != "y" ]; then
    echo "Setup stopped by user"
    exit 0
fi

# Camera 2 (East)
setup_camera_zones 2 "normanniles2" "East"

echo ""
read -p "Continue to Camera 3? (y/n): " continue
if [ "$continue" != "y" ]; then
    echo "Setup stopped by user"
    exit 0
fi

# Camera 3 (South)
setup_camera_zones 3 "normanniles3" "South"

echo ""
read -p "Continue to Camera 4? (y/n): " continue
if [ "$continue" != "y" ]; then
    echo "Setup stopped by user"
    exit 0
fi

# Camera 4 (West)
setup_camera_zones 4 "normanniles4" "West"

# Summary
echo ""
echo "=========================================="
echo "Zone Setup Complete!"
echo "=========================================="
echo ""
echo "Configuration files created:"
for i in 1 2 3 4; do
    if [ -f "camera_configs/camera_${i}_full.json" ]; then
        ZONE_COUNT=$(python -c "import json; data=json.load(open('camera_configs/camera_${i}_full.json')); print(len(data.get('zones', {})))" 2>/dev/null)
        echo "  ✓ camera_configs/camera_${i}_full.json ($ZONE_COUNT zones)"
    else
        echo "  ✗ camera_configs/camera_${i}_full.json (not created)"
    fi
done

echo ""
echo "Next steps:"
echo "  1. Process videos with your new zone configurations:"
echo "     python vehicle_counting.py VIDEO --config camera_configs/camera_X_full.json"
echo ""
echo "  2. Batch process all videos:"
echo "     python batch_process_videos.py --csv data_challenge/Train.csv"
echo "     (Update batch_process_videos.py to use *_full.json configs)"
echo ""
echo "  3. View zone designer guide:"
echo "     cat ZONE_DESIGNER_GUIDE.md"
echo ""
