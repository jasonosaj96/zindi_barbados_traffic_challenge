#!/bin/bash
# Setup zones for all 4 cameras using the interactive zone designer

echo "=== Roundabout Camera Zone Setup ==="
echo "This will guide you through setting up zones for all 4 cameras"
echo ""
echo "Setup Order:"
echo "  Camera 1 (North) - Define: north_entry, north_exit, north_queue, north_circulating"
echo "  Camera 2 (East)  - Define: east_entry, east_exit, east_queue, east_circulating"
echo "  Camera 3 (South) - Define: south_entry, south_exit, south_queue, south_circulating"
echo "  Camera 4 (West)  - Define: west_entry, west_exit, west_queue, west_circulating"
echo ""
echo "Zone Colors:"
echo "  - Entry (Green): Vehicles entering from the approach"
echo "  - Exit (Red): Vehicles exiting to the approach"
echo "  - Queue (Yellow): Waiting area before entry"
echo "  - Circulating (Orange): Vehicles circulating in the roundabout"
echo ""
echo "================================================================"
echo ""

# Camera 1 (North)
echo "CAMERA 1 - NORTH APPROACH"
echo "========================="
echo "Please define the following zones for the NORTH approach:"
echo "  1. north_entry (Green)"
echo "  2. north_exit (Red)"
echo "  3. north_queue (Yellow)"
echo "  4. north_circulating (Orange)"
echo ""
if [ -d "_data/normanniles1" ]; then
    VIDEO=$(find _data/normanniles1 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) | awk '{print length, $0}' | sort -n | head -n 1 | cut -d' ' -f2-)
    if [ -n "$VIDEO" ]; then
        echo "Found video: $VIDEO"
        python 1_zone_designer.py "$VIDEO" --config camera_configs/camera_1_zones.json --direction north
    else
        echo "No video found in _data/normanniles1/"
    fi
else
    echo "Directory _data/normanniles1 not found"
fi

echo ""
echo "Camera 1 (North) setup complete!"
echo ""
read -p "Press Enter to continue to Camera 2 (East)..."
echo ""

# Camera 2 (East)
echo "CAMERA 2 - EAST APPROACH"
echo "========================"
echo "Please define the following zones for the EAST approach:"
echo "  1. east_entry (Green)"
echo "  2. east_exit (Red)"
echo "  3. east_queue (Yellow)"
echo "  4. east_circulating (Orange)"
echo ""
if [ -d "_data/normanniles2" ]; then
    VIDEO=$(find _data/normanniles2 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) | awk '{print length, $0}' | sort -n | head -n 1 | cut -d' ' -f2-)
    if [ -n "$VIDEO" ]; then
        echo "Found video: $VIDEO"
        python 1_zone_designer.py "$VIDEO" --config camera_configs/camera_2_zones.json --direction east
    else
        echo "No video found in _data/normanniles2/"
    fi
else
    echo "Directory _data/normanniles2 not found"
fi

echo ""
echo "Camera 2 (East) setup complete!"
echo ""
read -p "Press Enter to continue to Camera 3 (South)..."
echo ""

# Camera 3 (South)
echo "CAMERA 3 - SOUTH APPROACH"
echo "========================="
echo "Please define the following zones for the SOUTH approach:"
echo "  1. south_entry (Green)"
echo "  2. south_exit (Red)"
echo "  3. south_queue (Yellow)"
echo "  4. south_circulating (Orange)"
echo ""
if [ -d "_data/normanniles3" ]; then
    VIDEO=$(find _data/normanniles3 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) | awk '{print length, $0}' | sort -n | head -n 1 | cut -d' ' -f2-)
    if [ -n "$VIDEO" ]; then
        echo "Found video: $VIDEO"
        python 1_zone_designer.py "$VIDEO" --config camera_configs/camera_3_zones.json --direction south
    else
        echo "No video found in _data/normanniles3/"
    fi
else
    echo "Directory _data/normanniles3 not found"
fi

echo ""
echo "Camera 3 (South) setup complete!"
echo ""
read -p "Press Enter to continue to Camera 4 (West)..."
echo ""

# Camera 4 (West)
echo "CAMERA 4 - WEST APPROACH"
echo "========================"
echo "Please define the following zones for the WEST approach:"
echo "  1. west_entry (Green)"
echo "  2. west_exit (Red)"
echo "  3. west_queue (Yellow)"
echo "  4. west_circulating (Orange)"
echo ""
if [ -d "_data/normanniles4" ]; then
    VIDEO=$(find _data/normanniles4 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) | awk '{print length, $0}' | sort -n | head -n 1 | cut -d' ' -f2-)
    if [ -n "$VIDEO" ]; then
        echo "Found video: $VIDEO"
        python 1_zone_designer.py "$VIDEO" --config camera_configs/camera_4_zones.json --direction west
    else
        echo "No video found in _data/normanniles4/"
    fi
else
    echo "Directory _data/normanniles4 not found"
fi

echo ""
echo "Camera 4 (West) setup complete!"
echo ""
echo "================================================================"
echo "=== ALL CAMERAS SETUP COMPLETE ==="
echo "================================================================"
echo ""
echo "Configuration files saved:"
echo "  - camera_configs/camera_1_zones.json (North)"
echo "  - camera_configs/camera_2_zones.json (East)"
echo "  - camera_configs/camera_3_zones.json (South)"
echo "  - camera_configs/camera_4_zones.json (West)"
echo ""
