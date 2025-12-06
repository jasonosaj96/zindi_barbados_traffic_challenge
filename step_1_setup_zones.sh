#!/bin/bash
# Setup zones for all 4 cameras using the interactive zone designer

echo "=== Roundabout Camera Zone Setup ==="
echo "This will guide you through setting up zones for all 4 cameras"
echo ""
echo "Setup Order:"
echo "  Camera 1 - Define: enter, exit, circulating_left, circulating_right"
echo "  Camera 2 - Define: enter, exit, circulating_left, circulating_right"
echo "  Camera 3 - Define: enter, exit, circulating_left, circulating_right"
echo "  Camera 4 - Define: enter, exit, circulating_left, circulating_right"
echo ""
echo "Zone Colors:"
echo "  - enter (Green): Vehicles entering the roundabout"
echo "  - exit (Red): Vehicles exiting the roundabout"
echo "  - circulating_left (Orange): Vehicles circulating on left side"
echo "  - circulating_right (Magenta): Vehicles circulating on right side"
echo ""
echo "================================================================"
echo ""

# Camera 1
echo "CAMERA 1"
echo "========"
echo "Please define the following zones for Camera 1:"
echo "  1. enter (Green)"
echo "  2. exit (Red)"
echo "  3. circulating_left (Orange)"
echo "  4. circulating_right (Magenta)"
echo ""
if [ -d "_data/normanniles1" ]; then
    VIDEO=$(find _data/normanniles1 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) | awk '{print length, $0}' | sort -n | head -n 1 | cut -d' ' -f2-)
    if [ -n "$VIDEO" ]; then
        echo "Found video: $VIDEO"
        python step_1_setup_zones.py "$VIDEO" --config camera_configs/camera_1_zones.json
    else
        echo "No video found in _data/normanniles1/"
    fi
else
    echo "Directory _data/normanniles1 not found"
fi

echo ""
echo "Camera 1 setup complete!"
echo ""
read -p "Press Enter to continue to Camera 2..."
echo ""

# Camera 2
echo "CAMERA 2"
echo "========"
echo "Please define the following zones for Camera 2:"
echo "  1. enter (Green)"
echo "  2. exit (Red)"
echo "  3. circulating_left (Orange)"
echo "  4. circulating_right (Magenta)"
echo ""
if [ -d "_data/normanniles2" ]; then
    VIDEO=$(find _data/normanniles2 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) | awk '{print length, $0}' | sort -n | head -n 1 | cut -d' ' -f2-)
    if [ -n "$VIDEO" ]; then
        echo "Found video: $VIDEO"
        python step_1_setup_zones.py "$VIDEO" --config camera_configs/camera_2_zones.json
    else
        echo "No video found in _data/normanniles2/"
    fi
else
    echo "Directory _data/normanniles2 not found"
fi

echo ""
echo "Camera 2 setup complete!"
echo ""
read -p "Press Enter to continue to Camera 3..."
echo ""

# Camera 3
echo "CAMERA 3"
echo "========"
echo "Please define the following zones for Camera 3:"
echo "  1. enter (Green)"
echo "  2. exit (Red)"
echo "  3. circulating_left (Orange)"
echo "  4. circulating_right (Magenta)"
echo ""
if [ -d "_data/normanniles3" ]; then
    VIDEO=$(find _data/normanniles3 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) | awk '{print length, $0}' | sort -n | head -n 1 | cut -d' ' -f2-)
    if [ -n "$VIDEO" ]; then
        echo "Found video: $VIDEO"
        python step_1_setup_zones.py "$VIDEO" --config camera_configs/camera_3_zones.json
    else
        echo "No video found in _data/normanniles3/"
    fi
else
    echo "Directory _data/normanniles3 not found"
fi

echo ""
echo "Camera 3 setup complete!"
echo ""
read -p "Press Enter to continue to Camera 4..."
echo ""

# Camera 4
echo "CAMERA 4"
echo "========"
echo "Please define the following zones for Camera 4:"
echo "  1. enter (Green)"
echo "  2. exit (Red)"
echo "  3. circulating_left (Orange)"
echo "  4. circulating_right (Magenta)"
echo ""
if [ -d "_data/normanniles4" ]; then
    VIDEO=$(find _data/normanniles4 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) | awk '{print length, $0}' | sort -n | head -n 1 | cut -d' ' -f2-)
    if [ -n "$VIDEO" ]; then
        echo "Found video: $VIDEO"
        python step_1_setup_zones.py "$VIDEO" --config camera_configs/camera_4_zones.json
    else
        echo "No video found in _data/normanniles4/"
    fi
else
    echo "Directory _data/normanniles4 not found"
fi

echo ""
echo "Camera 4 setup complete!"
echo ""
echo "================================================================"
echo "=== ALL CAMERAS SETUP COMPLETE ==="
echo "================================================================"
echo ""
echo "Configuration files saved:"
echo "  - camera_configs/camera_1_zones.json"
echo "  - camera_configs/camera_2_zones.json"
echo "  - camera_configs/camera_3_zones.json"
echo "  - camera_configs/camera_4_zones.json"
echo ""
echo "Each config contains 4 zones:"
echo "  • enter (Green)"
echo "  • exit (Red)"
echo "  • circulating_left (Orange)"
echo "  • circulating_right (Magenta)"
echo ""
