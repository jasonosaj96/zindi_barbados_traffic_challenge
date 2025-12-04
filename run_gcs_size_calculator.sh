#!/bin/bash

# Shell script to run the GCS size calculator for Barbados Traffic Challenge
# Usage: ./run_gcs_size_calculator.sh [OPTIONS]
#
# Examples:
#   ./run_gcs_size_calculator.sh                          # Analyze both buckets
#   ./run_gcs_size_calculator.sh --dataset small          # Only small dataset
#   ./run_gcs_size_calculator.sh --dataset full           # Only full dataset
#   ./run_gcs_size_calculator.sh --camera 1 --breakdown   # Camera 1 with breakdown
#   ./run_gcs_size_calculator.sh --bucket my-bucket       # Custom bucket

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/calculate_gcs_size.py"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Check if virtual environment exists and activate it
if [ -d "${SCRIPT_DIR}/.venv" ]; then
    echo "✓ Activating virtual environment..."
    source "${SCRIPT_DIR}/.venv/bin/activate"
else
    echo "⚠ No virtual environment found at ${SCRIPT_DIR}/.venv"
fi

# Run the Python script with all passed arguments
echo ""
echo "Running GCS size calculator..."
python "$PYTHON_SCRIPT" "$@"
