# Traffic Data Automation Guide

This guide explains how to use the automated pipeline for downloading and processing traffic data from Google Cloud Storage.

## Overview

The automation system consists of three main components:

1. **download_and_process.py** - Downloads videos from GCS buckets
2. **parallel_process.py** - Processes videos in parallel with YOLOv8n
3. **automated_pipeline.sh** - Complete end-to-end pipeline

## Data Sources

Videos are available in two GCS buckets:

- **Small dataset** (re-encoded): `gs://brb-traffic/`
- **Full dataset** (>500GB): `gs://brb-traffic-full/`

## Prerequisites

### 1. Install Google Cloud SDK

**macOS:**
```bash
brew install --cask google-cloud-sdk
```

**Linux:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

**Windows:**
Download from: https://cloud.google.com/sdk/docs/install

### 2. Authenticate (if required)

```bash
gcloud auth login
```

For public buckets, authentication may not be required.

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Quick Test (Recommended for First Run)

Test the pipeline with a small sample:

```bash
./quick_test.sh
```

This downloads and processes just 5 files per camera to verify everything works.

### Option 2: Complete Automated Pipeline

Run the entire pipeline with one command:

```bash
./automated_pipeline.sh
```

Or with options:

```bash
./automated_pipeline.sh --dataset small --cameras all --max-files 10
```

This will:
1. Download videos from GCS
2. Check/setup zone configurations
3. Process videos with YOLOv8n in parallel
4. Generate traffic analysis data

### Available Command Line Options

```bash
./automated_pipeline.sh --help
```

Options:
- `--dataset small|full` - Choose dataset (default: small)
- `--cameras all|1,2,3,4` - Select cameras (default: all)
- `--max-files N` - Limit downloads per camera (default: no limit)
- `--workers N` - Parallel workers (default: 4)
- `--start-date YYYY-MM-DD` - Filter by start date
- `--end-date YYYY-MM-DD` - Filter by end date

### Option 2: Step-by-Step

#### Step 1: Download Videos

Download from small dataset (all cameras):
```bash
python download_and_process.py --dataset small --cameras all
```

Download from full dataset (specific cameras):
```bash
python download_and_process.py --dataset full --cameras 1,2 --max-files 10
```

Download with date filter:
```bash
python download_and_process.py --cameras 1 --start-date 2025-10-01 --end-date 2025-10-31
```

#### Step 2: Setup Zone Configurations (if needed)

```bash
./setup_all_cameras.sh
```

#### Step 3: Process Videos

Process with parallel workers:
```bash
python parallel_process.py --workers 4
```

Process specific cameras only:
```bash
python parallel_process.py --cameras 1,2 --workers 2
```

## Configuration

### Command Line Arguments (Recommended)

Use command line arguments for explicit control:

```bash
./automated_pipeline.sh \
  --dataset small \
  --cameras 1,2 \
  --max-files 10 \
  --workers 4 \
  --start-date 2025-10-01 \
  --end-date 2025-10-31
```

### Environment Variables (Alternative)

Or use environment variables:

```bash
export DATASET=small          # small or full
export CAMERAS=all            # all or comma-separated (1,2,3,4)
export MAX_FILES=10           # Limit files per camera
export WORKERS=4              # Parallel processing workers
export START_DATE=2025-10-01  # Optional date filter
export END_DATE=2025-10-31    # Optional date filter

./automated_pipeline.sh
```

### Combining Both

Command line arguments override environment variables:

```bash
DATASET=full ./automated_pipeline.sh --dataset small --max-files 5
# Will use: dataset=small, max-files=5
```

### Camera Mapping

- Camera 1: normanniles1 (North)
- Camera 2: normanniles2 (East)
- Camera 3: normanniles3 (South)
- Camera 4: normanniles4 (West)

## Output Structure

After processing, your data directory will look like:

```
_data/
├── normanniles1/
│   ├── video1.mp4
│   ├── video1.counts.json
│   ├── video1.movements.json
│   └── video1.detections.json
├── normanniles2/
│   └── ...
├── normanniles3/
│   └── ...
└── normanniles4/
    └── ...
```

## Advanced Usage

### Download Only (No Processing)

```bash
python download_and_process.py --dataset small --cameras all --download-only
```

### Process Existing Videos

```bash
python parallel_process.py --data-dir _data --workers 8
```

### Limit Processing

Process only first N videos:
```bash
python parallel_process.py --max-videos 20
```

### Custom Output Directory

```bash
python download_and_process.py --dataset small --cameras all --output-dir my_data
python parallel_process.py --data-dir my_data
```

## Performance Tips

1. **Workers**: Set workers based on CPU cores (recommended: CPU count - 1)
   ```bash
   python parallel_process.py --workers 7  # For 8-core CPU
   ```

2. **Small Dataset First**: Start with the small dataset for faster iteration
   ```bash
   DATASET=small ./automated_pipeline.sh
   ```

3. **Batch Processing**: Process in batches with MAX_FILES
   ```bash
   MAX_FILES=50 ./automated_pipeline.sh
   ```

4. **Skip Processed Videos**: The scripts automatically skip already-processed videos

## Troubleshooting

### gsutil not found

Install Google Cloud SDK:
```bash
# macOS
brew install --cask google-cloud-sdk

# Linux
curl https://sdk.cloud.google.com | bash
```

### Permission denied

Make scripts executable:
```bash
chmod +x automated_pipeline.sh download_and_process.py parallel_process.py
```

### Out of disk space

Monitor disk usage:
```bash
du -sh _data/
```

The full dataset is >500GB. Use `--max-files` to limit downloads:
```bash
python download_and_process.py --dataset full --max-files 10
```

### Processing timeout

Increase timeout in `parallel_process.py` (default: 1 hour per video):
```python
timeout=3600  # Increase this value
```

## Monitoring

### Check Download Progress

```bash
watch -n 5 'du -sh _data/*'
```

### Check Processing Progress

```bash
watch -n 5 'find _data -name "*.counts.json" | wc -l'
```

### View Processing Summary

```bash
cat processing_summary.json | jq
```

## Examples

### Quick Test (5 Files Per Camera)

```bash
./quick_test.sh
```

### Small Sample for Development

```bash
./automated_pipeline.sh --max-files 10 --workers 2
```

### Process Recent Data for Camera 1

```bash
./automated_pipeline.sh \
  --dataset small \
  --cameras 1 \
  --start-date 2025-10-20 \
  --workers 2 \
  --max-files 20
```

### Download Full Dataset (Limited)

```bash
python download_and_process.py \
  --dataset full \
  --cameras all \
  --max-files 5 \
  --download-only
```

### Specific Cameras with Date Range

```bash
./automated_pipeline.sh \
  --cameras 1,3 \
  --start-date 2025-10-01 \
  --end-date 2025-10-15 \
  --max-files 50
```

### Parallel Processing All Cameras

```bash
python parallel_process.py \
  --cameras 1,2,3,4 \
  --workers 8 \
  --data-dir _data
```

### Environment Variables Style

```bash
DATASET=small CAMERAS=1,2 MAX_FILES=15 WORKERS=3 ./automated_pipeline.sh
```

## Data Format

### Counts JSON
```json
{
  "timestamp": "2025-10-26T10:30:00",
  "zones": {
    "north_entry": 45,
    "north_exit": 38,
    "north_queue": 12
  }
}
```

### Movements JSON
```json
{
  "movements": [
    {
      "id": 1,
      "path": ["north_entry", "circulating", "east_exit"],
      "duration": 15.2
    }
  ]
}
```

## Next Steps

After processing:

1. **Analyze Results**
   ```bash
   python analyze_counts.py
   ```

2. **Generate Visualizations**
   ```bash
   python batch_process_videos.py
   ```

3. **Create Submission**
   ```bash
   python create_submission.py
   ```

## Support

For issues or questions:
- Check existing GitHub issues
- Review error logs in `processing_summary.json`
- Verify zone configurations exist in `camera_configs/`
