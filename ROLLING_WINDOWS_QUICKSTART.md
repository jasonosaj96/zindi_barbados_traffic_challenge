# Rolling Windows Quick Start Guide

This guide shows you how to create and use rolling window sequences for your traffic congestion prediction model.

## What You Have Now

Three new files for rolling window sequence modeling:

1. **[create_rolling_window_dataset.py](create_rolling_window_dataset.py)** - Creates sequences from your training data
2. **[rolling_window_example.py](rolling_window_example.py)** - Example LSTM model training
3. **[docs/ROLLING_WINDOWS_README.md](docs/ROLLING_WINDOWS_README.md)** - Complete documentation

## Quick Start

### Step 1: Create Rolling Window Sequences

```bash
# Activate your virtual environment
source .venv/bin/activate

# Create sequences (6 windows = 30 minutes of history)
python create_rolling_window_dataset.py \
    --input training_dataset.csv \
    --output-dir sequences \
    --window-size 6 \
    --stride 1 \
    --test-size 0.2 \
    --min-valid-targets 1
```

**What this does:**
- Reads `training_dataset.csv` (4009 rows × 358 columns)
- Creates sequences of 6 consecutive 5-minute windows
- Each sequence = 30 minutes of traffic history
- Outputs training/test splits ready for model training

**Output:**
```
sequences/
├── X_train.npy          # (n_samples, 6, 340) - training features
├── X_test.npy           # Test features
├── y_train.npy          # (n_samples, 4) - training targets
├── y_test.npy           # Test targets
├── train_metadata.csv   # Timestamps and sequence info
├── test_metadata.csv
├── feature_columns.csv  # List of 340 feature names
├── target_columns.csv   # [north, east, south, west]_entry_congestion
├── encoding_map.csv     # Congestion level → integer mapping
└── dataset_info.csv     # Summary statistics
```

### Step 2: Train a Model (Example)

```bash
# Train example LSTM model
python rolling_window_example.py
```

**What this does:**
- Loads sequences from `sequences/` directory
- Normalizes features
- Builds and trains an LSTM model
- Evaluates on test set
- Saves model to `traffic_lstm_model.keras`

### Step 3: Use Sequences in Your Own Model

```python
import numpy as np

# Load sequences
X_train = np.load('sequences/X_train.npy')  # Shape: (samples, 6, 340)
y_train = np.load('sequences/y_train.npy')  # Shape: (samples, 4)

X_test = np.load('sequences/X_test.npy')
y_test = np.load('sequences/y_test.npy')

print(f"Training data: {X_train.shape}")
# Example output: (2661, 6, 340)
#                  ^^^^  ^  ^^^
#                  |     |  └─ 340 features per window
#                  |     └──── 6 consecutive windows (30 min)
#                  └────────── 2661 training sequences

print(f"Targets: {y_train.shape}")
# Example output: (2661, 4)
#                  ^^^^  ^
#                  |     └─ 4 directions (north, east, south, west)
#                  └─────── 2661 sequences
```

## Understanding the Approach

### Current Structure (Before)
```
Each row in training_dataset.csv = ONE 5-minute window
Row 1: [features at t=0] → predict congestion at t=0
```

### Rolling Window Structure (After)
```
Each sequence = MULTIPLE consecutive windows
Sequence 1: [t=0, t=5, t=10, t=15, t=20, t=25] → predict congestion at t=25
Sequence 2: [t=5, t=10, t=15, t=20, t=25, t=30] → predict congestion at t=30
```

**Benefits:**
- Capture temporal patterns (trends, acceleration)
- Learn from longer history (30 min vs 5 min)
- Better for LSTM/GRU/Transformer models
- More context = better predictions

## Target Encoding

Your congestion levels are automatically encoded as:

| Congestion Level | Encoded Value |
|-----------------|---------------|
| free flowing    | 0             |
| light delay     | 1             |
| moderate delay  | 2             |
| heavy delay     | 3             |
| Missing (NaN)   | -1            |

## Key Parameters

### --window-size (default: 6)
How many consecutive 5-minute windows per sequence.

```bash
--window-size 3   # 15 minutes (shorter sequences, faster training)
--window-size 6   # 30 minutes (good balance)
--window-size 12  # 60 minutes (more context, fewer sequences)
```

### --stride (default: 1)
How much to shift the window between sequences.

```bash
--stride 1   # Maximum overlap (more sequences)
--stride 2   # 50% overlap
--stride 3   # Minimal overlap (less data, faster)
```

### --min-valid-targets (default: 3)
Minimum number of labeled directions required per sequence.

```bash
--min-valid-targets 1   # Keep sequences with at least 1 labeled direction
--min-valid-targets 3   # Require 3 of 4 directions labeled (default)
--min-valid-targets 4   # Require all 4 directions labeled (strictest)
```

## Common Use Cases

### Use Case 1: Maximum Training Data
```bash
python create_rolling_window_dataset.py \
    --window-size 6 \
    --stride 1 \
    --min-valid-targets 1 \
    --test-size 0.15
```
**Result:** Most sequences, accepts partially labeled data

### Use Case 2: Long-term Patterns
```bash
python create_rolling_window_dataset.py \
    --window-size 12 \
    --stride 2 \
    --min-valid-targets 2
```
**Result:** 60-minute sequences for capturing rush hour dynamics

### Use Case 3: Fast Experimentation
```bash
python create_rolling_window_dataset.py \
    --window-size 3 \
    --stride 2 \
    --test-size 0.1
```
**Result:** Smaller sequences, faster training iterations

## Example: Custom PyTorch Model

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Load sequences
X_train = np.load('sequences/X_train.npy')
y_train = np.load('sequences/y_train.npy')

# Create PyTorch dataset
class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create data loader
train_dataset = TrafficDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define LSTM model
class TrafficLSTM(nn.Module):
    def __init__(self, n_features, n_targets=4, n_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 128, num_layers=2,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(128, n_targets * n_classes)
        self.n_targets = n_targets
        self.n_classes = n_classes

    def forward(self, x):
        # x: (batch, seq_len, features)
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])  # Last hidden state
        return out.view(-1, self.n_targets, self.n_classes)

# Initialize model
model = TrafficLSTM(n_features=X_train.shape[2])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(50):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)  # (batch, 4, 4)

        # Calculate loss for each direction
        loss = 0
        for i in range(4):  # 4 directions
            mask = batch_y[:, i] != -1  # Ignore missing values
            if mask.sum() > 0:
                loss += criterion(outputs[mask, i], batch_y[mask, i])

        loss.backward()
        optimizer.step()
```

## Troubleshooting

### "No sequences created"
- Try smaller `--window-size` (e.g., 3)
- Reduce `--min-valid-targets` (e.g., 1)
- Check if your `training_dataset.csv` has consecutive timestamps

### "Very few sequences"
- Your data may have temporal gaps
- Reduce `--window-size` to bridge smaller gaps
- Increase `--stride` (but this gives you less data)

### "Out of memory"
- Reduce `--window-size`
- Use smaller batch size during training
- Increase `--stride` to create fewer sequences

## What's Next?

1. **Create your sequences** with the settings that work best for your data
2. **Experiment with different window sizes** (3, 6, 12)
3. **Try different model architectures:**
   - LSTM (example provided)
   - GRU (similar to LSTM, often faster)
   - Transformer (for attention-based learning)
   - Bidirectional LSTM
4. **Feature engineering:**
   - Add more features to `training_dataset.csv`
   - The sequence creator will automatically include them
5. **Hyperparameter tuning:**
   - Window size
   - Model depth and width
   - Learning rate
   - Dropout rates

## Files Created

| File | Purpose |
|------|---------|
| `create_rolling_window_dataset.py` | Main script to create sequences |
| `rolling_window_example.py` | Example LSTM training pipeline |
| `docs/ROLLING_WINDOWS_README.md` | Detailed documentation |
| `ROLLING_WINDOWS_QUICKSTART.md` | This file (quick reference) |

## Getting Help

For detailed documentation, see [docs/ROLLING_WINDOWS_README.md](docs/ROLLING_WINDOWS_README.md)

For the full processing pipeline, see [docs/PARALLEL_PROCESSING_README.md](docs/PARALLEL_PROCESSING_README.md)

## Summary

You now have a complete pipeline for:
1. ✓ Creating temporal sequences from your traffic data
2. ✓ Encoding categorical congestion levels
3. ✓ Temporal train/test splitting
4. ✓ Example LSTM model for multi-output prediction
5. ✓ Evaluation and metrics

The rolling window approach will help you utilize more temporal information and improve your traffic congestion predictions!
