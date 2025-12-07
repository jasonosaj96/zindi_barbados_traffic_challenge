# Rolling Window Sequences for Traffic Prediction

This guide explains how to create and use rolling window sequences for temporal modeling of traffic congestion.

## Overview

The rolling window approach transforms your training data from individual 5-minute snapshots into **sequences** that capture temporal patterns. This is essential for models like LSTM, GRU, or Transformers that learn from time series.

### Current Data Structure

Your `training_dataset.csv` contains:
- **4009 rows** (one per 5-minute window)
- **358 columns** (features + targets)
- Each row = one 5-minute aggregation across 4 cameras
- Already includes temporal features (lags, rolling means)

### What Rolling Windows Add

Instead of predicting from a single 5-minute window, rolling windows create sequences:

```
Before (single window):
Row 1: [features at t=0] → predict congestion at t=0

After (sequence of windows):
Sequence 1: [features at t=0, t=5, t=10, t=15, t=20, t=25] → predict congestion at t=25
Sequence 2: [features at t=5, t=10, t=15, t=20, t=25, t=30] → predict congestion at t=30
...
```

**Key benefits:**
- Capture longer-term temporal patterns
- Model learns sequence-to-prediction mapping
- Better context for congestion prediction
- Suitable for LSTM/Transformer architectures

## Quick Start

### 1. Create Sequences

```bash
# Create sequences with default settings (6 windows = 30 minutes)
python create_rolling_window_dataset.py \
    --input training_dataset.csv \
    --output-dir sequences \
    --window-size 6 \
    --stride 1 \
    --test-size 0.2
```

### 2. Train a Model

```bash
# Train example LSTM model
python rolling_window_example.py
```

### 3. Load Sequences in Your Code

```python
import numpy as np

# Load sequences
X_train = np.load('sequences/X_train.npy')  # (samples, 6, 358)
y_train = np.load('sequences/y_train.npy')  # (samples, 4)

print(f"X_train shape: {X_train.shape}")
# Output: (3207, 6, 358)
#         ^^^^  ^  ^^^
#         |     |  └─ 358 features per window
#         |     └──── 6 consecutive windows (30 minutes)
#         └────────── 3207 training sequences
```

## Parameters Explained

### Window Size

Controls how many consecutive 5-minute windows are included in each sequence.

```bash
--window-size 6   # 6 × 5 min = 30 minutes of history
--window-size 12  # 12 × 5 min = 60 minutes of history
--window-size 3   # 3 × 5 min = 15 minutes of history
```

**Recommended:**
- `6` (30 min): Good balance between context and data efficiency
- `12` (60 min): Better for capturing rush hour patterns
- `3` (15 min): Faster training, less context

**Trade-off:** Larger windows = more context but fewer sequences and slower training.

### Stride

Controls overlap between consecutive sequences.

```bash
--stride 1   # No overlap (default)
--stride 2   # 50% overlap
--stride 3   # 33% overlap
```

**Example with stride=1:**
```
Sequence 1: [t=0, t=5, t=10, t=15, t=20, t=25]
Sequence 2: [t=5, t=10, t=15, t=20, t=25, t=30]  ← starts 1 step later
Sequence 3: [t=10, t=15, t=20, t=25, t=30, t=35]
```

**Example with stride=2:**
```
Sequence 1: [t=0, t=5, t=10, t=15, t=20, t=25]
Sequence 2: [t=10, t=15, t=20, t=25, t=30, t=35]  ← starts 2 steps later
Sequence 3: [t=20, t=25, t=30, t=35, t=40, t=45]
```

**Trade-off:** Smaller stride = more sequences (more training data) but higher overlap.

### Min Valid Targets

Minimum number of non-null target labels required per sequence.

```bash
--min-valid-targets 3   # Require at least 3 of 4 directions labeled
--min-valid-targets 4   # Require all 4 directions labeled (stricter)
--min-valid-targets 1   # Accept if at least 1 direction labeled (lenient)
```

**Recommended:** `3` (default) - balances data quantity and quality.

### Test Size

Fraction of data reserved for testing (temporal split, no shuffling).

```bash
--test-size 0.2   # 80% train, 20% test
--test-size 0.15  # 85% train, 15% test
```

## Output Files

After running `create_rolling_window_dataset.py`, you'll get:

```
sequences/
├── X_train.npy              # Training features (n_samples, window_size, n_features)
├── X_test.npy               # Test features
├── y_train.npy              # Training targets (n_samples, 4)
├── y_test.npy               # Test targets
├── train_metadata.csv       # Sequence metadata (timestamps, IDs)
├── test_metadata.csv
├── feature_columns.csv      # List of feature names
├── target_columns.csv       # List of target names (north/east/south/west)
├── encoding_map.csv         # Congestion level → integer mapping
└── dataset_info.csv         # Summary statistics
```

## Target Encoding

Congestion levels are encoded as integers:

| Congestion Level | Encoded Value |
|-----------------|---------------|
| free flowing    | 0             |
| light delay     | 1             |
| moderate delay  | 2             |
| heavy delay     | 3             |
| Missing (NaN)   | -1            |

**Note:** Missing values (-1) should be masked during loss calculation.

## Data Split Strategy

The script uses **temporal splitting** (not random):

1. Sort all sequences by end timestamp
2. First 80% → training set
3. Last 20% → test set

**Why temporal?** Ensures the model is tested on future data it hasn't seen, simulating real-world deployment.

```
Timeline: ═══════════════════════════════════════════>
          [    Training (80%)    ][ Test (20%) ]
                                 ↑
                           Split point (temporal)
```

## Usage Examples

### Example 1: Default Settings

```bash
python create_rolling_window_dataset.py
```

Creates sequences with:
- 6 windows per sequence (30 minutes)
- Stride of 1 (no overlap)
- 80/20 train/test split
- Minimum 3 valid targets

### Example 2: Longer Sequences

```bash
python create_rolling_window_dataset.py \
    --window-size 12 \
    --stride 2 \
    --test-size 0.15
```

Creates:
- 12 windows per sequence (60 minutes)
- Stride of 2 (50% overlap → more data)
- 85/15 train/test split

### Example 3: Short Sequences (Faster Training)

```bash
python create_rolling_window_dataset.py \
    --window-size 3 \
    --stride 1 \
    --min-valid-targets 2
```

Creates:
- 3 windows per sequence (15 minutes)
- Relaxed target requirement (2 of 4 directions)

### Example 4: Maximum Data Augmentation

```bash
python create_rolling_window_dataset.py \
    --window-size 6 \
    --stride 1 \
    --min-valid-targets 1 \
    --test-size 0.15
```

Maximizes training data by:
- Small stride (more sequences)
- Lenient target requirement
- Smaller test set

## Using Sequences in PyTorch

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TrafficSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Load data
X_train = np.load('sequences/X_train.npy')
y_train = np.load('sequences/y_train.npy')

# Create dataset
train_dataset = TrafficSequenceDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Example LSTM model
class TrafficLSTM(torch.nn.Module):
    def __init__(self, n_features, n_targets=4, n_classes=4):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_features, 128, num_layers=2,
                                    batch_first=True, dropout=0.2)
        self.fc = torch.nn.Linear(128, n_targets * n_classes)
        self.n_targets = n_targets
        self.n_classes = n_classes

    def forward(self, x):
        # x: (batch, seq_len, features)
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])  # Use last hidden state
        return out.view(-1, self.n_targets, self.n_classes)

model = TrafficLSTM(n_features=X_train.shape[2])
```

## Using Sequences in TensorFlow/Keras

See [rolling_window_example.py](../rolling_window_example.py) for a complete example.

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Load data
X_train = np.load('sequences/X_train.npy')
y_train = np.load('sequences/y_train.npy')

# Build model
inputs = keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = layers.LSTM(128, return_sequences=True)(inputs)
x = layers.LSTM(64)(x)

# Multi-output classification (4 directions)
outputs = [
    layers.Dense(4, activation='softmax', name=f'direction_{i}')(x)
    for i in range(4)
]

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train,
    [y_train[:, i] for i in range(4)],  # Split targets
    epochs=50,
    batch_size=32,
    validation_split=0.2
)
```

## Expected Sequence Counts

With 4009 rows in `training_dataset.csv`:

| Window Size | Stride | Approx Sequences |
|-------------|--------|------------------|
| 3           | 1      | ~4000            |
| 6           | 1      | ~4000            |
| 12          | 1      | ~4000            |
| 6           | 2      | ~2000            |
| 12          | 2      | ~2000            |

**Note:** Actual counts will be lower due to:
- Temporal gaps (sequences can't span different days)
- Missing targets (filtered by `--min-valid-targets`)

## Handling Missing Values

### In Features (X)

- Script normalizes features and replaces NaN/inf with 0
- Models with masking layers can handle this

### In Targets (y)

- Missing targets encoded as -1
- Filter during evaluation: `valid_mask = (y_test != -1)`
- For multi-task learning, use sample weighting or masked loss

```python
# Keras example with masked loss
import tensorflow as tf

def masked_categorical_crossentropy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    loss = keras.losses.sparse_categorical_crossentropy(
        tf.where(mask == 1, y_true, 0),  # Replace -1 with 0
        y_pred
    )
    return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
```

## Performance Tips

### Memory Optimization

If you run out of memory:

```bash
# Reduce window size
python create_rolling_window_dataset.py --window-size 3

# Increase stride (less overlap)
python create_rolling_window_dataset.py --stride 3

# Use smaller batch size during training
```

### Training Speed

For faster iteration:

```bash
# Create smaller test set
python create_rolling_window_dataset.py --test-size 0.1

# Use fewer windows
python create_rolling_window_dataset.py --window-size 3

# Reduce stride for less data
python create_rolling_window_dataset.py --stride 2
```

### Model Performance

For better accuracy:

```bash
# Longer sequences capture more patterns
python create_rolling_window_dataset.py --window-size 12

# More data with overlapping sequences
python create_rolling_window_dataset.py --stride 1

# Stricter target requirements (higher quality)
python create_rolling_window_dataset.py --min-valid-targets 4
```

## Common Issues

### Issue: "No sequences created"

**Cause:** Window size too large or too many missing targets.

**Solution:**
```bash
# Reduce window size
python create_rolling_window_dataset.py --window-size 3

# Relax target requirement
python create_rolling_window_dataset.py --min-valid-targets 1
```

### Issue: "Very few sequences created"

**Cause:** Large temporal gaps in your data.

**Solution:**
- Check your `training_dataset.csv` for consecutive timestamps
- Reduce `window_size` to bridge smaller gaps
- Accept shorter sequences

### Issue: "Out of memory during training"

**Solution:**
```python
# Use batch training
from tensorflow.keras.callbacks import ReduceLROnPlateau

model.fit(
    X_train, y_train,
    batch_size=16,  # Reduce batch size
    epochs=50,
    callbacks=[ReduceLROnPlateau()]
)
```

## Next Steps

1. **Create sequences:**
   ```bash
   python create_rolling_window_dataset.py
   ```

2. **Train example model:**
   ```bash
   python rolling_window_example.py
   ```

3. **Experiment with hyperparameters:**
   - Try different window sizes (3, 6, 12)
   - Adjust stride for more/less data
   - Test various model architectures (LSTM, GRU, Transformer)

4. **Feature engineering:**
   - Add more temporal features to `training_dataset.csv`
   - The rolling window script will automatically include them

5. **Production deployment:**
   - For real-time prediction, maintain a rolling buffer of recent windows
   - Feed the last N windows to your trained model

## See Also

- [PARALLEL_PROCESSING_README.md](PARALLEL_PROCESSING_README.md) - How to generate `training_dataset.csv`
- [rolling_window_example.py](../rolling_window_example.py) - Complete LSTM training example
- [create_rolling_window_dataset.py](../create_rolling_window_dataset.py) - Sequence creation script
