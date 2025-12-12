# Traffic Congestion Prediction - Model Architectures

This document describes the various deep learning models implemented for the Barbados traffic congestion prediction challenge.

## Overview

The notebook `train_ml_model_complete.ipynb` implements **5 different model architectures** for multi-directional traffic congestion prediction:

1. **LSTM** (Baseline)
2. **GRU** (Faster variant)
3. **Bidirectional LSTM** (Captures patterns in both directions)
4. **CNN-LSTM Hybrid** (Local + sequential features)
5. **Transformer** (Attention-based)

## Problem Setup

- **Input**: Rolling window sequences of shape `(samples, 6, 340)`
  - 6 time windows = 30 minutes of traffic history
  - 340 features per window (traffic metrics from 4 cameras)

- **Output**: Multi-class classification for 4 directions
  - North, East, South, West entry congestion
  - 4 classes: Free Flowing, Light Delay, Moderate Delay, Heavy Delay

- **Dataset**: 108 training sequences, 28 test sequences

## Model Architectures

### 1. LSTM Model (Baseline)

**Architecture:**
```
Input (6, 340)
  ↓
LSTM(128) + Dropout(0.3)
  ↓
LSTM(64) + Dropout(0.3)
  ↓
Dense(128, relu) + Dropout(0.2)
  ↓
Dense(64, relu)
  ↓
Output (4 directions × 4 classes)
```

**Strengths:**
- Captures long-term temporal dependencies
- Standard baseline for sequence modeling
- Good at learning temporal patterns in traffic flow

**Parameters:** ~350K

### 2. GRU Model

**Architecture:**
```
Input (6, 340)
  ↓
GRU(128) + Dropout(0.3)
  ↓
GRU(64) + Dropout(0.3)
  ↓
Dense(128, relu) + Dropout(0.2)
  ↓
Dense(64, relu)
  ↓
Output (4 directions × 4 classes)
```

**Strengths:**
- Faster training than LSTM (fewer parameters)
- Similar performance to LSTM in many cases
- Simpler gating mechanism

**Parameters:** ~290K

**Use when:** You want faster training with comparable performance

### 3. Bidirectional LSTM

**Architecture:**
```
Input (6, 340)
  ↓
BiLSTM(64) + Dropout(0.3)
  ↓
BiLSTM(32) + Dropout(0.3)
  ↓
Dense(128, relu) + Dropout(0.2)
  ↓
Dense(64, relu)
  ↓
Output (4 directions × 4 classes)
```

**Strengths:**
- Processes sequences in both forward and backward directions
- Can capture future context (within the 30-minute window)
- Better feature learning from complete temporal context

**Parameters:** ~310K

**Use when:** You want to leverage both past and future patterns

### 4. CNN-LSTM Hybrid

**Architecture:**
```
Input (6, 340)
  ↓
Conv1D(64, k=3) + BatchNorm
  ↓
Conv1D(64, k=3) + BatchNorm
  ↓
MaxPooling1D(2)
  ↓
LSTM(64) + Dropout(0.3)
  ↓
LSTM(32) + Dropout(0.3)
  ↓
Dense(128, relu) + Dropout(0.2)
  ↓
Dense(64, relu)
  ↓
Output (4 directions × 4 classes)
```

**Strengths:**
- CNN layers extract local temporal features
- LSTM layers model long-term dependencies
- Combines spatial and temporal feature learning
- Good for detecting short-term patterns + long-term trends

**Parameters:** ~280K

**Use when:** You have both local patterns and long-term trends

### 5. Transformer Model

**Architecture:**
```
Input (6, 340)
  ↓
Multi-Head Attention (4 heads, dim=64)
  ↓
Layer Normalization + Residual
  ↓
Feed-Forward Network (128)
  ↓
Layer Normalization + Residual
  ↓
[Repeat 2x]
  ↓
Global Average Pooling
  ↓
Dense(128, relu) + Dropout(0.2)
  ↓
Dense(64, relu)
  ↓
Output (4 directions × 4 classes)
```

**Strengths:**
- Self-attention mechanism captures dependencies across all time steps
- No recurrence = parallel processing
- Can learn complex temporal relationships
- State-of-the-art for many sequence tasks

**Parameters:** ~420K

**Use when:** You want maximum model capacity and have enough data

## Key Features

### Custom Loss Function

All models use a **masked categorical cross-entropy** loss that handles missing labels:

```python
def masked_categorical_crossentropy(y_true, y_pred):
    """Ignores -1 labels (missing data)"""
    mask = tf.not_equal(y_true, -1)
    # Only compute loss for valid labels
    return masked_loss
```

### Data Preprocessing

1. **Handle missing values**: Replace NaN with column means
2. **Standardization**: Zero mean, unit variance
3. **Temporal structure**: Preserve 6-window sequences

### Training Setup

- **Optimizer**: Adam (lr=0.001)
- **Batch size**: 16
- **Early stopping**: Patience=15 epochs
- **Learning rate reduction**: Factor=0.5, patience=5

### Evaluation Metrics

For each model and direction:
- **Accuracy** per direction
- **Precision, Recall, F1-score** per congestion class
- **Average accuracy** across all directions
- **Confusion matrices** for detailed error analysis

## How to Use

### 1. Run the Complete Notebook

```bash
jupyter notebook train_ml_model_complete.ipynb
```

This will:
1. Load the rolling window sequences
2. Preprocess the data
3. Train all 5 models
4. Evaluate and compare performance
5. Save trained models to `models/` directory

### 2. Load a Trained Model

```python
from tensorflow import keras
import joblib

# Load model
model = keras.models.load_model('models/lstm_traffic_model.keras',
                                custom_objects={'masked_categorical_crossentropy': masked_categorical_crossentropy})

# Load scaler
scaler = joblib.load('models/feature_scaler.pkl')

# Make predictions
y_pred = model.predict(X_test_scaled)
```

### 3. Use for Inference

```python
import numpy as np

# Preprocess new data
X_new_scaled = scaler.transform(X_new.reshape(-1, 340)).reshape(-1, 6, 340)

# Predict
predictions = model.predict(X_new_scaled)
predicted_classes = np.argmax(predictions, axis=-1)

# Decode predictions
class_names = ['Free Flowing', 'Light Delay', 'Moderate Delay', 'Heavy Delay']
directions = ['North', 'East', 'South', 'West']

for i, direction in enumerate(directions):
    pred_class = predicted_classes[0, i]
    print(f"{direction}: {class_names[pred_class]}")
```

## Model Selection Guide

| Model | Training Speed | Performance | Memory | Best For |
|-------|---------------|-------------|--------|----------|
| LSTM | Medium | Baseline | Medium | General purpose |
| GRU | Fast | Good | Low | Fast iteration |
| BiLSTM | Medium | Good | Medium | Bidirectional context |
| CNN-LSTM | Fast | Very Good | Low | Local + temporal patterns |
| Transformer | Slow | Excellent | High | Maximum accuracy |

## Hyperparameter Tuning

Consider experimenting with:

1. **Window size**: 3 (15min), 6 (30min), 12 (60min)
2. **Model depth**: More/fewer layers
3. **Hidden units**: 32, 64, 128, 256
4. **Dropout rates**: 0.2, 0.3, 0.4, 0.5
5. **Learning rate**: 0.0001, 0.001, 0.01
6. **Batch size**: 8, 16, 32
7. **Attention heads** (Transformer): 2, 4, 8

## Advanced Techniques

### Ensemble Learning

Combine predictions from multiple models:

```python
# Simple averaging
ensemble_pred = (pred_lstm + pred_gru + pred_bilstm) / 3

# Weighted averaging (based on validation performance)
weights = [0.3, 0.25, 0.25, 0.15, 0.05]  # LSTM, GRU, BiLSTM, CNN-LSTM, Transformer
ensemble_pred = sum(w * p for w, p in zip(weights, predictions))
```

### Data Augmentation

For small datasets, consider:
- **Temporal jittering**: Add small time shifts
- **Noise injection**: Add Gaussian noise to features
- **Mixup**: Blend samples

### Transfer Learning

If you get more data:
- Fine-tune pre-trained models
- Freeze early layers, train later layers
- Progressive unfreezing

## Troubleshooting

### Low accuracy?
1. Check data quality and preprocessing
2. Try different window sizes
3. Increase model capacity
4. Add more regularization (dropout)
5. Use ensemble methods

### Overfitting?
1. Increase dropout rates
2. Add L2 regularization
3. Reduce model size
4. Get more training data
5. Use data augmentation

### Slow training?
1. Use GRU instead of LSTM
2. Reduce batch size
3. Use fewer layers
4. Reduce hidden units
5. Use mixed precision training

## Results Interpretation

The notebook generates:

1. **Training curves**: Loss and accuracy over epochs
2. **Confusion matrices**: Per-direction error analysis
3. **Classification reports**: Precision/recall/F1 per class
4. **Comparison plots**: Model performance comparison

Look for:
- **Convergence**: Training loss should decrease smoothly
- **Generalization**: Val loss should track train loss
- **Class balance**: Check if some classes are harder to predict
- **Direction differences**: Some directions may be easier

## Next Steps

1. **Experiment with architectures**: Modify layer sizes, depths
2. **Try different window sizes**: 3, 6, 12 windows
3. **Feature engineering**: Add more features to input data
4. **Ensemble models**: Combine best performers
5. **Cross-validation**: K-fold for robust evaluation
6. **Deployment**: Export to TFLite or ONNX for production

## References

- LSTM: Hochreiter & Schmidhuber (1997)
- GRU: Cho et al. (2014)
- Attention: Vaswani et al. (2017)
- Traffic Prediction: Comprehensive review by Vlahogianni et al. (2014)

## Files

- `train_ml_model_complete.ipynb`: Main training notebook
- `models/*.keras`: Trained model files
- `models/feature_scaler.pkl`: Data scaler for preprocessing
- `sequences/`: Input data files

---

**Happy modeling!**
