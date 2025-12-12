# Quick Start Guide - Model Training

## TL;DR

```bash
# 1. Open the notebook
jupyter notebook train_ml_model_complete.ipynb

# 2. Run all cells (Kernel > Restart & Run All)

# 3. Models will be saved to models/ directory
```

That's it! The notebook trains 5 different models and compares their performance.

---

## What You Get

### 5 Pre-Configured Models

1. **LSTM** - Classic sequence model
2. **GRU** - Faster LSTM variant
3. **BiLSTM** - Bidirectional learning
4. **CNN-LSTM** - Hybrid architecture
5. **Transformer** - Attention-based (state-of-the-art)

### Automatic Evaluation

- Per-direction accuracy
- Confusion matrices
- Classification reports
- Model comparison plots
- Training history visualization

### Saved Outputs

```
models/
├── lstm_traffic_model.keras
├── gru_traffic_model.keras
├── bilstm_traffic_model.keras
├── cnn_lstm_traffic_model.keras
├── transformer_traffic_model.keras
└── feature_scaler.pkl
```

---

## Quick Training Commands

### Train All Models (Recommended)

```python
# Just run the notebook - it handles everything!
jupyter notebook train_ml_model_complete.ipynb
```

### Train Single Model

```python
# Copy one model section from the notebook
# Example: LSTM only

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load data
X_train = np.load('sequences/X_train.npy')
y_train = np.load('sequences/y_train.npy')
X_test = np.load('sequences/X_test.npy')
y_test = np.load('sequences/y_test.npy')

# Preprocess (use code from notebook)
# ... preprocessing code ...

# Create LSTM model
def create_lstm_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(16, activation='linear')(x)
    outputs = layers.Reshape((4, 4))(outputs)
    outputs = layers.Activation('softmax')(outputs)
    return keras.Model(inputs=inputs, outputs=outputs)

# Train
model = create_lstm_model((6, 340))
model.compile(optimizer='adam', loss=masked_categorical_crossentropy)
model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test),
          epochs=100, batch_size=16)
```

---

## Making Predictions

### Option 1: Use Prediction Script

```bash
# Basic prediction
python predict_with_model.py \
    --model models/lstm_traffic_model.keras \
    --data sequences/X_test.npy \
    --output predictions.csv

# With evaluation
python predict_with_model.py \
    --model models/lstm_traffic_model.keras \
    --data sequences/X_test.npy \
    --ground-truth sequences/y_test.npy \
    --metadata sequences/test_metadata.csv \
    --output predictions.csv

# Show full probabilities
python predict_with_model.py \
    --model models/lstm_traffic_model.keras \
    --data sequences/X_test.npy \
    --output predictions.csv \
    --show-probabilities
```

### Option 2: Python Code

```python
from tensorflow import keras
import numpy as np
import joblib

# Load model
model = keras.models.load_model('models/lstm_traffic_model.keras',
    custom_objects={'masked_categorical_crossentropy': masked_categorical_crossentropy})

# Load scaler
scaler = joblib.load('models/feature_scaler.pkl')

# Preprocess your data
X_scaled = preprocess_data(X_new, scaler)  # Use function from notebook

# Predict
predictions = model.predict(X_scaled)
predicted_classes = np.argmax(predictions, axis=-1)

# Interpret
directions = ['North', 'East', 'South', 'West']
classes = ['Free Flowing', 'Light Delay', 'Moderate Delay', 'Heavy Delay']

for i, direction in enumerate(directions):
    pred = predicted_classes[0, i]
    print(f"{direction}: {classes[pred]}")
```

---

## Common Tasks

### Compare Models

```python
# The notebook automatically generates comparison plots
# Look for the "Model Comparison" section at the end
```

### Save/Load Models

```python
# Save (done automatically by notebook)
model.save('my_model.keras')

# Load
from tensorflow import keras
model = keras.models.load_model('my_model.keras',
    custom_objects={'masked_categorical_crossentropy': masked_categorical_crossentropy})
```

### Get Model Performance

```python
# Use evaluate_model function from notebook
results = evaluate_model(model, X_test_scaled, y_test, "My Model")
print(f"Average Accuracy: {results['avg_accuracy']:.4f}")
```

### Visualize Training

```python
# Use plot_training_history from notebook
plot_training_history(history, "Model Name")
```

---

## Troubleshooting

### GPU Not Detected?

```python
# Check GPU availability
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Force CPU if needed
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Out of Memory?

```python
# Reduce batch size in training
model.fit(..., batch_size=8)  # Instead of 16

# Or use smaller model
# GRU instead of LSTM
# Fewer layers/units
```

### Model Not Learning?

```python
# Check learning rate
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

# Check data preprocessing
print("X_train min/max:", X_train_scaled.min(), X_train_scaled.max())
print("X_train mean/std:", X_train_scaled.mean(), X_train_scaled.std())

# Check labels
print("Unique labels:", np.unique(y_train))
print("Label distribution:", np.bincount(y_train[y_train != -1].flatten()))
```

### Predictions All Same Class?

```python
# Class imbalance - use class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced',
    classes=np.unique(y_train[y_train != -1]),
    y=y_train[y_train != -1].flatten())

# Pass to fit
model.fit(..., class_weight=dict(enumerate(class_weights)))
```

---

## Performance Tips

### Faster Training

1. Use GRU instead of LSTM
2. Reduce model size (fewer layers/units)
3. Increase batch size
4. Use mixed precision training:
```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

### Better Accuracy

1. Try Transformer model (best accuracy)
2. Use ensemble of models
3. Tune hyperparameters
4. Increase model size
5. Add more training data (adjust window creation parameters)

### Memory Efficiency

1. Use batch size = 8 or 16
2. Reduce model size
3. Use gradient checkpointing
4. Clear session between models:
```python
from tensorflow.keras import backend as K
K.clear_session()
```

---

## Next Steps

1. **Run the notebook** to train all models
2. **Compare results** in the final comparison section
3. **Choose best model** based on accuracy/speed trade-off
4. **Make predictions** using the prediction script
5. **Tune hyperparameters** if needed
6. **Deploy** best model for production

---

## File Reference

| File | Purpose |
|------|---------|
| `train_ml_model_complete.ipynb` | Main training notebook |
| `predict_with_model.py` | Prediction utility script |
| `MODELS_README.md` | Detailed architecture documentation |
| `QUICK_START_MODELS.md` | This quick start guide |
| `sequences/*.npy` | Training/test data |
| `models/*.keras` | Trained models |
| `models/feature_scaler.pkl` | Data preprocessor |

---

## Questions?

- **Model architectures**: See [MODELS_README.md](MODELS_README.md)
- **Data preparation**: See [ROLLING_WINDOWS_QUICKSTART.md](ROLLING_WINDOWS_QUICKSTART.md)
- **Full pipeline**: See [docs/PARALLEL_PROCESSING_README.md](docs/PARALLEL_PROCESSING_README.md)

Happy modeling! 🚗📊
