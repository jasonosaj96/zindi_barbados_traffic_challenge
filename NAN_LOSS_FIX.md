# Fixing NaN Loss Issue

## Problem

You're seeing `loss: nan` during training, which indicates the custom loss function is producing invalid (NaN) values.

## Root Cause

The issue is likely in the `masked_categorical_crossentropy` function. The built-in `tf.keras.losses.categorical_crossentropy` can produce NaN when:
1. Predictions contain values very close to 0 or 1
2. Division by very small numbers
3. Log of values near zero

## Solution

I've updated the loss function in the notebook with these improvements:

1. **Explicit type casting**: Ensure y_true is int32
2. **Prediction clipping**: Clip predictions to avoid log(0)
3. **Manual cross-entropy**: Calculate cross-entropy manually for better control
4. **Safe division**: Use tf.cond to avoid division by zero

## Updated Loss Function

The fixed version is now in cell 9 of the notebook. Run that cell again before training.

## Alternative Approach (Recommended for Stability)

If you continue to see NaN, use this simpler approach with **sample weights** instead:

###  Step 1: Add this cell before creating models

```python
# Create sample weights (1 for valid labels, 0 for missing)
def create_sample_weights(y):
    return np.where(y != -1, 1.0, 0.0)

train_weights = create_sample_weights(y_train)
test_weights = create_sample_weights(y_test)

# Simple loss function
def safe_sparse_crossentropy(y_true, y_pred):
    y_true_safe = tf.maximum(tf.cast(y_true, tf.int32), 0)
    return tf.keras.losses.sparse_categorical_crossentropy(y_true_safe, y_pred)
```

### Step 2: Change model compilation

```python
# Instead of:
# model.compile(optimizer=keras.optimizers.Adam(0.001),
#               loss=masked_categorical_crossentropy,
#               metrics=['accuracy'])

# Use:
model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=safe_sparse_crossentropy,
              metrics=['accuracy'])
```

### Step 3: Change training call

```python
# Instead of:
# history = model.fit(X_train_scaled, y_train, ...)

# Use:
history = model.fit(X_train_scaled, y_train,
                   sample_weight=train_weights,  # ADD THIS
                   validation_data=(X_test_scaled, y_test, test_weights),  # ADD test_weights
                   epochs=100,
                   batch_size=16,
                   callbacks=[early_stop, reduce_lr],
                   verbose=1)
```

## Quick Test

Before training all models, test with a simple model first:

```python
# Simple test model
test_model = keras.Sequential([
    keras.layers.Input(shape=(6, 340)),
    keras.layers.LSTM(32),
    keras.layers.Dense(16, activation='linear'),
    keras.layers.Reshape((4, 4)),
    keras.layers.Activation('softmax')
])

test_model.compile(
    optimizer='adam',
    loss=safe_sparse_crossentropy,
    metrics=['accuracy']
)

# Train for just 5 epochs
test_history = test_model.fit(
    X_train_scaled, y_train,
    sample_weight=train_weights,
    validation_data=(X_test_scaled, y_test, test_weights),
    epochs=5,
    batch_size=16,
    verbose=1
)

# Check if loss is valid
print(f"Final loss: {test_history.history['loss'][-1]}")
print(f"Is NaN? {np.isnan(test_history.history['loss'][-1])}")
```

If this works (no NaN), then use the same approach for all your models.

## Data Statistics

Your current data has:
- **Total labels**: 432 (108 samples × 4 directions)
- **Valid labels**: 325 (75%)
- **Missing labels**: 107 (25%)

Per direction:
- **North**: 107/108 valid (99%) - heavily imbalanced (103 class 0)
- **East**: 108/108 valid (100%) - heavily imbalanced (103 class 0)
- **South**: 8/108 valid (7%) - **very few labels!**
- **West**: 102/108 valid (94%) - heavily imbalanced (100 class 0)

## Recommendations

1. **Use sample weights approach** (more stable)
2. **Consider class weights** to handle imbalance:
   ```python
   from sklearn.utils.class_weight import compute_class_weight

   # Get all valid labels
   valid_labels = y_train[y_train != -1].flatten()

   # Compute weights
   class_weights = compute_class_weight(
       'balanced',
       classes=np.unique(valid_labels),
       y=valid_labels
   )

   class_weight_dict = dict(enumerate(class_weights))

   # Use in training
   model.fit(..., class_weight=class_weight_dict)
   ```

3. **Get more data** if possible - South direction has only 8 labeled samples!

4. **Reduce learning rate** to 0.0001 for more stable training:
   ```python
   model.compile(optimizer=keras.optimizers.Adam(0.0001), ...)
   ```

## Files to Update

If using the sample weights approach, you'll need to modify:

1. **Cell with utility functions** - add `create_sample_weights` function
2. **All model compilation cells** - use `safe_sparse_crossentropy`
3. **All training cells** - add `sample_weight` parameter

Let me know which approach you'd like to use, and I can create an updated notebook!
