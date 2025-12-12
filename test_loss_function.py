"""
Test the custom loss function to debug NaN issues
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load data
y_train = np.load('sequences/y_train.npy')
y_test = np.load('sequences/y_test.npy')

print("Data loaded:")
print(f"y_train shape: {y_train.shape}")
print(f"y_train unique values: {np.unique(y_train)}")
print(f"Valid labels: {(y_train != -1).sum()}/{y_train.size}")

# Define custom loss
def masked_categorical_crossentropy_v1(y_true, y_pred):
    """Original version"""
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    y_true_safe = tf.maximum(y_true, 0)
    y_true_onehot = tf.one_hot(tf.cast(y_true_safe, tf.int32), depth=4)
    loss = tf.keras.losses.categorical_crossentropy(y_true_onehot, y_pred)
    loss = loss * mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-7)

def masked_categorical_crossentropy_v2(y_true, y_pred):
    """Improved version with clipping"""
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    y_true_safe = tf.maximum(y_true, 0)
    y_true_onehot = tf.one_hot(y_true_safe, depth=4)

    # Clip predictions to avoid log(0)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

    # Manual cross entropy
    loss = -tf.reduce_sum(y_true_onehot * tf.math.log(y_pred), axis=-1)
    loss = loss * mask

    n_valid = tf.reduce_sum(mask)
    return tf.cond(
        n_valid > 0,
        lambda: tf.reduce_sum(loss) / n_valid,
        lambda: 0.0
    )

# Test with dummy predictions
print("\nTesting loss functions...")

# Create dummy predictions (batch_size=4, 4 directions, 4 classes)
y_true_batch = tf.constant([
    [0, 1, -1, 0],
    [0, 0, -1, -1],
    [2, 1, -1, 0],
    [-1, -1, -1, 1]
], dtype=tf.float32)

y_pred_batch = tf.constant([
    [[0.9, 0.05, 0.03, 0.02], [0.1, 0.8, 0.05, 0.05], [0.25, 0.25, 0.25, 0.25], [0.7, 0.1, 0.1, 0.1]],
    [[0.8, 0.1, 0.05, 0.05], [0.7, 0.15, 0.1, 0.05], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]],
    [[0.1, 0.1, 0.7, 0.1], [0.2, 0.6, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25], [0.6, 0.2, 0.1, 0.1]],
    [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.1, 0.8, 0.05, 0.05]]
], dtype=tf.float32)

print("Input shapes:")
print(f"  y_true_batch: {y_true_batch.shape}")
print(f"  y_pred_batch: {y_pred_batch.shape}")

try:
    loss_v1 = masked_categorical_crossentropy_v1(y_true_batch, y_pred_batch)
    print(f"\nLoss V1 (original): {loss_v1.numpy():.4f}")
except Exception as e:
    print(f"\nLoss V1 failed: {e}")

try:
    loss_v2 = masked_categorical_crossentropy_v2(y_true_batch, y_pred_batch)
    print(f"Loss V2 (improved): {loss_v2.numpy():.4f}")
except Exception as e:
    print(f"Loss V2 failed: {e}")

# Test with actual data
print("\n" + "="*60)
print("Testing with actual training data (first batch):")
print("="*60)

# Simulate model predictions (random)
np.random.seed(42)
batch_size = 16
y_true_real = tf.constant(y_train[:batch_size], dtype=tf.float32)
y_pred_real = tf.nn.softmax(tf.random.normal((batch_size, 4, 4)), axis=-1)

print(f"Batch size: {batch_size}")
print(f"y_true shape: {y_true_real.shape}")
print(f"y_pred shape: {y_pred_real.shape}")
print(f"Valid labels in batch: {tf.reduce_sum(tf.cast(tf.not_equal(y_true_real, -1), tf.int32)).numpy()}")

try:
    loss_v1_real = masked_categorical_crossentropy_v1(y_true_real, y_pred_real)
    print(f"\nLoss V1 on real data: {loss_v1_real.numpy():.4f}")
    if np.isnan(loss_v1_real.numpy()):
        print("  WARNING: Loss is NaN!")
except Exception as e:
    print(f"\nLoss V1 on real data failed: {e}")

try:
    loss_v2_real = masked_categorical_crossentropy_v2(y_true_real, y_pred_real)
    print(f"Loss V2 on real data: {loss_v2_real.numpy():.4f}")
    if np.isnan(loss_v2_real.numpy()):
        print("  WARNING: Loss is NaN!")
except Exception as e:
    print(f"Loss V2 on real data failed: {e}")

print("\nDone!")
