"""
Example: Training a simple LSTM model on rolling window sequences.

This demonstrates how to use the sequences created by create_rolling_window_dataset.py
to train a temporal model for traffic congestion prediction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def load_sequences(sequences_dir: str = 'sequences'):
    """Load pre-created sequences from disk."""
    seq_path = Path(sequences_dir)

    print(f"Loading sequences from {seq_path}...")

    # Load arrays
    X_train = np.load(seq_path / 'X_train.npy')
    X_test = np.load(seq_path / 'X_test.npy')
    y_train = np.load(seq_path / 'y_train.npy')
    y_test = np.load(seq_path / 'y_test.npy')

    # Load metadata
    train_meta = pd.read_csv(seq_path / 'train_metadata.csv')
    test_meta = pd.read_csv(seq_path / 'test_metadata.csv')
    feature_cols = pd.read_csv(seq_path / 'feature_columns.csv')['feature'].tolist()
    target_cols = pd.read_csv(seq_path / 'target_columns.csv')['target'].tolist()

    print(f"✓ Loaded sequences:")
    print(f"  X_train: {X_train.shape} (samples, windows, features)")
    print(f"  y_train: {y_train.shape} (samples, targets)")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Targets: {target_cols}")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_meta': train_meta,
        'test_meta': test_meta,
        'feature_cols': feature_cols,
        'target_cols': target_cols
    }


def prepare_targets_for_training(y_train, y_test):
    """
    Prepare targets for multi-output classification.

    Handles missing values (-1) by masking them during training.
    """
    print("\nPreparing targets...")

    # Create masks for valid targets
    train_mask = (y_train != -1)
    test_mask = (y_test != -1)

    print(f"Valid targets in train: {train_mask.sum()} / {train_mask.size} ({train_mask.sum()/train_mask.size*100:.1f}%)")
    print(f"Valid targets in test:  {test_mask.sum()} / {test_mask.size} ({test_mask.sum()/test_mask.size*100:.1f}%)")

    # Replace -1 with 0 (will be masked during loss calculation)
    y_train_clean = np.where(y_train == -1, 0, y_train).astype(np.int32)
    y_test_clean = np.where(y_test == -1, 0, y_test).astype(np.int32)

    return y_train_clean, y_test_clean, train_mask, test_mask


def normalize_features(X_train, X_test):
    """
    Normalize features using training set statistics.

    Applies per-feature normalization across all time windows.
    """
    print("\nNormalizing features...")

    # Calculate mean and std from training data
    # Shape: (samples, windows, features) -> calculate across samples and windows
    train_flat = X_train.reshape(-1, X_train.shape[-1])  # (samples*windows, features)

    # Use robust statistics (ignore NaN and inf)
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(train_flat, axis=0)
        std = np.nanstd(train_flat, axis=0)

    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    std = np.where(np.isnan(std), 1, std)
    mean = np.where(np.isnan(mean), 0, mean)

    # Normalize
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    # Handle any remaining NaN/inf
    X_train_norm = np.nan_to_num(X_train_norm, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_norm = np.nan_to_num(X_test_norm, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"✓ Features normalized (mean=0, std=1)")

    return X_train_norm, X_test_norm, mean, std


def build_lstm_model(input_shape, n_targets, n_classes=4):
    """
    Build a simple LSTM model for multi-output classification.

    Args:
        input_shape: (n_windows, n_features)
        n_targets: Number of target outputs (4 for north/east/south/west)
        n_classes: Number of congestion classes (0-3)
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("ERROR: TensorFlow not installed. Install with: pip install tensorflow")
        return None

    print(f"\nBuilding LSTM model...")
    print(f"  Input shape: {input_shape}")
    print(f"  Outputs: {n_targets} targets × {n_classes} classes")

    # Input
    inputs = keras.Input(shape=input_shape, name='sequence_input')

    # LSTM layers
    x = layers.Masking(mask_value=0.0)(inputs)  # Mask padded sequences
    x = layers.LSTM(128, return_sequences=True, dropout=0.2)(x)
    x = layers.LSTM(64, dropout=0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Multi-output classification heads (one per target direction)
    outputs = []
    target_names = ['north', 'east', 'south', 'west']

    for i, name in enumerate(target_names):
        out = layers.Dense(n_classes, activation='softmax', name=f'{name}_entry_congestion')(x)
        outputs.append(out)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='traffic_lstm')

    # Compile with multi-output loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=['sparse_categorical_crossentropy'] * n_targets,
        metrics=[['accuracy']] * n_targets
    )

    print(f"✓ Model built")
    model.summary()

    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """Train the LSTM model."""
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        print("ERROR: TensorFlow not installed.")
        return None

    print(f"\nTraining model for {epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_test)}")
    print("-" * 80)

    # Split targets for multi-output
    y_train_split = [y_train[:, i] for i in range(y_train.shape[1])]
    y_test_split = [y_test[:, i] for i in range(y_test.shape[1])]

    # Callbacks with progress
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=2
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=2
        )
    ]

    # Train with verbose=2 for one line per epoch (better for logging)
    # Use verbose=1 for progress bar per epoch, verbose=2 for one line per epoch
    history = model.fit(
        X_train,
        y_train_split,
        validation_data=(X_test, y_test_split),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2 # Progress bar per epoch
    )

    print("\n" + "=" * 80)
    print("✓ Training complete")
    print("=" * 80)

    return history


def evaluate_model(model, X_test, y_test, target_cols):
    """Evaluate model performance."""
    print("\nEvaluating model...")

    # Get predictions
    predictions = model.predict(X_test)

    # Convert probabilities to class predictions
    y_pred = np.array([np.argmax(pred, axis=1) for pred in predictions]).T

    # Overall accuracy
    results = {}
    for i, target in enumerate(target_cols):
        # Filter out missing values (-1)
        valid_mask = y_test[:, i] != -1
        if valid_mask.sum() == 0:
            continue

        y_true = y_test[valid_mask, i]
        y_pred_valid = y_pred[valid_mask, i]

        acc = accuracy_score(y_true, y_pred_valid)
        results[target] = acc

        print(f"\n{target}:")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  Valid samples: {valid_mask.sum()}")

        # Classification report
        class_names = ['free flowing', 'light delay', 'moderate delay', 'heavy delay']
        print(classification_report(y_true, y_pred_valid, target_names=class_names, zero_division=0))

    # Average accuracy
    avg_acc = np.mean(list(results.values()))
    print(f"\nAverage accuracy: {avg_acc:.3f}")

    return results, y_pred


def plot_training_history(history, output_path='training_history.png'):
    """Plot training history."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy (average across outputs)
    train_acc_keys = [k for k in history.history.keys() if 'accuracy' in k and not k.startswith('val')]
    if train_acc_keys:
        train_acc = np.mean([history.history[k] for k in train_acc_keys], axis=0)
        val_acc = np.mean([history.history[f'val_{k}'] for k in train_acc_keys], axis=0)

        axes[1].plot(train_acc, label='Train Accuracy')
        axes[1].plot(val_acc, label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Average Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training history plot to {output_path}")


def main():
    print("=" * 80)
    print("LSTM MODEL TRAINING - TRAFFIC CONGESTION PREDICTION")
    print("=" * 80)

    # Load sequences
    data = load_sequences('sequences')

    # Prepare targets
    y_train_clean, y_test_clean, train_mask, test_mask = prepare_targets_for_training(
        data['y_train'], data['y_test']
    )

    # Normalize features
    X_train_norm, X_test_norm, mean, std = normalize_features(
        data['X_train'], data['X_test']
    )

    # Build model
    model = build_lstm_model(
        input_shape=(X_train_norm.shape[1], X_train_norm.shape[2]),
        n_targets=len(data['target_cols']),
        n_classes=4
    )

    if model is None:
        return

    # Train model
    history = train_model(
        model,
        X_train_norm,
        y_train_clean,
        X_test_norm,
        y_test_clean,
        epochs=50,
        batch_size=32
    )

    # Evaluate
    results, predictions = evaluate_model(
        model,
        X_test_norm,
        y_test_clean,
        data['target_cols']
    )

    # Plot training history
    plot_training_history(history)

    # Save model
    model.save('traffic_lstm_model.keras')
    print("\n✓ Model saved to traffic_lstm_model.keras")

    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
