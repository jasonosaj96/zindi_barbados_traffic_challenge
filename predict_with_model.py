"""
Utility script for making predictions with trained models

Usage:
    python predict_with_model.py --model models/lstm_traffic_model.keras --data sequences/X_test.npy
"""

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib


def masked_categorical_crossentropy(y_true, y_pred):
    """Custom loss function to handle missing labels"""
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    y_true_safe = tf.maximum(y_true, 0)
    y_true_onehot = tf.one_hot(tf.cast(y_true_safe, tf.int32), depth=4)
    loss = tf.keras.losses.categorical_crossentropy(y_true_onehot, y_pred)
    loss = loss * mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-7)


def load_model_and_scaler(model_path, scaler_path='models/feature_scaler.pkl'):
    """Load trained model and scaler"""
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(
        model_path,
        custom_objects={'masked_categorical_crossentropy': masked_categorical_crossentropy}
    )

    print(f"Loading scaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)

    return model, scaler


def preprocess_data(X, scaler):
    """Preprocess data using saved scaler"""
    n_samples, window_size, n_features = X.shape

    # Flatten
    X_flat = X.reshape(-1, n_features)

    # Handle NaN
    col_means = np.nanmean(X_flat, axis=0)
    for i in range(n_features):
        mask = np.isnan(X_flat[:, i])
        X_flat[mask, i] = col_means[i]

    # Scale
    X_scaled = scaler.transform(X_flat)

    # Reshape back
    X_scaled = X_scaled.reshape(n_samples, window_size, n_features)

    return X_scaled


def predict(model, X_scaled):
    """Make predictions"""
    print("Making predictions...")
    predictions = model.predict(X_scaled, verbose=0)
    predicted_classes = np.argmax(predictions, axis=-1)
    predicted_probs = np.max(predictions, axis=-1)

    return predicted_classes, predicted_probs, predictions


def format_predictions(predicted_classes, predicted_probs, metadata=None):
    """Format predictions as DataFrame"""
    directions = ['north', 'east', 'south', 'west']
    class_names = ['free flowing', 'light delay', 'moderate delay', 'heavy delay']

    results = []
    for i in range(len(predicted_classes)):
        row = {}

        if metadata is not None:
            row['timestamp'] = metadata.iloc[i]['last_timestamp']
            row['sequence_id'] = metadata.iloc[i]['sequence_id']

        for j, direction in enumerate(directions):
            pred_class = predicted_classes[i, j]
            pred_prob = predicted_probs[i, j]
            row[f'{direction}_prediction'] = class_names[pred_class]
            row[f'{direction}_confidence'] = pred_prob

        results.append(row)

    return pd.DataFrame(results)


def evaluate_predictions(y_true, predicted_classes):
    """Evaluate predictions if ground truth is available"""
    from sklearn.metrics import accuracy_score, classification_report

    directions = ['North', 'East', 'South', 'West']
    class_names = ['Free Flowing', 'Light Delay', 'Moderate Delay', 'Heavy Delay']

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60 + "\n")

    accuracies = []
    for i, direction in enumerate(directions):
        valid_mask = y_true[:, i] != -1
        if valid_mask.sum() == 0:
            print(f"{direction}: No valid labels")
            continue

        y_true_dir = y_true[valid_mask, i]
        y_pred_dir = predicted_classes[valid_mask, i]

        acc = accuracy_score(y_true_dir, y_pred_dir)
        accuracies.append(acc)

        print(f"{direction} Entry - Accuracy: {acc:.4f}")
        print(classification_report(y_true_dir, y_pred_dir,
                                   target_names=class_names,
                                   zero_division=0))

    if accuracies:
        print(f"\nAverage Accuracy: {np.mean(accuracies):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained traffic model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.keras file)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to input data (.npy file)')
    parser.add_argument('--scaler', type=str, default='models/feature_scaler.pkl',
                       help='Path to scaler file')
    parser.add_argument('--metadata', type=str, default=None,
                       help='Path to metadata CSV (optional)')
    parser.add_argument('--ground-truth', type=str, default=None,
                       help='Path to ground truth labels (.npy file) for evaluation')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV file path')
    parser.add_argument('--show-probabilities', action='store_true',
                       help='Show full probability distributions')

    args = parser.parse_args()

    # Load model and scaler
    model, scaler = load_model_and_scaler(args.model, args.scaler)

    # Load data
    print(f"Loading data from {args.data}...")
    X = np.load(args.data)
    print(f"Data shape: {X.shape}")

    # Load metadata if provided
    metadata = None
    if args.metadata:
        print(f"Loading metadata from {args.metadata}...")
        metadata = pd.read_csv(args.metadata)

    # Preprocess
    print("Preprocessing data...")
    X_scaled = preprocess_data(X, scaler)

    # Predict
    predicted_classes, predicted_probs, full_predictions = predict(model, X_scaled)

    # Format results
    print("Formatting predictions...")
    results_df = format_predictions(predicted_classes, predicted_probs, metadata)

    # Add full probability distributions if requested
    if args.show_probabilities:
        directions = ['north', 'east', 'south', 'west']
        class_names = ['free_flowing', 'light_delay', 'moderate_delay', 'heavy_delay']
        for i, direction in enumerate(directions):
            for j, class_name in enumerate(class_names):
                results_df[f'{direction}_{class_name}_prob'] = full_predictions[:, i, j]

    # Save results
    print(f"Saving predictions to {args.output}...")
    results_df.to_csv(args.output, index=False)
    print(f"Saved {len(results_df)} predictions")

    # Evaluate if ground truth provided
    if args.ground_truth:
        print(f"\nLoading ground truth from {args.ground_truth}...")
        y_true = np.load(args.ground_truth)
        evaluate_predictions(y_true, predicted_classes)

    # Show sample predictions
    print("\nSample predictions:")
    print(results_df.head(10).to_string())

    print("\nDone!")


if __name__ == '__main__':
    main()
