#!/usr/bin/env python3
"""
Prediction Script for Barbados Traffic Challenge

This script generates predictions for test data and creates submission files.

Usage:
    python predict.py --features features/test_features.csv --model models/ --output submissions/submission.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from train_model import CongestionClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_submission(
    features_df: pd.DataFrame,
    predictions_enter: np.ndarray,
    predictions_exit: np.ndarray,
    output_path: Path
):
    """
    Create submission file in required format.

    Format:
        ID, Target, Target_Accuracy
        time_segment_181_Norman Niles #1_congestion_enter_rating, heavy delay, heavy delay
        time_segment_181_Norman Niles #1_congestion_exit_rating, heavy delay, heavy delay
    """
    logger.info("Creating submission file...")

    submissions = []

    for i in range(len(features_df)):
        camera = features_df.iloc[i]['camera']
        target_id = features_df.iloc[i]['window_end_id'] + 7  # +2 embargo +5 prediction

        # Entrance prediction
        id_enter = f"time_segment_{target_id}_{camera}_congestion_enter_rating"
        submissions.append({
            'ID': id_enter,
            'Target': predictions_enter[i],
            'Target_Accuracy': predictions_enter[i]
        })

        # Exit prediction
        id_exit = f"time_segment_{target_id}_{camera}_congestion_exit_rating"
        submissions.append({
            'ID': id_exit,
            'Target': predictions_exit[i],
            'Target_Accuracy': predictions_exit[i]
        })

    submission_df = pd.DataFrame(submissions)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    logger.info(f"Submission saved to {output_path}")
    logger.info(f"  Total predictions: {len(submission_df)}")
    logger.info(f"\nPrediction distribution:")
    logger.info(f"\n{submission_df['Target'].value_counts()}")

    return submission_df


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for test set')
    parser.add_argument('--features', type=str, required=True,
                        help='Path to test features CSV')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for submission CSV')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PREDICTION - BARBADOS TRAFFIC CHALLENGE")
    logger.info("=" * 80)

    # Load features
    logger.info(f"\nLoading test features from {args.features}...")
    X_test = pd.read_csv(args.features)
    logger.info(f"  Loaded {len(X_test)} samples")

    # Load model
    logger.info(f"\nLoading model from {args.model}...")
    classifier = CongestionClassifier()
    classifier.load(Path(args.model))

    # Generate predictions
    logger.info(f"\nGenerating predictions...")
    y_enter_pred, y_exit_pred = classifier.predict(X_test)

    logger.info(f"  Entrance predictions: {len(y_enter_pred)}")
    logger.info(f"  Exit predictions: {len(y_exit_pred)}")

    # Create submission
    submission_df = create_submission(X_test, y_enter_pred, y_exit_pred, Path(args.output))

    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
