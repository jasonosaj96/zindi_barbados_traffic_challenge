#!/usr/bin/env python3
"""
End-to-End Pipeline for Barbados Traffic Challenge

This script runs the complete ML pipeline:
1. Feature engineering from raw data
2. Model training with cross-validation
3. Evaluation on validation set
4. Prediction on test set
5. Submission file generation

Usage:
    # Full pipeline (train + predict)
    python pipeline.py --mode full

    # Train only
    python pipeline.py --mode train

    # Predict only (requires trained model)
    python pipeline.py --mode predict
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end ML pipeline orchestrator."""

    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize pipeline with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        logger.info("Pipeline initialized")
        logger.info(f"  Config: {config_path}")

    def run_feature_engineering(self, test_mode: bool = False):
        """Run feature engineering step."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: FEATURE ENGINEERING")
        logger.info("=" * 80)

        fe_config = self.config['feature_engineering']
        paths = self.config['paths']

        if test_mode:
            input_csv = paths['test_csv']
            output_csv = paths['test_features']
            mode_flag = '--test-mode'
        else:
            input_csv = paths['train_csv']
            output_csv = paths['train_features']
            mode_flag = ''

        cmd = [
            'python', 'feature_engineering.py',
            '--input', input_csv,
            '--output', output_csv,
            '--window-size', str(fe_config['window_size']),
            '--prediction-horizon', str(fe_config['prediction_horizon']),
            '--embargo-period', str(fe_config['embargo_period'])
        ]

        if mode_flag:
            cmd.append(mode_flag)

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)

        if result.returncode != 0:
            logger.error("Feature engineering failed!")
            sys.exit(1)

        logger.info("✓ Feature engineering complete")

    def run_training(self):
        """Run model training step."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: MODEL TRAINING")
        logger.info("=" * 80)

        training_config = self.config['training']
        paths = self.config['paths']

        cmd = [
            'python', 'train_model.py',
            '--features', paths['train_features'],
            '--labels', paths['train_labels'],
            '--output', paths['models_dir'],
            '--test-split', str(training_config['test_split']),
            '--cv', str(training_config['cross_validation_folds']),
            '--random-seed', str(training_config['random_seed'])
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)

        if result.returncode != 0:
            logger.error("Model training failed!")
            sys.exit(1)

        logger.info("✓ Model training complete")

    def run_prediction(self):
        """Run prediction step."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: PREDICTION")
        logger.info("=" * 80)

        paths = self.config['paths']

        # Check if test features exist
        if not Path(paths['test_features']).exists():
            logger.info("Test features not found, running feature engineering...")
            self.run_feature_engineering(test_mode=True)

        cmd = [
            'python', 'predict.py',
            '--features', paths['test_features'],
            '--model', paths['models_dir'],
            '--output', paths['submission']
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)

        if result.returncode != 0:
            logger.error("Prediction failed!")
            sys.exit(1)

        logger.info("✓ Prediction complete")

    def run_full_pipeline(self):
        """Run complete pipeline."""
        logger.info("\n" + "#" * 80)
        logger.info("# FULL PIPELINE - BARBADOS TRAFFIC CHALLENGE")
        logger.info("#" * 80)

        # Step 1: Feature engineering (train)
        self.run_feature_engineering(test_mode=False)

        # Step 2: Model training
        self.run_training()

        # Step 3: Feature engineering (test) + Prediction
        self.run_prediction()

        logger.info("\n" + "#" * 80)
        logger.info("# PIPELINE COMPLETE")
        logger.info("#" * 80)
        logger.info(f"\nSubmission file: {self.config['paths']['submission']}")


def main():
    parser = argparse.ArgumentParser(description='Run ML pipeline for traffic prediction')
    parser.add_argument('--mode', type=str, choices=['full', 'train', 'predict'],
                        default='full',
                        help='Pipeline mode: full, train, or predict')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = Pipeline(config_path=args.config)

    # Run appropriate mode
    if args.mode == 'full':
        pipeline.run_full_pipeline()
    elif args.mode == 'train':
        pipeline.run_feature_engineering(test_mode=False)
        pipeline.run_training()
    elif args.mode == 'predict':
        pipeline.run_prediction()


if __name__ == '__main__':
    main()
