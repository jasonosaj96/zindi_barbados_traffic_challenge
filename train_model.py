#!/usr/bin/env python3
"""
Model Training Module for Barbados Traffic Challenge

This module handles model training, validation, and evaluation:
- Trains XGBoost classifiers for enter/exit congestion
- Performs cross-validation
- Evaluates with Macro-F1 (70%) + Accuracy (30%)
- Saves trained models and metrics

Usage:
    python train_model.py --features features/train_features.csv --labels features/train_labels.csv
    python train_model.py --features features/train_features.csv --labels features/train_labels.csv --cv 5
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, make_scorer
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CongestionClassifier:
    """
    XGBoost-based congestion classifier with dual-metric evaluation.

    Trains separate models for entrance and exit congestion.
    Evaluates with weighted score: 70% Macro-F1 + 30% Accuracy.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize classifier.

        Args:
            config: Model configuration dictionary
        """
        self.config = config or self._default_config()
        self.model_enter = None
        self.model_exit = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None

        # Fit label encoder
        self.label_encoder.fit(['free flowing', 'light delay', 'moderate delay', 'heavy delay'])

        logger.info("Initialized CongestionClassifier")
        logger.info(f"  Model config: {self.config}")

    def _default_config(self) -> Dict:
        """Default XGBoost configuration."""
        return {
            'objective': 'multi:softmax',
            'num_class': 4,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'enable_categorical': False
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_enter_train: pd.Series,
        y_exit_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_enter_val: pd.Series = None,
        y_exit_val: pd.Series = None
    ):
        """
        Train both entrance and exit models.

        Args:
            X_train: Training features
            y_enter_train: Entrance congestion labels
            y_exit_train: Exit congestion labels
            X_val: Validation features (optional)
            y_enter_val: Validation entrance labels (optional)
            y_exit_val: Validation exit labels (optional)
        """
        logger.info("Training models...")

        # Store feature names
        self.feature_names = [col for col in X_train.columns
                             if col not in ['camera', 'window_start_id', 'window_end_id']]

        # Filter to feature columns only
        X_train_features = X_train[self.feature_names]

        # Encode labels
        y_enter_encoded = self.label_encoder.transform(y_enter_train)
        y_exit_encoded = self.label_encoder.transform(y_exit_train)

        # Prepare validation set if provided
        eval_set_enter = None
        eval_set_exit = None

        if X_val is not None and y_enter_val is not None:
            X_val_features = X_val[self.feature_names]
            y_enter_val_encoded = self.label_encoder.transform(y_enter_val)
            y_exit_val_encoded = self.label_encoder.transform(y_exit_val)

            eval_set_enter = [(X_train_features, y_enter_encoded),
                             (X_val_features, y_enter_val_encoded)]
            eval_set_exit = [(X_train_features, y_exit_encoded),
                            (X_val_features, y_exit_val_encoded)]

        # Train entrance model
        logger.info("  Training entrance congestion model...")
        self.model_enter = xgb.XGBClassifier(**self.config)

        if eval_set_enter:
            self.model_enter.fit(
                X_train_features, y_enter_encoded,
                eval_set=eval_set_enter,
                verbose=False
            )
        else:
            self.model_enter.fit(X_train_features, y_enter_encoded)

        logger.info("    ✓ Entrance model trained")

        # Train exit model
        logger.info("  Training exit congestion model...")
        self.model_exit = xgb.XGBClassifier(**self.config)

        if eval_set_exit:
            self.model_exit.fit(
                X_train_features, y_exit_encoded,
                eval_set=eval_set_exit,
                verbose=False
            )
        else:
            self.model_exit.fit(X_train_features, y_exit_encoded)

        logger.info("    ✓ Exit model trained")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict congestion levels.

        Args:
            X: Feature matrix

        Returns:
            (entrance_predictions, exit_predictions)
        """
        X_features = X[self.feature_names]

        y_enter_pred = self.model_enter.predict(X_features)
        y_exit_pred = self.model_exit.predict(X_features)

        # Decode labels
        y_enter_pred = self.label_encoder.inverse_transform(y_enter_pred)
        y_exit_pred = self.label_encoder.inverse_transform(y_exit_pred)

        return y_enter_pred, y_exit_pred

    def evaluate(
        self,
        X: pd.DataFrame,
        y_enter_true: pd.Series,
        y_exit_true: pd.Series
    ) -> Dict:
        """
        Evaluate model with dual metrics.

        Metric: 70% Macro-F1 + 30% Accuracy

        Args:
            X: Features
            y_enter_true: True entrance labels
            y_exit_true: True exit labels

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating models...")

        # Predictions
        y_enter_pred, y_exit_pred = self.predict(X)

        # Entrance metrics
        enter_f1 = f1_score(y_enter_true, y_enter_pred, average='macro')
        enter_acc = accuracy_score(y_enter_true, y_enter_pred)
        enter_score = 0.7 * enter_f1 + 0.3 * enter_acc

        # Exit metrics
        exit_f1 = f1_score(y_exit_true, y_exit_pred, average='macro')
        exit_acc = accuracy_score(y_exit_true, y_exit_pred)
        exit_score = 0.7 * exit_f1 + 0.3 * exit_acc

        # Overall score
        overall_score = (enter_score + exit_score) / 2

        results = {
            'entrance': {
                'macro_f1': float(enter_f1),
                'accuracy': float(enter_acc),
                'weighted_score': float(enter_score),
                'f1_by_class': dict(zip(
                    self.label_encoder.classes_,
                    f1_score(y_enter_true, y_enter_pred, average=None)
                ))
            },
            'exit': {
                'macro_f1': float(exit_f1),
                'accuracy': float(exit_acc),
                'weighted_score': float(exit_score),
                'f1_by_class': dict(zip(
                    self.label_encoder.classes_,
                    f1_score(y_exit_true, y_exit_pred, average=None)
                ))
            },
            'overall_score': float(overall_score)
        }

        # Print results
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 80)

        logger.info(f"\nENTRANCE CONGESTION:")
        logger.info(f"  Macro-F1:        {enter_f1:.4f}")
        logger.info(f"  Accuracy:        {enter_acc:.4f}")
        logger.info(f"  Weighted Score:  {enter_score:.4f} (70% F1 + 30% Acc)")

        logger.info(f"\nEXIT CONGESTION:")
        logger.info(f"  Macro-F1:        {exit_f1:.4f}")
        logger.info(f"  Accuracy:        {exit_acc:.4f}")
        logger.info(f"  Weighted Score:  {exit_score:.4f} (70% F1 + 30% Acc)")

        logger.info(f"\nOVERALL SCORE: {overall_score:.4f}")
        logger.info("=" * 80)

        # Detailed reports
        logger.info(f"\nClassification Report (Entrance):")
        logger.info("\n" + classification_report(y_enter_true, y_enter_pred))

        logger.info(f"\nClassification Report (Exit):")
        logger.info("\n" + classification_report(y_exit_true, y_exit_pred))

        return results

    def cross_validate(
        self,
        X: pd.DataFrame,
        y_enter: pd.Series,
        y_exit: pd.Series,
        cv: int = 5
    ) -> Dict:
        """
        Perform cross-validation.

        Args:
            X: Features
            y_enter: Entrance labels
            y_exit: Exit labels
            cv: Number of folds

        Returns:
            Cross-validation results
        """
        logger.info(f"\nPerforming {cv}-fold cross-validation...")

        X_features = X[self.feature_names]
        y_enter_encoded = self.label_encoder.transform(y_enter)
        y_exit_encoded = self.label_encoder.transform(y_exit)

        # Custom scorer for weighted metric
        def weighted_scorer(y_true, y_pred):
            f1 = f1_score(y_true, y_pred, average='macro')
            acc = accuracy_score(y_true, y_pred)
            return 0.7 * f1 + 0.3 * acc

        scorer = make_scorer(weighted_scorer)

        # Cross-validate entrance model
        model_enter = xgb.XGBClassifier(**self.config)
        cv_scores_enter = cross_val_score(
            model_enter, X_features, y_enter_encoded,
            cv=cv, scoring=scorer, n_jobs=-1
        )

        # Cross-validate exit model
        model_exit = xgb.XGBClassifier(**self.config)
        cv_scores_exit = cross_val_score(
            model_exit, X_features, y_exit_encoded,
            cv=cv, scoring=scorer, n_jobs=-1
        )

        results = {
            'entrance': {
                'cv_scores': cv_scores_enter.tolist(),
                'mean_score': float(cv_scores_enter.mean()),
                'std_score': float(cv_scores_enter.std())
            },
            'exit': {
                'cv_scores': cv_scores_exit.tolist(),
                'mean_score': float(cv_scores_exit.mean()),
                'std_score': float(cv_scores_exit.std())
            }
        }

        logger.info(f"\nCross-Validation Results:")
        logger.info(f"  Entrance: {cv_scores_enter.mean():.4f} (+/- {cv_scores_enter.std():.4f})")
        logger.info(f"  Exit:     {cv_scores_exit.mean():.4f} (+/- {cv_scores_exit.std():.4f})")

        return results

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance for interpretability.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        logger.info(f"\nAnalyzing feature importance (top {top_n})...")

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_enter': self.model_enter.feature_importances_,
            'importance_exit': self.model_exit.feature_importances_
        })

        importance_df['importance_avg'] = (
            importance_df['importance_enter'] + importance_df['importance_exit']
        ) / 2

        importance_df = importance_df.sort_values('importance_avg', ascending=False)

        logger.info(f"\n{importance_df.head(top_n).to_string(index=False)}")

        return importance_df

    def plot_confusion_matrices(
        self,
        X: pd.DataFrame,
        y_enter_true: pd.Series,
        y_exit_true: pd.Series,
        output_dir: Path
    ):
        """Plot confusion matrices."""
        y_enter_pred, y_exit_pred = self.predict(X)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Entrance confusion matrix
        cm_enter = confusion_matrix(
            y_enter_true, y_enter_pred,
            labels=self.label_encoder.classes_
        )
        sns.heatmap(cm_enter, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   ax=axes[0])
        axes[0].set_title('Entrance Congestion')
        axes[0].set_ylabel('True')
        axes[0].set_xlabel('Predicted')

        # Exit confusion matrix
        cm_exit = confusion_matrix(
            y_exit_true, y_exit_pred,
            labels=self.label_encoder.classes_
        )
        sns.heatmap(cm_exit, annot=True, fmt='d', cmap='Greens',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   ax=axes[1])
        axes[1].set_title('Exit Congestion')
        axes[1].set_ylabel('True')
        axes[1].set_xlabel('Predicted')

        plt.tight_layout()
        output_path = output_dir / 'confusion_matrices.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved confusion matrices to {output_path}")
        plt.close()

    def save(self, output_dir: Path):
        """Save trained models."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save XGBoost models
        self.model_enter.save_model(str(output_dir / 'model_enter.json'))
        self.model_exit.save_model(str(output_dir / 'model_exit.json'))

        # Save metadata
        metadata = {
            'config': self.config,
            'feature_names': self.feature_names,
            'label_encoder_classes': self.label_encoder.classes_.tolist()
        }

        with open(output_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\nModels saved to {output_dir}/")

    def load(self, output_dir: Path):
        """Load trained models."""
        output_dir = Path(output_dir)

        # Load metadata
        with open(output_dir / 'model_metadata.json', 'r') as f:
            metadata = json.load(f)

        self.config = metadata['config']
        self.feature_names = metadata['feature_names']

        # Load models
        self.model_enter = xgb.XGBClassifier()
        self.model_enter.load_model(str(output_dir / 'model_enter.json'))

        self.model_exit = xgb.XGBClassifier()
        self.model_exit.load_model(str(output_dir / 'model_exit.json'))

        logger.info(f"Models loaded from {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Train congestion prediction model')
    parser.add_argument('--features', type=str, required=True,
                        help='Path to features CSV')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to labels CSV')
    parser.add_argument('--output', type=str, default='models/',
                        help='Output directory for models')
    parser.add_argument('--test-split', type=float, default=0.2,
                        help='Test split ratio (default: 0.2)')
    parser.add_argument('--cv', type=int, default=0,
                        help='Cross-validation folds (0 = no CV, default: 0)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("MODEL TRAINING - BARBADOS TRAFFIC CHALLENGE")
    logger.info("=" * 80)

    # Load data
    logger.info(f"\nLoading features from {args.features}...")
    X = pd.read_csv(args.features)
    logger.info(f"  Shape: {X.shape}")

    logger.info(f"\nLoading labels from {args.labels}...")
    labels = pd.read_csv(args.labels)
    logger.info(f"  Shape: {labels.shape}")

    y_enter = labels['target_enter']
    y_exit = labels['target_exit']

    # Train/test split
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.test_split,
        random_state=args.random_seed,
        stratify=y_enter
    )

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_enter_train, y_enter_val = y_enter.iloc[train_idx], y_enter.iloc[val_idx]
    y_exit_train, y_exit_val = y_exit.iloc[train_idx], y_exit.iloc[val_idx]

    logger.info(f"\nTrain/Validation Split:")
    logger.info(f"  Training samples:   {len(X_train)}")
    logger.info(f"  Validation samples: {len(X_val)}")

    # Initialize classifier
    classifier = CongestionClassifier()

    # Cross-validation (if requested)
    if args.cv > 0:
        cv_results = classifier.cross_validate(X_train, y_enter_train, y_exit_train, cv=args.cv)

    # Train on full training set
    classifier.train(X_train, y_enter_train, y_exit_train, X_val, y_enter_val, y_exit_val)

    # Evaluate on validation set
    results = classifier.evaluate(X_val, y_enter_val, y_exit_val)

    # Feature importance
    importance = classifier.get_feature_importance()

    # Save everything
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier.save(output_dir)

    # Save metrics
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Saved evaluation results")

    # Save feature importance
    importance.to_csv(output_dir / 'feature_importance.csv', index=False)
    logger.info(f"  Saved feature importance")

    # Plot confusion matrices
    classifier.plot_confusion_matrices(X_val, y_enter_val, y_exit_val, output_dir)

    logger.info("\n" + "=" * 80)
    logger.info(f"TRAINING COMPLETE - Overall Score: {results['overall_score']:.4f}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
