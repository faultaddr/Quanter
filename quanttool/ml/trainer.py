"""Model trainer for QuantTool ML module."""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json

from ..core.logging import get_logger
from .models import BaseModel, ModelRegistry


logger = get_logger(__name__)


class ModelTrainer:
    """Trainer for ML models with cross-validation and hyperparameter tuning."""

    def __init__(self, model: BaseModel, registry: ModelRegistry = None):
        """Initialize the model trainer.

        Args:
            model: Model to train
            registry: Optional model registry for saving
        """
        self.model = model
        self.registry = registry or ModelRegistry()
        self.training_history = []

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Perform time series cross-validation.

        Args:
            X: Feature matrix
            y: Target vector
            n_splits: Number of CV splits
            metrics: List of metrics to compute

        Returns:
            Dictionary with CV results
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {
            'folds': [],
            'metrics': {m: [] for m in metrics}
        }

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Training fold {fold + 1}/{n_splits}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Create fresh model instance for this fold
            from .models import ModelFactory
            fold_model = ModelFactory.create(
                self.model.name.lower().replace(' ', '_'),
                self.model.params
            )

            # Train
            fold_model.train(X_train, y_train)

            # Evaluate
            y_pred = fold_model.predict(X_val)
            y_proba = fold_model.predict_proba(X_val)[:, 1]

            fold_metrics = self._compute_metrics(y_val, y_pred, y_proba, metrics)
            fold_metrics['fold'] = fold
            fold_metrics['train_size'] = len(train_idx)
            fold_metrics['val_size'] = len(val_idx)

            cv_results['folds'].append(fold_metrics)

            for metric in metrics:
                cv_results['metrics'][metric].append(fold_metrics[metric])

        # Compute aggregate statistics
        cv_results['summary'] = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            for metric, values in cv_results['metrics'].items()
        }

        return cv_results

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        metrics: List[str]
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        results = {}

        for metric in metrics:
            try:
                if metric == 'accuracy':
                    results[metric] = accuracy_score(y_true, y_pred)
                elif metric == 'precision':
                    results[metric] = precision_score(y_true, y_pred, zero_division=0)
                elif metric == 'recall':
                    results[metric] = recall_score(y_true, y_pred, zero_division=0)
                elif metric == 'f1':
                    results[metric] = f1_score(y_true, y_pred, zero_division=0)
                elif metric == 'auc':
                    results[metric] = roc_auc_score(y_true, y_proba)
            except Exception as e:
                logger.warning(f"Could not compute metric {metric}: {e}")
                results[metric] = np.nan

        return results

    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        perform_cv: bool = True,
        cv_splits: int = 5
    ) -> Dict[str, Any]:
        """Train model and evaluate on test set.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction of data to use for testing
            perform_cv: Whether to perform cross-validation
            cv_splits: Number of CV splits

        Returns:
            Training results dictionary
        """
        # Split data chronologically
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

        # Perform CV if requested
        cv_results = None
        if perform_cv:
            cv_results = self.cross_validate(X_train, y_train, n_splits=cv_splits)

        # Train final model on all training data
        self.model.train(X_train, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        test_metrics = self._compute_metrics(
            y_test.values, y_pred, y_proba,
            ['accuracy', 'precision', 'recall', 'f1', 'auc']
        )

        results = {
            'model_name': self.model.name,
            'model_params': self.model.params,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'test_metrics': test_metrics,
            'cv_results': cv_results,
            'feature_importance': None
        }

        # Get feature importance if available
        try:
            importance = self.model.get_feature_importance()
            if importance is not None:
                results['feature_importance'] = importance.to_dict()
        except:
            pass

        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'results': results
        })

        return results

    def save_model(self, model_id: str = None) -> str:
        """Save the trained model.

        Args:
            model_id: Optional model identifier

        Returns:
            Path to saved model
        """
        if model_id is None:
            model_id = f"{self.model.name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        path = self.registry.save(self.model, model_id)
        logger.info(f"Model saved with ID: {model_id}")

        return path

    def grid_search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List[Any]],
        cv_splits: int = 3
    ) -> Dict[str, Any]:
        """Perform grid search for hyperparameter tuning.

        Args:
            X: Feature matrix
            y: Target vector
            param_grid: Dictionary of parameter names to lists of values
            cv_splits: Number of CV splits

        Returns:
            Grid search results
        """
        from itertools import product

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        logger.info(f"Testing {len(combinations)} parameter combinations")

        best_score = -np.inf
        best_params = None
        all_results = []

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            logger.info(f"Testing combination {i+1}/{len(combinations)}: {params}")

            # Create model with these parameters
            from .models import ModelFactory
            model_type = self.model.name.lower().replace(' ', '_')
            test_model = ModelFactory.create(model_type, params)

            # Perform CV
            trainer = ModelTrainer(test_model)
            cv_results = trainer.cross_validate(X, y, n_splits=cv_splits)

            mean_auc = cv_results['summary']['auc']['mean']

            result = {
                'params': params,
                'cv_results': cv_results,
                'mean_auc': mean_auc
            }
            all_results.append(result)

            if mean_auc > best_score:
                best_score = mean_auc
                best_params = params

        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results,
            'param_grid': param_grid
        }
