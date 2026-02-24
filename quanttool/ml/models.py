"""Machine learning models for QuantTool."""

import pickle
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from ..core.logging import get_logger
from ..core.registry import registry, ComponentType


logger = get_logger(__name__)


class BaseModel(ABC):
    """Base class for all ML models."""

    def __init__(self, name: str, params: Dict[str, Any] = None):
        """Initialize the model.

        Args:
            name: Model name
            params: Model parameters
        """
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model.

        Args:
            X: Feature matrix
            y: Target vector
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions.

        Args:
            X: Feature matrix

        Returns:
            Probability predictions
        """
        pass

    def save(self, path: str) -> None:
        """Save the model to disk.

        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'name': self.name,
                'params': self.params,
                'is_trained': self.is_trained
            }, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load the model from disk.

        Args:
            path: Path to load the model from
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.name = data['name']
            self.params = data['params']
            self.is_trained = data['is_trained']

        logger.info(f"Model loaded from {path}")

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance if available.

        Returns:
            Feature importance series or None
        """
        return None


@registry.register(ComponentType.MODEL, "random_forest")
class RandomForestModel(BaseModel):
    """Random Forest model implementation."""

    def __init__(self, params: Dict[str, Any] = None):
        """Initialize Random Forest model.

        Args:
            params: Model parameters
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        if params:
            default_params.update(params)

        super().__init__("RandomForest", default_params)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the Random Forest model."""
        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y)
        self.is_trained = True

        logger.info(f"Random Forest model trained on {len(X)} samples")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance."""
        if not self.is_trained:
            raise ValueError("Model must be trained")

        importances = self.model.feature_importances_
        return pd.Series(importances, index=self.model.feature_names_in_)


@registry.register(ComponentType.MODEL, "xgboost")
class XGBoostModel(BaseModel):
    """XGBoost model implementation."""

    def __init__(self, params: Dict[str, Any] = None):
        """Initialize XGBoost model.

        Args:
            params: Model parameters
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        if params:
            default_params.update(params)

        super().__init__("XGBoost", default_params)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the XGBoost model."""
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.error("XGBoost not installed. Install with: pip install xgboost")
            raise

        self.model = XGBClassifier(**self.params)
        self.model.fit(X, y)
        self.is_trained = True

        logger.info(f"XGBoost model trained on {len(X)} samples")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance."""
        if not self.is_trained:
            raise ValueError("Model must be trained")

        importances = self.model.feature_importances_
        return pd.Series(importances, index=self.model.feature_names_in_)


@registry.register(ComponentType.MODEL, "lstm")
class LSTMModel(BaseModel):
    """LSTM neural network model for time series prediction."""

    def __init__(self, params: Dict[str, Any] = None):
        """Initialize LSTM model.

        Args:
            params: Model parameters
        """
        default_params = {
            'sequence_length': 20,
            'units': [64, 32],
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 32,
            'random_state': 42
        }
        if params:
            default_params.update(params)

        super().__init__("LSTM", default_params)
        self.scaler = None

    def _create_sequences(self, X: pd.DataFrame, y: pd.Series = None):
        """Create sequences for LSTM input."""
        seq_len = self.params['sequence_length']
        X_values = X.values

        sequences = []
        targets = []

        for i in range(len(X_values) - seq_len):
            sequences.append(X_values[i:i + seq_len])
            if y is not None:
                targets.append(y.iloc[i + seq_len])

        return np.array(sequences), np.array(targets) if y is not None else None

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the LSTM model."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
            raise

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)

        # Build model
        tf.random.set_seed(self.params['random_state'])

        model = Sequential()
        for i, units in enumerate(self.params['units']):
            if i == 0:
                model.add(LSTM(units, return_sequences=(i < len(self.params['units']) - 1),
                             input_shape=(X_seq.shape[1], X_seq.shape[2])))
            else:
                model.add(LSTM(units, return_sequences=(i < len(self.params['units']) - 1)))
            model.add(Dropout(self.params['dropout']))

        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Train
        model.fit(
            X_seq, y_seq,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            validation_split=0.2,
            verbose=0
        )

        self.model = model
        self.is_trained = True

        logger.info(f"LSTM model trained on {len(X_seq)} sequences")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

        X_seq, _ = self._create_sequences(X_scaled)

        if len(X_seq) == 0:
            return np.array([])

        predictions = self.model.predict(X_seq, verbose=0)
        return (predictions > 0.5).astype(int).flatten()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

        X_seq, _ = self._create_sequences(X_scaled)

        if len(X_seq) == 0:
            return np.array([])

        proba = self.model.predict(X_seq, verbose=0).flatten()
        return np.column_stack([1 - proba, proba])

    def save(self, path: str) -> None:
        """Save the model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        model_path = save_path.with_suffix('.keras')
        self.model.save(model_path)

        # Save scaler and params
        with open(save_path, 'wb') as f:
            pickle.dump({
                'name': self.name,
                'params': self.params,
                'is_trained': self.is_trained,
                'scaler': self.scaler
            }, f)

        logger.info(f"LSTM model saved to {path}")

    def load(self, path: str) -> None:
        """Load the model from disk."""
        try:
            import tensorflow as tf
        except ImportError:
            logger.error("TensorFlow not installed")
            raise

        # Load metadata
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.name = data['name']
            self.params = data['params']
            self.is_trained = data['is_trained']
            self.scaler = data['scaler']

        # Load Keras model
        model_path = Path(path).with_suffix('.keras')
        self.model = tf.keras.models.load_model(model_path)

        logger.info(f"LSTM model loaded from {path}")


class ModelFactory:
    """Factory for creating ML models."""

    @staticmethod
    def create(model_type: str, params: Dict[str, Any] = None) -> BaseModel:
        """Create a model instance.

        Args:
            model_type: Type of model to create
            params: Model parameters

        Returns:
            Model instance
        """
        model_class = registry.get(ComponentType.MODEL, model_type)
        return model_class(params)


class ModelRegistry:
    """Registry for managing trained models."""

    def __init__(self, models_dir: str = "./models"):
        """Initialize the model registry.

        Args:
            models_dir: Directory for saving/loading models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_models: Dict[str, BaseModel] = {}

    def save(self, model: BaseModel, model_id: str) -> str:
        """Save a model to the registry.

        Args:
            model: Model to save
            model_id: Unique model identifier

        Returns:
            Path to saved model
        """
        path = self.models_dir / f"{model_id}.pkl"
        model.save(path)
        return str(path)

    def load(self, model_id: str) -> BaseModel:
        """Load a model from the registry.

        Args:
            model_id: Model identifier

        Returns:
            Loaded model
        """
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        path = self.models_dir / f"{model_id}.pkl"

        # Determine model type from metadata
        with open(path, 'rb') as f:
            data = pickle.load(f)
            model_type = data['name'].lower().replace(' ', '_')

        # Create appropriate model instance
        if 'lstm' in model_type:
            model = LSTMModel()
        elif 'xgboost' in model_type:
            model = XGBoostModel()
        elif 'random_forest' in model_type:
            model = RandomForestModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.load(path)
        self._loaded_models[model_id] = model

        return model

    def list_models(self) -> list:
        """List all saved models."""
        return [f.stem for f in self.models_dir.glob("*.pkl")]

    def delete(self, model_id: str) -> None:
        """Delete a model from the registry."""
        path = self.models_dir / f"{model_id}.pkl"

        if path.exists():
            path.unlink()

        # Also delete Keras model if exists
        keras_path = path.with_suffix('.keras')
        if keras_path.exists():
            keras_path.unlink()

        if model_id in self._loaded_models:
            del self._loaded_models[model_id]
