"""Abstract interface for models in QuantTool."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import pandas as pd


class IModel(ABC):
    """Abstract interface for predictive models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model on input features and target.

        Args:
            X: Feature matrix
            y: Target values
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Make predictions on input features.

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get prediction probabilities (for classification).

        Args:
            X: Feature matrix

        Returns:
            DataFrame with probability for each class
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance scores.

        Returns:
            Series with feature names as index and importance scores as values
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        pass

    @abstractmethod
    def set_params(self, **params) -> None:
        """
        Set model parameters.

        Args:
            **params: Model parameters to set
        """
        pass
