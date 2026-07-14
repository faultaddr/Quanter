"""Prediction service for QuantTool."""

from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from ..domain.interfaces.model import IModel
from ..domain.interfaces.data_provider import IDataProvider
from ..core.registry import registry, ComponentType
from ..core.logging import get_logger
from ..infrastructure.data_providers.incremental.manager import IncrementalDataManager, DataType


logger = get_logger(__name__)


class PredictionService:
    """Service class for predictive modeling."""

    def __init__(self, use_incremental: bool = True):
        """Initialize prediction service.

        Args:
            use_incremental: 是否使用增量数据获取
        """
        self.default_horizon = 6  # Default prediction horizon (6 x 10min = 60 minutes)
        self.use_incremental = use_incremental
        self._incremental_manager: Optional[IncrementalDataManager] = None
        if use_incremental:
            try:
                self._incremental_manager = IncrementalDataManager()
                logger.info("增量数据管理器初始化成功")
            except Exception as e:
                logger.warning(f"增量数据管理器初始化失败: {e}")
                self._incremental_manager = None

    def _get_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        data_provider_instance,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """获取股票数据（优先使用增量数据管理器）"""
        # 优先使用增量数据管理器
        if self._incremental_manager:
            try:
                df = self._incremental_manager.get_data(
                    symbol,
                    start_date,
                    end_date,
                    data_provider_instance,
                    data_type=DataType.STOCK_BAR,
                )
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"增量获取失败 {symbol}: {e}，回退到直接获取")

        # 回退到直接获取
        data = data_provider_instance.get_bars(
            [symbol], start_date, end_date, timeframe
        )
        return data.get(symbol, pd.DataFrame())

    def prepare_features(self, bars: pd.DataFrame, horizon: int = 6) -> pd.DataFrame:
        """
        Prepare features for prediction model.

        Args:
            bars: Input price data
            horizon: Prediction horizon in number of bars

        Returns:
            DataFrame with features prepared for modeling
        """
        df = bars.copy()

        # Technical indicators as features
        df["returns"] = df["close"].pct_change()
        df["returns_lag1"] = df["returns"].shift(1)
        df["returns_lag2"] = df["returns"].shift(2)

        # Moving averages
        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_10"] = df["close"].rolling(10).mean()
        df["ma_20"] = df["close"].rolling(20).mean()

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )

        # Volume features
        df["volume_sma"] = df["volume"].rolling(10).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # Volatility
        df["volatility"] = df["returns"].rolling(10).std()

        # Future returns (our target variable)
        df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
        df["target"] = (df["future_return"] > 0).astype(
            int
        )  # Binary classification: 1 if positive return, 0 otherwise

        # Select feature columns
        feature_columns = [
            "returns_lag1",
            "returns_lag2",
            "ma_5",
            "ma_10",
            "ma_20",
            "rsi",
            "bb_position",
            "volume_ratio",
            "volatility",
        ]

        # Create the feature matrix
        features_df = df[feature_columns + ["target"]].copy()

        # Drop rows with NaN values
        features_df = features_df.dropna()

        return features_df

    def train_model(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        horizon: int = None,
        model_type: str = "logistic_regression",
        data_provider: str = "tushare",
        timeframe: str = "10m",
    ) -> Dict[str, Any]:
        """
        Train a prediction model for a symbol.

        Args:
            symbol: Symbol to train model for
            start_date: Start date for training data
            end_date: End date for training data
            horizon: Prediction horizon (default: use service default)
            model_type: Type of model to train
            data_provider: Data provider to use
            timeframe: Timeframe for the data

        Returns:
            Dictionary with model performance and metadata
        """
        horizon = horizon or self.default_horizon

        logger.info(
            f"Training {model_type} model for {symbol} with horizon {horizon} bars"
        )

        # Get data provider
        provider_class = registry.get(ComponentType.DATA_PROVIDER, data_provider)
        data_provider_instance = provider_class()
        if hasattr(data_provider_instance, "initialize"):
            data_provider_instance.initialize()

        # Get data (使用增量数据管理器)
        bars = self._get_data(symbol, start_date, end_date, data_provider_instance, timeframe)

        if bars.empty:
            raise ValueError(f"No data available for symbol {symbol}")

        # Prepare features
        features_df = self.prepare_features(bars, horizon)

        if features_df.empty or len(features_df) < 50:  # Need sufficient data
            raise ValueError(f"Not enough data to train model for {symbol}")

        # Separate features and target
        X = features_df.drop("target", axis=1)
        y = features_df["target"]

        # Perform time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        models = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train model
            if model_type == "logistic_regression":
                model = LogisticRegression(random_state=42, max_iter=1000)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            model.fit(X_train, y_train)

            # Validate model
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[
                :, 1
            ]  # Probability of positive return

            accuracy = accuracy_score(y_val, y_pred)
            auc_score = roc_auc_score(y_val, y_pred_proba)

            cv_scores.append(
                {"fold": len(cv_scores), "accuracy": accuracy, "auc": auc_score}
            )

            models.append(model)

        # Use the best performing model (highest AUC) for final model
        best_fold_idx = max(range(len(cv_scores)), key=lambda i: cv_scores[i]["auc"])
        final_model = models[best_fold_idx]

        # Calculate overall metrics
        overall_metrics = {
            "cv_mean_accuracy": np.mean([score["accuracy"] for score in cv_scores]),
            "cv_std_accuracy": np.std([score["accuracy"] for score in cv_scores]),
            "cv_mean_auc": np.mean([score["auc"] for score in cv_scores]),
            "cv_std_auc": np.std([score["auc"] for score in cv_scores]),
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "class_balance": y.value_counts(normalize=True).to_dict(),
        }

        # Generate a model ID
        import hashlib

        model_id = hashlib.md5(
            f"{symbol}_{start_date}_{end_date}_{horizon}_{model_type}".encode()
        ).hexdigest()[:12]

        logger.info(f"Model trained successfully for {symbol}. Model ID: {model_id}")

        return {
            "model_id": model_id,
            "symbol": symbol,
            "horizon": horizon,
            "model_type": model_type,
            "start_date": start_date,
            "end_date": end_date,
            "timeframe": timeframe,
            "cv_scores": cv_scores,
            "metrics": overall_metrics,
            "features": list(X.columns),
            "model": final_model,  # In a real implementation, you'd save this to a file
        }

    def predict(
        self,
        model_id: str,
        symbol: str,
        data: pd.DataFrame = None,
        data_provider: str = "tushare",
        timeframe: str = "10m",
    ) -> Dict[str, Any]:
        """
        Make predictions using a trained model.

        Args:
            model_id: ID of the trained model
            symbol: Symbol to predict for
            data: New data to make predictions on (if None, fetch latest)
            data_provider: Data provider to use if fetching data
            timeframe: Timeframe for the data

        Returns:
            Dictionary with predictions
        """
        logger.info(f"Making predictions for {symbol} using model {model_id}")

        # In a real implementation, you would load the trained model
        # For this example, we'll demonstrate the process with a fresh model

        if data is None:
            # Get the latest data for prediction
            provider_class = registry.get(ComponentType.DATA_PROVIDER, data_provider)
            data_provider_instance = provider_class()
            if hasattr(data_provider_instance, "initialize"):
                data_provider_instance.initialize()

            # Get recent data (last 30 bars to have enough features)
            end_date = datetime.now()
            # For 10min bars, 30 bars = 5 hours of data
            start_date = end_date - pd.Timedelta(hours=5)

            # 使用增量数据管理器
            bars = self._get_data(symbol, start_date, end_date, data_provider_instance, timeframe)

            if bars.empty:
                raise ValueError(f"No data available for prediction for {symbol}")
        else:
            bars = data

        # Prepare features
        # We need to adapt our feature preparation for making forward-looking predictions
        features_df = self.prepare_features(bars, horizon=self.default_horizon)

        if features_df.empty:
            raise ValueError(f"No valid features could be computed for {symbol}")

        # Take the last row for prediction (most recent data)
        latest_features = features_df.drop("target", axis=1).iloc[[-1]]

        # For this example, we'll create a simple model on the fly
        # In a real implementation, you would load the saved model
        X = features_df.drop("target", axis=1)
        y = features_df["target"]

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)

        # Make predictions
        prediction_proba = model.predict_proba(latest_features)[
            0
        ]  # Probabilities for each class
        prediction = model.predict(latest_features)[0]  # Predicted class

        # Get feature importances
        feature_importance = pd.Series(
            np.abs(model.coef_[0]), index=X.columns
        ).sort_values(ascending=False)

        return {
            "model_id": model_id,
            "symbol": symbol,
            "prediction": int(prediction),  # Convert numpy.int64 to Python int
            "probability_positive": float(
                prediction_proba[1]
            ),  # Convert numpy.float64 to Python float
            "probability_negative": float(prediction_proba[0]),
            "feature_importance": feature_importance.to_dict(),
            "timestamp": datetime.now(),
            "features_used": list(latest_features.columns),
        }
