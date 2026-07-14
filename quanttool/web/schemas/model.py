"""GBM and Qlib model API schemas."""

from datetime import datetime, timedelta
from typing import List, Optional

from pydantic import BaseModel


class QlibTrainRequest(BaseModel):
    """Qlib model training request."""

    model_type: str = "lgb"
    symbols: List[str] = []
    train_start: str = "2019-01-01"
    train_end: str = "2022-12-31"
    valid_start: str = "2023-01-01"
    valid_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2026-03-17"
    features: List[str] = []
    use_rich_features: bool = True
    feature_set: str = "Alpha158"
    label: str = "return_5d"
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.01
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    epochs: int = 100
    batch_size: int = 256
    early_stopping_rounds: int = 20
    n_head: int = 4
    max_train_stocks: int = 0
    initial_cash: float = 100000.0
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0001


class QlibPredictRequest(BaseModel):
    """Qlib model prediction request."""

    model_type: str = "lgb"
    model_path: str = ""
    symbols: List[str] = []
    features: List[str] = []
    use_rich_features: bool = True
    feature_set: str = "Alpha158"
    predict_start_date: Optional[str] = None
    predict_end_date: Optional[str] = None
    initial_cash: float = 100000.0
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0001

    def get_predict_start_date(self) -> str:
        """Return the prediction start date, defaulting to one year ago."""
        if self.predict_start_date:
            return self.predict_start_date
        return (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    def get_predict_end_date(self) -> str:
        """Return the prediction end date, defaulting to today."""
        if self.predict_end_date:
            return self.predict_end_date
        return datetime.now().strftime("%Y-%m-%d")


class GBMTrainRequest(BaseModel):
    """GBM training request."""

    symbols: List[str] = []
    max_train_stocks: int = 0


class GBMPredictRequest(BaseModel):
    """GBM prediction request."""

    model_path: str = ""
    symbols: List[str] = []


class GBMPicksRequest(BaseModel):
    """GBM stock-pick request."""

    top_n: int = 10
    force_train: bool = False
    model_path: Optional[str] = None
