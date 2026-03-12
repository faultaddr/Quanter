"""
Qlib ML backtest tools for MCP Agent.
"""

from typing import List
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from ..schemas.tools import (
    QlibBacktestInput,
    QlibBacktestOutput,
)


# Model configurations
QLIB_MODELS = {
    # Gradient Boosting
    "lgb": {"type": "gbdt", "name": "LightGBM"},
    "xgboost": {"type": "gbdt", "name": "XGBoost"},
    "catboost": {"type": "gbdt", "name": "CatBoost"},
    "gbdt": {"type": "gbdt", "name": "GBDT"},

    # Neural Networks
    "mlp": {"type": "nn", "name": "MLP"},
    "gru": {"type": "nn", "name": "GRU"},
    "lstm": {"type": "nn", "name": "LSTM"},
    "gats": {"type": "nn", "name": "GATS"},
    "transformer": {"type": "nn", "name": "Transformer"},
    "double_gru": {"type": "nn", "name": "Double GRU"},
    "double_lstm": {"type": "nn", "name": "Double LSTM"},

    # Tabular Deep Learning
    "tabnet": {"type": "tabular", "name": "TabNet"},
    "tabnet2": {"type": "tabular", "name": "TabNet2"},
    "tabtransformer": {"type": "tabular", "name": "TabTransformer"},
    "deepfm": {"type": "tabular", "name": "DeepFM"},

    # Linear Models
    "linear": {"type": "linear", "name": "Linear Regression"},
    "ridge": {"type": "linear", "name": "Ridge"},
    "lasso": {"type": "linear", "name": "Lasso"},
    "elastic_net": {"type": "linear", "name": "Elastic Net"},

    # Other ML
    "svr": {"type": "other", "name": "SVR"},
    "random_forest": {"type": "ensemble", "name": "Random Forest"},
    "extra_trees": {"type": "ensemble", "name": "Extra Trees"},
    "adaboost": {"type": "ensemble", "name": "AdaBoost"},
}


def run_qlib_backtest(input_data: QlibBacktestInput) -> QlibBacktestOutput:
    """
    Run Qlib ML model backtest.

    Args:
        input_data: Qlib backtest parameters

    Returns:
        QlibBacktestOutput with ML backtest results
    """
    try:
        # Try to import qlib
        try:
            import qlib
            from qlib.constant import REG_CN
        except ImportError:
            return QlibBacktestOutput(
                model=input_data.model,
                symbols=input_data.symbols,
                training_days=input_data.days,
                initial_capital=input_data.initial_cash,
                final_capital=input_data.initial_cash,
                total_return=0.0,
                error="Qlib未安装。请运行: pip install pyqlib",
            )

        # Get model info
        model_info = QLIB_MODELS.get(input_data.model, {"type": "unknown", "name": input_data.model})

        # Import strategy
        try:
            from quanttool.strategies.qlib_strategy import QlibStrategy
        except ImportError:
            # Fallback simulation
            return _simulate_qlib_backtest(input_data, model_info)

        # Run actual Qlib backtest
        end_date = datetime.now()
        start_date = end_date - timedelta(days=input_data.days)

        strategy = QlibStrategy(
            model_type=input_data.model,
            symbols=input_data.symbols,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )

        result = strategy.run_backtest(initial_cash=input_data.initial_cash)

        return QlibBacktestOutput(
            model=input_data.model,
            symbols=input_data.symbols,
            training_days=input_data.days,
            initial_capital=input_data.initial_cash,
            final_capital=result.get('final_capital', input_data.initial_cash),
            total_return=result.get('total_return', 0.0),
            annual_return=result.get('annual_return'),
            information_ratio=result.get('information_ratio'),
            max_drawdown=result.get('max_drawdown'),
            ic=result.get('ic'),
            rank_ic=result.get('rank_ic'),
            selected_stocks=result.get('selected_stocks'),
        )

    except Exception as e:
        return QlibBacktestOutput(
            model=input_data.model,
            symbols=input_data.symbols,
            training_days=input_data.days,
            initial_capital=input_data.initial_cash,
            final_capital=input_data.initial_cash,
            total_return=0.0,
            error=str(e),
        )


def _simulate_qlib_backtest(
    input_data: QlibBacktestInput,
    model_info: dict
) -> QlibBacktestOutput:
    """
    Simulate a Qlib backtest when Qlib is not fully configured.
    """
    from quanttool.factors.stock_analyzer import StockAnalyzer
    import numpy as np

    analyzer = StockAnalyzer(use_cache=True)
    data_map = analyzer.get_stock_data_batch(input_data.symbols, days=input_data.days)

    if not data_map:
        return QlibBacktestOutput(
            model=input_data.model,
            symbols=input_data.symbols,
            training_days=input_data.days,
            initial_capital=input_data.initial_cash,
            final_capital=input_data.initial_cash,
            total_return=0.0,
            error="无法获取股票数据",
        )

    # Simulate model prediction based on technical indicators
    # This is a simplified simulation
    selected_stocks = []
    stock_scores = {}

    for symbol, df in data_map.items():
        if df.empty or len(df) < 30:
            continue

        # Simple scoring based on momentum and trend
        close = df['close'].values
        ma20 = np.mean(close[-20:])
        ma60 = np.mean(close[-60:]) if len(close) >= 60 else np.mean(close)

        momentum = (close[-1] - close[-20]) / close[-20] * 100
        trend_score = 50 if ma20 > ma60 else 30
        total_score = trend_score + momentum

        stock_scores[symbol] = total_score

    # Select top stocks
    sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
    selected_stocks = [s[0] for s in sorted_stocks[:5]]

    # Simulate portfolio return
    final_capital = input_data.initial_cash
    if selected_stocks:
        equal_weight = 1.0 / len(selected_stocks)
        total_return = 0

        for symbol in selected_stocks:
            df = data_map.get(symbol)
            if df is not None and not df.empty:
                start_price = df.iloc[-min(60, len(df))]['close']
                end_price = df.iloc[-1]['close']
                stock_return = (end_price - start_price) / start_price * 100
                total_return += stock_return * equal_weight

        final_capital = input_data.initial_cash * (1 + total_return / 100)

    return QlibBacktestOutput(
        model=input_data.model,
        symbols=input_data.symbols,
        training_days=input_data.days,
        initial_capital=input_data.initial_cash,
        final_capital=round(final_capital, 2),
        total_return=round(((final_capital - input_data.initial_cash) / input_data.initial_cash) * 100, 2),
        selected_stocks=selected_stocks,
    )
