"""Analysis service for QuantTool."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from ..domain.interfaces.data_provider import IDataProvider
from ..core.registry import registry, ComponentType
from ..core.logging import get_logger
from ..infrastructure.data_providers.incremental_data_manager import IncrementalDataManager, DataType


logger = get_logger(__name__)


class AnalysisService:
    """Service class for financial analysis and reporting."""

    def __init__(self, use_incremental: bool = True):
        """Initialize analysis service.

        Args:
            use_incremental: 是否使用增量数据获取
        """
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

    def analyze_stock(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        data_provider: str = "tushare",
        timeframe: str = "1d",
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a single stock.

        Args:
            symbol: Symbol to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            data_provider: Data provider to use
            timeframe: Timeframe for the analysis

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing stock {symbol} from {start_date} to {end_date}")

        # Get data provider
        provider_class = registry.get(ComponentType.DATA_PROVIDER, data_provider)
        data_provider_instance = provider_class()
        if hasattr(data_provider_instance, "initialize"):
            data_provider_instance.initialize()

        # Get data (使用增量数据管理器)
        bars = self._get_data(symbol, start_date, end_date, data_provider_instance, timeframe)

        if bars.empty:
            raise ValueError(f"No data available for symbol {symbol}")

        # Calculate technical indicators
        analysis_result = self._calculate_technical_indicators(bars)

        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(bars)

        # Identify significant events
        events = self._identify_events(bars)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "period": {"start_date": start_date, "end_date": end_date},
            "technical_indicators": analysis_result,
            "risk_metrics": risk_metrics,
            "significant_events": events,
            "basic_stats": self._calculate_basic_stats(bars),
        }

    def _calculate_technical_indicators(self, bars: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various technical indicators."""
        df = bars.copy()

        # Calculate returns
        df["returns"] = df["close"].pct_change()

        # Moving averages
        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_10"] = df["close"].rolling(10).mean()
        df["ma_20"] = df["close"].rolling(20).mean()
        df["ma_50"] = df["close"].rolling(50).mean()

        # RSI (Relative Strength Index)
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

        # MACD
        exp1 = df["close"].ewm(span=12).mean()
        exp2 = df["close"].ewm(span=26).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # ATR (Average True Range)
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        df["atr"] = tr.rolling(14).mean()

        # Get the latest values for each indicator
        latest_data = df.iloc[-1] if not df.empty else pd.Series(dtype=object)

        indicators = {
            "moving_averages": {
                "ma_5": latest_data.get("ma_5"),
                "ma_10": latest_data.get("ma_10"),
                "ma_20": latest_data.get("ma_20"),
                "ma_50": latest_data.get("ma_50"),
            },
            "rsi": latest_data.get("rsi"),
            "bollinger_bands": {
                "middle": latest_data.get("bb_middle"),
                "upper": latest_data.get("bb_upper"),
                "lower": latest_data.get("bb_lower"),
                "position": latest_data.get("bb_position"),
            },
            "macd": {
                "macd": latest_data.get("macd"),
                "signal": latest_data.get("macd_signal"),
                "histogram": latest_data.get("macd_histogram"),
            },
            "atr": latest_data.get("atr"),
        }

        return indicators

    def _calculate_risk_metrics(self, bars: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics."""
        df = bars.copy()
        df["returns"] = df["close"].pct_change().dropna()

        # Basic stats
        returns = df["returns"].dropna()

        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() if not drawdown.empty else 0

        # Sharpe ratio (assuming risk-free rate of 0)
        avg_return = returns.mean() * 252  # Annualized
        sharpe_ratio = avg_return / volatility if volatility != 0 else 0

        # Value at Risk (VaR) at 95%
        var_95 = np.percentile(returns.dropna(), 5) if len(returns) > 0 else 0

        # Beta (relative to market would require market data, using simplified version)
        # Here we'll use the market as the equal weighted average of the entire dataset (which doesn't exist)
        # So we'll just use a simple proxy - the standard deviation
        beta = (
            volatility / np.std(returns)
            if len(returns) > 1 and np.std(returns) != 0
            else 1
        )

        return {
            "volatility": volatility,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "var_95": var_95,
            "beta": beta,
            "skewness": returns.skew() if len(returns) > 2 else 0,
            "kurtosis": returns.kurtosis() if len(returns) > 3 else 0,
        }

    def _identify_events(self, bars: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify significant market events."""
        df = bars.copy()
        df["returns"] = df["close"].pct_change()

        events = []

        # Large moves (> 3% in a single day)
        large_moves = df[np.abs(df["returns"]) > 0.03]
        for idx, row in large_moves.iterrows():
            events.append(
                {
                    "type": "large_move",
                    "date": row["timestamp"],
                    "direction": "up" if row["returns"] > 0 else "down",
                    "magnitude": row["returns"],
                    "close": row["close"],
                }
            )

        # Volume spikes (volume > 2x average)
        avg_volume = df["volume"].mean()
        volume_spikes = df[df["volume"] > 2 * avg_volume]
        for idx, row in volume_spikes.iterrows():
            events.append(
                {
                    "type": "volume_spike",
                    "date": row["timestamp"],
                    "volume": row["volume"],
                    "close": row["close"],
                    "ratio_to_average": row["volume"] / avg_volume,
                }
            )

        # New highs/lows (relative to 50-period high/low)
        df["high_50d"] = df["high"].rolling(50).max()
        df["low_50d"] = df["low"].rolling(50).min()

        new_highs = df[df["close"] >= df["high_50d"]]
        for idx, row in new_highs.iterrows():
            events.append(
                {
                    "type": "new_high",
                    "date": row["timestamp"],
                    "close": row["close"],
                    "period": "50d",
                }
            )

        new_lows = df[df["close"] <= df["low_50d"]]
        for idx, row in new_lows.iterrows():
            events.append(
                {
                    "type": "new_low",
                    "date": row["timestamp"],
                    "close": row["close"],
                    "period": "50d",
                }
            )

        # Sort events by date
        events.sort(key=lambda x: x["date"], reverse=True)

        return events[:10]  # Return top 10 most recent events

    def _calculate_basic_stats(self, bars: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistical measures."""
        df = bars.copy()
        df["returns"] = df["close"].pct_change()

        returns = df["returns"].dropna()

        return {
            "total_return": (
                (df["close"].iloc[-1] / df["close"].iloc[0] - 1) if len(df) > 0 else 0
            ),
            "avg_daily_return": returns.mean() if len(returns) > 0 else 0,
            "std_daily_return": returns.std() if len(returns) > 1 else 0,
            "best_day": returns.max() if len(returns) > 0 else 0,
            "worst_day": returns.min() if len(returns) > 0 else 0,
            "avg_volume": df["volume"].mean() if len(df) > 0 else 0,
            "trading_days": len(df),
            "start_price": df["close"].iloc[0] if len(df) > 0 else 0,
            "end_price": df["close"].iloc[-1] if len(df) > 0 else 0,
        }

    def analyze_portfolio(
        self,
        holdings: Dict[str, float],  # symbol -> weight
        start_date: datetime,
        end_date: datetime,
        data_provider: str = "tushare",
        timeframe: str = "1d",
    ) -> Dict[str, Any]:
        """
        Analyze a portfolio of holdings.

        Args:
            holdings: Dictionary mapping symbols to weights
            start_date: Start date for analysis
            end_date: End date for analysis
            data_provider: Data provider to use
            timeframe: Timeframe for the analysis

        Returns:
            Dictionary with portfolio analysis results
        """
        logger.info(
            f"Analyzing portfolio with {len(holdings)} holdings from {start_date} to {end_date}"
        )

        # Get data for all symbols
        symbols = list(holdings.keys())

        # Get data provider
        provider_class = registry.get(ComponentType.DATA_PROVIDER, data_provider)
        data_provider_instance = provider_class()
        if hasattr(data_provider_instance, "initialize"):
            data_provider_instance.initialize()

        # Get data (使用增量数据管理器)
        all_data = {}
        for symbol in symbols:
            df = self._get_data(symbol, start_date, end_date, data_provider_instance, timeframe)
            if not df.empty:
                all_data[symbol] = df

        # Ensure all required symbols have data
        missing_symbols = [s for s in symbols if s not in all_data or all_data[s].empty]
        if missing_symbols:
            logger.warning(f"Missing data for symbols: {missing_symbols}")

        # Filter to only symbols with data
        available_symbols = [
            s for s in symbols if s in all_data and not all_data[s].empty
        ]
        available_holdings = {s: holdings[s] for s in available_symbols}

        if not available_holdings:
            raise ValueError("No data available for any of the specified symbols")

        # Align data for all symbols (same timestamps)
        # Find common timestamps across all available symbols
        common_timestamps = None
        for symbol in available_symbols:
            symbol_timestamps = set(all_data[symbol]["timestamp"])
            if common_timestamps is None:
                common_timestamps = symbol_timestamps
            else:
                common_timestamps = common_timestamps.intersection(symbol_timestamps)

        if not common_timestamps:
            raise ValueError("No common timestamps found across symbols")

        common_timestamps = sorted(list(common_timestamps))

        # Create aligned dataset
        aligned_data = {}
        for symbol in available_symbols:
            # Filter to common timestamps
            symbol_data = all_data[symbol]
            mask = symbol_data["timestamp"].isin(common_timestamps)
            aligned_data[symbol] = symbol_data[mask].set_index("timestamp").sort_index()

        # Calculate portfolio metrics
        portfolio_result = self._calculate_portfolio_metrics(
            aligned_data, available_holdings
        )

        return {
            "holdings": available_holdings,
            "timeframe": timeframe,
            "period": {"start_date": start_date, "end_date": end_date},
            "portfolio_metrics": portfolio_result,
            "constituent_analysis": {
                symbol: self.analyze_stock(
                    symbol, start_date, end_date, data_provider, timeframe
                )
                for symbol in available_symbols
            },
        }

    def _calculate_portfolio_metrics(
        self, aligned_data: Dict[str, pd.DataFrame], holdings: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate portfolio-level metrics."""
        # Normalize weights to sum to 1
        total_weight = sum(holdings.values())
        normalized_weights = {s: w / total_weight for s, w in holdings.items()}

        # Calculate portfolio returns
        # First, ensure all assets have the same timestamps
        all_timestamps = sorted(
            set.union(*(set(df.index) for df in aligned_data.values()))
        )

        # Create a combined returns dataframe
        returns_df = pd.DataFrame(index=all_timestamps)

        for symbol, data in aligned_data.items():
            # Ensure all timestamps are represented
            data_full = data.reindex(all_timestamps)
            # Forward-fill to handle missing values
            data_full["close"] = data_full["close"].fillna(method="ffill")

            # Calculate returns
            data_full["returns"] = data_full["close"].pct_change()
            returns_df[symbol] = data_full["returns"]

        # Calculate weighted portfolio returns
        weights_series = pd.Series(normalized_weights)
        portfolio_returns = (returns_df * weights_series).sum(axis=1)

        # Calculate portfolio metrics
        portfolio_cumulative = (1 + portfolio_returns).cumprod()

        # Portfolio volatility (annualized)
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)

        # Portfolio return (annualized)
        portfolio_return = portfolio_returns.mean() * 252

        # Max drawdown
        running_max = portfolio_cumulative.expanding().max()
        drawdown = (portfolio_cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Sharpe ratio
        sharpe_ratio = (
            portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0
        )

        # Correlation matrix
        correlation_matrix = returns_df.corr()

        # Risk contribution by asset (simplified version)
        asset_vols = returns_df.std() * np.sqrt(252)
        weights_array = np.array(list(normalized_weights.values()))
        individual_risks = weights_array * asset_vols.values
        total_risk = portfolio_volatility * np.sqrt(252)
        risk_contributions = (
            individual_risks / total_risk
            if total_risk != 0
            else np.zeros(len(individual_risks))
        )

        risk_contribution_dict = {}
        for i, symbol in enumerate(normalized_weights.keys()):
            risk_contribution_dict[symbol] = (
                risk_contributions[i] if i < len(risk_contributions) else 0
            )

        return {
            "total_return": (
                portfolio_cumulative.iloc[-1] - 1
                if len(portfolio_cumulative) > 0
                else 0
            ),
            "annual_return": portfolio_return,
            "volatility": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "correlation_matrix": (
                correlation_matrix.to_dict() if not correlation_matrix.empty else {}
            ),
            "risk_contributions": risk_contribution_dict,
            "tracking_error": None,  # Would require benchmark data
            "information_ratio": None,  # Would require benchmark data
        }
