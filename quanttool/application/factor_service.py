"""Factor service for QuantTool."""

from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
from ..domain.interfaces.factor import IFactor
from ..domain.interfaces.data_provider import IDataProvider
from ..domain.models import FactorEvaluationResult
from ..core.registry import registry, ComponentType
from ..core.logging import get_logger
from ..infrastructure.data_providers.incremental.manager import IncrementalDataManager, DataType


logger = get_logger(__name__)


class FactorService:
    """Service class for factor mining and evaluation."""

    def __init__(self, use_incremental: bool = True):
        """Initialize factor service.

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
        data = data_provider_instance.get_bars([symbol], start_date, end_date, "1d")
        return data.get(symbol, pd.DataFrame())

    def mine_factor(
        self,
        factor_name: str,
        factor_params: Dict[str, Any],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        data_provider: str = "tushare",
    ) -> Dict[str, FactorEvaluationResult]:
        """
        Mine a factor across the specified universe of symbols.

        Args:
            factor_name: Name of the factor to mine
            factor_params: Parameters for the factor
            symbols: List of symbols to mine the factor for
            start_date: Start date for factor calculation
            end_date: End date for factor calculation
            data_provider: Name of the data provider to use

        Returns:
            Dictionary mapping symbols to factor evaluation results
        """
        logger.info(
            f"Mining factor {factor_name} for {len(symbols)} symbols "
            f"from {start_date} to {end_date}"
        )

        # Get factor from registry
        factor_class = registry.get(ComponentType.FACTOR, factor_name)
        factor = factor_class()
        factor.initialize(factor_params)

        # Get data provider
        provider_class = registry.get(ComponentType.DATA_PROVIDER, data_provider)
        data_provider_instance = provider_class()
        if hasattr(data_provider_instance, "initialize"):
            data_provider_instance.initialize()

        # Get data (使用增量数据管理器)
        data = {}
        for symbol in symbols:
            df = self._get_data(symbol, start_date, end_date, data_provider_instance)
            if not df.empty:
                data[symbol] = df

        results = {}

        for symbol, bars in data.items():
            logger.info(f"Calculating factor for {symbol}")

            # Calculate factor values
            factor_values = factor.compute(bars)

            # Combine factor values with price data to calculate returns
            combined_data = pd.merge(
                bars[["timestamp", "close"]], factor_values, on="timestamp", how="inner"
            )

            # Calculate forward returns (for example, 5-day forward returns)
            combined_data["future_return"] = (
                combined_data["close"].shift(-5) / combined_data["close"] - 1
            )

            # Calculate factor metrics
            ic = self._calculate_ic(
                combined_data["factor_value"], combined_data["future_return"]
            )
            rank_ic = self._calculate_rank_ic(
                combined_data["factor_value"], combined_data["future_return"]
            )

            # Basic performance metrics
            avg_return = combined_data["future_return"].mean()
            volatility = combined_data["future_return"].std()
            sharpe_ratio = avg_return / volatility if volatility != 0 else 0

            # Win rate (percentage of positive returns when factor is in top quantile)
            top_quantile = combined_data["factor_value"].quantile(0.8)
            top_quintile_returns = combined_data[
                combined_data["factor_value"] >= top_quantile
            ]["future_return"]
            win_rate = (
                (top_quintile_returns > 0).sum() / len(top_quintile_returns)
                if len(top_quintile_returns) > 0
                else 0
            )

            # Create result
            result = FactorEvaluationResult(
                factor_name=factor_name,
                ic=ic,
                rank_ic=rank_ic,
                ic_ir=(
                    ic
                    / self._calculate_ic_std(
                        combined_data["factor_value"], combined_data["future_return"]
                    )
                    if self._calculate_ic_std(
                        combined_data["factor_value"], combined_data["future_return"]
                    )
                    != 0
                    else 0
                ),
                win_rate=win_rate,
                avg_return=avg_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                turnover=0.0,  # Placeholder
                max_exposure=1.0,  # Placeholder
                data=combined_data,
            )

            results[symbol] = result

        return results

    def _calculate_ic(self, factor_values: pd.Series, returns: pd.Series) -> float:
        """Calculate information coefficient (correlation between factor and returns)."""
        # Drop NaN values
        clean_data = pd.concat([factor_values, returns], axis=1).dropna()

        if len(clean_data) < 2:
            return 0.0

        correlation = clean_data.iloc[:, 0].corr(clean_data.iloc[:, 1])
        return correlation if pd.notna(correlation) else 0.0

    def _calculate_rank_ic(self, factor_values: pd.Series, returns: pd.Series) -> float:
        """Calculate rank information coefficient."""
        # Drop NaN values
        clean_data = pd.concat([factor_values, returns], axis=1).dropna()

        if len(clean_data) < 2:
            return 0.0

        # Rank the factor values and returns
        ranked_factors = clean_data.iloc[:, 0].rank(pct=True)
        ranked_returns = clean_data.iloc[:, 1].rank(pct=True)

        correlation = ranked_factors.corr(ranked_returns)
        return correlation if pd.notna(correlation) else 0.0

    def _calculate_ic_std(self, factor_values: pd.Series, returns: pd.Series) -> float:
        """Calculate standard deviation of IC."""
        # This is a simplified version - in practice, you'd calculate IC over time periods
        # and then calculate the standard deviation of those IC values
        ic = self._calculate_ic(factor_values, returns)
        # For now, returning absolute IC as a simple proxy for standard deviation
        return abs(ic) if abs(ic) > 0.01 else 0.01  # Avoid division by zero
