"""Backtest service for QuantTool."""

from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
from ..domain.models import BacktestResult, Trade, Metric
from ..backtest.engine import BacktestEngine
from ..core.registry import registry, ComponentType
from ..core.logging import get_logger

logger = get_logger(__name__)


class BacktestService:
    """Service class for running backtests."""

    def __init__(self, use_qlib: bool = True, use_realtime_price: bool = False):
        """Initialize backtest service.

        Args:
            use_qlib: 是否使用 qlib 数据（默认True，qlib数据是完整数据集）
            use_realtime_price: 优先使用实时股价，避免复权价格显示异常（默认False）
        """
        self.engine = BacktestEngine()
        self.use_qlib = use_qlib
        self.use_realtime_price = use_realtime_price
        self._qlib_loader = None
        self._stock_analyzer = None

        if use_qlib and not use_realtime_price:
            try:
                from ..infrastructure.data_providers.qlib_data_loader import QlibDataLoader
                self._qlib_loader = QlibDataLoader()
                logger.info("使用 qlib 数据加载器")
            except Exception as e:
                logger.warning(f"qlib 数据加载器初始化失败: {e}")

        if use_realtime_price:
            try:
                from ..factors.stock_analyzer import StockAnalyzer
                self._stock_analyzer = StockAnalyzer(use_realtime_price=True)
                logger.info("使用实时价格数据")
            except Exception as e:
                logger.warning(f"StockAnalyzer 初始化失败: {e}")

    def _get_data_qlib(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """使用 qlib 获取股票数据"""
        if not self._qlib_loader:
            return pd.DataFrame()

        try:
            df = self._qlib_loader.load_stock_data(symbol, start_date, end_date)
            if df.empty:
                return pd.DataFrame()

            # 重命名列以适配回测引擎（需要 timestamp 列）
            df = df.reset_index()
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})
            elif 'trade_date' in df.columns:
                df = df.rename(columns={'trade_date': 'timestamp'})

            # 移除 instrument 列（如果存在）
            if 'instrument' in df.columns:
                df = df.drop(columns=['instrument'])

            return df
        except Exception as e:
            logger.error(f"qlib 获取数据失败 {symbol}: {e}")
            return pd.DataFrame()

    def _get_data_realtime(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """使用 StockAnalyzer 获取实时价格数据"""
        if not self._stock_analyzer:
            return pd.DataFrame()

        try:
            # 计算天数
            from datetime import datetime as dt
            start_dt = dt.strptime(start_date, '%Y-%m-%d')
            end_dt = dt.strptime(end_date, '%Y-%m-%d')
            days = (end_dt - start_dt).days + 30  # 多加载一些数据确保覆盖

            df = self._stock_analyzer.get_stock_data(symbol, days)
            if df.empty:
                return pd.DataFrame()

            # 过滤日期范围
            if 'trade_date' in df.columns:
                df = df.rename(columns={'trade_date': 'timestamp'})
            elif 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})

            if 'timestamp' in df.columns:
                df = df[df['timestamp'] >= start_dt]
                df = df[df['timestamp'] <= end_dt]

            return df
        except Exception as e:
            logger.error(f"实时数据获取失败 {symbol}: {e}")
            return pd.DataFrame()

    def run_backtest(
        self,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
        initial_cash: float = 100000.0,
        commission_rate: float = 0.0003,
    ) -> BacktestResult:
        """
        Run a backtest with the specified parameters.

        每只股票独立回测，各自使用初始资金。

        Args:
            strategy_name: Name of the strategy to use
            strategy_params: Parameters for the strategy
            symbols: List of symbols to trade
            start_date: Start date for the backtest
            end_date: End date for the backtest
            timeframe: Timeframe for the backtest
            initial_cash: Initial capital for each stock
            commission_rate: Commission rate per trade

        Returns:
            Backtest result object (aggregated across all symbols)
        """
        logger.info(
            f"Starting backtest for strategy: {strategy_name}, symbols: {symbols}, "
            f"timeframe: {timeframe}, period: {start_date} to {end_date}"
        )

        # Get data (根据配置选择数据源)
        start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)

        data = {}
        for symbol in symbols:
            # 优先使用实时价格数据，否则使用 qlib 数据
            if self.use_realtime_price:
                df = self._get_data_realtime(symbol, start_str, end_str)
            else:
                df = self._get_data_qlib(symbol, start_str, end_str)
            if not df.empty:
                data[symbol] = df
                logger.info(f"加载 {symbol} 数据: {len(df)} 条")

        if not data:
            logger.error("没有获取到任何数据")
            raise ValueError("回测数据为空，请检查股票代码和日期范围")

        # 为每只股票独立运行回测
        all_results = []
        all_trades = []

        for symbol, df in data.items():
            # 为每只股票创建新的策略实例
            strategy_class = registry.get(ComponentType.STRATEGY, strategy_name)
            strategy = strategy_class()
            strategy.initialize(strategy_params)

            # 如果是 GBM 策略，自动加载最新模型
            if strategy_name == "gbm":
                import glob
                import os
                model_files = glob.glob("models/gbm/lgbm_*.pkl")
                if model_files:
                    model_path = max(model_files, key=os.path.getmtime)
                    logger.info(f"GBM 策略自动加载模型: {model_path}")
                    strategy.load_model(model_path)
                else:
                    logger.warning("GBM 策略未找到模型，回测将不会产生交易信号")

            # 配置引擎
            self.engine.set_initial_cash(initial_cash)
            self.engine.set_commission_rate(commission_rate)
            # 独立回测模式下，允许使用全部资金
            self.engine.set_max_position_size(1.0)  # 100% 仓位

            # 运行单只股票回测
            symbol_data = {symbol: df}
            result = self.engine.run_backtest(strategy, symbol_data, start_date, end_date)

            # 为交易记录添加股票标识
            for trade in result.trades:
                trade.symbol = symbol

            all_results.append(result)
            all_trades.extend(result.trades)
            logger.info(f"{symbol} 回测完成: 收益率 {result.total_return:.2%}")

        # 汇总所有股票的结果
        total_initial_cash = initial_cash * len(data)
        total_final_value = sum(r.final_capital for r in all_results)
        total_return = (total_final_value - total_initial_cash) / total_initial_cash

        # 计算年化收益
        days_diff = (end_date - start_date).days
        annual_return = (
            ((total_final_value / total_initial_cash) ** (365.0 / days_diff) - 1)
            if days_diff > 0
            else 0.0
        )

        # 合并权益曲线
        equity_curve_list = []
        for result in all_results:
            equity_curve_list.extend(result.equity_curve)

        # 计算综合指标
        avg_win_rate = sum(r.win_rate for r in all_results) / len(all_results) if all_results else 0
        total_trades = len(all_trades)

        logger.info(
            f"Backtest completed. Total stocks: {len(data)}, "
            f"Total value: {total_final_value:.2f}, Total return: {total_return:.2%}"
        )

        # 创建汇总结果
        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_cash,
            final_capital=total_final_value / len(data),  # 平均每只股票
            total_return=total_return,
            annual_return=annual_return,
            volatility=all_results[0].volatility if all_results else 0,
            sharpe_ratio=all_results[0].sharpe_ratio if all_results else 0,
            sortino_ratio=all_results[0].sortino_ratio if all_results else 0,
            max_drawdown=max(r.max_drawdown for r in all_results) if all_results else 0,
            win_rate=avg_win_rate,
            profit_factor=sum(r.profit_factor for r in all_results) / len(all_results) if all_results else 0,
            total_trades=total_trades,
            winning_trades=sum(r.winning_trades for r in all_results),
            losing_trades=sum(r.losing_trades for r in all_results),
            trades=all_trades,
            orders=[],
            metrics=[],
            equity_curve=equity_curve_list,
        )

        return result

    def calculate_metrics(
        self, trades: List[Trade], initial_capital: float
    ) -> List[Metric]:
        """
        Calculate performance metrics from trade history.

        Args:
            trades: List of trades from the backtest
            initial_capital: Initial capital for the backtest

        Returns:
            List of calculated metrics
        """
        return self.engine.calculate_metrics(trades, initial_capital)
