"""Signal service for QuantTool."""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from ..domain.interfaces.strategy import IStrategy
from ..domain.interfaces.data_provider import IDataProvider
from ..domain.interfaces.notifier import INotifier
from ..domain.models import Signal
from ..core.registry import registry, ComponentType
from ..core.logging import get_logger
from ..core.timeutils import get_next_trading_bar_timestamp
from ..infrastructure.data_providers.incremental_data_manager import IncrementalDataManager, DataType


logger = get_logger(__name__)


class SignalService:
    """Service class for generating trading signals."""

    def __init__(self, use_incremental: bool = True):
        """Initialize signal service.

        Args:
            use_incremental: 是否使用增量数据获取
        """
        self.cooldown_bars = (
            3  # Minimum bars between same-direction signals for same symbol
        )
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

    def scan_signals(
        self,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "10m",
        data_provider: str = "tushare",
    ) -> List[Signal]:
        """
        Scan for signals across historical data.

        Args:
            strategy_name: Name of the strategy to use for signal generation
            strategy_params: Parameters for the strategy
            symbols: List of symbols to scan
            start_date: Start date for scanning
            end_date: End date for scanning
            timeframe: Timeframe for the scan
            data_provider: Name of the data provider to use

        Returns:
            List of generated signals
        """
        logger.info(
            f"Scanning for signals using strategy {strategy_name} for {len(symbols)} symbols "
            f"from {start_date} to {end_date}"
        )

        # Get strategy from registry
        strategy_class = registry.get(ComponentType.STRATEGY, strategy_name)
        strategy = strategy_class()
        strategy.initialize(strategy_params)

        # Get data provider
        provider_class = registry.get(ComponentType.DATA_PROVIDER, data_provider)
        data_provider_instance = provider_class()
        if hasattr(data_provider_instance, "initialize"):
            data_provider_instance.initialize()

        # Get data (使用增量数据管理器)
        data = {}
        for symbol in symbols:
            df = self._get_data(symbol, start_date, end_date, data_provider_instance, timeframe)
            if not df.empty:
                data[symbol] = df

        signals = []
        # Track last signal for each symbol to enforce cooldown
        last_signal_time = {}

        for symbol, df in data.items():
            logger.info(f"Processing {len(df)} bars for {symbol}")

            # Process each bar in the data
            for idx, row in df.iterrows():
                # Get historical data up to current point
                hist_data = df.iloc[: idx + 1].copy()

                # Get signal from strategy
                signal = strategy.get_signal(row, hist_data)

                if signal and signal.get("direction") and signal["direction"] != "hold":
                    # Check cooldown period
                    if symbol in last_signal_time:
                        last_time = last_signal_time[symbol]
                        # Find how many bars have passed since last signal
                        current_idx = idx
                        last_idx = (
                            df.index.get_loc(last_time) if last_time in df.index else -1
                        )

                        if current_idx - last_idx < self.cooldown_bars:
                            continue  # Skip if cooldown period hasn't passed

                    # Create signal object
                    signal_obj = Signal(
                        symbol=symbol,
                        timestamp=row["timestamp"],
                        direction=signal["direction"],
                        strength=signal.get("strength", 1.0),
                        reason=signal.get("reason", "unknown"),
                        predicted_return=signal.get("predicted_return"),
                        confidence=signal.get("confidence"),
                    )

                    signals.append(signal_obj)

                    # Update last signal time for this symbol
                    last_signal_time[symbol] = row["timestamp"]

        logger.info(f"Generated {len(signals)} signals across all symbols")
        return signals

    def live_signals(
        self,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        symbols: List[str],
        notifier: INotifier,
        timeframe: str = "10m",
        data_provider: str = "ashare",
        run_duration: int = None,  # Duration in minutes to run, None for indefinite
    ):
        """
        Monitor for live signals and notify when detected.

        Args:
            strategy_name: Name of the strategy to use for signal generation
            strategy_params: Parameters for the strategy
            symbols: List of symbols to monitor
            notifier: Notifier to use for sending alerts
            timeframe: Timeframe for monitoring
            data_provider: Name of the data provider to use (should be real-time capable)
            run_duration: How long to run in minutes (None for indefinite)
        """
        logger.info(
            f"Starting live signal monitoring for {len(symbols)} symbols using strategy {strategy_name}"
        )

        # Get strategy from registry
        strategy_class = registry.get(ComponentType.STRATEGY, strategy_name)
        strategy = strategy_class()
        strategy.initialize(strategy_params)

        # Get data provider
        provider_class = registry.get(ComponentType.DATA_PROVIDER, data_provider)
        data_provider_instance = provider_class()
        if hasattr(data_provider_instance, "initialize"):
            data_provider_instance.initialize()

        # Track start time if duration is specified
        start_time = datetime.now()

        # Track last signal for each symbol to enforce cooldown
        last_signal_time = {}

        try:
            while True:
                # Check if we should stop based on run duration
                if run_duration is not None:
                    elapsed = datetime.now() - start_time
                    if elapsed.total_seconds() / 60 >= run_duration:
                        logger.info(
                            "Live signal monitoring stopped: reached duration limit"
                        )
                        break

                # Get latest data for each symbol
                for symbol in symbols:
                    latest_bar = data_provider_instance.get_latest_bar(
                        symbol, timeframe
                    )

                    if latest_bar is not None and not latest_bar.empty:
                        current_bar = latest_bar.iloc[0]  # Get the row

                        # Get historical data for signal calculation
                        # In a real implementation, we'd maintain a sliding window of history
                        # For now, we'll use just the current bar to demonstrate the concept
                        hist_data = (
                            latest_bar.copy()
                        )  # In practice, this would be a longer history

                        # Get signal from strategy
                        signal = strategy.get_signal(current_bar, hist_data)

                        if (
                            signal
                            and signal.get("direction")
                            and signal["direction"] != "hold"
                        ):
                            # Check cooldown period
                            current_time = current_bar["timestamp"]

                            if symbol in last_signal_time:
                                last_time = last_signal_time[symbol]

                                # In a real system, we'd need to compare bar indices or time differences
                                # For simplicity, we'll check if enough time has passed
                                time_diff = current_time - last_time
                                # Assuming 10-minute bars, we need at least 3 bars * 10 minutes = 30 minutes
                                min_wait = timedelta(
                                    minutes=self.cooldown_bars
                                    * (10 if timeframe == "10m" else 1)
                                )

                                if time_diff < min_wait:
                                    continue  # Skip if cooldown period hasn't passed

                            # Create signal object
                            signal_obj = Signal(
                                symbol=symbol,
                                timestamp=current_time,
                                direction=signal["direction"],
                                strength=signal.get("strength", 1.0),
                                reason=signal.get("reason", "live_signal"),
                                predicted_return=signal.get("predicted_return"),
                                confidence=signal.get("confidence"),
                            )

                            # Notify about the signal
                            message = (
                                f"LIVE SIGNAL: {signal_obj.direction.upper()} {signal_obj.symbol} at {signal_obj.timestamp}\n"
                                f"Reason: {signal_obj.reason}\n"
                                f"Strength: {signal_obj.strength}"
                            )

                            if signal_obj.confidence:
                                message += f"\nConfidence: {signal_obj.confidence:.2f}"

                            subject = f"Live Trading Signal: {signal_obj.direction.upper()} {signal_obj.symbol}"

                            success = notifier.send_notification(message, subject)

                            if success:
                                logger.info(
                                    f"Sent signal notification: {signal_obj.direction} {signal_obj.symbol}"
                                )
                            else:
                                logger.error(
                                    f"Failed to send signal notification: {signal_obj.direction} {signal_obj.symbol}"
                                )

                            # Update last signal time for this symbol
                            last_signal_time[symbol] = current_time

                # Sleep before checking again (in a real implementation, this might be event-driven)
                import time

                time.sleep(60)  # Wait 1 minute between checks

        except KeyboardInterrupt:
            logger.info("Live signal monitoring interrupted by user")
        except Exception as e:
            logger.error(f"Error in live signal monitoring: {str(e)}")
