"""
实时信号监控服务

提供实时股票信号监控能力:
- 定时检查股票评分
- 自动生成买入/卖出信号
- 多渠道通知
- 信号历史记录
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

from ..domain.models import Signal, OrderSide
from ..factors.breakout_scoring_system import BreakoutScoringSystem
from ..factors.trend_scoring_system import TrendScoringSystem
from ..factors.trend_momentum_scoring import TrendMomentumScoring
from ..strategies.qlib_strategy import QlibStrategy
from ..core.timeutils import is_trading_time
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MonitorConfig:
    """监控配置"""
    symbols: List[str]                           # 监控股票列表
    interval_minutes: int = 5                    # 检查间隔(分钟)
    strategy: str = "breakout"                   # 策略名称
    score_threshold: int = 50                    # 评分阈值
    buy_threshold: int = 50                      # 买入阈值
    sell_threshold: int = 40                     # 卖出阈值
    notifiers: List[str] = field(default_factory=lambda: ["console"])
    history_days: int = 60                       # 历史数据天数
    trading_hours_only: bool = True              # 仅交易时间运行
    cooldown_minutes: int = 15                   # 冷却期(分钟)


@dataclass
class SignalResult:
    """信号结果"""
    signal: Optional[Signal]
    score: float
    score_details: Dict[str, Any]
    passed_filter: bool
    filter_reason: str


class RealtimeMonitorService:
    """
    实时信号监控服务

    功能:
    - 定时检查股票评分 (每5分钟)
    - 仅在交易时间内运行
    - 多渠道通知
    - 信号冷却机制
    - 历史记录保存
    """

    def __init__(
        self,
        config: MonitorConfig,
        data_provider=None,
        notifiers: List[Any] = None,
        signal_store=None
    ):
        """
        初始化监控服务

        Args:
            config: 监控配置
            data_provider: 数据提供者
            notifiers: 通知器列表
            signal_store: 信号存储
        """
        self.config = config
        self.data_provider = data_provider
        self.notifiers = notifiers or []
        self.signal_store = signal_store

        # 初始化评分系统
        if config.strategy == "breakout":
            self.scoring_system = BreakoutScoringSystem()
        elif config.strategy == "trend":
            self.scoring_system = TrendScoringSystem()
        elif config.strategy == "momentum":
            self.scoring_system = TrendMomentumScoring()
        elif config.strategy == "qlib":
            self.scoring_system = QlibStrategy()
            self._is_qlib = True
        else:
            self.scoring_system = BreakoutScoringSystem()

        self._is_qlib = config.strategy == "qlib"

        # 状态
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_signal_time: Dict[str, datetime] = {}  # 冷却期追踪
        self._signal_history: List[SignalResult] = []

        # 统计
        self._check_count = 0
        self._signal_count = 0

    async def start(self) -> None:
        """
        启动监控服务

        此方法会阻塞运行，直到被停止（通过 stop() 或 KeyboardInterrupt）
        """
        if self._running:
            logger.warning("Monitor service is already running")
            return

        self._running = True
        logger.info(
            f"Starting realtime monitor service for {len(self.config.symbols)} symbols, "
            f"interval={self.config.interval_minutes}min, strategy={self.config.strategy}"
        )

        # 直接运行监控循环（阻塞）
        await self._monitor_loop()

    async def stop(self) -> None:
        """停止监控服务"""
        if not self._running:
            return

        self._running = False
        logger.info("Monitor service stopped")

    async def _monitor_loop(self) -> None:
        """监控循环"""
        while self._running:
            try:
                await self._check_signals()
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")

            # 等待下一个检查周期
            await asyncio.sleep(self.config.interval_minutes * 60)

    async def _check_signals(self) -> None:
        """检查所有股票的信号"""
        now = datetime.now()

        # 检查是否在交易时间
        if self.config.trading_hours_only and not is_trading_time(now):
            logger.debug(f"Not trading time: {now}")
            return

        self._check_count += 1
        logger.info(f"Checking signals for {len(self.config.symbols)} symbols (check #{self._check_count})")

        for symbol in self.config.symbols:
            try:
                await self._check_symbol(symbol)
            except Exception as e:
                logger.error(f"Error checking {symbol}: {e}")

    async def _check_symbol(self, symbol: str) -> None:
        """检查单个股票的信号"""
        now = datetime.now()

        # 检查冷却期
        if symbol in self._last_signal_time:
            elapsed = now - self._last_signal_time[symbol]
            if elapsed < timedelta(minutes=self.config.cooldown_minutes):
                logger.debug(f"{symbol} in cooldown period ({elapsed.total_seconds()/60:.1f}min < {self.config.cooldown_minutes}min)")
                return

        # 获取数据
        df = await self._get_stock_data(symbol)
        if df is None or df.empty:
            logger.warning(f"No data available for {symbol}")
            return

        # 计算评分
        result = self._generate_signal(symbol, df)

        # 检查是否有信号
        if result.signal is not None:
            # 更新冷却期
            self._last_signal_time[symbol] = now
            self._signal_count += 1

            # 发送通知
            await self._notify(result)

            # 保存到历史
            self._signal_history.append(result)
            if self.signal_store:
                self.signal_store.save_signal(result.signal, result.score_details)

            logger.info(
                f"Signal generated: {result.signal.direction} {symbol} "
                f"score={result.score:.1f} passed={result.passed_filter}"
            )

    async def _get_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取股票历史数据"""
        if self.data_provider is None:
            logger.error("No data provider configured")
            return None

        try:
            # 获取日线数据用于评分计算
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config.history_days)

            data = self.data_provider.get_bars(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
                timeframe='1d'
            )

            return data.get(symbol, pd.DataFrame())

        except Exception as e:
            logger.error(f"Failed to get data for {symbol}: {e}")
            return None

    def _generate_signal(self, symbol: str, df: pd.DataFrame) -> SignalResult:
        """使用评分系统生成信号"""
        # Qlib 策略使用不同的接口
        if self._is_qlib:
            return self._generate_qlib_signal(symbol, df)

        # 计算评分
        score_result = self.scoring_system.calculate_score(df)

        # 提取关键信息
        if hasattr(score_result, 'final_score'):
            # BreakoutScoringSystem 或 TrendScoringSystem
            final_score = score_result.final_score
            passed_filter = getattr(score_result, 'passed_hard_filter',
                                   getattr(score_result, 'passed_filter', True))
            filter_reason = getattr(score_result, 'hard_filter_reason',
                                   getattr(score_result, 'filter_reason', ''))
            score_details = score_result.to_dict() if hasattr(score_result, 'to_dict') else {}
        else:
            # 字典格式
            final_score = score_result.get('final_score', 50)
            passed_filter = score_result.get('passed_filter', True)
            filter_reason = score_result.get('filter_reason', '')
            score_details = score_result

        # 判断信号方向
        signal = None
        if passed_filter and final_score >= self.config.buy_threshold:
            signal = Signal(
                symbol=symbol,
                timestamp=datetime.now(),
                direction=OrderSide.BUY,
                strength=min(1.0, final_score / 100),
                reason=f"Score={final_score:.1f} >= {self.config.buy_threshold}",
                confidence=final_score / 100
            )
        elif final_score <= self.config.sell_threshold:
            signal = Signal(
                symbol=symbol,
                timestamp=datetime.now(),
                direction=OrderSide.SELL,
                strength=max(0.5, 1 - final_score / 100),
                reason=f"Score={final_score:.1f} <= {self.config.sell_threshold}",
                confidence=1 - final_score / 100
            )

        return SignalResult(
            signal=signal,
            score=final_score,
            score_details=score_details,
            passed_filter=passed_filter,
            filter_reason=filter_reason
        )

    def _generate_qlib_signal(self, symbol: str, df: pd.DataFrame) -> SignalResult:
        """使用 Qlib 策略生成信号"""
        try:
            # Qlib 需要至少 120 条数据
            if len(df) < 120:
                return SignalResult(
                    signal=None,
                    score=50.0,
                    score_details={'error': f'数据不足，需要120条，当前{len(df)}条'},
                    passed_filter=False,
                    filter_reason='数据不足'
                )

            # 调用 Qlib 策略的 get_signal 方法
            current_bar = df.iloc[-1]
            signal_dict = self.scoring_system.get_signal(current_bar, df)

            # 提取信号信息
            direction = signal_dict.get('direction')
            probability = signal_dict.get('probability', 0.5)
            score = signal_dict.get('score', 0.5)
            confidence = signal_dict.get('confidence', 0.5)

            # 将 score (0-1) 转换为 0-100 的评分
            final_score = score * 100

            # 生成信号
            signal = None
            if direction == 'buy':
                signal = Signal(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    direction=OrderSide.BUY,
                    strength=min(1.0, confidence),
                    reason=f"Qlib概率={probability:.2f}, 评分={final_score:.1f}",
                    confidence=confidence
                )
            elif direction == 'sell':
                signal = Signal(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    direction=OrderSide.SELL,
                    strength=min(1.0, confidence),
                    reason=f"Qlib概率={probability:.2f}, 评分={final_score:.1f}",
                    confidence=confidence
                )

            return SignalResult(
                signal=signal,
                score=final_score,
                score_details={
                    'probability': probability,
                    'confidence': confidence,
                    'qlib_score': score,
                    'strategy': 'qlib'
                },
                passed_filter=True,
                filter_reason=''
            )

        except Exception as e:
            logger.error(f"Qlib signal generation error for {symbol}: {e}")
            return SignalResult(
                signal=None,
                score=50.0,
                score_details={'error': str(e)},
                passed_filter=False,
                filter_reason=str(e)
            )

    async def _notify(self, result: SignalResult) -> None:
        """发送通知"""
        signal = result.signal

        # 构建消息
        direction_emoji = "🟢" if signal.direction == OrderSide.BUY else "🔴"
        direction_cn = "买入" if signal.direction == OrderSide.BUY else "卖出"

        message = f"""
{direction_emoji} {direction_cn}信号

股票: {signal.symbol}
时间: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
评分: {result.score:.1f}
强度: {signal.strength:.2f}
原因: {signal.reason}

评分详情:
- 通过过滤: {result.passed_filter}
- 过滤原因: {result.filter_reason}
"""

        subject = f"[{direction_cn}信号] {signal.symbol} 评分{result.score:.0f}"

        # 发送到所有通知渠道
        for notifier in self.notifiers:
            try:
                if hasattr(notifier, 'send_notification'):
                    notifier.send_notification(message.strip(), subject)
                elif hasattr(notifier, 'notify'):
                    notifier.notify(message.strip(), subject)
            except Exception as e:
                logger.error(f"Failed to send notification via {notifier}: {e}")

    def get_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        return {
            "running": self._running,
            "symbols": self.config.symbols,
            "strategy": self.config.strategy,
            "interval_minutes": self.config.interval_minutes,
            "score_threshold": self.config.score_threshold,
            "check_count": self._check_count,
            "signal_count": self._signal_count,
            "last_check": datetime.now().isoformat() if self._check_count > 0 else None
        }

    def get_recent_signals(self, limit: int = 20) -> List[SignalResult]:
        """获取最近的信号"""
        return self._signal_history[-limit:]

    def get_signal_stats(self) -> Dict[str, Any]:
        """获取信号统计"""
        if not self._signal_history:
            return {}

        buy_count = sum(1 for s in self._signal_history if s.signal and s.signal.direction == OrderSide.BUY)
        sell_count = sum(1 for s in self._signal_history if s.signal and s.signal.direction == OrderSide.SELL)
        avg_score = sum(s.score for s in self._signal_history) / len(self._signal_history)

        return {
            "total_signals": len(self._signal_history),
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "avg_score": avg_score,
            "pass_rate": sum(1 for s in self._signal_history if s.passed_filter) / len(self._signal_history)
        }