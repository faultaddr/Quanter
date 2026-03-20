"""
趋势动量评分系统

核心思路：抓住趋势启动点，而非等待形态确认

评分构成:
- 动量突破信号 (40分): 5/10/20日动量
- 均线系统 (30分): MA5/10/20/60排列
- 量能确认 (20分): 量比、价涨量增
- 位置判断 (10分): 60日位置
- 突破确认 (加分): 突破新高
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class ScoringConfig:
    """
    评分系统配置参数

    用于优化过程中调整评分系统的各项阈值
    """

    # 动量阈值
    mom_5_strong: float = 3.0
    """5日动量强阈值"""
    mom_5_medium: float = 1.0
    """5日动量中等阈值"""
    mom_10_strong: float = 5.0
    """10日动量强阈值"""
    mom_10_medium: float = 2.0
    """10日动量中等阈值"""
    mom_20_strong: float = 10.0
    """20日动量强阈值"""
    mom_20_medium: float = 5.0
    """20日动量中等阈值"""

    # 动量得分
    mom_5_strong_score: float = 15.0
    mom_5_medium_score: float = 8.0
    mom_10_strong_score: float = 10.0
    mom_10_medium_score: float = 5.0
    mom_20_strong_score: float = 15.0
    mom_20_medium_score: float = 8.0

    # 均线斜率阈值
    ma20_slope_threshold: float = 2.0
    """MA20斜率阈值"""

    # 量比阈值
    vol_ratio_huge: float = 2.0
    """巨量量比阈值"""
    vol_ratio_large: float = 1.5
    """放量量比阈值"""

    # 量能得分
    vol_huge_score: float = 12.0
    vol_large_score: float = 8.0
    vol_normal_score: float = 4.0
    vol_price_up_score: float = 8.0

    # 位置区间
    position_mid_low: float = 0.3
    """中位区间下限"""
    position_mid_high: float = 0.6
    """中位区间上限"""
    position_high_risk: float = 0.8
    """高位风险阈值"""

    # 位置得分
    position_mid_score: float = 10.0
    position_low_score: float = 5.0
    position_high_penalty: float = -5.0

    # 突破阈值
    breakout_10d_threshold: float = 0.99
    """10日突破阈值比例"""

    # 策略参数
    buy_threshold: float = 55.0
    """买入阈值"""
    sell_threshold: float = 40.0
    """卖出阈值"""
    stop_loss_pct: float = 0.07
    """止损比例"""
    take_profit_pct: float = 0.15
    """止盈比例"""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'mom_5_strong': self.mom_5_strong,
            'mom_5_medium': self.mom_5_medium,
            'mom_10_strong': self.mom_10_strong,
            'mom_10_medium': self.mom_10_medium,
            'mom_20_strong': self.mom_20_strong,
            'mom_20_medium': self.mom_20_medium,
            'ma20_slope_threshold': self.ma20_slope_threshold,
            'vol_ratio_huge': self.vol_ratio_huge,
            'vol_ratio_large': self.vol_ratio_large,
            'position_mid_low': self.position_mid_low,
            'position_mid_high': self.position_mid_high,
            'position_high_risk': self.position_high_risk,
            'breakout_10d_threshold': self.breakout_10d_threshold,
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScoringConfig':
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class TrendMomentumResult:
    """趋势动量评分结果"""

    final_score: float
    """最终评分 (0-100)"""

    signal: bool
    """是否生成信号 (score >= threshold)"""

    stop_loss: float
    """止损价"""

    take_profit: float
    """止盈价"""

    signals: List[str] = field(default_factory=list)
    """信号明细列表"""

    details: Dict[str, Any] = field(default_factory=dict)
    """各项得分详情"""

    momentum_score: float = 0.0
    """动量得分"""

    ma_score: float = 0.0
    """均线得分"""

    volume_score: float = 0.0
    """量能得分"""

    position_score: float = 0.0
    """位置得分"""

    breakout_score: float = 0.0
    """突破得分"""


class TrendMomentumScoring:
    """
    趋势动量评分系统

    核心思路：抓住趋势启动点，而非等待形态确认

    评分构成:
    - 动量突破信号 (40分): 5/10/20日动量
    - 均线系统 (30分): MA5/10/20/60排列
    - 量能确认 (20分): 量比、价涨量增
    - 位置判断 (10分): 60日位置
    - 突破确认 (加分): 突破新高
    """

    def __init__(
        self,
        buy_threshold: float = 55.0,
        stop_loss_pct: float = 0.07,
        take_profit_pct: float = 0.15,
        config: Optional[ScoringConfig] = None,
    ):
        """
        初始化趋势动量评分系统

        Args:
            buy_threshold: 买入阈值 (默认55)
            stop_loss_pct: 止损比例 (默认7%)
            take_profit_pct: 止盈比例 (默认15%)
            config: 评分配置参数 (可选)
        """
        if config is not None:
            self.config = config
        else:
            self.config = ScoringConfig(
                buy_threshold=buy_threshold,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )

        # 向后兼容属性
        self.buy_threshold = self.config.buy_threshold
        self.stop_loss_pct = self.config.stop_loss_pct
        self.take_profit_pct = self.config.take_profit_pct

    def update_config(self, **kwargs) -> None:
        """
        更新配置参数

        Args:
            **kwargs: 要更新的参数
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # 同步兼容属性
        self.buy_threshold = self.config.buy_threshold
        self.stop_loss_pct = self.config.stop_loss_pct
        self.take_profit_pct = self.config.take_profit_pct

    def calculate_score(self, df: pd.DataFrame) -> TrendMomentumResult:
        """
        计算趋势动量评分

        Args:
            df: K线数据，需要包含 close, high, low, volume 列

        Returns:
            TrendMomentumResult: 评分结果
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        if len(close) < 60:
            return TrendMomentumResult(
                final_score=0,
                signal=False,
                stop_loss=0,
                take_profit=0,
                signals=['数据不足(需60日)'],
                details={'error': 'insufficient_data'}
            )

        score = 0.0
        signals: List[str] = []
        details: Dict[str, Any] = {}

        # 1. 动量突破信号 (40分)
        momentum_score, momentum_signals, momentum_details = self._calculate_momentum(close)
        score += momentum_score
        signals.extend(momentum_signals)
        details['momentum'] = momentum_details

        # 2. 均线系统 (30分)
        ma_score, ma_signals, ma_details = self._calculate_ma_system(close)
        score += ma_score
        signals.extend(ma_signals)
        details['ma'] = ma_details

        # 3. 量能确认 (20分)
        volume_score, volume_signals, volume_details = self._calculate_volume(close, volume)
        score += volume_score
        signals.extend(volume_signals)
        details['volume'] = volume_details

        # 4. 位置判断 (10分)
        position_score, position_signals, position_details = self._calculate_position(close, high, low)
        score += position_score
        signals.extend(position_signals)
        details['position'] = position_details

        # 5. 突破确认 (加分项)
        breakout_score, breakout_signals, breakout_details = self._calculate_breakout(close, high)
        score += breakout_score
        signals.extend(breakout_signals)
        details['breakout'] = breakout_details

        # 计算止损止盈
        stop_loss = close[-1] * (1 - self.stop_loss_pct)
        take_profit = close[-1] * (1 + self.take_profit_pct)

        # 最终评分限制在0-100
        final_score = min(100.0, max(0.0, score))

        return TrendMomentumResult(
            final_score=final_score,
            signal=final_score >= self.buy_threshold,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signals=signals,
            details=details,
            momentum_score=momentum_score,
            ma_score=ma_score,
            volume_score=volume_score,
            position_score=position_score,
            breakout_score=breakout_score,
        )

    def _calculate_momentum(self, close: np.ndarray) -> tuple:
        """计算动量突破信号 (40分)"""
        score = 0.0
        signals: List[str] = []
        details: Dict[str, Any] = {}
        cfg = self.config

        # 计算多周期动量
        mom_5 = (close[-1] - close[-5]) / close[-5] * 100
        mom_10 = (close[-1] - close[-10]) / close[-10] * 100
        mom_20 = (close[-1] - close[-20]) / close[-20] * 100

        details['mom_5'] = mom_5
        details['mom_10'] = mom_10
        details['mom_20'] = mom_20

        # 短期动量爆发
        if mom_5 > cfg.mom_5_strong:
            score += cfg.mom_5_strong_score
            signals.append(f"5日动量{mom_5:.1f}%(+{cfg.mom_5_strong_score:.0f})")
        elif mom_5 > cfg.mom_5_medium:
            score += cfg.mom_5_medium_score
            signals.append(f"5日动量{mom_5:.1f}%(+{cfg.mom_5_medium_score:.0f})")

        # 中期动量确认
        if mom_10 > cfg.mom_10_strong:
            score += cfg.mom_10_strong_score
            signals.append(f"10日动量{mom_10:.1f}%(+{cfg.mom_10_strong_score:.0f})")
        elif mom_10 > cfg.mom_10_medium:
            score += cfg.mom_10_medium_score
            signals.append(f"10日动量{mom_10:.1f}%(+{cfg.mom_10_medium_score:.0f})")

        # 长期趋势
        if mom_20 > cfg.mom_20_strong:
            score += cfg.mom_20_strong_score
            signals.append(f"20日动量{mom_20:.1f}%(+{cfg.mom_20_strong_score:.0f})")
        elif mom_20 > cfg.mom_20_medium:
            score += cfg.mom_20_medium_score
            signals.append(f"20日动量{mom_20:.1f}%(+{cfg.mom_20_medium_score:.0f})")

        details['score'] = score
        return score, signals, details

    def _calculate_ma_system(self, close: np.ndarray) -> tuple:
        """计算均线系统得分 (30分)"""
        score = 0.0
        signals: List[str] = []
        details: Dict[str, Any] = {}
        cfg = self.config

        ma5 = np.mean(close[-5:])
        ma10 = np.mean(close[-10:])
        ma20 = np.mean(close[-20:])
        ma60 = np.mean(close[-60:])

        details['ma5'] = ma5
        details['ma10'] = ma10
        details['ma20'] = ma20
        details['ma60'] = ma60

        # 均线多头
        if ma5 > ma10 > ma20 > ma60:
            score += 20
            signals.append("均线完美多头(+20)")
        elif ma5 > ma10 > ma20:
            score += 12
            signals.append("均线多头(+12)")
        elif ma5 > ma10:
            score += 6
            signals.append("短期金叉(+6)")

        # 突破MA20
        if close[-1] > ma20:
            score += 5
            signals.append("站上MA20(+5)")

        # MA20斜率
        ma20_slope = (ma20 - np.mean(close[-30:-10])) / np.mean(close[-30:-10]) * 100
        details['ma20_slope'] = ma20_slope

        if ma20_slope > cfg.ma20_slope_threshold:
            score += 5
            signals.append(f"MA20斜率{ma20_slope:.1f}%(+5)")

        details['score'] = score
        return score, signals, details

    def _calculate_volume(self, close: np.ndarray, volume: np.ndarray) -> tuple:
        """计算量能确认得分 (20分)"""
        score = 0.0
        signals: List[str] = []
        details: Dict[str, Any] = {}
        cfg = self.config

        vol_ma5 = np.mean(volume[-5:])
        vol_ma20 = np.mean(volume[-20:])

        details['vol_ma5'] = vol_ma5
        details['vol_ma20'] = vol_ma20
        details['vol_ratio'] = vol_ma5 / vol_ma20 if vol_ma20 > 0 else 0

        if vol_ma5 > vol_ma20 * cfg.vol_ratio_huge:
            score += cfg.vol_huge_score
            signals.append(f"巨量(+{cfg.vol_huge_score:.0f})")
        elif vol_ma5 > vol_ma20 * cfg.vol_ratio_large:
            score += cfg.vol_large_score
            signals.append(f"放量(+{cfg.vol_large_score:.0f})")
        elif vol_ma5 > vol_ma20:
            score += cfg.vol_normal_score
            signals.append(f"量增(+{cfg.vol_normal_score:.0f})")

        # 价涨量增
        if close[-1] > close[-2] and volume[-1] > volume[-2]:
            score += cfg.vol_price_up_score
            signals.append(f"价涨量增(+{cfg.vol_price_up_score:.0f})")

        details['score'] = score
        return score, signals, details

    def _calculate_position(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> tuple:
        """计算位置判断得分 (10分)"""
        score = 0.0
        signals: List[str] = []
        details: Dict[str, Any] = {}
        cfg = self.config

        high_60 = np.max(high[-60:])
        low_60 = np.min(low[-60:])
        position = (close[-1] - low_60) / (high_60 - low_60 + 1e-10)

        details['high_60'] = high_60
        details['low_60'] = low_60
        details['position'] = position

        # 中位启动最佳
        if cfg.position_mid_low < position < cfg.position_mid_high:
            score += cfg.position_mid_score
            signals.append(f"中位启动({position*100:.0f}%)(+{cfg.position_mid_score:.0f})")
        elif position < cfg.position_mid_low:
            score += cfg.position_low_score
            signals.append(f"低位({position*100:.0f}%)(+{cfg.position_low_score:.0f})")
        elif position > cfg.position_high_risk:
            score += cfg.position_high_penalty
            signals.append(f"高位风险({position*100:.0f}%)({cfg.position_high_penalty:.0f})")

        details['score'] = score
        return score, signals, details

    def _calculate_breakout(self, close: np.ndarray, high: np.ndarray) -> tuple:
        """计算突破确认得分 (加分项)"""
        score = 0.0
        signals: List[str] = []
        details: Dict[str, Any] = {}
        cfg = self.config

        # 突破10日高
        high_10 = np.max(high[-10:])
        details['high_10'] = high_10

        if close[-1] >= high_10 * cfg.breakout_10d_threshold:
            score += 10
            signals.append("突破10日高(+10)")

        # 创20日新高
        high_20 = np.max(high[-20:])
        details['high_20'] = high_20

        if close[-1] >= high_20:
            score += 5
            signals.append("创20日新高(+5)")

        details['score'] = score
        return score, signals, details