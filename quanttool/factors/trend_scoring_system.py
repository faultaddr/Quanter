"""
趋势选股评分系统 - 纯趋势强度评分

核心理念：
- 找到趋势已经确立、当前仍在运行的股票
- 趋势中途入场，不抄底，不追顶
- 趋势持续就持有，趋势破坏就离场

评分架构：
1. 硬过滤层：不满足直接排除
2. 趋势强度评分：均线结构、价格动能、量能配合、相对强度
3. 入场时机修正：0.7~1.2系数
4. 最终评分：趋势总分 × 时机系数
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TrendScoreResult:
    """趋势评分结果"""
    # 最终评分
    final_score: float
    # 趋势总分（不含时机系数）
    trend_total_score: float
    # 时机系数
    timing_coefficient: float
    # 各因子得分
    ma_structure_score: float  # 均线结构分
    price_momentum_score: float  # 价格动能分
    volume_score: float  # 量能配合分
    relative_strength_score: float  # 相对强度分
    # 硬过滤结果
    passed_hard_filter: bool
    hard_filter_reason: str = ""
    # 时机分析
    timing_type: str = "standard"  # 回踩/突破/标准/过热/风险
    # 明细数据
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'final_score': self.final_score,
            'trend_total_score': self.trend_total_score,
            'timing_coefficient': self.timing_coefficient,
            'ma_structure_score': self.ma_structure_score,
            'price_momentum_score': self.price_momentum_score,
            'volume_score': self.volume_score,
            'relative_strength_score': self.relative_strength_score,
            'passed_hard_filter': self.passed_hard_filter,
            'hard_filter_reason': self.hard_filter_reason,
            'timing_type': self.timing_type,
            'details': self.details
        }


class TrendScoringSystem:
    """
    趋势选股评分系统

    纯趋势强度评分，不再使用位置惩罚系数

    评分公式：
    趋势总分 = 均线结构分×0.30 + 价格动能分×0.30 + 量能配合分×0.25 + 相对强度分×0.15
    最终评分 = 趋势总分 × 时机系数
    """

    # 权重配置
    WEIGHTS = {
        'ma_structure': 0.30,
        'price_momentum': 0.30,
        'volume': 0.25,
        'relative_strength': 0.15
    }

    # 时机系数范围
    TIMING_RANGE = (0.7, 1.2)

    def __init__(
        self,
        min_amount: float = 1e8,  # 最低日均成交额 1亿
        min_list_days: int = 120,  # 最低上市天数
        ma_periods: Tuple[int, ...] = (5, 20, 50, 200)
    ):
        """
        初始化趋势评分系统

        Args:
            min_amount: 最低日均成交额
            min_list_days: 最低上市天数
            ma_periods: 均线周期
        """
        self.min_amount = min_amount
        self.min_list_days = min_list_days
        self.ma_periods = ma_periods

    def calculate_score(self, df: pd.DataFrame) -> TrendScoreResult:
        """
        计算趋势评分

        Args:
            df: 股票数据，需包含 open, high, low, close, volume 等列

        Returns:
            TrendScoreResult: 评分结果
        """
        # 1. 硬过滤
        passed, reason = self._hard_filter(df)
        if not passed:
            return TrendScoreResult(
                final_score=0,
                trend_total_score=0,
                timing_coefficient=0,
                ma_structure_score=0,
                price_momentum_score=0,
                volume_score=0,
                relative_strength_score=0,
                passed_hard_filter=False,
                hard_filter_reason=reason
            )

        # 2. 计算各因子得分
        ma_score, ma_details = self._calculate_ma_structure_score(df)
        momentum_score, momentum_details = self._calculate_price_momentum_score(df)
        volume_score, volume_details = self._calculate_volume_score(df)
        rs_score, rs_details = self._calculate_relative_strength_score(df)

        # 3. 计算趋势总分
        trend_total = (
            ma_score * self.WEIGHTS['ma_structure'] +
            momentum_score * self.WEIGHTS['price_momentum'] +
            volume_score * self.WEIGHTS['volume'] +
            rs_score * self.WEIGHTS['relative_strength']
        )

        # 4. 计算时机系数
        timing_coef, timing_type, timing_details = self._calculate_timing_coefficient(df)

        # 5. 计算最终评分
        final_score = min(100, trend_total * timing_coef)

        return TrendScoreResult(
            final_score=round(final_score, 2),
            trend_total_score=round(trend_total, 2),
            timing_coefficient=round(timing_coef, 2),
            ma_structure_score=round(ma_score, 2),
            price_momentum_score=round(momentum_score, 2),
            volume_score=round(volume_score, 2),
            relative_strength_score=round(rs_score, 2),
            passed_hard_filter=True,
            timing_type=timing_type,
            details={
                'ma_structure': ma_details,
                'price_momentum': momentum_details,
                'volume': volume_details,
                'relative_strength': rs_details,
                'timing': timing_details
            }
        )

    def _hard_filter(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        硬过滤：不满足直接排除

        检查项：
        1. 流动性过滤：20日均成交额 > 1亿
        2. 趋势存在过滤：MA20斜率 > 0，股价 > MA20
        3. 基本过滤：排除ST、上市<6个月（需外部判断）
        """
        if len(df) < 60:
            return False, "数据不足60天"

        # 计算均线
        close = df['close'].values
        volume = df['volume'].values

        # MA20斜率
        ma20 = self._ma(close, 20)
        if len(ma20) < 20:
            return False, "MA20计算数据不足"

        ma20_slope = (ma20[-1] - ma20[-5]) / ma20[-5] if ma20[-5] > 0 else 0

        # 1. 流动性过滤
        avg_amount_20d = np.mean(close[-20:] * volume[-20:])
        if avg_amount_20d < self.min_amount:
            return False, f"流动性不足: 20日均成交额 {avg_amount_20d/1e8:.2f}亿 < 1亿"

        # 2. 趋势存在过滤
        if ma20_slope <= 0:
            return False, f"MA20斜率非正: {ma20_slope*100:.2f}%"

        if close[-1] <= ma20[-1]:
            return False, f"股价 {close[-1]:.2f} 低于MA20 {ma20[-1]:.2f}"

        return True, "通过硬过滤"

    def _calculate_ma_structure_score(self, df: pd.DataFrame) -> Tuple[float, Dict]:
        """
        计算均线结构分（30%权重）

        评分标准：
        - MA5 > MA20 > MA50 > MA200 → 100分（完美多头）
        - MA5 > MA20 > MA50 → 85分（强势多头）
        - MA5 > MA20, MA50向上 → 70分（中期多头确立）
        - 其他情况根据排列情况打分

        加分项：
        - 均线发散 +10分
        - MA20斜率增强 +5分
        """
        close = df['close'].values

        # 计算各均线
        ma5 = self._ma(close, 5)
        ma20 = self._ma(close, 20)
        ma50 = self._ma(close, 50)
        ma200 = self._ma(close, 200) if len(close) >= 200 else None

        score = 50  # 基础分
        details = {}

        # 检查均线排列
        ma5_above_ma20 = ma5[-1] > ma20[-1]
        ma20_above_ma50 = ma20[-1] > ma50[-1]

        if ma200 is not None and len(ma200) > 0:
            ma50_above_ma200 = ma50[-1] > ma200[-1]
            if ma5_above_ma20 and ma20_above_ma50 and ma50_above_ma200:
                score = 100  # 完美多头
                details['ma_pattern'] = "完美多头排列(MA5>MA20>MA50>MA200)"
            elif ma5_above_ma20 and ma20_above_ma50:
                score = 85  # 强势多头
                details['ma_pattern'] = "强势多头排列(MA5>MA20>MA50)"
        elif ma5_above_ma20 and ma20_above_ma50:
            score = 85
            details['ma_pattern'] = "强势多头排列(MA5>MA20>MA50)"
        elif ma5_above_ma20:
            # 检查MA50方向
            ma50_slope = (ma50[-1] - ma50[-5]) / ma50[-5] if ma50[-5] > 0 else 0
            if ma50_slope > 0:
                score = 70
                details['ma_pattern'] = "中期多头确立(MA5>MA20,MA50向上)"
            else:
                score = 60
                details['ma_pattern'] = "短期多头(MA5>MA20)"
        else:
            # 计算得分比例
            ma5_distance = (ma5[-1] - ma20[-1]) / ma20[-1] * 100
            score = max(40, 50 + ma5_distance)  # MA5低于MA20时扣分
            details['ma_pattern'] = f"均线整理中(MA5距MA20 {ma5_distance:.1f}%)"

        # 加分项：均线发散
        ma5_slope = (ma5[-1] - ma5[-5]) / ma5[-5] * 100 if ma5[-5] > 0 else 0
        ma20_slope = (ma20[-1] - ma20[-5]) / ma20[-5] * 100 if ma20[-5] > 0 else 0

        if ma5_slope > 0 and ma20_slope > 0:
            score = min(100, score + 10)
            details['divergence_bonus'] = "+10 均线发散向上"

        # 加分项：MA20斜率增强
        if ma20_slope > 2:  # 周涨幅超过2%
            score = min(100, score + 5)
            details['slope_bonus'] = f"+5 MA20斜率增强({ma20_slope:.1f}%)"

        details['ma5_slope'] = round(ma5_slope, 2)
        details['ma20_slope'] = round(ma20_slope, 2)
        details['score'] = round(score, 2)

        return min(100, max(0, score)), details

    def _calculate_price_momentum_score(self, df: pd.DataFrame) -> Tuple[float, Dict]:
        """
        计算价格动能分（30%权重）

        组成：
        - MACD动能（50%权重）：连续扩大天数×15分
        - RSI健康度（30%权重）：50-70区间100分，70-80区间85分
        - 涨幅持续性（20%权重）：近期vs中期涨幅比值
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        details = {}
        score = 50

        # 1. MACD动能（50%权重）
        macd, signal, hist = self._macd(close)
        if len(hist) >= 5:
            # 统计连续扩大天数
            expanding_days = 0
            for i in range(-1, -min(6, len(hist)), -1):
                if hist[i] > hist[i-1] and hist[i] > 0:
                    expanding_days += 1
                else:
                    break

            macd_score = min(100, expanding_days * 15 + 40)  # 基础40分

            # MACD金叉加分
            if len(hist) >= 2 and hist[-2] <= 0 and hist[-1] > 0:
                macd_score = min(100, macd_score + 15)
                details['macd_cross'] = "MACD金叉"

            details['macd_expanding_days'] = expanding_days
            details['macd_score'] = macd_score
        else:
            macd_score = 50
            details['macd_score'] = 50

        # 2. RSI健康度（30%权重）
        rsi = self._rsi(close, 14)
        if len(rsi) > 0:
            rsi_val = rsi[-1]
            if 50 <= rsi_val <= 70:
                rsi_score = 100  # 健康区间
                details['rsi_status'] = f"健康区间({rsi_val:.1f})"
            elif 70 < rsi_val <= 80:
                rsi_score = 85  # 略超买
                details['rsi_status'] = f"略超买({rsi_val:.1f})"
            elif rsi_val > 80:
                rsi_score = 60  # 超买
                details['rsi_status'] = f"超买({rsi_val:.1f})"
            elif rsi_val >= 40:
                rsi_score = 70
                details['rsi_status'] = f"中性({rsi_val:.1f})"
            else:
                rsi_score = 50
                details['rsi_status'] = f"弱势({rsi_val:.1f})"

            details['rsi_value'] = round(rsi_val, 2)
            details['rsi_score'] = rsi_score
        else:
            rsi_score = 50
            details['rsi_score'] = 50

        # 3. 涨幅持续性（20%权重）
        if len(close) >= 20:
            recent_return = (close[-1] - close[-5]) / close[-5] * 100
            mid_return = (close[-1] - close[-20]) / close[-20] * 100

            if mid_return > 0 and recent_return > 0:
                # 持续上涨
                consistency = min(100, recent_return / max(abs(mid_return), 0.1) * 50 + 50)
                sustain_score = min(100, consistency)
            elif mid_return > 0 and recent_return <= 0:
                # 中期上涨但近期回调
                sustain_score = 60
                details['sustain_status'] = "近期回调"
            else:
                sustain_score = 50

            details['recent_return'] = round(recent_return, 2)
            details['mid_return'] = round(mid_return, 2)
            details['sustain_score'] = sustain_score
        else:
            sustain_score = 50
            details['sustain_score'] = 50

        # 综合得分
        score = macd_score * 0.50 + rsi_score * 0.30 + sustain_score * 0.20
        details['total_score'] = round(score, 2)

        return round(score, 2), details

    def _calculate_volume_score(self, df: pd.DataFrame) -> Tuple[float, Dict]:
        """
        计算量能配合分（25%权重）

        评分标准：
        - 量比 = 近5日均量 / 近60日均量
        - 量比1.2-1.5（温和放量）→ 75分最健康
        - OBV修正系数：向上×1.1，向下×0.8
        """
        close = df['close'].values
        volume = df['volume'].values

        details = {}
        score = 50

        # 计算量比
        if len(volume) >= 60:
            vol_5d = np.mean(volume[-5:])
            vol_60d = np.mean(volume[-60:])
            vol_ratio = vol_5d / vol_60d if vol_60d > 0 else 1

            details['vol_ratio'] = round(vol_ratio, 2)

            # 量比评分
            if 1.2 <= vol_ratio <= 1.5:
                vol_score = 75  # 温和放量，最健康
                details['vol_status'] = "温和放量(最健康)"
            elif 1.5 < vol_ratio <= 2.0:
                vol_score = 65  # 放量明显
                details['vol_status'] = "放量明显"
            elif 1.0 <= vol_ratio < 1.2:
                vol_score = 60  # 量能平稳
                details['vol_status'] = "量能平稳"
            elif vol_ratio > 2.0:
                vol_score = 55  # 放量过大，有风险
                details['vol_status'] = "放量过大(谨慎)"
            else:
                vol_score = 50  # 缩量
                details['vol_status'] = "量能萎缩"
        else:
            vol_score = 50
            vol_ratio = 1.0
            details['vol_status'] = "数据不足"

        # OBV方向修正
        obv = self._obv(close, volume)
        if len(obv) >= 5:
            obv_slope = (obv[-1] - obv[-5]) / abs(obv[-5]) if obv[-5] != 0 else 0

            if obv_slope > 0:
                obv_modifier = 1.1
                details['obv_direction'] = "向上(+10%)"
            else:
                obv_modifier = 0.8
                details['obv_direction'] = "向下(-20%)"

            details['obv_slope'] = round(obv_slope, 4)
        else:
            obv_modifier = 1.0
            details['obv_direction'] = "数据不足"

        # 最终得分
        score = vol_score * obv_modifier
        score = min(100, max(0, score))
        details['final_score'] = round(score, 2)

        return round(score, 2), details

    def _calculate_relative_strength_score(self, df: pd.DataFrame) -> Tuple[float, Dict]:
        """
        计算相对强度分（15%权重）

        计算个股相对沪深300的超额收益
        """
        close = df['close'].values
        details = {}

        # 由于没有基准数据，使用简化的相对强度计算
        # 基于个股自身的涨幅分布

        if len(close) >= 60:
            # 计算不同周期的涨幅
            return_5d = (close[-1] - close[-5]) / close[-5] * 100
            return_20d = (close[-1] - close[-20]) / close[-20] * 100
            return_60d = (close[-1] - close[-60]) / close[-60] * 100

            details['return_5d'] = round(return_5d, 2)
            details['return_20d'] = round(return_20d, 2)
            details['return_60d'] = round(return_60d, 2)

            # 计算相对强度（假设基准为市场平均10%年化）
            # 这里简化处理：根据绝对涨幅评分
            if return_20d >= 15:
                rs_score = 100
                details['rs_status'] = "极强(20日涨幅>15%)"
            elif return_20d >= 10:
                rs_score = 90
                details['rs_status'] = "强(20日涨幅10-15%)"
            elif return_20d >= 5:
                rs_score = 75
                details['rs_status'] = "中等(20日涨幅5-10%)"
            elif return_20d >= 0:
                rs_score = 60
                details['rs_status'] = "一般(20日涨幅0-5%)"
            else:
                rs_score = max(40, 50 + return_20d)  # 下跌扣分
                details['rs_status'] = f"弱势(20日涨幅{return_20d:.1f}%)"

            # 加分：涨幅加速
            if return_5d > return_20d / 4:  # 近5日涨幅超过20日的25%
                rs_score = min(100, rs_score + 5)
                details['acceleration_bonus'] = "+5 涨幅加速"

        else:
            rs_score = 50
            details['rs_status'] = "数据不足"

        details['final_score'] = round(rs_score, 2)

        return round(rs_score, 2), details

    def _calculate_timing_coefficient(
        self, df: pd.DataFrame
    ) -> Tuple[float, str, Dict]:
        """
        计算入场时机系数

        时机系数（0.7 ~ 1.2）：
        - 回踩买点（最佳）：回调5-10%，MA20企稳 → 1.2
        - 突破买点（次佳）：放量突破平台 → 1.1
        - 趋势运行中（标准）→ 1.0
        - 短期过热（谨慎）：5日涨幅>15%或RSI>80 → 0.8
        - 追高风险（等待）：5日涨幅>25%或量价背离 → 0.7
        """
        close = df['close'].values
        volume = df['volume'].values

        details = {}

        if len(close) < 20:
            return 1.0, "标准", {'status': '数据不足'}

        # 计算关键指标
        ma20 = self._ma(close, 20)
        rsi = self._rsi(close, 14)

        # 5日涨幅
        return_5d = (close[-1] - close[-5]) / close[-5] * 100
        # 10日涨幅
        return_10d = (close[-1] - close[-10]) / close[-10] * 100 if len(close) >= 10 else 0
        # 距离MA20距离
        dist_to_ma20 = (close[-1] - ma20[-1]) / ma20[-1] * 100

        # 计算近期高点
        high_20d = np.max(close[-20:])
        dist_from_high = (high_20d - close[-1]) / high_20d * 100

        details['return_5d'] = round(return_5d, 2)
        details['return_10d'] = round(return_10d, 2)
        details['dist_to_ma20'] = round(dist_to_ma20, 2)
        details['dist_from_high'] = round(dist_from_high, 2)

        rsi_val = rsi[-1] if len(rsi) > 0 else 50
        details['rsi'] = round(rsi_val, 2)

        # 判断时机类型
        # 1. 追高风险：5日涨幅>25%或量价背离
        if return_5d > 25:
            return 0.7, "追高风险", dict(details, timing_reason="5日涨幅>25%")

        # 量价背离检测
        if return_5d > 15 and len(volume) >= 10:
            vol_ratio_recent = np.mean(volume[-5:]) / np.mean(volume[-10:])
            if vol_ratio_recent < 0.8:  # 缩量上涨
                return 0.7, "追高风险", dict(details, timing_reason="量价背离(缩量大涨)")

        # 2. 短期过热：5日涨幅>15%或RSI>80
        if return_5d > 15 or rsi_val > 80:
            reason = "5日涨幅>15%" if return_5d > 15 else f"RSI={rsi_val:.1f}>80"
            return 0.8, "短期过热", dict(details, timing_reason=reason)

        # 3. 回踩买点：回调5-10%，MA20企稳
        if 5 <= dist_from_high <= 10 and dist_to_ma20 <= 5:
            # 检查MA20附近企稳
            if len(close) >= 3:
                recent_volatility = np.std(close[-3:]) / np.mean(close[-3:])
                if recent_volatility < 0.03:  # 波动较小，企稳
                    return 1.2, "回踩买点", dict(details, timing_reason="回调至MA20企稳")

        # 4. 突破买点：放量突破近期平台
        if len(close) >= 20:
            high_20d_before = np.max(close[-20:-5])  # 排除最近5天
            if close[-1] > high_20d_before and return_5d > 3:
                # 放量突破
                vol_ratio = np.mean(volume[-3:]) / np.mean(volume[-20:])
                if vol_ratio > 1.3:
                    return 1.1, "突破买点", dict(details, timing_reason="放量突破平台")

        # 5. 趋势运行中（标准）
        return 1.0, "趋势运行", dict(details, timing_reason="趋势运行中")

    # ==================== 技术指标辅助函数 ====================

    def _ma(self, data: np.ndarray, period: int) -> np.ndarray:
        """简单移动平均"""
        result = np.full(len(data), np.nan)
        if len(data) >= period:
            result[period-1:] = np.convolve(data, np.ones(period)/period, mode='valid')
        return result

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """指数移动平均"""
        return pd.Series(data).ewm(span=period, adjust=False).mean().values

    def _macd(
        self,
        data: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD指标"""
        ema_fast = self._ema(data, fast)
        ema_slow = self._ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, signal)
        histogram = (macd_line - signal_line) * 2  # 柱状图放大显示
        return macd_line, signal_line, histogram

    def _rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """RSI指标"""
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = pd.Series(gains).rolling(period).mean().values
        avg_loss = pd.Series(losses).rolling(period).mean().values

        rsi = np.full_like(avg_gain, 50, dtype=float)
        valid_loss = avg_loss > 0
        rsi[valid_loss] = 100 - (100 / (1 + avg_gain[valid_loss] / avg_loss[valid_loss]))
        no_loss_with_gain = (avg_loss == 0) & (avg_gain > 0)
        rsi[no_loss_with_gain] = 100

        # 填充前面无效值
        rsi[:period] = 50

        return rsi

    def _obv(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """OBV指标"""
        obv = np.zeros(len(close))
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        return obv


def analyze_trend_quality(df: pd.DataFrame) -> Dict:
    """
    分析股票趋势质量

    Args:
        df: 股票数据

    Returns:
        Dict: 趋势分析结果
    """
    system = TrendScoringSystem()
    result = system.calculate_score(df)

    return {
        'final_score': result.final_score,
        'trend_score': result.trend_total_score,
        'timing_coefficient': result.timing_coefficient,
        'ma_score': result.ma_structure_score,
        'momentum_score': result.price_momentum_score,
        'volume_score': result.volume_score,
        'rs_score': result.relative_strength_score,
        'timing_type': result.timing_type,
        'passed_filter': result.passed_hard_filter,
        'filter_reason': result.hard_filter_reason,
        'recommendation': _get_recommendation(result)
    }


def _get_recommendation(result: TrendScoreResult) -> str:
    """根据评分生成推荐"""
    if not result.passed_hard_filter:
        return f"不推荐: {result.hard_filter_reason}"

    if result.final_score >= 90:
        return "强烈推荐: 极强趋势+时机好"
    elif result.final_score >= 75:
        return "推荐: 强趋势"
    elif result.final_score >= 60:
        return "谨慎: 趋势一般"
    else:
        return "不推荐: 趋势弱"
