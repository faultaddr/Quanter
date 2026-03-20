"""
低位盘整突破评分系统

核心理念：
- 技术面负责"形态与时点"（低位盘整 + 放量突破）
- 因子面负责"质量与胜率"（基本面/估值/资金/波动/动量筛选与打分）

评分架构：
1. 形态检测层：低位 + 盘整 + 突破三要素检测
2. 因子评分层：质量/成长/估值/动量/资金/风险六大因子
3. 综合评分：形态得分 × 因子质量得分
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class BreakoutScoreResult:
    """低位盘整突破评分结果"""
    # 最终评分
    final_score: float
    # 形态检测
    is_low_position: bool
    is_consolidating: bool
    has_breakout: bool
    # 因子得分
    quality_score: float      # 基本面质量
    growth_score: float       # 成长性
    value_score: float        # 估值
    momentum_score: float     # 动量
    flow_score: float         # 资金流向
    risk_score: float         # 风险
    # 形态细节
    consolidation_days: int
    price_range: float        # 盘整振幅
    volume_ratio: float       # 量比
    breakout_strength: float  # 突破强度
    # 交易参数
    stop_loss_price: float
    take_profit_price: float
    # 过滤结果
    passed_filter: bool
    filter_reason: str
    # 明细
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'final_score': self.final_score,
            'is_low_position': self.is_low_position,
            'is_consolidating': self.is_consolidating,
            'has_breakout': self.has_breakout,
            'quality_score': self.quality_score,
            'growth_score': self.growth_score,
            'value_score': self.value_score,
            'momentum_score': self.momentum_score,
            'flow_score': self.flow_score,
            'risk_score': self.risk_score,
            'consolidation_days': self.consolidation_days,
            'price_range': self.price_range,
            'volume_ratio': self.volume_ratio,
            'breakout_strength': self.breakout_strength,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
            'passed_filter': self.passed_filter,
            'filter_reason': self.filter_reason,
            'details': self.details
        }


class BreakoutScoringSystem:
    """
    低位盘整突破评分系统

    寻找处于低位、经历盘整后放量突破的股票
    """

    # 盘整参数
    CONSOLIDATION_PARAMS = {
        'min_days': 20,          # 最少盘整天数
        'max_days': 40,          # 最多盘整天数
        'max_range': 0.18,       # 最大振幅 18%
        'min_range': 0.12,       # 最小振幅 12%（确保有足够波动）
    }

    # 突破参数
    BREAKOUT_PARAMS = {
        'price_breakout_epsilon': 0.02,  # 价格突破阈值 2%
        'volume_ratio_min': 1.5,          # 最小量比
        'close_confirm_max': 0.3,         # 收盘价确认比例（上影线不超过30%）
    }

    # 低位参数
    LOW_POSITION_PARAMS = {
        'lookback_period': 250,           # 回看周期
        'drawdown_threshold': 0.30,       # 回撤阈值 30%
        'percentile_threshold': 0.35,     # 分位数阈值 35%
    }

    # 因子权重
    FACTOR_WEIGHTS = {
        'quality': 0.25,    # 基本面质量
        'growth': 0.20,     # 成长性
        'value': 0.20,      # 估值
        'momentum': 0.20,   # 动量
        'flow': 0.10,       # 资金流向
        'risk': 0.05,       # 风险
    }

    def __init__(
        self,
        min_amount: float = 5e7,   # 最低日均成交额 5000万
        min_list_days: int = 120,  # 最低上市天数
    ):
        """
        初始化低位盘整突破评分系统

        Args:
            min_amount: 最低日均成交额
            min_list_days: 最低上市天数
        """
        self.min_amount = min_amount
        self.min_list_days = min_list_days

    def calculate_score(self, df: pd.DataFrame) -> BreakoutScoreResult:
        """
        计算低位盘整突破评分

        Args:
            df: 股票数据，需包含 open, high, low, close, volume 等列

        Returns:
            BreakoutScoreResult: 评分结果
        """
        # 初始化默认结果
        default_result = self._create_default_result()

        # 数据验证
        if len(df) < 60:
            default_result.filter_reason = "数据不足60天"
            return default_result

        # 1. 检测低位
        is_low, low_details = self._detect_low_position(df)

        # 2. 检测盘整
        is_consolidating, consolidation_details = self._detect_consolidation(df)

        # 3. 检测突破
        has_breakout, breakout_details = self._detect_breakout(df)

        # 如果不满足形态条件，直接返回
        if not (is_low and is_consolidating and has_breakout):
            reasons = []
            if not is_low:
                reasons.append("非低位")
            if not is_consolidating:
                reasons.append("非盘整形态")
            if not has_breakout:
                reasons.append("无突破信号")

            default_result.is_low_position = is_low
            default_result.is_consolidating = is_consolidating
            default_result.has_breakout = has_breakout
            default_result.passed_filter = False
            default_result.filter_reason = "、".join(reasons)
            default_result.details = {
                'low_position': low_details,
                'consolidation': consolidation_details,
                'breakout': breakout_details
            }
            return default_result

        # 4. 计算因子得分
        factor_scores = self._calculate_factor_scores(df)

        # 5. 计算形态得分
        pattern_score = self._calculate_pattern_score(
            consolidation_details, breakout_details
        )

        # 6. 计算综合评分
        factor_total = sum(
            factor_scores.get(key, 50) * weight
            for key, weight in self.FACTOR_WEIGHTS.items()
        )

        # 最终评分 = 形态得分 × 因子质量系数
        # 因子质量系数：因子得分/50（50分为基准）
        quality_factor = factor_total / 50.0
        final_score = min(100, pattern_score * quality_factor)

        # 7. 计算交易参数
        stop_loss, take_profit = self._calculate_trading_params(
            df, consolidation_details, breakout_details
        )

        return BreakoutScoreResult(
            final_score=round(final_score, 2),
            is_low_position=True,
            is_consolidating=True,
            has_breakout=True,
            quality_score=round(factor_scores.get('quality', 50), 2),
            growth_score=round(factor_scores.get('growth', 50), 2),
            value_score=round(factor_scores.get('value', 50), 2),
            momentum_score=round(factor_scores.get('momentum', 50), 2),
            flow_score=round(factor_scores.get('flow', 50), 2),
            risk_score=round(factor_scores.get('risk', 50), 2),
            consolidation_days=consolidation_details.get('days', 0),
            price_range=consolidation_details.get('range', 0),
            volume_ratio=breakout_details.get('volume_ratio', 1.0),
            breakout_strength=breakout_details.get('strength', 0),
            stop_loss_price=round(stop_loss, 2),
            take_profit_price=round(take_profit, 2),
            passed_filter=True,
            filter_reason="通过形态筛选",
            details={
                'low_position': low_details,
                'consolidation': consolidation_details,
                'breakout': breakout_details,
                'pattern_score': pattern_score,
                'quality_factor': quality_factor
            }
        )

    def _create_default_result(self) -> BreakoutScoreResult:
        """创建默认结果"""
        return BreakoutScoreResult(
            final_score=0,
            is_low_position=False,
            is_consolidating=False,
            has_breakout=False,
            quality_score=50,
            growth_score=50,
            value_score=50,
            momentum_score=50,
            flow_score=50,
            risk_score=50,
            consolidation_days=0,
            price_range=0,
            volume_ratio=1.0,
            breakout_strength=0,
            stop_loss_price=0,
            take_profit_price=0,
            passed_filter=False,
            filter_reason="未检测"
        )

    def _detect_low_position(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        检测低位形态

        低位定义（满足其一）：
        1. 距250日高点回撤 >= 30%
        2. 价格处于250日分位 <= 35%

        Args:
            df: 股票数据

        Returns:
            Tuple[bool, Dict]: (是否低位, 详情)
        """
        close = df['close'].values
        high = df['high'].values

        lookback = self.LOW_POSITION_PARAMS['lookback_period']
        drawdown_threshold = self.LOW_POSITION_PARAMS['drawdown_threshold']
        percentile_threshold = self.LOW_POSITION_PARAMS['percentile_threshold']

        details = {}

        if len(close) < lookback:
            # 数据不足，使用可用数据
            lookback = len(close)

        # 计算250日最高价
        high_n = np.max(high[-lookback:])
        current_price = close[-1]

        # 计算回撤幅度
        drawdown = (high_n - current_price) / high_n

        # 计算价格分位数
        low_n = np.min(close[-lookback:])
        price_range = high_n - low_n
        if price_range > 0:
            percentile = (current_price - low_n) / price_range
        else:
            percentile = 0.5

        details['high_n'] = round(high_n, 2)
        details['low_n'] = round(low_n, 2)
        details['current_price'] = round(current_price, 2)
        details['drawdown'] = round(drawdown, 4)
        details['percentile'] = round(percentile, 4)
        details['drawdown_threshold'] = drawdown_threshold
        details['percentile_threshold'] = percentile_threshold

        # 判断低位
        is_low = drawdown >= drawdown_threshold or percentile <= percentile_threshold

        if is_low:
            if drawdown >= drawdown_threshold:
                details['reason'] = f"回撤达标: {drawdown*100:.1f}% >= {drawdown_threshold*100:.0f}%"
            else:
                details['reason'] = f"分位达标: {percentile*100:.1f}% <= {percentile_threshold*100:.0f}%"
        else:
            details['reason'] = f"非低位: 回撤{drawdown*100:.1f}%, 分位{percentile*100:.1f}%"

        return is_low, details

    def _detect_consolidation(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        检测盘整形态

        盘整特征：
        1. 振幅收敛：近期振幅小于前期
        2. 均线走平：MA20斜率接近0
        3. 波动率下降

        Args:
            df: 股票数据

        Returns:
            Tuple[bool, Dict]: (是否盘整, 详情)
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        min_days = self.CONSOLIDATION_PARAMS['min_days']
        max_days = self.CONSOLIDATION_PARAMS['max_days']
        max_range = self.CONSOLIDATION_PARAMS['max_range']

        details = {}

        if len(close) < min_days + 10:
            return False, {'reason': '数据不足'}

        # 寻找盘整区间
        best_consolidation = None
        best_score = 0

        for days in range(min_days, min(max_days + 1, len(close) - 5)):
            # 检查过去days天的数据
            recent_high = np.max(high[-days-5:-5])  # 排除最近5天（可能是突破）
            recent_low = np.min(low[-days-5:-5])
            recent_range = (recent_high - recent_low) / recent_low

            # 计算MA20斜率
            ma20 = self._ma(close, 20)
            if len(ma20) >= days:
                ma20_slope = abs(ma20[-5] - ma20[-days-5]) / ma20[-days-5] if ma20[-days-5] > 0 else 1
            else:
                ma20_slope = 1

            # 计算波动率
            returns = np.diff(close[-days-5:-5]) / close[-days-5:-6]
            volatility = np.std(returns)

            # 盘整评分：振幅小、斜率小、波动率小
            if recent_range <= max_range:
                score = (1 - recent_range/max_range) * 0.5 + \
                        (1 - min(ma20_slope, 0.1)/0.1) * 0.3 + \
                        (1 - min(volatility, 0.03)/0.03) * 0.2

                if score > best_score:
                    best_score = score
                    best_consolidation = {
                        'days': days,
                        'range': recent_range,
                        'high': recent_high,
                        'low': recent_low,
                        'ma20_slope': ma20_slope,
                        'volatility': volatility,
                        'score': score
                    }

        if best_consolidation is None:
            details['reason'] = '未找到符合条件的盘整区间'
            return False, details

        details.update(best_consolidation)
        details['reason'] = f"盘整{best_consolidation['days']}天, 振幅{best_consolidation['range']*100:.1f}%"

        # 判断是否有效盘整
        is_consolidating = best_score >= 0.4

        return is_consolidating, details

    def _detect_breakout(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        检测突破信号

        突破特征：
        1. 价格突破：收盘价突破盘整上沿
        2. 成交放量：量比 >= 1.5
        3. 收盘确认：收盘价接近最高价

        Args:
            df: 股票数据

        Returns:
            Tuple[bool, Dict]: (是否突破, 详情)
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        epsilon = self.BREAKOUT_PARAMS['price_breakout_epsilon']
        vol_ratio_min = self.BREAKOUT_PARAMS['volume_ratio_min']
        close_confirm_max = self.BREAKOUT_PARAMS['close_confirm_max']

        details = {}

        if len(close) < 30:
            return False, {'reason': '数据不足'}

        # 计算近期高点（排除当天）
        recent_high = np.max(high[-21:-1])  # 过去20天最高

        # 当日数据
        today_high = high[-1]
        today_low = low[-1]
        today_close = close[-1]
        today_open = df['open'].values[-1] if 'open' in df.columns else close[-1]

        # 价格突破检测
        price_breakout = today_close > recent_high * (1 - epsilon)

        # 成交量检测
        vol_ma = np.mean(volume[-21:-1])  # 过去20天均量
        today_vol = volume[-1]
        vol_ratio = today_vol / vol_ma if vol_ma > 0 else 1

        volume_breakout = vol_ratio >= vol_ratio_min

        # 收盘确认（上影线比例）
        day_range = today_high - today_low
        upper_shadow = today_high - today_close
        upper_shadow_ratio = upper_shadow / day_range if day_range > 0 else 0

        close_confirmed = upper_shadow_ratio <= close_confirm_max

        # 突破强度
        breakout_strength = 0
        if price_breakout:
            breakout_strength += 40
        if volume_breakout:
            breakout_strength += 30
        if close_confirmed:
            breakout_strength += 30

        details['recent_high'] = round(recent_high, 2)
        details['today_close'] = round(today_close, 2)
        details['price_breakout'] = price_breakout
        details['vol_ratio'] = round(vol_ratio, 2)
        details['volume_breakout'] = volume_breakout
        details['upper_shadow_ratio'] = round(upper_shadow_ratio, 2)
        details['close_confirmed'] = close_confirmed
        details['strength'] = breakout_strength

        # 判断是否有效突破
        has_breakout = price_breakout and volume_breakout

        if has_breakout:
            details['reason'] = f"突破成功: 价格+{((today_close/recent_high-1)*100):.1f}%, 量比{vol_ratio:.1f}"
        else:
            reasons = []
            if not price_breakout:
                reasons.append("价格未突破")
            if not volume_breakout:
                reasons.append(f"量比不足({vol_ratio:.1f}<{vol_ratio_min})")
            details['reason'] = "、".join(reasons) if reasons else "未突破"

        return has_breakout, details

    def _calculate_factor_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        计算因子得分

        Args:
            df: 股票数据

        Returns:
            Dict[str, float]: 各因子得分
        """
        scores = {}

        # 1. 质量因子（基本面质量）- 使用技术指标代理
        scores['quality'] = self._calculate_quality_score(df)

        # 2. 成长因子（成长性）
        scores['growth'] = self._calculate_growth_score(df)

        # 3. 价值因子（估值）- 使用相对位置代理
        scores['value'] = self._calculate_value_score(df)

        # 4. 动量因子
        scores['momentum'] = self._calculate_momentum_score(df)

        # 5. 资金流向因子
        scores['flow'] = self._calculate_flow_score(df)

        # 6. 风险因子
        scores['risk'] = self._calculate_risk_score(df)

        return scores

    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """
        计算质量得分

        使用ROE趋势代理（基于价格与MA关系）
        """
        close = df['close'].values

        # 价格相对于长期均线的稳定性
        ma60 = self._ma(close, 60)
        ma120 = self._ma(close, 120)

        if len(ma60) > 0 and len(ma120) > 0 and not np.isnan(ma60[-1]) and not np.isnan(ma120[-1]):
            # 价格稳定性
            price_stability = 1 - abs(close[-1] - ma60[-1]) / ma60[-1]
            # 均线稳定性
            ma_stability = 1 - abs(ma60[-1] - ma120[-1]) / ma120[-1] if ma120[-1] > 0 else 0.5

            score = (price_stability * 0.5 + ma_stability * 0.5) * 100
        else:
            score = 50

        return max(0, min(100, score))

    def _calculate_growth_score(self, df: pd.DataFrame) -> float:
        """
        计算成长得分

        基于中长期涨幅趋势
        """
        close = df['close'].values

        if len(close) < 120:
            return 50

        # 60日涨幅
        return_60 = (close[-1] - close[-60]) / close[-60] if close[-60] > 0 else 0
        # 120日涨幅
        return_120 = (close[-1] - close[-120]) / close[-120] if close[-120] > 0 else 0

        # 成长性评分：中期涨幅 > 长期涨幅 表示加速
        if return_120 > 0:
            acceleration = return_60 / return_120
            if acceleration > 1:
                score = 60 + min(40, acceleration * 20)
            else:
                score = 40 + acceleration * 20
        else:
            score = 30

        return max(0, min(100, score))

    def _calculate_value_score(self, df: pd.DataFrame) -> float:
        """
        计算价值得分

        基于价格相对位置（低位高分）
        """
        close = df['close'].values
        low = df['low'].values
        high = df['high'].values

        if len(close) < 250:
            lookback = len(close)
        else:
            lookback = 250

        # 计算价格分位
        low_n = np.min(low[-lookback:])
        high_n = np.max(high[-lookback:])
        price_range = high_n - low_n

        if price_range > 0:
            percentile = (close[-1] - low_n) / price_range
        else:
            percentile = 0.5

        # 低位高分，高位低分
        score = (1 - percentile) * 100

        return max(0, min(100, score))

    def _calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """
        计算动量得分

        基于60日/120日收益率和相对沪深300超额
        """
        close = df['close'].values

        if len(close) < 60:
            return 50

        # 60日收益率
        return_60 = (close[-1] - close[-60]) / close[-60] if close[-60] > 0 else 0

        # 相对强度（与自身历史比较）
        returns_20 = []
        for i in range(20, min(120, len(close)), 20):
            r = (close[-1] - close[-i]) / close[-i] if close[-i] > 0 else 0
            returns_20.append(r)

        if returns_20:
            avg_return = np.mean(returns_20)
            return_consistency = sum(1 for r in returns_20 if r > 0) / len(returns_20)
        else:
            avg_return = 0
            return_consistency = 0.5

        # 动量评分
        if return_60 > 0.1:  # 60日涨幅超过10%
            score = 70 + return_consistency * 30
        elif return_60 > 0:
            score = 50 + return_60 * 200
        else:
            score = max(30, 50 + return_60 * 100)

        return max(0, min(100, score))

    def _calculate_flow_score(self, df: pd.DataFrame) -> float:
        """
        计算资金流向得分

        基于OBV和成交量变化
        """
        close = df['close'].values
        volume = df['volume'].values

        # OBV
        obv = self._obv(close, volume)

        if len(obv) < 20:
            return 50

        # OBV趋势
        obv_change = (obv[-1] - obv[-20]) / abs(obv[-20]) if obv[-20] != 0 else 0

        # 量价配合
        price_change = (close[-1] - close[-20]) / close[-20] if close[-20] > 0 else 0

        # 量价同向为佳
        if price_change > 0 and obv_change > 0:
            score = 70 + min(30, obv_change * 50)
        elif price_change > 0 and obv_change < 0:
            score = 40  # 量价背离
        elif price_change < 0 and obv_change > 0:
            score = 60  # 可能是底部吸筹
        else:
            score = 50

        return max(0, min(100, score))

    def _calculate_risk_score(self, df: pd.DataFrame) -> float:
        """
        计算风险得分

        基于波动率、最大回撤、Beta
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        if len(close) < 60:
            return 50

        # 计算波动率
        returns = np.diff(close[-60:]) / close[-60:-1]
        volatility = np.std(returns)

        # 计算最大回撤
        cumulative = np.maximum.accumulate(close[-60:])
        drawdown = (cumulative - close[-60:]) / cumulative
        max_drawdown = np.max(drawdown)

        # 风险评分（低波动、低回撤 = 高分）
        vol_score = max(0, 100 - volatility * 1000)  # 波动率越低分越高
        dd_score = max(0, 100 - max_drawdown * 200)  # 回撤越低分越高

        score = vol_score * 0.5 + dd_score * 0.5

        return max(0, min(100, score))

    def _calculate_pattern_score(
        self,
        consolidation_details: Dict,
        breakout_details: Dict
    ) -> float:
        """
        计算形态得分

        Args:
            consolidation_details: 盘整详情
            breakout_details: 突破详情

        Returns:
            float: 形态得分
        """
        # 盘整质量得分
        consolidation_score = consolidation_details.get('score', 0.5) * 100

        # 突破强度得分
        breakout_strength = breakout_details.get('strength', 0)

        # 综合形态得分
        pattern_score = consolidation_score * 0.4 + breakout_strength * 0.6

        return pattern_score

    def _calculate_trading_params(
        self,
        df: pd.DataFrame,
        consolidation_details: Dict,
        breakout_details: Dict
    ) -> Tuple[float, float]:
        """
        计算交易参数（止损/止盈）

        Args:
            df: 股票数据
            consolidation_details: 盘整详情
            breakout_details: 突破详情

        Returns:
            Tuple[float, float]: (止损价, 止盈价)
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        current_price = close[-1]

        # 止损：跌破盘整上沿×0.98 或 ATR止损
        consolidation_low = consolidation_details.get('low', current_price * 0.95)
        atr = self._atr(df, 14)

        # 止损价：取较低者（更宽松）
        stop_loss = max(consolidation_low * 0.98, current_price - atr * 2)

        # 止盈：前高压力位 或 2:1 盈亏比
        lookback = min(60, len(high))
        prev_high = np.max(high[-lookback:-1])

        risk = current_price - stop_loss
        take_profit_ratio = current_price + risk * 2

        # 止盈价：取较低者（更保守）
        take_profit = min(prev_high, take_profit_ratio)

        return stop_loss, take_profit

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

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """ATR指标"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )

        return np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)


def analyze_breakout_quality(df: pd.DataFrame) -> Dict:
    """
    分析股票低位盘整突破质量

    Args:
        df: 股票数据

    Returns:
        Dict: 分析结果
    """
    system = BreakoutScoringSystem()
    result = system.calculate_score(df)

    return {
        'final_score': result.final_score,
        'is_low_position': result.is_low_position,
        'is_consolidating': result.is_consolidating,
        'has_breakout': result.has_breakout,
        'quality_score': result.quality_score,
        'growth_score': result.growth_score,
        'value_score': result.value_score,
        'momentum_score': result.momentum_score,
        'flow_score': result.flow_score,
        'risk_score': result.risk_score,
        'consolidation_days': result.consolidation_days,
        'volume_ratio': result.volume_ratio,
        'breakout_strength': result.breakout_strength,
        'stop_loss': result.stop_loss_price,
        'take_profit': result.take_profit_price,
        'passed_filter': result.passed_filter,
        'filter_reason': result.filter_reason,
        'recommendation': _get_breakout_recommendation(result)
    }


def _get_breakout_recommendation(result: BreakoutScoreResult) -> str:
    """根据评分生成推荐"""
    if not result.passed_filter:
        return f"不推荐: {result.filter_reason}"

    if not (result.is_low_position and result.is_consolidating and result.has_breakout):
        return f"不推荐: 形态不完整"

    if result.final_score >= 80:
        return "强烈推荐: 完美形态+优质因子"
    elif result.final_score >= 70:
        return "推荐: 优质突破形态"
    elif result.final_score >= 60:
        return "谨慎: 形态尚可，关注因子质量"
    else:
        return "不推荐: 因子质量较差"