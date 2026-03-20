"""
机器学习特征工程模块

使用技术指标生成机器学习特征，支持XGBoost等模型训练
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FeatureConfig:
    """特征配置"""
    # 趋势类
    ma_periods: List[int] = None
    ema_periods: List[int] = None

    # 动量类
    mom_periods: List[int] = None
    rsi_periods: List[int] = None

    # 波动率类
    boll_period: int = 20
    boll_std: float = 2.0
    atr_period: int = 14

    # 位置类
    position_periods: List[int] = None

    # 成交量类
    vol_ma_period: int = 20

    def __post_init__(self):
        if self.ma_periods is None:
            self.ma_periods = [5, 10, 20, 60, 120]
        if self.ema_periods is None:
            self.ema_periods = [12, 26, 50]
        if self.mom_periods is None:
            self.mom_periods = [5, 10, 20, 60]
        if self.rsi_periods is None:
            self.rsi_periods = [6, 12, 24]
        if self.position_periods is None:
            self.position_periods = [20, 60]


class TechnicalIndicators:
    """技术指标计算器"""

    @staticmethod
    def MA(series: pd.Series, period: int) -> pd.Series:
        """简单移动平均"""
        return series.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def EMA(series: pd.Series, period: int) -> pd.Series:
        """指数移动平均"""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def STD(series: pd.Series, period: int) -> pd.Series:
        """标准差"""
        return series.rolling(window=period, min_periods=period).std(ddof=0)

    @staticmethod
    def HHV(series: pd.Series, period: int) -> pd.Series:
        """最高值"""
        return series.rolling(window=period, min_periods=period).max()

    @staticmethod
    def LLV(series: pd.Series, period: int) -> pd.Series:
        """最低值"""
        return series.rolling(window=period, min_periods=period).min()

    @staticmethod
    def REF(series: pd.Series, periods: int) -> pd.Series:
        """前值"""
        return series.shift(periods)

    @staticmethod
    def RSI(close: pd.Series, period: int) -> pd.Series:
        """相对强弱指标"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def MACD(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD指标"""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        macd_bar = 2 * (dif - dea)
        return dif, dea, macd_bar

    @staticmethod
    def KDJ(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9, m1: int = 3, m2: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """KDJ指标"""
        hhv = high.rolling(window=n, min_periods=n).max()
        llv = low.rolling(window=n, min_periods=n).min()
        rsv = (close - llv) / (hhv - llv + 1e-10) * 100

        k = rsv.ewm(alpha=1/m1, adjust=False).mean()
        d = k.ewm(alpha=1/m2, adjust=False).mean()
        j = 3 * k - 2 * d
        return k, d, j

    @staticmethod
    def BOLL(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """布林带"""
        mid = close.rolling(window=period, min_periods=period).mean()
        std = close.rolling(window=period, min_periods=period).std(ddof=0)
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        return upper, mid, lower

    @staticmethod
    def ATR(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """平均真实波幅"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def AROON(high: pd.Series, low: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """阿隆指标"""
        aroon_up = high.rolling(window=period + 1).apply(
            lambda x: (period - (period - x.argmax())) / period * 100, raw=True
        )
        aroon_down = low.rolling(window=period + 1).apply(
            lambda x: (period - (period - x.argmin())) / period * 100, raw=True
        )
        return aroon_up, aroon_down

    @staticmethod
    def CCI(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """商品通道指标"""
        tp = (high + low + close) / 3
        ma = tp.rolling(window=period, min_periods=period).mean()
        md = tp.rolling(window=period, min_periods=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        return (tp - ma) / (0.015 * md + 1e-10)

    @staticmethod
    def WILLR(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """威廉指标"""
        hhv = high.rolling(window=period, min_periods=period).max()
        llv = low.rolling(window=period, min_periods=period).min()
        return (hhv - close) / (hhv - llv + 1e-10) * -100

    @staticmethod
    def MFI(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """资金流量指标"""
        tp = (high + low + close) / 3
        mf = tp * volume

        positive_mf = pd.Series(np.where(tp > tp.shift(1), mf, 0), index=tp.index)
        negative_mf = pd.Series(np.where(tp < tp.shift(1), mf, 0), index=tp.index)

        positive_sum = positive_mf.rolling(window=period).sum()
        negative_sum = negative_mf.rolling(window=period).sum()

        mfr = positive_sum / (negative_sum + 1e-10)
        return 100 - (100 / (1 + mfr))

    @staticmethod
    def OBV(close: pd.Series, volume: pd.Series) -> pd.Series:
        """能量潮指标"""
        direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
        return (pd.Series(direction, index=close.index) * volume).cumsum()


class MLFeatureEngineer:
    """
    机器学习特征工程

    生成多维度技术指标特征用于机器学习模型
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.indicators = TechnicalIndicators()
        self.feature_names: List[str] = []
        self.feature_groups: Dict[str, List[str]] = {}

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成所有特征

        Args:
            df: OHLCV数据，需要包含 open, high, low, close, volume 列

        Returns:
            DataFrame: 特征矩阵
        """
        if len(df) < 120:
            raise ValueError(f"数据不足，需要至少120条数据，当前只有{len(df)}条")

        features = {}
        close = df['close']
        high = df['high']
        low = df['low']
        open_ = df['open']
        volume = df.get('volume', df.get('vol', pd.Series(0, index=df.index)))

        # ==================== 1. 趋势类特征 ====================
        trend_features = self._generate_trend_features(close, high, low)
        features.update(trend_features)

        # ==================== 2. 动量类特征 ====================
        momentum_features = self._generate_momentum_features(close, high, low)
        features.update(momentum_features)

        # ==================== 3. 波动率类特征 ====================
        volatility_features = self._generate_volatility_features(close, high, low)
        features.update(volatility_features)

        # ==================== 4. 位置类特征 ====================
        position_features = self._generate_position_features(close, high, low)
        features.update(position_features)

        # ==================== 5. 成交量类特征 ====================
        volume_features = self._generate_volume_features(close, high, low, volume)
        features.update(volume_features)

        # ==================== 6. 价格形态特征 ====================
        pattern_features = self._generate_pattern_features(close, high, low, open_)
        features.update(pattern_features)

        # ==================== 7. 衍生特征 ====================
        derived_features = self._generate_derived_features(features, close)
        features.update(derived_features)

        # 构建 DataFrame
        feature_df = pd.DataFrame(features, index=df.index)

        # 处理 NaN 和无穷值
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.ffill().bfill().fillna(0)

        self.feature_names = list(feature_df.columns)
        self._build_feature_groups()

        return feature_df

    def _generate_trend_features(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Dict:
        """生成趋势类特征"""
        features = {}

        # 移动平均
        for period in self.config.ma_periods:
            features[f'ma_{period}'] = self.indicators.MA(close, period)
            features[f'price_ma{period}_ratio'] = close / (features[f'ma_{period}'] + 1e-10)

        # 指数移动平均
        for period in self.config.ema_periods:
            features[f'ema_{period}'] = self.indicators.EMA(close, period)

        # MACD
        dif, dea, macd_bar = self.indicators.MACD(close)
        features['macd_dif'] = dif
        features['macd_dea'] = dea
        features['macd_bar'] = macd_bar
        features['macd_cross'] = ((dif > dea) & (dif.shift(1) <= dea.shift(1))).astype(float)

        # 均线斜率
        features['ma20_slope'] = self.indicators.MA(close, 20).diff(5) / (self.indicators.MA(close, 20) + 1e-10)
        features['ma60_slope'] = self.indicators.MA(close, 60).diff(10) / (self.indicators.MA(close, 60) + 1e-10)

        # 均线排列
        ma5 = self.indicators.MA(close, 5)
        ma10 = self.indicators.MA(close, 10)
        ma20 = self.indicators.MA(close, 20)
        ma60 = self.indicators.MA(close, 60)
        features['ma_bullish_alignment'] = ((ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60)).astype(float)
        features['ma_bearish_alignment'] = ((ma5 < ma10) & (ma10 < ma20) & (ma20 < ma60)).astype(float)

        return features

    def _generate_momentum_features(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Dict:
        """生成动量类特征"""
        features = {}

        # 动量
        for period in self.config.mom_periods:
            features[f'mom_{period}'] = (close - self.indicators.REF(close, period)) / (self.indicators.REF(close, period) + 1e-10) * 100

        # RSI
        for period in self.config.rsi_periods:
            features[f'rsi_{period}'] = self.indicators.RSI(close, period)

        # KDJ
        k, d, j = self.indicators.KDJ(high, low, close)
        features['kdj_k'] = k
        features['kdj_d'] = d
        features['kdj_j'] = j
        features['kdj_cross'] = ((k > d) & (k.shift(1) <= d.shift(1))).astype(float)

        # CCI
        features['cci_14'] = self.indicators.CCI(high, low, close, 14)

        # Williams %R
        features['willr_14'] = self.indicators.WILLR(high, low, close, 14)

        # Aroon
        aroon_up, aroon_down = self.indicators.AROON(high, low, 14)
        features['aroon_up'] = aroon_up
        features['aroon_down'] = aroon_down
        features['aroon_osc'] = aroon_up - aroon_down

        # 动量加速度
        features['mom_accel_5'] = features['mom_5'] - features.get('mom_5', pd.Series(0, index=close.index)).shift(5)

        return features

    def _generate_volatility_features(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Dict:
        """生成波动率类特征"""
        features = {}

        # 布林带
        upper, mid, lower = self.indicators.BOLL(close, self.config.boll_period, self.config.boll_std)
        features['boll_upper'] = upper
        features['boll_mid'] = mid
        features['boll_lower'] = lower
        features['boll_width'] = (upper - lower) / (mid + 1e-10)
        features['boll_position'] = (close - lower) / (upper - lower + 1e-10)
        features['boll_squeeze'] = features['boll_width'] / features['boll_width'].rolling(20).mean()

        # ATR
        features['atr_14'] = self.indicators.ATR(high, low, close, 14)
        features['atr_ratio'] = features['atr_14'] / (close + 1e-10)

        # 标准差
        features['std_20'] = self.indicators.STD(close, 20)
        features['std_60'] = self.indicators.STD(close, 60)

        # 波动率变化
        features['volatility_change'] = features['std_20'] / (self.indicators.REF(features['std_20'], 10) + 1e-10)

        # 振幅
        features['amplitude'] = (high - low) / (close + 1e-10)

        return features

    def _generate_position_features(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Dict:
        """生成位置类特征"""
        features = {}

        for period in self.config.position_periods:
            hhv = self.indicators.HHV(close, period)
            llv = self.indicators.LLV(close, period)

            # 位置指标 (0-1之间，越接近1越接近高点)
            features[f'pos_{period}'] = (close - llv) / (hhv - llv + 1e-10)

            # 距离高点天数
            features[f'hhvbars_{period}'] = close.rolling(window=period).apply(
                lambda x: period - 1 - np.argmax(x) if len(x) == period else 0, raw=True
            )

            # 距离低点天数
            features[f'llvbars_{period}'] = close.rolling(window=period).apply(
                lambda x: period - 1 - np.argmin(x) if len(x) == period else 0, raw=True
            )

        # 乖离率
        ma20 = self.indicators.MA(close, 20)
        ma60 = self.indicators.MA(close, 60)
        features['bias_20'] = (close - ma20) / (ma20 + 1e-10) * 100
        features['bias_60'] = (close - ma60) / (ma60 + 1e-10) * 100

        # 相对位置
        features['price_vs_ma20'] = np.where(close > ma20, 1, -1)
        features['price_vs_ma60'] = np.where(close > ma60, 1, -1)

        return features

    def _generate_volume_features(self, close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> Dict:
        """生成成交量类特征"""
        features = {}

        if volume.sum() == 0:
            # 如果没有成交量数据，填充默认值
            for key in ['vol_ma20', 'vol_ratio', 'obv', 'mfi', 'vol_change', 'vol_trend']:
                features[key] = pd.Series(0, index=close.index)
            return features

        # 成交量均线
        features['vol_ma20'] = self.indicators.MA(volume, 20)

        # 量比
        features['vol_ratio'] = volume / (features['vol_ma20'] + 1e-10)

        # OBV
        features['obv'] = self.indicators.OBV(close, volume)
        features['obv_ma20'] = self.indicators.MA(features['obv'], 20)

        # MFI
        features['mfi'] = self.indicators.MFI(high, low, close, volume, 14)

        # 成交量变化
        features['vol_change'] = volume / (self.indicators.REF(volume, 5) + 1e-10)

        # 量价配合
        features['vol_price_trend'] = np.where(
            (close > close.shift(1)) & (volume > volume.shift(1)), 1,
            np.where((close < close.shift(1)) & (volume < volume.shift(1)), -1, 0)
        )

        return features

    def _generate_pattern_features(self, close: pd.Series, high: pd.Series, low: pd.Series, open_: pd.Series) -> Dict:
        """生成价格形态特征"""
        features = {}

        # K线形态
        features['body_ratio'] = (close - open_) / (high - low + 1e-10)
        features['upper_shadow'] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / (high - low + 1e-10)
        features['lower_shadow'] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / (high - low + 1e-10)

        # 连续上涨/下跌
        features['consecutive_up'] = (close > close.shift(1)).rolling(5).sum()
        features['consecutive_down'] = (close < close.shift(1)).rolling(5).sum()

        # 跳空
        features['gap_up'] = (low > high.shift(1)).astype(float)
        features['gap_down'] = (high < low.shift(1)).astype(float)

        # 新高/新低
        features['new_high_20'] = (close >= self.indicators.HHV(close, 20)).astype(float)
        features['new_low_20'] = (close <= self.indicators.LLV(close, 20)).astype(float)
        features['new_high_60'] = (close >= self.indicators.HHV(close, 60)).astype(float)
        features['new_low_60'] = (close <= self.indicators.LLV(close, 60)).astype(float)

        return features

    def _generate_derived_features(self, features: Dict, close: pd.Series) -> Dict:
        """生成衍生特征 - 增强预测力"""
        derived = {}

        # 多周期趋势确认
        derived['trend_strength'] = (
            (features.get('ma_bullish_alignment', pd.Series(0, index=close.index)) * 2 +
             features.get('rsi_12', pd.Series(50, index=close.index)) / 100 +
             features.get('pos_20', pd.Series(0.5, index=close.index)))
        )

        # 动量-波动率组合
        derived['mom_vol_ratio'] = features.get('mom_10', pd.Series(0, index=close.index)) / (features.get('std_20', pd.Series(1, index=close.index)) * 100 + 1e-10)

        # 位置-动量组合
        derived['pos_mom_score'] = (
            features.get('pos_20', pd.Series(0.5, index=close.index)) * 0.5 +
            (features.get('rsi_6', pd.Series(50, index=close.index)) / 100) * 0.5
        )

        # ========== 核心特征：趋势评分系统 ==========
        # 使用已有的趋势评分作为最重要的特征
        try:
            from .trend_scoring_system import TrendScoringSystem
            trend_system = TrendScoringSystem(min_amount=0)  # 禁用金额过滤

            trend_scores = []
            for i in range(len(close)):
                if i < 60:
                    trend_scores.append(0)
                else:
                    try:
                        sub_df = pd.DataFrame({
                            'close': close.iloc[:i+1],
                            'high': close.index.to_series().iloc[:i+1],  # 占位
                            'low': close.iloc[:i+1],
                            'open': close.iloc[:i+1],
                            'volume': pd.Series(1e9, index=close.index[:i+1])  # 默认大成交量
                        })
                        result = trend_system.calculate_score(sub_df)
                        trend_scores.append(result.final_score)
                    except:
                        trend_scores.append(0)

            derived['trend_score'] = pd.Series(trend_scores, index=close.index)
            derived['trend_score_ma5'] = derived['trend_score'].rolling(5).mean()
            derived['trend_score_change'] = derived['trend_score'].diff(5)
        except:
            # 如果趋势评分系统不可用，使用简化版
            derived['trend_score'] = features.get('trend_strength', pd.Series(0, index=close.index))
            derived['trend_score_ma5'] = derived['trend_score'].rolling(5).mean()
            derived['trend_score_change'] = derived['trend_score'].diff(5)

        # ========== 新增：更有预测力的特征 ==========

        # 1. 趋势加速度 (二阶导数)
        ma20 = features.get('ma_20', pd.Series(close, index=close.index))
        derived['ma_acceleration'] = ma20.diff(5).diff(5) / (ma20 + 1e-10)

        # 2. 价格动量强度 (标准化)
        mom_5 = features.get('mom_5', pd.Series(0, index=close.index))
        mom_20 = features.get('mom_20', pd.Series(0, index=close.index))
        derived['momentum_strength'] = (mom_5 - mom_20) / (abs(mom_20) + 1e-10)

        # 3. RSI超买超卖反转信号
        rsi_6 = features.get('rsi_6', pd.Series(50, index=close.index))
        rsi_12 = features.get('rsi_12', pd.Series(50, index=close.index))
        derived['rsi_oversold'] = (rsi_6 < 30).astype(float)  # 超卖信号
        derived['rsi_overbought'] = (rsi_6 > 70).astype(float)  # 超买信号
        derived['rsi_cross_up'] = ((rsi_6 > rsi_12) & (rsi_6.shift(1) <= rsi_12.shift(1))).astype(float)

        # 4. KDJ 金叉/死叉
        kdj_k = features.get('kdj_k', pd.Series(50, index=close.index))
        kdj_d = features.get('kdj_d', pd.Series(50, index=close.index))
        kdj_j = features.get('kdj_j', pd.Series(50, index=close.index))
        derived['kdj_oversold'] = (kdj_j < 20).astype(float)
        derived['kdj_cross'] = ((kdj_k > kdj_d) & (kdj_k.shift(1) <= kdj_d.shift(1))).astype(float)

        # 5. MACD 金叉强度
        macd_dif = features.get('macd_dif', pd.Series(0, index=close.index))
        macd_dea = features.get('macd_dea', pd.Series(0, index=close.index))
        derived['macd_golden_cross'] = ((macd_dif > macd_dea) & (macd_dif.shift(1) <= macd_dea.shift(1))).astype(float)
        derived['macd_hist_trend'] = (macd_dif - macd_dea).diff(3)  # MACD柱趋势

        # 6. 布林带突破
        boll_position = features.get('boll_position', pd.Series(0.5, index=close.index))
        derived['boll_breakout_up'] = (boll_position > 0.95).astype(float)
        derived['boll_breakout_down'] = (boll_position < 0.05).astype(float)

        # 7. 位置反转信号
        pos_20 = features.get('pos_20', pd.Series(0.5, index=close.index))
        pos_60 = features.get('pos_60', pd.Series(0.5, index=close.index))
        derived['position_reversal'] = ((pos_20 < 0.2) & (pos_20 > pos_20.shift(5))).astype(float)  # 低位反弹
        derived['position_breakdown'] = ((pos_20 > 0.8) & (pos_20 < pos_20.shift(5))).astype(float)  # 高位回落

        # 8. 量价配合度
        vol_ratio = features.get('vol_ratio', pd.Series(1, index=close.index))
        derived['volume_breakout'] = (vol_ratio > 2.0).astype(float)  # 放量
        derived['volume_dry'] = (vol_ratio < 0.5).astype(float)  # 缩量

        # 9. 综合打分 (关键特征)
        derived['buy_signal_score'] = (
            derived['rsi_oversold'] * 2 +
            derived['kdj_cross'] * 2 +
            derived['macd_golden_cross'] * 2 +
            derived['position_reversal'] * 1.5 +
            (1 - pos_20) * 0.5  # 低位加分
        )

        return derived

    def _build_feature_groups(self):
        """构建特征分组"""
        self.feature_groups = {
            '趋势类': [f for f in self.feature_names if any(x in f for x in ['ma_', 'ema_', 'macd', 'slope', 'alignment'])],
            '动量类': [f for f in self.feature_names if any(x in f for x in ['mom', 'rsi', 'kdj', 'cci', 'willr', 'aroon'])],
            '波动率类': [f for f in self.feature_names if any(x in f for x in ['boll', 'atr', 'std', 'volatility', 'amplitude'])],
            '位置类': [f for f in self.feature_names if any(x in f for x in ['pos_', 'hhvbars', 'llvbars', 'bias', 'price_vs'])],
            '成交量类': [f for f in self.feature_names if any(x in f for x in ['vol', 'obv', 'mfi'])],
            '形态类': [f for f in self.feature_names if any(x in f for x in ['body', 'shadow', 'consecutive', 'gap', 'new_high', 'new_low'])],
            '衍生类': [f for f in self.feature_names if f in ['trend_strength', 'mom_vol_ratio', 'pos_mom_score']]
        }

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """返回特征分组"""
        return self.feature_groups

    def select_features_by_importance(self, feature_df: pd.DataFrame, importance_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        """
        根据重要性选择特征

        Args:
            feature_df: 特征DataFrame
            importance_df: 特征重要性DataFrame，需要有'feature'和'importance'列
            top_n: 选择的特征数量

        Returns:
            选择后的特征DataFrame
        """
        top_features = importance_df.nlargest(top_n, 'importance')['feature'].tolist()
        selected_features = [f for f in top_features if f in feature_df.columns]
        return feature_df[selected_features]


class LabelGenerator:
    """
    标签生成器

    支持多种标签生成方法
    """

    def __init__(
        self,
        method: str = 'simple',
        horizon: int = 10,
        profit_target: float = 0.05,
        stop_loss: float = 0.03,
        time_limit: int = 20
    ):
        """
        Args:
            method: 标签方法 ('simple', 'triple_barrier', 'regression', 'relative')
            horizon: 预测周期
            profit_target: 止盈目标
            stop_loss: 止损比例
            time_limit: 时间限制
        """
        self.method = method
        self.horizon = horizon
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.time_limit = time_limit

    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        生成标签

        Args:
            df: 包含close列的DataFrame

        Returns:
            标签Series
        """
        close = df['close']

        if self.method == 'simple':
            # 简单方法：未来N天收益率 > 0 则为1
            future_ret = close.shift(-self.horizon) / close - 1
            labels = (future_ret > 0).astype(int)

        elif self.method == 'relative':
            # 相对收益法：未来收益 > 历史中位数则为1
            future_ret = close.shift(-self.horizon) / close - 1
            median_ret = future_ret.rolling(60).median()
            labels = (future_ret > median_ret).astype(float)
            # 最后 horizon 个样本设为 NaN
            labels.iloc[-self.horizon:] = np.nan

        elif self.method == 'triple_barrier':
            # 三重屏障法
            labels = self._triple_barrier_label(close.values)

        elif self.method == 'regression':
            # 回归方法：预测实际收益率
            labels = close.shift(-self.horizon) / close - 1

        else:
            raise ValueError(f"未知标签方法: {self.method}")

        return pd.Series(labels, index=df.index)

    def _triple_barrier_label(self, close: np.ndarray) -> np.ndarray:
        """三重屏障法标签 - 改进版"""
        n = len(close)
        labels = np.zeros(n)

        for i in range(n - self.time_limit):
            entry_price = close[i]
            profit_barrier = entry_price * (1 + self.profit_target)
            loss_barrier = entry_price * (1 - self.stop_loss)

            outcome = 0  # 0: 时间退出
            exit_price = close[i + self.time_limit] if i + self.time_limit < n else close[-1]

            for j in range(1, self.time_limit + 1):
                if i + j >= n:
                    break
                if close[i + j] >= profit_barrier:
                    outcome = 1  # 止盈
                    exit_price = close[i + j]
                    break
                elif close[i + j] <= loss_barrier:
                    outcome = -1  # 止损
                    exit_price = close[i + j]
                    break

            # 改进：根据实际收益率判断，而不是只看是否触及止盈
            # 如果实际收益率 > 0，标记为1，否则为0
            actual_return = (exit_price - entry_price) / entry_price
            labels[i] = 1 if actual_return > 0 else 0

        return labels