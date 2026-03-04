"""
动态阈值系统模块

根据市场状态和波动率动态调整买卖阈值：
- 市场状态识别（牛市/熊市/震荡市）
- 自适应阈值调整
- 波动率敏感的阈值优化
- 双重市场状态系统（大盘+个股）
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class MarketRegime(str, Enum):
    """市场状态"""
    BULL = "bull"           # 牛市
    BEAR = "bear"           # 熊市
    SIDEWAY = "sideway"     # 震荡市
    VOLATILE = "volatile"   # 剧烈波动


class VolatilityLevel(str, Enum):
    """波动率水平"""
    LOW = "low"         # 低波动
    NORMAL = "normal"   # 正常波动
    HIGH = "high"       # 高波动
    EXTREME = "extreme" # 极端波动


class CombinedSignal(str, Enum):
    """综合信号"""
    STRONG_BUY = "强买入"      # 大盘牛 + 个股牛
    WATCH = "关注"             # 大盘牛 + 个股震荡
    AVOID = "回避"             # 大盘牛 + 个股熊 或 大盘震荡 + 个股熊
    LIGHT_POSITION = "轻仓"    # 大盘震荡 + 个股牛
    WAIT = "观望"              # 大盘震荡 + 个股震荡
    CASH = "空仓"              # 大盘熊


@dataclass
class DualMarketState:
    """双重市场状态"""
    index_regime: MarketRegime      # 大盘状态（影响阈值）
    stock_regime: MarketRegime      # 个股状态（影响评分）
    combined_signal: CombinedSignal # 综合信号
    confidence: float               # 置信度 (0-1)
    index_code: str                 # 大盘指数代码
    index_name: str                 # 大盘指数名称

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'index_regime': self.index_regime.value,
            'stock_regime': self.stock_regime.value,
            'combined_signal': self.combined_signal.value,
            'confidence': self.confidence,
            'index_code': self.index_code,
            'index_name': self.index_name,
        }


@dataclass
class AdaptiveThresholdConfig:
    """自适应阈值配置"""
    buy_threshold: float
    sell_threshold: float
    strong_buy_threshold: float
    strong_sell_threshold: float
    market_regime: MarketRegime
    volatility_level: VolatilityLevel
    confidence: float
    dual_market_state: Optional[DualMarketState] = None  # 双重市场状态

    def to_dict(self) -> Dict:
        """转换为字典"""
        result = {
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold,
            'strong_buy_threshold': self.strong_buy_threshold,
            'strong_sell_threshold': self.strong_sell_threshold,
            'market_regime': self.market_regime.value,
            'volatility_level': self.volatility_level.value,
            'confidence': self.confidence,
        }
        if self.dual_market_state:
            result['dual_market_state'] = self.dual_market_state.to_dict()
        return result


class AdaptiveThresholdManager:
    """
    自适应阈值管理器

    根据市场状态和波动率动态调整买卖阈值
    """

    # 不同市场状态的默认阈值配置
    REGIME_THRESHOLDS = {
        MarketRegime.BULL: {
            'buy_threshold': 65.0,       # 牛市：提高买入门槛，避免追高
            'sell_threshold': 40.0,      # 牛市：降低卖出门槛，让利润奔跑
            'strong_buy': 80.0,
            'strong_sell': 25.0,
        },
        MarketRegime.BEAR: {
            'buy_threshold': 75.0,       # 熊市：大幅提高买入门槛，防守为主
            'sell_threshold': 55.0,      # 熊市：提高卖出门槛，及时止损
            'strong_buy': 85.0,
            'strong_sell': 40.0,
        },
        MarketRegime.SIDEWAY: {
            'buy_threshold': 50.0,       # 震荡市：低买高卖，均值回归
            'sell_threshold': 50.0,      # 震荡市：对称阈值
            'strong_buy': 70.0,
            'strong_sell': 30.0,
        },
        MarketRegime.VOLATILE: {
            'buy_threshold': 70.0,       # 剧烈波动：谨慎操作
            'sell_threshold': 45.0,      # 剧烈波动：及时止盈
            'strong_buy': 80.0,
            'strong_sell': 35.0,
        },
    }

    # 波动率阈值
    VOLATILITY_THRESHOLDS = {
        'low': 0.015,        # 日波动率 < 1.5%
        'normal_low': 0.02,  # 1.5% - 2%
        'normal_high': 0.03, # 2% - 3%
        'high': 0.05,        # 3% - 5%
        # > 5% 为极端波动
    }

    def __init__(
        self,
        lookback_period: int = 60,
        regime_smoothing: int = 5,
        volatility_adjustment_factor: float = 0.3
    ):
        """
        初始化自适应阈值管理器

        Args:
            lookback_period: 市场状态识别回看周期
            regime_smoothing: 状态平滑周期
            volatility_adjustment_factor: 波动率调整系数
        """
        self.lookback_period = lookback_period
        self.regime_smoothing = regime_smoothing
        self.volatility_adjustment_factor = volatility_adjustment_factor

        # 历史记录
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []
        self.threshold_history: List[Tuple[datetime, AdaptiveThresholdConfig]] = []

    def detect_market_regime(
        self,
        df: pd.DataFrame,
        price_column: str = 'close'
    ) -> MarketRegime:
        """
        识别市场状态

        Args:
            df: 价格数据DataFrame
            price_column: 价格列名

        Returns:
            MarketRegime: 市场状态
        """
        if len(df) < self.lookback_period:
            return MarketRegime.SIDEWAY

        # 取最近数据
        recent_data = df.tail(self.lookback_period)
        prices = recent_data[price_column].values

        # 计算收益率
        returns = np.diff(prices) / prices[:-1]

        # 计算特征
        # 1. 趋势（年化收益率）
        trend = np.mean(returns) * 252

        # 2. 波动率（年化）
        volatility = np.std(returns) * np.sqrt(252)

        # 3. 价格位置（相对区间高低点）
        price_position = (prices[-1] - np.min(prices)) / (np.max(prices) - np.min(prices))

        # 4. 方向一致性
        up_days = np.sum(returns > 0) / len(returns)

        # 状态判定
        # 牛市：趋势向上 + 波动率适中 + 价格位置高
        # 熊市：趋势向下 + 波动率可能高
        # 震荡：趋势接近零 + 波动率低
        # 剧烈波动：波动率极高

        annual_vol_threshold_high = 0.25
        annual_trend_bull = 0.10
        annual_trend_bear = -0.10

        # 首先判断波动率
        if volatility > annual_vol_threshold_high:
            if trend > annual_trend_bull:
                regime = MarketRegime.BULL  # 波动但上涨
            elif trend < annual_trend_bear:
                regime = MarketRegime.BEAR  # 波动且下跌
            else:
                regime = MarketRegime.VOLATILE  # 剧烈震荡
        else:
            if trend > annual_trend_bull:
                regime = MarketRegime.BULL
            elif trend < annual_trend_bear:
                regime = MarketRegime.BEAR
            else:
                regime = MarketRegime.SIDEWAY

        # 记录历史
        self.regime_history.append((datetime.now(), regime))

        return regime

    def detect_volatility_level(
        self,
        df: pd.DataFrame,
        price_column: str = 'close'
    ) -> VolatilityLevel:
        """
        检测波动率水平

        Args:
            df: 价格数据
            price_column: 价格列名

        Returns:
            VolatilityLevel: 波动率水平
        """
        if len(df) < 20:
            return VolatilityLevel.NORMAL

        # 计算日波动率
        returns = df[price_column].pct_change().dropna()
        daily_vol = returns.tail(20).std()

        if daily_vol < self.VOLATILITY_THRESHOLDS['low']:
            return VolatilityLevel.LOW
        elif daily_vol < self.VOLATILITY_THRESHOLDS['normal_high']:
            return VolatilityLevel.NORMAL
        elif daily_vol < self.VOLATILITY_THRESHOLDS['high']:
            return VolatilityLevel.HIGH
        else:
            return VolatilityLevel.EXTREME

    def adjust_thresholds_by_market(
        self,
        regime: MarketRegime,
        base_config: Optional[Dict] = None
    ) -> Dict:
        """
        根据市场状态调整阈值

        Args:
            regime: 市场状态
            base_config: 基础配置（可选）

        Returns:
            Dict: 调整后的阈值配置
        """
        # 获取基础阈值
        if base_config:
            thresholds = base_config.copy()
        else:
            thresholds = {
                'buy_threshold': 70.0,
                'sell_threshold': 50.0,
                'strong_buy': 80.0,
                'strong_sell': 35.0,
            }

        # 获取状态对应阈值
        regime_thresholds = self.REGIME_THRESHOLDS[regime]

        # 融合调整
        # 使用加权平均：基础阈值权重0.3，状态阈值权重0.7
        adjusted = {}
        for key in thresholds:
            base_val = thresholds[key]
            regime_val = regime_thresholds[key]
            adjusted[key] = base_val * 0.3 + regime_val * 0.7

        return adjusted

    def adjust_thresholds_by_volatility(
        self,
        volatility_level: VolatilityLevel,
        current_thresholds: Dict
    ) -> Dict:
        """
        根据波动率调整阈值

        高波动时提高买入门槛，低波动时降低买入门槛

        Args:
            volatility_level: 波动率水平
            current_thresholds: 当前阈值

        Returns:
            Dict: 调整后的阈值
        """
        # 波动率调整系数
        vol_adjustments = {
            VolatilityLevel.LOW: {'buy': -5.0, 'sell': 5.0},      # 低波动：降低买入门槛
            VolatilityLevel.NORMAL: {'buy': 0.0, 'sell': 0.0},    # 正常：不调整
            VolatilityLevel.HIGH: {'buy': 5.0, 'sell': -5.0},     # 高波动：提高买入门槛
            VolatilityLevel.EXTREME: {'buy': 10.0, 'sell': -10.0}, # 极端：大幅调整
        }

        adjustment = vol_adjustments[volatility_level]

        return {
            'buy_threshold': current_thresholds['buy_threshold'] + adjustment['buy'],
            'sell_threshold': current_thresholds['sell_threshold'] + adjustment['sell'],
            'strong_buy': current_thresholds.get('strong_buy', 80.0) + adjustment['buy'] * 0.5,
            'strong_sell': current_thresholds.get('strong_sell', 35.0) + adjustment['sell'] * 0.5,
        }

    def get_adaptive_thresholds(
        self,
        df: pd.DataFrame,
        base_config: Optional[Dict] = None
    ) -> AdaptiveThresholdConfig:
        """
        获取自适应阈值

        综合市场状态和波动率调整阈值

        Args:
            df: 价格数据
            base_config: 基础配置

        Returns:
            AdaptiveThresholdConfig: 自适应阈值配置
        """
        # 检测市场状态
        regime = self.detect_market_regime(df)

        # 检测波动率水平
        volatility_level = self.detect_volatility_level(df)

        # 根据市场状态调整
        thresholds = self.adjust_thresholds_by_market(regime, base_config)

        # 根据波动率调整
        thresholds = self.adjust_thresholds_by_volatility(volatility_level, thresholds)

        # 限制范围
        thresholds['buy_threshold'] = max(50.0, min(85.0, thresholds['buy_threshold']))
        thresholds['sell_threshold'] = max(30.0, min(60.0, thresholds['sell_threshold']))

        # 计算置信度
        confidence = self._calculate_confidence(df, regime, volatility_level)

        config = AdaptiveThresholdConfig(
            buy_threshold=thresholds['buy_threshold'],
            sell_threshold=thresholds['sell_threshold'],
            strong_buy_threshold=thresholds['strong_buy'],
            strong_sell_threshold=thresholds['strong_sell'],
            market_regime=regime,
            volatility_level=volatility_level,
            confidence=confidence
        )

        # 记录历史
        self.threshold_history.append((datetime.now(), config))

        return config

    def _calculate_confidence(
        self,
        df: pd.DataFrame,
        regime: MarketRegime,
        volatility_level: VolatilityLevel
    ) -> float:
        """
        计算阈值置信度
        """
        confidence = 0.5

        # 数据量充足
        if len(df) >= 120:
            confidence += 0.2
        elif len(df) >= 60:
            confidence += 0.1

        # 状态稳定性
        if len(self.regime_history) >= 5:
            recent_regimes = [r for _, r in self.regime_history[-5:]]
            if all(r == regime for r in recent_regimes):
                confidence += 0.2

        # 波动率正常
        if volatility_level == VolatilityLevel.NORMAL:
            confidence += 0.1

        return min(1.0, confidence)

    def get_threshold_statistics(self) -> Dict:
        """
        获取阈值统计信息
        """
        if not self.threshold_history:
            return {}

        recent = self.threshold_history[-20:]
        buy_thresholds = [t.buy_threshold for _, t in recent]
        sell_thresholds = [t.sell_threshold for _, t in recent]

        regime_counts = {}
        for _, config in self.threshold_history:
            r = config.market_regime.value
            regime_counts[r] = regime_counts.get(r, 0) + 1

        return {
            'avg_buy_threshold': np.mean(buy_thresholds),
            'avg_sell_threshold': np.mean(sell_thresholds),
            'current_regime': self.threshold_history[-1][1].market_regime.value,
            'regime_distribution': regime_counts,
            'total_adjustments': len(self.threshold_history)
        }


class ScoreNormalizer:
    """
    评分归一化器

    解决评分分布偏低问题
    """

    def __init__(
        self,
        target_mean: float = 50.0,
        target_std: float = 15.0,
        min_score: float = 0.0,
        max_score: float = 100.0
    ):
        """
        初始化评分归一化器

        Args:
            target_mean: 目标均值
            target_std: 目标标准差
            min_score: 最小评分
            max_score: 最大评分
        """
        self.target_mean = target_mean
        self.target_std = target_std
        self.min_score = min_score
        self.max_score = max_score

    def normalize_cross_sectional(
        self,
        scores: pd.Series
    ) -> pd.Series:
        """
        横截面归一化（市场中性）

        将评分归一化到目标分布，保持相对排序

        Args:
            scores: 评分序列

        Returns:
            pd.Series: 归一化后的评分
        """
        if len(scores) < 2:
            return scores

        # 计算Z-score
        mean = scores.mean()
        std = scores.std()

        if std < 1e-6:
            # 方差太小，返回原值
            return scores

        z_scores = (scores - mean) / std

        # 映射到目标分布
        normalized = z_scores * self.target_std + self.target_mean

        # 限制范围
        return normalized.clip(self.min_score, self.max_score)

    def normalize_time_series(
        self,
        scores: pd.Series,
        window: int = 60
    ) -> pd.Series:
        """
        时间序列归一化

        使用滚动窗口进行归一化

        Args:
            scores: 评分序列（带时间索引）
            window: 滚动窗口大小

        Returns:
            pd.Series: 归一化后的评分
        """
        if len(scores) < window:
            return self.normalize_cross_sectional(scores)

        # 计算滚动统计
        rolling_mean = scores.rolling(window=window, min_periods=int(window/2)).mean()
        rolling_std = scores.rolling(window=window, min_periods=int(window/2)).std()

        # 计算Z-score
        z_scores = (scores - rolling_mean) / (rolling_std + 1e-6)

        # 映射到目标分布
        normalized = z_scores * self.target_std + self.target_mean

        # 填充初始NaN
        normalized = normalized.fillna(self.target_mean)

        # 限制范围
        return normalized.clip(self.min_score, self.max_score)

    def normalize_rank_based(
        self,
        scores: pd.Series
    ) -> pd.Series:
        """
        秩归一化

        将评分转换为排名百分位数

        Args:
            scores: 评分序列

        Returns:
            pd.Series: 归一化后的评分（百分位）
        """
        if len(scores) < 2:
            return scores

        # 计算排名百分位
        ranks = scores.rank(method='average', pct=True)

        # 映射到目标范围
        normalized = ranks * 100

        return normalized

    def normalize_mixed(
        self,
        scores: pd.Series,
        cross_sectional_weight: float = 0.6,
        time_series_weight: float = 0.4,
        window: int = 60
    ) -> pd.Series:
        """
        混合归一化

        结合横截面和时间序列归一化

        Args:
            scores: 评分序列
            cross_sectional_weight: 横截面权重
            time_series_weight: 时间序列权重
            window: 时间序列窗口

        Returns:
            pd.Series: 归一化后的评分
        """
        # 归一化权重
        total_weight = cross_sectional_weight + time_series_weight
        cs_w = cross_sectional_weight / total_weight
        ts_w = time_series_weight / total_weight

        # 分别归一化
        cs_normalized = self.normalize_cross_sectional(scores)
        ts_normalized = self.normalize_time_series(scores, window)

        # 加权组合
        mixed = cs_normalized * cs_w + ts_normalized * ts_w

        return mixed.clip(self.min_score, self.max_score)


def get_adaptive_thresholds(
    df: pd.DataFrame,
    base_buy: float = 70.0,
    base_sell: float = 50.0
) -> Dict:
    """
    便捷函数：获取自适应阈值

    Args:
        df: 价格数据
        base_buy: 基础买入阈值
        base_sell: 基础卖出阈值

    Returns:
        Dict: 阈值配置
    """
    manager = AdaptiveThresholdManager()
    config = manager.get_adaptive_thresholds(
        df,
        base_config={
            'buy_threshold': base_buy,
            'sell_threshold': base_sell,
            'strong_buy': 80.0,
            'strong_sell': 35.0,
        }
    )
    return {
        'buy_threshold': config.buy_threshold,
        'sell_threshold': config.sell_threshold,
        'strong_buy': config.strong_buy_threshold,
        'strong_sell': config.strong_sell_threshold,
        'market_regime': config.market_regime.value,
        'volatility_level': config.volatility_level.value,
        'confidence': config.confidence
    }


class IndexMarketDetector:
    """
    大盘市场状态检测器

    基于大盘指数（上证指数、深证成指、沪深300）判断市场整体状态。
    这是真正的"市场状态"，用于影响整体阈值调整。
    """

    # 大盘指数代码
    INDEX_CODES = {
        'sh': '000001.SH',      # 上证指数
        'sz': '399001.SZ',      # 深证成指
        'hs300': '000300.SH',   # 沪深300
    }

    INDEX_NAMES = {
        '000001.SH': '上证指数',
        '399001.SZ': '深证成指',
        '000300.SH': '沪深300',
    }

    # 缓存大盘数据（避免重复获取）
    _index_cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}
    _cache_ttl = 3600  # 缓存有效期（秒）

    def __init__(
        self,
        default_index: str = 'hs300',
        lookback_period: int = 60
    ):
        """
        初始化大盘市场状态检测器

        Args:
            default_index: 默认使用的指数 ('sh', 'sz', 'hs300')
            lookback_period: 状态识别回看周期
        """
        self.default_index = default_index
        self.lookback_period = lookback_period
        self._data_fetcher = None

    def _get_data_fetcher(self):
        """懒加载数据获取器"""
        if self._data_fetcher is None:
            from quanttool.infrastructure.data_providers.data_fetcher import create_data_fetcher_with_credentials
            self._data_fetcher = create_data_fetcher_with_credentials()
            self._data_fetcher.initialize()
        return self._data_fetcher

    def get_index_data(self, index_code: str, days: int = 120) -> pd.DataFrame:
        """
        获取大盘指数数据

        Args:
            index_code: 指数代码
            days: 获取天数

        Returns:
            pd.DataFrame: 指数数据
        """
        # 检查缓存
        now = datetime.now()
        if index_code in self._index_cache:
            cache_time, cached_df = self._index_cache[index_code]
            if (now - cache_time).total_seconds() < self._cache_ttl:
                return cached_df

        # 获取数据
        try:
            fetcher = self._get_data_fetcher()
            from datetime import timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            result = fetcher.get_bars([index_code], start_date, end_date)
            df = result.get(index_code, pd.DataFrame())

            # 更新缓存
            if not df.empty:
                self._index_cache[index_code] = (now, df)

            return df
        except Exception as e:
            print(f"获取大盘指数数据失败: {e}")
            return pd.DataFrame()

    def detect_index_regime(
        self,
        index_code: Optional[str] = None
    ) -> Tuple[MarketRegime, float]:
        """
        检测大盘市场状态

        Args:
            index_code: 指数代码（None则使用默认指数）

        Returns:
            Tuple[MarketRegime, float]: (市场状态, 置信度)
        """
        if index_code is None:
            index_code = self.INDEX_CODES.get(self.default_index, '000300.SH')

        # 获取指数数据
        df = self.get_index_data(index_code, days=self.lookback_period + 20)

        if df.empty or len(df) < self.lookback_period:
            # 数据不足，返回默认状态
            return MarketRegime.SIDEWAY, 0.3

        # 计算市场状态特征
        recent_data = df.tail(self.lookback_period)
        prices = recent_data['close'].values

        # 计算收益率
        returns = np.diff(prices) / prices[:-1]

        # 1. 趋势（年化收益率）
        trend = np.mean(returns) * 252

        # 2. 波动率（年化）
        volatility = np.std(returns) * np.sqrt(252)

        # 3. 价格位置（相对区间高低点）
        price_position = (prices[-1] - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-10)

        # 4. 方向一致性
        up_days = np.sum(returns > 0) / len(returns)

        # 状态判定阈值
        annual_vol_threshold_high = 0.25
        annual_trend_bull = 0.10
        annual_trend_bear = -0.10

        # 计算置信度
        confidence = 0.5
        if len(df) >= 120:
            confidence += 0.2
        if price_position > 0.8 or price_position < 0.2:
            confidence += 0.1  # 极端位置更可信
        if volatility < 0.20:
            confidence += 0.1  # 低波动更稳定

        # 状态判定
        if volatility > annual_vol_threshold_high:
            if trend > annual_trend_bull:
                regime = MarketRegime.BULL  # 波动但上涨
            elif trend < annual_trend_bear:
                regime = MarketRegime.BEAR  # 波动且下跌
            else:
                regime = MarketRegime.VOLATILE  # 剧烈震荡
        else:
            if trend > annual_trend_bull:
                regime = MarketRegime.BULL
            elif trend < annual_trend_bear:
                regime = MarketRegime.BEAR
            else:
                regime = MarketRegime.SIDEWAY

        return regime, min(1.0, confidence)

    def get_dual_market_state(
        self,
        stock_df: pd.DataFrame,
        index_code: Optional[str] = None
    ) -> DualMarketState:
        """
        获取双重市场状态（大盘 + 个股）

        Args:
            stock_df: 个股数据
            index_code: 大盘指数代码

        Returns:
            DualMarketState: 双重市场状态
        """
        # 获取大盘状态
        if index_code is None:
            index_code = self.INDEX_CODES.get(self.default_index, '000300.SH')

        index_regime, index_confidence = self.detect_index_regime(index_code)

        # 获取个股状态
        stock_manager = AdaptiveThresholdManager(lookback_period=self.lookback_period)
        stock_regime = stock_manager.detect_market_regime(stock_df)

        # 计算综合信号
        combined_signal = self._determine_combined_signal(index_regime, stock_regime)

        # 计算综合置信度
        confidence = (index_confidence + 0.5) / 2  # 个股置信度默认0.5

        return DualMarketState(
            index_regime=index_regime,
            stock_regime=stock_regime,
            combined_signal=combined_signal,
            confidence=confidence,
            index_code=index_code,
            index_name=self.INDEX_NAMES.get(index_code, index_code),
        )

    def _determine_combined_signal(
        self,
        index_regime: MarketRegime,
        stock_regime: MarketRegime
    ) -> CombinedSignal:
        """
        根据大盘和个股状态确定综合信号

        综合信号逻辑：
        | 大盘状态 | 个股状态 | 综合信号 | 说明 |
        |---------|---------|---------|------|
        | BULL    | BULL    | 强买入  | 顺势而为 |
        | BULL    | SIDEWAY | 关注    | 等待个股启动 |
        | BULL    | BEAR    | 回避    | 逆势个股风险大 |
        | SIDEWAY | BULL    | 轻仓    | 个股独立行情 |
        | SIDEWAY | SIDEWAY | 观望    | 无明确方向 |
        | SIDEWAY | BEAR    | 回避    | 下跌趋势 |
        | BEAR    | *       | 空仓    | 大盘下跌，整体回避 |
        | VOLATILE| BULL    | 轻仓    | 波动中的机会 |
        | VOLATILE| *       | 观望    | 波动剧烈，谨慎为上 |
        """
        # 大盘熊市，无论个股如何都空仓
        if index_regime == MarketRegime.BEAR:
            return CombinedSignal.CASH

        # 大盘牛市
        if index_regime == MarketRegime.BULL:
            if stock_regime == MarketRegime.BULL:
                return CombinedSignal.STRONG_BUY
            elif stock_regime == MarketRegime.SIDEWAY:
                return CombinedSignal.WATCH
            else:  # BEAR or VOLATILE
                return CombinedSignal.AVOID

        # 大盘震荡市
        if index_regime == MarketRegime.SIDEWAY:
            if stock_regime == MarketRegime.BULL:
                return CombinedSignal.LIGHT_POSITION
            elif stock_regime == MarketRegime.SIDEWAY:
                return CombinedSignal.WAIT
            else:  # BEAR or VOLATILE
                return CombinedSignal.AVOID

        # 大盘剧烈波动
        if index_regime == MarketRegime.VOLATILE:
            if stock_regime == MarketRegime.BULL:
                return CombinedSignal.LIGHT_POSITION
            else:
                return CombinedSignal.WAIT

        # 默认观望
        return CombinedSignal.WAIT


def get_dual_market_state(
    stock_df: pd.DataFrame,
    index_code: Optional[str] = None,
    default_index: str = 'hs300'
) -> DualMarketState:
    """
    便捷函数：获取双重市场状态

    Args:
        stock_df: 个股数据
        index_code: 大盘指数代码（None则使用默认指数）
        default_index: 默认指数类型 ('sh', 'sz', 'hs300')

    Returns:
        DualMarketState: 双重市场状态
    """
    detector = IndexMarketDetector(default_index=default_index)
    return detector.get_dual_market_state(stock_df, index_code)