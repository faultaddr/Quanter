"""
筹码分布模块 (Chip Distribution / CYQ)

实现与东方财富一致的筹码分布算法，用于分析：
- 筹码集中度
- 套牢盘分布
- 支撑/阻力位识别
- 主力成本估算

核心原理：
通过计算一定时间范围内股票的最高价、最低价、成交数，
输出对应价格成交数占整个流通盘比值的分布图形。

参考：InStock 项目实现
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ChipDistributionResult:
    """筹码分布计算结果"""
    price_levels: np.ndarray      # 价格区间
    chip_distribution: np.ndarray # 筹码分布（占比）
    concentration_ratio: float    # 筹码集中度 (0-100)
    avg_cost: float               # 平均持仓成本
    profit_ratio: float           # 当前价位获利盘比例 (0-100)
    upper_pressure: float         # 上方套牢盘压力 (0-100)
    lower_support: float          # 下方支撑强度 (0-100)
    support_levels: List[float]   # 支撑位列表
    resistance_levels: List[float] # 阻力位列表
    peak_prices: List[float]      # 筹码峰价格
    score: float                  # 筹码评分 (0-100)


class ChipDistributionCalculator:
    """
    筹码分布计算器

    计算原理：
    1. 将价格区间划分为N个等分
    2. 对于每个交易日，将成交量均匀分布在当日的价格区间内
    3. 累积所有交易日的成交量分布
    4. 归一化得到筹码分布

    与东方财富一致的计算方法：
    - 默认计算210个交易日
    - 考虑当日最高价、最低价、成交量
    - 成交量均匀分布在当日价格区间
    """

    # 默认参数
    DEFAULT_LOOKBACK = 210  # 默认回看天数
    DEFAULT_PRICE_BINS = 100  # 价格区间数量

    def __init__(
        self,
        lookback_days: int = 210,
        price_bins: int = 100,
        decay_factor: float = 1.0
    ):
        """
        初始化筹码分布计算器

        Args:
            lookback_days: 回看天数，默认210日
            price_bins: 价格区间数量，默认100
            decay_factor: 衰减因子，用于模拟筹码换手
        """
        self.lookback_days = lookback_days
        self.price_bins = price_bins
        self.decay_factor = decay_factor

    def calculate(
        self,
        df: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> ChipDistributionResult:
        """
        计算筹码分布

        Args:
            df: 股票数据DataFrame，需包含 open, high, low, close, volume 列
            current_price: 当前价格，默认使用最新收盘价

        Returns:
            ChipDistributionResult: 筹码分布结果
        """
        if df is None or len(df) < 10:
            return self._empty_result()

        # 获取最近N天的数据
        lookback = min(self.lookback_days, len(df))
        recent_df = df.tail(lookback).copy()

        # 提取数据
        high = recent_df['high'].values
        low = recent_df['low'].values
        close = recent_df['close'].values
        volume = recent_df['volume'].values

        # 确保数据有效
        if len(high) < 10:
            return self._empty_result()

        # 当前价格
        if current_price is None:
            current_price = close[-1]

        # 计算价格范围
        price_min = np.min(low)
        price_max = np.max(high)

        if price_max <= price_min:
            price_max = price_min * 1.01

        # 创建价格区间
        price_step = (price_max - price_min) / self.price_bins
        price_levels = np.linspace(price_min, price_max, self.price_bins + 1)

        # 初始化筹码分布
        chip_distribution = np.zeros(self.price_bins + 1)

        # 计算每个交易日的筹码分布
        for i in range(len(high)):
            h = high[i]
            l = low[i]
            v = volume[i]

            # 跳过无效数据
            if h <= l or v <= 0:
                continue

            # 计算当日价格区间对应的筹码位置
            low_idx = int((l - price_min) / price_step)
            high_idx = int((h - price_min) / price_step)

            # 边界处理
            low_idx = max(0, min(low_idx, self.price_bins))
            high_idx = max(0, min(high_idx, self.price_bins))

            # 将成交量均匀分布在当日价格区间
            if high_idx > low_idx:
                chip_per_bin = v / (high_idx - low_idx + 1)
                chip_distribution[low_idx:high_idx + 1] += chip_per_bin

        # 应用衰减因子（模拟筹码换手）
        if self.decay_factor < 1.0:
            for i in range(len(chip_distribution)):
                # 越老的筹码衰减越多
                chip_distribution[i] *= (self.decay_factor ** (len(high) - i))

        # 归一化
        total_chip = np.sum(chip_distribution)
        if total_chip > 0:
            chip_distribution = chip_distribution / total_chip * 100

        # 计算统计数据
        concentration_ratio = self._calculate_concentration(chip_distribution)
        avg_cost = self._calculate_avg_cost(price_levels, chip_distribution)
        profit_ratio = self._calculate_profit_ratio(
            price_levels, chip_distribution, current_price
        )
        upper_pressure = self._calculate_upper_pressure(
            price_levels, chip_distribution, current_price
        )
        lower_support = self._calculate_lower_support(
            price_levels, chip_distribution, current_price
        )
        support_levels, resistance_levels = self._find_support_resistance(
            price_levels, chip_distribution, current_price
        )
        peak_prices = self._find_chip_peaks(price_levels, chip_distribution)
        score = self._calculate_chip_score(
            concentration_ratio, profit_ratio, upper_pressure, lower_support
        )

        return ChipDistributionResult(
            price_levels=price_levels,
            chip_distribution=chip_distribution,
            concentration_ratio=concentration_ratio,
            avg_cost=avg_cost,
            profit_ratio=profit_ratio,
            upper_pressure=upper_pressure,
            lower_support=lower_support,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            peak_prices=peak_prices,
            score=score
        )

    def _calculate_concentration(self, distribution: np.ndarray) -> float:
        """
        计算筹码集中度

        集中度 = 主要筹码区间占比
        高集中度意味着主力控盘强
        """
        # 找到筹码最集中的区间
        sorted_dist = np.sort(distribution)[::-1]
        # 前30%价格区间的筹码占比
        top_bins = int(len(distribution) * 0.3)
        concentration = np.sum(sorted_dist[:top_bins])

        return min(100, concentration)

    def _calculate_avg_cost(
        self,
        price_levels: np.ndarray,
        distribution: np.ndarray
    ) -> float:
        """
        计算平均持仓成本

        加权平均价格
        """
        if np.sum(distribution) == 0:
            return price_levels[len(price_levels) // 2]

        # 使用价格区间的中点
        mid_prices = (price_levels[:-1] + price_levels[1:]) / 2
        # 扩展distribution以匹配mid_prices长度
        if len(distribution) > len(mid_prices):
            mid_prices = np.append(mid_prices, mid_prices[-1])

        avg_cost = np.sum(mid_prices * distribution[:len(mid_prices)]) / np.sum(distribution)
        return avg_cost

    def _calculate_profit_ratio(
        self,
        price_levels: np.ndarray,
        distribution: np.ndarray,
        current_price: float
    ) -> float:
        """
        计算当前价位的获利盘比例

        低于当前价的筹码为获利盘
        """
        if current_price <= price_levels[0]:
            return 0.0

        # 找到当前价格对应的位置
        profit_mask = price_levels < current_price
        profit_chip = np.sum(distribution[profit_mask])
        total_chip = np.sum(distribution)

        if total_chip == 0:
            return 50.0

        return profit_chip / total_chip * 100

    def _calculate_upper_pressure(
        self,
        price_levels: np.ndarray,
        distribution: np.ndarray,
        current_price: float
    ) -> float:
        """
        计算上方套牢盘压力

        高于当前价的筹码为套牢盘
        """
        if current_price >= price_levels[-1]:
            return 0.0

        # 找到当前价格对应的位置
        upper_mask = price_levels > current_price
        upper_chip = np.sum(distribution[upper_mask])
        total_chip = np.sum(distribution)

        if total_chip == 0:
            return 50.0

        return upper_chip / total_chip * 100

    def _calculate_lower_support(
        self,
        price_levels: np.ndarray,
        distribution: np.ndarray,
        current_price: float
    ) -> float:
        """
        计算下方支撑强度

        基于下方筹码密度和分布
        """
        if current_price <= price_levels[0]:
            return 0.0

        # 找到当前价格下方最近的筹码峰
        lower_mask = price_levels < current_price
        lower_dist = distribution[lower_mask]

        if len(lower_dist) == 0 or np.sum(lower_dist) == 0:
            return 0.0

        # 支撑强度 = 下方筹码集中度
        max_support = np.max(lower_dist)
        avg_support = np.mean(lower_dist[lower_dist > 0]) if np.any(lower_dist > 0) else 0

        # 综合评估
        support_score = (max_support * 0.6 + avg_support * 0.4) * 5
        return min(100, support_score)

    def _find_support_resistance(
        self,
        price_levels: np.ndarray,
        distribution: np.ndarray,
        current_price: float
    ) -> Tuple[List[float], List[float]]:
        """
        识别支撑位和阻力位

        支撑位：下方筹码峰
        阻力位：上方筹码峰
        """
        support_levels = []
        resistance_levels = []

        # 找筹码峰
        peaks = self._find_peaks(distribution)

        mid_prices = (price_levels[:-1] + price_levels[1:]) / 2
        if len(distribution) > len(mid_prices):
            mid_prices = np.append(mid_prices, mid_prices[-1])

        for peak_idx in peaks:
            if peak_idx < len(mid_prices):
                peak_price = mid_prices[peak_idx]
                if peak_price < current_price:
                    support_levels.append(peak_price)
                else:
                    resistance_levels.append(peak_price)

        # 按距离当前价格排序
        support_levels = sorted(support_levels, reverse=True)[:3]
        resistance_levels = sorted(resistance_levels)[:3]

        return support_levels, resistance_levels

    def _find_peaks(self, data: np.ndarray, min_prominence: float = 0.5) -> List[int]:
        """
        寻找数据中的峰值

        Args:
            data: 输入数据
            min_prominence: 最小峰显著度

        Returns:
            峰值位置列表
        """
        peaks = []

        for i in range(1, len(data) - 1):
            if data[i] > data[i - 1] and data[i] > data[i + 1]:
                # 检查峰的显著性
                left_min = np.min(data[max(0, i - 5):i])
                right_min = np.min(data[i + 1:min(len(data), i + 6)])
                prominence = data[i] - max(left_min, right_min)

                if prominence > min_prominence:
                    peaks.append(i)

        return peaks

    def _find_chip_peaks(
        self,
        price_levels: np.ndarray,
        distribution: np.ndarray
    ) -> List[float]:
        """
        找到所有筹码峰的价格

        Returns:
            筹码峰价格列表
        """
        peaks = self._find_peaks(distribution)
        mid_prices = (price_levels[:-1] + price_levels[1:]) / 2
        if len(distribution) > len(mid_prices):
            mid_prices = np.append(mid_prices, mid_prices[-1])

        peak_prices = []
        for idx in peaks:
            if idx < len(mid_prices):
                peak_prices.append(mid_prices[idx])

        return peak_prices

    def _calculate_chip_score(
        self,
        concentration: float,
        profit_ratio: float,
        upper_pressure: float,
        lower_support: float
    ) -> float:
        """
        计算筹码评分

        评分逻辑：
        - 高集中度 = 高分（主力控盘）
        - 适度获利盘 = 高分（不超买）
        - 低上方压力 = 高分
        - 高下方支撑 = 高分

        Returns:
            评分 0-100
        """
        score = 0.0

        # 集中度评分 (权重30%)
        # 适度集中最好（50-80%）
        if 50 <= concentration <= 80:
            score += 30
        elif concentration > 80:
            score += 25  # 过于集中可能有风险
        else:
            score += concentration * 0.4

        # 获利盘评分 (权重25%)
        # 适度获利盘（30-70%）最好
        if 30 <= profit_ratio <= 70:
            score += 25
        elif profit_ratio < 30:
            score += 20  # 低获利盘，有上涨空间
        else:
            score += max(0, 25 - (profit_ratio - 70) * 0.5)

        # 上方压力评分 (权重20%)
        # 压力越小越好
        score += max(0, 20 - upper_pressure * 0.2)

        # 下方支撑评分 (权重25%)
        # 支撑越强越好
        score += min(25, lower_support * 0.25)

        return min(100, max(0, score))

    def _empty_result(self) -> ChipDistributionResult:
        """返回空结果"""
        return ChipDistributionResult(
            price_levels=np.array([0.0]),
            chip_distribution=np.array([0.0]),
            concentration_ratio=0.0,
            avg_cost=0.0,
            profit_ratio=0.0,
            upper_pressure=0.0,
            lower_support=0.0,
            support_levels=[],
            resistance_levels=[],
            peak_prices=[],
            score=0.0
        )

    def get_chip_distribution_chart(
        self,
        result: ChipDistributionResult,
        width: int = 40,
        height: int = 15
    ) -> str:
        """
        生成筹码分布的ASCII图表

        Args:
            result: 筹码分布结果
            width: 图表宽度
            height: 图表高度

        Returns:
            ASCII图表字符串
        """
        if result.price_levels is None or len(result.price_levels) == 0:
            return "无数据"

        # 创建画布
        canvas = [[' ' for _ in range(width)] for _ in range(height)]

        # 归一化分布数据
        max_dist = np.max(result.chip_distribution)
        if max_dist == 0:
            return "无数据"

        normalized = result.chip_distribution / max_dist

        # 计算价格范围
        price_min = result.price_levels[0]
        price_max = result.price_levels[-1]
        price_range = price_max - price_min if price_max > price_min else 1

        # 绘制筹码分布（从左到右）
        for i, dist in enumerate(normalized):
            if i >= width:
                break
            bar_height = int(dist * height)
            for h in range(height - bar_height, height):
                if h >= 0:
                    canvas[h][i] = '█'

        # 添加价格标签
        lines = []
        lines.append(f"筹码分布图 (集中度: {result.concentration_ratio:.1f}%)")
        lines.append(f"最高: ¥{price_max:.2f}")
        lines.append("")

        for row in canvas:
            lines.append(''.join(row))

        lines.append("")
        lines.append(f"最低: ¥{price_min:.2f}")
        lines.append(f"平均成本: ¥{result.avg_cost:.2f}")
        lines.append(f"获利盘: {result.profit_ratio:.1f}%")
        lines.append(f"筹码评分: {result.score:.1f}")

        return '\n'.join(lines)


def calculate_chip_distribution(
    df: pd.DataFrame,
    lookback_days: int = 210,
    current_price: Optional[float] = None
) -> Dict:
    """
    便捷函数：计算筹码分布

    Args:
        df: 股票数据DataFrame
        lookback_days: 回看天数
        current_price: 当前价格

    Returns:
        Dict: 筹码分布结果字典
    """
    calculator = ChipDistributionCalculator(lookback_days=lookback_days)
    result = calculator.calculate(df, current_price)

    return {
        'concentration_ratio': result.concentration_ratio,
        'avg_cost': result.avg_cost,
        'profit_ratio': result.profit_ratio,
        'upper_pressure': result.upper_pressure,
        'lower_support': result.lower_support,
        'support_levels': result.support_levels,
        'resistance_levels': result.resistance_levels,
        'peak_prices': result.peak_prices,
        'score': result.score
    }


def get_chip_assessment(result: ChipDistributionResult) -> str:
    """
    获取筹码分布的定性评估

    Args:
        result: 筹码分布结果

    Returns:
        评估描述字符串
    """
    assessments = []

    # 集中度评估
    if result.concentration_ratio >= 70:
        assessments.append("筹码高度集中，主力控盘明显")
    elif result.concentration_ratio >= 50:
        assessments.append("筹码较为集中，关注主力动向")
    else:
        assessments.append("筹码分散，多空分歧较大")

    # 获利盘评估（改进：更中性的表述）
    if result.profit_ratio >= 80:
        assessments.append("获利盘较多，注意回调风险")
    elif result.profit_ratio >= 50:
        # 改进：移除"上涨动力充足"，改为更中性的表述
        assessments.append("获利盘适中，抛压不算沉重")
    elif result.profit_ratio <= 20:
        assessments.append("获利盘较少，有修复空间")
    else:
        assessments.append("获利盘分布均衡")

    # 套牢盘评估
    if result.upper_pressure >= 50:
        assessments.append(f"上方套牢盘压力大({result.upper_pressure:.0f}%)")
    elif result.upper_pressure >= 30:
        assessments.append(f"上方有一定阻力({result.upper_pressure:.0f}%)")
    else:
        assessments.append("上方阻力较小")

    # 支撑评估（改进：更准确的阈值）
    if result.lower_support >= 50:
        assessments.append("下方支撑强劲")
    elif result.lower_support >= 25:
        assessments.append("下方有支撑")
    elif result.lower_support >= 10:
        assessments.append("下方支撑偏弱")
    else:
        assessments.append("下方支撑较弱，注意风险")

    # 综合评分
    if result.score >= 70:
        assessments.append("【筹码形态良好】")
    elif result.score >= 50:
        assessments.append("【筹码形态一般】")
    else:
        assessments.append("【筹码形态较差】")

    return "；".join(assessments)
