"""Base classes for the scoring system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import pandas as pd


@dataclass
class ScoreResult:
    """评分结果基类

    所有评分策略返回统一的结果格式。
    """
    # 最终评分 (0-100)
    final_score: float

    # 是否通过硬过滤
    passed_filter: bool = True

    # 过滤/排除原因
    filter_reason: str = ""

    # 各维度得分明细
    details: Dict[str, Any] = field(default_factory=dict)

    # 策略名称
    strategy_name: str = ""

    # 时机系数 (可选)
    timing_coefficient: Optional[float] = None

    # 交易参数 (可选)
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'final_score': self.final_score,
            'passed_filter': self.passed_filter,
            'filter_reason': self.filter_reason,
            'details': self.details,
            'strategy_name': self.strategy_name,
            'timing_coefficient': self.timing_coefficient,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
        }

    @property
    def score_level(self) -> str:
        """评分等级"""
        if not self.passed_filter:
            return "filtered"
        if self.final_score >= 80:
            return "excellent"
        if self.final_score >= 60:
            return "good"
        if self.final_score >= 40:
            return "average"
        return "weak"


class ScoringStrategy(ABC):
    """评分策略基类

    所有评分策略必须实现此接口。支持策略模式，便于扩展新的评分方法。

    Usage:
        class MyStrategy(ScoringStrategy):
            @property
            def name(self) -> str:
                return "my_strategy"

            def calculate_score(self, df: pd.DataFrame, **kwargs) -> ScoreResult:
                # 实现评分逻辑
                return ScoreResult(final_score=75.0, strategy_name=self.name)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称（唯一标识）"""
        pass

    @property
    def description(self) -> str:
        """策略描述"""
        return ""

    @property
    def version(self) -> str:
        """策略版本"""
        return "1.0.0"

    @abstractmethod
    def calculate_score(self, df: pd.DataFrame, **kwargs) -> ScoreResult:
        """计算评分

        Args:
            df: K线数据，至少包含 open, high, low, close, volume 列
            **kwargs: 策略特定参数

        Returns:
            ScoreResult: 评分结果
        """
        pass

    def validate_data(self, df: pd.DataFrame) -> bool:
        """验证数据是否满足最低要求

        Args:
            df: K线数据

        Returns:
            bool: 数据是否有效
        """
        if df is None or len(df) == 0:
            return False

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                return False

        return True

    def get_required_data_length(self) -> int:
        """获取所需的最小数据长度

        Returns:
            int: 最小K线数量
        """
        return 60  # 默认60根K线

    def get_default_params(self) -> Dict[str, Any]:
        """获取默认参数

        Returns:
            Dict: 默认参数配置
        """
        return {}
