"""Unified scoring system that combines multiple scoring strategies.

This module provides a facade for using multiple scoring strategies
and getting comprehensive scoring results.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd

from .base import ScoreResult, ScoringStrategy


class UnifiedScoringSystem:
    """统一评分系统

    组合多个评分策略，提供统一的评分接口。

    Usage:
        scorer = UnifiedScoringSystem([
            TrendScoringStrategy(),
            BreakoutScoringStrategy(),
            MultiDimensionScoringStrategy(),
        ])

        results = scorer.calculate_scores(df)
        best_strategy, best_result = scorer.get_best_strategy(df)
    """

    def __init__(self, strategies: Optional[List[ScoringStrategy]] = None):
        """
        初始化统一评分系统

        Args:
            strategies: 评分策略列表，如果为 None 则使用默认策略
        """
        self.strategies = strategies or []

        # 如果没有提供策略，使用默认策略
        if not self.strategies:
            self._load_default_strategies()

    def _load_default_strategies(self):
        """加载默认策略"""
        try:
            from .strategies import (
                TrendScoringStrategy,
                BreakoutScoringStrategy,
                MultiDimensionScoringStrategy,
            )
            self.strategies = [
                TrendScoringStrategy(),
                BreakoutScoringStrategy(),
                MultiDimensionScoringStrategy(),
            ]
        except ImportError:
            # 如果导入失败，保持空列表
            pass

    def add_strategy(self, strategy: ScoringStrategy) -> 'UnifiedScoringSystem':
        """添加评分策略

        Args:
            strategy: 评分策略实例

        Returns:
            self，支持链式调用
        """
        self.strategies.append(strategy)
        return self

    def remove_strategy(self, name: str) -> 'UnifiedScoringSystem':
        """移除评分策略

        Args:
            name: 策略名称

        Returns:
            self，支持链式调用
        """
        self.strategies = [s for s in self.strategies if s.name != name]
        return self

    def get_strategy(self, name: str) -> Optional[ScoringStrategy]:
        """获取指定策略

        Args:
            name: 策略名称

        Returns:
            策略实例，如果不存在则返回 None
        """
        for strategy in self.strategies:
            if strategy.name == name:
                return strategy
        return None

    def calculate_scores(
        self,
        df: pd.DataFrame,
        **kwargs
    ) -> Dict[str, ScoreResult]:
        """使用所有策略计算评分

        Args:
            df: K线数据
            **kwargs: 传递给策略的额外参数

        Returns:
            Dict[str, ScoreResult]: 策略名称到评分结果的映射
        """
        results = {}

        for strategy in self.strategies:
            try:
                if strategy.validate_data(df):
                    result = strategy.calculate_score(df, **kwargs)
                    results[strategy.name] = result
            except Exception as e:
                # 策略执行失败时，记录错误但不中断
                results[strategy.name] = ScoreResult(
                    final_score=0,
                    passed_filter=False,
                    filter_reason=f"策略执行失败: {str(e)}",
                    strategy_name=strategy.name,
                )

        return results

    def get_best_strategy(
        self,
        df: pd.DataFrame,
        **kwargs
    ) -> Tuple[Optional[str], Optional[ScoreResult]]:
        """获取最高评分的策略

        Args:
            df: K线数据
            **kwargs: 传递给策略的额外参数

        Returns:
            Tuple[策略名称, 评分结果]，如果没有有效结果则返回 (None, None)
        """
        scores = self.calculate_scores(df, **kwargs)

        # 只考虑通过过滤的结果
        valid_scores = {
            name: result for name, result in scores.items()
            if result.passed_filter
        }

        if not valid_scores:
            return None, None

        return max(valid_scores.items(), key=lambda x: x[1].final_score)

    def get_average_score(
        self,
        df: pd.DataFrame,
        **kwargs
    ) -> float:
        """获取所有策略的平均评分

        Args:
            df: K线数据
            **kwargs: 传递给策略的额外参数

        Returns:
            float: 平均评分
        """
        scores = self.calculate_scores(df, **kwargs)

        valid_scores = [
            result.final_score for result in scores.values()
            if result.passed_filter
        ]

        if not valid_scores:
            return 0.0

        return sum(valid_scores) / len(valid_scores)

    def get_summary(
        self,
        df: pd.DataFrame,
        **kwargs
    ) -> Dict:
        """获取评分摘要

        Args:
            df: K线数据
            **kwargs: 传递给策略的额外参数

        Returns:
            Dict: 包含各策略评分和综合评价的摘要
        """
        scores = self.calculate_scores(df, **kwargs)

        best_name, best_result = self.get_best_strategy(df, **kwargs)
        avg_score = self.get_average_score(df, **kwargs)

        return {
            'scores': {name: result.to_dict() for name, result in scores.items()},
            'best_strategy': best_name,
            'best_score': best_result.final_score if best_result else 0,
            'average_score': round(avg_score, 2),
            'overall_level': self._get_overall_level(avg_score),
            'passed_strategies': [
                name for name, result in scores.items()
                if result.passed_filter
            ],
        }

    def _get_overall_level(self, avg_score: float) -> str:
        """获取综合评级"""
        if avg_score >= 80:
            return "优秀"
        elif avg_score >= 60:
            return "良好"
        elif avg_score >= 40:
            return "一般"
        else:
            return "较弱"
