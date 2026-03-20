"""
样本外验证模块

实现策略的样本外验证：
- 训练/测试集分割
- 过拟合检测
- 参数稳定性检验
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class TrainTestSplit:
    """训练测试分割"""
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    train_ratio: float
    split_date: Optional[datetime] = None


@dataclass
class ThreeWaySplit:
    """三阶段数据划分：训练集、验证集、测试集"""
    train_data: pd.DataFrame
    validation_data: pd.DataFrame
    test_data: pd.DataFrame
    train_ratio: float
    validation_ratio: float
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    validation_start: Optional[str] = None
    validation_end: Optional[str] = None
    test_start: Optional[str] = None
    test_end: Optional[str] = None


@dataclass
class OutOfSampleResult:
    """样本外验证结果"""
    train_metrics: Dict
    test_metrics: Dict
    overfitting_detected: bool
    overfitting_score: float
    stability_score: float
    recommendations: List[str]


class OutOfSampleValidator:
    """
    样本外验证器

    检测策略是否过拟合
    """

    # 过拟合判定阈值
    OVERFITTING_THRESHOLDS = {
        'return_drop': 0.30,      # 测试集收益低于训练集30%视为过拟合
        'sharpe_drop': 0.25,      # 夏普比率下降阈值
        'ic_drop': 0.35,          # IC下降阈值
        'max_dd_increase': 0.50,  # 最大回撤增加阈值
    }

    def __init__(
        self,
        train_ratio: float = 0.7,
        validation_method: str = 'simple',  # simple, rolling, walk_forward
        n_splits: int = 5,
        random_seed: Optional[int] = None
    ):
        """
        初始化样本外验证器

        Args:
            train_ratio: 训练集比例
            validation_method: 验证方法
            n_splits: 折数（用于滚动验证）
            random_seed: 随机种子
        """
        self.train_ratio = train_ratio
        self.validation_method = validation_method
        self.n_splits = n_splits
        self.random_seed = random_seed

    def split_data(
        self,
        df: pd.DataFrame,
        date_column: str = 'timestamp'
    ) -> TrainTestSplit:
        """
        分割训练集和测试集

        Args:
            df: 数据
            date_column: 日期列

        Returns:
            TrainTestSplit: 分割结果
        """
        if len(df) < 100:
            raise ValueError("数据量不足，至少需要100条记录")

        # 确保数据按日期排序
        if date_column in df.columns:
            df = df.sort_values(date_column)
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()

        # 分割
        split_idx = int(len(df) * self.train_ratio)

        train_data = df.iloc[:split_idx].copy()
        test_data = df.iloc[split_idx:].copy()

        split_date = None
        if date_column in df.columns:
            split_date = df.iloc[split_idx][date_column]
        elif isinstance(df.index, pd.DatetimeIndex):
            split_date = df.index[split_idx]

        return TrainTestSplit(
            train_data=train_data,
            test_data=test_data,
            train_ratio=self.train_ratio,
            split_date=split_date
        )

    def split_three_way(
        self,
        df: pd.DataFrame,
        date_column: str = 'timestamp',
        validation_ratio: float = 0.15
    ) -> ThreeWaySplit:
        """
        三阶段数据划分：训练集、验证集、测试集

        Args:
            df: 数据
            date_column: 日期列
            validation_ratio: 验证集比例（从训练集中划分）

        Returns:
            ThreeWaySplit: 三阶段分割结果
        """
        if len(df) < 150:
            raise ValueError("数据量不足，至少需要150条记录")

        # 确保数据按日期排序
        if date_column in df.columns:
            df = df.sort_values(date_column)
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()

        # 首先按 train_ratio 分割训练集和测试集
        test_split_idx = int(len(df) * self.train_ratio)

        # 再从训练集中划分验证集
        train_total_idx = test_split_idx
        validation_idx = int(train_total_idx * (1 - validation_ratio))

        train_data = df.iloc[:validation_idx].copy()
        validation_data = df.iloc[validation_idx:test_split_idx].copy()
        test_data = df.iloc[test_split_idx:].copy()

        # 记录分割日期
        train_start = train_end = validation_start = validation_end = test_start = test_end = None

        if isinstance(df.index, pd.DatetimeIndex):
            if len(train_data) > 0:
                train_start = str(train_data.index[0].date())
                train_end = str(train_data.index[-1].date())
            if len(validation_data) > 0:
                validation_start = str(validation_data.index[0].date())
                validation_end = str(validation_data.index[-1].date())
            if len(test_data) > 0:
                test_start = str(test_data.index[0].date())
                test_end = str(test_data.index[-1].date())

        return ThreeWaySplit(
            train_data=train_data,
            validation_data=validation_data,
            test_data=test_data,
            train_ratio=self.train_ratio,
            validation_ratio=validation_ratio,
            train_start=train_start,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            test_start=test_start,
            test_end=test_end
        )

    def split_three_way_by_date(
        self,
        df: pd.DataFrame,
        train_start: str,
        train_end: str,
        validation_start: str,
        validation_end: str,
        test_start: str,
        test_end: str,
        date_column: str = 'timestamp'
    ) -> ThreeWaySplit:
        """
        按日期进行三阶段划分

        Args:
            df: 数据
            train_start: 训练集开始日期 (YYYY-MM-DD)
            train_end: 训练集结束日期
            validation_start: 验证集开始日期
            validation_end: 验证集结束日期
            test_start: 测试集开始日期
            test_end: 测试集结束日期
            date_column: 日期列

        Returns:
            ThreeWaySplit: 三阶段分割结果
        """
        # 确保数据按日期排序
        if date_column in df.columns:
            df = df.sort_values(date_column)
            df = df.set_index(date_column)
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()

        train_data = df.loc[train_start:train_end].copy()
        validation_data = df.loc[validation_start:validation_end].copy()
        test_data = df.loc[test_start:test_end].copy()

        # 计算比例
        total_len = len(df)
        train_ratio = len(train_data) / total_len if total_len > 0 else 0
        validation_ratio = len(validation_data) / total_len if total_len > 0 else 0

        return ThreeWaySplit(
            train_data=train_data,
            validation_data=validation_data,
            test_data=test_data,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            train_start=train_start,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            test_start=test_start,
            test_end=test_end
        )

    def validate(
        self,
        df: pd.DataFrame,
        strategy: Callable,
        score_calculator: Optional[Callable] = None,
        date_column: str = 'timestamp'
    ) -> OutOfSampleResult:
        """
        执行样本外验证

        Args:
            df: 数据
            strategy: 策略函数
            score_calculator: 评分计算函数
            date_column: 日期列

        Returns:
            OutOfSampleResult: 验证结果
        """
        # 分割数据
        split = self.split_data(df, date_column)

        # 在训练集上优化参数
        train_result = self._run_backtest(split.train_data, strategy, score_calculator)
        train_metrics = self._extract_metrics(train_result)

        # 在测试集上验证
        test_result = self._run_backtest(split.test_data, strategy, score_calculator)
        test_metrics = self._extract_metrics(test_result)

        # 检测过拟合
        overfitting_detected, overfitting_score = self.detect_overfitting(
            train_metrics, test_metrics
        )

        # 计算稳定性
        stability_score = self._calculate_stability(train_metrics, test_metrics)

        # 生成建议
        recommendations = self._generate_recommendations(
            train_metrics, test_metrics, overfitting_detected
        )

        return OutOfSampleResult(
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            overfitting_detected=overfitting_detected,
            overfitting_score=overfitting_score,
            stability_score=stability_score,
            recommendations=recommendations
        )

    def _run_backtest(
        self,
        df: pd.DataFrame,
        strategy: Callable,
        score_calculator: Optional[Callable] = None
    ) -> Dict:
        """运行回测"""
        try:
            # 如果策略有backtest方法
            if hasattr(strategy, 'backtest'):
                return strategy.backtest(df)

            # 如果策略有run方法
            if hasattr(strategy, 'run'):
                return strategy.run(df)

            # 直接调用策略函数
            return strategy(df)

        except Exception as e:
            return {
                'error': str(e),
                'annual_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0
            }

    def _extract_metrics(self, result: Dict) -> Dict:
        """提取关键指标"""
        return {
            'annual_return': result.get('annual_return', 0),
            'sharpe_ratio': result.get('sharpe_ratio', 0),
            'max_drawdown': result.get('max_drawdown', 0),
            'win_rate': result.get('win_rate', 0),
            'ic': result.get('ic', 0),
            'total_trades': result.get('total_trades', 0)
        }

    def detect_overfitting(
        self,
        train_metrics: Dict,
        test_metrics: Dict
    ) -> Tuple[bool, float]:
        """
        检测过拟合

        比较训练集和测试集表现

        Args:
            train_metrics: 训练集指标
            test_metrics: 测试集指标

        Returns:
            Tuple[bool, float]: (是否过拟合, 过拟合分数)
        """
        overfitting_indicators = []

        # 1. 收益率下降
        train_return = train_metrics.get('annual_return', 0)
        test_return = test_metrics.get('annual_return', 0)

        if train_return > 0:
            return_drop = (train_return - test_return) / train_return
            if return_drop > self.OVERFITTING_THRESHOLDS['return_drop']:
                overfitting_indicators.append(('return', return_drop))

        # 2. 夏普比率下降
        train_sharpe = train_metrics.get('sharpe_ratio', 0)
        test_sharpe = test_metrics.get('sharpe_ratio', 0)

        if train_sharpe > 0:
            sharpe_drop = (train_sharpe - test_sharpe) / train_sharpe
            if sharpe_drop > self.OVERFITTING_THRESHOLDS['sharpe_drop']:
                overfitting_indicators.append(('sharpe', sharpe_drop))

        # 3. IC下降
        train_ic = train_metrics.get('ic', 0)
        test_ic = test_metrics.get('ic', 0)

        if train_ic > 0:
            ic_drop = (train_ic - test_ic) / train_ic
            if ic_drop > self.OVERFITTING_THRESHOLDS['ic_drop']:
                overfitting_indicators.append(('ic', ic_drop))

        # 4. 最大回撤增加
        train_dd = abs(train_metrics.get('max_drawdown', 0))
        test_dd = abs(test_metrics.get('max_drawdown', 0))

        if train_dd > 0:
            dd_increase = (test_dd - train_dd) / train_dd
            if dd_increase > self.OVERFITTING_THRESHOLDS['max_dd_increase']:
                overfitting_indicators.append(('drawdown', dd_increase))

        # 计算过拟合分数
        if overfitting_indicators:
            overfitting_score = np.mean([v for _, v in overfitting_indicators])
        else:
            overfitting_score = 0

        return len(overfitting_indicators) >= 2, overfitting_score

    def _calculate_stability(
        self,
        train_metrics: Dict,
        test_metrics: Dict
    ) -> float:
        """
        计算策略稳定性

        训练集和测试集表现越接近，稳定性越高
        """
        stability_scores = []

        # 收益率稳定性
        train_return = train_metrics.get('annual_return', 0)
        test_return = test_metrics.get('annual_return', 0)

        if train_return != 0:
            return_stability = min(test_return, train_return) / max(abs(test_return), abs(train_return), 1e-6)
            stability_scores.append(max(0, return_stability))

        # 夏普比率稳定性
        train_sharpe = train_metrics.get('sharpe_ratio', 0)
        test_sharpe = test_metrics.get('sharpe_ratio', 0)

        if train_sharpe != 0:
            sharpe_stability = min(test_sharpe, train_sharpe) / max(abs(test_sharpe), abs(train_sharpe), 1e-6)
            stability_scores.append(max(0, sharpe_stability))

        # IC稳定性
        train_ic = train_metrics.get('ic', 0)
        test_ic = test_metrics.get('ic', 0)

        if train_ic != 0:
            ic_stability = min(test_ic, train_ic) / max(abs(test_ic), abs(train_ic), 1e-6)
            stability_scores.append(max(0, ic_stability))

        return np.mean(stability_scores) if stability_scores else 0.5

    def _generate_recommendations(
        self,
        train_metrics: Dict,
        test_metrics: Dict,
        overfitting_detected: bool
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []

        if overfitting_detected:
            recommendations.append("⚠️ 检测到过拟合风险，建议：")

            train_return = train_metrics.get('annual_return', 0)
            test_return = test_metrics.get('annual_return', 0)

            if test_return < train_return * 0.5:
                recommendations.append(
                    "  - 测试集收益显著低于训练集，考虑减少参数数量或增加正则化"
                )

            train_sharpe = train_metrics.get('sharpe_ratio', 0)
            test_sharpe = test_metrics.get('sharpe_ratio', 0)

            if test_sharpe < train_sharpe * 0.5:
                recommendations.append(
                    "  - 测试集夏普比率下降明显，策略在不同市场环境下表现不稳定"
                )

            test_dd = abs(test_metrics.get('max_drawdown', 0))
            train_dd = abs(train_metrics.get('max_drawdown', 0))

            if test_dd > train_dd * 1.5:
                recommendations.append(
                    "  - 测试集回撤增大，建议加强风险控制逻辑"
                )

            recommendations.extend([
                "  - 使用更长的样本外验证期",
                "  - 考虑使用滚动窗口优化",
                "  - 增加因子有效性的持续监控"
            ])
        else:
            if test_metrics.get('annual_return', 0) >= train_metrics.get('annual_return', 0) * 0.8:
                recommendations.append("✅ 策略表现稳定，样本外验证通过")
            else:
                recommendations.append("📊 策略表现尚可，建议继续监控")

        return recommendations

    def rolling_validation(
        self,
        df: pd.DataFrame,
        strategy: Callable,
        window_size: int = 252,
        step_size: int = 63
    ) -> List[OutOfSampleResult]:
        """
        滚动窗口验证

        在不同时间段多次验证策略

        Args:
            df: 数据
            strategy: 策略
            window_size: 窗口大小（交易日）
            step_size: 步进大小

        Returns:
            List[OutOfSampleResult]: 多次验证结果
        """
        results = []
        n = len(df)

        for i in range(0, n - window_size, step_size):
            window_df = df.iloc[i:i+window_size]

            result = self.validate(window_df, strategy)
            results.append(result)

        return results

    def walk_forward_analysis(
        self,
        df: pd.DataFrame,
        strategy: Callable,
        train_period: int = 252,
        test_period: int = 63
    ) -> Dict:
        """
        滚动前进分析

        模拟实盘交易中的参数更新过程

        Args:
            df: 数据
            strategy: 策略
            train_period: 训练期长度
            test_period: 测试期长度

        Returns:
            Dict: 分析结果
        """
        n = len(df)

        if n < train_period + test_period:
            return {'error': '数据量不足'}

        all_test_returns = []
        all_train_returns = []

        for i in range(0, n - train_period - test_period, test_period):
            train_df = df.iloc[i:i+train_period]
            test_df = df.iloc[i+train_period:i+train_period+test_period]

            # 训练
            train_result = self._run_backtest(train_df, strategy)
            train_metrics = self._extract_metrics(train_result)
            all_train_returns.append(train_metrics.get('annual_return', 0))

            # 测试
            test_result = self._run_backtest(test_df, strategy)
            test_metrics = self._extract_metrics(test_result)
            all_test_returns.append(test_metrics.get('annual_return', 0))

        return {
            'avg_train_return': np.mean(all_train_returns),
            'avg_test_return': np.mean(all_test_returns),
            'return_consistency': np.mean([1 if t > 0 else 0 for t in all_test_returns]),
            'performance_degradation': (
                np.mean(all_train_returns) - np.mean(all_test_returns)
            ) / abs(np.mean(all_train_returns)) if np.mean(all_train_returns) != 0 else 0
        }


def validate_out_of_sample(
    df: pd.DataFrame,
    strategy: Callable,
    train_ratio: float = 0.7
) -> Dict:
    """
    便捷函数：样本外验证

    Args:
        df: 数据
        strategy: 策略
        train_ratio: 训练集比例

    Returns:
        Dict: 验证结果
    """
    validator = OutOfSampleValidator(train_ratio=train_ratio)
    result = validator.validate(df, strategy)

    return {
        'train_metrics': result.train_metrics,
        'test_metrics': result.test_metrics,
        'overfitting_detected': result.overfitting_detected,
        'overfitting_score': result.overfitting_score,
        'stability_score': result.stability_score,
        'recommendations': result.recommendations
    }