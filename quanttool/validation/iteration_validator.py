"""
迭代验证器模块

实现RL式闭环优化流程：
- 综合奖励计算
- 迭代优化循环
- 权重动态调整
- 过拟合检测
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from scipy.optimize import minimize
import warnings
import copy

warnings.filterwarnings('ignore')


@dataclass
class IterationResult:
    """单次迭代结果"""
    iteration: int
    config: Dict
    ic: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    reward: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationResult:
    """验证结果"""
    best_config: Dict
    best_reward: float
    best_iteration: int
    total_iterations: int
    improvement_rate: float
    converged: bool
    history: List[IterationResult]


class IterationValidator:
    """
    RL式迭代验证器

    通过迭代优化提升策略表现
    """

    # 验证目标
    VALIDATION_TARGETS = {
        'annual_return': {'target': 0.10, 'weight': 0.30},
        'ic': {'target': 0.05, 'weight': 0.25},
        'sharpe_ratio': {'target': 1.0, 'weight': 0.20},
        'max_drawdown': {'target': -0.15, 'weight': 0.15},
        'win_rate': {'target': 0.50, 'weight': 0.10},
    }

    def __init__(
        self,
        max_iterations: int = 50,
        early_stop_patience: int = 10,
        improvement_threshold: float = 0.001,
        reward_function: Optional[Callable] = None
    ):
        """
        初始化迭代验证器

        Args:
            max_iterations: 最大迭代次数
            early_stop_patience: 早停耐心值
            improvement_threshold: 改进阈值
            reward_function: 自定义奖励函数
        """
        self.max_iterations = max_iterations
        self.early_stop_patience = early_stop_patience
        self.improvement_threshold = improvement_threshold
        self.reward_function = reward_function

        # 状态
        self.current_iteration = 0
        self.best_config = None
        self.best_reward = -float('inf')
        self.best_result = None
        self.history: List[IterationResult] = []
        self.no_improvement_count = 0

    def calculate_reward(
        self,
        ic: float,
        annual_return: float,
        sharpe: float,
        max_dd: float,
        win_rate: float
    ) -> float:
        """
        计算综合奖励值

        reward = Σ(weight_i * metric_i / target_i)

        Args:
            ic: 信息系数
            annual_return: 年化收益率
            sharpe: 夏普比率
            max_dd: 最大回撤
            win_rate: 胜率

        Returns:
            float: 综合奖励值
        """
        if self.reward_function:
            return self.reward_function(ic, annual_return, sharpe, max_dd, win_rate)

        # 默认奖励函数
        reward = 0.0

        # IC贡献（目标>0.05）
        ic_target = self.VALIDATION_TARGETS['ic']['target']
        ic_weight = self.VALIDATION_TARGETS['ic']['weight']
        reward += ic_weight * (ic / ic_target) if ic_target != 0 else 0

        # 年化收益贡献（目标>10%）
        return_target = self.VALIDATION_TARGETS['annual_return']['target']
        return_weight = self.VALIDATION_TARGETS['annual_return']['weight']
        reward += return_weight * (annual_return / return_target) if return_target != 0 else 0

        # 夏普比率贡献（目标>1.0）
        sharpe_target = self.VALIDATION_TARGETS['sharpe_ratio']['target']
        sharpe_weight = self.VALIDATION_TARGETS['sharpe_ratio']['weight']
        reward += sharpe_weight * (sharpe / sharpe_target) if sharpe_target != 0 else 0

        # 最大回撤贡献（目标<15%）
        dd_target = self.VALIDATION_TARGETS['max_drawdown']['target']
        dd_weight = self.VALIDATION_TARGETS['max_drawdown']['weight']
        # 回撤为负，越小越好
        dd_score = -max_dd / abs(dd_target) if dd_target != 0 else 0
        reward += dd_weight * min(1.0, dd_score)

        # 胜率贡献（目标>50%）
        win_target = self.VALIDATION_TARGETS['win_rate']['target']
        win_weight = self.VALIDATION_TARGETS['win_rate']['weight']
        reward += win_weight * (win_rate / win_target) if win_target != 0 else 0

        return reward

    def run_iteration(
        self,
        df: pd.DataFrame,
        config: Dict,
        score_calculator: Callable,
        backtest_engine: Callable
    ) -> IterationResult:
        """
        执行单次迭代

        Args:
            df: 数据
            config: 配置参数
            score_calculator: 评分计算函数
            backtest_engine: 回测引擎函数

        Returns:
            IterationResult: 迭代结果
        """
        self.current_iteration += 1

        try:
            # 应用配置计算评分
            scores = score_calculator(df, config)

            # 执行回测
            backtest_result = backtest_engine(df, scores)

            # 提取指标
            ic = backtest_result.get('ic', 0)
            annual_return = backtest_result.get('annual_return', 0)
            sharpe = backtest_result.get('sharpe_ratio', 0)
            max_dd = backtest_result.get('max_drawdown', 0)
            win_rate = backtest_result.get('win_rate', 0)

            # 计算奖励
            reward = self.calculate_reward(ic, annual_return, sharpe, max_dd, win_rate)

            # 记录结果
            result = IterationResult(
                iteration=self.current_iteration,
                config=copy.deepcopy(config),
                ic=ic,
                annual_return=annual_return,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                win_rate=win_rate,
                reward=reward
            )

            self.history.append(result)

            # 更新最佳配置
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_config = copy.deepcopy(config)
                self.best_result = result
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            return result

        except Exception as e:
            # 返回空结果
            return IterationResult(
                iteration=self.current_iteration,
                config=config,
                ic=0,
                annual_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                reward=-1.0
            )

    def optimize_weights(
        self,
        current_weights: Dict,
        reward: float,
        learning_rate: float = 0.1
    ) -> Dict:
        """
        根据奖励优化权重

        使用梯度上升法优化

        Args:
            current_weights: 当前权重
            reward: 当前奖励
            learning_rate: 学习率

        Returns:
            Dict: 优化后的权重
        """
        if not self.history or len(self.history) < 2:
            return current_weights

        # 计算奖励变化
        prev_reward = self.history[-2].reward if len(self.history) >= 2 else reward
        reward_change = reward - prev_reward

        # 获取历史最佳权重
        if self.best_config and 'weights' in self.best_config:
            best_weights = self.best_config['weights']
        else:
            best_weights = current_weights

        # 如果奖励增加，向当前方向移动
        # 如果奖励减少，向历史最佳方向移动
        new_weights = {}

        for key in current_weights:
            if reward_change > 0:
                # 继续当前方向
                movement = (current_weights[key] - best_weights.get(key, current_weights[key])) * learning_rate
                new_weights[key] = current_weights[key] + movement
            else:
                # 回归最佳方向
                new_weights[key] = current_weights[key] + learning_rate * (best_weights.get(key, current_weights[key]) - current_weights[key])

        # 归一化权重
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}

        return new_weights

    def run_iteration_loop(
        self,
        df: pd.DataFrame,
        initial_config: Dict,
        score_calculator: Callable,
        backtest_engine: Callable,
        config_optimizer: Optional[Callable] = None
    ) -> ValidationResult:
        """
        执行完整迭代循环

        终止条件:
        1. 年化收益 > 10% 且 IC > 0.05
        2. 连续N次无改善
        3. 达到最大迭代次数

        Args:
            df: 数据
            initial_config: 初始配置
            score_calculator: 评分计算函数
            backtest_engine: 回测引擎
            config_optimizer: 配置优化器（可选）

        Returns:
            ValidationResult: 验证结果
        """
        current_config = copy.deepcopy(initial_config)
        initial_reward = 0

        for i in range(self.max_iterations):
            # 执行迭代
            result = self.run_iteration(
                df, current_config, score_calculator, backtest_engine
            )

            if i == 0:
                initial_reward = result.reward

            # 检查是否达到目标
            if self._check_target_reached(result):
                break

            # 检查早停
            if self.no_improvement_count >= self.early_stop_patience:
                break

            # 优化配置
            if config_optimizer:
                current_config = config_optimizer(current_config, result)
            else:
                # 默认优化权重
                if 'weights' in current_config:
                    current_config['weights'] = self.optimize_weights(
                        current_config['weights'],
                        result.reward
                    )

        # 计算改进率
        improvement_rate = (self.best_reward - initial_reward) / abs(initial_reward) if initial_reward != 0 else 0

        return ValidationResult(
            best_config=self.best_config or initial_config,
            best_reward=self.best_reward,
            best_iteration=self.history.index(self.best_result) if self.best_result else 0,
            total_iterations=self.current_iteration,
            improvement_rate=improvement_rate,
            converged=self.no_improvement_count < self.early_stop_patience,
            history=self.history
        )

    def _check_target_reached(self, result: IterationResult) -> bool:
        """
        检查是否达到目标

        年化收益 > 10% 且 IC > 0.05 且 夏普 > 1.0
        """
        return (
            result.annual_return > 0.10 and
            result.ic > 0.05 and
            result.sharpe_ratio > 1.0
        )

    def bayesian_optimization(
        self,
        df: pd.DataFrame,
        param_space: Dict,
        score_calculator: Callable,
        backtest_engine: Callable,
        n_iterations: int = 20
    ) -> Dict:
        """
        贝叶斯优化

        使用高斯过程优化超参数

        Args:
            df: 数据
            param_space: 参数空间 {'param': (min, max)}
            score_calculator: 评分计算函数
            backtest_engine: 回测引擎
            n_iterations: 迭代次数

        Returns:
            Dict: 最优参数
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real

            # 定义目标函数
            def objective(params):
                config = {}
                for i, (key, (low, high)) in enumerate(param_space.items()):
                    config[key] = params[i]

                result = self.run_iteration(
                    df, config, score_calculator, backtest_engine
                )
                return -result.reward  # 最小化负奖励

            # 定义搜索空间
            dimensions = [
                Real(low, high, name=key)
                for key, (low, high) in param_space.items()
            ]

            # 运行优化
            result = gp_minimize(
                objective,
                dimensions,
                n_calls=n_iterations,
                random_state=42
            )

            # 返回最优参数
            best_params = {}
            for i, key in enumerate(param_space.keys()):
                best_params[key] = result.x[i]

            return best_params

        except ImportError:
            # 如果没有scikit-optimize，使用随机搜索
            return self._random_search(
                df, param_space, score_calculator, backtest_engine, n_iterations
            )

    def _random_search(
        self,
        df: pd.DataFrame,
        param_space: Dict,
        score_calculator: Callable,
        backtest_engine: Callable,
        n_iterations: int
    ) -> Dict:
        """
        随机搜索
        """
        best_params = None
        best_reward = -float('inf')

        for _ in range(n_iterations):
            # 随机采样参数
            config = {}
            for key, (low, high) in param_space.items():
                config[key] = np.random.uniform(low, high)

            # 评估
            result = self.run_iteration(
                df, config, score_calculator, backtest_engine
            )

            if result.reward > best_reward:
                best_reward = result.reward
                best_params = config

        return best_params

    def detect_overfitting(
        self,
        train_result: IterationResult,
        test_result: IterationResult,
        threshold: float = 0.3
    ) -> bool:
        """
        检测过拟合

        如果测试集表现显著低于训练集，则认为过拟合

        Args:
            train_result: 训练集结果
            test_result: 测试集结果
            threshold: 差异阈值

        Returns:
            bool: 是否过拟合
        """
        # 计算各指标差异
        return_diff = abs(train_result.annual_return - test_result.annual_return)
        ic_diff = abs(train_result.ic - test_result.ic)
        sharpe_diff = abs(train_result.sharpe_ratio - test_result.sharpe_ratio)

        # 测试集表现显著低于训练集
        if test_result.annual_return < train_result.annual_return * (1 - threshold):
            return True

        if test_result.ic < train_result.ic * (1 - threshold):
            return True

        return False

    def get_iteration_summary(self) -> Dict:
        """
        获取迭代摘要
        """
        if not self.history:
            return {}

        rewards = [r.reward for r in self.history]
        returns = [r.annual_return for r in self.history]
        ics = [r.ic for r in self.history]

        return {
            'total_iterations': len(self.history),
            'best_reward': self.best_reward,
            'best_iteration': self.best_result.iteration if self.best_result else None,
            'avg_reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'avg_return': np.mean(returns),
            'avg_ic': np.mean(ics),
            'improvement_trend': np.polyfit(range(len(rewards)), rewards, 1)[0] if len(rewards) > 1 else 0
        }


def run_iteration_validation(
    df: pd.DataFrame,
    initial_config: Dict,
    score_calculator: Callable,
    backtest_engine: Callable
) -> Dict:
    """
    便捷函数：运行迭代验证

    Args:
        df: 数据
        initial_config: 初始配置
        score_calculator: 评分计算函数
        backtest_engine: 回测引擎

    Returns:
        Dict: 验证结果
    """
    validator = IterationValidator()
    result = validator.run_iteration_loop(
        df, initial_config, score_calculator, backtest_engine
    )

    return {
        'best_config': result.best_config,
        'best_reward': result.best_reward,
        'total_iterations': result.total_iterations,
        'improvement_rate': result.improvement_rate,
        'converged': result.converged
    }