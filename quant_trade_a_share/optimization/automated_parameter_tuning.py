"""
自动化调参模块
利用Qlib的自动化流程优化策略参数
"""
import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Callable
from itertools import product
import warnings
warnings.filterwarnings('ignore')

try:
    import qlib
    from qlib.config import REG_CN as REGION_CN
    from qlib.workflow import R
    from qlib.tests.data import GetData
    from qlib.utils import init_instance_by_config
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    print("⚠️ Qlib 未安装，将使用基础参数优化功能")

from quant_trade_a_share.models.model_fusion import ModelFusion
from quant_trade_a_share.factors.factor_library_expansion import FactorLibraryExpansion
from quant_trade_a_share.risk.portfolio_risk_management import PortfolioRiskManagement

# Handle MyTTIndicators import with error handling
try:
    from quant_trade_a_share.utils.mytt_indicators import *
    MYTT_AVAILABLE = True

    # Create a wrapper class for compatibility
    class MyTTIndicators:
        def __init__(self):
            pass

        def MA(self, S, N):
            from quant_trade_a_share.utils.mytt_indicators import MA
            return MA(S, N)

        def EMA(self, S, N):
            from quant_trade_a_share.utils.mytt_indicators import EMA
            return EMA(S, N)

        def MACD(self, S, SHORT=12, LONG=26, M=9):
            from quant_trade_a_share.utils.mytt_indicators import MACD
            return MACD(S, SHORT, LONG, M)

        def KDJ(self, CLOSE, HIGH, LOW, N=9, M1=3, M2=3):
            from quant_trade_a_share.utils.mytt_indicators import KDJ
            return KDJ(CLOSE, HIGH, LOW, N, M1, M2)

        def RSI(self, CLOSE, N=24):
            from quant_trade_a_share.utils.mytt_indicators import RSI
            return RSI(CLOSE, N)

        def BOLL(self, CLOSE, N=20, P=2):
            from quant_trade_a_share.utils.mytt_indicators import BOLL
            return BOLL(CLOSE, N, P)

        def CCI(self, CLOSE, HIGH, LOW, N=14):
            from quant_trade_a_share.utils.mytt_indicators import CCI
            return CCI(CLOSE, HIGH, LOW, N)

        def ATR(self, CLOSE, HIGH, LOW, N=20):
            from quant_trade_a_share.utils.mytt_indicators import ATR
            return ATR(CLOSE, HIGH, LOW, N)

        def DMA(self, CLOSE, M1=10, M2=50):
            from quant_trade_a_share.utils.mytt_indicators import DFMA
            return DFMA(CLOSE, M1, M2)

        def DMI(self, HIGH, LOW, CLOSE, M1=14, M2=6):
            from quant_trade_a_share.utils.mytt_indicators import DMI
            return DMI(CLOSE, HIGH, LOW, M1, M2)

        def TRIX(self, CLOSE, M1=12, M2=20):
            from quant_trade_a_share.utils.mytt_indicators import TRIX
            return TRIX(CLOSE, M1, M2)

        def VR(self, CLOSE, VOL, M1=26):
            from quant_trade_a_share.utils.mytt_indicators import VR
            return VR(CLOSE, VOL, M1)

        def WR(self, CLOSE, HIGH, LOW, N=10, N1=6):
            from quant_trade_a_share.utils.mytt_indicators import WR
            return WR(CLOSE, HIGH, LOW, N, N1)

except ImportError:
    print("⚠️ MyTT 指标不可用，将使用基础功能")
    MYTT_AVAILABLE = False

    # Dummy class as fallback
    class MyTTIndicators:
        def __init__(self):
            pass

        def MA(self, S, N):
            if hasattr(pd, 'Series'):
                return pd.Series(S).rolling(N).mean().values
            else:
                return np.full_like(S, np.mean(S) if len(S) > 0 else 0)

        def EMA(self, S, N):
            if hasattr(pd, 'Series'):
                return pd.Series(S).ewm(span=N, adjust=False).mean().values
            else:
                return S

        def MACD(self, S, SHORT=12, LONG=26, M=9):
            return np.zeros(len(S)), np.zeros(len(S)), np.zeros(len(S))

        def KDJ(self, CLOSE, HIGH, LOW, N=9, M1=3, M2=3):
            return np.zeros(len(CLOSE)), np.zeros(len(CLOSE)), np.zeros(len(CLOSE))

        def RSI(self, CLOSE, N=24):
            return np.zeros(len(CLOSE))

        def BOLL(self, CLOSE, N=20, P=2):
            mid = self.MA(CLOSE, N)
            std = pd.Series(CLOSE).rolling(N).std().values if hasattr(pd, 'Series') else np.zeros(len(CLOSE))
            return mid + P * std, mid, mid - P * std

        def CCI(self, CLOSE, HIGH, LOW, N=14):
            return np.zeros(len(CLOSE))

        def ATR(self, CLOSE, HIGH, LOW, N=20):
            return np.zeros(len(CLOSE))

        def DMA(self, CLOSE, M1=10, M2=50):
            return np.zeros(len(CLOSE)), np.zeros(len(CLOSE))

        def DMI(self, HIGH, LOW, CLOSE, M1=14, M2=6):
            return np.zeros(len(CLOSE)), np.zeros(len(CLOSE)), np.zeros(len(CLOSE)), np.zeros(len(CLOSE))

        def TRIX(self, CLOSE, M1=12, M2=20):
            return np.zeros(len(CLOSE)), np.zeros(len(CLOSE))

        def VR(self, CLOSE, VOL, M1=26):
            return np.zeros(len(CLOSE))

        def WR(self, CLOSE, HIGH, LOW, N=10, N1=6):
            return np.zeros(len(CLOSE)), np.zeros(len(CLOSE))

class AutomatedParameterTuning:
    """
    自动化参数调优类
    利用Qlib和其他方法优化策略参数
    """

    def __init__(self):
        """初始化参数调优器"""
        self.model_fusion = ModelFusion()
        self.factor_library = FactorLibraryExpansion()
        self.risk_manager = PortfolioRiskManagement()
        self.mytt_indicators = MyTTIndicators()

        # 评估指标权重
        self.metric_weights = {
            'sharpe_ratio': 0.4,
            'max_drawdown': -0.3,  # 负值因为我们要最大化
            'total_return': 0.2,
            'win_rate': 0.1
        }

        print("✅ 自动化参数调优器初始化完成")

    def grid_search_optimization(self, data: pd.DataFrame, param_grid: Dict[str, List],
                               target_metric: str = 'sharpe_ratio',
                               scoring_func: Callable = None) -> Tuple[Dict, float]:
        """
        网格搜索参数优化

        Args:
            data: 输入数据
            param_grid: 参数网格 {param_name: [values]}
            target_metric: 目标指标
            scoring_func: 自定义评分函数
        """
        print("🔍 开始网格搜索参数优化...")

        # 获取所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(product(*param_values))

        best_score = float('-inf')
        best_params = {}

        print(f"📊 将测试 {len(all_combinations)} 种参数组合")

        for i, combination in enumerate(all_combinations):
            current_params = dict(zip(param_names, combination))

            try:
                # 使用当前参数评估策略
                score = self.evaluate_strategy_with_params(data, current_params)

                if score > best_score:
                    best_score = score
                    best_params = current_params.copy()

                if (i + 1) % 10 == 0:  # 每10次打印一次进度
                    print(f"📈 已完成 {i + 1}/{len(all_combinations)}, "
                          f"当前最佳得分: {best_score:.4f}")

            except Exception as e:
                print(f"⚠️ 参数组合 {current_params} 评估失败: {e}")
                continue

        print(f"✅ 网格搜索完成，最佳参数: {best_params}, 得分: {best_score:.4f}")
        return best_params, best_score

    def evaluate_strategy_with_params(self, data: pd.DataFrame, params: Dict) -> float:
        """
        使用特定参数评估策略

        Args:
            data: 输入数据
            params: 参数字典
        """
        try:
            # 根据参数类型选择评估方法
            if any(key in params for key in ['ma_short', 'ma_long', 'rsi_period']):
                # 技术指标策略评估
                score = self._evaluate_technical_strategy(data, params)
            elif any(key in params for key in ['ml_model', 'lookback_window']):
                # ML策略评估
                score = self._evaluate_ml_strategy(data, params)
            elif any(key in params for key in ['factor_weight', 'signal_threshold']):
                # 融合策略评估
                score = self._evaluate_fusion_strategy(data, params)
            else:
                # 默认评估方法
                score = self._evaluate_generic_strategy(data, params)

            return score

        except Exception as e:
            print(f"⚠️ 参数评估失败: {e}")
            return float('-inf')  # 返回极小值

    def _evaluate_technical_strategy(self, data: pd.DataFrame, params: Dict) -> float:
        """
        评估技术指标策略
        """
        try:
            # 设置默认参数
            ma_short = params.get('ma_short', 5)
            ma_long = params.get('ma_long', 20)
            rsi_period = params.get('rsi_period', 14)
            rsi_buy_threshold = params.get('rsi_buy_threshold', 30)
            rsi_sell_threshold = params.get('rsi_sell_threshold', 70)

            # 计算技术指标
            signals = pd.Series(0, index=data.index)

            # MA交叉信号
            if 'close' in data.columns:
                ma_short_series = self.mytt_indicators.MA(data['close'], ma_short)
                ma_long_series = self.mytt_indicators.MA(data['close'], ma_long)

                buy_condition = (ma_short_series > ma_long_series) & (ma_short_series.shift(1) <= ma_long_series.shift(1))
                sell_condition = (ma_short_series < ma_long_series) & (ma_short_series.shift(1) >= ma_long_series.shift(1))

                signals[buy_condition] = 1
                signals[sell_condition] = -1

            # RSI信号
            if 'close' in data.columns:
                rsi_values = self.mytt_indicators.RSI(data['close'], rsi_period)
                rsi_buy = (rsi_values < rsi_buy_threshold) & (rsi_values.shift(1) >= rsi_buy_threshold)
                rsi_sell = (rsi_values > rsi_sell_threshold) & (rsi_values.shift(1) <= rsi_sell_threshold)

                signals[rsi_buy] = 1
                signals[rsi_sell] = -1

            # 计算收益
            if 'close' in data.columns:
                returns = data['close'].pct_change().fillna(0)
                strategy_returns = signals.shift(1).fillna(0) * returns  # 信号滞后一期

                # 计算评估指标
                metrics = self.risk_manager.calculate_basic_risk_metrics(strategy_returns)

                # 计算综合得分
                score = (metrics.get('sharpe_ratio', 0) * self.metric_weights['sharpe_ratio'] +
                        abs(metrics.get('max_drawdown', 0)) * self.metric_weights['max_drawdown'] +
                        metrics.get('annual_return', 0) * self.metric_weights['total_return'] +
                        metrics.get('win_rate', 0) * self.metric_weights['win_rate'])

                return score
            else:
                return 0.0

        except Exception as e:
            print(f"⚠️ 技术指标策略评估失败: {e}")
            return float('-inf')

    def _evaluate_ml_strategy(self, data: pd.DataFrame, params: Dict) -> float:
        """
        评估机器学习策略
        """
        try:
            lookback_window = params.get('lookback_window', 20)
            feature_lag = params.get('feature_lag', 1)
            model_type = params.get('ml_model', 'ensemble')

            # 准备特征
            features = pd.DataFrame(index=data.index)
            for lag in range(1, feature_lag + 1):
                features[f'close_lag_{lag}'] = data['close'].shift(lag) if 'close' in data.columns else pd.Series(0, index=data.index)
                features[f'volume_lag_{lag}'] = data['volume'].shift(lag) if 'volume' in data.columns else pd.Series(0, index=data.index)

            # 计算技术指标作为特征
            if 'close' in data.columns:
                rsi = self.mytt_indicators.RSI(data['close'], 14)
                features['rsi'] = rsi

            features = features.dropna()

            if len(features) == 0:
                return float('-inf')

            # 使用模型融合进行评估
            ml_signals = self.model_fusion.calculate_ml_signals(
                pd.concat([data.reindex(features.index), features], axis=1),
                model_type=model_type
            )

            # 计算收益
            if 'close' in data.columns:
                returns = data['close'].pct_change().reindex(features.index).fillna(0)
                strategy_returns = ml_signals.reindex(returns.index, fill_value=0) * returns

                # 计算评估指标
                metrics = self.risk_manager.calculate_basic_risk_metrics(strategy_returns)

                # 计算综合得分
                score = (metrics.get('sharpe_ratio', 0) * self.metric_weights['sharpe_ratio'] +
                        abs(metrics.get('max_drawdown', 0)) * self.metric_weights['max_drawdown'] +
                        metrics.get('annual_return', 0) * self.metric_weights['total_return'] +
                        metrics.get('win_rate', 0) * self.metric_weights['win_rate'])

                return score
            else:
                return 0.0

        except Exception as e:
            print(f"⚠️ ML策略评估失败: {e}")
            return float('-inf')

    def _evaluate_fusion_strategy(self, data: pd.DataFrame, params: Dict) -> float:
        """
        评估融合策略
        """
        try:
            # 获取融合策略参数
            factor_weight = params.get('factor_weight', 0.5)
            signal_threshold = params.get('signal_threshold', 0.1)

            # 计算技术信号
            technical_signals = self.model_fusion.calculate_technical_signals(data)

            # 计算ML信号
            ml_signals = self.model_fusion.calculate_ml_signals(data)

            # 计算加权融合信号
            combined_signal = pd.Series(0.0, index=data.index)

            # 平均技术信号
            if technical_signals:
                avg_tech_signal = pd.Series(0.0, index=data.index)
                for _, sig in technical_signals.items():
                    avg_tech_signal += sig
                avg_tech_signal /= len(technical_signals)
                combined_signal += avg_tech_signal * factor_weight

            # 添加ML信号
            if not ml_signals.empty:
                ml_aligned = ml_signals.reindex(data.index, fill_value=0.0)
                combined_signal += ml_aligned * (1 - factor_weight)

            # 应用阈值
            discrete_signal = pd.Series(0, index=combined_signal.index)
            discrete_signal[combined_signal > signal_threshold] = 1
            discrete_signal[combined_signal < -signal_threshold] = -1

            # 计算收益
            if 'close' in data.columns:
                returns = data['close'].pct_change().fillna(0)
                strategy_returns = discrete_signal.shift(1).fillna(0) * returns

                # 计算评估指标
                metrics = self.risk_manager.calculate_basic_risk_metrics(strategy_returns)

                # 计算综合得分
                score = (metrics.get('sharpe_ratio', 0) * self.metric_weights['sharpe_ratio'] +
                        abs(metrics.get('max_drawdown', 0)) * self.metric_weights['max_drawdown'] +
                        metrics.get('annual_return', 0) * self.metric_weights['total_return'] +
                        metrics.get('win_rate', 0) * self.metric_weights['win_rate'])

                return score
            else:
                return 0.0

        except Exception as e:
            print(f"⚠️ 融合策略评估失败: {e}")
            return float('-inf')

    def _evaluate_generic_strategy(self, data: pd.DataFrame, params: Dict) -> float:
        """
        通用策略评估
        """
        try:
            # 这里可以实现其他类型的策略评估
            # 简单返回基于波动率的分数
            if 'close' in data.columns:
                returns = data['close'].pct_change().fillna(0)
                sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
                return sharpe
            return 0.0
        except:
            return 0.0

    def bayesian_optimization(self, data: pd.DataFrame, param_space: Dict[str, Tuple],
                            n_iterations: int = 50, target_metric: str = 'sharpe_ratio') -> Tuple[Dict, float]:
        """
        贝叶斯优化（简化版本）

        Args:
            data: 输入数据
            param_space: 参数空间 {param_name: (min_val, max_val)}
            n_iterations: 迭代次数
            target_metric: 目标指标
        """
        print("🔮 开始贝叶斯优化...")

        # 简化的贝叶斯优化实现
        best_score = float('-inf')
        best_params = {}

        # 随机搜索作为近似
        for iteration in range(n_iterations):
            # 随机采样参数
            current_params = {}
            for param_name, (min_val, max_val) in param_space.items():
                if isinstance(min_val, int):
                    current_params[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    current_params[param_name] = np.random.uniform(min_val, max_val)

            # 评估参数
            score = self.evaluate_strategy_with_params(data, current_params)

            if score > best_score:
                best_score = score
                best_params = current_params.copy()

            if (iteration + 1) % 10 == 0:
                print(f"📈 贝叶斯优化迭代 {iteration + 1}/{n_iterations}, "
                      f"当前最佳得分: {best_score:.4f}")

        print(f"✅ 贝叶斯优化完成，最佳参数: {best_params}, 得分: {best_score:.4f}")
        return best_params, best_score

    def genetic_algorithm_optimization(self, data: pd.DataFrame, param_ranges: Dict[str, List],
                                    population_size: int = 20, generations: int = 30) -> Tuple[Dict, float]:
        """
        遗传算法优化

        Args:
            data: 输入数据
            param_ranges: 参数范围 {param_name: [possible_values]}
            population_size: 种群大小
            generations: 代数
        """
        print("🧬 开始遗传算法优化...")

        # 初始化种群
        population = []
        for _ in range(population_size):
            individual = {}
            for param_name, values in param_ranges.items():
                individual[param_name] = np.random.choice(values)
            population.append(individual)

        best_score = float('-inf')
        best_params = {}

        for gen in range(generations):
            # 评估种群
            fitness_scores = []
            for individual in population:
                score = self.evaluate_strategy_with_params(data, individual)
                fitness_scores.append(score)

            # 找到最佳个体
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > best_score:
                best_score = fitness_scores[max_idx]
                best_params = population[max_idx].copy()

            # 选择、交叉、变异
            new_population = []

            # 精英保留
            elite_indices = np.argsort(fitness_scores)[-2:]  # 保留最好的2个
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            # 生成新个体
            while len(new_population) < population_size:
                # 选择父母（锦标赛选择）
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                # 交叉
                child = self._crossover(parent1, parent2, param_ranges)

                # 变异
                child = self._mutate(child, param_ranges)

                new_population.append(child)

            population = new_population

            if (gen + 1) % 10 == 0:
                print(f"📈 遗传算法第 {gen + 1} 代, 当前最佳得分: {best_score:.4f}")

        print(f"✅ 遗传算法优化完成，最佳参数: {best_params}, 得分: {best_score:.4f}")
        return best_params, best_score

    def _tournament_selection(self, population: List[Dict], scores: List[float], tournament_size: int = 3) -> Dict:
        """锦标赛选择"""
        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        winner_idx = tournament_indices[np.argmax([scores[i] for i in tournament_indices])]
        return population[winner_idx].copy()

    def _crossover(self, parent1: Dict, parent2: Dict, param_ranges: Dict[str, List]) -> Dict:
        """交叉操作"""
        child = {}
        for param_name in param_ranges.keys():
            if np.random.rand() < 0.5:  # 50% 概率选择parent1的基因
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        return child

    def _mutate(self, individual: Dict, param_ranges: Dict[str, List], mutation_rate: float = 0.1) -> Dict:
        """变异操作"""
        mutated = individual.copy()
        for param_name, values in param_ranges.items():
            if np.random.rand() < mutation_rate:
                mutated[param_name] = np.random.choice(values)
        return mutated

    def multi_objective_optimization(self, data: pd.DataFrame, param_ranges: Dict[str, List],
                                   objectives: List[str] = ['sharpe_ratio', 'max_drawdown', 'total_return']) -> List[Tuple[Dict, Dict]]:
        """
        多目标优化

        Args:
            data: 输入数据
            param_ranges: 参数范围
            objectives: 优化目标列表
        """
        print("🎯 开始多目标优化...")

        # 简化版本：计算多组帕累托前沿参数
        results = []

        # 随机生成候选参数
        for _ in range(50):  # 测试50组参数
            params = {}
            for param_name, values in param_ranges.items():
                params[param_name] = np.random.choice(values)

            # 评估所有目标
            scores = {}
            temp_data = data.copy()

            # 计算技术指标策略得分
            tech_score = self._evaluate_technical_strategy(temp_data, params)
            scores['technical_strategy'] = tech_score

            # 计算ML策略得分
            ml_score = self._evaluate_ml_strategy(temp_data, params)
            scores['ml_strategy'] = ml_score

            # 计算融合策略得分
            fusion_score = self._evaluate_fusion_strategy(temp_data, params)
            scores['fusion_strategy'] = fusion_score

            results.append((params, scores))

        # 简单排序返回最佳几组
        results.sort(key=lambda x: sum(x[1].values()), reverse=True)
        top_results = results[:10]  # 返回前10组

        print(f"✅ 多目标优化完成，找到 {len(top_results)} 组优秀参数")
        return top_results

    def parameter_stability_analysis(self, data: pd.DataFrame, best_params: Dict,
                                  n_bootstrap: int = 10) -> Dict[str, Dict[str, float]]:
        """
        参数稳定性分析

        Args:
            data: 输入数据
            best_params: 最佳参数
            n_bootstrap: 自助采样次数
        """
        print("🔍 进行参数稳定性分析...")

        scores = {param_name: [] for param_name in best_params.keys()}
        metrics = ['sharpe_ratio', 'max_drawdown', 'total_return', 'win_rate']

        for i in range(n_bootstrap):
            # 随机采样数据进行自助法验证
            bootstrap_data = data.sample(frac=0.8, replace=True, random_state=i).sort_index()

            # 使用最佳参数评估
            score = self.evaluate_strategy_with_params(bootstrap_data, best_params)

            # 计算具体指标（这里需要重新计算以获取详细指标）
            # 为简化，我们重新运行评估
            try:
                if 'close' in bootstrap_data.columns:
                    returns = bootstrap_data['close'].pct_change().fillna(0)
                    if len(returns) > 1:
                        # 重新运行策略以获取完整指标
                        tech_signals = self.model_fusion.calculate_technical_signals(bootstrap_data)
                        ml_signals = self.model_fusion.calculate_ml_signals(bootstrap_data)

                        # 使用最佳参数的融合策略
                        factor_weight = best_params.get('factor_weight', 0.5)
                        signal_threshold = best_params.get('signal_threshold', 0.1)

                        combined_signal = pd.Series(0.0, index=bootstrap_data.index)
                        if tech_signals:
                            avg_tech_signal = pd.Series(0.0, index=bootstrap_data.index)
                            for _, sig in tech_signals.items():
                                avg_tech_signal += sig
                            avg_tech_signal /= len(tech_signals)
                            combined_signal += avg_tech_signal * factor_weight

                        if not ml_signals.empty:
                            ml_aligned = ml_signals.reindex(bootstrap_data.index, fill_value=0.0)
                            combined_signal += ml_aligned * (1 - factor_weight)

                        discrete_signal = pd.Series(0, index=combined_signal.index)
                        discrete_signal[combined_signal > signal_threshold] = 1
                        discrete_signal[combined_signal < -signal_threshold] = -1

                        strategy_returns = discrete_signal.shift(1).fillna(0) * returns
                        metrics_vals = self.risk_manager.calculate_basic_risk_metrics(strategy_returns)

                        for metric in metrics:
                            if metric in metrics_vals:
                                scores[f"{metric}_scores"].append(metrics_vals[metric])
            except:
                continue

        # 计算稳定性统计
        stability_report = {}
        for metric in metrics:
            metric_key = f"{metric}_scores"
            if metric_key in scores and len(scores[metric_key]) > 0:
                stability_report[metric] = {
                    'mean': np.mean(scores[metric_key]),
                    'std': np.std(scores[metric_key]),
                    'min': np.min(scores[metric_key]),
                    'max': np.max(scores[metric_key]),
                    'cv': np.std(scores[metric_key]) / (np.mean(scores[metric_key]) + 1e-10),  # 变异系数
                    'stable': np.std(scores[metric_key]) / (np.mean(scores[metric_key]) + 1e-10) < 0.5  # 稳定性判断
                }

        print("✅ 参数稳定性分析完成")
        return stability_report

    def run_comprehensive_optimization(self, data: pd.DataFrame,
                                     optimization_methods: List[str] = ['grid_search', 'bayesian', 'genetic'],
                                     param_configs: Dict = None) -> Dict[str, Any]:
        """
        运行综合优化

        Args:
            data: 输入数据
            optimization_methods: 优化方法列表
            param_configs: 参数配置
        """
        print("🚀 开始综合参数优化...")

        if param_configs is None:
            # 默认参数配置
            param_configs = {
                'grid_search': {
                    'param_grid': {
                        'ma_short': [5, 10, 15],
                        'ma_long': [20, 30, 40],
                        'rsi_period': [10, 14, 20],
                        'rsi_buy_threshold': [25, 30, 35],
                        'rsi_sell_threshold': [65, 70, 75]
                    }
                },
                'bayesian': {
                    'param_space': {
                        'ma_short': (3, 20),
                        'ma_long': (15, 50),
                        'rsi_period': (7, 30),
                        'lookback_window': (10, 40)
                    }
                },
                'genetic': {
                    'param_ranges': {
                        'ma_short': [3, 5, 10, 15, 20],
                        'ma_long': [15, 20, 30, 40, 50],
                        'rsi_period': [7, 14, 21, 28],
                        'factor_weight': [0.3, 0.4, 0.5, 0.6, 0.7],
                        'signal_threshold': [0.05, 0.1, 0.15, 0.2]
                    }
                }
            }

        results = {}

        for method in optimization_methods:
            print(f"\n🔄 执行 {method} 优化...")
            try:
                if method == 'grid_search' and 'param_grid' in param_configs[method]:
                    best_params, best_score = self.grid_search_optimization(
                        data, param_configs[method]['param_grid']
                    )
                elif method == 'bayesian' and 'param_space' in param_configs[method]:
                    best_params, best_score = self.bayesian_optimization(
                        data, param_configs[method]['param_space']
                    )
                elif method == 'genetic' and 'param_ranges' in param_configs[method]:
                    best_params, best_score = self.genetic_algorithm_optimization(
                        data, param_configs[method]['param_ranges']
                    )
                else:
                    print(f"⚠️ 未知的优化方法或配置: {method}")
                    continue

                results[method] = {
                    'best_params': best_params,
                    'best_score': best_score,
                    'optimization_method': method
                }

            except Exception as e:
                print(f"⚠️ {method} 优化失败: {e}")
                continue

        # 选择最佳结果
        if results:
            best_result_key = max(results.keys(), key=lambda k: results[k]['best_score'])
            overall_best = results[best_result_key]

            print(f"\n🏆 综合优化结果:")
            print(f"最佳方法: {overall_best['optimization_method']}")
            print(f"最佳参数: {overall_best['best_params']}")
            print(f"最佳得分: {overall_best['best_score']:.4f}")

            # 进行稳定性分析
            if 'best_params' in overall_best:
                stability = self.parameter_stability_analysis(data, overall_best['best_params'])
                results['stability_analysis'] = stability

        return results


if __name__ == "__main__":
    print("🧪 测试自动化调参模块...")

    # 创建示例数据
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)

    sample_data = pd.DataFrame({
        'high': prices * (1 + np.abs(np.random.randn(200)) * 0.02),
        'low': prices * (1 - np.abs(np.random.randn(200)) * 0.02),
        'open': prices + np.random.randn(200) * 0.1,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 200)
    }, index=dates)

    # 测试自动化调参
    tuner = AutomatedParameterTuning()

    print(f"\n📋 自动化调参器状态: 初始化完成")

    print("\n🎯 主要功能:")
    print("• 网格搜索优化")
    print("• 贝叶斯优化")
    print("• 遗传算法优化")
    print("• 多目标优化")
    print("• 参数稳定性分析")
    print("• 综合优化流程")

    print("\n💡 应用场景:")
    print("1. 策略参数寻优")
    print("2. 模型超参数调整")
    print("3. 组合参数优化")
    print("4. 稳健性验证")