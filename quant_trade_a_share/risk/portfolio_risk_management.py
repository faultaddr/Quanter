"""
风险管理模块
利用Qlib的风险模型加强投资组合管理
"""
import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import qlib
    from qlib.config import REG_CN as REGION_CN
    from qlib.data import D
    from qlib.riskmodel import RiskModel
    from qlib.portfolio import Portfolio
    from qlib.contrib.riskmodel import StructuredRM
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    print("⚠️ Qlib 未安装，将使用基础风险管理功能")

from quant_trade_a_share.models.model_fusion import ModelFusion
from quant_trade_a_share.factors.factor_library_expansion import FactorLibraryExpansion

class PortfolioRiskManagement:
    """
    投资组合风险管理类
    利用 Qlib 风险模型加强投资组合管理
    """

    def __init__(self, provider_uri="~/.qlib/qlib_data/cn_data"):
        """初始化风险管理器"""
        self.provider_uri = provider_uri
        self.risk_model = None
        self.initialized = False

        # 子模块
        self.model_fusion = ModelFusion()
        self.factor_library = FactorLibraryExpansion()

        if QLIB_AVAILABLE:
            try:
                qlib.init(provider_uri=self.provider_uri, region=REGION_CN)
                self.initialized = True
                print("✅ 风险管理系统初始化成功")
            except Exception as e:
                print(f"⚠️ Qlib 初始化失败: {e}")
                print("💡 提示: 安装 Qlib 并下载数据以启用完整风险功能")
        else:
            # Qlib not available, will use basic risk functionality
            pass

        # 风险参数设置
        self.risk_limits = {
            'max_position_size': 0.1,      # 最大单股占比 10%
            'max_sector_exposure': 0.3,    # 最大行业暴露 30%
            'max_beta': 1.2,              # 最大贝塔值
            'max_drawdown': 0.15,         # 最大回撤 15%
            'volatility_target': 0.2      # 波动率目标 20%
        }

        # 风险指标存储
        self.risk_metrics = {}
        self.portfolio_history = []

    def calculate_basic_risk_metrics(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """
        计算基础风险指标

        Args:
            returns: 投资组合收益率序列
            benchmark_returns: 基准收益率序列
        """
        if returns.empty:
            return {}

        metrics = {}

        # 收益率统计
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annual_return'] = (1 + returns.mean()) ** 252 - 1
        metrics['volatility'] = returns.std() * np.sqrt(252)  # 年化波动率

        # 夏普比率
        risk_free_rate = 0.03  # 假设无风险利率为 3%
        excess_return = metrics['annual_return'] - risk_free_rate
        metrics['sharpe_ratio'] = excess_return / metrics['volatility'] if metrics['volatility'] != 0 else 0

        # 最大回撤
        metrics['max_drawdown'] = self._calculate_max_drawdown(returns)

        # 胜率
        metrics['win_rate'] = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        # 波动率下行比率
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_dev = np.sqrt((negative_returns ** 2).mean())
            metrics['sortino_ratio'] = excess_return / (downside_dev * np.sqrt(252)) if downside_dev != 0 else 0
        else:
            metrics['sortino_ratio'] = float('inf')

        # Calmar比率（回撤比率）
        metrics['calmar_ratio'] = excess_return / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0

        # Alpha 和 Beta (如果有基准)
        if benchmark_returns is not None and not benchmark_returns.empty:
            # 对齐索引
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')

            # Beta
            if aligned_benchmark.var() != 0:
                metrics['beta'] = aligned_returns.cov(aligned_benchmark) / aligned_benchmark.var()
            else:
                metrics['beta'] = 0

            # Alpha
            expected_return = risk_free_rate + metrics['beta'] * (aligned_benchmark.mean() * 252 - risk_free_rate)
            metrics['alpha'] = metrics['annual_return'] - expected_return
        else:
            metrics['beta'] = 1.0  # 默认Beta为1.0
            metrics['alpha'] = 0.0

        return metrics

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        try:
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max
            return drawdowns.min()
        except:
            return 0.0

    def estimate_covariance_matrix(self, returns: pd.DataFrame, method: str = 'ledoit_wolf') -> pd.DataFrame:
        """
        估计协方差矩阵

        Args:
            returns: 资产收益率矩阵
            method: 估计方法 ('sample', 'ledoit_wolf', 'constant_correlation')
        """
        if returns.empty or returns.isna().all().all():
            return pd.DataFrame()

        try:
            if method == 'ledoit_wolf':
                # Ledoit-Wolf shrinkage estimator
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf()
                cov_matrix = lw.fit(returns.dropna()).covariance_
                return pd.DataFrame(cov_matrix,
                                  index=returns.columns,
                                  columns=returns.columns)

            elif method == 'constant_correlation':
                # Constant correlation model
                corrmatrix = returns.corr()
                stds = returns.std()
                cov_matrix = pd.DataFrame(index=returns.columns, columns=returns.columns)

                for i in returns.columns:
                    for j in returns.columns:
                        if i == j:
                            cov_matrix.loc[i, j] = stds[i]**2
                        else:
                            avg_corr = corrmatrix.values[corrmatrix.columns != i, corrmatrix.columns != j].mean()
                            cov_matrix.loc[i, j] = avg_corr * stds[i] * stds[j]

                return cov_matrix

            else:  # Sample covariance
                return returns.cov().fillna(0)

        except Exception as e:
            print(f"⚠️ 协方差矩阵估计失败: {e}")
            # 返回单位矩阵作为备选
            identity = np.eye(len(returns.columns))
            return pd.DataFrame(identity,
                              index=returns.columns,
                              columns=returns.columns)

    def calculate_portfolio_risk_contributions(self, weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
        """
        计算投资组合中各资产的风险贡献

        Args:
            weights: 资产权重
            returns: 资产收益率矩阵
        """
        if weights.empty or returns.empty:
            return pd.Series()

        try:
            # 计算协方差矩阵
            cov_matrix = self.estimate_covariance_matrix(returns)

            # 投资组合总体风险
            portfolio_variance = weights.dot(cov_matrix).dot(weights)
            portfolio_vol = np.sqrt(portfolio_variance)

            # 边际风险贡献
            marginal_contrib = (2 * cov_matrix.dot(weights)) / (2 * portfolio_vol)

            # 个体风险贡献
            risk_contributions = weights * marginal_contrib

            return risk_contributions

        except Exception as e:
            print(f"⚠️ 风险贡献计算失败: {e}")
            return pd.Series()

    def optimize_portfolio(self, returns: pd.DataFrame,
                         risk_model: str = 'min_variance',
                         constraints: Dict = None) -> pd.Series:
        """
        投资组合优化

        Args:
            returns: 资产收益率矩阵
            risk_model: 风险模型类型 ('min_variance', 'risk_parity', 'max_diversification')
            constraints: 约束条件
        """
        if returns.empty:
            return pd.Series()

        try:
            n_assets = len(returns.columns)

            # 设置默认约束
            if constraints is None:
                constraints = {
                    'min_weight': 0.0,      # 最小权重
                    'max_weight': 0.3,      # 最大权重
                    'long_only': True       # 只做多
                }

            # 计算期望收益率（使用历史平均）
            expected_returns = returns.mean() * 252

            # 计算协方差矩阵
            cov_matrix = self.estimate_covariance_matrix(returns)

            # 根据不同模型计算最优权重
            if risk_model == 'min_variance':
                weights = self._min_variance_optimization(cov_matrix, constraints)
            elif risk_model == 'risk_parity':
                weights = self._risk_parity_optimization(cov_matrix, constraints)
            elif risk_model == 'max_diversification':
                weights = self._max_diversification_optimization(cov_matrix, expected_returns, constraints)
            else:  # mean_variance
                weights = self._mean_variance_optimization(expected_returns, cov_matrix, constraints)

            # 标准化权重使总和为1
            if weights.sum() != 0:
                weights = weights / weights.sum()

            return weights

        except Exception as e:
            print(f"⚠️ 投资组合优化失败: {e}")
            # 返回等权重作为备选
            equal_weights = pd.Series(1.0/n_assets, index=returns.columns)
            return equal_weights

    def _min_variance_optimization(self, cov_matrix: pd.DataFrame, constraints: Dict) -> pd.Series:
        """最小方差优化"""
        try:
            from scipy.optimize import minimize

            n = len(cov_matrix)
            def objective(w):
                return w.T @ cov_matrix @ w

            constraints_list = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # 权重和为1
            ]

            bounds = [(constraints.get('min_weight', 0), constraints.get('max_weight', 1)) for _ in range(n)]

            if constraints.get('long_only', True):
                bounds = [(max(0, b[0]), min(1, b[1])) for b in bounds]

            # 初始权重（等权）
            x0 = np.array([1/n] * n)

            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)

            if result.success:
                return pd.Series(result.x, index=cov_matrix.columns)
            else:
                print("⚠️ 优化未成功，使用等权重")
                return pd.Series([1/n] * n, index=cov_matrix.columns)

        except:
            # 如果scipy不可用，返回等权重
            n = len(cov_matrix)
            return pd.Series([1/n] * n, index=cov_matrix.columns)

    def _risk_parity_optimization(self, cov_matrix: pd.DataFrame, constraints: Dict) -> pd.Series:
        """风险平价优化"""
        try:
            # 简化的风险平价算法
            # 计算每个资产的波动率作为初始估计
            volatilities = np.sqrt(np.diag(cov_matrix))

            # 使用逆波动率加权（简化版风险平价）
            weights = 1 / volatilities
            weights = weights / weights.sum()

            # 应用约束
            weights = np.clip(weights, constraints.get('min_weight', 0), constraints.get('max_weight', 1))
            weights = weights / weights.sum()  # 重新标准化

            return pd.Series(weights, index=cov_matrix.columns)

        except:
            # 如果计算失败，返回等权重
            n = len(cov_matrix)
            return pd.Series([1/n] * n, index=cov_matrix.columns)

    def _max_diversification_optimization(self, cov_matrix: pd.DataFrame, expected_returns: pd.Series, constraints: Dict) -> pd.Series:
        """最大分散化优化"""
        try:
            # 最大分散化比率 = 投资组合波动率 / 各资产权重*波动率之和
            volatilities = np.sqrt(np.diag(cov_matrix))

            # 初始权重估计（等权）
            n = len(cov_matrix)
            weights = np.array([1/n] * n)

            # 计算相关系数矩阵
            correlation = cov_matrix / np.outer(volatilities, volatilities)

            # 使用启发式方法：最大化分散化比率
            for _ in range(100):  # 迭代优化
                # 当前权重下的分散化比率
                portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
                weighted_vols = np.sum(weights * volatilities)
                diversification_ratio = portfolio_vol / weighted_vols if weighted_vols != 0 else 0

                # 梯度上升更新（简化版）
                grad = (weights * volatilities) / weighted_vols - (cov_matrix @ weights) / portfolio_vol
                weights = weights + 0.01 * grad  # 学习率0.01

                # 应用约束并重新标准化
                weights = np.clip(weights, constraints.get('min_weight', 0), constraints.get('max_weight', 1))
                weights = np.maximum(weights, 0)  # 确保非负
                weights = weights / weights.sum() if weights.sum() != 0 else np.array([1/n] * n)

            return pd.Series(weights, index=cov_matrix.columns)

        except:
            # 如果计算失败，返回等权重
            n = len(cov_matrix)
            return pd.Series([1/n] * n, index=cov_matrix.columns)

    def _mean_variance_optimization(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, constraints: Dict) -> pd.Series:
        """均值-方差优化"""
        try:
            from scipy.optimize import minimize

            n = len(expected_returns)
            target_return = expected_returns.mean()  # 目标收益率设为平均值

            def objective(w):
                return w.T @ cov_matrix @ w  # 最小化风险（方差）

            constraints_list = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # 权重和为1
                {'type': 'eq', 'fun': lambda w: w.T @ expected_returns - target_return}  # 目标收益率
            ]

            bounds = [(constraints.get('min_weight', 0), constraints.get('max_weight', 1)) for _ in range(n)]

            if constraints.get('long_only', True):
                bounds = [(max(0, b[0]), min(1, b[1])) for b in bounds]

            # 初始权重（等权）
            x0 = np.array([1/n] * n)

            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)

            if result.success:
                return pd.Series(result.x, index=expected_returns.index)
            else:
                print("⚠️ 优化未成功，使用等权重")
                return pd.Series([1/n] * n, index=expected_returns.index)

        except:
            # 如果scipy不可用，返回等权重
            n = len(expected_returns)
            return pd.Series([1/n] * n, index=expected_returns.index)

    def monitor_position_risk(self, positions: Dict[str, float], current_prices: Dict[str, float],
                            threshold_alerts: bool = True) -> Dict[str, Any]:
        """
        监控头寸风险

        Args:
            positions: 持仓字典 {symbol: quantity}
            current_prices: 当前价格字典 {symbol: price}
            threshold_alerts: 是否启用阈值警报
        """
        if not positions or not current_prices:
            return {}

        risk_report = {}

        # 计算市值
        market_values = {}
        total_value = 0
        for symbol, qty in positions.items():
            if symbol in current_prices:
                value = qty * current_prices[symbol]
                market_values[symbol] = value
                total_value += value

        risk_report['total_portfolio_value'] = total_value
        risk_report['position_sizes'] = {}
        risk_report['alerts'] = []

        if total_value > 0:
            # 计算各头寸占比和风险
            for symbol, mv in market_values.items():
                pct_of_portfolio = mv / total_value
                risk_report['position_sizes'][symbol] = {
                    'market_value': mv,
                    'percentage_of_portfolio': pct_of_portfolio
                }

                # 检查阈值警报
                if threshold_alerts:
                    if pct_of_portfolio > self.risk_limits['max_position_size']:
                        risk_report['alerts'].append({
                            'type': 'POSITION_SIZE_EXCEEDED',
                            'symbol': symbol,
                            'current': f"{pct_of_portfolio:.2%}",
                            'limit': f"{self.risk_limits['max_position_size']:.2%}",
                            'severity': 'HIGH'
                        })

        return risk_report

    def simulate_portfolio_scenario(self, returns: pd.DataFrame, scenario: str = 'stress',
                                  severity: float = 1.0) -> Dict[str, Any]:
        """
        情景分析

        Args:
            returns: 历史收益率数据
            scenario: 情景类型 ('stress', 'normal', 'bull', 'bear')
            severity: 严重程度 (0-1)
        """
        if returns.empty:
            return {}

        scenarios = {
            'stress': {
                'return_multiplier': -1.5,
                'volatility_multiplier': 2.0,
                'correlation_shift': 0.3
            },
            'normal': {
                'return_multiplier': 1.0,
                'volatility_multiplier': 1.0,
                'correlation_shift': 0.0
            },
            'bull': {
                'return_multiplier': 1.5,
                'volatility_multiplier': 0.8,
                'correlation_shift': -0.1
            },
            'bear': {
                'return_multiplier': -1.0,
                'volatility_multiplier': 1.5,
                'correlation_shift': 0.2
            }
        }

        if scenario not in scenarios:
            scenario = 'normal'

        scenario_params = scenarios[scenario]

        # 调整收益率
        adj_returns = returns * scenario_params['return_multiplier'] * severity
        adj_returns = adj_returns * scenario_params['volatility_multiplier'] * severity

        # 调整相关性
        # (简化处理，实际情况需要更复杂的协方差矩阵调整)

        # 计算调整后的风险指标
        scenario_metrics = self.calculate_basic_risk_metrics(adj_returns.mean(axis=1))  # 简化为等权投资组合

        return {
            'scenario': scenario,
            'severity': severity,
            'adjusted_returns': adj_returns,
            'risk_metrics': scenario_metrics,
            'params_used': scenario_params
        }

    def apply_risk_adjustment(self, signals: pd.Series, risk_metrics: Dict[str, float],
                            adjustment_method: str = 'volatility_scaling') -> pd.Series:
        """
        应用风险调整

        Args:
            signals: 原始信号
            risk_metrics: 风险指标
            adjustment_method: 调整方法
        """
        if signals.empty or not risk_metrics:
            return signals

        adjusted_signals = signals.copy()

        if adjustment_method == 'volatility_scaling':
            # 波动率缩放
            current_vol = risk_metrics.get('volatility', 0.2)  # 默认20%年化波动率
            target_vol = self.risk_limits['volatility_target']

            if current_vol > 0:
                scaling_factor = target_vol / current_vol
                # 缩放信号强度
                adjusted_signals = adjusted_signals * min(scaling_factor, 1.0)  # 不增加风险，只减少

        elif adjustment_method == 'drawdown_control':
            # 回撤控制
            current_drawdown = risk_metrics.get('max_drawdown', 0)
            max_allowed_drawdown = self.risk_limits['max_drawdown']

            if abs(current_drawdown) > max_allowed_drawdown:
                # 如果超过最大回撤限制，降低信号强度
                reduction_factor = max(0, 1 - (abs(current_drawdown) - max_allowed_drawdown) / max_allowed_drawdown)
                adjusted_signals = adjusted_signals * reduction_factor

        elif adjustment_method == 'beta_adjustment':
            # Beta调整
            current_beta = risk_metrics.get('beta', 1.0)
            max_beta = self.risk_limits['max_beta']

            if current_beta > max_beta:
                # 如果Beta过高，降低信号强度
                reduction_factor = max(0, max_beta / current_beta)
                adjusted_signals = adjusted_signals * reduction_factor

        return adjusted_signals

    def generate_risk_report(self, portfolio_data: Dict[str, Any]) -> str:
        """
        生成风险报告

        Args:
            portfolio_data: 投资组合数据
        """
        report = []
        report.append("="*60)
        report.append("投资组合风险管理报告")
        report.append("="*60)

        if 'risk_metrics' in portfolio_data:
            metrics = portfolio_data['risk_metrics']
            report.append(f"总收益: {metrics.get('total_return', 0):.2%}")
            report.append(f"年化收益: {metrics.get('annual_return', 0):.2%}")
            report.append(f"波动率: {metrics.get('volatility', 0):.2%}")
            report.append(f"夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
            report.append(f"最大回撤: {metrics.get('max_drawdown', 0):.2%}")
            report.append(f"Beta: {metrics.get('beta', 0):.3f}")
            report.append(f"Alpha: {metrics.get('alpha', 0):.3f}")
            report.append(f"胜率: {metrics.get('win_rate', 0):.2%}")
            report.append(f"Sortino比率: {metrics.get('sortino_ratio', 0):.3f}")
            report.append(f"Calmar比率: {metrics.get('calmar_ratio', 0):.3f}")

        if 'position_sizes' in portfolio_data:
            report.append("\n头寸规模:")
            for symbol, pos_info in portfolio_data['position_sizes'].items():
                report.append(f"  {symbol}: {pos_info['percentage_of_portfolio']:.2%} ({pos_info['market_value']:.2f}元)")

        if 'alerts' in portfolio_data and portfolio_data['alerts']:
            report.append("\n风险警报:")
            for alert in portfolio_data['alerts']:
                report.append(f"  [{alert['severity']}] {alert['type']}: {alert['symbol']} - {alert['current']} (限制: {alert['limit']})")

        report.append("="*60)
        return "\n".join(report)


if __name__ == "__main__":
    print("🧪 测试风险管理模块...")

    # 创建示例数据
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    # 创建多资产收益率数据
    n_assets = 5
    asset_names = [f'STOCK_{i}' for i in range(n_assets)]

    returns_data = pd.DataFrame(
        np.random.randn(100, n_assets) * 0.02,  # 2%日波动率
        index=dates,
        columns=asset_names
    )

    # 测试风险管理
    risk_manager = PortfolioRiskManagement()

    print(f"\n📋 风险管理系统状态: {'可用' if risk_manager.initialized else '不可用'}")

    print("\n🎯 主要功能:")
    print("• 基础风险指标计算")
    print("• 协方差矩阵估计")
    print("• 投资组合优化")
    print("• 风险贡献分析")
    print("• 头寸监控")
    print("• 情景分析")
    print("• 风险调整")
    print("• 风险报告生成")

    print("\n💡 应用场景:")
    print("1. 投资组合构建")
    print("2. 风险控制")
    print("3. 绩效归因")
    print("4. 监管合规")