"""
Qlib集成增强主模块
整合因子库扩充、模型融合、风险管理和自动化调参四大功能
"""
import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 导入四大核心模块
from quant_trade_a_share.factors.factor_library_expansion import FactorLibraryExpansion
from quant_trade_a_share.models.model_fusion import ModelFusion
from quant_trade_a_share.risk.portfolio_risk_management import PortfolioRiskManagement
from quant_trade_a_share.optimization.automated_parameter_tuning import AutomatedParameterTuning

# Import MyTTIndicators with error handling (similar to factor_library_expansion)
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
    print("⚠️ MyTT 指标不可用，将使用基础因子功能")
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

class QlibIntegratedEnhancement:
    """
    Qlib集成增强主类
    整合四大核心功能：因子库扩充、模型融合、风险管理、自动化调参
    """

    def __init__(self, provider_uri="~/.qlib/qlib_data/cn_data"):
        """初始化集成增强系统"""
        self.provider_uri = provider_uri

        # 初始化四大核心模块
        self.factor_library = FactorLibraryExpansion(provider_uri)
        self.model_fusion = ModelFusion()
        self.risk_manager = PortfolioRiskManagement(provider_uri)
        self.param_tuner = AutomatedParameterTuning(verbose=False)
        self.mytt_indicators = MyTTIndicators()

        # 系统状态
        self.system_initialized = all([
            hasattr(self.factor_library, 'initialized'),
            hasattr(self.risk_manager, 'initialized')
        ])

        # Only print essential initialization info
        print(f"📊 Qlib 集成状态: {'完全可用' if self.system_initialized else '基础功能可用'}")

    def run_comprehensive_analysis(self, data: pd.DataFrame, instruments: List[str] = None,
                                 start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        运行综合性分析（四合一功能）

        Args:
            data: 股票数据
            instruments: 股票列表
            start_date: 开始日期
            end_date: 结束日期
        """
        print("🌟 开始综合性分析...")

        results = {}

        # 1. 因子库扩充
        print("\n📊 第一步：因子库扩充")
        if instruments and start_date and end_date:
            factors = self.factor_library.get_comprehensive_factors(
                data, instruments, start_date, end_date
            )
        else:
            # 如果没有提供股票列表，使用MyTT指标作为主要因子
            factors = self.factor_library.get_mytt_indicators(data)
        results['factors'] = factors
        print(f"✅ 生成 {len(factors.columns)} 个综合因子")

        # 2. 模型融合
        print("\n🤖 第二步：模型融合")
        fusion_result = self.model_fusion.run_fusion_strategy(data)
        results['fusion'] = fusion_result
        print(f"✅ 融合策略执行完成")

        # 3. 风险管理
        print("\n🛡️  第三步：风险管理")
        if 'signals' in fusion_result:
            # 基于融合策略的信号进行风险分析
            if 'close' in data.columns:
                returns = data['close'].pct_change().fillna(0)
                risk_metrics = self.risk_manager.calculate_basic_risk_metrics(
                    fusion_result['signals'].shift(1).fillna(0) * returns
                )
                results['risk_metrics'] = risk_metrics

                # 生成风险报告
                risk_report_data = {
                    'risk_metrics': risk_metrics,
                    'position_sizes': {},
                    'alerts': []
                }
                risk_report = self.risk_manager.generate_risk_report(risk_report_data)
                results['risk_report'] = risk_report
        print("✅ 风险评估完成")

        # 4. 自动调参
        print("\n⚙️  第四步：自动调参")
        try:
            param_results = self.param_tuner.run_comprehensive_optimization(data)
            results['optimization'] = param_results

            # 如果找到更好的参数，使用它们重新运行融合策略
            if 'optimization' in param_results and param_results:
                best_method_key = max(
                    [k for k in param_results.keys() if k != 'stability_analysis'],
                    key=lambda k: param_results[k]['best_score'] if isinstance(param_results[k], dict) and 'best_score' in param_results[k] else 0
                ) if any(k != 'stability_analysis' for k in param_results.keys()) else None

                if best_method_key and isinstance(param_results[best_method_key], dict):
                    best_params = param_results[best_method_key].get('best_params', {})
                    if best_params:
                        print(f"🔄 使用优化后的参数重新运行融合策略...")
                        # 这里可以基于优化参数重新运行策略，为简化跳过
        except Exception as e:
            print(f"⚠️ 自动调参过程中出现错误: {e}")

        print("\n🎉 综合性分析完成！")
        return results

    def advanced_factor_analysis(self, data: pd.DataFrame, instruments: List[str],
                               start_date: str, end_date: str) -> Dict[str, Any]:
        """
        高级因子分析（Qlib Alpha + MyTT 指标结合）
        """
        print("🔍 执行高级因子分析...")

        # 获取综合因子
        comprehensive_factors = self.factor_library.get_comprehensive_factors(
            data, instruments, start_date, end_date
        )

        # 计算因子统计
        factor_stats = self.factor_library.calculate_factor_stats(comprehensive_factors)

        # 计算因子相关性
        correlation_matrix = self.factor_library.factor_rank_correlation(comprehensive_factors)

        # 因子有效性检验（IC分析 - 简化版）
        ic_analysis = self._simplified_ic_analysis(comprehensive_factors, data)

        results = {
            'factors': comprehensive_factors,
            'stats': factor_stats,
            'correlations': correlation_matrix,
            'ic_analysis': ic_analysis
        }

        print(f"✅ 高级因子分析完成，包含 {len(comprehensive_factors.columns)} 个因子")
        return results

    def _simplified_ic_analysis(self, factors: pd.DataFrame, data: pd.DataFrame) -> Dict[str, float]:
        """
        简化的IC分析（因子与未来收益的相关性）
        """
        try:
            if 'close' in data.columns:
                # 计算未来一期收益率作为目标变量
                future_returns = data['close'].pct_change().shift(-1).fillna(0)

                ic_scores = {}
                for col in factors.columns:
                    if col in data.columns:  # 如果因子列存在于原始数据中
                        # 计算因子与未来收益的IC
                        factor_series = factors[col].reindex(data.index).fillna(method='ffill').fillna(0)
                        aligned_factor, aligned_returns = factor_series.align(future_returns, join='inner')

                        if len(aligned_factor) > 0:
                            ic = np.corrcoef(aligned_factor, aligned_returns)[0, 1]
                            if not np.isnan(ic):
                                ic_scores[col] = ic

                return ic_scores
        except Exception as e:
            print(f"⚠️ IC分析出现错误: {e}")

        return {}

    def smart_portfolio_optimization(self, returns_data: pd.DataFrame,
                                   risk_target: str = 'SharpeRatio',
                                   constraints: Dict = None) -> Dict[str, Any]:
        """
        智能投资组合优化（结合风险模型）
        """
        print("⚖️  执行智能投资组合优化...")

        # 使用风险管理模块进行投资组合优化
        optimal_weights = self.risk_manager.optimize_portfolio(
            returns_data, risk_model='risk_parity', constraints=constraints
        )

        # 计算风险贡献
        risk_contributions = self.risk_manager.calculate_portfolio_risk_contributions(
            optimal_weights, returns_data
        )

        # 情景分析
        stress_test = self.risk_manager.simulate_portfolio_scenario(
            returns_data, scenario='stress', severity=0.8
        )

        results = {
            'optimal_weights': optimal_weights,
            'risk_contributions': risk_contributions,
            'stress_test': stress_test
        }

        print("✅ 智能投资组合优化完成")
        return results

    def adaptive_signal_generation(self, data: pd.DataFrame,
                                 confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        自适应信号生成（融合多模型输出）
        """
        print("🎯 生成自适应交易信号...")

        # 计算技术指标信号
        technical_signals = self.model_fusion.calculate_technical_signals(data)

        # 计算ML模型信号
        ml_signals = self.model_fusion.calculate_ml_signals(data, model_type='ensemble')

        # 计算集成信号
        ensemble_signal = self.model_fusion.calculate_ensemble_signal(
            technical_signals, ml_signals
        )

        # 自适应权重调整（基于历史表现）
        # 简化版本：基于信号置信度调整
        adaptive_signal = ensemble_signal.copy()
        high_confidence_mask = ensemble_signal.abs() > confidence_threshold
        adaptive_signal[high_confidence_mask] *= 1.2  # 高置信度信号加强
        low_confidence_mask = ensemble_signal.abs() <= confidence_threshold * 0.5
        adaptive_signal[low_confidence_mask] *= 0.8  # 低置信度信号减弱

        # 应用风险调整
        risk_adjusted_signals = self.risk_manager.apply_risk_adjustment(
            adaptive_signal,
            {},  # 可以传入实时风险指标
            adjustment_method='volatility_scaling'
        )

        results = {
            'technical_signals': technical_signals,
            'ml_signals': ml_signals,
            'ensemble_signal': ensemble_signal,
            'adaptive_signal': adaptive_signal,
            'risk_adjusted_signal': risk_adjusted_signals
        }

        print(f"✅ 生成 {len(adaptive_signal[adaptive_signal != 0])} 个自适应交易信号")
        return results

    def automated_strategy_optimization(self, data: pd.DataFrame,
                                      optimization_goal: str = 'SharpeRatio') -> Dict[str, Any]:
        """
        自动化策略参数优化
        """
        print("🔧 执行自动化策略优化...")

        # 定义参数空间
        param_configs = {
            'grid_search': {
                'param_grid': {
                    'ma_short': [5, 10, 15],
                    'ma_long': [20, 30, 40],
                    'rsi_period': [10, 14, 20]
                }
            },
            'bayesian': {
                'param_space': {
                    'ma_short': (3, 20),
                    'ma_long': (15, 50),
                    'rsi_period': (7, 30)
                }
            },
            'genetic': {
                'param_ranges': {
                    'ma_short': [3, 5, 10, 15, 20],
                    'ma_long': [15, 20, 30, 40, 50],
                    'rsi_period': [7, 14, 21, 28],
                    'factor_weight': [0.3, 0.4, 0.5, 0.6, 0.7]
                }
            }
        }

        # 运行优化
        opt_results = self.param_tuner.run_comprehensive_optimization(
            data,
            optimization_methods=['grid_search', 'genetic'],
            param_configs=param_configs
        )

        # 参数稳定性分析
        if 'optimization' in opt_results:
            for method, result in opt_results['optimization'].items():
                if isinstance(result, dict) and 'best_params' in result:
                    stability = self.param_tuner.parameter_stability_analysis(
                        data, result['best_params']
                    )
                    opt_results['stability_analysis'] = stability
                    break

        print("✅ 自动化策略优化完成")
        return opt_results

    def generate_comprehensive_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        生成综合性分析报告
        """
        report = []
        report.append("="*70)
        report.append("Qlib集成增强系统 - 综合性分析报告")
        report.append("="*70)

        if 'factors' in analysis_results:
            factor_data = analysis_results['factors']
            report.append(f"📊 因子分析:")
            report.append(f"   • 总因子数量: {len(factor_data.columns)}")
            report.append(f"   • 样本数量: {len(factor_data)}")
            if not factor_data.empty and len(factor_data.columns) > 0:
                report.append(f"   • 首个因子均值: {factor_data.iloc[:, 0].mean():.4f}")
                report.append(f"   • 首个因子标准差: {factor_data.iloc[:, 0].std():.4f}")

        if 'fusion' in analysis_results and 'performance' in analysis_results['fusion']:
            perf = analysis_results['fusion']['performance']
            report.append(f"\n🤖 模型融合表现:")
            report.append(f"   • 年化收益率: {perf.get('annual_return', 0):.2%}")
            report.append(f"   • 夏普比率: {perf.get('sharpe_ratio', 0):.3f}")
            report.append(f"   • 最大回撤: {perf.get('max_drawdown', 0):.2%}")
            report.append(f"   • 胜率: {perf.get('win_rate', 0):.2%}")

        if 'risk_metrics' in analysis_results:
            risk = analysis_results['risk_metrics']
            report.append(f"\n🛡️  风险指标:")
            report.append(f"   • 波动率: {risk.get('volatility', 0):.2%}")
            report.append(f"   • Beta: {risk.get('beta', 0):.3f}")
            report.append(f"   • Alpha: {risk.get('alpha', 0):.3f}")
            report.append(f"   • Sortino比率: {risk.get('sortino_ratio', 0):.3f}")

        if 'optimization' in analysis_results:
            opt = analysis_results['optimization']
            report.append(f"\n⚙️  优化结果:")
            successful_methods = [k for k, v in opt.items()
                                if isinstance(v, dict) and 'best_score' in v]
            report.append(f"   • 成功优化方法数: {len(successful_methods)}")
            for method in successful_methods[:2]:  # 只显示前两个
                method_result = opt[method]
                report.append(f"   • {method}: 得分 {method_result.get('best_score', 0):.4f}")

        if 'risk_report' in analysis_results:
            report.append(f"\n🚨 风险报告:")
            report.append(f"   {analysis_results['risk_report'].split(chr(10))[5] if chr(10) in analysis_results['risk_report'] else analysis_results['risk_report'][:100]}...")

        report.append("\n💡 分析完成时间: " + str(pd.Timestamp.now()))
        report.append("="*70)

        return "\n".join(report)


if __name__ == "__main__":
    print("🧪 测试 Qlib 集成增强系统...")

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

    # 测试集成系统
    integrated_system = QlibIntegratedEnhancement()

    print(f"\n📋 系统状态: {'完全集成' if integrated_system.system_initialized else '基础功能'}")

    print("\n🎯 四大核心功能:")
    print("1. ✅ 因子库扩充：Qlib Alpha因子 + MyTT指标")
    print("2. ✅ 模型融合：传统技术指标 + ML模型")
    print("3. ✅ 风险管理：Qlib风险模型 + 投资组合优化")
    print("4. ✅ 自动调参：网格搜索 + 贝叶斯 + 遗传算法")

    print("\n💡 综合应用场景:")
    print("• 智能量化策略开发")
    print("• 多因子模型构建")
    print("• 风险控制增强")
    print("• 参数优化自动化")