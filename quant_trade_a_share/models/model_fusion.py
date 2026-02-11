"""
模型融合模块
将传统技术指标策略与ML模型预测相结合
"""
import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Callable
import warnings
warnings.filterwarnings('ignore')

try:
    import qlib
    from qlib.config import REG_CN as REGION_CN
    from qlib.contrib.model.gbdt import GBDT
    from qlib.contrib.model.linear import LinearModel
    from qlib.contrib.model.rnn import RNNModel
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
    QLIB_SKLEARN_AVAILABLE = True
except ImportError:
    QLIB_SKLEARN_AVAILABLE = False
    print("⚠️ Qlib 或 scikit-learn 未安装，将使用基础融合功能")

# 导入策略工具 and handle MyTTIndicators with error handling
from quant_trade_a_share.strategies.strategy_tools import Strategy, StrategyManager

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

from quant_trade_a_share.integration.deep_qlib_integration import DeepQlibIntegration

class ModelFusion:
    """
    模型融合类
    将传统技术指标策略与ML模型预测相结合
    """

    def __init__(self):
        """初始化模型融合器"""
        self.traditional_strategies = StrategyManager()
        self.mytt_indicators = MyTTIndicators()
        self.deep_qlib = DeepQlibIntegration() if QLIB_SKLEARN_AVAILABLE else None
        self.ml_models = {}
        self.weights = {}  # 模型权重
        self.performance_history = {}  # 模型历史表现

        print("✅ 模型融合器初始化完成")

    def calculate_technical_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算传统技术指标信号

        Args:
            data: 股票数据
        """
        # 确保数据索引唯一，如果有重复则重置索引但保留原始索引作为列
        if data.index.duplicated().any():
            original_index_name = data.index.name
            data = data.reset_index()
            data = data.set_index(data.columns[0])  # 使用第一列作为新索引
            data.index.name = original_index_name

        signals = {}

        # MA 交叉策略信号
        ma5_values = self.mytt_indicators.MA(data['close'], 5)
        ma20_values = self.mytt_indicators.MA(data['close'], 20)
        ma5 = pd.Series(ma5_values, index=data.index)
        ma20 = pd.Series(ma20_values, index=data.index)
        ma_cross_signal = pd.Series(0, index=data.index)
        ma_cross_signal[(ma5 > ma20) & (ma5.shift(1) <= ma20.shift(1))] = 1  # 金叉买入
        ma_cross_signal[(ma5 < ma20) & (ma5.shift(1) >= ma20.shift(1))] = -1  # 死叉卖出
        signals['MA_CROSS'] = ma_cross_signal

        # MACD 策略信号
        dif_values, dea_values, bar = self.mytt_indicators.MACD(data['close'])
        dif = pd.Series(dif_values, index=data.index) if not isinstance(dif_values, pd.Series) else dif_values
        dea = pd.Series(dea_values, index=data.index) if not isinstance(dea_values, pd.Series) else dea_values
        bar = pd.Series(bar, index=data.index) if not isinstance(bar, pd.Series) else bar
        macd_signal = pd.Series(0, index=data.index)
        macd_signal[(dif > dea) & (dif.shift(1) <= dea.shift(1))] = 1  # 金叉买入
        macd_signal[(dif < dea) & (dif.shift(1) >= dea.shift(1))] = -1  # 死叉卖出
        signals['MACD_CROSS'] = macd_signal

        # KDJ 策略信号
        k_values, d_values, j_values = self.mytt_indicators.KDJ(data['high'], data['low'], data['close'])
        k = pd.Series(k_values, index=data.index)
        d = pd.Series(d_values, index=data.index)
        kdj_signal = pd.Series(0, index=data.index)
        kdj_signal[(k > d) & (k < 20) & (k.shift(1) >= d.shift(1))] = 1  # 超卖买入
        kdj_signal[(k < d) & (k > 80) & (k.shift(1) <= d.shift(1))] = -1  # 超卖卖出
        signals['KDJ_SIGNAL'] = kdj_signal

        # RSI 策略信号
        rsi_14_values = self.mytt_indicators.RSI(data['close'], 14)
        rsi_14 = pd.Series(rsi_14_values, index=data.index)
        rsi_signal = pd.Series(0, index=data.index)
        rsi_signal[(rsi_14 < 30) & (rsi_14.shift(1) >= 30)] = 1  # 超卖买入
        rsi_signal[(rsi_14 > 70) & (rsi_14.shift(1) <= 70)] = -1  # 超买卖出
        signals['RSI_SIGNAL'] = rsi_signal

        # BOLL 策略信号
        upper_values, middle_values, lower_values = self.mytt_indicators.BOLL(data['close'])
        upper = pd.Series(upper_values, index=data.index)
        middle = pd.Series(middle_values, index=data.index)
        lower = pd.Series(lower_values, index=data.index)
        boll_signal = pd.Series(0, index=data.index)
        boll_signal[(data['close'] <= lower) & (data['close'].shift(1) > lower)] = 1  # 触底反弹
        boll_signal[(data['close'] >= upper) & (data['close'].shift(1) < upper)] = -1  # 触顶回落
        signals['BOLL_SIGNAL'] = boll_signal

        # CCI 策略信号
        cci_values = self.mytt_indicators.CCI(data['high'], data['low'], data['close'])
        cci = pd.Series(cci_values, index=data.index)
        cci_signal = pd.Series(0, index=data.index)
        cci_signal[(cci < -100) & (cci.shift(1) >= -100)] = 1  # 超卖买入
        cci_signal[(cci > 100) & (cci.shift(1) <= 100)] = -1  # 超买卖出
        signals['CCI_SIGNAL'] = cci_signal

        print(f"✅ 计算完成 {len(signals)} 种技术指标信号")
        return signals

    def calculate_ml_signals(self, data: pd.DataFrame, model_type: str = 'ensemble') -> pd.Series:
        """
        计算机器学习模型信号

        Args:
            data: 股票数据
            model_type: 模型类型
        """
        if self.deep_qlib:
            try:
                print(f"🤖 使用 {model_type} ML 模型计算信号...")
                ml_signals = self.deep_qlib.get_ml_signals(data, method=model_type)
                if not ml_signals.empty:
                    print(f"✅ ML 模型生成 {len(ml_signals[ml_signals != 0])} 个信号")
                    return ml_signals
                else:
                    print("⚠️ ML 模型未返回有效信号，使用基础计算")
            except Exception as e:
                print(f"⚠️ ML 信号计算失败: {e}")

        # 基础 ML 信号计算（如果 deep_qlib 不可用或出错）
        return self._basic_ml_signals(data)

    def _basic_ml_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        基础机器学习信号计算
        """
        # 确保数据索引唯一：删除重复索引保留第一次出现
        if data.index.duplicated().any():
            print("🔍 检测到重复索引，正在处理...")
            original_length = len(data)
            data = data[~data.index.duplicated(keep='first')]
            print(f"✅ 从 {original_length} 行清理到 {len(data)} 行")

        signals = pd.Series(0.0, index=data.index)

        try:
            # 创建基础特征
            features = pd.DataFrame(index=data.index)
            features['close_lag1'] = data['close'].shift(1)
            features['pct_chg'] = data['close'].pct_change()
            features['volume_lag1'] = data['volume'].shift(1)
            features['volume_pct_chg'] = data['volume'].pct_change()

            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

            # 防止除零和重复索引
            rs = gain / (loss + 1e-10)  # 添加小值防止除零
            rsi = 100 - (100 / (1 + rs))

            # 确保rsi与data索引一致（这里可能是关键问题所在）
            if not rsi.index.equals(data.index):
                rsi = rsi.reindex(data.index)
            features['rsi'] = rsi

            # MACD
            exp12 = data['close'].ewm(span=12).mean()
            exp26 = data['close'].ewm(span=26).mean()
            macd = exp12 - exp26
            signal_line = macd.ewm(span=9).mean()

            # 确保MACD相关指标与data索引一致
            if not macd.index.equals(data.index):
                macd = macd.reindex(data.index)
            if not signal_line.index.equals(data.index):
                signal_line = signal_line.reindex(data.index)
            features['macd'] = macd
            features['macd_signal'] = signal_line

            # 删除含 NaN 的行
            features = features.dropna()

            if len(features) > 10:  # 需要有足够的数据点
                # 基于 RSI 的 ML 信号
                rsi_values = features['rsi']
                rsi_sig = pd.Series(0.0, index=features.index)

                # 确保索引一致后再进行shift操作
                rsi_shifted = rsi_values.shift(1)
                if not rsi_shifted.index.equals(rsi_values.index):
                    rsi_shifted = rsi_shifted.reindex(rsi_values.index)

                rsi_sig[(rsi_values < 30) & (rsi_shifted >= 30)] = 0.8  # 超卖
                rsi_sig[(rsi_values > 70) & (rsi_shifted <= 70)] = -0.8  # 超买
                signals.loc[features.index] += rsi_sig

                # 基于 MACD 的 ML 信号
                macd_values = features['macd']
                macd_signal_values = features['macd_signal']
                macd_sig = pd.Series(0.0, index=features.index)

                # 确保索引一致后再进行shift操作
                macd_shifted = macd_values.shift(1)
                macd_signal_shifted = macd_signal_values.shift(1)

                if not macd_shifted.index.equals(macd_values.index):
                    macd_shifted = macd_shifted.reindex(macd_values.index)
                if not macd_signal_shifted.index.equals(macd_signal_values.index):
                    macd_signal_shifted = macd_signal_shifted.reindex(macd_signal_values.index)

                macd_sig[(macd_values > macd_signal_values) &
                        (macd_shifted <= macd_signal_shifted)] = 0.6  # 金叉
                macd_sig[(macd_values < macd_signal_values) &
                        (macd_shifted >= macd_signal_shifted)] = -0.6  # 死叉
                signals.loc[features.index] += macd_sig

                # 趋势信号
                trend_sig = pd.Series(0.0, index=features.index)
                trend_sig[features['pct_chg'] > 0.02] = 0.4  # 上涨趋势
                trend_sig[features['pct_chg'] < -0.02] = -0.4  # 下跌趋势
                signals.loc[features.index] += trend_sig

        except Exception as e:
            print(f"⚠️ 基础 ML 信号计算失败: {e}")
            import traceback
            traceback.print_exc()

        return signals
    def train_ml_model(self, data: pd.DataFrame, target_col: str = 'future_return', model_name: str = 'default'):
        """
        训练机器学习模型

        Args:
            data: 训练数据
            target_col: 目标列
            model_name: 模型名称
        """
        if not QLIB_SKLEARN_AVAILABLE:
            print("❌ scikit-learn 不可用，无法训练 ML 模型")
            return None

        try:
            # 准备特征和目标
            feature_cols = [col for col in data.columns if col != target_col and not col.startswith('target')]

            if target_col not in data.columns:
                # 如果没有目标列，创建未来收益率作为目标
                data['future_return'] = data['close'].pct_change().shift(-1).fillna(0)
                target_col = 'future_return'
                feature_cols = [col for col in data.columns if col not in ['future_return', 'instrument']]

            X = data[feature_cols].fillna(0)
            y = data[target_col]

            # 训练多个模型
            models = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'lr': LogisticRegression(random_state=42, max_iter=1000)
            }

            trained_models = {}
            for name, model in models.items():
                try:
                    model.fit(X, y)
                    trained_models[name] = model
                    print(f"✅ {name.upper()} 模型训练完成")
                except Exception as e:
                    print(f"⚠️ {name.upper()} 模型训练失败: {e}")

            self.ml_models[model_name] = trained_models
            return trained_models

        except Exception as e:
            print(f"❌ 模型训练失败: {e}")
            return None

    def predict_with_ml_model(self, data: pd.DataFrame, model_name: str = 'default') -> pd.Series:
        """
        使用训练好的模型进行预测

        Args:
            data: 预测数据
            model_name: 模型名称
        """
        if model_name not in self.ml_models:
            print(f"❌ 模型 {model_name} 未训练")
            return pd.Series(0.0, index=data.index)

        try:
            # 准备特征
            feature_cols = [col for col in data.columns if not col.startswith('target')]
            X = data[feature_cols].fillna(0)

            # 获取所有模型的预测并平均
            predictions = pd.DataFrame(index=X.index)
            for model_name_key, model in self.ml_models[model_name].items():
                try:
                    pred = model.predict(X)
                    predictions[model_name_key] = pred
                except Exception as e:
                    print(f"⚠️ {model_name_key.upper()} 模型预测失败: {e}")

            # 计算平均预测
            if not predictions.empty:
                avg_pred = predictions.mean(axis=1)
                pred_series = pd.Series(avg_pred, index=data.index)
                print(f"✅ ML 模型预测完成，返回 {len(pred_series)} 个预测值")
                return pred_series
            else:
                print("⚠️ 所有模型预测失败")
                return pd.Series(0.0, index=data.index)

        except Exception as e:
            print(f"❌ 模型预测失败: {e}")
            return pd.Series(0.0, index=data.index)

    def calculate_ensemble_signal(self, technical_signals: Dict[str, pd.Series],
                                 ml_signals: pd.Series,
                                 weights: Dict[str, float] = None) -> pd.Series:
        """
        计算集成信号

        Args:
            technical_signals: 技术指标信号
            ml_signals: 机器学习信号
            weights: 各信号权重
        """
        # 默认权重
        if weights is None:
            weights = {}
            # 技术指标权重平均分配
            tech_weight_per = 0.3 / len(technical_signals) if technical_signals else 0
            for key in technical_signals:
                weights[key] = tech_weight_per
            # ML 信号权重
            weights['ml_signal'] = 0.4
            # 剩余权重给传统移动平均信号
            weights['fallback'] = 0.3

        # 确保所有输入都有唯一的索引，从第一个信号获取索引
        if technical_signals:
            first_signal = next(iter(technical_signals.values()))
            # 检查索引是否唯一
            if first_signal.index.duplicated().any():
                original_index_name = first_signal.index.name
                first_signal = first_signal.reset_index()
                first_signal = first_signal.set_index(first_signal.columns[0])
                first_signal.index.name = original_index_name
        else:
            first_signal = ml_signals

        # 创建综合信号 Series
        combined_signal = pd.Series(0.0, index=first_signal.index)

        # 加权组合技术指标信号
        for sig_name, sig_series in technical_signals.items():
            if sig_name in weights:
                # 确保信号系列与目标索引对齐
                sig_aligned = sig_series.reindex(combined_signal.index, fill_value=0.0)
                combined_signal += sig_aligned * weights[sig_name]

        # 加入 ML 信号
        if not ml_signals.empty and 'ml_signal' in weights:
            # 确保索引对齐
            ml_aligned = ml_signals.reindex(combined_signal.index, fill_value=0.0)
            combined_signal += ml_aligned * weights['ml_signal']

        # 标准化信号到 [-1, 1] 范围
        max_abs = combined_signal.abs().max()
        if max_abs > 0:
            combined_signal = combined_signal / max_abs

        # 将连续信号转换为离散信号（可选）
        discrete_signal = combined_signal.copy()
        discrete_signal[combined_signal > 0.1] = 1    # 买入信号
        discrete_signal[combined_signal < -0.1] = -1  # 卖出信号
        discrete_signal[(combined_signal >= -0.1) & (combined_signal <= 0.1)] = 0  # 持有信号

        print(f"✅ 生成集成信号，包含 {len(discrete_signal[discrete_signal != 0])} 个交易信号")
        return discrete_signal

    def adaptive_weighting(self, historical_performance: Dict[str, float]) -> Dict[str, float]:
        """
        自适应权重调整

        Args:
            historical_performance: 历史表现字典
        """
        # 基于历史表现调整权重
        total_perf = sum(max(0, perf) for perf in historical_performance.values())  # 只考虑正收益
        if total_perf == 0:
            # 如果所有模型表现都不好，恢复默认权重
            return {key: 1.0/len(historical_performance) for key in historical_performance}

        # 按表现比例分配权重
        weights = {}
        for model_name, perf in historical_performance.items():
            # 只给正收益模型分配权重
            if perf > 0:
                weights[model_name] = max(0, perf) / total_perf
            else:
                weights[model_name] = 0.01  # 给极小权重以保持多样性

        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            for model_name in weights:
                weights[model_name] /= total_weight

        return weights

    def evaluate_signal_performance(self, signals: pd.Series, actual_returns: pd.Series) -> Dict[str, float]:
        """
        评估信号表现

        Args:
            signals: 交易信号
            actual_returns: 实际收益率
        """
        if len(signals) != len(actual_returns):
            print("❌ 信号和收益率长度不匹配")
            return {}

        try:
            # 对齐索引
            aligned_signals, aligned_returns = signals.align(actual_returns, join='inner')

            # 计算策略收益率（信号滞后一期以避免前瞻偏差）
            strategy_returns = aligned_signals.shift(1).fillna(0) * aligned_returns

            # 计算各项指标
            total_return = strategy_returns.sum()
            avg_return = strategy_returns.mean()
            volatility = strategy_returns.std() * np.sqrt(252)  # 年化波动率
            sharpe = avg_return / (strategy_returns.std() + 1e-10) * np.sqrt(252)  # 夏普比率
            max_drawdown = self._calculate_max_drawdown(strategy_returns)

            # 胜率
            profitable_trades = (strategy_returns > 0).sum()
            total_trades = (strategy_returns != 0).sum()
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0

            performance = {
                'total_return': total_return,
                'avg_return': avg_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades
            }

            return performance

        except Exception as e:
            print(f"❌ 信号表现评估失败: {e}")
            return {}

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        try:
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max
            return drawdowns.min()
        except:
            return 0.0

    def run_fusion_strategy(self, data: pd.DataFrame,
                          rebalance_freq: str = 'daily',
                          initial_capital: float = 100000) -> Dict[str, Any]:
        """
        运行融合策略

        Args:
            data: 股票数据
            rebalance_freq: 再平衡频率
            initial_capital: 初始资金
        """
        print("🔄 开始运行融合策略...")

        # 计算各类信号
        print("📊 计算技术指标信号...")
        technical_signals = self.calculate_technical_signals(data)

        print("🤖 计算机器学习信号...")
        ml_signals = self.calculate_ml_signals(data)

        # 计算集成信号
        print("🔗 计算集成信号...")
        ensemble_signal = self.calculate_ensemble_signal(technical_signals, ml_signals)

        # 计算策略表现
        print("📈 评估策略表现...")
        if 'close' in data.columns:
            returns = data['close'].pct_change().fillna(0)
            performance = self.evaluate_signal_performance(ensemble_signal, returns)
        else:
            performance = {}
            returns = pd.Series(0, index=data.index)

        # 模拟交易过程
        portfolio_values = [initial_capital]
        positions = [0]  # 持仓数量
        cash = initial_capital

        for i in range(1, len(ensemble_signal)):
            current_signal = ensemble_signal.iloc[i-1]  # 使用前一期信号
            current_price = data['close'].iloc[i] if 'close' in data.columns else 100

            # 根据信号调整仓位
            if current_signal == 1:  # 买入
                shares_to_buy = int(cash * 0.9 / current_price)  # 使用90%现金买入
                positions.append(positions[-1] + shares_to_buy)
                cash -= shares_to_buy * current_price
            elif current_signal == -1:  # 卖出
                cash += positions[-1] * current_price  # 清空所有持仓
                positions.append(0)
            else:  # 持有
                positions.append(positions[-1])

            # 更新投资组合价值
            portfolio_value = cash + positions[-1] * current_price
            portfolio_values.append(portfolio_value)

        result = {
            'signals': ensemble_signal,
            'performance': performance,
            'portfolio_values': pd.Series(portfolio_values, index=data.index),
            'positions': pd.Series(positions, index=data.index),
            'cash': cash,
            'technical_signals': technical_signals,
            'ml_signals': ml_signals
        }

        print("✅ 融合策略运行完成")
        print(f"💰 初始资金: {initial_capital}")
        print(f"💰 最终价值: {portfolio_values[-1]:.2f}")
        if performance:
            print(f"📊 年化收益: {performance.get('avg_return', 0)*252:.4f}")
            print(f"📊 夏普比率: {performance.get('sharpe_ratio', 0):.4f}")
            print(f"📊 最大回撤: {performance.get('max_drawdown', 0):.4f}")

        return result


if __name__ == "__main__":
    print("🧪 测试模型融合模块...")

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

    # 测试模型融合
    fusion = ModelFusion()

    print(f"\n📋 模型融合器状态: 初始化完成")
    print(f"📊 Qlib/Scikit-learn 可用: {QLIB_SKLEARN_AVAILABLE}")

    print("\n🎯 主要功能:")
    print("• 传统技术指标信号计算")
    print("• 机器学习模型信号生成")
    print("• 集成信号融合")
    print("• 自适应权重调整")
    print("• 策略表现评估")
    print("• 投资组合模拟")

    print("\n💡 应用场景:")
    print("1. 多策略集成")
    print("2. 风险控制优化")
    print("3. 收益增强")
    print("4. 稳定性提升")