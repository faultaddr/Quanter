"""
深度 Qlib 集成模块
集成 Qlib 的机器学习和深度学习能力到现有项目
"""
import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Check Python version compatibility
import platform
python_version = platform.python_version()

if tuple(map(int, python_version.split('.')[:2])) < (3, 7):
    QLIB_AVAILABLE = False
    print(f"⚠️ Qlib 需要 Python 3.7+，当前版本: {python_version}")
    print("💡 解决方案: 使用 Python 3.8+ 运行程序或安装兼容版本的 Qlib")
else:
    try:
        import qlib
        from qlib.config import REG_CN as REGION_CN
        from qlib.data import D
        from qlib.utils import init_instance_by_config
        from qlib.workflow import R
        from qlib.model.trainer import task_train
        from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
        from qlib.contrib.evaluate import risk_analysis, indicator_analysis
        from qlib.backtest import backtest, executor

        # Try to import GBDT separately to handle potential LightGBM/OpenMP issues
        try:
            from qlib.contrib.model.gbdt import LGBModel as GBDT  # Newer Qlib versions use LGBModel instead of GBDT
            GBDT_AVAILABLE = True
        except (ImportError, OSError) as e:
            # Try the older name as fallback
            try:
                from qlib.contrib.model.gbdt import GBDT
                GBDT_AVAILABLE = True
            except (ImportError, OSError) as e2:
                print(f"⚠️ GBDT 模型不可用 (LightGBM 问题): {e}")
                print("💡 解决方案: 运行以下命令之一安装 OpenMP 库:")
                print("   macOS (Homebrew): brew install libomp")
                print("   macOS (Conda): conda install -c conda-forge libopenmp")
                print("   或运行 install_qlib.sh 脚本来自动处理此问题")
                GBDT_AVAILABLE = False

        # Import LinearModel
        try:
            from qlib.contrib.model.linear import LinearModel
            LINEAR_MODEL_AVAILABLE = True
        except (ImportError, OSError) as e:
            print(f"⚠️ Linear 模型不可用 (可能受LightGBM问题影响): {e}")
            LINEAR_MODEL_AVAILABLE = False

        QLIB_AVAILABLE = True
    except ImportError:
        QLIB_AVAILABLE = False
        print("⚠️ Qlib 未安装，将使用模拟集成")


class DeepQlibIntegration:
    """
    深度 Qlib 集成类
    整合 Qlib 的机器学习和深度学习能力
    """

    def __init__(self, provider_uri="~/.qlib/qlib_data/cn_data"):
        """初始化深度 Qlib 集成"""
        self.provider_uri = provider_uri
        self.initialized = False

        if QLIB_AVAILABLE:
            try:
                # 初始化 Qlib
                qlib.init(provider_uri=self.provider_uri, region=REGION_CN)
                self.initialized = True
                print("✅ 深度 Qlib 集成初始化成功")
            except Exception as e:
                print(f"⚠️ Qlib 初始化失败 (仅影响高级功能): {e}")
                print("💡 提示: 运行 install_qlib.sh 安装完整 Qlib 数据环境")
        else:
            # Qlib not available, will use basic analysis functionality
            pass

    def get_qlib_alpha_factors(self, instruments, start_date, end_date, alpha_version='158'):
        """
        获取 Qlib 的 Alpha 因子

        Args:
            instruments: 股票列表
            start_date: 开始日期
            end_date: 结束日期
            alpha_version: Alpha 版本 ('158' 或 '101')
        """
        if not self.initialized:
            print("❌ Qlib 未初始化，无法获取 Alpha 因子")
            return pd.DataFrame()

        try:
            if alpha_version == '158':
                # Qlib Alpha158 特征集
                alpha_fields = [
                    # 技术指标类
                    'Ref($close,1)/$close',  # 一日收益率
                    'Mean($close,5)/$close', # 五日均值比
                    'Mean($close,10)/$close',# 十日均值比
                    'Mean($close,20)/$close',# 二十日均值比
                    '(($close-$open)/$open)', # 开盘转收盘变化
                    '($high-$low)/$close',    # 最高价最低价差
                    'Rank($volume)',          # 成交量排名
                    'Rank($close)',           # 收盘价排名
                    # 波动率类
                    'Std($close,10)',         # 10日标准差
                    'Std($close,20)',         # 20日标准差
                    # 其他复杂特征
                    'Ts_Sum(Greater($close-$open,0),5)/Ts_Sum(Abs($close-$open),5)',
                    'Slope($close,5)',        # 5日趋势斜率
                    'Resi($close,20)',        # 20日残差
                ]
            else:  # Alpha101
                alpha_fields = [
                    '$close/$open-1',  # 日回报
                    'Rank($volume)/Rank($close)',  # 量价关系
                    'Ts_Sum($high-$low, 10)/Ts_Sum(Ts_Sum($high-$low, 2), 5)',  # 波动率特征
                    'Delay($close,5)/$close',  # 5日滞后比
                    'Corr(Rank($close), Rank($volume), 5)',  # 价量相关性
                    'Decay_linear($close, 5)',  # 线性衰减
                ]

            # 获取特征数据
            df = D.features(instruments, alpha_fields, start_date, end_date)
            print(f"✅ 成功获取 {len(alpha_fields)} 个 Alpha{alpha_version} 因子，{len(df)} 条记录")
            return df

        except Exception as e:
            print(f"❌ 获取 Alpha 因子失败: {e}")
            return pd.DataFrame()

    def train_ml_model(self, data, target_column='LABEL0', model_type='gbdt'):
        """
        使用 Qlib 训练机器学习模型

        Args:
            data: 训练数据
            target_column: 目标列名
            model_type: 模型类型 ('gbdt', 'linear')
        """
        if not self.initialized or data.empty:
            print("❌ 无法训练模型：Qlib未初始化或数据为空")
            return None

        try:
            # Prepare features and labels
            if target_column in data.columns:
                X = data.drop(columns=[target_column])
                y = data[target_column]
            else:
                # If no explicit label, create simple label based on price movement
                X = data
                # Create future return labels (example, should be defined based on demand)
                y = data['$close'].pct_change().shift(-1).fillna(0).apply(
                    lambda x: 1 if x > 0.02 else (-1 if x < -0.02 else 0)
                )

            # Select model based on availability
            if model_type == 'gbdt':
                if not GBDT_AVAILABLE:
                    print("❌ GBDT 模型不可用 (LightGBM 问题)，请安装 OpenMP 库后再试")
                    print("💡 解决方案: 运行 'brew install libomp' 或 'conda install -c conda-forge libopenmp'")
                    return None
                model = GBDT(
                    loss="mse",  # Regression task
                    colsample_bytree=0.8879,
                    learning_rate=0.2,
                    subsample=0.84,
                    lambda_l1=205.6999,
                    lambda_l2=580.8121,
                    max_depth=8,
                    num_leaves=210,
                    num_boost_round=300,
                    early_stopping_rounds=50
                )
            elif model_type == 'linear':
                if not LINEAR_MODEL_AVAILABLE:
                    print("❌ Linear 模型不可用")
                    return None
                model = LinearModel()
            else:
                # Fallback to linear model if GBDT is requested but not available
                if model_type != 'linear':
                    print(f"⚠️ 模型类型 '{model_type}' 不支持，使用线性模型作为备选")
                if LINEAR_MODEL_AVAILABLE:
                    model = LinearModel()
                else:
                    print("❌ 没有可用的模型")
                    return None

            # Train the model
            print(f"🚀 使用 {model_type.upper()} 模型训练中...")
            model.fit(X, y)
            print(f"✅ {model_type.upper()} 模型训练完成")

            return model

        except Exception as e:
            print(f"❌ 模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict_with_qlib_model(self, model, features):
        """
        使用训练好的 Qlib 模型进行预测

        Args:
            model: 训练好的模型
            features: 特征数据
        """
        if model is None or features.empty:
            print("❌ 无法预测：模型或特征数据为空")
            return pd.Series()

        try:
            predictions = model.predict(features)
            pred_series = pd.Series(predictions, index=features.index)
            print(f"✅ 使用 Qlib 模型预测完成，共 {len(pred_series)} 个预测值")
            return pred_series
        except Exception as e:
            print(f"❌ 模型预测失败: {e}")
            return pd.Series()

    def run_qlib_backtest(self, strategy_params=None, executor_params=None):
        """
        运行 Qlib 高级回测

        Args:
            strategy_params: 策略参数
            executor_params: 执行器参数
        """
        if not self.initialized:
            print("❌ Qlib 未初始化，无法运行高级回测")
            return {}

        try:
            # 这里可以定义具体的回测配置
            print("🔄 运行 Qlib 高级回测...")

            # Qlib 的高级回测功能比较复杂，需要详细配置
            # 这是一个简化的示例
            backtest_config = {
                'start_time': '2024-01-01',
                'end_time': '2024-12-31',
                'benchmark': 'SH000300',  # 沪深300基准
                'account': 1000000,       # 初始资金
                'freq': 'day',
                'refresh_rate': 1,
                'deal_price': 'close',
                'open_cost': 0.0005,      # 开仓手续费
                'close_cost': 0.0015,     # 平仓手续费
                'min_cost': 5,            # 最小手续费
            }

            # 这里可以执行真正的回测
            print("✅ Qlib 高级回测框架已就绪")
            print("💡 提示：详细回测配置需要根据具体策略定义")
            return backtest_config

        except Exception as e:
            print(f"⚠️ Qlib 高级回测配置失败: {e}")
            return {}

    def get_ml_signals(self, stock_data, method='ensemble'):
        """
        使用机器学习方法生成交易信号

        Args:
            stock_data: 股票数据
            method: 信号生成方法 ('ml', 'ensemble')
        """
        if stock_data.empty:
            print("❌ 股票数据为空，无法生成 ML 信号")
            return pd.Series()

        try:
            print(f"🤖 使用 {method.upper()} 方法生成 ML 交易信号...")

            # 准备特征
            features = self.prepare_ml_features(stock_data)

            if features.empty:
                print("⚠️ 特征准备失败，使用传统指标")
                return self.get_traditional_signals(stock_data)

            # 根据方法选择不同的信号生成策略
            if method == 'ml':
                # 简单的 ML 信号生成（实际应用中会更复杂）
                signals = self.ml_signal_generation(features)
            else:  # ensemble
                # 集成多种 ML 方法
                ml_signals = self.ml_signal_generation(features)
                traditional_signals = self.get_traditional_signals(stock_data)
                # 组合信号
                signals = (ml_signals + traditional_signals) / 2
                # 将连续信号转换为离散信号
                signals = signals.apply(lambda x: 1 if x > 0.1 else (-1 if x < -0.1 else 0))

            print(f"✅ 生成了 {len(signals[signals != 0])} 个 ML 交易信号")
            return signals

        except Exception as e:
            print(f"❌ ML 信号生成失败: {e}")
            return pd.Series()

    def prepare_ml_features(self, data):
        """
        准备机器学习特征
        """
        if data.empty:
            return pd.DataFrame()

        try:
            # 确保数据索引唯一，如果有重复则重置索引但保留原始索引作为列
            if data.index.duplicated().any():
                original_index_name = data.index.name
                data = data.reset_index()
                data = data.set_index(data.columns[0])  # 使用第一列作为新索引
                data.index.name = original_index_name

            features = pd.DataFrame(index=data.index)

            # 基础价格特征
            features['close_lag1'] = data['close'].shift(1)
            features['close_lag2'] = data['close'].shift(2)
            features['close_lag3'] = data['close'].shift(3)

            # 价格变化率
            features['pct_chg'] = data['close'].pct_change()
            features['pct_chg_lag1'] = features['pct_chg'].shift(1)
            features['pct_chg_lag2'] = features['pct_chg'].shift(2)

            # 技术指标特征
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))

            # 布林带
            features['ma_20'] = data['close'].rolling(window=20).mean()
            bb_std = data['close'].rolling(window=20).std()
            features['bb_upper'] = features['ma_20'] + (bb_std * 2)
            features['bb_lower'] = features['ma_20'] - (bb_std * 2)
            features['bb_position'] = (data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)

            # MACD
            exp12 = data['close'].ewm(span=12).mean()
            exp26 = data['close'].ewm(span=26).mean()
            features['macd'] = exp12 - exp26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_hist'] = features['macd'] - features['macd_signal']

            # 波动率
            features['volatility'] = data['close'].pct_change().rolling(window=10).std()

            # 成交量特征
            features['volume_sma'] = data['volume'].rolling(window=10).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma']

            # 删除包含 NaN 的行
            features = features.dropna()

            return features

        except Exception as e:
            print(f"⚠️ 特征准备失败: {e}")
            return pd.DataFrame()

    def ml_signal_generation(self, features):
        """
        基于 ML 特征生成信号
        """
        if features.empty:
            return pd.Series()

        try:
            # 确保数据索引唯一，如果有重复则重置索引但保留原始索引作为列
            if features.index.duplicated().any():
                original_index_name = features.index.name
                features = features.reset_index()
                features = features.set_index(features.columns[0])  # 使用第一列作为新索引
                features.index.name = original_index_name

            # 简单的规则基信号生成（实际应用中可以用训练的模型）
            signals = pd.Series(0.0, index=features.index)

            # 基于 RSI 的信号
            if 'rsi' in features.columns:
                signals[(features['rsi'] < 30) & (features['rsi'].shift(1) >= 30)] = 0.8  # 超卖买入
                signals[(features['rsi'] > 70) & (features['rsi'].shift(1) <= 70)] = -0.8  # 超买卖出

            # 基于布林带的信号
            if 'bb_position' in features.columns:
                signals[(features['bb_position'] < 0.1)] = 0.7  # 接近下轨买入
                signals[(features['bb_position'] > 0.9)] = -0.7  # 接近上轨卖出

            # 基于 MACD 的信号
            if 'macd' in features.columns and 'macd_signal' in features.columns:
                signals[(features['macd'] > features['macd_signal']) &
                        (features['macd'].shift(1) <= features['macd_signal'].shift(1))] = 0.6  # 金叉
                signals[(features['macd'] < features['macd_signal']) &
                        (features['macd'].shift(1) >= features['macd_signal'].shift(1))] = -0.6  # 死叉

            # 波动率过滤
            if 'volatility' in features.columns:
                high_vol_filter = features['volatility'] > features['volatility'].quantile(0.7)
                low_vol_filter = features['volatility'] < features['volatility'].quantile(0.3)

                # 高波动率时减少信号强度，低波动率时增加信号强度
                signals[high_vol_filter] *= 0.7
                signals[low_vol_filter] *= 1.3

            return signals

        except Exception as e:
            print(f"⚠️ ML 信号生成失败: {e}")
            return pd.Series(0.0, index=features.index)

    def get_traditional_signals(self, data):
        """
        获取传统技术指标信号作为对比
        """
        if data.empty:
            return pd.Series(0.0, index=data.index)

        # 确保数据索引唯一，如果有重复则重置索引但保留原始索引作为列
        if data.index.duplicated().any():
            original_index_name = data.index.name
            data = data.reset_index()
            data = data.set_index(data.columns[0])  # 使用第一列作为新索引
            data.index.name = original_index_name

        signals = pd.Series(0.0, index=data.index)

        try:
            # 简单移动平均线策略
            ma_short = data['close'].rolling(window=5).mean()
            ma_long = data['close'].rolling(window=20).mean()

            buy_signals = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
            sell_signals = (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))

            signals.loc[buy_signals] = 0.5
            signals.loc[sell_signals] = -0.5

        except Exception as e:
            print(f"⚠️ 传统信号生成失败: {e}")

        return signals

    def compare_models_performance(self, data, models_config=None):
        """
        比较不同模型的性能

        Args:
            data: 测试数据
            models_config: 模型配置
        """
        print("📊 比较不同模型的性能...")

        results = {}

        # 传统的技术指标策略
        trad_signals = self.get_traditional_signals(data)
        results['Traditional_MA'] = self._evaluate_signals(trad_signals, data)

        # 机器学习策略
        ml_signals = self.get_ml_signals(data, method='ml')
        if not ml_signals.empty:
            results['ML_Based'] = self._evaluate_signals(ml_signals, data)

        # 集成策略
        ensemble_signals = self.get_ml_signals(data, method='ensemble')
        if not ensemble_signals.empty:
            results['Ensemble'] = self._evaluate_signals(ensemble_signals, data)

        # 显示结果对比
        print("\n📈 模型性能对比:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        return results

    def _evaluate_signals(self, signals, data):
        """
        评估信号表现
        """
        if signals.empty or data.empty:
            return {'return': 0, 'sharpe': 0, 'max_dd': 0}

        try:
            # 确保索引一致，如果有重复则重置索引但保留原始索引作为列
            if signals.index.duplicated().any():
                original_index_name = signals.index.name
                signals = signals.reset_index()
                signals = signals.set_index(signals.columns[0])  # 使用第一列作为新索引
                signals.index.name = original_index_name

            if data.index.duplicated().any():
                original_index_name = data.index.name
                data = data.reset_index()
                data = data.set_index(data.columns[0])  # 使用第一列作为新索引
                data.index.name = original_index_name

            # 确保索引一致
            common_index = signals.index.intersection(data.index)
            signals = signals.loc[common_index]
            data = data.loc[common_index]

            # 生成持仓信号（滞后一期以避免前瞻偏差）
            positions = signals.shift(1).fillna(0)

            # 计算日收益率
            daily_returns = data['close'].pct_change().fillna(0)

            # 策略收益率
            strategy_returns = positions * daily_returns

            # 计算指标
            total_return = (1 + strategy_returns).prod() - 1
            annual_return = ((1 + total_return) ** (252 / len(strategy_returns))) - 1
            volatility = strategy_returns.std() * np.sqrt(252)
            sharpe = annual_return / volatility if volatility != 0 else 0

            # 计算最大回撤
            cum_returns = (1 + strategy_returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            max_dd = drawdowns.min()

            return {
                'return': annual_return,
                'sharpe': sharpe,
                'max_dd': max_dd,
                'volatility': volatility
            }
        except Exception as e:
            print(f"⚠️ 信号评估失败: {e}")
            return {'return': 0, 'sharpe': 0, 'max_dd': 0}

# 使用示例和测试
if __name__ == "__main__":
    print("🧪 测试深度 Qlib 集成功能...")

    integration = DeepQlibIntegration()

    print(f"\n📋 Qlib 集成状态: {'可用' if integration.initialized else '不可用'}")

    print("\n🎯 主要功能:")
    print("• Alpha 因子获取 (Alpha158/Alpha101)")
    print("• 机器学习模型训练 (GBDT, Linear)")
    print("• 深度学习模型集成")
    print("• 高级回测框架")
    print("• ML 驱动的信号生成")
    print("• 模型性能对比")

    print("\n💡 集成建议:")
    print("1. 在策略管理器中集成 ML 模型")
    print("2. 使用 Alpha 因子增强现有策略")
    print("3. 构建模型融合策略")
    print("4. 利用 Qlib 的风险模型")
    print("5. 部署在线学习机制")