"""
GBM 策略

简洁的 Gradient Boosting Machine 策略：
1. 使用 LightGBM (sklearn 接口)
2. Alpha158 特征工程
3. 按年份固定划分数据
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from ..domain.interfaces.strategy import IStrategy
from ..core.registry import registry, ComponentType
from ..core.logging import get_logger

logger = get_logger(__name__)

# 尝试导入 LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logger.warning("lightgbm 未安装，请运行: pip install lightgbm")

# 导入数据加载器
try:
    from ..infrastructure.data_providers.qlib_data_loader import QlibDataLoader
    QLIB_LOADER_AVAILABLE = True
except ImportError:
    QLIB_LOADER_AVAILABLE = False

# 导入特征工程
try:
    from ..infrastructure.data_providers.qlib_data_converter import Alpha158Features, Alpha360Features
    ALPHA_FEATURES_AVAILABLE = True
except ImportError:
    ALPHA_FEATURES_AVAILABLE = False


@dataclass
class GBMConfig:
    """GBM 策略配置"""

    # 特征集
    feature_type: str = "alpha158"  # alpha158 或 alpha360

    # 模型参数 (优化后的参数)
    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.2
    num_leaves: int = 210
    min_child_samples: int = 20
    subsample: float = 0.8789
    colsample_bytree: float = 0.8879
    reg_alpha: float = 205.6999  # lambda_l1
    reg_lambda: float = 580.9768  # lambda_l2
    n_jobs: int = 20  # num_threads

    # 训练参数
    early_stopping_rounds: int = 50
    random_state: int = 42
    verbose: int = -1

    # 标签参数
    label_horizon: int = 10  # 预测未来10天收益率

    # 信号参数（使用百分位排名）
    # buy_threshold=0.25 表示前75%百分位触发买入
    # sell_threshold=0.25 表示后25%百分位触发卖出
    # 中间50%为持有区间（减少频繁交易）
    buy_threshold: float = 0.25
    sell_threshold: float = 0.25
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10

    # 数据划分（按年份固定）
    train_end: str = "2022-12-31"
    valid_end: str = "2024-06-30"


class GBMModel:
    """
    LightGBM 模型封装

    使用 sklearn 接口
    """

    def __init__(self, config: GBMConfig):
        """
        初始化模型

        Args:
            config: 模型配置
        """
        self.config = config
        self.model = None
        self.feature_names: List[str] = []
        self.is_fitted = False

        if not LGB_AVAILABLE:
            raise ImportError("lightgbm 未安装，请运行: pip install lightgbm")

        self._init_model()

    def _init_model(self):
        """初始化 LightGBM 模型"""
        self.model = lgb.LGBMRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
        )
        logger.info(f"LightGBM 模型初始化完成 (n_estimators={self.config.n_estimators})")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None
    ) -> 'GBMModel':
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_valid: 验证特征
            y_valid: 验证标签

        Returns:
            self
        """
        self.feature_names = list(X_train.columns)

        logger.info(f"开始训练 LightGBM 模型...")
        logger.info(f"  - 训练样本: {len(X_train)}")
        logger.info(f"  - 特征数量: {len(self.feature_names)}")

        # 设置验证集
        eval_set = None
        callbacks = None
        if X_valid is not None and y_valid is not None:
            eval_set = [(X_valid, y_valid)]
            logger.info(f"  - 验证样本: {len(X_valid)}")
            # 构建 callbacks，过滤 None 值
            callbacks = [lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)]
            if self.config.verbose >= 0:
                callbacks.append(lgb.log_evaluation(period=100))

        # 训练模型
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )

        self.is_fitted = True
        logger.info(f"模型训练完成，最佳迭代: {self.model.best_iteration_}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测收益率

        Args:
            X: 特征 DataFrame

        Returns:
            预测值数组
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测上涨概率

        将收益率预测转换为概率

        Args:
            X: 特征 DataFrame

        Returns:
            概率数组
        """
        pred = self.predict(X)
        # 使用 sigmoid 将收益率映射到概率
        prob = 1 / (1 + np.exp(-10 * pred))
        return np.clip(prob, 0, 1)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        获取特征重要性

        Args:
            top_n: 返回前 N 个特征

        Returns:
            特征重要性 DataFrame
        """
        if not self.is_fitted:
            return pd.DataFrame()

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance.head(top_n)

    def save(self, filepath: str):
        """保存模型"""
        data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"模型已保存: {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.is_fitted = True
        logger.info(f"模型已加载: {filepath}")


@registry.register(ComponentType.STRATEGY, "gbm")
class GBMStrategy(IStrategy):
    """
    GBM 策略

    特点:
    1. 使用 LightGBM (sklearn 接口)
    2. Alpha158 特征工程
    3. 按年份固定划分数据
    4. 简洁易用
    """

    def __init__(self, config: Optional[GBMConfig] = None):
        """
        初始化策略

        Args:
            config: 策略配置
        """
        self.config = config or GBMConfig()
        self.model: Optional[GBMModel] = None
        self.data_loader: Optional[QlibDataLoader] = None

        # 初始化数据加载器
        if QLIB_LOADER_AVAILABLE:
            self.data_loader = QlibDataLoader()

        # 策略参数
        self.parameters = {
            'feature_type': self.config.feature_type,
            'n_estimators': self.config.n_estimators,
            'learning_rate': self.config.learning_rate,
            'buy_threshold': self.config.buy_threshold,
            'sell_threshold': self.config.sell_threshold,
            'stop_loss_pct': self.config.stop_loss_pct,
            'take_profit_pct': self.config.take_profit_pct,
        }

    def initialize(self, parameters: Dict[str, Any]) -> None:
        """初始化策略参数"""
        self.parameters.update(parameters)

        # 更新配置
        for key, value in self.parameters.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def train(
        self,
        instruments: List[str],
        start_date: str = "2017-01-01",
        end_date: str = "2026-12-31",
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        训练模型

        Args:
            instruments: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            show_progress: 是否显示进度

        Returns:
            训练结果
        """
        logger.info(f"开始训练 GBM 策略...")
        logger.info(f"  - 股票数量: {len(instruments)}")
        logger.info(f"  - 日期范围: {start_date} ~ {end_date}")
        logger.info(f"  - 特征类型: {self.config.feature_type}")

        # 初始化数据加载器
        if not self.data_loader:
            self.data_loader = QlibDataLoader()

        if not self.data_loader.init_qlib():
            raise RuntimeError("Qlib 初始化失败")

        # 创建数据集
        features, labels = self.data_loader.create_dataset(
            instruments=instruments,
            start_date=start_date,
            end_date=end_date,
            feature_type=self.config.feature_type,
            label_horizon=self.config.label_horizon
        )

        if features.empty:
            raise ValueError("没有有效的训练数据")

        # 按日期划分数据
        train_end = self.config.train_end
        valid_end = self.config.valid_end

        # 训练集
        train_mask = features.index.get_level_values('datetime') <= train_end
        X_train = features[train_mask]
        y_train = labels[train_mask]

        # 验证集
        valid_mask = (features.index.get_level_values('datetime') > train_end) & \
                     (features.index.get_level_values('datetime') <= valid_end)
        X_valid = features[valid_mask]
        y_valid = labels[valid_mask]

        # 测试集
        test_mask = features.index.get_level_values('datetime') > valid_end
        X_test = features[test_mask]
        y_test = labels[test_mask]

        logger.info(f"数据划分完成:")
        logger.info(f"  - 训练集: {len(X_train)} 条 (截止 {train_end})")
        logger.info(f"  - 验证集: {len(X_valid)} 条 ({train_end} ~ {valid_end})")
        logger.info(f"  - 测试集: {len(X_test)} 条 ({valid_end} 之后)")

        # 创建并训练模型
        self.model = GBMModel(self.config)
        self.model.fit(X_train, y_train, X_valid, y_valid)

        # 计算训练指标
        train_pred = self.model.predict(X_train)
        train_ic = np.corrcoef(train_pred, y_train)[0, 1]

        valid_ic = 0
        if len(X_valid) > 0:
            valid_pred = self.model.predict(X_valid)
            valid_ic = np.corrcoef(valid_pred, y_valid)[0, 1]

        test_ic = 0
        if len(X_test) > 0:
            test_pred = self.model.predict(X_test)
            test_ic = np.corrcoef(test_pred, y_test)[0, 1]

        result = {
            'success': True,
            'train_samples': len(X_train),
            'valid_samples': len(X_valid),
            'test_samples': len(X_test),
            'feature_count': len(features.columns),
            'train_ic': train_ic,
            'valid_ic': valid_ic,
            'test_ic': test_ic,
            'best_iteration': self.model.model.best_iteration_,
        }

        logger.info(f"训练完成:")
        logger.info(f"  - 训练 IC: {train_ic:.4f}")
        logger.info(f"  - 验证 IC: {valid_ic:.4f}")
        logger.info(f"  - 测试 IC: {test_ic:.4f}")

        return result

    def predict(
        self,
        instrument: str,
        historical_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        预测单只股票

        Args:
            instrument: 股票代码
            historical_data: 历史数据（可选，如未提供则从 qlib 加载）

        Returns:
            预测结果
        """
        if not self.model or not self.model.is_fitted:
            raise RuntimeError("模型未训练")

        # 加载数据
        if historical_data is None:
            if not self.data_loader:
                self.data_loader = QlibDataLoader()
            historical_data = self.data_loader.load_stock_data(instrument)

        if historical_data.empty:
            raise ValueError(f"无法获取股票数据: {instrument}")

        # 生成特征
        if self.config.feature_type == "alpha158":
            features = Alpha158Features.generate(historical_data)
        else:
            features = Alpha360Features.generate(historical_data)

        # 获取最新特征
        latest_features = features.iloc[[-1]]

        # 预测
        pred_return = self.model.predict(latest_features)[0]
        pred_prob = self.model.predict_proba(latest_features)[0]

        # 生成信号
        signal = self._generate_signal(pred_prob, pred_return)

        return {
            'instrument': instrument,
            'return_pred': pred_return,
            'probability': pred_prob,
            'signal': signal['signal'],
            'direction': signal['direction'],
            'confidence': signal['confidence'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'close': historical_data['close'].iloc[-1],
        }

    def _generate_signal(
        self,
        probability: float,
        pred_return: float
    ) -> Dict[str, Any]:
        """
        生成交易信号

        Args:
            probability: 上涨概率
            pred_return: 预测收益率

        Returns:
            信号字典
        """
        direction = None
        signal = 'hold'

        if probability >= self.config.buy_threshold:
            direction = 'buy'
            signal = 'buy'
        elif probability <= self.config.sell_threshold:
            direction = 'sell'
            signal = 'sell'

        # 置信度
        confidence = abs(probability - 0.5) * 2

        return {
            'signal': signal,
            'direction': direction,
            'probability': probability,
            'confidence': confidence,
            'pred_return': pred_return,
            'stop_loss': None,  # 需要当前价格计算
            'take_profit': None,
        }

    def get_signal(
        self,
        current_bar: pd.Series,
        historical_bars: pd.DataFrame,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        获取交易信号（实现 IStrategy 接口）

        Args:
            current_bar: 当前 K 线
            historical_bars: 历史 K 线

        Returns:
            信号字典
        """
        if not self.model or not self.model.is_fitted:
            return {
                'direction': None,
                'signal': 'hold',
                'reason': '模型未训练'
            }

        min_bars = 120
        if len(historical_bars) < min_bars:
            return {
                'direction': None,
                'signal': 'hold',
                'reason': f'数据不足(需{min_bars}日)'
            }

        try:
            # 生成特征
            if self.config.feature_type == "alpha158":
                features = Alpha158Features.generate(historical_bars)
            else:
                features = Alpha360Features.generate(historical_bars)

            latest_features = features.iloc[[-1]]

            # 预测
            pred_return = self.model.predict(latest_features)[0]
            pred_prob = self.model.predict_proba(latest_features)[0]

            # 计算预测值在历史预测中的百分位排名
            # 使用最近60天的预测值作为参考
            lookback = min(60, len(features) - 1)
            if lookback > 10:
                historical_features = features.iloc[-lookback:]
                historical_preds = self.model.predict(historical_features)
                # 计算当前预测值的百分位
                percentile = (historical_preds < pred_return).sum() / len(historical_preds)
            else:
                percentile = 0.5  # 数据不足，使用中位数

            # ========== 趋势判断（避免高位追涨）==========
            close_series = historical_bars['close']
            close = current_bar['close']

            # 1. 均线位置判断
            ma20 = close_series.rolling(window=20).mean().iloc[-1]
            ma60 = close_series.rolling(window=60).mean().iloc[-1] if len(close_series) >= 60 else ma20

            # 价格相对 MA20 的偏离度
            price_deviation_ma20 = (close - ma20) / ma20

            # 2. 近期涨幅判断（过去10日涨幅）
            if len(close_series) >= 10:
                recent_return_10d = (close - close_series.iloc[-10]) / close_series.iloc[-10]
            else:
                recent_return_10d = 0

            # 3. RSI 判断（避免超买区买入）
            delta = close_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1] if not rs.empty else 50

            # 趋势安全标志
            trend_safe_to_buy = True
            trend_warning = []

            # 价格高于 MA20 超过 15%，认为短期偏高
            if price_deviation_ma20 > 0.15:
                trend_safe_to_buy = False
                trend_warning.append(f"价格偏离MA20 {price_deviation_ma20:.1%}")

            # 近10日涨幅超过 20%，避免追涨
            if recent_return_10d > 0.20:
                trend_safe_to_buy = False
                trend_warning.append(f"近10日涨幅 {recent_return_10d:.1%}")

            # RSI 超过 70，超买区不买入
            if rsi > 70:
                trend_safe_to_buy = False
                trend_warning.append(f"RSI超买 {rsi:.0f}")

            # ========== 生成信号 ==========
            direction = None
            signal_type = 'hold'

            # 使用百分位排名判断
            buy_percentile = 1 - self.config.buy_threshold
            sell_percentile = self.config.sell_threshold

            if percentile >= buy_percentile:
                # 买入信号：需要趋势安全
                if trend_safe_to_buy:
                    direction = 'buy'
                    signal_type = 'buy'
                else:
                    # 趋势不安全，保持持有
                    signal_type = 'hold'
                    direction = None

            elif percentile <= sell_percentile:
                direction = 'sell'
                signal_type = 'sell'

            # 使用 ATR 计算止损止盈
            close = current_bar['close']

            # 计算 ATR（从历史数据）
            try:
                high = historical_bars['high']
                low = historical_bars['low']
                close_series = historical_bars['close']

                # 计算 True Range
                tr1 = high - low
                tr2 = abs(high - close_series.shift(1))
                tr3 = abs(low - close_series.shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

                # ATR (14日)
                atr = tr.rolling(window=14).mean().iloc[-1]

                if pd.notna(atr) and atr > 0:
                    # 止损：价格 - 2倍ATR
                    stop_loss = close - 2 * atr
                    # 止盈：价格 + 3倍ATR
                    take_profit = close + 3 * atr
                else:
                    # 回退到固定百分比
                    stop_loss = close * (1 - self.config.stop_loss_pct)
                    take_profit = close * (1 + self.config.take_profit_pct)
            except Exception:
                # 回退到固定百分比
                stop_loss = close * (1 - self.config.stop_loss_pct)
                take_profit = close * (1 + self.config.take_profit_pct)

            # 置信度基于百分位排名
            confidence = abs(percentile - 0.5) * 2

            return {
                'direction': direction,
                'signal': signal_type,
                'probability': pred_prob,
                'pred_return': pred_return,
                'percentile': percentile,
                'confidence': confidence,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy_name': f'GBMStrategy({self.config.feature_type})',
                'timestamp': current_bar.get('timestamp', datetime.now()),
            }

        except Exception as e:
            logger.error(f"信号生成失败: {e}")
            return {
                'direction': None,
                'signal': 'hold',
                'error': str(e)
            }

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有信号

        Args:
            bars: K 线数据

        Returns:
            包含信号的 DataFrame
        """
        result = bars.copy()
        min_bars = 120

        # 初始化列
        result['signal'] = 'hold'
        result['probability'] = np.nan
        result['confidence'] = np.nan

        for i in range(min_bars, len(bars)):
            window_data = bars.iloc[:i+1]
            current_bar = bars.iloc[i]

            signal = self.get_signal(current_bar, window_data)

            result.loc[result.index[i], 'signal'] = signal.get('signal', 'hold')
            result.loc[result.index[i], 'probability'] = signal.get('probability', np.nan)
            result.loc[result.index[i], 'confidence'] = signal.get('confidence', np.nan)

        return result

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """获取特征重要性"""
        if self.model:
            return self.model.get_feature_importance(top_n)
        return pd.DataFrame()

    def save_model(self, filepath: str):
        """保存模型"""
        if self.model:
            self.model.save(filepath)

    def load_model(self, filepath: str):
        """加载模型"""
        self.model = GBMModel(self.config)
        self.model.load(filepath)

    def get_name(self) -> str:
        """获取策略名称"""
        return f"GBMStrategy({self.config.feature_type})"

    def get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return self.parameters.copy()

    def get_description(self) -> str:
        """获取策略描述"""
        return (
            f"GBM 策略: 特征集={self.config.feature_type}, "
            f"模型=LightGBM (n_estimators={self.config.n_estimators}), "
            f"买入阈值={self.config.buy_threshold}"
        )


# 便捷函数
def create_gbm_strategy(
    feature_type: str = "alpha158",
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    **kwargs
) -> GBMStrategy:
    """
    创建 GBM 策略

    Args:
        feature_type: 特征类型
        n_estimators: 树数量
        learning_rate: 学习率
        **kwargs: 其他参数

    Returns:
        GBMStrategy 实例
    """
    config = GBMConfig(
        feature_type=feature_type,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        **kwargs
    )
    return GBMStrategy(config)
