"""
机器学习选股策略模块

基于XGBoost模型的智能选股策略，结合多因子特征和机器学习预测
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
import os
import warnings
warnings.filterwarnings('ignore')

from ..domain.interfaces.strategy import IStrategy
from ..factors.ml_feature_engineer import MLFeatureEngineer, LabelGenerator
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MLPredictionResult:
    """ML预测结果"""
    probability: float  # 上涨概率
    signal: str  # 信号类型
    confidence: float  # 置信度
    position_score: float  # 位置得分
    stop_loss: float  # 止损价
    take_profit: float  # 止盈价


class MLStockSelectionStrategy(IStrategy):
    """
    机器学习选股策略

    特点：
    1. 使用多因子特征工程
    2. XGBoost模型预测上涨概率
    3. 结合位置区间优化选股
    4. 内置止盈止损
    """

    def __init__(
        self,
        buy_prob_threshold: float = 0.60,
        sell_prob_threshold: float = 0.40,
        position_range: tuple = (0.20, 0.50),  # 位置区间
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        hold_days: int = 10,
        use_pretrained_model: bool = False,
        model_path: Optional[str] = None,
    ):
        """
        初始化机器学习选股策略

        Args:
            buy_prob_threshold: 买入概率阈值
            sell_prob_threshold: 卖出概率阈值
            position_range: 位置区间 (低位, 高位)
            stop_loss_pct: 止损比例
            take_profit_pct: 止盈比例
            hold_days: 持有天数
            use_pretrained_model: 是否使用预训练模型
            model_path: 模型路径
        """
        self.buy_prob_threshold = buy_prob_threshold
        self.sell_prob_threshold = sell_prob_threshold
        self.position_range = position_range
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.hold_days = hold_days
        self.use_pretrained_model = use_pretrained_model
        self.model_path = model_path

        # 初始化特征工程
        self.feature_engineer = MLFeatureEngineer()

        # 模型训练器 (延迟导入)
        self.trainer = None

        # 策略参数
        self.parameters = {
            'buy_prob_threshold': buy_prob_threshold,
            'sell_prob_threshold': sell_prob_threshold,
            'position_range': position_range,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'hold_days': hold_days,
        }

        # 信号历史
        self.signals_history: List[Dict] = []
        self.last_prediction: Optional[MLPredictionResult] = None

        # 训练数据缓存
        self._training_cache: Dict[str, pd.DataFrame] = {}

    def initialize(self, parameters: Dict[str, Any]) -> None:
        """初始化策略参数"""
        self.parameters.update(parameters)
        self.buy_prob_threshold = self.parameters.get('buy_prob_threshold', 0.60)
        self.sell_prob_threshold = self.parameters.get('sell_prob_threshold', 0.40)
        self.position_range = self.parameters.get('position_range', (0.20, 0.50))
        self.stop_loss_pct = self.parameters.get('stop_loss_pct', 0.05)
        self.take_profit_pct = self.parameters.get('take_profit_pct', 0.10)
        self.hold_days = self.parameters.get('hold_days', 10)

        # 加载预训练模型
        if self.use_pretrained_model and self.model_path and os.path.exists(self.model_path):
            self._load_model()

    def train_model(self, data: pd.DataFrame, retrain: bool = False,
                     model_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        训练模型

        Args:
            data: 历史数据
            retrain: 是否重新训练
            model_params: 模型超参数 (可选)

        Returns:
            训练是否成功
        """
        try:
            # 延迟导入避免依赖问题
            from ..ml.xgboost_trainer import XGBoostTrainer

            logger.info("开始训练机器学习模型...")

            # 生成特征
            features = self.feature_engineer.generate_features(data)

            # 生成标签 - 使用简单方法：未来N天收益 > 0
            label_generator = LabelGenerator(
                method='simple',  # 改用简单方法，标签更平衡
                horizon=self.hold_days
            )
            labels = label_generator.generate_labels(data)

            # 对齐特征和标签
            common_idx = features.index.intersection(labels.dropna().index)
            X = features.loc[common_idx]
            y = labels.loc[common_idx]

            # 默认模型参数
            default_params = {
                'n_estimators': 300,
                'max_depth': 5,
                'learning_rate': 0.05,
                'use_feature_selection': True,
                'feature_selection_threshold': 50,
                'random_state': None  # 默认不固定随机种子
            }

            # 使用传入的参数覆盖默认值
            if model_params:
                default_params.update(model_params)

            # 训练模型
            self.trainer = XGBoostTrainer(
                n_estimators=default_params['n_estimators'],
                max_depth=default_params['max_depth'],
                learning_rate=default_params['learning_rate'],
                use_feature_selection=default_params['use_feature_selection'],
                feature_selection_threshold=default_params['feature_selection_threshold'],
                random_state=default_params.get('random_state', None)
            )
            self.trainer.train(X, y)

            logger.info("模型训练完成")
            return True

        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            return False

    def _load_model(self):
        """加载预训练模型"""
        try:
            from ..ml.xgboost_trainer import XGBoostTrainer
            self.trainer = XGBoostTrainer()
            self.trainer.load_model(self.model_path)
            logger.info(f"模型已加载: {self.model_path}")
        except Exception as e:
            logger.warning(f"模型加载失败: {e}")
            self.trainer = None

    def get_signal(
        self,
        current_bar: pd.Series,
        historical_bars: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        获取交易信号

        Args:
            current_bar: 当前K线
            historical_bars: 历史K线

        Returns:
            信号字典
        """
        # 检查数据充足性
        if len(historical_bars) < 120:
            return {
                'direction': None,
                'signal': 'hold',
                'reason': '数据不足(需120日)'
            }

        # 如果模型未训练，使用规则策略
        if self.trainer is None or self.trainer.model is None:
            return self._get_fallback_signal(current_bar, historical_bars)

        try:
            # 生成特征
            features = self.feature_engineer.generate_features(historical_bars)

            # 获取最新特征
            latest_features = features.iloc[[-1]]

            # 预测上涨概率
            prob = self.trainer.predict_proba(latest_features)[0]

            # 计算位置
            close = current_bar['close']
            low_60 = historical_bars['low'].rolling(60).min().iloc[-1]
            high_60 = historical_bars['high'].rolling(60).max().iloc[-1]
            position = (close - low_60) / (high_60 - low_60 + 1e-10)

            # 生成信号
            direction = None
            signal_type = 'hold'

            # 买入条件：概率高 + 位置低
            if (prob >= self.buy_prob_threshold and
                self.position_range[0] <= position <= self.position_range[1]):
                direction = 'buy'
                signal_type = 'buy'
            # 卖出条件：概率低
            elif prob <= self.sell_prob_threshold:
                direction = 'sell'
                signal_type = 'sell'

            # 计算止损止盈
            stop_loss = close * (1 - self.stop_loss_pct)
            take_profit = close * (1 + self.take_profit_pct)

            # 置信度
            confidence = abs(prob - 0.5) * 2

            # 保存预测结果
            self.last_prediction = MLPredictionResult(
                probability=prob,
                signal=signal_type,
                confidence=confidence,
                position_score=position,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            signal = {
                'direction': direction,
                'signal': signal_type,
                'probability': prob,
                'confidence': confidence,
                'position': position,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy_name': 'MLStockSelectionStrategy',
                'timestamp': current_bar.get('timestamp', datetime.now())
            }

            self.signals_history.append(signal)
            return signal

        except Exception as e:
            logger.error(f"信号生成失败: {e}")
            return {
                'direction': None,
                'signal': 'hold',
                'error': str(e)
            }

    def _get_fallback_signal(self, current_bar: pd.Series, historical_bars: pd.DataFrame) -> Dict[str, Any]:
        """
        备用规则策略 (当模型不可用时)

        基于位置区间和动量的简单规则
        """
        close = current_bar['close']

        # 计算位置
        low_60 = historical_bars['low'].rolling(60).min().iloc[-1]
        high_60 = historical_bars['high'].rolling(60).max().iloc[-1]
        position = (close - low_60) / (high_60 - low_60 + 1e-10)

        # 计算动量
        mom_10 = (close - historical_bars['close'].iloc[-10]) / historical_bars['close'].iloc[-10]

        # 均线排列
        ma5 = historical_bars['close'].rolling(5).mean().iloc[-1]
        ma10 = historical_bars['close'].rolling(10).mean().iloc[-1]
        ma20 = historical_bars['close'].rolling(20).mean().iloc[-1]
        ma_bullish = ma5 > ma10 > ma20

        # 买入条件
        direction = None
        signal_type = 'hold'

        if (self.position_range[0] <= position <= self.position_range[1] and
            mom_10 > 0 and ma_bullish):
            direction = 'buy'
            signal_type = 'buy'
        elif position > 0.8 or mom_10 < -0.05:
            direction = 'sell'
            signal_type = 'sell'

        stop_loss = close * (1 - self.stop_loss_pct)
        take_profit = close * (1 + self.take_profit_pct)

        return {
            'direction': direction,
            'signal': signal_type,
            'probability': 0.5,  # 默认概率
            'confidence': 0.3,  # 低置信度
            'position': position,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'strategy_name': 'MLStockSelectionStrategy (Rule Fallback)',
            'timestamp': current_bar.get('timestamp', datetime.now())
        }

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        计算信号

        Args:
            bars: K线数据

        Returns:
            DataFrame: 包含信号列的数据
        """
        result = bars.copy()
        min_bars = 120

        # 初始化列
        result['signal'] = 'hold'
        result['probability'] = np.nan
        result['position'] = np.nan
        result['confidence'] = np.nan
        result['stop_loss'] = np.nan
        result['take_profit'] = np.nan

        for i in range(min_bars, len(bars)):
            window_data = bars.iloc[:i+1]
            current_bar = bars.iloc[i]

            signal = self.get_signal(current_bar, window_data)

            result.loc[result.index[i], 'signal'] = signal.get('signal', 'hold')
            result.loc[result.index[i], 'probability'] = signal.get('probability', np.nan)
            result.loc[result.index[i], 'position'] = signal.get('position', np.nan)
            result.loc[result.index[i], 'confidence'] = signal.get('confidence', np.nan)
            result.loc[result.index[i], 'stop_loss'] = signal.get('stop_loss', np.nan)
            result.loc[result.index[i], 'take_profit'] = signal.get('take_profit', np.nan)

        return result

    def get_name(self) -> str:
        """获取策略名称"""
        return "MLStockSelectionStrategy"

    def get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return self.parameters.copy()

    def get_description(self) -> str:
        """获取策略描述"""
        return (
            f"机器学习选股策略: 买入概率阈值={self.buy_prob_threshold}, "
            f"位置区间={self.position_range}, "
            f"止损={self.stop_loss_pct*100}%, "
            f"止盈={self.take_profit_pct*100}%"
        )

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """获取特征重要性"""
        if self.trainer is not None:
            return self.trainer.get_feature_importance(top_n)
        return pd.DataFrame()

    def get_signal_statistics(self) -> Dict:
        """获取信号统计"""
        if not self.signals_history:
            return {}

        buy_signals = [s for s in self.signals_history if s.get('direction') == 'buy']
        sell_signals = [s for s in self.signals_history if s.get('direction') == 'sell']

        return {
            'total_signals': len(self.signals_history),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'avg_buy_prob': np.mean([s.get('probability', 0) for s in buy_signals]) if buy_signals else 0,
            'avg_sell_prob': np.mean([s.get('probability', 0) for s in sell_signals]) if sell_signals else 0,
        }


class MLStockSelector:
    """
    机器学习股票筛选器

    用于批量筛选股票池
    """

    def __init__(
        self,
        strategy: Optional[MLStockSelectionStrategy] = None,
        top_k: int = 10
    ):
        """
        初始化筛选器

        Args:
            strategy: ML策略实例
            top_k: 筛选数量
        """
        self.strategy = strategy or MLStockSelectionStrategy()
        self.top_k = top_k

    def select_stocks(
        self,
        stock_data: Dict[str, pd.DataFrame],
        min_data_days: int = 120
    ) -> pd.DataFrame:
        """
        筛选股票

        Args:
            stock_data: 股票数据字典 {stock_code: DataFrame}
            min_data_days: 最小数据天数

        Returns:
            筛选结果DataFrame
        """
        results = []

        for stock_code, df in stock_data.items():
            if len(df) < min_data_days:
                continue

            try:
                signal = self.strategy.get_signal(df.iloc[-1], df)

                if signal.get('direction') == 'buy':
                    results.append({
                        'stock_code': stock_code,
                        'probability': signal.get('probability', 0),
                        'confidence': signal.get('confidence', 0),
                        'position': signal.get('position', 0),
                        'stop_loss': signal.get('stop_loss', 0),
                        'take_profit': signal.get('take_profit', 0),
                        'close': df['close'].iloc[-1],
                        'timestamp': df.index[-1] if df.index.name else df.iloc[-1].get('timestamp', datetime.now())
                    })
            except Exception as e:
                logger.warning(f"股票 {stock_code} 筛选失败: {e}")
                continue

        if not results:
            return pd.DataFrame()

        # 按概率排序
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('probability', ascending=False)

        return results_df.head(self.top_k)

    def batch_predict(
        self,
        stock_data: Dict[str, pd.DataFrame],
        min_data_days: int = 120
    ) -> pd.DataFrame:
        """
        批量预测所有股票

        Args:
            stock_data: 股票数据字典
            min_data_days: 最小数据天数

        Returns:
            预测结果DataFrame
        """
        results = []

        for stock_code, df in stock_data.items():
            if len(df) < min_data_days:
                continue

            try:
                signal = self.strategy.get_signal(df.iloc[-1], df)

                results.append({
                    'stock_code': stock_code,
                    'signal': signal.get('signal', 'hold'),
                    'probability': signal.get('probability', 0.5),
                    'confidence': signal.get('confidence', 0),
                    'position': signal.get('position', 0),
                    'close': df['close'].iloc[-1],
                })
            except Exception as e:
                logger.warning(f"股票 {stock_code} 预测失败: {e}")
                continue

        return pd.DataFrame(results)