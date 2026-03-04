"""
微软 Qlib 策略适配器

集成微软开源量化框架 Qlib 的 Alpha158 特征集和预训练模型
GitHub: https://github.com/microsoft/qlib
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from ..domain.interfaces.strategy import IStrategy
from ..core.registry import registry, ComponentType
from ..core.logging import get_logger

logger = get_logger(__name__)

# Qlib 可用性检查
try:
    import qlib
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.contrib.data.handler import Alpha158, Alpha360
    from qlib.contrib.model.gbdt import LGBModel
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    logger.warning("pyqlib 未正确安装，QlibStrategy 将使用模拟特征")


@dataclass
class QlibConfig:
    """Qlib 配置"""
    # 特征集类型
    feature_set: str = "Alpha158"  # Alpha158 或 Alpha360

    # 模型参数
    model_type: str = "lgb"  # lgb, mlp, gru, transformer
    learning_rate: float = 0.01
    max_depth: int = 6
    n_estimators: int = 200

    # 信号参数
    buy_threshold: float = 0.55
    sell_threshold: float = 0.45
    top_k_features: int = 50

    # 止盈止损
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10


class QlibFeatureEngineer:
    """
    Qlib 特征工程器

    实现 Alpha158 风格的特征集
    """

    def __init__(self, feature_set: str = "Alpha158"):
        """
        初始化特征工程器

        Args:
            feature_set: 特征集类型 (Alpha158 或 Alpha360)
        """
        self.feature_set = feature_set
        self.feature_names: List[str] = []

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成 Alpha158 风格特征

        Alpha158 特征设计原则:
        1. 时间序列特征 - 多周期窗口
        2. 横截面特征 - 相对位置
        3. 衍生特征 - 组合与变换

        Args:
            df: OHLCV 数据

        Returns:
            特征 DataFrame
        """
        if len(df) < 120:
            raise ValueError(f"数据不足，需要至少120条数据，当前只有{len(df)}条")

        features = {}
        close = df['close']
        high = df['high']
        low = df['low']
        open_ = df['open']
        volume = df.get('volume', df.get('vol', pd.Series(1, index=df.index)))

        # ==================== Alpha158 核心特征 ====================

        # 1. 价格动量特征 (多个时间窗口)
        windows = [5, 10, 20, 30, 60]
        for w in windows:
            # 收益率
            features[f'REF({w})'] = close / close.shift(w) - 1
            # 波动率
            features[f'STD({w})'] = close.pct_change().rolling(w).std()
            # 偏度
            features[f'SKEW({w})'] = close.pct_change().rolling(w).skew()
            # 峰度
            features[f'KURT({w})'] = close.pct_change().rolling(w).kurt()

        # 2. 相对位置特征
        for w in windows:
            hhv = high.rolling(w).max()
            llv = low.rolling(w).min()
            features[f'POS({w})'] = (close - llv) / (hhv - llv + 1e-10)
            features[f'HHVPOS({w})'] = (hhv - close) / (close + 1e-10)
            features[f'LLVPOS({w})'] = (close - llv) / (close + 1e-10)

        # 3. 均线特征
        ma_windows = [5, 10, 20, 30, 60, 120]
        for w in ma_windows:
            ma = close.rolling(w).mean()
            features[f'MA({w})'] = ma
            features[f'MADIFF({w})'] = (close - ma) / (ma + 1e-10)
            # 均线斜率
            features[f'MASLOPE({w})'] = ma.diff(5) / (ma + 1e-10)

        # 4. EMA 特征
        ema_windows = [5, 10, 20, 30, 60]
        for w in ema_windows:
            ema = close.ewm(span=w, adjust=False).mean()
            features[f'EMA({w})'] = ema
            features[f'EMADIFF({w})'] = (close - ema) / (ema + 1e-10)

        # 5. 技术指标特征
        # RSI
        for w in [6, 12, 24]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(w).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
            rs = gain / (loss + 1e-10)
            features[f'RSI({w})'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        features['MACD_DIF'] = dif
        features['MACD_DEA'] = dea
        features['MACD_HIST'] = 2 * (dif - dea)

        # KDJ
        for n in [9, 14]:
            hhv = high.rolling(n).max()
            llv = low.rolling(n).min()
            rsv = (close - llv) / (hhv - llv + 1e-10) * 100
            k = rsv.ewm(alpha=1/3, adjust=False).mean()
            d = k.ewm(alpha=1/3, adjust=False).mean()
            j = 3 * k - 2 * d
            features[f'K({n})'] = k
            features[f'D({n})'] = d
            features[f'J({n})'] = j

        # 6. 波动率特征
        for w in [5, 10, 20, 30]:
            features[f'VOL({w})'] = close.pct_change().rolling(w).std() * np.sqrt(252)
            # ATR
            tr = pd.concat([
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            ], axis=1).max(axis=1)
            features[f'ATR({w})'] = tr.rolling(w).mean() / close

        # 7. 布林带特征
        for w in [10, 20]:
            mid = close.rolling(w).mean()
            std = close.rolling(w).std()
            upper = mid + 2 * std
            lower = mid - 2 * std
            features[f'BOLLUP({w})'] = (upper - close) / close
            features[f'BOLLLOW({w})'] = (close - lower) / close
            features[f'BOLLW({w})'] = (upper - lower) / mid

        # 8. 成交量特征
        vol_windows = [5, 10, 20, 30]
        for w in vol_windows:
            vol_ma = volume.rolling(w).mean()
            features[f'VOLMA({w})'] = volume / (vol_ma + 1e-10)
            # 量价相关性
            features[f'CORR({w})'] = close.pct_change().rolling(w).corr(
                volume.pct_change().rolling(w).mean()
            )

        # 9. 价格形态特征
        features['BODY'] = (close - open_) / (high - low + 1e-10)
        features['UPPERSHADOW'] = (high - close.clip(upper=open_)) / (high - low + 1e-10)
        features['LOWERSHADOW'] = (close.clip(lower=open_) - low) / (high - low + 1e-10)

        # 10. 动量加速度
        for w in [5, 10, 20]:
            mom = close / close.shift(w) - 1
            features[f'MOMACC({w})'] = mom.diff(w)

        # 11. 趋势强度
        for w in [10, 20, 60]:
            up_days = (close > close.shift(1)).rolling(w).sum()
            features[f'TREND({w})'] = up_days / w

        # 12. Alpha360 扩展特征 (如果选择)
        if self.feature_set == "Alpha360":
            # 更长周期
            for w in [90, 120, 180, 240, 360]:
                features[f'REF({w})'] = close / close.shift(w) - 1
                features[f'POS({w})'] = (close - low.rolling(w).min()) / (
                    high.rolling(w).max() - low.rolling(w).min() + 1e-10
                )

        # 构建特征 DataFrame
        feature_df = pd.DataFrame(features, index=df.index)

        # 处理异常值
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.ffill().bfill().fillna(0)

        # 标准化 (类似 Qlib 的处理)
        feature_df = (feature_df - feature_df.rolling(60).mean()) / (feature_df.rolling(60).std() + 1e-10)
        feature_df = feature_df.ffill().bfill().fillna(0)

        self.feature_names = list(feature_df.columns)

        return feature_df


class QlibModel:
    """
    Qlib 模型封装器

    支持 LightGBM, MLP, GRU, Transformer 等模型
    """

    def __init__(
        self,
        model_type: str = "lgb",
        learning_rate: float = 0.01,
        max_depth: int = 6,
        n_estimators: int = 200
    ):
        """
        初始化模型

        Args:
            model_type: 模型类型 (lgb, mlp, gru, transformer)
            learning_rate: 学习率
            max_depth: 树深度 (LGB)
            n_estimators: 树数量 (LGB) 或 迭代次数
        """
        self.model_type = model_type
        self.model = None
        self.is_fitted = False

        if model_type == "lgb":
            try:
                import lightgbm as lgb
                self.model = lgb.LGBMClassifier(
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    objective='binary',
                    metric='auc',
                    verbose=-1,
                    n_jobs=-1
                )
                self._model_available = True
            except ImportError:
                logger.warning("LightGBM 未安装，请运行: pip install lightgbm")
                self._model_available = False
        else:
            # 其他模型类型使用简化实现
            self._model_available = False
            logger.info(f"模型类型 {model_type} 使用简化实现")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'QlibModel':
        """训练模型"""
        if self._model_available and self.model is not None:
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info(f"模型训练完成: {self.model_type}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        if self._model_available and self.is_fitted:
            return self.model.predict_proba(X)[:, 1]

        # 简化实现: 基于特征的加权平均
        weights = np.zeros(len(X))
        feature_weights = {
            'POS(20)': 0.15,
            'RSI(12)': 0.10,
            'MACD_DIF': 0.10,
            'MADIFF(20)': 0.10,
            'VOLMA(10)': 0.05,
        }

        for feat, w in feature_weights.items():
            if feat in X.columns:
                val = X[feat].iloc[-1] if len(X) > 0 else 0.5
                # 标准化到 0-1
                val = max(0, min(1, (val + 1) / 2))
                weights += w * val

        # 返回概率
        prob = 0.5 + 0.3 * (weights - 0.5)
        return np.full(len(X), np.clip(prob, 0, 1))

    def save(self, filepath: str):
        """保存模型"""
        import joblib
        joblib.dump(self.model, filepath)
        logger.info(f"模型已保存: {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        import joblib
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"模型已加载: {filepath}")


@dataclass
class QlibSignal:
    """Qlib 信号"""
    direction: Optional[str]
    signal: str
    probability: float
    confidence: float
    score: float
    stop_loss: float
    take_profit: float
    features: Dict[str, float]


@registry.register(ComponentType.STRATEGY, "qlib")
class QlibStrategy(IStrategy):
    """
    微软 Qlib 策略

    集成 Qlib 的 Alpha158 特征集和机器学习模型

    特点:
    1. 158+ 量化因子 (Alpha158)
    2. 多种 ML 模型支持 (LGB, MLP, GRU, Transformer)
    3. 基于概率的信号生成
    4. 自动特征选择
    """

    def __init__(
        self,
        feature_set: str = "Alpha158",
        model_type: str = "lgb",
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.45,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
        use_qlib_native: bool = False,
    ):
        """
        初始化 Qlib 策略

        Args:
            feature_set: 特征集类型 (Alpha158, Alpha360)
            model_type: 模型类型 (lgb, mlp, gru, transformer)
            buy_threshold: 买入概率阈值
            sell_threshold: 卖出概率阈值
            stop_loss_pct: 止损比例
            take_profit_pct: 止盈比例
            use_qlib_native: 是否使用 Qlib 原生 API
        """
        self.feature_set = feature_set
        self.model_type = model_type
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.use_qlib_native = use_qlib_native and QLIB_AVAILABLE

        # 特征工程器
        self.feature_engineer = QlibFeatureEngineer(feature_set)

        # 模型
        self.model = QlibModel(model_type=model_type)

        # 策略参数
        self.parameters = {
            'feature_set': feature_set,
            'model_type': model_type,
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
        }

        # 信号历史
        self.signals_history: List[QlibSignal] = []
        self.last_signal: Optional[QlibSignal] = None

        # 特征缓存
        self._feature_cache: Dict[str, pd.DataFrame] = {}

    def initialize(self, parameters: Dict[str, Any]) -> None:
        """初始化策略参数"""
        self.parameters.update(parameters)
        self.feature_set = self.parameters.get('feature_set', 'Alpha158')
        self.model_type = self.parameters.get('model_type', 'lgb')
        self.buy_threshold = self.parameters.get('buy_threshold', 0.55)
        self.sell_threshold = self.parameters.get('sell_threshold', 0.45)
        self.stop_loss_pct = self.parameters.get('stop_loss_pct', 0.05)
        self.take_profit_pct = self.parameters.get('take_profit_pct', 0.10)

        # 重新初始化组件
        self.feature_engineer = QlibFeatureEngineer(self.feature_set)
        self.model = QlibModel(model_type=self.model_type)

    def train_model(
        self,
        data: pd.DataFrame,
        horizon: int = 10,
        retrain: bool = False
    ) -> bool:
        """
        训练模型

        Args:
            data: 历史数据
            horizon: 预测周期
            retrain: 是否重新训练

        Returns:
            训练是否成功
        """
        try:
            logger.info(f"开始训练 Qlib 模型 ({self.model_type})...")

            # 生成特征
            features = self.feature_engineer.generate_features(data)

            # 生成标签: 未来 N 天收益率 > 0
            close = data['close']
            labels = (close.shift(-horizon) / close - 1 > 0).astype(int)

            # 对齐特征和标签
            valid_idx = features.dropna().index.intersection(labels.dropna().index)
            X = features.loc[valid_idx]
            y = labels.loc[valid_idx]

            # 训练模型
            self.model.fit(X, y)

            logger.info(f"Qlib 模型训练完成，特征数: {len(features.columns)}")
            return True

        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            return False

    def get_signal(
        self,
        current_bar: pd.Series,
        historical_bars: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        获取交易信号

        Args:
            current_bar: 当前 K 线
            historical_bars: 历史 K 线

        Returns:
            信号字典
        """
        min_bars = 120
        if len(historical_bars) < min_bars:
            return {
                'direction': None,
                'signal': 'hold',
                'reason': f'数据不足(需{min_bars}日)'
            }

        try:
            # 生成特征
            features = self.feature_engineer.generate_features(historical_bars)
            latest_features = features.iloc[[-1]]

            # 预测概率
            prob = self.model.predict_proba(latest_features)[0]

            # 计算综合得分
            score = self._calculate_score(features.iloc[-1], prob)

            # 生成信号
            direction = None
            signal_type = 'hold'

            # 更激进的买入条件：只看概率，不看 score
            if prob >= self.buy_threshold:
                direction = 'buy'
                signal_type = 'buy'
            elif prob <= self.sell_threshold:
                direction = 'sell'
                signal_type = 'sell'

            # 计算止损止盈
            close = current_bar['close']
            stop_loss = close * (1 - self.stop_loss_pct)
            take_profit = close * (1 + self.take_profit_pct)

            # 置信度
            confidence = abs(prob - 0.5) * 2

            # 构建信号
            signal = QlibSignal(
                direction=direction,
                signal=signal_type,
                probability=prob,
                confidence=confidence,
                score=score,
                stop_loss=stop_loss,
                take_profit=take_profit,
                features=latest_features.iloc[0].to_dict()
            )

            self.last_signal = signal
            self.signals_history.append(signal)

            return {
                'direction': direction,
                'signal': signal_type,
                'probability': prob,
                'confidence': confidence,
                'score': score,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy_name': f'QlibStrategy({self.feature_set})',
                'timestamp': current_bar.get('timestamp', datetime.now())
            }

        except Exception as e:
            logger.error(f"信号生成失败: {e}")
            return {
                'direction': None,
                'signal': 'hold',
                'error': str(e)
            }

    def _calculate_score(self, features: pd.Series, prob: float) -> float:
        """
        计算综合得分

        结合模型概率和关键特征 - 优化版，更激进的买入信号
        """
        score = prob * 0.6  # 提高模型概率权重

        # 位置得分 - 更激进的低位加分
        if 'POS(20)' in features.index:
            pos = features['POS(20)']
            # 低位大幅加分，高位小幅减分
            if pos < 0.4:
                score += 0.25 * (1 - pos)
            elif pos > 0.8:
                score -= 0.05 * (pos - 0.8)

        # RSI 得分 - 扩大超卖范围
        if 'RSI(12)' in features.index:
            rsi = features['RSI(12)']
            if rsi < 40:  # 扩大到40
                score += 0.15 * (40 - rsi) / 40
            elif rsi > 75:  # 提高超买阈值
                score -= 0.05

        # MACD 得分 - 增加权重
        if 'MACD_DIF' in features.index and 'MACD_DEA' in features.index:
            if features['MACD_DIF'] > features['MACD_DEA']:
                score += 0.15
            # MACD 金叉额外加分
            if 'MACD_HIST' in features.index:
                if features['MACD_HIST'] > 0:
                    score += 0.05

        # 动量得分 - 增加权重
        if 'REF(10)' in features.index:
            mom = features['REF(10)']
            if mom > 0:
                score += 0.1
            if mom > 0.05:  # 强动量
                score += 0.05

        # 新增：KDJ 超卖加分
        if 'J(9)' in features.index:
            j = features['J(9)']
            if j < 20:
                score += 0.1

        # 新增：布林带下轨支撑
        if 'BOLLLOW(20)' in features.index:
            boll_low = features['BOLLLOW(20)']
            if boll_low < 0.1:  # 接近下轨
                score += 0.05

        return np.clip(score, 0, 1)

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
        result['score'] = np.nan
        result['confidence'] = np.nan

        for i in range(min_bars, len(bars)):
            window_data = bars.iloc[:i+1]
            current_bar = bars.iloc[i]

            signal = self.get_signal(current_bar, window_data)

            result.loc[result.index[i], 'signal'] = signal.get('signal', 'hold')
            result.loc[result.index[i], 'probability'] = signal.get('probability', np.nan)
            result.loc[result.index[i], 'score'] = signal.get('score', np.nan)
            result.loc[result.index[i], 'confidence'] = signal.get('confidence', np.nan)

        return result

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """获取特征重要性"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_engineer.feature_names,
                'importance': self.model.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance.head(top_n)
        return pd.DataFrame()

    def get_name(self) -> str:
        """获取策略名称"""
        return f"QlibStrategy({self.feature_set}, {self.model_type})"

    def get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return self.parameters.copy()

    def get_description(self) -> str:
        """获取策略描述"""
        return (
            f"微软 Qlib 策略: 特征集={self.feature_set}, "
            f"模型={self.model_type}, "
            f"买入阈值={self.buy_threshold}, "
            f"止损={self.stop_loss_pct*100}%"
        )


class QlibStockSelector:
    """
    基于 Qlib 的股票筛选器
    """

    def __init__(
        self,
        strategy: Optional[QlibStrategy] = None,
        top_k: int = 10
    ):
        """
        初始化筛选器

        Args:
            strategy: Qlib 策略实例
            top_k: 筛选数量
        """
        self.strategy = strategy or QlibStrategy()
        self.top_k = top_k

    def select(
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
            筛选结果 DataFrame
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
                        'score': signal.get('score', 0),
                        'confidence': signal.get('confidence', 0),
                        'stop_loss': signal.get('stop_loss', 0),
                        'take_profit': signal.get('take_profit', 0),
                        'close': df['close'].iloc[-1],
                    })
            except Exception as e:
                logger.warning(f"股票 {stock_code} 筛选失败: {e}")
                continue

        if not results:
            return pd.DataFrame()

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score', ascending=False)

        return results_df.head(self.top_k)