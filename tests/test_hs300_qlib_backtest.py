#!/usr/bin/env python
"""
沪深300 Qlib 模型回测脚本

使用沪深300成分股近五年数据训练 Qlib 模型：
- 近五年数据用于训练和验证 (前4年训练，第5年验证)
- 最近一年数据用于预测和回测

支持所有 Qlib 原生模型
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from quanttool.strategies.qlib.models import QlibModelFactory, QlibModelConfig
from quanttool.strategies.qlib.pytorch_models import create_pytorch_sequence_model
from quanttool.strategies.qlib.advanced_models import create_advanced_model
from quanttool.strategies.qlib.data_adapter import create_qlib_dataset_from_dataframe, create_ts_compatible_dataset
from quanttool.backtest.engine import BacktestEngine
from quanttool.infrastructure.data_providers.data_fetcher import AshareFetcher


# ============================================================================
# 配置
# ============================================================================

# 时间配置（调整为适应缓存数据量）
TRAIN_YEARS = 3      # 训练数据年数
VALID_YEARS = 1      # 验证数据年数
TEST_YEARS = 1       # 预测/回测数据年数
TOTAL_YEARS = TRAIN_YEARS + VALID_YEARS + TEST_YEARS  # 总共需要的数据年数

# 回测配置
INITIAL_CAPITAL = 100000.0
COMMISSION_RATE = 0.0003  # 万三

# PyTorch 模型配置
PYTORCH_EPOCHS = 50
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 256

# 策略参数
BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.10

# 支持的模型
ALL_MODELS = {
    # GBDT 系列
    'lgb': {'name': 'LightGBM', 'category': 'GBDT', 'fast': True},
    'xgboost': {'name': 'XGBoost', 'category': 'GBDT', 'fast': True},
    'catboost': {'name': 'CatBoost', 'category': 'GBDT', 'fast': True},
    'double_ensemble': {'name': 'DoubleEnsemble', 'category': 'GBDT', 'fast': True},

    # PyTorch 序列模型
    'lstm': {'name': 'LSTM', 'category': 'PyTorch序列', 'fast': False},
    'gru': {'name': 'GRU', 'category': 'PyTorch序列', 'fast': False},
    'alstm': {'name': 'Attention LSTM', 'category': 'PyTorch序列', 'fast': False},
    'transformer': {'name': 'Transformer', 'category': 'PyTorch序列', 'fast': False},
    'tcn': {'name': 'TCN', 'category': 'PyTorch序列', 'fast': False},

    # PyTorch 高级模型
    'gats': {'name': 'GATs', 'category': 'PyTorch高级', 'fast': False},
    'sfm': {'name': 'SFM', 'category': 'PyTorch高级', 'fast': False},
    'tabnet': {'name': 'TabNet', 'category': 'PyTorch高级', 'fast': False},
    'adarnn': {'name': 'ADARNN', 'category': 'PyTorch高级', 'fast': False},
    'add': {'name': 'ADD', 'category': 'PyTorch高级', 'fast': False},
    'hist': {'name': 'HIST', 'category': 'PyTorch高级', 'fast': False},
    'igmtf': {'name': 'IGMTF', 'category': 'PyTorch高级', 'fast': False},
    'krnn': {'name': 'KRNN', 'category': 'PyTorch高级', 'fast': False},
    'tra': {'name': 'TRA', 'category': 'PyTorch高级', 'fast': False},
    'tcts': {'name': 'TCTS', 'category': 'PyTorch高级', 'fast': False},
    'sandwich': {'name': 'Sandwich', 'category': 'PyTorch高级', 'fast': False},
}


# ============================================================================
# 数据获取
# ============================================================================

def get_hs300_constituents(use_cache: bool = True) -> list:
    """获取沪深300成分股列表，优先从缓存获取"""
    # 尝试从缓存目录获取
    if use_cache:
        cache_dir = project_root / '.cache' / 'incremental_data'
        if cache_dir.exists():
            cache_files = list(cache_dir.glob('*_stock_bar.parquet'))
            if cache_files:
                # 从文件名提取股票代码
                symbols = []
                for f in cache_files:
                    # 文件名格式: 000001_SZ_stock_bar.parquet
                    name = f.stem.replace('_stock_bar', '')
                    parts = name.rsplit('_', 1)
                    if len(parts) == 2:
                        symbols.append(parts[0])
                if symbols:
                    print(f"  从缓存获取 {len(symbols)} 只股票")
                    return symbols

    try:
        # 尝试从本地文件获取
        hs300_file = project_root / 'data' / 'hs300_constituents.csv'
        if hs300_file.exists():
            df = pd.read_csv(hs300_file)
            return df['code'].tolist()
    except Exception:
        pass

    # 沪深300部分代表性成分股
    return [
        '600519',  # 贵州茅台
        '601318',  # 中国平安
        '600036',  # 招商银行
        '601166',  # 兴业银行
        '600000',  # 浦发银行
        '601398',  # 工商银行
        '601288',  # 农业银行
        '600030',  # 中信证券
        '601888',  # 中国中免
        '600276',  # 恒瑞医药
        '000858',  # 五粮液
        '000333',  # 美的集团
        '002415',  # 海康威视
        '600900',  # 长江电力
        '601012',  # 隆基绿能
        '002594',  # 比亚迪
        '300750',  # 宁德时代
        '603259',  # 药明康德
        '600887',  # 伊利股份
        '000001',  # 平安银行
    ]


def fetch_stock_data(symbol: str, total_years: int = TOTAL_YEARS, use_cache: bool = True) -> pd.DataFrame:
    """获取股票历史数据，优先使用缓存"""

    # 标准化股票代码格式
    if symbol.startswith(('6', '5', '9')):
        cache_symbol = f"{symbol}_SH"
    else:
        cache_symbol = f"{symbol}_SZ"

    # 尝试从缓存读取
    if use_cache:
        cache_file = project_root / '.cache' / 'incremental_data' / f'{cache_symbol}_stock_bar.parquet'
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if 'timestamp' not in df.columns and 'time' in df.columns:
                    df = df.rename(columns={'time': 'timestamp'})
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp').reset_index(drop=True)
                print(f"  ✅ 从缓存读取 {len(df)} 条", end=" ")
                return df
            except Exception as e:
                print(f"  ⚠️ 缓存读取失败: {e}", end=" ")

    # 网络获取（备用）
    end_date = datetime.now().strftime('%Y-%m-%d')
    days = total_years * 365 + 100  # 额外获取一些数据

    try:
        df = AshareFetcher.get_price(
            code=symbol,
            end_date=end_date,
            count=days,
            frequency='1d'
        )

        if df.empty:
            return pd.DataFrame()

        # 标准化列名
        if 'timestamp' not in df.columns:
            if 'time' in df.columns:
                df = df.rename(columns={'time': 'timestamp'})

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    except Exception as e:
        print(f"  ❌ {symbol} 获取失败: {e}")
        return pd.DataFrame()


def split_data_by_time(df: pd.DataFrame) -> dict:
    """
    按时间划分数据

    近五年数据划分：
    - 训练集：前4年
    - 验证集：第5年
    - 测试集（预测）：最近1年
    """
    if len(df) < 252 * TOTAL_YEARS:
        return None

    # 计算切分点
    total_days = len(df)
    test_start = total_days - 252 * TEST_YEARS  # 最近1年开始
    valid_start = test_start - 252 * VALID_YEARS  # 验证集开始

    train_df = df.iloc[:valid_start].copy()
    valid_df = df.iloc[valid_start:test_start].copy()
    test_df = df.iloc[test_start:].copy()

    return {
        'train': train_df,
        'valid': valid_df,
        'test': test_df,
        'train_dates': (train_df['timestamp'].iloc[0], train_df['timestamp'].iloc[-1]),
        'valid_dates': (valid_df['timestamp'].iloc[0], valid_df['timestamp'].iloc[-1]),
        'test_dates': (test_df['timestamp'].iloc[0], test_df['timestamp'].iloc[-1]),
    }


# ============================================================================
# 特征工程
# ============================================================================

def generate_alpha158_features(df: pd.DataFrame) -> pd.DataFrame:
    """生成 Alpha158 风格特征"""
    features = {}
    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']
    volume = df.get('volume', df.get('vol', pd.Series(1, index=df.index)))

    # 保存原始时间戳索引
    original_index = df.index

    windows = [5, 10, 20, 30, 60]

    # 1. 价格动量特征
    for w in windows:
        features[f'REF({w})'] = close / close.shift(w) - 1
        features[f'STD({w})'] = close.pct_change().rolling(w).std()
        features[f'SKEW({w})'] = close.pct_change().rolling(w).skew()
        features[f'KURT({w})'] = close.pct_change().rolling(w).kurt()

    # 2. 相对位置特征
    for w in windows:
        hhv = high.rolling(w).max()
        llv = low.rolling(w).min()
        features[f'POS({w})'] = (close - llv) / (hhv - llv + 1e-10)

    # 3. 均线特征
    for w in [5, 10, 20, 30, 60, 120]:
        ma = close.rolling(w).mean()
        features[f'MA({w})'] = ma
        features[f'MADIFF({w})'] = (close - ma) / (ma + 1e-10)

    # 4. EMA 特征
    for w in [5, 10, 20, 30, 60]:
        ema = close.ewm(span=w, adjust=False).mean()
        features[f'EMA({w})'] = ema
        features[f'EMADIFF({w})'] = (close - ema) / (ema + 1e-10)

    # 5. RSI
    for w in [6, 12, 24]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
        rs = gain / (loss + 1e-10)
        features[f'RSI({w})'] = 100 - (100 / (1 + rs))

    # 6. MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    features['MACD_DIF'] = dif
    features['MACD_DEA'] = dea
    features['MACD_HIST'] = 2 * (dif - dea)

    # 7. KDJ
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

    # 8. 波动率特征
    for w in [5, 10, 20, 30]:
        features[f'VOL({w})'] = close.pct_change().rolling(w).std() * np.sqrt(252)
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        features[f'ATR({w})'] = tr.rolling(w).mean() / close

    # 9. 布林带特征
    for w in [10, 20]:
        mid = close.rolling(w).mean()
        std = close.rolling(w).std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        features[f'BOLLUP({w})'] = (upper - close) / close
        features[f'BOLLLOW({w})'] = (close - lower) / close

    # 10. 成交量特征
    for w in [5, 10, 20, 30]:
        vol_ma = volume.rolling(w).mean()
        features[f'VOLMA({w})'] = volume / (vol_ma + 1e-10)

    # 构建特征 DataFrame
    feature_df = pd.DataFrame(features, index=df.index)

    # 处理异常值
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.ffill().bfill().fillna(0)

    # 标准化
    feature_df = (feature_df - feature_df.rolling(60).mean()) / (feature_df.rolling(60).std() + 1e-10)
    feature_df = feature_df.ffill().bfill().fillna(0)

    # 恢复时间戳索引（如果原始数据有时间戳列）
    if 'timestamp' in df.columns:
        feature_df.index = pd.DatetimeIndex(df['timestamp'].values)

    return feature_df


def generate_labels(df: pd.DataFrame, horizon: int = 10) -> pd.Series:
    """生成标签：未来N天收益率"""
    close = df['close']
    returns = close.shift(-horizon) / close - 1
    # 转换为二分类标签：收益率>0为1，否则为0
    labels = (returns > 0).astype(float)

    # 恢复时间戳索引（如果原始数据有时间戳列）
    if 'timestamp' in df.columns:
        labels.index = pd.DatetimeIndex(df['timestamp'].values)

    return labels


# ============================================================================
# 模型训练和预测
# ============================================================================

def train_and_predict(
    model_type: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int = PYTORCH_EPOCHS,
) -> dict:
    """训练模型并进行预测"""

    # 生成特征和标签
    train_features = generate_alpha158_features(train_df)
    train_labels = generate_labels(train_df)

    valid_features = generate_alpha158_features(valid_df)
    valid_labels = generate_labels(valid_df)

    test_features = generate_alpha158_features(test_df)

    # 去除 NaN
    valid_idx = train_features.dropna().index.intersection(train_labels.dropna().index)
    train_features = train_features.loc[valid_idx]
    train_labels = train_labels.loc[valid_idx]

    valid_idx = valid_features.dropna().index.intersection(valid_labels.dropna().index)
    valid_features = valid_features.loc[valid_idx]
    valid_labels = valid_labels.loc[valid_idx]

    # 创建配置
    config = QlibModelConfig(
        model_type=model_type,
        epochs=epochs,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        batch_size=BATCH_SIZE,
        learning_rate=0.01,
        early_stopping_rounds=10,
    )

    # 创建模型
    try:
        model = QlibModelFactory.create(model_type, config)
    except Exception as e:
        print(f"  模型创建失败: {e}")
        return {'success': False, 'error': str(e)}

    # 合并训练和验证数据
    all_features = pd.concat([train_features, valid_features])
    all_labels = pd.concat([train_labels, valid_labels])

    # 训练模型
    try:
        model.fit(all_features, all_labels)
    except Exception as e:
        print(f"  训练失败: {e}")
        return {'success': False, 'error': str(e)}

    # 预测
    try:
        predictions = model.predict(test_features)
        if predictions is None or len(predictions) == 0:
            return {'success': False, 'error': '预测结果为空'}
    except Exception as e:
        print(f"  预测失败: {e}")
        return {'success': False, 'error': str(e)}

    return {
        'success': True,
        'predictions': predictions,
        'test_features': test_features,
        'model': model,
    }


# ============================================================================
# 回测
# ============================================================================

class SimpleQlibStrategy:
    """简化的 Qlib 策略，用于回测"""

    def __init__(
        self,
        predictions: np.ndarray,
        buy_threshold: float = BUY_THRESHOLD,
        sell_threshold: float = SELL_THRESHOLD,
        stop_loss_pct: float = STOP_LOSS_PCT,
        take_profit_pct: float = TAKE_PROFIT_PCT,
    ):
        self.predictions = predictions
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position = 0
        self.entry_price = 0

    def get_signal(self, current_bar, historical_bars, **kwargs):
        """获取交易信号"""
        idx = kwargs.get('bar_index', 0)

        if idx >= len(self.predictions):
            return {'direction': None, 'signal': 'hold'}

        pred = self.predictions[idx]

        # 根据预测概率生成信号
        if pred > self.buy_threshold and self.position == 0:
            self.position = 1
            self.entry_price = current_bar['close']
            return {
                'direction': 'buy',
                'signal': 'buy',
                'stop_loss': self.entry_price * (1 - self.stop_loss_pct),
                'take_profit': self.entry_price * (1 + self.take_profit_pct),
            }
        elif pred < self.sell_threshold and self.position > 0:
            self.position = 0
            return {'direction': 'sell', 'signal': 'sell'}
        else:
            return {'direction': None, 'signal': 'hold'}


def run_backtest(test_df: pd.DataFrame, predictions: np.ndarray) -> dict:
    """运行回测"""
    symbol = 'stock'
    strategy = SimpleQlibStrategy(predictions)

    engine = BacktestEngine()
    engine.set_initial_cash(INITIAL_CAPITAL)
    engine.set_commission_rate(COMMISSION_RATE)
    engine.set_t_plus_1(True)

    data = {symbol: test_df.copy()}

    try:
        result = engine.run_backtest(
            strategy=strategy,
            data=data,
            start_date=test_df['timestamp'].iloc[0],
            end_date=test_df['timestamp'].iloc[-1]
        )

        return {
            'success': True,
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ============================================================================
# 主程序
# ============================================================================

def run_single_stock_backtest(
    symbol: str,
    model_type: str,
    stock_data: dict,
    epochs: int = PYTORCH_EPOCHS,
) -> dict:
    """对单只股票运行回测"""
    df = stock_data.get(symbol)
    if df is None or len(df) < 252 * TOTAL_YEARS:
        return {'success': False, 'error': '数据不足'}

    # 划分数据
    split = split_data_by_time(df)
    if split is None:
        return {'success': False, 'error': '数据划分失败'}

    # 训练和预测
    result = train_and_predict(
        model_type,
        split['train'],
        split['valid'],
        split['test'],
        epochs=epochs
    )

    if not result.get('success'):
        return result

    # 回测
    backtest_result = run_backtest(split['test'], result['predictions'])

    return backtest_result


def main():
    """主函数"""
    print("=" * 80)
    print("沪深300 Qlib 模型回测")
    print("=" * 80)
    print(f"\n数据划分:")
    print(f"  训练集: {TRAIN_YEARS}年")
    print(f"  验证集: {VALID_YEARS}年")
    print(f"  测试集: {TEST_YEARS}年 (预测)")
    print(f"\n模型参数:")
    print(f"  PyTorch epochs: {PYTORCH_EPOCHS}")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")

    # 获取股票列表
    print("\n获取沪深300成分股...")
    stocks = get_hs300_constituents()
    print(f"  共 {len(stocks)} 只股票")

    # 获取数据
    print("\n获取股票数据...")
    stock_data = {}
    for symbol in stocks:
        print(f"  获取 {symbol}...", end=" ")
        df = fetch_stock_data(symbol, TOTAL_YEARS)
        if not df.empty and len(df) >= 252 * TOTAL_YEARS:
            stock_data[symbol] = df
            print(f"✅ {len(df)} 条")
        else:
            print(f"❌ 数据不足 (需要 {252 * TOTAL_YEARS} 条)")

    if not stock_data:
        print("❌ 没有有效数据")
        return

    print(f"\n有效股票: {len(stock_data)} 只")

    # 选择一只股票进行测试
    test_symbol = list(stock_data.keys())[0]
    print(f"\n使用 {test_symbol} 进行模型对比测试...")

    # 测试所有模型
    all_results = []

    # 先测试 GBDT 模型
    print("\n" + "=" * 80)
    print("测试 GBDT 模型")
    print("=" * 80)

    gbdt_models = {k: v for k, v in ALL_MODELS.items() if v['category'] == 'GBDT'}

    for model_type, model_info in gbdt_models.items():
        print(f"\n测试 {model_info['name']} ({model_type})...", end=" ")
        start_time = time.time()

        try:
            result = run_single_stock_backtest(test_symbol, model_type, stock_data)
            elapsed = time.time() - start_time

            if result.get('success'):
                print(f"✅ 年化收益: {result['annual_return']*100:.2f}% (耗时: {elapsed:.1f}s)")
                result['model_type'] = model_type
                result['model_name'] = model_info['name']
                result['category'] = model_info['category']
                all_results.append(result)
            else:
                print(f"❌ 失败: {result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"❌ 异常: {e}")

    # 测试 PyTorch 序列模型
    print("\n" + "=" * 80)
    print("测试 PyTorch 序列模型")
    print("=" * 80)

    pytorch_seq_models = {k: v for k, v in ALL_MODELS.items() if v['category'] == 'PyTorch序列'}

    for model_type, model_info in pytorch_seq_models.items():
        print(f"\n测试 {model_info['name']} ({model_type})...", end=" ")
        start_time = time.time()

        try:
            result = run_single_stock_backtest(test_symbol, model_type, stock_data, epochs=PYTORCH_EPOCHS)
            elapsed = time.time() - start_time

            if result.get('success'):
                print(f"✅ 年化收益: {result['annual_return']*100:.2f}% (耗时: {elapsed:.1f}s)")
                result['model_type'] = model_type
                result['model_name'] = model_info['name']
                result['category'] = model_info['category']
                all_results.append(result)
            else:
                print(f"❌ 失败: {result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"❌ 异常: {e}")

    # 测试 PyTorch 高级模型
    print("\n" + "=" * 80)
    print("测试 PyTorch 高级模型")
    print("=" * 80)

    pytorch_adv_models = {k: v for k, v in ALL_MODELS.items() if v['category'] == 'PyTorch高级'}

    for model_type, model_info in pytorch_adv_models.items():
        print(f"\n测试 {model_info['name']} ({model_type})...", end=" ")
        start_time = time.time()

        try:
            result = run_single_stock_backtest(test_symbol, model_type, stock_data, epochs=PYTORCH_EPOCHS)
            elapsed = time.time() - start_time

            if result.get('success'):
                print(f"✅ 年化收益: {result['annual_return']*100:.2f}% (耗时: {elapsed:.1f}s)")
                result['model_type'] = model_type
                result['model_name'] = model_info['name']
                result['category'] = model_info['category']
                all_results.append(result)
            else:
                print(f"❌ 失败: {result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"❌ 异常: {e}")

    # 结果汇总
    print("\n" + "=" * 80)
    print("回测结果汇总")
    print("=" * 80)

    if all_results:
        # 按年化收益排序
        all_results.sort(key=lambda x: x.get('annual_return', -999), reverse=True)

        print("\n📊 模型收益排名:")
        print("-" * 80)
        print(f"{'排名':<4} {'模型':<20} {'类型':<12} {'年化收益':>10} {'夏普':>8} {'胜率':>8} {'交易':>6}")
        print("-" * 80)

        for i, r in enumerate(all_results, 1):
            print(f"{i:<4} {r.get('model_name', r.get('model_type')):<20} "
                  f"{r.get('category', 'N/A'):<12} "
                  f"{r.get('annual_return', 0)*100:>9.2f}% "
                  f"{r.get('sharpe_ratio', 0):>8.2f} "
                  f"{r.get('win_rate', 0)*100:>7.1f}% "
                  f"{r.get('total_trades', 0):>6}")

        print("-" * 80)

        # 最佳模型
        best = all_results[0]
        print(f"\n🏆 最佳模型: {best.get('model_name', best.get('model_type'))}")
        print(f"   年化收益率: {best.get('annual_return', 0)*100:.2f}%")
        print(f"   夏普比率: {best.get('sharpe_ratio', 0):.2f}")
        print(f"   最大回撤: {best.get('max_drawdown', 0)*100:.2f}%")
        print(f"   胜率: {best.get('win_rate', 0)*100:.1f}%")
        print(f"   交易次数: {best.get('total_trades', 0)}")

    return all_results


if __name__ == "__main__":
    results = main()
