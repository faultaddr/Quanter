#!/usr/bin/env python
"""
Qlib 官方训练流程示例

展示如何使用 Quanter 系统的数据进行 qlib 官方训练：
1. 将数据转换为 qlib 二进制格式
2. 使用 Alpha158/Alpha360 官方特征
3. 使用 qlib 原生模型进行训练
4. 回测和评估

使用方法:
    python examples/qlib_official_training.py --mode dump    # 转换数据
    python examples/qlib_official_training.py --mode train   # 训练模型
    python examples/qlib_official_training.py --mode backtest # 回测
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from quanttool.infrastructure.data_providers.qlib_data_converter import (
    QlibDataConverter,
    QlibDataConfig,
    QlibTrainingPipeline,
    Alpha158Features,
    Alpha360Features,
)
from quanttool.core.logging import get_logger

logger = get_logger(__name__)


def dump_qlib_data(args):
    """
    步骤1：将数据转换为 qlib 二进制格式

    完全遵循 qlib 官方数据结构:
    - calendars/day.txt      # 交易日历
    - instruments/all.txt    # 股票列表
    - features/{symbol}/     # 每只股票的数据
    """
    print("=" * 80)
    print("Qlib 数据转换")
    print("=" * 80)

    # 创建数据转换器
    config = QlibDataConfig(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        feature_type=args.feature_type,
    )
    converter = QlibDataConverter(config)

    # 获取可用的股票代码
    symbols = converter.get_available_symbols()
    print(f"\n缓存中共有 {len(symbols)} 只股票")

    if not symbols:
        print("错误：没有可用的缓存数据")
        print("请先运行数据获取命令，例如:")
        print("  python -m quanttool.cli.main data fetch-stock 000001 365")
        return

    # 转换数据
    result = converter.dump_data(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        feature_type=args.feature_type,
    )

    print(f"\n转换完成:")
    print(f"  股票数量: {result['symbol_count']}")
    print(f"  交易日数: {result['date_count']}")
    print(f"  特征数量: {result['feature_count']}")
    print(f"  输出目录: {result['output_dir']}")

    # 显示下一步操作
    print(f"\n下一步：使用 qlib 初始化数据")
    print(f"  import qlib")
    print(f"  qlib.init(provider_uri='{args.output_dir}')")


def train_qlib_model(args):
    """
    步骤2：使用 qlib 官方流程训练模型
    """
    print("=" * 80)
    print("Qlib 模型训练")
    print("=" * 80)

    # 创建数据转换器
    config = QlibDataConfig(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        feature_type=args.feature_type,
    )
    converter = QlibDataConverter(config)
    pipeline = QlibTrainingPipeline(converter)

    # 获取股票列表
    symbols = converter.get_available_symbols()
    if args.symbols:
        symbols = [s for s in symbols if s in args.symbols.split(',')]

    print(f"\n训练参数:")
    print(f"  股票数量: {len(symbols)}")
    print(f"  模型类型: {args.model}")
    print(f"  特征类型: {args.feature_type}")
    print(f"  时间范围: {args.start_date} ~ {args.end_date}")

    # 训练模型
    if args.model in ['lgb', 'xgboost', 'catboost', 'double_ensemble']:
        result = pipeline.train_gbdt_model(
            symbols=symbols,
            model_type=args.model,
            feature_type=args.feature_type,
            start_date=args.start_date,
            end_date=args.end_date,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
        )
    else:
        result = pipeline.train_pytorch_model(
            symbols=symbols,
            model_type=args.model,
            feature_type=args.feature_type,
            start_date=args.start_date,
            end_date=args.end_date,
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        )

    if result['success']:
        print(f"\n训练完成:")
        print(f"  特征数量: {result['feature_count']}")
        print(f"  样本数量: {result['sample_count']}")

        # 保存模型
        model_path = Path(args.output_dir) / f"model_{args.model}.pkl"
        result['model'].save(str(model_path))
        print(f"  模型保存: {model_path}")
    else:
        print(f"训练失败: {result.get('error', 'Unknown')}")


def backtest_qlib_strategy(args):
    """
    步骤3：回测 qlib 模型策略
    """
    print("=" * 80)
    print("Qlib 模型回测")
    print("=" * 80)

    from quanttool.backtest.engine import BacktestEngine
    from quanttool.strategies.qlib.models import QlibModelFactory

    # 加载模型
    model_path = Path(args.output_dir) / f"model_{args.model}.pkl"
    if not model_path.exists():
        print(f"错误：模型文件不存在: {model_path}")
        print("请先运行训练命令")
        return

    # 创建数据转换器
    config = QlibDataConfig(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
    converter = QlibDataConverter(config)

    # 获取测试股票数据
    symbols = converter.get_available_symbols()
    test_symbol = symbols[0] if symbols else None
    if not test_symbol:
        print("错误：没有可用的股票数据")
        return

    df = converter.load_stock_data(test_symbol)
    if df.empty:
        print(f"错误：无法加载股票数据: {test_symbol}")
        return

    # 过滤时间范围
    if args.start_date:
        df = df[df.index >= pd.to_datetime(args.start_date)]
    if args.end_date:
        df = df[df.index <= pd.to_datetime(args.end_date)]

    print(f"\n回测参数:")
    print(f"  股票: {test_symbol}")
    print(f"  时间范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  数据条数: {len(df)}")

    # 生成特征
    features = Alpha158Features.generate(df) if args.feature_type == 'alpha158' else Alpha360Features.generate(df)

    # 加载模型并预测
    from quanttool.strategies.qlib.models import QlibModelConfig
    model_config = QlibModelConfig(model_type=args.model)
    model = QlibModelFactory.create(args.model, model_config)
    model.load(str(model_path))

    predictions = model.predict(features)

    # 创建策略
    class QlibStrategy:
        def __init__(self, predictions, buy_threshold=0.55, sell_threshold=0.45):
            self.predictions = predictions
            self.buy_threshold = buy_threshold
            self.sell_threshold = sell_threshold
            self.position = 0
            self.entry_price = 0

        def get_signal(self, current_bar, historical_bars, **kwargs):
            idx = kwargs.get('bar_index', 0)
            if idx >= len(self.predictions):
                return {'direction': None, 'signal': 'hold'}

            pred = self.predictions[idx]

            if pred > self.buy_threshold and self.position == 0:
                self.position = 1
                self.entry_price = current_bar['close']
                return {
                    'direction': 'buy',
                    'signal': 'buy',
                    'stop_loss': self.entry_price * 0.95,
                    'take_profit': self.entry_price * 1.10,
                }
            elif pred < self.sell_threshold and self.position > 0:
                self.position = 0
                return {'direction': 'sell', 'signal': 'sell'}

            return {'direction': None, 'signal': 'hold'}

    strategy = QlibStrategy(predictions)

    # 运行回测
    engine = BacktestEngine()
    engine.set_initial_cash(args.capital)
    engine.set_commission_rate(0.0003)
    engine.set_t_plus_1(True)

    result = engine.run_backtest(
        strategy=strategy,
        data={test_symbol: df},
        start_date=df.index[0],
        end_date=df.index[-1]
    )

    print(f"\n回测结果:")
    print(f"  总收益率: {result.total_return * 100:.2f}%")
    print(f"  年化收益率: {result.annual_return * 100:.2f}%")
    print(f"  夏普比率: {result.sharpe_ratio:.2f}")
    print(f"  最大回撤: {result.max_drawdown * 100:.2f}%")
    print(f"  胜率: {result.win_rate * 100:.1f}%")
    print(f"  交易次数: {result.total_trades}")


def run_full_pipeline(args):
    """
    完整流程：数据转换 -> 训练 -> 回测
    """
    print("=" * 80)
    print("Qlib 官方训练流程 - 完整运行")
    print("=" * 80)

    # 步骤1：转换数据
    print("\n[1/3] 转换数据...")
    dump_qlib_data(args)

    # 步骤2：训练模型
    print("\n[2/3] 训练模型...")
    train_qlib_model(args)

    # 步骤3：回测
    print("\n[3/3] 回测...")
    backtest_qlib_strategy(args)


def main():
    parser = argparse.ArgumentParser(description='Qlib 官方训练流程')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['dump', 'train', 'backtest', 'full'],
                       help='运行模式')
    parser.add_argument('--cache-dir', type=str, default='.cache/incremental_data',
                       help='缓存目录')
    parser.add_argument('--output-dir', type=str, default='qlib_data/cn_data',
                       help='输出目录')
    parser.add_argument('--feature-type', type=str, default='alpha158',
                       choices=['alpha158', 'alpha360'],
                       help='特征类型')
    parser.add_argument('--model', type=str, default='lgb',
                       help='模型类型 (lgb, xgboost, catboost, lstm, gru, transformer)')
    parser.add_argument('--start-date', type=str, default=None,
                       help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--symbols', type=str, default=None,
                       help='股票代码列表 (逗号分隔)')

    # 模型参数
    parser.add_argument('--n-estimators', type=int, default=200)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2)

    # 回测参数
    parser.add_argument('--capital', type=float, default=100000.0,
                       help='初始资金')

    args = parser.parse_args()

    # 设置默认日期范围
    if not args.start_date:
        args.start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
    if not args.end_date:
        args.end_date = datetime.now().strftime('%Y-%m-%d')

    # 运行
    if args.mode == 'dump':
        dump_qlib_data(args)
    elif args.mode == 'train':
        train_qlib_model(args)
    elif args.mode == 'backtest':
        backtest_qlib_strategy(args)
    else:
        run_full_pipeline(args)


if __name__ == '__main__':
    main()
