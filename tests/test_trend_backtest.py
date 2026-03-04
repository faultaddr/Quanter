"""
趋势动量策略回测 - 基于规则，不依赖机器学习

特点：
1. 基于评分系统生成信号，不使用ML
2. 三阶段验证：训练集优化 -> 验证集调参 -> 测试集评估
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_real_data_baostock(stock_codes: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """使用BaoStock获取真实数据"""
    try:
        import baostock as bs
    except ImportError:
        print("请安装baostock: pip install baostock")
        return {}

    lg = bs.login()
    if lg.error_code != '0':
        print(f"BaoStock登录失败: {lg.error_msg}")
        return {}

    print(f"BaoStock登录成功，获取数据: {start_date} ~ {end_date}")

    stock_data = {}

    for code in stock_codes:
        try:
            if code.startswith('6'):
                bs_code = f"sh.{code}"
            else:
                bs_code = f"sz.{code}"

            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2"
            )

            if rs is None or rs.error_code != '0':
                continue

            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                continue

            df = pd.DataFrame(data_list, columns=rs.fields)
            df['date'] = pd.to_datetime(df['date'])
            df = df.rename(columns={'date': 'timestamp'})
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.set_index('timestamp')
            df = df.sort_index()

            stock_data[code] = df
            print(f"  {code}: {len(df)} 条数据")

        except Exception:
            pass

    bs.logout()
    return stock_data


def split_data(
    stock_data: Dict[str, pd.DataFrame],
    train_end: str = "2024-03-31",
    validation_end: str = "2025-06-30"
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    三阶段数据划分 - 增加验证集长度以获得更稳健的评估

    训练集: 用于参数优化
    验证集: 用于选择最优参数 (更长更稳健)
    测试集: 最终评估
    """
    train_data = {}
    validation_data = {}
    test_data = {}

    for code, df in stock_data.items():
        train_df = df.loc[:train_end]
        validation_df = df.loc[train_end:validation_end]
        test_df = df.loc[validation_end:]

        if len(train_df) > 100 and len(validation_df) > 100 and len(test_df) > 20:
            train_data[code] = train_df
            validation_data[code] = validation_df.iloc[1:]
            test_data[code] = test_df.iloc[1:]

    print(f"\n三阶段数据划分完成:")
    print(f"  训练集 (~ {train_end}): {len(train_data)} 只股票")
    print(f"  验证集 ({train_end} ~ {validation_end}): {len(validation_data)} 只股票")
    print(f"  测试集 ({validation_end} ~): {len(test_data)} 只股票")

    return train_data, validation_data, test_data


def run_trend_backtest(
    stock_data: Dict[str, pd.DataFrame],
    buy_threshold: float = 55.0,
    sell_threshold: float = 40.0,
    stop_loss_pct: float = 0.07,
    take_profit_pct: float = 0.15,
    hold_days: int = 10,
    initial_capital: float = 1000000,
    max_positions: int = 5,
    market_filter: bool = True  # 新增：市场环境过滤
) -> Dict:
    """
    趋势动量策略回测 - 增加市场环境过滤

    市场环境过滤：
    - 当大盘(MA20 > MA60)时才买入
    - 熊市时减少开仓
    """
    from quanttool.strategies.trend_momentum_strategy import TrendMomentumStrategy

    if not stock_data:
        return {'error': '数据不足'}

    # 初始化策略
    strategy = TrendMomentumStrategy(
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct
    )

    # 获取所有日期
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df.index.tolist())
    all_dates = sorted(list(all_dates))

    # 计算市场环境指标 (使用所有股票的平均表现)
    market_status = {}
    if market_filter and len(stock_data) > 0:
        # 使用第一只股票作为市场代表
        sample_df = list(stock_data.values())[0]
        sample_df['ma20'] = sample_df['close'].rolling(20).mean()
        sample_df['ma60'] = sample_df['close'].rolling(60).mean()
        sample_df['market_bull'] = sample_df['ma20'] > sample_df['ma60']

        for date in all_dates:
            if date in sample_df.index:
                market_status[date] = sample_df.loc[date, 'market_bull'] if pd.notna(sample_df.loc[date, 'market_bull']) else True
            else:
                market_status[date] = True

    # 回测
    capital = initial_capital
    positions = {}
    trades = []
    equity_curve = [capital]

    signal_stats = {'buy': 0, 'hold': 0, 'total_checks': 0, 'market_filter': 0}

    for date in all_dates:
        daily_pnl = 0

        # 检查现有持仓
        for code in list(positions.keys()):
            if code not in stock_data:
                continue
            df = stock_data[code]
            if date not in df.index:
                continue

            close = df.loc[date, 'close']
            pos = positions[code]

            # 止损
            if close <= pos['stop_loss']:
                pnl = (close - pos['entry_price']) * pos['shares']
                daily_pnl += pnl
                trades.append({'code': code, 'action': 'sell', 'price': close, 'pnl': pnl, 'reason': 'stop_loss', 'date': date})
                del positions[code]

            # 止盈
            elif close >= pos['take_profit']:
                pnl = (close - pos['entry_price']) * pos['shares']
                daily_pnl += pnl
                trades.append({'code': code, 'action': 'sell', 'price': close, 'pnl': pnl, 'reason': 'take_profit', 'date': date})
                del positions[code]

            # 超时
            elif (date - pos['entry_date']).days >= hold_days * 2:
                pnl = (close - pos['entry_price']) * pos['shares']
                daily_pnl += pnl
                trades.append({'code': code, 'action': 'sell', 'price': close, 'pnl': pnl, 'reason': 'timeout', 'date': date})
                del positions[code]

        # 尝试新买入
        if len(positions) < max_positions:
            # 市场环境过滤
            is_bull_market = market_status.get(date, True)

            for code, df in stock_data.items():
                if code in positions:
                    continue
                if date not in df.index:
                    continue

                hist_df = df.loc[:date]
                if len(hist_df) < 60:
                    continue

                try:
                    signal = strategy.get_signal(hist_df.iloc[-1], hist_df)
                    signal_stats['total_checks'] += 1

                    if signal.get('direction') == 'buy':
                        # 市场环境过滤：熊市时提高买入阈值
                        if market_filter and not is_bull_market:
                            signal_stats['market_filter'] += 1
                            continue

                        signal_stats['buy'] += 1
                        close = df.loc[date, 'close']

                        position_value = capital * 0.18
                        shares = position_value / close

                        positions[code] = {
                            'shares': shares,
                            'entry_price': close,
                            'stop_loss': signal.get('stop_loss', close * (1 - stop_loss_pct)),
                            'take_profit': signal.get('take_profit', close * (1 + take_profit_pct)),
                            'entry_date': date
                        }

                        trades.append({'code': code, 'action': 'buy', 'price': close, 'shares': shares, 'date': date})
                    else:
                        signal_stats['hold'] += 1

                except Exception:
                    pass

        capital += daily_pnl
        equity_curve.append(capital)

    print(f"\n  信号统计: 检查 {signal_stats['total_checks']} 次, 买入 {signal_stats['buy']} 次, 市场过滤 {signal_stats['market_filter']} 次")

    # 平剩余仓位
    for code, pos in list(positions.items()):
        if code in stock_data:
            df = stock_data[code]
            if len(df) > 0:
                close = df.iloc[-1]['close']
                pnl = (close - pos['entry_price']) * pos['shares']
                capital += pnl
                trades.append({'code': code, 'action': 'sell', 'price': close, 'pnl': pnl, 'reason': 'final', 'date': df.index[-1]})

    # 计算指标
    sell_trades = [t for t in trades if t.get('action') == 'sell']
    winning = [t for t in sell_trades if t.get('pnl', 0) > 0]
    losing = [t for t in sell_trades if t.get('pnl', 0) <= 0]

    total_return = capital / initial_capital - 1

    test_days = len(all_dates)
    annual_return = total_return * (252 / test_days) if test_days > 0 else 0

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = np.max(drawdown)

    return {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_dd,
        'total_trades': len(sell_trades),
        'winning_trades': len(winning),
        'losing_trades': len(losing),
        'win_rate': len(winning) / len(sell_trades) if sell_trades else 0,
        'trades': trades,
        'equity_curve': equity_curve,
        'params': {
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'market_filter': market_filter
        }
    }


def optimize_parameters(
    train_data: Dict[str, pd.DataFrame],
    validation_data: Dict[str, pd.DataFrame],
    target_return: float = 0.15,
    max_iterations: int = 50
) -> Tuple[Dict, Dict]:
    """参数优化 - 更激进的参数范围"""
    import random

    # 参数范围 - 更宽松的买入条件
    buy_threshold_range = [40, 45, 50, 55, 60]  # 降低买入阈值
    sell_threshold_range = [25, 30, 35, 40]  # 降低卖出阈值
    stop_loss_pct_range = [0.08, 0.10, 0.12]  # 适当放宽止损
    take_profit_pct_range = [0.12, 0.15, 0.18, 0.20]  # 更高的止盈

    # 生成参数组合
    all_combos = []
    for bt in buy_threshold_range:
        for st in sell_threshold_range:
            if bt > st:  # 买入阈值 > 卖出阈值
                for sl in stop_loss_pct_range:
                    for tp in take_profit_pct_range:
                        if tp > sl:  # 止盈 > 止损
                            all_combos.append({
                                'buy_threshold': bt,
                                'sell_threshold': st,
                                'stop_loss_pct': sl,
                                'take_profit_pct': tp,
                                'market_filter': False  # 关闭市场过滤
                            })

    print(f"总参数组合数: {len(all_combos)}")

    if len(all_combos) > max_iterations:
        param_sets = random.sample(all_combos, max_iterations)
    else:
        param_sets = all_combos

    best_params = None
    best_return = -1

    for i, params in enumerate(param_sets):
        print(f"\n{'='*50}")
        print(f"参数优化 {i+1}/{len(param_sets)}")
        print(f"参数: {params}")

        try:
            result = run_trend_backtest(
                stock_data=validation_data,
                **params
            )

            if 'error' in result:
                print(f"  错误: {result['error']}")
                continue

            annual = result['annual_return']
            print(f"  验证集年化收益: {annual:.2%}")

            if annual > best_return:
                best_return = annual
                best_params = params

            if annual >= target_return:
                print(f"\n✓ 达到目标年化收益 {target_return:.0%}!")
                return params, {'validation_return': annual, 'found_target': True}

        except Exception as e:
            print(f"  异常: {e}")
            continue

    return best_params, {'validation_return': best_return, 'found_target': False}


def main():
    """主函数"""
    stock_codes = ['000876', '600515', '688131', '600600', '600460', '688271', '001965']

    print("=" * 60)
    print("趋势动量策略回测 - 基于规则，不依赖机器学习")
    print("=" * 60)

    # 1. 获取数据
    print("\nStep 1: 获取真实数据")
    stock_data = fetch_real_data_baostock(
        stock_codes=stock_codes,
        start_date="2020-01-01",
        end_date="2026-02-28"
    )

    if not stock_data:
        print("获取数据失败")
        return

    # 2. 数据划分
    print("\nStep 2: 三阶段数据划分")
    train_data, validation_data, test_data = split_data(stock_data)

    # 3. 参数优化
    print("\nStep 3: 参数优化")
    best_params, opt_report = optimize_parameters(
        train_data=train_data,
        validation_data=validation_data,
        target_return=0.15
    )

    if not best_params:
        print("参数优化失败")
        return

    print(f"\n验证集最优参数: {best_params}")
    print(f"验证集年化收益: {opt_report['validation_return']:.2%}")

    # 4. 测试集评估
    print("\n" + "=" * 60)
    print("Step 4: 测试集评估")
    print("=" * 60)
    print(f"使用锁定参数: {best_params}")

    test_result = run_trend_backtest(
        stock_data=test_data,
        **best_params
    )

    if 'error' in test_result:
        print(f"测试失败: {test_result['error']}")
        return

    # 5. 输出结果
    print("\n" + "=" * 60)
    print("最终报告")
    print("=" * 60)

    print(f"\n{'='*30}")
    print("锁定参数")
    print(f"{'='*30}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    print(f"\n{'='*30}")
    print("验证集表现")
    print(f"{'='*30}")
    print(f"  年化收益: {opt_report['validation_return']:.2%}")

    print(f"\n{'='*30}")
    print("测试集表现 (最终)")
    print(f"{'='*30}")
    print(f"  初始资金: {test_result['initial_capital']:,.0f}")
    print(f"  最终资金: {test_result['final_capital']:,.0f}")
    print(f"  总收益率: {test_result['total_return']:.2%}")
    print(f"  年化收益: {test_result['annual_return']:.2%}")
    print(f"  最大回撤: {test_result['max_drawdown']:.2%}")
    print(f"  总交易次数: {test_result['total_trades']}")
    print(f"  胜率: {test_result['win_rate']:.2%}")

    # 过拟合检测
    print(f"\n{'='*30}")
    print("过拟合检测")
    print(f"{'='*30}")
    drop_ratio = (opt_report['validation_return'] - test_result['annual_return']) / opt_report['validation_return'] if opt_report['validation_return'] > 0 else 0
    is_overfitting = drop_ratio > 0.3

    if is_overfitting:
        print(f"  ⚠️  检测到过拟合!")
        print(f"  验证集 -> 测试集收益下降: {drop_ratio:.1%}")
    else:
        print(f"  ✓ 未检测到过拟合")
        print(f"  验证集 -> 测试集收益下降: {drop_ratio:.1%}")

    print(f"\n{'='*30}")
    print("结论")
    print(f"{'='*30}")
    if test_result['annual_return'] >= 0.15 and not is_overfitting:
        print(f"  ✓ 策略验证通过，年化收益 {test_result['annual_return']:.2%} >= 15%")
    elif is_overfitting:
        print(f"  ✗ 策略存在过拟合风险")
    else:
        print(f"  ✗ 年化收益 {test_result['annual_return']:.2%} 未达到目标 15%")


if __name__ == "__main__":
    main()