"""
突破策略回测 - 海龟交易法则简化版

核心逻辑：价格突破N日高点时买入，跌破N日低点时卖出
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

import baostock as bs


def fetch_data(stock_codes: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """获取数据"""
    lg = bs.login()
    print(f"BaoStock登录: {lg.error_msg}")

    stock_data = {}
    for code in stock_codes:
        bs_code = f"sh.{code}" if code.startswith('6') else f"sz.{code}"
        rs = bs.query_history_k_data_plus(bs_code, "date,open,high,low,close,volume",
                                          start_date=start_date, end_date=end_date,
                                          frequency="d", adjustflag="2")
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        if data_list:
            df = pd.DataFrame(data_list, columns=rs.fields)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            stock_data[code] = df.sort_index()
            print(f"  {code}: {len(df)} 条")

    bs.logout()
    return stock_data


def run_backtest(stock_data: Dict[str, pd.DataFrame],
                 breakout_period: int = 20,
                 stop_loss_pct: float = 0.10,
                 take_profit_pct: float = 0.20) -> Dict:
    """突破策略回测"""

    all_dates = sorted(set(d for df in stock_data.values() for d in df.index))

    capital = 1000000
    positions = {}
    trades = []

    for date in all_dates:
        daily_pnl = 0

        # 检查持仓
        for code in list(positions.keys()):
            df = stock_data.get(code)
            if df is None or date not in df.index:
                continue

            close = df.loc[date, 'close']
            pos = positions[code]

            # 止损
            if close <= pos['stop_loss']:
                daily_pnl += (close - pos['entry']) * pos['shares']
                trades.append({'code': code, 'pnl': (close - pos['entry']) * pos['shares']})
                del positions[code]
            # 止盈
            elif close >= pos['take_profit']:
                daily_pnl += (close - pos['entry']) * pos['shares']
                trades.append({'code': code, 'pnl': (close - pos['entry']) * pos['shares']})
                del positions[code]
            # 跌破N日低点
            elif len(df.loc[:date]) >= breakout_period:
                period_low = df.loc[:date]['low'].iloc[-breakout_period:].min()
                if close < period_low:
                    daily_pnl += (close - pos['entry']) * pos['shares']
                    trades.append({'code': code, 'pnl': (close - pos['entry']) * pos['shares']})
                    del positions[code]

        # 买入
        if len(positions) < 5:
            for code, df in stock_data.items():
                if code in positions or date not in df.index:
                    continue

                hist = df.loc[:date]
                if len(hist) < breakout_period + 1:
                    continue

                period_high = hist['high'].iloc[-breakout_period-1:-1].max()
                close = hist['close'].iloc[-1]

                if close > period_high:
                    shares = (capital * 0.18) / close
                    positions[code] = {
                        'shares': shares,
                        'entry': close,
                        'stop_loss': close * (1 - stop_loss_pct),
                        'take_profit': close * (1 + take_profit_pct)
                    }

        capital += daily_pnl

    # 平仓
    for code, pos in positions.items():
        df = stock_data.get(code)
        if df is not None and len(df) > 0:
            close = df.iloc[-1]['close']
            capital += (close - pos['entry']) * pos['shares']

    total_return = capital / 1000000 - 1
    days = len(all_dates)
    annual_return = total_return * (252 / days) if days > 0 else 0

    wins = [t for t in trades if t.get('pnl', 0) > 0]

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'trades': len(trades),
        'win_rate': len(wins) / len(trades) if trades else 0
    }


def main():
    stock_codes = ['000876', '600515', '688131', '600600', '600460', '688271', '001965']

    print("=" * 60)
    print("突破策略回测")
    print("=" * 60)

    # 获取数据
    print("\n获取数据...")
    all_data = fetch_data(stock_codes, "2020-01-01", "2026-02-28")

    # 划分数据
    validation_data = {c: df.loc["2024-07-01":"2025-06-30"] for c, df in all_data.items()}
    test_data = {c: df.loc["2025-07-01":] for c, df in all_data.items()}

    # 测试多个参数
    params_list = [
        {'breakout_period': 20, 'stop_loss_pct': 0.10, 'take_profit_pct': 0.20},
        {'breakout_period': 30, 'stop_loss_pct': 0.10, 'take_profit_pct': 0.20},
        {'breakout_period': 20, 'stop_loss_pct': 0.08, 'take_profit_pct': 0.15},
    ]

    print("\n验证集测试:")
    best_params = None
    best_return = -1

    for params in params_list:
        result = run_backtest(validation_data, **params)
        print(f"  参数 {params}: 年化 {result['annual_return']:.2%}, 胜率 {result['win_rate']:.2%}")
        if result['annual_return'] > best_return:
            best_return = result['annual_return']
            best_params = params

    print(f"\n最优参数: {best_params}")
    print(f"验证集年化: {best_return:.2%}")

    # 测试集
    print("\n测试集评估:")
    test_result = run_backtest(test_data, **best_params)
    print(f"  年化收益: {test_result['annual_return']:.2%}")
    print(f"  总收益: {test_result['total_return']:.2%}")
    print(f"  交易次数: {test_result['trades']}")
    print(f"  胜率: {test_result['win_rate']:.2%}")

    # 结论
    print("\n" + "=" * 60)
    if test_result['annual_return'] >= 0.15:
        print(f"✓ 达到目标！年化收益 {test_result['annual_return']:.2%} >= 15%")
    else:
        print(f"✗ 未达标，年化收益 {test_result['annual_return']:.2%}")


if __name__ == "__main__":
    main()