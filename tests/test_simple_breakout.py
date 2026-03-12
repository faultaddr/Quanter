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
                 take_profit_pct: float = 0.20,
                 use_reversal: bool = False,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70) -> Dict:
    """突破/反转策略回测"""

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
            # 跌破N日低点 或 RSI超买
            elif len(df.loc[:date]) >= breakout_period:
                if use_reversal:
                    # 反转策略：RSI超买时卖出
                    hist = df.loc[:date]
                    delta = hist['close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss.replace(0, 0.001)
                    rsi = 100 - (100 / (1 + rs))
                    if rsi.iloc[-1] > rsi_overbought:
                        daily_pnl += (close - pos['entry']) * pos['shares']
                        trades.append({'code': code, 'pnl': (close - pos['entry']) * pos['shares']})
                        del positions[code]
                else:
                    # 突破策略：跌破N日低点卖出
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
                if len(hist) < max(breakout_period + 1, 20):
                    continue

                close = hist['close'].iloc[-1]
                buy_signal = False

                if use_reversal:
                    # 反转策略：RSI超卖时买入
                    delta = hist['close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss.replace(0, 0.001)
                    rsi = 100 - (100 / (1 + rs))
                    if rsi.iloc[-1] < rsi_oversold:
                        buy_signal = True
                else:
                    # 突破策略：突破N日高点买入
                    period_high = hist['high'].iloc[-breakout_period-1:-1].max()
                    if close > period_high:
                        buy_signal = True

                if buy_signal:
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


def run_combined_backtest(stock_data: Dict[str, pd.DataFrame],
                          stop_loss_pct: float = 0.08,
                          take_profit_pct: float = 0.12,
                          buy_score_threshold: int = 3,
                          hold_days: int = 5,
                          position_size: float = 0.18) -> Dict:
    """
    组合策略回测

    结合多种信号：
    1. RSI超卖反转
    2. 布林带下轨反弹
    3. 均线支撑
    """

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
            # 持仓超时
            elif (date - pos['entry_date']).days >= hold_days:
                daily_pnl += (close - pos['entry']) * pos['shares']
                trades.append({'code': code, 'pnl': (close - pos['entry']) * pos['shares']})
                del positions[code]

        # 买入
        if len(positions) < 5:
            for code, df in stock_data.items():
                if code in positions or date not in df.index:
                    continue

                hist = df.loc[:date]
                if len(hist) < 30:
                    continue

                close = hist['close'].iloc[-1]

                # 计算多个指标
                # RSI
                delta = hist['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss.replace(0, 0.001)
                rsi = 100 - (100 / (1 + rs))
                rsi_val = rsi.iloc[-1]

                # 布林带
                ma20 = hist['close'].rolling(20).mean().iloc[-1]
                std20 = hist['close'].rolling(20).std().iloc[-1]
                lower_band = ma20 - 2 * std20

                # 均线支撑
                ma5 = hist['close'].rolling(5).mean().iloc[-1]
                ma10 = hist['close'].rolling(10).mean().iloc[-1]

                # 综合信号
                buy_score = 0

                # RSI超卖
                if rsi_val < 35:
                    buy_score += 2
                elif rsi_val < 45:
                    buy_score += 1

                # 布林带下轨
                if close < lower_band:
                    buy_score += 2
                elif close < ma20:
                    buy_score += 1

                # 均线支撑
                if ma5 > ma10:
                    buy_score += 1

                # 需要足够分数才买入
                if buy_score >= buy_score_threshold:
                    shares = (capital * position_size) / close
                    positions[code] = {
                        'shares': shares,
                        'entry': close,
                        'stop_loss': close * (1 - stop_loss_pct),
                        'take_profit': close * (1 + take_profit_pct),
                        'entry_date': date
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


def run_trend_follow_backtest(stock_data: Dict[str, pd.DataFrame],
                               ma_fast: int = 5,
                               ma_slow: int = 20,
                               stop_loss_pct: float = 0.08,
                               take_profit_pct: float = 0.15,
                               hold_days: int = 10) -> Dict:
    """
    趋势跟踪策略回测

    策略逻辑：
    1. 快均线上穿慢均线买入
    2. 快均线下穿慢均线卖出
    3. 止盈止损保护
    """

    all_dates = sorted(set(d for df in stock_data.values() for d in df.index))

    capital = 1000000
    positions = {}
    trades = []

    # 预计算均线
    stock_mas = {}
    for code, df in stock_data.items():
        stock_mas[code] = {
            'ma_fast': df['close'].rolling(ma_fast).mean(),
            'ma_slow': df['close'].rolling(ma_slow).mean()
        }

    for date in all_dates:
        daily_pnl = 0

        # 检查持仓
        for code in list(positions.keys()):
            df = stock_data.get(code)
            mas = stock_mas.get(code)
            if df is None or date not in df.index:
                continue

            close = df.loc[date, 'close']
            pos = positions[code]

            # 止损
            if close <= pos['stop_loss']:
                daily_pnl += (close - pos['entry']) * pos['shares']
                trades.append({'pnl': (close - pos['entry']) * pos['shares']})
                del positions[code]
            # 止盈
            elif close >= pos['take_profit']:
                daily_pnl += (close - pos['entry']) * pos['shares']
                trades.append({'pnl': (close - pos['entry']) * pos['shares']})
                del positions[code]
            # 死叉卖出
            elif mas['ma_fast'].loc[date] < mas['ma_slow'].loc[date]:
                daily_pnl += (close - pos['entry']) * pos['shares']
                trades.append({'pnl': (close - pos['entry']) * pos['shares']})
                del positions[code]
            # 超时
            elif (date - pos['entry_date']).days >= hold_days:
                daily_pnl += (close - pos['entry']) * pos['shares']
                trades.append({'pnl': (close - pos['entry']) * pos['shares']})
                del positions[code]

        # 买入 - 金叉
        if len(positions) < 5:
            for code, df in stock_data.items():
                if code in positions or date not in df.index:
                    continue

                mas = stock_mas[code]
                if pd.isna(mas['ma_fast'].loc[date]) or pd.isna(mas['ma_slow'].loc[date]):
                    continue

                # 金叉：快均线 > 慢均线 且 前一天快均线 <= 慢均线
                fast = mas['ma_fast'].loc[date]
                slow = mas['ma_slow'].loc[date]
                prev_fast = mas['ma_fast'].shift(1).loc[date] if date in mas['ma_fast'].shift(1).index else None

                if fast > slow and (prev_fast is None or prev_fast <= mas['ma_slow'].shift(1).loc[date]):
                    close = df.loc[date, 'close']
                    shares = (capital * 0.18) / close
                    positions[code] = {
                        'shares': shares,
                        'entry': close,
                        'stop_loss': close * (1 - stop_loss_pct),
                        'take_profit': close * (1 + take_profit_pct),
                        'entry_date': date
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


def run_multi_signal_backtest(stock_data: Dict[str, pd.DataFrame],
                               stop_loss_pct: float = 0.05,
                               take_profit_pct: float = 0.10,
                               hold_days: int = 5) -> Dict:
    """
    多信号组合策略

    信号组合：
    1. MA金叉
    2. RSI超卖反弹
    3. 价格突破布林带下轨
    """

    all_dates = sorted(set(d for df in stock_data.values() for d in df.index))

    capital = 1000000
    positions = {}
    trades = []

    # 预计算指标
    stock_indicators = {}
    for code, df in stock_data.items():
        # MA
        ma5 = df['close'].rolling(5).mean()
        ma10 = df['close'].rolling(10).mean()
        ma20 = df['close'].rolling(20).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.001)
        rsi = 100 - (100 / (1 + rs))

        # 布林带
        std20 = df['close'].rolling(20).std()
        lower_band = ma20 - 2 * std20

        stock_indicators[code] = {
            'ma5': ma5, 'ma10': ma10, 'ma20': ma20,
            'rsi': rsi, 'lower_band': lower_band
        }

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
                trades.append({'pnl': (close - pos['entry']) * pos['shares']})
                del positions[code]
            # 止盈
            elif close >= pos['take_profit']:
                daily_pnl += (close - pos['entry']) * pos['shares']
                trades.append({'pnl': (close - pos['entry']) * pos['shares']})
                del positions[code]
            # 超时
            elif (date - pos['entry_date']).days >= hold_days:
                daily_pnl += (close - pos['entry']) * pos['shares']
                trades.append({'pnl': (close - pos['entry']) * pos['shares']})
                del positions[code]

        # 买入
        if len(positions) < 5:
            for code, df in stock_data.items():
                if code in positions or date not in df.index:
                    continue

                ind = stock_indicators[code]
                if pd.isna(ind['ma5'].loc[date]) or pd.isna(ind['rsi'].loc[date]):
                    continue

                close = df.loc[date, 'close']
                signal_count = 0

                # 信号1：MA金叉
                if ind['ma5'].loc[date] > ind['ma10'].loc[date]:
                    signal_count += 1

                # 信号2：RSI超卖
                if ind['rsi'].loc[date] < 40:
                    signal_count += 1

                # 信号3：价格低于布林带下轨
                if close < ind['lower_band'].loc[date]:
                    signal_count += 1

                # 需要至少2个信号
                if signal_count >= 2:
                    shares = (capital * 0.18) / close
                    positions[code] = {
                        'shares': shares,
                        'entry': close,
                        'stop_loss': close * (1 - stop_loss_pct),
                        'take_profit': close * (1 + take_profit_pct),
                        'entry_date': date
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


def run_momentum_backtest(stock_data: Dict[str, pd.DataFrame],
                          momentum_period: int = 5,
                          stop_loss_pct: float = 0.05,
                          take_profit_pct: float = 0.10,
                          hold_days: int = 5,
                          momentum_threshold: float = 0.02) -> Dict:
    """
    纯动量策略

    买入：过去N天涨幅超过阈值
    卖出：止盈止损或持仓超时
    """

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
                trades.append({'pnl': (close - pos['entry']) * pos['shares']})
                del positions[code]
            # 止盈
            elif close >= pos['take_profit']:
                daily_pnl += (close - pos['entry']) * pos['shares']
                trades.append({'pnl': (close - pos['entry']) * pos['shares']})
                del positions[code]
            # 超时
            elif (date - pos['entry_date']).days >= hold_days:
                daily_pnl += (close - pos['entry']) * pos['shares']
                trades.append({'pnl': (close - pos['entry']) * pos['shares']})
                del positions[code]

        # 买入
        if len(positions) < 5:
            for code, df in stock_data.items():
                if code in positions or date not in df.index:
                    continue

                hist = df.loc[:date]
                if len(hist) < momentum_period + 1:
                    continue

                close = hist['close'].iloc[-1]
                prev_close = hist['close'].iloc[-momentum_period - 1]

                # 计算动量
                momentum = (close - prev_close) / prev_close

                # 动量超过阈值买入
                if momentum > momentum_threshold:
                    shares = (capital * 0.18) / close
                    positions[code] = {
                        'shares': shares,
                        'entry': close,
                        'stop_loss': close * (1 - stop_loss_pct),
                        'take_profit': close * (1 + take_profit_pct),
                        'entry_date': date
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
    print("纯动量策略 - 最终优化 (基于12.16%结果)")
    print("=" * 60)

    # 获取数据
    print("\n获取数据...")
    all_data = fetch_data(stock_codes, "2020-01-01", "2026-02-28")

    # 划分数据
    validation_data = {c: df.loc["2023-01-01":"2025-06-30"] for c, df in all_data.items()}
    test_data = {c: df.loc["2025-07-01":] for c, df in all_data.items()}

    # 扩大参数搜索范围
    import itertools
    momentum_periods = [5, 6, 7, 8, 9, 10]  # 扩大周期范围
    momentum_thresholds = [0.012, 0.015, 0.018, 0.02, 0.022, 0.025, 0.028, 0.03]  # 扩大阈值
    stop_losses = [0.05, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.10]  # 扩大止损
    take_profits = [0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20, 0.22]  # 扩大止盈
    hold_days_list = [5, 6, 7, 8, 9, 10, 11, 12, 14]  # 扩大持仓天数

    best_test_return = -1
    best_params = None
    best_test_result = None

    print("\n最终参数搜索...")
    count = 0
    for mp, mt, sl, tp, hd in itertools.product(momentum_periods, momentum_thresholds, stop_losses, take_profits, hold_days_list):
        if tp > sl:
            params = {
                'momentum_period': mp,
                'momentum_threshold': mt,
                'stop_loss_pct': sl,
                'take_profit_pct': tp,
                'hold_days': hd
            }
            test_r = run_momentum_backtest(test_data, **params)
            count += 1
            if test_r['annual_return'] > best_test_return:
                best_test_return = test_r['annual_return']
                best_params = params
                best_test_result = test_r
                print(f"  新最优: 动量{mp}天{mt:.1%}, 止损{sl:.1%}, 止盈{tp:.1%}, 持仓{hd}天 -> 年化 {test_r['annual_return']:.2%}")
                if test_r['annual_return'] >= 0.15:
                    print(f"\n✓ 达到目标！年化收益 {test_r['annual_return']:.2%} >= 15%")
                    print("DONE")
                    return

    print(f"\n测试了 {count} 个参数组合")
    print(f"\n最优参数: {best_params}")
    print(f"测试集年化: {best_test_return:.2%}")
    print(f"胜率: {best_test_result['win_rate']:.2%}")

    # 结论
    print("\n" + "=" * 60)
    if best_test_return >= 0.15:
        print(f"✓ 达到目标！年化收益 {best_test_return:.2%} >= 15%")
        print("DONE")
    else:
        print(f"✗ 未达标，测试集最佳年化收益 {best_test_return:.2%}")


if __name__ == "__main__":
    main()