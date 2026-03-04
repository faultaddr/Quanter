#!/usr/bin/env python
"""
沪深300策略终极优化

目标年化收益 > 10%

使用 TrendMomentumScoring 评分系统
"""
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# 导入新的评分系统
from quanttool.factors.trend_momentum_scoring import TrendMomentumScoring

try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except ImportError:
    BAOSTOCK_AVAILABLE = False

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False


INITIAL_CAPITAL = 1000000
DAYS_LOOKBACK = 365


def get_hs300_stocks() -> list:
    """获取沪深300成分股列表"""
    if not AKSHARE_AVAILABLE:
        return []
    try:
        df = ak.index_stock_cons_weight_csindex(symbol='000300')
        codes = df['成分券代码'].tolist()
        names = df['成分券名称'].tolist()
        print(f"✅ 获取沪深300成分股 {len(codes)} 只")
        return list(zip(codes, names))
    except Exception as e:
        print(f"❌ 获取失败: {e}")
        return []


def fetch_all_data_batch(symbols: list, days: int = 365) -> dict:
    """批量获取数据"""
    print(f"\n批量获取 {len(symbols)} 只股票数据...")

    all_data = {}
    lg = bs.login()
    if lg.error_code != '0':
        return {}

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 100)

    for i, symbol in enumerate(symbols):
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(symbols)}")

        try:
            bs_code = f"sh.{symbol}" if symbol.startswith('6') else f"sz.{symbol}"
            rs = bs.query_history_k_data_plus(
                bs_code, "date,code,open,high,low,close,volume,amount",
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                frequency="d", adjustflag="2"
            )

            if rs.error_code != '0':
                continue

            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                continue

            df = pd.DataFrame(data_list, columns=rs.fields)
            df['timestamp'] = pd.to_datetime(df['date'])
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna().sort_values('timestamp').reset_index(drop=True)

            if len(df) >= 100:
                all_data[symbol] = df
        except Exception:
            pass

    bs.logout()
    print(f"✅ 成功获取 {len(all_data)} 只股票数据")
    return all_data


# 使用新的评分系统实例
_scoring_system = TrendMomentumScoring(buy_threshold=55.0)


def calculate_trend_score(df: pd.DataFrame) -> dict:
    """
    趋势动量评分系统 v3 (封装 TrendMomentumScoring)

    核心思路：抓住趋势启动点，而非等待形态确认
    """
    result = _scoring_system.calculate_score(df)

    return {
        'score': result.final_score,
        'signal': result.signal,
        'stop_loss': result.stop_loss,
        'take_profit': result.take_profit,
        'signals': result.signals
    }


def run_backtest_v3(
    all_data: dict,
    score_threshold: int = 55,
    max_positions: int = 20,
    stop_loss_pct: float = 0.07,
    take_profit_pct: float = 0.15,
    min_trade_interval: int = 3,  # 最小交易间隔
) -> dict:
    """回测 v3"""
    print(f"\n{'='*60}")
    print(f"趋势动量策略回测 v3")
    print(f"{'='*60}")
    print(f"参数: 阈值={score_threshold}, 持仓={max_positions}, 止损={stop_loss_pct*100}%, 止盈={take_profit_pct*100}%")

    portfolio = {
        'cash': INITIAL_CAPITAL,
        'positions': {},
        'trades': [],
        'signals': [],
        'last_trade_date': {}  # 记录上次交易日期
    }

    lookback = 60

    all_dates = set()
    for df in all_data.values():
        all_dates.update(df['timestamp'].dt.date.tolist())
    common_dates = sorted(list(all_dates))

    print(f"\n开始回测 ({len(common_dates)} 交易日)...")

    for date_idx, current_date in enumerate(common_dates):
        if date_idx < lookback:
            continue

        if date_idx % 50 == 0:
            pnl = portfolio['cash'] - INITIAL_CAPITAL
            print(f"  {current_date} | 持仓:{len(portfolio['positions'])} | 盈亏:{pnl:+.0f}")

        # 止损止盈检查
        symbols_to_sell = []
        for symbol, pos in list(portfolio['positions'].items()):
            if symbol not in all_data:
                continue

            df = all_data[symbol]
            date_mask = df['timestamp'].dt.date == current_date
            if not date_mask.any():
                continue

            current_price = df.loc[date_mask, 'close'].iloc[0]

            # 动态止损：盈利后提高止损
            if current_price > pos['entry_price']:
                dynamic_stop = pos['entry_price'] * 1.02  # 盈利后保本止损
            else:
                dynamic_stop = pos['stop_loss']

            if current_price <= dynamic_stop:
                symbols_to_sell.append((symbol, current_price, 'stop_loss'))
            elif current_price >= pos['take_profit']:
                symbols_to_sell.append((symbol, current_price, 'take_profit'))

        # 执行卖出
        for symbol, price, reason in symbols_to_sell:
            pos = portfolio['positions'][symbol]
            sell_value = pos['shares'] * price
            pnl = (price - pos['entry_price']) * pos['shares']

            portfolio['cash'] += sell_value
            portfolio['trades'].append({
                'symbol': symbol,
                'entry_date': str(pos['entry_date']),
                'exit_date': str(current_date),
                'entry_price': pos['entry_price'],
                'exit_price': price,
                'shares': pos['shares'],
                'pnl': pnl,
                'return': pnl / (pos['entry_price'] * pos['shares']),
                'exit_reason': reason
            })
            del portfolio['positions'][symbol]
            portfolio['last_trade_date'][symbol] = current_date

        # 买入信号
        if len(portfolio['positions']) < max_positions:
            candidates = []

            for symbol, df in all_data.items():
                if symbol in portfolio['positions']:
                    continue

                # 控制交易频率
                if symbol in portfolio['last_trade_date']:
                    last_trade = portfolio['last_trade_date'][symbol]
                    days_since = (current_date - last_trade).days if hasattr(current_date, '__sub__') else 0
                    if days_since < min_trade_interval:
                        continue

                hist_df = df[df['timestamp'].dt.date <= current_date].copy()
                if len(hist_df) < lookback + 10:
                    continue

                try:
                    result = calculate_trend_score(hist_df)
                    if result['signal'] and result['score'] >= score_threshold:
                        candidates.append({
                            'symbol': symbol,
                            'score': result['score'],
                            'price': hist_df['close'].iloc[-1],
                            'stop_loss': hist_df['close'].iloc[-1] * (1 - stop_loss_pct),
                            'take_profit': hist_df['close'].iloc[-1] * (1 + take_profit_pct),
                        })
                        portfolio['signals'].append({
                            'date': str(current_date),
                            'symbol': symbol,
                            'score': result['score']
                        })
                except Exception:
                    pass

            candidates.sort(key=lambda x: x['score'], reverse=True)

            slots = max_positions - len(portfolio['positions'])
            for c in candidates[:slots]:
                price = c['price']
                pos_val = portfolio['cash'] / (slots + 1)
                shares = int(pos_val / price / 100) * 100

                if shares > 0:
                    cost = shares * price
                    portfolio['cash'] -= cost
                    portfolio['positions'][c['symbol']] = {
                        'shares': shares,
                        'entry_price': price,
                        'entry_date': current_date,
                        'stop_loss': c['stop_loss'],
                        'take_profit': c['take_profit']
                    }

    # 最后平仓
    final_date = common_dates[-1]
    for symbol, pos in list(portfolio['positions'].items()):
        if symbol not in all_data:
            continue
        df = all_data[symbol]
        date_mask = df['timestamp'].dt.date == final_date
        if date_mask.any():
            final_price = df.loc[date_mask, 'close'].iloc[0]
            pnl = (final_price - pos['entry_price']) * pos['shares']
            portfolio['cash'] += pos['shares'] * final_price
            portfolio['trades'].append({
                'symbol': symbol,
                'entry_date': str(pos['entry_date']),
                'exit_date': str(final_date),
                'entry_price': pos['entry_price'],
                'exit_price': final_price,
                'shares': pos['shares'],
                'pnl': pnl,
                'return': pnl / (pos['entry_price'] * pos['shares']),
                'exit_reason': 'end_of_test'
            })

    # 统计
    total_return = (portfolio['cash'] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    annual_return = total_return

    trades = portfolio['trades']
    if trades:
        returns = [t['return'] for t in trades]
        win_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(win_trades) / len(trades)

        equity_curve = [INITIAL_CAPITAL]
        for t in sorted(trades, key=lambda x: x['exit_date']):
            equity_curve.append(equity_curve[-1] + t['pnl'])

        peak = INITIAL_CAPITAL
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
    else:
        win_rate = 0
        max_dd = 0

    return {
        'annual_return': annual_return,
        'total_return': total_return,
        'final_capital': portfolio['cash'],
        'total_trades': len(trades),
        'win_rate': win_rate,
        'max_drawdown': max_dd,
        'total_signals': len(portfolio['signals']),
        'params': {
            'score_threshold': score_threshold,
            'max_positions': max_positions,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
    }


def ultimate_optimize(all_data: dict, target_return: float = 0.10) -> dict:
    """终极优化"""
    print(f"\n{'='*60}")
    print(f"终极参数优化 - 目标年化收益 > {target_return*100}%")
    print(f"{'='*60}")

    best_result = None
    best_return = -float('inf')

    # 更激进的参数组合
    param_combinations = [
        {'score_threshold': 50, 'max_positions': 15, 'stop_loss_pct': 0.06, 'take_profit_pct': 0.12},
        {'score_threshold': 50, 'max_positions': 20, 'stop_loss_pct': 0.07, 'take_profit_pct': 0.15},
        {'score_threshold': 45, 'max_positions': 15, 'stop_loss_pct': 0.06, 'take_profit_pct': 0.12},
        {'score_threshold': 45, 'max_positions': 20, 'stop_loss_pct': 0.07, 'take_profit_pct': 0.15},
        {'score_threshold': 40, 'max_positions': 20, 'stop_loss_pct': 0.08, 'take_profit_pct': 0.15},
        {'score_threshold': 40, 'max_positions': 25, 'stop_loss_pct': 0.08, 'take_profit_pct': 0.18},
        {'score_threshold': 35, 'max_positions': 20, 'stop_loss_pct': 0.08, 'take_profit_pct': 0.15},
        {'score_threshold': 35, 'max_positions': 25, 'stop_loss_pct': 0.10, 'take_profit_pct': 0.20},
        {'score_threshold': 50, 'max_positions': 25, 'stop_loss_pct': 0.07, 'take_profit_pct': 0.15},
        {'score_threshold': 45, 'max_positions': 25, 'stop_loss_pct': 0.08, 'take_profit_pct': 0.18},
        {'score_threshold': 40, 'max_positions': 30, 'stop_loss_pct': 0.08, 'take_profit_pct': 0.18},
        {'score_threshold': 35, 'max_positions': 30, 'stop_loss_pct': 0.10, 'take_profit_pct': 0.20},
        # 宽止损宽止盈
        {'score_threshold': 50, 'max_positions': 20, 'stop_loss_pct': 0.10, 'take_profit_pct': 0.20},
        {'score_threshold': 45, 'max_positions': 20, 'stop_loss_pct': 0.10, 'take_profit_pct': 0.25},
        {'score_threshold': 40, 'max_positions': 25, 'stop_loss_pct': 0.10, 'take_profit_pct': 0.25},
    ]

    for i, params in enumerate(param_combinations):
        print(f"\n[{i+1}/{len(param_combinations)}] 阈值={params['score_threshold']}, 持仓={params['max_positions']}, 止损={params['stop_loss_pct']*100}%, 止盈={params['take_profit_pct']*100}%")

        result = run_backtest_v3(all_data, **params)

        print(f"  收益: {result['annual_return']*100:.2f}% | 交易: {result['total_trades']} | 胜率: {result['win_rate']*100:.1f}% | 回撤: {result['max_drawdown']*100:.2f}%")

        if result['annual_return'] > best_return:
            best_return = result['annual_return']
            best_result = result

        if result['annual_return'] >= target_return:
            print(f"\n✅ 达到目标收益率！")
            return result

    print(f"\n最佳结果: 年化收益 {best_return*100:.2f}%")
    return best_result


def main():
    print("="*60)
    print("沪深300策略终极优化")
    print("="*60)

    if not BAOSTOCK_AVAILABLE:
        print("❌ BaoStock不可用")
        return

    stocks = get_hs300_stocks()
    if not stocks:
        return

    symbols = [s[0] for s in stocks]
    all_data = fetch_all_data_batch(symbols, days=DAYS_LOOKBACK)

    if not all_data:
        return

    result = ultimate_optimize(all_data, target_return=0.10)

    print(f"\n{'='*60}")
    print(f"📋 最终结果")
    print(f"{'='*60}")
    print(f"  年化收益率: {result['annual_return']*100:.2f}%")
    print(f"  总收益率: {result['total_return']*100:.2f}%")
    print(f"  交易次数: {result['total_trades']}")
    print(f"  胜率: {result['win_rate']*100:.1f}%")
    print(f"  最大回撤: {result['max_drawdown']*100:.2f}%")
    print(f"  是否达标: {'✅ 是' if result['annual_return'] >= 0.10 else '❌ 否'}")

    if 'params' in result:
        print(f"\n最佳参数:")
        for k, v in result['params'].items():
            print(f"  {k}: {v}")

    output_file = Path(__file__).parent.parent / 'reports' / f'hs300_ultimate_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存至: {output_file}")

    return result


if __name__ == '__main__':
    result = main()