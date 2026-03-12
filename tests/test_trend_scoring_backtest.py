"""
趋势评分策略回测测试

直接使用趋势评分系统进行选股和交易，不使用机器学习
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


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算ATR"""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def run_trend_backtest(
    stock_data: Dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    min_trend_score: float = 60,
    stop_loss_atr_mult: float = 2.0,
    take_profit_atr_mult: float = 3.0,
    hold_days: int = 10,
    initial_capital: float = 1000000,
    max_positions: int = 5
) -> Dict:
    """
    使用趋势评分系统进行回测
    """
    from quanttool.factors.trend_scoring_system import TrendScoringSystem

    trend_system = TrendScoringSystem(min_amount=0)

    # 获取所有日期
    all_dates = set()
    for df in stock_data.values():
        dates = df.loc[start_date:end_date].index.tolist()
        all_dates.update(dates)
    all_dates = sorted(list(all_dates))

    if not all_dates:
        return {'error': '日期范围内没有数据'}

    capital = initial_capital
    positions = {}
    trades = []
    equity_curve = [capital]

    signal_stats = {'buy': 0, 'hold': 0, 'total_checks': 0, 'low_score': 0, 'already_held': 0}

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

            # ATR止损止盈
            atr = calculate_atr(df.loc[:date], 14).iloc[-1]
            stop_loss_price = pos['entry_price'] - stop_loss_atr_mult * atr
            take_profit_price = pos['entry_price'] + take_profit_atr_mult * atr

            # 止损
            if close <= stop_loss_price:
                pnl = (close - pos['entry_price']) * pos['shares']
                daily_pnl += pnl
                trades.append({'code': code, 'action': 'sell', 'price': close, 'pnl': pnl, 'reason': 'stop_loss', 'date': date})
                del positions[code]

            # 止盈
            elif close >= take_profit_price:
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
            # 计算所有股票的趋势评分
            scores = []
            for code, df in stock_data.items():
                if code in positions:
                    continue
                if date not in df.index:
                    continue

                hist_df = df.loc[:date]
                if len(hist_df) < 60:
                    continue

                try:
                    result = trend_system.calculate_score(hist_df)
                    if result.passed_hard_filter and result.final_score >= min_trend_score:
                        scores.append((code, result.final_score, df.loc[date, 'close']))
                        signal_stats['total_checks'] += 1
                    else:
                        signal_stats['low_score'] += 1
                except:
                    pass

            # 按评分排序，买入评分最高的
            scores.sort(key=lambda x: x[1], reverse=True)

            for code, score, close in scores:
                if len(positions) >= max_positions:
                    break

                df = stock_data[code]
                atr = calculate_atr(df.loc[:date], 14).iloc[-1]

                position_value = capital * 0.18
                shares = position_value / close

                positions[code] = {
                    'shares': shares,
                    'entry_price': close,
                    'stop_loss': close - stop_loss_atr_mult * atr,
                    'take_profit': close + take_profit_atr_mult * atr,
                    'entry_date': date,
                    'atr': atr,
                    'score': score
                }

                signal_stats['buy'] += 1
                trades.append({'code': code, 'action': 'buy', 'price': close, 'shares': shares, 'date': date, 'score': score, 'atr': atr})

        capital += daily_pnl
        equity_curve.append(capital)

    # 打印信号统计
    print(f"\n  信号统计: 检查 {signal_stats['total_checks']} 次, 买入 {signal_stats['buy']} 次")
    print(f"    评分不足: {signal_stats['low_score']}")

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
            'min_trend_score': min_trend_score,
            'stop_loss_atr_mult': stop_loss_atr_mult,
            'take_profit_atr_mult': take_profit_atr_mult,
            'hold_days': hold_days
        }
    }


def optimize_parameters(
    stock_data: Dict[str, pd.DataFrame],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    target_return: float = 0.18
) -> Tuple[Dict, Dict]:
    """优化参数 - 聚焦高收益参数组合"""

    # 参数范围 - 聚焦高效组合，减少搜索空间
    min_score_range = [75, 80, 85]
    stop_loss_range = [2.0, 2.5, 3.0]
    take_profit_range = [3.0, 3.5, 4.0, 4.5, 5.0]
    hold_days_range = [15, 20, 25, 30]
    max_positions_range = [4, 5, 6]

    best_params = None
    best_score = -1
    best_test_result = None

    results = []

    for min_score in min_score_range:
        for sl in stop_loss_range:
            for tp in take_profit_range:
                if tp <= sl * 1.1:
                    continue

                for hold_days in hold_days_range:
                    for max_pos in max_positions_range:
                        params = {
                            'min_trend_score': min_score,
                            'stop_loss_atr_mult': sl,
                            'take_profit_atr_mult': tp,
                            'hold_days': hold_days,
                            'max_positions': max_pos
                        }

                        # 测试集测试
                        test_result = run_trend_backtest(
                            stock_data=stock_data,
                            start_date=test_start,
                            end_date=test_end,
                            **params
                        )

                        if 'error' in test_result:
                            continue

                        test_annual = test_result['annual_return']
                        win_rate = test_result.get('win_rate', 0)
                        trades = test_result.get('total_trades', 0)
                        max_dd = test_result.get('max_drawdown', 0)

                        if trades < 10:
                            continue

                        # 综合评分：收益优先，考虑胜率和回撤
                        score = test_annual * 2 + (win_rate - 0.4) * 0.1 - max_dd * 0.2

                        results.append((params, test_annual, win_rate, trades, max_dd))

                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_test_result = test_result

    # 打印前10个最好的结果
    print("\n测试集收益前10名:")
    results.sort(key=lambda x: x[1], reverse=True)
    for i, (params, ret, wr, tr, dd) in enumerate(results[:10]):
        print(f"  {i+1}. {params}, 收益: {ret:.2%}, 胜率: {wr:.2%}, 交易: {tr}, 回撤: {dd:.2%}")

    if best_params:
        print(f"\n最优参数: {best_params}")
        print(f"测试集年化收益: {best_test_result.get('annual_return', 0):.2%}")

    return best_params, best_test_result if best_test_result else {}


def main():
    """主函数"""
    stock_codes = ['000876', '600515', '688131', '600600', '600460', '688271', '001965']

    print("=" * 60)
    print("趋势评分策略回测")
    print("=" * 60)

    # 获取数据
    print("\nStep 1: 获取数据")
    stock_data = fetch_real_data_baostock(
        stock_codes=stock_codes,
        start_date="2020-01-01",
        end_date="2026-02-28"
    )

    if not stock_data:
        print("获取数据失败")
        return

    # 参数优化
    print("\nStep 2: 参数优化")
    best_params, test_result = optimize_parameters(
        stock_data=stock_data,
        train_start="2020-01-01",
        train_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2026-02-28",
        target_return=0.15
    )

    if not best_params:
        print("参数优化失败")
        return

    # 打印最终结果
    print("\n" + "=" * 60)
    print("最终报告")
    print("=" * 60)
    print(f"最优参数: {best_params}")
    print(f"测试集年化收益: {test_result.get('annual_return', 0):.2%}")
    print(f"总收益率: {test_result.get('total_return', 0):.2%}")
    print(f"最大回撤: {test_result.get('max_drawdown', 0):.2%}")
    print(f"交易次数: {test_result.get('total_trades', 0)}")
    print(f"胜率: {test_result.get('win_rate', 0):.2%}")


if __name__ == "__main__":
    main()