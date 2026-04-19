"""Simple breakout strategy backtest script with trend-based holding."""

import sys
sys.path.insert(0, '/Users/missy/PROJ/NEW_Quanter/Quanter')

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from quanttool.infrastructure.data_providers.historical.enhanced_fetcher import create_data_fetcher_with_credentials
from quanttool.factors.breakout_scoring_system import BreakoutScoringSystem
from quanttool.factors.trend_scoring_system import TrendScoringSystem


def run_breakout_backtest(
    symbols: list,
    start_date: datetime,
    end_date: datetime,
    initial_cash: float = 100000.0,
    commission: float = 0.0003,
    # 可调参数
    buy_score_threshold: int = 70,      # 买入信号阈值
    sell_score_threshold: int = 25,     # 卖出信号阈值
    max_hold_days: int = 20,            # 最大持仓天数
    stop_loss_pct: float = -8.0,        # 止损百分比
    take_profit_pct: float = 15.0,      # 止盈百分比
    max_positions: int = 5,             # 最大持仓数
    position_pct: float = 0.2,          # 单只股票仓位比例
    trend_sell_threshold: int = 40,     # 趋势评分卖出阈值
):
    """
    Run breakout strategy with trend-based holding.
    - Use breakout score for BUY signals
    - Use trend score for HOLD/SELL decisions
    """
    print("=" * 70)
    print("📊 低位盘整突破策略回测 (趋势跟踪版)")
    print("=" * 70)
    print(f"股票池: {symbols}")
    print(f"回测区间: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    print(f"初始资金: ¥{initial_cash:,.2f}")
    print(f"手续费率: {commission * 100:.2f}%")
    print("-" * 70)
    print(f"策略参数:")
    print(f"  买入阈值(breakout): {buy_score_threshold}分")
    print(f"  卖出阈值(trend): {trend_sell_threshold}分")
    print(f"  最大持仓: {max_hold_days}天")
    print(f"  止损: {stop_loss_pct}%")
    print(f"  止盈: {take_profit_pct}%")
    print("-" * 70)

    # Initialize data fetcher
    print("\n正在初始化数据获取器...")
    fetcher = create_data_fetcher_with_credentials()

    # Fetch all data
    print(f"\n正在获取 {len(symbols)} 只股票数据...")
    data = fetcher.get_bars_cached(symbols, start_date, end_date)
    print(f"成功获取 {len(data)} 只股票数据\n")

    # Initialize scoring systems
    breakout_system = BreakoutScoringSystem()
    trend_system = TrendScoringSystem()

    # Track portfolio with trailing stop
    cash = initial_cash
    positions = {}  # symbol -> {'shares': n, 'buy_price': p, 'buy_date': d, 'high_price': h}
    trades = []
    portfolio_values = []

    # 移动止盈参数
    trailing_stop_pct = 0.05  # 从最高点回撤5%止盈

    # Get all trading dates
    all_dates = set()
    for symbol, df in data.items():
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            all_dates.update(df['timestamp'].dt.date.tolist())

    trading_dates = sorted(all_dates)

    print("开始回测...")
    print("-" * 70)

    for current_date in trading_dates:
        current_dt = datetime.combine(current_date, datetime.min.time())

        # Calculate portfolio value
        portfolio_value = cash
        for symbol, pos in positions.items():
            if symbol in data and not data[symbol].empty:
                df = data[symbol]
                day_data = df[df['timestamp'].dt.date == current_date]
                if not day_data.empty:
                    current_price = day_data.iloc[0]['close']
                    portfolio_value += pos['shares'] * current_price

        portfolio_values.append({
            'date': current_date,
            'value': portfolio_value,
            'cash': cash,
            'positions': len(positions)
        })

        # Process existing positions - check for sell signals
        positions_to_sell = []
        for symbol, pos in list(positions.items()):
            if symbol not in data or data[symbol].empty:
                continue

            df = data[symbol]
            day_data = df[df['timestamp'].dt.date == current_date]

            if day_data.empty:
                continue

            current_price = day_data.iloc[0]['close']
            profit_pct = (current_price - pos['buy_price']) / pos['buy_price'] * 100
            days_held = (current_date - pos['buy_date']).days

            # 更新最高价（用于移动止盈）
            if 'high_price' not in pos:
                pos['high_price'] = pos['buy_price']
            if current_price > pos['high_price']:
                pos['high_price'] = current_price

            # 移动止盈检查 - 从最高点回撤超过阈值
            drawdown_from_high = (pos['high_price'] - current_price) / pos['high_price'] * 100
            if profit_pct > 5 and drawdown_from_high >= trailing_stop_pct * 100:
                positions_to_sell.append((symbol, f"移动止盈(回撤{drawdown_from_high:.1f}%)"))
                continue

            # 止损检查
            if profit_pct <= stop_loss_pct:
                positions_to_sell.append((symbol, f"止损({profit_pct:.1f}%)"))
                continue

            # 止盈检查
            if profit_pct >= take_profit_pct:
                positions_to_sell.append((symbol, f"止盈({profit_pct:.1f}%)"))
                continue

            # 持仓期满
            if days_held >= max_hold_days:
                positions_to_sell.append((symbol, f"持仓期满({days_held}天)"))
                continue

            # 趋势评分检查 - 使用趋势评分做持仓决策
            hist_data = df[df['timestamp'].dt.date <= current_date].tail(100)
            if len(hist_data) >= 60:
                try:
                    trend_result = trend_system.calculate_score(hist_data)
                    if trend_result.final_score < trend_sell_threshold:
                        positions_to_sell.append((symbol, f"趋势转弱({trend_result.final_score:.0f}分)"))
                except:
                    pass

        # Execute sells
        for symbol, reason in positions_to_sell:
            if symbol in positions:
                df = data[symbol]
                day_data = df[df['timestamp'].dt.date == current_date]
                if not day_data.empty:
                    sell_price = day_data.iloc[0]['close']
                    pos = positions[symbol]
                    sell_value = pos['shares'] * sell_price
                    commission_cost = sell_value * commission
                    cash += sell_value - commission_cost

                    profit = (sell_price - pos['buy_price']) / pos['buy_price'] * 100
                    trades.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': sell_price,
                        'shares': pos['shares'],
                        'value': sell_value,
                        'profit_pct': profit,
                        'hold_days': (current_date - pos['buy_date']).days,
                        'reason': reason
                    })

                    del positions[symbol]
                    print(f"  [{current_date}] 卖出 {symbol}: ¥{sell_price:.2f} ({profit:+.1f}%) - {reason}")

        # Look for buy signals - use breakout score
        if len(positions) < max_positions:
            buy_candidates = []

            for symbol in symbols:
                if symbol in positions:
                    continue
                if symbol not in data or data[symbol].empty:
                    continue

                df = data[symbol]
                hist_data = df[df['timestamp'].dt.date <= current_date].tail(100)

                if len(hist_data) < 60:
                    continue

                try:
                    # 用breakout评分作为买入信号
                    breakout_result = breakout_system.calculate_score(hist_data)
                    if breakout_result.passed_filter and breakout_result.final_score >= buy_score_threshold:
                        # 同时检查趋势评分，确保趋势向上
                        trend_result = trend_system.calculate_score(hist_data)
                        if trend_result.final_score >= 60:  # 趋势评分>=60才买入（提高门槛）
                            buy_candidates.append((symbol, breakout_result.final_score, trend_result.final_score))
                except:
                    pass

            # 按breakout评分排序
            buy_candidates.sort(key=lambda x: x[1], reverse=True)

            for symbol, breakout_score, trend_score in buy_candidates:
                if len(positions) >= max_positions:
                    break

                df = data[symbol]
                day_data = df[df['timestamp'].dt.date == current_date]

                if day_data.empty:
                    continue

                buy_price = day_data.iloc[0]['close']
                position_size = int((cash * position_pct) // buy_price)

                if position_size > 0:
                    buy_value = position_size * buy_price
                    commission_cost = buy_value * commission

                    if buy_value + commission_cost <= cash:
                        cash -= buy_value + commission_cost
                        positions[symbol] = {
                            'shares': position_size,
                            'buy_price': buy_price,
                            'buy_date': current_date,
                            'buy_score': breakout_score
                        }
                        trades.append({
                            'date': current_date,
                            'symbol': symbol,
                            'action': 'BUY',
                            'price': buy_price,
                            'shares': position_size,
                            'value': buy_value,
                            'profit_pct': 0,
                            'hold_days': 0,
                            'reason': f"突破{breakout_score:.0f}分+趋势{trend_score:.0f}分"
                        })
                        print(f"  [{current_date}] 买入 {symbol}: ¥{buy_price:.2f} x {position_size}股 (突破{breakout_score:.0f}+趋势{trend_score:.0f})")

    # Close all remaining positions at end
    final_date = trading_dates[-1]
    print(f"\n{final_date}: 清算剩余持仓...")
    for symbol, pos in list(positions.items()):
        if symbol in data and not data[symbol].empty:
            df = data[symbol]
            day_data = df[df['timestamp'].dt.date == final_date]
            if not day_data.empty:
                sell_price = day_data.iloc[0]['close']
                sell_value = pos['shares'] * sell_price
                commission_cost = sell_value * commission
                cash += sell_value - commission_cost

                profit = (sell_price - pos['buy_price']) / pos['buy_price'] * 100
                trades.append({
                    'date': final_date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': sell_price,
                    'shares': pos['shares'],
                    'value': sell_value,
                    'profit_pct': profit,
                    'hold_days': (final_date - pos['buy_date']).days,
                    'reason': "回测结束"
                })
                print(f"  [{final_date}] 卖出 {symbol}: ¥{sell_price:.2f} ({profit:+.1f}%)")

    # Calculate results
    final_value = cash
    total_return = (final_value - initial_cash) / initial_cash * 100

    # Calculate annualized return
    days = (end_date - start_date).days
    annual_return = ((final_value / initial_cash) ** (365 / max(days, 1)) - 1) * 100

    # Calculate win rate
    sell_trades = [t for t in trades if t['action'] == 'SELL']
    winning_trades = [t for t in sell_trades if t['profit_pct'] > 0]

    # Print results
    print("\n" + "=" * 70)
    print("📈 回测结果")
    print("=" * 70)
    print(f"初始资金: ¥{initial_cash:,.2f}")
    print(f"最终资金: ¥{final_value:,.2f}")
    print(f"总收益: {total_return:+.2f}%")
    print(f"年化收益: {annual_return:+.2f}%")
    print(f"总交易次数: {len(trades)}")
    print(f"买入次数: {len([t for t in trades if t['action'] == 'BUY'])}")
    print(f"卖出次数: {len(sell_trades)}")
    print(f"胜率: {len(winning_trades) / max(len(sell_trades), 1) * 100:.1f}%")

    if sell_trades:
        avg_profit = np.mean([t['profit_pct'] for t in sell_trades])
        max_profit = max([t['profit_pct'] for t in sell_trades])
        max_loss = min([t['profit_pct'] for t in sell_trades])
        avg_hold = np.mean([t['hold_days'] for t in sell_trades])
        print(f"平均单笔收益: {avg_profit:+.2f}%")
        print(f"平均持仓天数: {avg_hold:.1f}天")
        print(f"最大盈利: {max_profit:+.2f}%")
        print(f"最大亏损: {max_loss:+.2f}%")

        # 盈亏比
        if winning_trades:
            avg_win = np.mean([t['profit_pct'] for t in winning_trades])
            losing_trades = [t for t in sell_trades if t['profit_pct'] <= 0]
            if losing_trades:
                avg_loss = np.mean([abs(t['profit_pct']) for t in losing_trades])
                print(f"盈亏比: {avg_win / avg_loss:.2f}")

    print("=" * 70)

    # 按卖出原因统计
    if sell_trades:
        print("\n卖出原因统计:")
        reasons = {}
        for t in sell_trades:
            reason_type = t['reason'].split('(')[0]
            reasons[reason_type] = reasons.get(reason_type, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}次")

    return {
        'initial_cash': initial_cash,
        'final_value': final_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'trades': trades,
        'win_rate': len(winning_trades) / max(len(sell_trades), 1)
    }


if __name__ == "__main__":
    symbols = ['000876.SZ', '600515.SH', '688131.SH', '600600.SH', '600460.SH', '688271.SH', '001965.SZ']

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # 迭代14：增加仓位
    result = run_breakout_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        buy_score_threshold=60,      # 放宽买入
        trend_sell_threshold=25,     # 放宽卖出
        max_hold_days=40,            # 延长持仓
        stop_loss_pct=-10.0,         # 放宽止损
        take_profit_pct=30.0,        # 提高止盈
        max_positions=4,
        position_pct=0.28            # 增加单只仓位到28%
    )