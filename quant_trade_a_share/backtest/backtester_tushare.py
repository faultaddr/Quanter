"""
Backtesting module using Tushare data for historical analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tushare as ts


class BacktesterWithTushare:
    """
    Backtester that uses Tushare data for historical analysis
    """
    def __init__(self, tushare_token):
        ts.set_token(tushare_token)
        self.pro = ts.pro_api()
        print("✅ Tushare回测模块初始化完成")
    
    def get_historical_data(self, symbol, start_date, end_date, freq='D'):
        """
        Get historical data from Tushare for backtesting
        """
        try:
            if freq != 'D':
                # Get minute-level data for backtesting
                df = self.pro.query('bar', ts_code=symbol, freq=freq, start_date=start_date, end_date=end_date)
            else:
                # Get daily data for backtesting
                df = self.pro.query('daily', ts_code=symbol, start_date=start_date, end_date=end_date)
            
            if df is not None and not df.empty:
                # Rename columns to standard format
                if freq != 'D':
                    df.rename(columns={
                        'trade_time': 'date',
                        'ts_code': 'symbol',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'vol': 'volume'
                    }, inplace=True)
                else:
                    df.rename(columns={
                        'trade_date': 'date',
                        'ts_code': 'symbol',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'vol': 'volume'
                    }, inplace=True)
                
                # Convert date column to datetime
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                print(f"✅ 获取到 {len(df)} 条 {symbol} 历史数据")
                return df
            else:
                print(f"❌ 未获取到 {symbol} 的历史数据")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ 获取 {symbol} 历史数据失败: {e}")
            return pd.DataFrame()
    
    def run_backtest(self, strategy, symbol, start_date, end_date, initial_capital=100000, freq='D'):
        """
        Run a backtest using Tushare historical data
        """
        print(f"🚀 开始对 {symbol} 进行回测...")
        print(f"📊 回测周期: {start_date} 至 {end_date}")
        
        # Get historical data
        data = self.get_historical_data(symbol, start_date, end_date, freq)
        
        if data.empty:
            print("❌ 历史数据为空，无法进行回测")
            return None
        
        # Generate signals using the strategy
        signals = strategy.generate_signals(data)
        
        # Initialize portfolio
        cash = initial_capital
        position = 0  # Number of shares held
        position_value = 0
        portfolio_values = []
        dates = []
        
        # Run the backtest
        for i, (date, row) in enumerate(data.iterrows()):
            current_price = row['close']
            
            # Process signal if available
            if i < len(signals) and not pd.isna(signals.iloc[i]):
                signal = int(signals.iloc[i])
                
                if signal == 1:  # Buy signal
                    # Calculate how much to buy (use 90% of available cash to allow for fluctuations)
                    buy_amount = cash * 0.9
                    shares_to_buy = int(buy_amount / current_price)
                    
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        cash -= cost
                        position += shares_to_buy
                        print(f"🟢 {date.strftime('%Y-%m-%d')} 买入 {shares_to_buy} 股 @ ¥{current_price:.2f}")
                
                elif signal == -1:  # Sell signal
                    if position > 0:
                        proceeds = position * current_price
                        cash += proceeds
                        print(f"🔴 {date.strftime('%Y-%m-%d')} 卖出 {position} 股 @ ¥{current_price:.2f}")
                        position = 0  # Clear position
            
            # Calculate current portfolio value
            current_value = cash + (position * current_price)
            portfolio_values.append(current_value)
            dates.append(date)
        
        # Calculate performance metrics
        if len(portfolio_values) > 1:
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            total_return = (final_value - initial_value) / initial_value
            peak_value = max(portfolio_values)
            lowest_value = min(portfolio_values)
            
            # Calculate max drawdown
            max_so_far = portfolio_values[0]
            max_drawdown = 0
            for value in portfolio_values:
                if value > max_so_far:
                    max_so_far = value
                drawdown = (max_so_far - value) / max_so_far
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # Calculate annualized return if period is longer than 1 year
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            years = (end_dt - start_dt).days / 365.25
            
            if years > 0:
                annualized_return = (final_value / initial_value) ** (1/years) - 1
            else:
                annualized_return = 0
            
            results = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'max_drawdown': max_drawdown,
                'trading_days': len(portfolio_values),
                'peak_value': peak_value,
                'lowest_value': lowest_value,
                'portfolio_values': portfolio_values,
                'dates': dates
            }
            
            print(f"✅ 回测完成!")
            print(f"📈 总收益率: {total_return:.2%}")
            print(f"📊 年化收益率: {annualized_return:.2%}")
            print(f"📉 最大回撤: {max_drawdown:.2%}")
            print(f"💰 最终价值: ¥{final_value:,.2f}")
            
            return results
        else:
            print("❌ 回测数据不足")
            return None


def run_example_backtest():
    """
    Example of how to run a backtest
    """
    from quant_trade_a_share.strategies.strategy_tools import MovingAverageCrossoverStrategy
    
    # Use the provided token
    token = "744295f7af6adf63074518f919f5ad5054caf8b84d3c07c066f5c42e"
    
    # Initialize backtester
    backtester = BacktesterWithTushare(token)
    
    # Create a strategy
    strategy = MovingAverageCrossoverStrategy(short_window=10, long_window=30)
    
    # Run backtest
    results = backtester.run_backtest(
        strategy=strategy,
        symbol='000001.SZ',  # Ping An Bank
        start_date='20220101',
        end_date='20221231',
        initial_capital=100000,
        freq='D'  # Daily data for this example
    )
    
    return results


if __name__ == "__main__":
    print("🔍 Tushare回测模块测试")
    results = run_example_backtest()
    if results:
        print("\\n🎯 回测结果摘要:")
        for key, value in results.items():
            if key not in ['portfolio_values', 'dates']:  # Skip large arrays
                print(f"  {key}: {value}")