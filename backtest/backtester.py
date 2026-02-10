"""
Backtesting engine for quantitative trading strategies
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from ..data.data_fetcher import DataFetcher


class Position:
    def __init__(self, symbol, quantity, entry_price, entry_date):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.exit_price = None
        self.exit_date = None
        self.is_closed = False
    
    def close(self, exit_price, exit_date):
        self.exit_price = exit_price
        self.exit_date = exit_date
        self.is_closed = True
    
    def current_value(self, current_price):
        return self.quantity * current_price
    
    def unrealized_pnl(self, current_price):
        if not self.is_closed:
            return (current_price - self.entry_price) * self.quantity
        return 0
    
    def realized_pnl(self):
        if self.is_closed and self.exit_price:
            return (self.exit_price - self.entry_price) * self.quantity
        return 0


class Portfolio:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> position
        self.cash = initial_capital
        self.transaction_log = []
        self.historical_values = []
        self.dates = []
    
    def buy(self, symbol, quantity, price, date, commission_rate=0.001):
        cost = quantity * price
        commission = cost * commission_rate
        total_cost = cost + commission
        
        if total_cost <= self.cash:
            self.cash -= total_cost
            
            # Close existing position if exists
            if symbol in self.positions and not self.positions[symbol].is_closed:
                self.close_position(symbol, price, date, commission_rate)
            
            # Create new position
            position = Position(symbol, quantity, price, date)
            self.positions[symbol] = position
            
            # Log transaction
            self.transaction_log.append({
                'date': date,
                'symbol': symbol,
                'action': 'BUY',
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'total_cost': total_cost
            })
            
            return True
        return False
    
    def sell(self, symbol, quantity, price, date, commission_rate=0.001):
        if symbol in self.positions and not self.positions[symbol].is_closed:
            position = self.positions[symbol]
            
            if quantity <= position.quantity:
                proceeds = quantity * price
                commission = proceeds * commission_rate
                net_proceeds = proceeds - commission
                
                self.cash += net_proceeds
                
                # Reduce position size or close it
                if quantity == position.quantity:
                    position.close(price, date)
                else:
                    position.quantity -= quantity
                
                # Log transaction
                self.transaction_log.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': quantity,
                    'price': price,
                    'commission': commission,
                    'net_proceeds': net_proceeds
                })
                
                return True
        return False
    
    def close_position(self, symbol, price, date, commission_rate=0.001):
        if symbol in self.positions and not self.positions[symbol].is_closed:
            position = self.positions[symbol]
            return self.sell(symbol, position.quantity, price, date, commission_rate)
        return False
    
    def calculate_total_value(self, current_prices):
        """Calculate total portfolio value including cash and positions"""
        positions_value = 0
        for symbol, position in self.positions.items():
            if not position.is_closed and symbol in current_prices:
                positions_value += position.current_value(current_prices[symbol])
        return self.cash + positions_value
    
    def record_value(self, date, current_prices):
        """Record portfolio value at given date"""
        total_value = self.calculate_total_value(current_prices)
        self.historical_values.append(total_value)
        self.dates.append(date)


class Backtester:
    def __init__(self):
        self.data_fetcher = DataFetcher()
    
    def run(self, strategy, start_date, end_date, initial_capital=100000, symbol='000001',
            data_source='eastmoney', commission_rate=0.001):
        """
        Run a backtest for a given strategy
        
        Args:
            strategy: Trading strategy object with generate_signals method
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            initial_capital (float): Initial capital for backtest
            symbol (str): Stock symbol to trade
            data_source (str): Data source to use
            commission_rate (float): Commission rate per transaction
        
        Returns:
            dict: Backtest results
        """
        # Fetch historical data
        if data_source == 'mock':
            # Use mock data generator
            from ..utils.mock_data_generator import generate_mock_data
            data = generate_mock_data(symbol, start_date, end_date)
        else:
            data = self.data_fetcher.fetch(symbol, start_date, end_date, data_source)
        
        if data.empty:
            print(f"No data available for {symbol} in the specified date range.")
            return {}
        
        return self.run_with_data(strategy, data, initial_capital, symbol, commission_rate)
    
    def run_with_data(self, strategy, data, initial_capital=100000, symbol='000001', commission_rate=0.001):
        """
        Run a backtest for a given strategy using provided data
        
        Args:
            strategy: Trading strategy object with generate_signals method
            data (pd.DataFrame): Historical price data
            initial_capital (float): Initial capital for backtest
            symbol (str): Stock symbol to trade
            commission_rate (float): Commission rate per transaction
        
        Returns:
            dict: Backtest results
        """
        # Initialize portfolio
        portfolio = Portfolio(initial_capital)
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Align signals with data
        data = data.copy()
        data['signal'] = signals
        
        # Execute trades based on signals
        for idx, row in data.iterrows():
            date = idx
            price = row['close']
            
            # Record portfolio value
            current_prices = {symbol: price}
            portfolio.record_value(date, current_prices)
            
            # Process signal
            if 'signal' in row and not pd.isna(row['signal']):
                signal = int(row['signal'])
                
                if signal == 1:  # Buy signal
                    # Calculate how many shares we can buy with available cash
                    max_shares = int(portfolio.cash / (price * (1 + commission_rate)))
                    if max_shares > 0:
                        portfolio.buy(symbol, max_shares, price, date, commission_rate)
                        
                elif signal == -1:  # Sell signal
                    if symbol in portfolio.positions and not portfolio.positions[symbol].is_closed:
                        pos = portfolio.positions[symbol]
                        portfolio.sell(symbol, pos.quantity, price, date, commission_rate)
        
        # Prepare results
        results = {
            'portfolio': portfolio,
            'trades': portfolio.transaction_log,
            'equity_curve': pd.DataFrame({
                'date': portfolio.dates,
                'value': portfolio.historical_values
            }).set_index('date'),
            'final_value': portfolio.calculate_total_value({symbol: data.iloc[-1]['close']}),
            'total_return': (portfolio.calculate_total_value({symbol: data.iloc[-1]['close']}) - portfolio.initial_capital) / portfolio.initial_capital,
            'symbol': symbol,
            'start_date': data.index[0].strftime('%Y-%m-%d'),
            'end_date': data.index[-1].strftime('%Y-%m-%d'),
            'initial_capital': portfolio.initial_capital
        }
        
        return results
    
    def plot_results(self, results):
        """Plot backtest results"""
        if 'equity_curve' not in results:
            print("No equity curve in results to plot.")
            return
        
        plt.figure(figsize=(14, 8))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(results['equity_curve'].index, results['equity_curve']['value'], label='Portfolio Value')
        plt.axhline(y=results['initial_capital'], color='r', linestyle='--', label='Initial Capital')
        plt.title(f'Equity Curve for {results["symbol"]}')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (CNY)')
        plt.legend()
        plt.grid(True)
        
        # Plot returns
        equity_curve = results['equity_curve']['value']
        returns = equity_curve.pct_change().dropna()
        
        plt.subplot(2, 1, 2)
        plt.plot(returns.index, returns, label='Daily Returns', alpha=0.7)
        plt.title('Daily Returns')
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


class BaseStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, **kwargs):
        self.parameters = kwargs
    
    def generate_signals(self, data):
        """
        Generate trading signals based on historical data
        
        Args:
            data (pd.DataFrame): Historical price data with columns [open, high, low, close, volume]
        
        Returns:
            pd.Series: Trading signals (1 for buy, -1 for sell, 0 for hold)
        """
        raise NotImplementedError("Subclasses must implement generate_signals method")