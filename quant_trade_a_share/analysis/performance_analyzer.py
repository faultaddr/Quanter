"""
Performance analysis tools for backtesting results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """
    Analyzes the performance of trading strategies
    """
    
    def __init__(self):
        pass
    
    def analyze(self, results):
        """
        Analyze backtest results and generate performance metrics
        
        Args:
            results (dict): Backtest results from backtester.run()
        
        Returns:
            dict: Performance metrics
        """
        if not results or 'equity_curve' not in results:
            return {}
        
        # Extract equity curve
        equity_curve = results['equity_curve']['value']
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Calculate performance metrics
        metrics = {}
        
        # Total return
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        metrics['total_return'] = total_return
        
        # Annualized return (assuming 252 trading days in a year for Chinese market)
        trading_days = len(equity_curve)
        years = trading_days / 244  # Approximate number of trading days in China per year
        annualized_return = (1 + total_return) ** (1 / years) - 1
        metrics['annualized_return'] = annualized_return
        
        # Volatility (annualized standard deviation)
        annualized_volatility = returns.std() * np.sqrt(244)
        metrics['annualized_volatility'] = annualized_volatility
        
        # Sharpe ratio (risk-free rate assumed to be 0.03 or 3% annually)
        risk_free_rate = 0.03
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        metrics['sharpe_ratio'] = sharpe_ratio
        
        # Max drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        metrics['max_drawdown'] = max_drawdown
        
        # Calmar ratio (return over max drawdown)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
        metrics['calmar_ratio'] = calmar_ratio
        
        # Win rate
        win_rate = len(returns[returns > 0]) / len(returns[returns != 0])
        metrics['win_rate'] = win_rate
        
        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses != 0 else np.inf
        metrics['profit_factor'] = profit_factor
        
        # Value at Risk (VaR) at 5% confidence level
        var_5percent = np.percentile(returns.dropna(), 5)
        metrics['var_5percent'] = var_5percent
        
        # Sortino ratio (only considers downside volatility)
        downside_returns = returns[returns < 0]
        downside_deviation = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(244)
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else np.inf
        metrics['sortino_ratio'] = sortino_ratio
        
        # Beta and Alpha relative to benchmark (e.g., SSE Composite Index)
        # For now, we'll skip this as we don't have benchmark data
        metrics['beta'] = np.nan
        metrics['alpha'] = np.nan
        
        # Number of trades
        metrics['num_trades'] = len(results.get('trades', []))
        
        # Store returns for further analysis
        metrics['returns'] = returns
        metrics['equity_curve'] = equity_curve
        
        return metrics
    
    def plot_performance(self, results, benchmark_symbol=None):
        """
        Plot performance charts
        
        Args:
            results (dict): Backtest results
            benchmark_symbol (str): Symbol for benchmark comparison (optional)
        """
        if 'equity_curve' not in results:
            print("No equity curve in results to plot.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        equity_curve = results['equity_curve']['value']
        
        # Equity curve
        axes[0, 0].plot(equity_curve.index, equity_curve.values, label='Strategy', linewidth=2)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Portfolio Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        axes[0, 1].fill_between(drawdown.index, 0, drawdown * 100, alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown (%)')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True)
        
        # Returns distribution
        returns = equity_curve.pct_change().dropna()
        axes[1, 0].hist(returns, bins=50, density=True, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Return')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True)
        
        # Rolling Sharpe ratio (using 252-day window)
        rolling_returns = returns.rolling(window=252).mean()
        rolling_vol = returns.rolling(window=252).std() * np.sqrt(244)
        rolling_sharpe = rolling_returns / rolling_vol
        axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe, label='Rolling Sharpe Ratio', color='purple')
        axes[1, 1].set_title('Rolling Sharpe Ratio (252 days)')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def compare_strategies(self, results_list, labels):
        """
        Compare performance of multiple strategies
        
        Args:
            results_list (list): List of backtest results
            labels (list): Labels for each strategy
        
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for i, results in enumerate(results_list):
            metrics = self.analyze(results)
            
            row = {
                'Strategy': labels[i],
                'Total Return': f"{metrics.get('total_return', 0):.2%}",
                'Annualized Return': f"{metrics.get('annualized_return', 0):.2%}",
                'Volatility': f"{metrics.get('annualized_volatility', 0):.2%}",
                'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
                'Win Rate': f"{metrics.get('win_rate', 0):.2%}",
                'Profit Factor': f"{metrics.get('profit_factor', 0):.2f}",
                'Sortino Ratio': f"{metrics.get('sortino_ratio', 0):.2f}"
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df
    
    def generate_report(self, results, title="Strategy Performance Report"):
        """
        Generate a comprehensive performance report
        
        Args:
            results (dict): Backtest results
            title (str): Title for the report
        
        Returns:
            str: Performance report as string
        """
        metrics = self.analyze(results)
        
        report = f"""
{title}
{'='*len(title)}

SUMMARY STATISTICS
------------------
Total Return:           {metrics.get('total_return', 0):.2%}
Annualized Return:      {metrics.get('annualized_return', 0):.2%}
Annualized Volatility:  {metrics.get('annualized_volatility', 0):.2%}
Max Drawdown:           {metrics.get('max_drawdown', 0):.2%}

RISK-ADJUSTED METRICS
---------------------
Sharpe Ratio:           {metrics.get('sharpe_ratio', 0):.2f}
Sortino Ratio:          {metrics.get('sortino_ratio', 0):.2f}
Calmar Ratio:           {metrics.get('calmar_ratio', 0):.2f}

TRADE STATISTICS
----------------
Win Rate:               {metrics.get('win_rate', 0):.2%}
Profit Factor:          {metrics.get('profit_factor', 0):.2f}
Number of Trades:       {metrics.get('num_trades', 0)}
Value at Risk (5%):     {metrics.get('var_5percent', 0):.2%}

Additional Notes:
- Risk-free rate assumed at 3% annually
- Annualized using 244 trading days per year (approximation for Chinese market)
- All metrics calculated based on daily returns
        """
        
        return report