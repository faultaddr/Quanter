"""
Dash-based user interface for the quantitative trading tool
"""
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from quant_trade_a_share import get_data, run_backtest, analyze_performance
from quant_trade_a_share.strategies import (
    MovingAverageCrossoverStrategy, 
    RSIStrategy, 
    MeanReversionStrategy, 
    BollingerBandsStrategy, 
    MACDStrategy
)
from quant_trade_a_share.analysis.performance_analyzer import PerformanceAnalyzer

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Define the layout
app.layout = html.Div([
    html.H1("Quantitative Trading Tool for A-Share Market", 
            style={'textAlign': 'center', 'marginBottom': 30}),
    
    # Input section
    html.Div([
        html.Div([
            html.Label('Stock Symbol:', style={'font-weight': 'bold'}),
            dcc.Input(id='symbol-input', value='000001', type='text', 
                      style={'width': '100%', 'padding': '5px'})
        ], className='six columns'),
        
        html.Div([
            html.Label('Start Date:', style={'font-weight': 'bold'}),
            dcc.DatePickerSingle(
                id='start-date',
                date='2020-01-01',
                display_format='YYYY-MM-DD'
            )
        ], className='three columns'),
        
        html.Div([
            html.Label('End Date:', style={'font-weight': 'bold'}),
            dcc.DatePickerSingle(
                id='end-date',
                date='2021-01-01',
                display_format='YYYY-MM-DD'
            )
        ], className='three columns'),
    ], className='row', style={'marginBottom': 20}),
    
    html.Div([
        html.Div([
            html.Label('Initial Capital:', style={'font-weight': 'bold'}),
            dcc.Input(id='capital-input', value=100000, type='number', 
                      style={'width': '100%', 'padding': '5px'})
        ], className='three columns'),
        
        html.Div([
            html.Label('Data Source:', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='data-source',
                options=[
                    {'label': 'EastMoney', 'value': 'eastmoney'},
                    {'label': 'Tushare', 'value': 'tushare'},
                    {'label': 'Baostock', 'value': 'baostock'},
                    {'label': 'Yahoo Finance', 'value': 'yahoo'}
                ],
                value='eastmoney'
            )
        ], className='three columns'),
        
        html.Div([
            html.Label('Trading Strategy:', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='strategy',
                options=[
                    {'label': 'Moving Average Crossover', 'value': 'ma_crossover'},
                    {'label': 'RSI Strategy', 'value': 'rsi'},
                    {'label': 'Mean Reversion', 'value': 'mean_reversion'},
                    {'label': 'Bollinger Bands', 'value': 'bollinger'},
                    {'label': 'MACD Strategy', 'value': 'macd'}
                ],
                value='ma_crossover'
            )
        ], className='three columns'),
        
        html.Div([
            html.Label('Run Backtest:', style={'font-weight': 'bold'}),
            html.Button('Run', id='run-button', n_clicks=0,
                        style={'width': '100%', 'padding': '10px', 'marginTop': '10px'})
        ], className='three columns'),
    ], className='row', style={'marginBottom': 20}),
    
    # Strategy-specific parameters
    html.Div(id='strategy-params', children=[
        # Default params for Moving Average Crossover
        html.Div([
            html.Div([
                html.Label('Short Window:', style={'font-weight': 'bold'}),
                dcc.Input(id='short-window', value=10, type='number', 
                          style={'width': '100%', 'padding': '5px'})
            ], className='three columns'),
            
            html.Div([
                html.Label('Long Window:', style={'font-weight': 'bold'}),
                dcc.Input(id='long-window', value=30, type='number', 
                          style={'width': '100%', 'padding': '5px'})
            ], className='three columns'),
        ], className='row', style={'marginBottom': 20})
    ]),
    
    # Results section
    html.Div([
        html.H3('Price Chart', style={'textAlign': 'center'}),
        dcc.Graph(id='price-chart')
    ], style={'marginBottom': 30}),
    
    html.Div([
        html.H3('Equity Curve', style={'textAlign': 'center'}),
        dcc.Graph(id='equity-curve')
    ], style={'marginBottom': 30}),
    
    html.Div([
        html.H3('Performance Metrics', style={'textAlign': 'center'}),
        html.Div(id='performance-metrics')
    ], style={'marginBottom': 30}),
    
    # Hidden div to store backtest results
    html.Div(id='backtest-results', style={'display': 'none'})
], style={'padding': '20px'})


@callback(
    Output('strategy-params', 'children'),
    [Input('strategy', 'value')]
)
def update_strategy_params(strategy):
    if strategy == "ma_crossover":
        result = [
            html.Div([
                html.Div([
                    html.Label("Short Window:", style={"font-weight": "bold"}),
                    dcc.Input(id="short-window", value=10, type="number", 
                              style={"width": "100%", "padding": "5px"})
                ], className="three columns"),
                
                html.Div([
                    html.Label("Long Window:", style={"font-weight": "bold"}),
                    dcc.Input(id="long-window", value=30, type="number", 
                              style={"width": "100%", "padding": "5px"})
                ], className="three columns"),
            ], className="row", style={"marginBottom": 20})
        ]
    elif strategy == "rsi":
        result = [
            html.Div([
                html.Div([
                    html.Label("RSI Period:", style={"font-weight": "bold"}),
                    dcc.Input(id="rsi-period", value=14, type="number", 
                              style={"width": "100%", "padding": "5px"})
                ], className="three columns"),
                
                html.Div([
                    html.Label("Oversold Level:", style={"font-weight": "bold"}),
                    dcc.Input(id="oversold", value=30, type="number", 
                              style={"width": "100%", "padding": "5px"})
                ], className="three columns"),
                
                html.Div([
                    html.Label("Overbought Level:", style={"font-weight": "bold"}),
                    dcc.Input(id="overbought", value=70, type="number", 
                              style={"width": "100%", "padding": "5px"})
                ], className="three columns"),
            ], className="row", style={"marginBottom": 20})
        ]
    elif strategy == "mean_reversion":
        result = [
            html.Div([
                html.Div([
                    html.Label("Window:", style={"font-weight": "bold"}),
                    dcc.Input(id="mr-window", value=20, type="number", 
                              style={"width": "100%", "padding": "5px"})
                ], className="three columns"),
                
                html.Div([
                    html.Label("Threshold (Std Devs):", style={"font-weight": "bold"}),
                    dcc.Input(id="mr-threshold", value=1.5, type="number", 
                              style={"width": "100%", "padding": "5px"})
                ], className="three columns"),
            ], className="row", style={"marginBottom": 20})
        ]
    elif strategy == "bollinger":
        result = [
            html.Div([
                html.Div([
                    html.Label("Window:", style={"font-weight": "bold"}),
                    dcc.Input(id="bb-window", value=20, type="number", 
                              style={"width": "100%", "padding": "5px"})
                ], className="three columns"),
                
                html.Div([
                    html.Label("Num Std Devs:", style={"font-weight": "bold"}),
                    dcc.Input(id="bb-num-std", value=2, type="number", 
                              style={"width": "100%", "padding": "5px"})
                ], className="three columns"),
            ], className="row", style={"marginBottom": 20})
        ]
    elif strategy == "macd":
        result = [
            html.Div([
                html.Div([
                    html.Label("Fast Period:", style={"font-weight": "bold"}),
                    dcc.Input(id="fast-period", value=12, type="number", 
                              style={"width": "100%", "padding": "5px"})
                ], className="three columns"),
                
                html.Div([
                    html.Label("Slow Period:", style={"font-weight": "bold"}),
                    dcc.Input(id="slow-period", value=26, type="number", 
                              style={"width": "100%", "padding": "5px"})
                ], className="three columns"),
                
                html.Div([
                    html.Label("Signal Period:", style={"font-weight": "bold"}),
                    dcc.Input(id="signal-period", value=9, type="number", 
                              style={"width": "100%", "padding": "5px"})
                ], className="three columns"),
            ], className="row", style={"marginBottom": 20})
        ]
    else:
        result = []
    
    return result
"""
Dash-based user interface for the quantitative trading tool
"""
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from quant_trade_a_share import get_data, run_backtest, analyze_performance
from quant_trade_a_share.strategies import (
    MovingAverageCrossoverStrategy, 
    RSIStrategy, 
    MeanReversionStrategy, 
    BollingerBandsStrategy, 
    MACDStrategy
)
from quant_trade_a_share.analysis.performance_analyzer import PerformanceAnalyzer

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Define the layout
app.layout = html.Div([
    html.H1("Quantitative Trading Tool for A-Share Market", 
            style={'textAlign': 'center', 'marginBottom': 30}),
    
    # Input section
    html.Div([
        html.Div([
            html.Label('Stock Symbol:', style={'font-weight': 'bold'}),
            dcc.Input(id='symbol-input', value='000001', type='text', 
                      style={'width': '100%', 'padding': '5px'})
        ], className='six columns'),
        
        html.Div([
            html.Label('Start Date:', style={'font-weight': 'bold'}),
            dcc.DatePickerSingle(
                id='start-date',
                date='2020-01-01',
                display_format='YYYY-MM-DD'
            )
        ], className='three columns'),
        
        html.Div([
            html.Label('End Date:', style={'font-weight': 'bold'}),
            dcc.DatePickerSingle(
                id='end-date',
                date='2021-01-01',
                display_format='YYYY-MM-DD'
            )
        ], className='three columns'),
    ], className='row', style={'marginBottom': 20}),
    
    html.Div([
        html.Div([
            html.Label('Initial Capital:', style={'font-weight': 'bold'}),
            dcc.Input(id='capital-input', value=100000, type='number', 
                      style={'width': '100%', 'padding': '5px'})
        ], className='three columns'),
        
        html.Div([
            html.Label('Data Source:', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='data-source',
                options=[
                    {'label': 'EastMoney', 'value': 'eastmoney'},
                    {'label': 'Tushare', 'value': 'tushare'},
                    {'label': 'Baostock', 'value': 'baostock'},
                    {'label': 'Yahoo Finance', 'value': 'yahoo'}
                ],
                value='eastmoney'
            )
        ], className='three columns'),
        
        html.Div([
            html.Label('Trading Strategy:', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='strategy',
                options=[
                    {'label': 'Moving Average Crossover', 'value': 'ma_crossover'},
                    {'label': 'RSI Strategy', 'value': 'rsi'},
                    {'label': 'Mean Reversion', 'value': 'mean_reversion'},
                    {'label': 'Bollinger Bands', 'value': 'bollinger'},
                    {'label': 'MACD Strategy', 'value': 'macd'}
                ],
                value='ma_crossover'
            )
        ], className='three columns'),
        
        html.Div([
            html.Label('Run Backtest:', style={'font-weight': 'bold'}),
            html.Button('Run', id='run-button', n_clicks=0,
                        style={'width': '100%', 'padding': '10px', 'marginTop': '10px'})
        ], className='three columns'),
    ], className='row', style={'marginBottom': 20}),
    
    # Strategy-specific parameters
    html.Div(id='strategy-params', children=[
        # Default params for Moving Average Crossover
        html.Div([
            html.Div([
                html.Label('Short Window:', style={'font-weight': 'bold'}),
                dcc.Input(id='short-window', value=10, type='number', 
                          style={'width': '100%', 'padding': '5px'})
            ], className='three columns'),
            
            html.Div([
                html.Label('Long Window:', style={'font-weight': 'bold'}),
                dcc.Input(id='long-window', value=30, type='number', 
                          style={'width': '100%', 'padding': '5px'})
            ], className='three columns'),
        ], className='row', style={'marginBottom': 20})
    ]),
    
    # Results section
    html.Div([
        html.H3('Price Chart', style={'textAlign': 'center'}),
        dcc.Graph(id='price-chart')
    ], style={'marginBottom': 30}),
    
    html.Div([
        html.H3('Equity Curve', style={'textAlign': 'center'}),
        dcc.Graph(id='equity-curve')
    ], style={'marginBottom': 30}),
    
    html.Div([
        html.H3('Performance Metrics', style={'textAlign': 'center'}),
        html.Div(id='performance-metrics')
    ], style={'marginBottom': 30}),
    
    # Hidden div to store backtest results
    html.Div(id='backtest-results', style={'display': 'none'})
], style={'padding': '20px'})


@callback(
    [Output('backtest-results', 'children'),
     Output('price-chart', 'figure'),
     Output('equity-curve', 'figure'),
     Output('performance-metrics', 'children')],
    [Input('run-button', 'n_clicks')],
    [State('symbol-input', 'value'),
     State('start-date', 'date'),
     State('end-date', 'date'),
     State('capital-input', 'value'),
     State('data-source', 'value'),
     State('strategy', 'value'),
     State('short-window', 'value'),
     State('long-window', 'value'),
     State('rsi-period', 'value'),
     State('oversold', 'value'),
     State('overbought', 'value'),
     State('mr-window', 'value'),
     State('mr-threshold', 'value'),
     State('bb-window', 'value'),
     State('bb-num-std', 'value'),
     State('fast-period', 'value'),
     State('slow-period', 'value'),
     State('signal-period', 'value')]
)
def run_backtest_callback(n_clicks, symbol, start_date, end_date, capital, 
                         data_source, strategy, short_window, long_window,
                         rsi_period, oversold, overbought, mr_window, mr_threshold,
                         bb_window, bb_num_std, fast_period, slow_period, signal_period):
    
    if n_clicks == 0:
        # Return empty figures and metrics on initial load
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Run a backtest to see results")
        
        metrics_div = html.Div("Run a backtest to see performance metrics")
        
        return "", empty_fig, empty_fig, metrics_div
    
    # Create strategy based on selected type and parameters
    if strategy == 'ma_crossover':
        strat = MovingAverageCrossoverStrategy(short_window=short_window, long_window=long_window)
    elif strategy == 'rsi':
        strat = RSIStrategy(rsi_period=rsi_period, oversold=oversold, overbought=overbought)
    elif strategy == 'mean_reversion':
        strat = MeanReversionStrategy(window=mr_window, threshold=mr_threshold)
    elif strategy == 'bollinger':
        strat = BollingerBandsStrategy(window=bb_window, num_std=bb_num_std)
    elif strategy == 'macd':
        strat = MACDStrategy(fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)
    else:
        strat = MovingAverageCrossoverStrategy()
    
    # Run the backtest
    try:
        results = run_backtest(strat, start_date, end_date, initial_capital=capital, 
                              symbol=symbol, data_source=data_source)
        
        # Get the historical data for the price chart
        data = get_data(symbol, start_date, end_date, source=data_source)
        
        # Create price chart
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=data.index, y=data['close'], 
                                      mode='lines', name='Close Price'))
        
        # If we have signals, plot them
        if hasattr(strat, 'generate_signals'):
            signals = strat.generate_signals(data)
            buy_signals = data[signals == 1]
            sell_signals = data[signals == -1]
            
            if not buy_signals.empty:
                price_fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['close'],
                                              mode='markers', name='Buy Signal', 
                                              marker=dict(color='green', size=10, symbol='triangle-up')))
            
            if not sell_signals.empty:
                price_fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['close'],
                                              mode='markers', name='Sell Signal', 
                                              marker=dict(color='red', size=10, symbol='triangle-down')))
        
        price_fig.update_layout(title=f'Price Chart for {symbol}',
                               xaxis_title='Date',
                               yaxis_title='Price (CNY)')
        
        # Create equity curve
        equity_fig = go.Figure()
        if 'equity_curve' in results:
            equity_fig.add_trace(go.Scatter(x=results['equity_curve'].index, 
                                           y=results['equity_curve']['value'],
                                           mode='lines', name='Equity Curve'))
            equity_fig.update_layout(title='Equity Curve',
                                    xaxis_title='Date',
                                    yaxis_title='Portfolio Value (CNY)')
        else:
            equity_fig.update_layout(title='No equity curve available')
        
        # Analyze performance
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.analyze(results)
        
        # Create metrics display
        metrics_div = html.Div([
            html.Table([
                html.Tr([html.Td("Total Return:", style={'fontWeight': 'bold'}), 
                        html.Td(f"{metrics.get('total_return', 0):.2%}")]),
                html.Tr([html.Td("Annualized Return:", style={'fontWeight': 'bold'}), 
                        html.Td(f"{metrics.get('annualized_return', 0):.2%}")]),
                html.Tr([html.Td("Volatility:", style={'fontWeight': 'bold'}), 
                        html.Td(f"{metrics.get('annualized_volatility', 0):.2%}")]),
                html.Tr([html.Td("Sharpe Ratio:", style={'fontWeight': 'bold'}), 
                        html.Td(f"{metrics.get('sharpe_ratio', 0):.2f}")]),
                html.Tr([html.Td("Max Drawdown:", style={'fontWeight': 'bold'}), 
                        html.Td(f"{metrics.get('max_drawdown', 0):.2%}")]),
                html.Tr([html.Td("Win Rate:", style={'fontWeight': 'bold'}), 
                        html.Td(f"{metrics.get('win_rate', 0):.2%}")]),
                html.Tr([html.Td("Profit Factor:", style={'fontWeight': 'bold'}), 
                        html.Td(f"{metrics.get('profit_factor', 0):.2f}")]),
                html.Tr([html.Td("Number of Trades:", style={'fontWeight': 'bold'}), 
                        html.Td(metrics.get('num_trades', 0))]),
            ], style={'width': '100%', 'borderCollapse': 'collapse'})
        ])
        
        # Return results as JSON string to store in hidden div
        import json
        results_json = json.dumps(results, default=str)
        
        return results_json, price_fig, equity_fig, metrics_div
        
    except Exception as e:
        # Handle errors gracefully
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Error: {str(e)}")
        
        error_metrics = html.Div(f"Error running backtest: {str(e)}")
        
        return "", error_fig, error_fig, error_metrics


if __name__ == '__main__':
    app.run_server(debug=True)