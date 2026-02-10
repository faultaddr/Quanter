"""
Visualization Dashboard for A-Share Market Analysis
Interactive dashboard for stock screening, strategy analysis, and signal generation
"""
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import our custom modules
from quant_trade_a_share.screeners.stock_screener import StockScreener
from quant_trade_a_share.strategies.strategy_tools import StrategyManager


# Initialize the app
app = dash.Dash(__name__)
server = app.server  # Expose the server for deployment

# Initialize components
screener = StockScreener()
strategy_manager = StrategyManager()


# Define the layout
app.layout = html.Div([
    html.H1("A股市场量化分析仪表板", style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    html.Div([
        # Stock Screener Section
        html.Div([
            html.H2("股票筛选器", style={'color': '#34495e'}),
            
            html.Div([
                html.Label("最低价格:"),
                dcc.Input(id='min-price', type='number', value=10, style={'width': '45%', 'marginRight': '5%'}),
                
                html.Label("最高价格:"),
                dcc.Input(id='max-price', type='number', value=150, style={'width': '45%'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '10px'}),
            
            html.Div([
                html.Label("最小成交量:"),
                dcc.Input(id='min-volume', type='number', value=5000000, style={'width': '45%', 'marginRight': '5%'}),
                
                html.Label("分析天数:"),
                dcc.Input(id='days-back', type='number', value=60, style={'width': '45%'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '10px'}),
            
            html.Button('筛选股票', id='screen-btn', n_clicks=0, 
                       style={'backgroundColor': '#3498db', 'color': 'white', 'border': 'none', 
                              'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer'}),
            
            html.Div(id='screener-output', style={'marginTop': '20px'}),
        ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 
                  'border': '1px solid #ddd', 'borderRadius': '5px', 'marginRight': '5%'}),
        
        # Strategy Analysis Section
        html.Div([
            html.H2("策略分析", style={'color': '#34495e'}),
            
            html.Div([
                html.Label("选择股票:"),
                dcc.Dropdown(
                    id='stock-selector',
                    options=[],
                    placeholder="请选择股票"
                ),
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.Label("选择策略:"),
                dcc.Dropdown(
                    id='strategy-selector',
                    options=[
                        {'label': '移动平均线交叉', 'value': 'ma_crossover'},
                        {'label': 'RSI策略', 'value': 'rsi'},
                        {'label': 'MACD策略', 'value': 'macd'},
                        {'label': '布林带策略', 'value': 'bollinger'},
                        {'label': '均值回归', 'value': 'mean_reversion'},
                        {'label': '突破策略', 'value': 'breakout'}
                    ],
                    value='ma_crossover',
                    placeholder="请选择策略"
                ),
            ], style={'marginBottom': '10px'}),
            
            html.Button('运行策略', id='run-strategy-btn', n_clicks=0,
                       style={'backgroundColor': '#2ecc71', 'color': 'white', 'border': 'none', 
                              'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer'}),
            
            html.Div(id='strategy-output', style={'marginTop': '20px'}),
        ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 
                  'border': '1px solid #ddd', 'borderRadius': '5px'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    
    # Chart Section
    html.Div([
        html.H2("图表分析", style={'color': '#34495e', 'textAlign': 'center'}),
        dcc.Graph(id='price-chart', style={'height': '600px'}),
    ], style={'marginTop': '20px', 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
    
    # Signal Notifications
    html.Div([
        html.H2("信号通知", style={'color': '#34495e', 'textAlign': 'center'}),
        html.Div(id='signal-notifications', style={'padding': '10px', 'backgroundColor': '#f8f9fa', 
                                                  'borderRadius': '5px', 'minHeight': '100px'}),
    ], style={'marginTop': '20px', 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
    
    # Hidden div to store data
    html.Div(id='stored-data', style={'display': 'none'}),
])


@app.callback(
    Output('screener-output', 'children'),
    [Input('screen-btn', 'n_clicks')],
    [State('min-price', 'value'),
     State('max-price', 'value'),
     State('min-volume', 'value'),
     State('days-back', 'value')]
)
def screen_stocks(n_clicks, min_price, max_price, min_volume, days_back):
    if n_clicks > 0:
        try:
            # Apply filters
            filters = {
                'min_price': min_price or 10,
                'max_price': max_price or 150,
                'min_volume': min_volume or 5000000,
                'days_back': days_back or 60
            }
            
            results = screener.screen_stocks(filters)
            
            if results is not None and not results.empty:
                # Update dropdown options with screened stocks
                stock_options = [{'label': f"{row['name']} ({row['code']})", 'value': row['symbol']} 
                                for _, row in results.iterrows()]
                
                # Store options in a hidden div for later use
                return html.Div([
                    html.H3(f"找到 {len(results)} 只符合条件的股票", style={'color': '#27ae60'}),
                    html.Table([
                        html.Thead([
                            html.Tr([html.Th(col) for col in results.columns])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(results.iloc[i][col]) for col in results.columns
                            ]) for i in range(min(10, len(results)))  # Show top 10
                        ])
                    ], style={'width': '100%', 'borderCollapse': 'collapse', 'marginTop': '10px'}),
                    dcc.Store(id='stock-options-store', data=stock_options)
                ])
            else:
                return html.Div("未找到符合条件的股票", style={'color': '#e74c3c'})
        except Exception as e:
            return html.Div(f"筛选过程中出现错误: {str(e)}", style={'color': '#e74c3c'})
    
    return html.Div("点击'筛选股票'按钮开始筛选")


@app.callback(
    Output('stock-selector', 'options'),
    [Input('screener-output', 'children')]
)
def update_stock_dropdown(screener_output):
    # This would normally get the options from the screener output
    # For now, return a default set
    try:
        # In a real implementation, we would extract the stock options from the screener output
        # Since we can't extract from the output directly, we'll return a default list
        return [{'label': '贵州茅台 (600519)', 'value': 'sh600519'}, 
                {'label': '五粮液 (000858)', 'value': 'sz000858'},
                {'label': '招商银行 (600036)', 'value': 'sh600036'}]
    except:
        return []


@app.callback(
    Output('price-chart', 'figure'),
    [Input('run-strategy-btn', 'n_clicks')],
    [State('stock-selector', 'value'),
     State('strategy-selector', 'value')]
)
def update_chart(n_clicks, selected_stock, selected_strategy):
    if n_clicks > 0 and selected_stock and selected_strategy:
        try:
            # Get stock data
            data = screener.fetch_stock_data(selected_stock, period='180')
            if data is None or data.empty:
                return go.Figure().update_layout(title="无法获取股票数据")
            
            # Run selected strategy
            strategy = strategy_manager.get_strategy(selected_strategy)
            if strategy is None:
                return go.Figure().update_layout(title="策略未找到")
            
            signals = strategy.generate_signals(data)
            
            # Create chart
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='价格'
            ))
            
            # Add moving averages if available
            if 'ma_10' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['ma_10'],
                    mode='lines',
                    name='MA10',
                    line=dict(color='orange', width=1)
                ))
            
            if 'ma_20' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['ma_20'],
                    mode='lines',
                    name='MA20',
                    line=dict(color='blue', width=1)
                ))
            
            # Add buy/sell signals
            buy_signals = data[signals == 1]
            sell_signals = data[signals == -1]
            
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['close'],
                    mode='markers',
                    name='买入信号',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ))
            
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['close'],
                    mode='markers',
                    name='卖出信号',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ))
            
            # Add Bollinger Bands if available
            if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['bb_upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='rgba(255,0,0,0.3)', width=1, dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['bb_lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='rgba(255,0,0,0.3)', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.1)'
                ))
            
            fig.update_layout(
                title=f"{selected_stock} - {strategy.name} 策略分析",
                xaxis_title="日期",
                yaxis_title="价格",
                hovermode='x unified',
                height=600
            )
            
            return fig
        except Exception as e:
            return go.Figure().update_layout(title=f"图表生成错误: {str(e)}")
    
    return go.Figure().update_layout(title="选择股票和策略后点击'运行策略'按钮")


@app.callback(
    Output('signal-notifications', 'children'),
    [Input('run-strategy-btn', 'n_clicks')],
    [State('stock-selector', 'value'),
     State('strategy-selector', 'value')]
)
def update_notifications(n_clicks, selected_stock, selected_strategy):
    if n_clicks > 0 and selected_stock and selected_strategy:
        try:
            # Get stock data
            data = screener.fetch_stock_data(selected_stock, period='30')  # Last 30 days
            if data is None or data.empty:
                return html.P("无法获取股票数据", style={'color': '#e74c3c'})
            
            # Run selected strategy
            strategy = strategy_manager.get_strategy(selected_strategy)
            if strategy is None:
                return html.P("策略未找到", style={'color': '#e74c3c'})
            
            signals = strategy.generate_signals(data)
            
            # Get latest signals
            recent_signals = signals.tail(5)  # Last 5 days
            
            notification_items = []
            for date, signal in recent_signals.items():
                if signal == 1:
                    notification_items.append(
                        html.Div([
                            html.Strong(f"[{date.strftime('%Y-%m-%d')}] {selected_stock}"),
                            html.Span(" → 买入信号", style={'color': '#27ae60', 'marginLeft': '10px'})
                        ], style={'padding': '5px', 'borderBottom': '1px solid #eee'})
                    )
                elif signal == -1:
                    notification_items.append(
                        html.Div([
                            html.Strong(f"[{date.strftime('%Y-%m-%d')}] {selected_stock}"),
                            html.Span(" → 卖出信号", style={'color': '#e74c3c', 'marginLeft': '10px'})
                        ], style={'padding': '5px', 'borderBottom': '1px solid #eee'})
                    )
            
            if notification_items:
                return html.Div(notification_items)
            else:
                return html.P("最近几天无交易信号", style={'color': '#7f8c8d'})
        
        except Exception as e:
            return html.P(f"生成信号通知时出错: {str(e)}", style={'color': '#e74c3c'})
    
    return html.P("运行策略后将显示信号通知", style={'color': '#7f8c8d'})


@app.callback(
    Output('strategy-output', 'children'),
    [Input('run-strategy-btn', 'n_clicks')],
    [State('stock-selector', 'value'),
     State('strategy-selector', 'value')]
)
def run_strategy_analysis(n_clicks, selected_stock, selected_strategy):
    if n_clicks > 0 and selected_stock and selected_strategy:
        try:
            # Get stock data
            data = screener.fetch_stock_data(selected_stock, period='180')
            if data is None or data.empty:
                return html.Div("无法获取股票数据", style={'color': '#e74c3c'})
            
            # Run selected strategy
            strategy = strategy_manager.get_strategy(selected_strategy)
            if strategy is None:
                return html.Div("策略未找到", style={'color': '#e74c3c'})
            
            signals = strategy.generate_signals(data)
            
            # Calculate strategy performance
            buy_signals = signals[signals == 1]
            sell_signals = signals[signals == -1]
            
            total_signals = len(buy_signals) + len(sell_signals)
            buy_count = len(buy_signals)
            sell_count = len(sell_signals)
            
            # Calculate returns based on signals
            cumulative_returns = 0
            if len(buy_signals) > 0 and len(sell_signals) > 0:
                # Simple return calculation (would be more complex in reality)
                start_price = data['close'].iloc[0]
                end_price = data['close'].iloc[-1]
                simple_return = (end_price - start_price) / start_price * 100
            
            return html.Div([
                html.H4(f"{strategy.name} 策略分析结果", style={'color': '#2980b9'}),
                html.Ul([
                    html.Li(f"总信号数: {total_signals}"),
                    html.Li(f"买入信号: {buy_count}"),
                    html.Li(f"卖出信号: {sell_count}"),
                    html.Li(f"期间收益率: {((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100):.2f}%") if len(data) > 0 else html.Li("数据不足"),
                ]),
                html.Div(f"当前状态: {strategy.name} 策略已运行", style={'marginTop': '10px', 'color': '#27ae60'})
            ])
        
        except Exception as e:
            return html.Div(f"策略分析出错: {str(e)}", style={'color': '#e74c3c'})
    
    return html.Div("选择股票和策略后点击'运行策略'按钮")


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)