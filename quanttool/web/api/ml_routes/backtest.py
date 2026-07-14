"""ML backtest API route."""

from datetime import datetime, timedelta
import os
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from ....core.logging import get_logger
from ...schemas.ml import MLBacktestRequest
from ..utils import to_python_types


logger = get_logger(__name__)
router = APIRouter()

@router.post("/ml/backtest")
async def run_ml_backtest(request: MLBacktestRequest) -> Dict[str, Any]:
    """
    使用 ML 模型进行回测

    使用训练好的 GBM 模型对指定股票进行回测
    """
    try:
        from quanttool.strategies.gbm_strategy import GBMStrategy, GBMConfig
        from quanttool.infrastructure.data_providers.qlib_data_loader import QlibDataLoader
        import glob

        # 查找模型
        model_path = request.model_path
        if not model_path:
            model_files = glob.glob("models/gbm/lgbm_*.pkl")
            if not model_files:
                raise HTTPException(status_code=404, detail="未找到训练好的模型，请先训练模型")
            model_path = max(model_files, key=os.path.getmtime)
            logger.info(f"使用最新模型: {model_path}")

        # 解析日期
        end_date = datetime.now() if not request.end_date else datetime.fromisoformat(request.end_date)
        start_date = end_date - timedelta(days=365) if not request.start_date else datetime.fromisoformat(request.start_date)

        # 加载模型
        config = GBMConfig(
            buy_threshold=request.buy_threshold,
            sell_threshold=request.sell_threshold,
        )
        strategy = GBMStrategy(config)
        strategy.load_model(model_path)

        # 初始化数据加载器
        data_loader = QlibDataLoader()
        if not data_loader.init_qlib():
            raise HTTPException(status_code=500, detail="Qlib 初始化失败")

        # 回测逻辑
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # 加载所有股票数据
        # 注意：使用 StockAnalyzer 获取真实价格数据，而非 qlib 数据
        from quanttool.factors.stock_analyzer import StockAnalyzer
        stock_analyzer = StockAnalyzer(use_realtime_price=True)

        all_data = {}
        for symbol in request.symbols:
            df = stock_analyzer.get_stock_data(symbol, days=365)
            if df.empty:
                # 回退到 qlib
                df = data_loader.load_stock_data(symbol, start_str, end_str, use_adjclose=False)
            if not df.empty:
                df = df.reset_index()
                if 'date' in df.columns:
                    df = df.rename(columns={'date': 'timestamp'})
                all_data[symbol] = df

        if not all_data:
            raise HTTPException(status_code=400, detail="没有获取到任何数据")

        # 模拟回测
        cash = request.initial_cash
        position = {}  # 持仓 {symbol: shares}
        trades = []
        portfolio_values = []

        # 获取所有交易日
        all_dates = set()
        for symbol, df in all_data.items():
            for t in df['timestamp']:
                all_dates.add(t)
        sorted_dates = sorted(all_dates)

        for current_date in sorted_dates:
            # 计算当前组合价值
            position_value = 0
            for symbol, shares in position.items():
                if symbol in all_data:
                    df = all_data[symbol]
                    row = df[df['timestamp'] == current_date]
                    if not row.empty:
                        position_value += row['close'].values[0] * shares

            portfolio_value = cash + position_value
            portfolio_values.append({
                'date': current_date,
                'value': portfolio_value
            })

            # 对每只股票生成信号
            for symbol in request.symbols:
                if symbol not in all_data:
                    continue

                df = all_data[symbol]
                historical = df[df['timestamp'] <= current_date]

                if len(historical) < 120:  # 需要足够的历史数据
                    continue

                current_bar = historical.iloc[-1]

                try:
                    signal = strategy.get_signal(current_bar, historical)
                except Exception as e:
                    continue

                # 执行交易
                close = current_bar['close']
                signal_type = signal.get('signal', 'hold')

                if signal_type == 'buy' and symbol not in position:
                    # 买入
                    shares = int(cash * 0.2 / close)  # 每次20%仓位
                    if shares > 0:
                        cost = shares * close * (1 + request.commission_rate)
                        if cost <= cash:
                            cash -= cost
                            position[symbol] = shares
                            trades.append({
                                'symbol': symbol,
                                'action': 'buy',
                                'price': close,
                                'shares': shares,
                                'timestamp': current_date,
                                'reason': f"ML预测概率: {signal.get('probability', 0):.2%}"
                            })

                elif signal_type == 'sell' and symbol in position:
                    # 卖出
                    shares = position[symbol]
                    revenue = shares * close * (1 - request.commission_rate)
                    cash += revenue
                    del position[symbol]
                    trades.append({
                        'symbol': symbol,
                        'action': 'sell',
                        'price': close,
                        'shares': shares,
                        'timestamp': current_date,
                        'reason': f"ML预测概率: {signal.get('probability', 0):.2%}"
                    })

        # 最终价值
        final_position_value = 0
        for symbol, shares in position.items():
            if symbol in all_data:
                df = all_data[symbol]
                if not df.empty:
                    final_position_value += df['close'].iloc[-1] * shares

        final_value = cash + final_position_value
        total_return = (final_value - request.initial_cash) / request.initial_cash

        # 计算最大回撤
        values = [p['value'] for p in portfolio_values]
        max_drawdown = 0
        peak = values[0] if values else 0
        for v in values:
            if v > peak:
                peak = v
            drawdown = (peak - v) / peak if peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # 计算年化收益
        days = (end_date - start_date).days if isinstance(end_date, datetime) else 365
        annual_return = total_return * (365 / max(days, 1)) if total_return else 0

        # 计算胜率：盈利的卖出次数 / 总卖出次数
        sell_trades = [t for t in trades if t['action'] == 'sell']
        # 计算每笔卖出的盈亏
        buy_prices = {}
        for t in trades:
            if t['action'] == 'buy':
                buy_prices[t['symbol']] = t['price']
            elif t['action'] == 'sell' and t['symbol'] in buy_prices:
                t['profit'] = (t['price'] - buy_prices[t['symbol']]) * t['shares']

        win_count = sum(1 for t in sell_trades if t.get('profit', 0) > 0)
        win_rate = win_count / max(len(sell_trades), 1)

        return to_python_types({
            "success": True,
            "strategy": "ML-GBM",
            "model_path": model_path,
            "symbols": request.symbols,
            "start_date": start_str,
            "end_date": end_str,
            "initial_capital": request.initial_cash,
            "final_capital": final_value,
            "total_return": total_return,
            "annual_return": annual_return,
            "excess_return": annual_return - 0.05,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": annual_return / max(0.15, max_drawdown) if max_drawdown > 0 else 0,
            "total_trades": len(trades),
            "win_rate": win_rate,
            "trades": trades[-50:],  # 最近50笔交易
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ML 回测失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"回测失败: {str(e)}")
