"""ML strategy API routes."""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
import json
import os
import queue
import threading
import time
import uuid

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...core.logging import get_logger
from .utils import get_cached_analysis, set_cached_analysis, to_python_types


logger = get_logger(__name__)
router = APIRouter()

from ..schemas.ml import MLBacktestRequest, MLMonitorRequest, MLScanRequest


_monitor_services: Dict[str, Any] = {}


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


@router.post("/ml/scan")
async def scan_with_ml_model(request: MLScanRequest) -> Dict[str, Any]:
    """
    使用 ML 模型进行智能选股

    对候选股票进行预测，返回得分最高的股票
    """
    try:
        from quanttool.strategies.gbm_strategy import GBMStrategy, GBMConfig
        from quanttool.infrastructure.data_providers.qlib_data_loader import QlibDataLoader
        from quanttool.cli.commands.analysis_commands import get_csi300_constituents
        import glob

        # 查找模型
        model_path = request.model_path
        if not model_path:
            model_files = glob.glob("models/gbm/lgbm_*.pkl")
            if not model_files:
                raise HTTPException(status_code=404, detail="未找到训练好的模型，请先训练模型")
            model_path = max(model_files, key=os.path.getmtime)
            logger.info(f"使用最新模型: {model_path}")

        # 获取候选股票
        symbols = request.symbols
        if not symbols:
            # 默认使用沪深300成分股
            csi300 = get_csi300_constituents()
            symbols = [s['code'] if isinstance(s, dict) else s for s in csi300]

        # 加载模型
        config = GBMConfig()
        strategy = GBMStrategy(config)
        strategy.load_model(model_path)

        # 初始化数据加载器
        data_loader = QlibDataLoader()
        if not data_loader.init_qlib():
            raise HTTPException(status_code=500, detail="Qlib 初始化失败")

        # 预测所有股票
        results = []
        for symbol in symbols:
            try:
                pred = strategy.predict(symbol)
                if pred.get('probability', 0) >= request.min_probability:
                    results.append({
                        'symbol': symbol,
                        'probability': pred['probability'],
                        'pred_return': pred.get('return_pred', 0),
                        'signal': pred.get('signal', 'hold'),
                        'close': pred.get('close', 0),
                    })
            except Exception as e:
                logger.debug(f"预测失败 {symbol}: {e}")
                continue

        # 按概率排序
        results.sort(key=lambda x: x['probability'], reverse=True)
        top_results = results[:request.top_n]

        return {
            "success": True,
            "model_path": model_path,
            "total_scanned": len(symbols),
            "qualified_count": len(results),
            "min_probability": request.min_probability,
            "results": to_python_types(top_results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ML 选股失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"选股失败: {str(e)}")


@router.post("/ml/monitor/start")
async def start_ml_monitor(request: MLMonitorRequest) -> Dict[str, Any]:
    """
    启动 ML 模型实时监控

    定时对指定股票进行预测并生成信号
    """
    try:
        from quanttool.strategies.gbm_strategy import GBMStrategy, GBMConfig
        import glob

        # 查找模型
        model_path = request.model_path
        if not model_path:
            model_files = glob.glob("models/gbm/lgbm_*.pkl")
            if not model_files:
                raise HTTPException(status_code=404, detail="未找到训练好的模型")
            model_path = max(model_files, key=os.path.getmtime)

        # 加载模型
        config = GBMConfig()
        strategy = GBMStrategy(config)
        strategy.load_model(model_path)

        monitor_id = str(uuid.uuid4())[:8]

        # 存储监控信息
        _monitor_services[monitor_id] = {
            "service": strategy,
            "model_path": model_path,
            "symbols": request.symbols,
            "signals": [],
            "started_at": datetime.now(),
            "task": None,
        }

        async def run_ml_monitor():
            while True:
                try:
                    for symbol in request.symbols:
                        try:
                            pred = strategy.predict(symbol)
                            signal = {
                                "symbol": symbol,
                                "probability": pred.get('probability', 0),
                                "signal": pred.get('signal', 'hold'),
                                "timestamp": datetime.now().isoformat(),
                            }
                            _monitor_services[monitor_id]["signals"].insert(0, signal)
                            # 保留最近100条信号
                            _monitor_services[monitor_id]["signals"] = _monitor_services[monitor_id]["signals"][:100]
                        except Exception as e:
                            logger.debug(f"监控预测失败 {symbol}: {e}")

                    await asyncio.sleep(request.interval_seconds)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"ML 监控错误: {e}")
                    await asyncio.sleep(5)

        task = asyncio.create_task(run_ml_monitor())
        _monitor_services[monitor_id]["task"] = task

        return {
            "monitor_id": monitor_id,
            "model_path": model_path,
            "symbols": request.symbols,
            "interval_seconds": request.interval_seconds,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动 ML 监控失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动失败: {str(e)}")


@router.get("/ml/monitor/{monitor_id}/signals")
async def get_ml_monitor_signals(monitor_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """获取 ML 监控信号"""
    if monitor_id not in _monitor_services:
        raise HTTPException(status_code=404, detail=f"监控 {monitor_id} 不存在")

    monitor = _monitor_services[monitor_id]
    signals = monitor.get("signals", [])[:limit]
    return to_python_types(signals)
