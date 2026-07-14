"""Single backtest execution API routes."""

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from ...schemas.backtest import BacktestRequest


router = APIRouter()

@router.get("/backtest/history")


@router.post("/backtest/run")
async def run_backtest(request: BacktestRequest) -> Dict[str, Any]:
    """
    运行回测

    支持的策略：
    - ma_cross: 均线交叉
    - breakout: 突破策略
    - trend_momentum: 趋势动量
    - adaptive_threshold: 自适应阈值

    数据提供者：
    - enhanced_data_fetcher: 直接从网络获取
    - incremental_data_fetcher: 优先使用缓存，增量拉取（推荐）
    """
    try:
        from quanttool.application.backtest_service import BacktestService

        # 解析日期（使用动态默认值：最近一年）
        start_date = datetime.fromisoformat(request.get_start_date())
        end_date = datetime.fromisoformat(request.get_end_date())

        # 初始化回测服务（使用实时价格数据，避免复权价格显示异常）
        backtest_service = BacktestService(use_qlib=True, use_realtime_price=True)

        # 运行回测
        result = backtest_service.run_backtest(
            strategy_name=request.strategy_name,
            strategy_params=request.strategy_params,
            symbols=request.symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe="1d",
            initial_cash=request.initial_cash,
            commission_rate=request.commission_rate,
        )

        # 转换结果
        result_dict = {
            "strategy": request.strategy_name,
            "symbols": request.symbols,
            "start_date": result.start_date.isoformat(),
            "end_date": result.end_date.isoformat(),
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "excess_return": result.annual_return - 0.05 if result.annual_return else 0,  # 假设基准5%
            "max_drawdown": getattr(result, 'max_drawdown', 0),
            "sharpe_ratio": getattr(result, 'sharpe_ratio', 0),
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "profit_factor": getattr(result, 'profit_factor', 0),
            "trades": []
        }

        # 添加交易记录（如果有）
        if hasattr(result, 'trades') and result.trades:
            for trade in result.trades[:50]:  # 限制返回数量
                # Trade 对象使用 side 字段 (OrderSide 枚举)
                side = getattr(trade, 'side', None)
                if side is None:
                    side = getattr(trade, 'action', 'sell')
                action = side.value if hasattr(side, 'value') else str(side)

                result_dict["trades"].append({
                    "strategy": request.strategy_name,
                    "symbol": getattr(trade, 'symbol', ''),
                    "action": action,
                    "type": action,
                    "price": getattr(trade, 'price', 0),
                    "shares": getattr(trade, 'quantity', getattr(trade, 'shares', 0)),
                    "timestamp": str(getattr(trade, 'timestamp', '')),
                    "profit": getattr(trade, 'pnl', getattr(trade, 'profit', None)),
                })

        return result_dict

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"回测失败: {str(e)}")

