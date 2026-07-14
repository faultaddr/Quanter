"""Backtest strategy catalog API route."""

from typing import Any, Dict, List

from fastapi import APIRouter


router = APIRouter()

@router.get("/backtest/strategies")
async def list_backtest_strategies() -> List[Dict[str, Any]]:
    """列出可用的回测策略"""
    return [
        {
            "name": "ma_cross",
            "display_name": "均线交叉策略",
            "description": "短期均线上穿长期均线买入，下穿卖出",
            "category": "traditional",
            "params": {
                "short_window": {"type": "int", "default": 10, "description": "短期均线周期"},
                "long_window": {"type": "int", "default": 30, "description": "长期均线周期"}
            }
        },
        {
            "name": "breakout",
            "display_name": "突破策略",
            "description": "价格突破N日高点买入，跌破N日低点卖出",
            "category": "traditional",
            "params": {
                "period": {"type": "int", "default": 20, "description": "突破周期"}
            }
        },
        {
            "name": "score",
            "display_name": "评分策略",
            "description": "首次突破策略：评分首次突破阈值时买入/卖出。买入=80,卖出=60为最优参数",
            "category": "traditional",
            "params": {
                "buy_threshold": {"type": "float", "default": 80.0, "description": "买入评分阈值（首次突破触发）"},
                "sell_threshold": {"type": "float", "default": 60.0, "description": "卖出评分阈值（首次跌破触发）"}
            }
        },
        {
            "name": "enhanced_score",
            "display_name": "增强评分策略",
            "description": "首次突破+动态权重+风险控制。评分首次突破80买入，首次跌破60卖出",
            "category": "traditional",
            "params": {
                "buy_threshold": {"type": "float", "default": 80.0, "description": "买入评分阈值"},
                "sell_threshold": {"type": "float", "default": 60.0, "description": "卖出评分阈值"},
                "use_dynamic_weights": {"type": "bool", "default": True, "description": "使用动态权重"},
                "use_risk_control": {"type": "bool", "default": True, "description": "使用风险控制"}
            }
        },
        {
            "name": "dual_ma",
            "display_name": "双均线策略",
            "description": "经典双均线交叉策略，支持多周期组合",
            "category": "traditional",
            "params": {
                "fast_period": {"type": "int", "default": 5, "description": "快线周期"},
                "slow_period": {"type": "int", "default": 20, "description": "慢线周期"}
            }
        },
        {
            "name": "macd",
            "display_name": "MACD策略",
            "description": "基于MACD指标的金叉死叉信号",
            "category": "traditional",
            "params": {
                "fast_period": {"type": "int", "default": 12, "description": "快线周期"},
                "slow_period": {"type": "int", "default": 26, "description": "慢线周期"},
                "signal_period": {"type": "int", "default": 9, "description": "信号线周期"}
            }
        },
        {
            "name": "rsi",
            "display_name": "RSI策略",
            "description": "基于RSI超买超卖信号",
            "category": "traditional",
            "params": {
                "period": {"type": "int", "default": 14, "description": "RSI周期"},
                "oversold": {"type": "int", "default": 30, "description": "超卖阈值"},
                "overbought": {"type": "int", "default": 70, "description": "超买阈值"}
            }
        },
        {
            "name": "kdj",
            "display_name": "KDJ策略",
            "description": "基于KDJ指标的金叉死叉信号",
            "category": "traditional",
            "params": {
                "n": {"type": "int", "default": 9, "description": "KDJ周期"},
                "m1": {"type": "int", "default": 3, "description": "K平滑周期"},
                "m2": {"type": "int", "default": 3, "description": "D平滑周期"}
            }
        },
        {
            "name": "bollinger",
            "display_name": "布林带策略",
            "description": "基于布林带上下轨突破信号",
            "category": "traditional",
            "params": {
                "period": {"type": "int", "default": 20, "description": "布林带周期"},
                "std_dev": {"type": "float", "default": 2.0, "description": "标准差倍数"}
            }
        },
        {
            "name": "turtle",
            "display_name": "海龟交易策略",
            "description": "经典海龟交易系统，基于通道突破",
            "category": "traditional",
            "params": {
                "entry_period": {"type": "int", "default": 20, "description": "入场周期"},
                "exit_period": {"type": "int", "default": 10, "description": "出场周期"}
            }
        },
        {
            "name": "gbm",
            "display_name": "GBM机器学习策略",
            "description": "基于LightGBM的机器学习策略，使用Alpha158特征和百分位排名信号",
            "category": "ml",
            "params": {
                "buy_threshold": {"type": "float", "default": 0.35, "description": "买入百分位阈值（前65%触发买入）"},
                "sell_threshold": {"type": "float", "default": 0.35, "description": "卖出百分位阈值（后35%触发卖出）"},
                "stop_loss_pct": {"type": "float", "default": 0.05, "description": "止损比例"},
                "take_profit_pct": {"type": "float", "default": 0.10, "description": "止盈比例"}
            }
        }
    ]


