"""
QuantTool MCP Server

This module implements an MCP (Model Context Protocol) server that exposes
QuantTool's stock analysis, backtesting, and ML model capabilities as tools
for Claude Code to call.

Usage:
    python -m quanttool.agent.server

Configuration for Claude Code:
    Add to ~/.claude/settings.json or project .claude/settings.json:
    {
        "mcpServers": {
            "quanttool": {
                "command": "python",
                "args": ["-m", "quanttool.agent.server"],
                "cwd": "/path/to/Quanter"
            }
        }
    }
"""

import asyncio
import json
import sys
import os
from typing import Any, Dict, List

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: MCP SDK not installed. Run: pip install mcp", file=sys.stderr)

# Import tools
from .tools import (
    analyze_stock,
    get_stock_score,
    run_backtest,
    run_qlib_backtest,
    screen_stocks,
)
from .schemas.tools import (
    AnalyzeStockInput,
    GetStockScoreInput,
    RunBacktestInput,
    QlibBacktestInput,
    ScreenStocksInput,
)


# Tool definitions for MCP
TOOL_DEFINITIONS = [
    {
        "name": "analyze_stock",
        "description": """对A股进行综合技术分析，包括技术指标、评分、买卖建议。

功能：
- 获取股票历史数据
- 计算多种技术指标（MACD、KDJ、RSI、布林带等）
- 生成综合评分（0-100分）
- 提供买卖操作建议

适用场景：
- 分析单只股票的技术面
- 查看股票当前趋势状态
- 获取买卖时机建议""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "股票代码，如 '600519'（茅台）或 '000001.SZ'（平安银行）"
                },
                "days": {
                    "type": "integer",
                    "default": 360,
                    "minimum": 30,
                    "maximum": 720,
                    "description": "分析天数，默认360天"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_stock_score",
        "description": """获取股票的多维度评分。

评分维度：
- 趋势因子：均线系统、DMI、MACD
- 动能因子：MTM、ROC、KDJ、RSI
- 资金因子：OBV、MFI、成交量

返回：
- 综合评分（0-100）
- 各维度分项得分
- 操作建议""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "股票代码"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "run_backtest",
        "description": """使用指定策略进行历史回测。

支持策略：
- ma_cross: 均线交叉策略
- dual_ma: 双均线策略
- rsi: RSI超买超卖策略
- macd: MACD策略
- bollinger: 布林带策略
- kdj: KDJ策略
- turtle: 海龟交易策略

返回：
- 总收益率
- 年化收益
- 最大回撤
- 胜率
- 交易记录""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "股票代码列表"
                },
                "strategy": {
                    "type": "string",
                    "enum": ["ma_cross", "dual_ma", "rsi", "macd", "bollinger", "kdj", "turtle"],
                    "default": "ma_cross",
                    "description": "交易策略"
                },
                "start_date": {
                    "type": "string",
                    "format": "date",
                    "description": "开始日期 (YYYY-MM-DD)"
                },
                "end_date": {
                    "type": "string",
                    "format": "date",
                    "description": "结束日期 (YYYY-MM-DD)"
                },
                "initial_cash": {
                    "type": "number",
                    "default": 100000,
                    "description": "初始资金"
                },
                "commission_rate": {
                    "type": "number",
                    "default": 0.0003,
                    "description": "手续费率"
                }
            },
            "required": ["symbols"]
        }
    },
    {
        "name": "qlib_backtest",
        "description": """使用Qlib机器学习模型进行回测（支持23种模型）。

支持模型：
梯度提升类：lgb, xgboost, catboost, gbdt
神经网络类：mlp, gru, lstm, transformer, gats
表格学习类：tabnet, tabtransformer, deepfm
线性模型类：linear, ridge, lasso, elastic_net
集成学习类：random_forest, extra_trees, adaboost

特点：
- 基于历史数据训练ML模型
- 预测股票未来收益
- 自动选股和调仓""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "股票代码列表（训练和测试用）"
                },
                "model": {
                    "type": "string",
                    "enum": [
                        "lgb", "xgboost", "catboost", "mlp", "gru", "lstm",
                        "gats", "tabnet", "transformer", "double_gru", "double_lstm",
                        "linear", "ridge", "lasso", "elastic_net", "svr",
                        "random_forest", "extra_trees", "adaboost", "gbdt",
                        "tabnet2", "tabtransformer", "deepfm"
                    ],
                    "default": "lgb",
                    "description": "ML模型选择"
                },
                "days": {
                    "type": "integer",
                    "default": 180,
                    "minimum": 60,
                    "maximum": 720,
                    "description": "训练数据天数"
                },
                "epochs": {
                    "type": "integer",
                    "default": 50,
                    "description": "神经网络训练轮数"
                },
                "initial_cash": {
                    "type": "number",
                    "default": 100000,
                    "description": "初始资金"
                }
            },
            "required": ["symbols"]
        }
    },
    {
        "name": "screen_stocks",
        "description": """根据技术指标条件筛选股票。

支持的指数：
- hs300: 沪深300
- zz500: 中证500
- sz50: 上证50
- all: 全市场

筛选功能：
- 按评分筛选
- 按技术指标筛选
- 返回排名靠前的股票""",
        "inputSchema": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "string",
                    "enum": ["hs300", "zz500", "sz50", "all"],
                    "default": "hs300",
                    "description": "筛选范围"
                },
                "min_score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "最低评分阈值"
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "返回数量上限"
                }
            },
            "required": []
        }
    },
]


def create_server() -> 'Server':
    """Create and configure the MCP server."""
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP SDK not installed. Run: pip install mcp")

    server = Server("quanttool")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """Return list of available tools."""
        return [
            Tool(
                name=tool["name"],
                description=tool["description"],
                inputSchema=tool["inputSchema"],
            )
            for tool in TOOL_DEFINITIONS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Execute a tool and return results."""
        try:
            result = await _execute_tool(name, arguments)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        except Exception as e:
            error_result = {"error": str(e), "tool": name, "arguments": arguments}
            return [TextContent(type="text", text=json.dumps(error_result, ensure_ascii=False, indent=2))]

    return server


async def _execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool based on name and arguments."""

    if name == "analyze_stock":
        input_data = AnalyzeStockInput(**arguments)
        result = analyze_stock(input_data)
        return result.model_dump()

    elif name == "get_stock_score":
        input_data = GetStockScoreInput(**arguments)
        result = get_stock_score(input_data)
        return result.model_dump()

    elif name == "run_backtest":
        input_data = RunBacktestInput(**arguments)
        result = run_backtest(input_data)
        return result.model_dump()

    elif name == "qlib_backtest":
        input_data = QlibBacktestInput(**arguments)
        result = run_qlib_backtest(input_data)
        return result.model_dump()

    elif name == "screen_stocks":
        input_data = ScreenStocksInput(**arguments)
        result = screen_stocks(input_data)
        return result.model_dump()

    else:
        raise ValueError(f"Unknown tool: {name}")


async def run_server():
    """Run the MCP server."""
    if not MCP_AVAILABLE:
        print("Error: MCP SDK not installed.", file=sys.stderr)
        print("Please install it with: pip install mcp", file=sys.stderr)
        sys.exit(1)

    server = create_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main():
    """Main entry point."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
