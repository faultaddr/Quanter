# QuantTool Stock API Routes Phase 3 Design

> 日期: 2026-07-09
> 状态: 用户已授权继续整理

## 背景

前两轮已经把 API 总入口和 model API 拆成聚合 router，但 `quanttool/web/api/stock.py` 仍有 1400 多行。它同时包含股票分析、增强分析、K 线、筹码、技术信号、完整分析、资金流、风险、因子、交易可行性、回测对比和指数数据接口。

这次继续整理只拆 API 编排文件边界，不改 `StockAnalyzer`、筹码计算、技术信号规则、风险计算或回测策略逻辑。

## 目标

1. 保持所有股票相关 HTTP 路径、方法、查询参数和 broad response shape 不变。
2. 将 `quanttool/web/api/stock.py` 收缩为薄聚合 router。
3. 将股票 API 按职责拆到独立模块，降低单文件认知成本。
4. 增加结构测试，防止 `stock.py` 再次长成巨石。
5. 保持 smoke tests、compileall 和 frontend lint 通过。

## 非目标

- 不拆 `quanttool/factors/stock_analyzer.py`。
- 不重写增强分析、技术信号、筹码分布、资金流、风险或回测对比逻辑。
- 不改变 `/api/analyze*`、`/api/stock/{symbol}/*`、`/api/index/{index_code}/data` 路径。
- 不新增运行依赖。

## 目标结构

```text
quanttool/web/api/
├── stock.py                         # 只 include stock_routes 子 router
└── stock_routes/
    ├── __init__.py                  # 聚合股票子 router
    ├── analysis.py                  # /analyze、/analyze/enhanced、/stock/{symbol}/analysis
    ├── market_data.py               # /stock/{symbol}/info、/stock/{symbol}/kline、/index/{index_code}/data
    ├── chip_signals.py              # /stock/{symbol}/chip、/stock/{symbol}/signals
    └── insights.py                  # /flow、/risk、/factors、/feasibility、/backtest-compare
```

共享原则：

- 每个子模块只定义一个 `router = APIRouter()`。
- 子模块之间不互相导入 endpoint 函数。
- 缓存继续通过 `quanttool.web.api.utils` 使用。
- `quanttool/web/api/stock.py` 继续导出 `router`，让上层 `routes.py` 无需改动。

## 验证策略

新增结构测试：

- `quanttool/web/api/stock.py` 行数不超过 120 行。
- `quanttool/web/api/stock_routes/` 必须存在，并包含四个业务模块。
- 股票相关关键路由仍注册在 FastAPI app 中。

现有验证继续执行：

```bash
.venv-mcp/bin/python -m unittest discover -s tests -v
.venv-mcp/bin/python -m compileall -q quanttool
cd quanttool/web/frontend && npm run lint
```

## 风险控制

- 拆分采用机械搬迁，先保持函数体不改。
- 用 route introspection 对比 `/api` 路由清单，防止路径丢失或重复。
- 如果发现循环导入，只移动共享工具到 `utils.py`，不让业务 router 互相依赖。
