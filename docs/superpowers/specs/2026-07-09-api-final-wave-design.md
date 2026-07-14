# QuantTool API Final Wave Design

> 日期: 2026-07-09
> 状态: 用户要求完成本轮重构

## 背景

本轮整理已经完成 API 总入口、model API、stock API 和 Qlib training API 的拆分。API 编排层剩余的多 endpoint 热点主要是：

- `quanttool/web/api/backtest.py`
- `quanttool/web/api/ml.py`

这两个文件仍然把多个 endpoint 放在同一个模块里，但都属于入口编排代码，适合继续机械拆分。相比之下，`qlib_prediction.py` 和 `qlib_training_routes/stream.py` 虽然仍较长，但当前各自承载单一 endpoint 的预测回测和 SSE 流式训练闭包，继续拆会开始改变内部语义边界，暂不纳入本轮完成线。

## 完成边界

本轮“全部完成”定义为：

1. API route 聚合层全部拆到清晰目录边界。
2. 多 endpoint API 文件变成薄聚合 router。
3. `/api` 路由清单保持 70 条且和重构前快照一致。
4. smoke tests、compileall、frontend lint 全部通过。
5. 工作区只剩用户自有未跟踪文件，不混入提交。

## 非目标

- 不拆 `quanttool/factors/stock_analyzer.py`、`scoring_system.py` 等算法内核巨石。
- 不拆 `BacktestService`、`GBMStrategy`、`Qlib` 模型实现。
- 不改变回测、ML 选股、ML 监控、SSE 输出或实验查询的业务逻辑。
- 不新增运行依赖。

## 目标结构

```text
quanttool/web/api/
├── backtest.py                       # 只 include backtest_routes 子 router
├── backtest_routes/
│   ├── __init__.py                   # 聚合回测子 router
│   ├── catalog.py                    # /backtest/strategies
│   ├── execution.py                  # /backtest/history、/backtest/run
│   ├── comparison.py                 # /backtest/run-all
│   ├── stream.py                     # /backtest/run-all-stream
│   └── experiments.py                # /experiments、/backtest/runs/{run_id}
├── ml.py                             # 只 include ml_routes 子 router
└── ml_routes/
    ├── __init__.py                   # 聚合 ML 子 router
    ├── backtest.py                   # /ml/backtest
    ├── scan.py                       # /ml/scan
    └── monitor.py                    # /ml/monitor/start、/ml/monitor/{monitor_id}/signals
```

共享原则：

- 每个子模块只定义一个 `router = APIRouter()`。
- 子模块之间不互相导入 endpoint 函数。
- ML monitor 的 `_monitor_services` 放在 `ml_routes/monitor.py` 内部，保持监控状态和对应 endpoint 同域。
- `quanttool/web/api/backtest.py` 和 `quanttool/web/api/ml.py` 继续导出 `router`，让上层 `routes.py` 无需改动。

## 验证策略

新增结构测试：

- `quanttool/web/api/backtest.py` 行数不超过 120 行。
- `quanttool/web/api/ml.py` 行数不超过 120 行。
- `backtest_routes/` 和 `ml_routes/` 必须包含目标模块。
- 回测和 ML 关键路由仍注册在 FastAPI app 中。

现有验证继续执行：

```bash
.venv-mcp/bin/python -m unittest discover -s tests -v
.venv-mcp/bin/python -m compileall -q quanttool
cd quanttool/web/frontend && npm run lint
```

## 风险控制

- 拆分采用机械搬迁，函数体尽量原样保留。
- 用 route introspection 对比 `/api` 路由清单，防止路径丢失或重复。
- 如出现 import 问题，只修相对路径和局部 import，不改业务逻辑。
