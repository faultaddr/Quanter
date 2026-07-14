# QuantTool Qlib Training Routes Phase 4 Design

> 日期: 2026-07-09
> 状态: 用户已授权继续整理

## 背景

API 总入口、model API 和 stock API 已经完成拆分。当前 API 层剩下的最大热点是 `quanttool/web/api/model_routes/qlib_training.py`，它有接近 1000 行，同时包含普通 Qlib 训练接口和 SSE 流式训练接口。

这次继续整理只拆 Qlib 训练 API 文件边界，不改数据收集、特征工程、Qlib 原生训练、sklearn fallback、评估指标、SSE 事件内容或模型保存格式。

## 目标

1. 保持 `/api/qlib/train` 和 `/api/qlib/train/stream` HTTP 路径、方法、请求 schema 和 broad response shape 不变。
2. 将 `quanttool/web/api/model_routes/qlib_training.py` 收缩为薄聚合 router。
3. 将同步训练和流式训练拆到独立模块。
4. 增加结构测试，防止 Qlib 训练聚合文件再次变大。
5. 保持 smoke tests、compileall 和 frontend lint 通过。

## 非目标

- 不重写 Qlib 训练算法。
- 不修改 `QlibTrainRequest`。
- 不改变 SSE event 类型、字段或结束条件。
- 不新增运行依赖。

## 目标结构

```text
quanttool/web/api/model_routes/
├── qlib_training.py                  # 只 include qlib_training_routes 子 router
└── qlib_training_routes/
    ├── __init__.py                   # 聚合 Qlib 训练子 router
    ├── batch.py                      # /qlib/train
    └── stream.py                     # /qlib/train/stream
```

共享原则：

- 每个子模块只定义一个 `router = APIRouter()`。
- `batch.py` 和 `stream.py` 不互相导入 endpoint 函数。
- 先做机械搬迁，重复 import 可以后续再收敛。
- `quanttool/web/api/model_routes/qlib_training.py` 继续导出 `router`，让 `model_routes/__init__.py` 无需改动。

## 验证策略

新增结构测试：

- `quanttool/web/api/model_routes/qlib_training.py` 行数不超过 120 行。
- `quanttool/web/api/model_routes/qlib_training_routes/` 必须存在，并包含 `batch.py` 和 `stream.py`。
- `/api/qlib/train` 和 `/api/qlib/train/stream` 仍注册在 FastAPI app 中。

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
