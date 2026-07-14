# QuantTool API Model Routes Phase 2 Design

> 日期: 2026-07-09
> 状态: 用户已授权继续整理

## 背景

上一轮整理已经把 `quanttool/web/api/routes.py` 拆成聚合入口和多个业务 router，但 `quanttool/web/api/models.py` 仍有 2000 多行。它同时承载 Qlib 模型发现、GBM 训练与预测、Qlib 训练、SSE 训练流、Qlib 预测回测和模型分类，仍然是 API 层最大的维护热点之一。

这次继续整理只处理 API 编排边界，不改训练算法、特征工程、回测交易规则或模型保存格式。

## 目标

1. 保持所有 `/api/gbm/*` 和 `/api/qlib/*` HTTP 路径、方法和请求 schema 不变。
2. 将 `quanttool/web/api/models.py` 收缩为薄聚合 router。
3. 将 GBM、Qlib 模型发现、Qlib 训练、Qlib 预测拆到独立模块。
4. 增加结构测试，防止聚合文件再次长成巨石。
5. 保持当前 smoke test、compileall 和 frontend lint 通过。

## 非目标

- 不拆 `quanttool/factors/stock_analyzer.py`。
- 不重写 Qlib/GBM 训练逻辑。
- 不修改模型文件目录、文件名格式或响应字段。
- 不引入新的运行依赖。

## 目标结构

```text
quanttool/web/api/
├── models.py                         # 只 include model_routes 子 router
└── model_routes/
    ├── __init__.py                   # 聚合 GBM/Qlib 子 router
    ├── discovery.py                  # /qlib/models、saved/pretrained/all/categories/detail
    ├── gbm.py                        # /gbm/train、predict、models、delete、progress、qrun-models、picks
    ├── qlib_training.py              # /qlib/train、/qlib/train/stream
    └── qlib_prediction.py            # /qlib/predict
```

共享原则：

- 每个子模块只定义一个 `router = APIRouter()`。
- 子模块之间不互相导入 endpoint 函数。
- 共享 logger 和常量在本地模块内保留，避免为了拆分制造新的全局依赖。
- `quanttool/web/api/models.py` 继续导出 `router`，让上层 `routes.py` 无需改动。

## 验证策略

新增结构测试：

- `quanttool/web/api/models.py` 行数不超过 120 行。
- `quanttool/web/api/model_routes/` 必须存在，并包含四个业务模块。
- `/api/gbm/*` 和 `/api/qlib/*` 关键路由仍注册在 FastAPI app 中。

现有验证继续执行：

```bash
.venv-mcp/bin/python -m unittest discover -s tests -v
.venv-mcp/bin/python -m compileall -q quanttool
cd quanttool/web/frontend && npm run lint
```

## 风险控制

- 拆分采用机械搬迁，先保持函数体不改。
- 用 route introspection 对比关键模型端点，防止路径丢失。
- 如果发现循环导入，优先移动共享函数到新模块，不让业务 router 互相依赖。
