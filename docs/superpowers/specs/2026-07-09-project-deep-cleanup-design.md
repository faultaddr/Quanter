# QuantTool 项目深度整理设计文档

> 日期: 2026-07-09
> 状态: 待实施

## 1. 背景

QuantTool 已完成一轮底层模块迁移，数据提供者、评分系统、报告系统已经按目录初步分层。但当前项目仍然显得混乱，主要原因不在底层算法本身，而在入口层和工程边界没有收束：

- `quanttool/web/api/routes.py` 集中承载任务、股票分析、扫描、回测、GBM、Qlib、实时行情、监控和 ML 接口，共 5903 行，维护和测试成本高。
- API 请求模型一部分定义在 `routes.py`，一部分定义在 `quanttool/web/schemas/`，schema 边界不清。
- 上轮重构后仍有残留 import，例如 `analysis_service.py` 引用已不存在的 `incremental_data_manager`。
- `pyproject.toml` 只声明 `packages = ["quanttool"]`，正式打包时不会完整包含子包。
- `pyproject.toml` 与 `requirements.txt` 依赖清单不一致，入口运行依赖和可选依赖边界不清。
- `.DS_Store`、`tsconfig.tsbuildinfo`、嵌套的 `quanttool/cli/quanttool/config/default.yaml` 和 `quanttool/web/frontend/quanttool/config/default.yaml` 等生成物或误拷贝文件被跟踪。
- README 和实际目录结构存在偏差，文档降低了新维护者判断力。
- 仓库没有可执行的 Python 测试基线，当前只能依赖 import 检查和前端 lint。

## 2. 目标

本次整理的目标是稳定项目的工程边界，让后续功能开发、算法拆分和性能优化有可依赖的结构。

1. 保持现有 HTTP API 路径和响应格式兼容，前端调用不需要改造。
2. 将巨型 API 文件拆成按业务域组织的 router 模块。
3. 将 API 请求/响应模型迁移到 `quanttool/web/schemas/`，集中管理 schema。
4. 修复明确的重构残留 import 和工程配置问题。
5. 清理确认无引用的误入版本库文件和生成物。
6. 建立最小 smoke test 基线，覆盖 CLI、FastAPI app、关键 service import 和 router 注册。
7. 更新 README/文档中的目录说明和启动验证说明。

## 3. 非目标

本次不拆分以下内部算法大文件，避免一次性改变业务逻辑：

- `quanttool/factors/stock_analyzer.py`
- `quanttool/infrastructure/data_providers/historical/enhanced_fetcher.py`
- `quanttool/factors/scoring_system.py`
- `quanttool/strategies/qlib_strategy.py`

这些文件的拆分应在 API 边界稳定后另起小步重构，每次配套专门测试。

本次也不删除仍被公开导入的 legacy 类。对仍有兼容价值的 legacy 入口，只补明确导出和 deprecation 说明；只删除确认无引用、无运行价值的误拷贝或生成物。

## 4. 目标结构

### 4.1 API 路由结构

保留一个总入口 `quanttool/web/api/routes.py`，但让它只负责聚合子 router。

目标文件：

```text
quanttool/web/api/
├── __init__.py
├── routes.py                  # 聚合 router，保持 app.py include 方式不变
├── utils.py                   # to_python_types、轻量缓存、共享转换函数
├── dependencies.py            # provider/service lazy factory 和熔断状态
├── tasks.py                   # /tasks、异步任务入口
├── stock.py                   # /stock/{symbol}/analysis、kline、signals、chip、flow、risk
├── scan.py                    # /scan、/scan/market
├── backtest.py                # /backtest/run、history、runs、strategies
├── models.py                  # /gbm/train、predict、models；/qlib/train、predict、models
├── realtime.py                # /realtime/quote、batch、kline、search
├── monitor.py                 # /monitor/start、stop、status、list、signals
├── ml.py                      # /ml/backtest、scan、monitor/start、monitor/signals
├── factors.py                 # /factors/mine、validate、optimize
├── risk.py                    # /risk/portfolio/check
└── registry.py                # /data/providers、/strategies、/factors 列表
```

拆分原则：

- 每个模块只定义一个 `router = APIRouter()`。
- `routes.py` 只 include 子 router，不包含业务处理函数。
- URL 路径保持现状，避免前端同步改造。
- 共享工具从子模块导入 `utils.py` 或 `dependencies.py`，禁止子 router 互相导入业务函数。

### 4.2 Schema 结构

目标文件：

```text
quanttool/web/schemas/
├── __init__.py
├── common.py                  # 通用响应、序列化辅助类型
├── tasks.py                   # TaskCreateRequest 等
├── stock.py                   # AnalyzeRequest、EnhancedAnalyzeRequest 等
├── scan.py                    # ScanRequest
├── backtest.py                # BacktestRequest 和回测响应
├── model.py                   # GBM/Qlib 请求模型
├── realtime.py                # 实时行情请求模型
├── monitor.py                 # 监控请求模型
├── ml.py                      # ML 请求模型
├── factor.py                  # 因子请求/响应
├── risk.py                    # 风控请求模型
└── experiment.py              # 保留实验 schema
```

拆分原则：

- 当前 `routes.py` 内部所有 `BaseModel` 子类迁入 schemas。
- 若已有同名 schema 文件，优先合并到已有文件，避免重复定义。
- endpoint 内部仍可返回 `Dict[str, Any]`，本次不强制完整 response model 化。

### 4.3 工程配置

打包配置改为自动发现全部子包：

```toml
[tool.setuptools.packages.find]
include = ["quanttool*"]
```

依赖治理原则：

- `pyproject.toml` 作为主依赖来源。
- `requirements.txt` 保留给传统 pip 安装，但内容与主依赖保持一致或明确标记为运行时全集。
- 数据库、Web、分析、Qlib、Agent 等可选能力继续放在 `[project.optional-dependencies]`。
- 启动后端所需的 `uvicorn`、`asyncpg`、`sqlalchemy` 等依赖要能通过文档明确安装。

### 4.4 清理范围

确认无引用后删除或停止跟踪：

- 根目录 `.DS_Store`
- 根目录空的 `__init__.py`
- `quanttool/web/frontend/tsconfig.tsbuildinfo`
- `quanttool/cli/quanttool/config/default.yaml`
- `quanttool/web/frontend/quanttool/config/default.yaml`

保留但加入 ignore 或文档说明：

- `reports/` 作为运行产物目录，不纳入版本控制。
- `quanttool.log` 作为运行日志，不纳入版本控制。
- `.venv-mcp/` 和 `node_modules/` 继续忽略。

### 4.5 文档更新

更新 README 和相关文档：

- 真实目录结构。
- 后端和前端启动命令。
- Python 环境建议使用 `python3` 或项目 venv，不假设系统存在 `python`。
- 最小验证命令：Python smoke tests、FastAPI import、前端 lint。
- 说明 `reports/`、`.cache/`、日志、模型文件属于运行产物。

## 5. 迁移策略

迁移按低风险到高风险推进：

1. 先补 smoke tests，使当前失败点可见。
2. 修复 `analysis_service.py` 的旧 import。
3. 清理无引用生成物和误拷贝配置文件。
4. 修正打包配置和依赖清单。
5. 迁移 schema，保持原 endpoint 逻辑不变。
6. 分批拆 router，每拆一组运行 smoke tests。
7. 更新文档。

API 拆分顺序：

1. `tasks.py`，依赖最清晰，能先移出异步任务管理。
2. `registry.py`、`risk.py`、`factors.py`，接口较短。
3. `realtime.py`、`monitor.py`，共享 provider factory 迁入 dependencies。
4. `backtest.py`、`scan.py`、`stock.py`，核心业务接口。
5. `models.py`、`ml.py`，Qlib/GBM/ML 逻辑最长，最后拆。

## 6. 验证策略

新增 `tests/`，至少覆盖：

- `test_imports.py`
  - `import quanttool.web.app`
  - `from quanttool.application.analysis_service import AnalysisService`
  - `from quanttool.cli.main import app`
- `test_api_router.py`
  - FastAPI app 注册 `/api/stock/{symbol}/analysis`
  - 注册 `/api/backtest/run`
  - 注册 `/api/realtime/search`
  - 注册 `/api/gbm/train`
- `test_packaging.py`
  - 验证 `setuptools` 配置使用 `quanttool*` 子包发现。

手动验证命令：

```bash
.venv-mcp/bin/python -m compileall -q quanttool
.venv-mcp/bin/python -c "import quanttool.web.app; print('web app import ok')"
.venv-mcp/bin/python -c "from quanttool.application.analysis_service import AnalysisService; print('analysis service import ok')"
.venv-mcp/bin/python -m quanttool --help
cd quanttool/web/frontend && npm run lint
```

如果项目环境安装了 pytest，则运行：

```bash
.venv-mcp/bin/python -m pytest
```

## 7. 风险与控制

### 7.1 API 行为回归

风险：拆分 router 时路径、方法、默认参数或响应字段变化。

控制：先用 introspection 记录现有路由清单，拆分后比较路径和 HTTP method 集合完全一致。

### 7.2 循环导入

风险：子 router 之间互相导入工具函数导致循环依赖。

控制：共享函数只放 `utils.py` 和 `dependencies.py`；子 router 不互相导入。

### 7.3 依赖安装变化

风险：调整依赖后本地运行环境和文档不一致。

控制：保留 `requirements.txt`，并在 README 明确推荐安装命令。

### 7.4 误删仍有价值文件

风险：清理生成物时删除真实配置。

控制：删除前用 `rg` 确认无引用；配置只保留 `quanttool/config/default.yaml` 作为默认源。

## 8. 完成标准

本次整理完成时应满足：

- `quanttool/web/api/routes.py` 低于 100 行，只负责聚合 router。
- 所有 API request schema 从 `quanttool/web/schemas/` 导入。
- 现有 API 路径和 HTTP method 集合保持一致。
- `analysis_service.py` 可正常 import。
- `pyproject.toml` 能发现 `quanttool*` 子包。
- 误入版本库的生成物和嵌套配置副本被清理。
- README 反映真实目录结构和验证命令。
- 前端 `npm run lint` 通过。
- Python compile/import smoke checks 通过。
