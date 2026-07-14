# QuantTool Algorithm Core Refactor Design

> 日期: 2026-07-09
> 状态: 已批准进入第一波实施计划

## 背景

API 层已经完成路由拆分，算法相关的巨型模块成为下一轮维护瓶颈。当前主要问题不是算法数量多，而是计算规则、数据访问、报告文案、风险建议和兼容入口混在同一批对象里，导致任何算法调整都容易牵动 API、CLI 和报告输出。

本轮探索确认了几个核心热点：

- `quanttool/factors/stock_analyzer.py` 约 4857 行，同时负责取数、技术指标、三套评分、基本面、筛选、止损、推荐和 Markdown 报告。
- `quanttool/factors/scoring_system.py` 约 2843 行，同时包含 MyTT 风格指标函数、多维因子抽取、评分规则、触发信号、执行建议和报告辅助信息。
- `quanttool/factors/scoring/` 已经存在 `ScoringStrategy`、`ScoreResult` 和 `UnifiedScoringSystem`，但当前只有 `TrendScoringStrategy` 接近纯实现，`BreakoutScoringStrategy` 和 `MultiDimensionScoringStrategy` 仍主要包住旧系统。
- `quanttool/backtest/engine.py`、`quanttool/strategies/qlib_strategy.py` 和 `quanttool/factors/ml_feature_engineer.py` 也有算法重复和职责混合，但它们外部依赖和交易语义风险更高，暂不作为第一刀。

## 目标

第一波目标是让评分和单股分析形成稳定、可测试的算法内核，为后续回测、Qlib 和 ML 特征工程重构提供安全基础。

1. 给评分系统和分析上下文建立行为锁定测试，覆盖确定性样本上的输出形状和关键数值。
2. 沿用现有 `quanttool/factors/scoring/` 策略化接口，不另起新框架。
3. 把多维评分的旧实现逐步包进统一评分门面，先保持行为兼容，再拆内部责任。
4. 把 `StockAnalyzer` 从“大而全对象”收窄为兼容 facade，新增专门的分析编排器和报告生成器。
5. 保持现有 API、CLI、agent tools 和历史导入路径兼容。

## 非目标

第一波不做以下事情：

- 不删除 `quanttool/factors/scoring_system.py`、`trend_scoring_system.py`、`breakout_scoring_system.py` 的公开导入路径。
- 不改变三套评分算法的阈值、权重、推荐规则或报告字段。
- 不重构 `quanttool/backtest/engine.py` 的交易撮合、T+1、止损止盈或绩效指标。
- 不重构 `quanttool/strategies/qlib_strategy.py`、`gbm_strategy.py`、`ml_feature_engineer.py` 的训练、预测或特征定义。
- 不新增运行依赖。
- 不要求真实外部数据源可用；第一波测试使用本地构造的确定性 K 线样本。

## 推荐方案

采用“评分/分析内核优先”的增量路线。

### 方案 A: 评分与分析内核优先（推荐）

先围绕 `factors/scoring/` 和 `StockAnalyzer` 做行为锁定、接口统一和编排拆分。

优点：

- 已有 `ScoringStrategy` 和 `UnifiedScoringSystem` 可以承接迁移。
- 主要是纯 DataFrame 计算和对象组装，外部依赖少。
- 直接减少 API、CLI、agent tools 对巨型 `StockAnalyzer` 的隐式依赖。

代价：

- 第一波不会立刻解决回测撮合和 Qlib 模型文件过大的问题。

### 方案 B: 回测引擎优先

先拆事件循环、订单执行、A 股约束和绩效指标。

优点：交易语义会更清晰。

代价：T+1、滑点、涨跌停、止损止盈很容易行为漂移，当前测试基线不足，不适合作为第一刀。

### 方案 C: Qlib/GBM 优先

先拆特征工程、模型适配和信号解释。

优点：训练和预测链路更清晰。

代价：依赖 pyqlib、lightgbm 等环境状态，且当前有 fallback 伪概率逻辑，验证成本最高。

## 目标结构

第一波完成后，目标结构如下：

```text
quanttool/factors/
├── analysis_context.py              # 保留统一上下文数据结构
├── analysis_orchestrator.py         # 新增：纯编排分析上下文，不做取数和报告
├── stock_analyzer.py                # 兼容 facade：取数、指标、调用 orchestrator/report
├── reports/
│   ├── __init__.py
│   └── stock_report.py              # 新增：从 AnalysisContext 生成 Markdown
├── scoring/
│   ├── __init__.py
│   ├── base.py                      # 现有 ScoreResult / ScoringStrategy
│   ├── unified_scoring_system.py    # 现有统一评分门面，补充兼容输出
│   └── strategies/
│       ├── trend.py                 # 现有趋势策略
│       ├── breakout.py              # 继续适配旧突破系统
│       └── multi_dimension.py       # 继续适配旧多维评分系统，补齐 legacy 输出
```

测试目标结构：

```text
tests/
├── test_smoke.py                    # 保留工程 smoke tests
├── fixtures/
│   └── algorithm_data.py             # 新增确定性 K 线样本构造
├── test_scoring_contracts.py         # 新增评分策略/统一门面行为测试
└── test_analysis_orchestrator.py     # 新增分析上下文编排测试
```

## 组件设计

### 1. 确定性算法样本

新增测试辅助函数，构造无需网络和数据库的 OHLCV DataFrame：

- `make_trending_ohlcv(rows: int = 260) -> pd.DataFrame`
- `make_sideways_ohlcv(rows: int = 260) -> pd.DataFrame`
- `make_breakout_ohlcv(rows: int = 260) -> pd.DataFrame`

样本必须包含 `timestamp`、`date`、`open`、`high`、`low`、`close`、`volume`、`amount`。数值采用固定公式，不使用随机数。

### 2. 统一评分门面

现有 `UnifiedScoringSystem.calculate_scores(df, **kwargs)` 保留。第一波新增一个兼容汇总方法：

```python
def calculate_context_scores(
    self,
    df: pd.DataFrame,
    symbol: str = "",
    trade_date: str = "",
) -> dict:
    """Return classic/trend/breakout score payloads for AnalysisContext builders."""
```

该方法不替代旧 `ScoringSystem.calculate_all_scores`，只为新的编排器提供统一入口。旧类继续存在，`MultiDimensionScoringStrategy` 内部仍可委托旧类。

### 3. 分析编排器

新增 `AnalysisOrchestrator`，只接受已准备好的 DataFrame 和可注入组件，不直接取外部行情：

```python
class AnalysisOrchestrator:
    def __init__(
        self,
        scoring_system: Optional[UnifiedScoringSystem] = None,
        recommendation_engine: Optional[RecommendationEngine] = None,
        stop_loss_calculator: Optional[UnifiedStopLossCalculator] = None,
        market_state_builder: Optional[Callable[[pd.DataFrame], UnifiedMarketState]] = None,
        fundamental_provider: Optional[Callable[[str], FundamentalData]] = None,
    ) -> None:
        ...

    def build_context(
        self,
        df: pd.DataFrame,
        symbol: str,
        primary_system: ScoringSystemType = ScoringSystemType.AUTO,
        current_price: Optional[float] = None,
    ) -> AnalysisContext:
        ...
```

默认情况下，编排器复用现有评分、推荐、止损、筛选和传统策略逻辑；测试中可注入轻量 fake，避免网络、数据库和实时行情。

### 4. 报告生成器

新增 `StockReportGenerator`，接收 `AnalysisContext` 和已计算指标的 DataFrame 生成 Markdown。第一波只搬迁 `generate_report_from_context` 及其 `_generate_*_v2` 辅助函数，不改文案和章节。

`StockAnalyzer.generate_report_from_context(...)` 保留，但改为委托 `StockReportGenerator.generate(...)`。

### 5. 兼容 facade

`StockAnalyzer` 继续提供现有公开方法：

- `get_stock_data`
- `calculate_technical_indicators`
- `run_trading_strategies`
- `build_analysis_context`
- `analyze_stock_with_context`
- `generate_report_from_context`
- `analyze_stock`
- `analyze_stock_enhanced`

第一波只让 `build_analysis_context` 和 `generate_report_from_context` 转向新模块。外部调用方式和返回类型不变。

## 数据流

第一波目标数据流：

```text
StockAnalyzer.get_stock_data(symbol)
    -> DataFrame
StockAnalyzer.calculate_technical_indicators(df)
    -> DataFrame with indicators
AnalysisOrchestrator.build_context(df, symbol)
    -> AnalysisContext
StockReportGenerator.generate(df, context, symbol)
    -> Markdown report
```

评分数据流：

```text
DataFrame
    -> UnifiedScoringSystem.calculate_context_scores(...)
    -> ClassicScore / TrendScore / BreakoutScore compatible payloads
    -> AnalysisContext
    -> RecommendationEngine.generate_recommendation(context)
```

## 错误处理

- 数据为空时返回空 `AnalysisContext`，保持现有行为。
- 某个评分策略失败时生成默认分数对象，并记录失败原因，不中断其他评分。
- 基本面获取失败时保留空 `FundamentalData`，不影响技术分析。
- 报告生成器只消费 `AnalysisContext`，不做外部 IO。
- 兼容 facade 中的异常捕获先保持旧行为，后续再统一 logger。

## 测试策略

第一波新增测试分三层：

1. 样本测试：确保测试 K 线列完整、长度固定、无随机性。
2. 评分契约测试：确保 `UnifiedScoringSystem` 返回三套策略结果，结果可序列化，关键字段存在。
3. 分析编排测试：注入 fake 组件，验证 `AnalysisOrchestrator.build_context` 能生成完整 `AnalysisContext`，并且 `StockAnalyzer` facade 委托不改变返回类型。

现有验证继续执行：

```bash
.venv-mcp/bin/python -m unittest discover -s tests -v
.venv-mcp/bin/python -m compileall -q quanttool
cd quanttool/web/frontend && npm run lint
```

## 风险与控制

### 评分行为漂移

风险：迁移到统一门面时改变分数、字段名或过滤状态。

控制：第一波先使用 legacy 适配，不重写核心算法。新增测试检查字段和基本数值范围；后续若要替换内部算法，再用黄金样本做数值对比。

### 循环导入

风险：`StockAnalyzer`、`AnalysisOrchestrator`、`StockReportGenerator` 互相导入导致循环。

控制：`AnalysisOrchestrator` 不导入 `StockAnalyzer`；报告生成器只导入上下文模型；`StockAnalyzer` 单向依赖 orchestrator/report。

### 外部数据依赖拖慢测试

风险：测试触发 qlib、数据库或实时行情。

控制：测试使用本地确定性 DataFrame，并通过依赖注入 fake 基本面、市场状态、推荐和止损组件。

### 文件搬迁过大

风险：一次性移动几千行报告函数导致难以 review。

控制：第一波只搬 `generate_report_from_context` 的新版上下文报告链路；旧版增强报告和其他长函数留在 `StockAnalyzer`。

## 完成边界

第一波完成定义：

1. 新增算法测试基线通过。
2. `AnalysisOrchestrator` 可独立构建 `AnalysisContext`。
3. `StockReportGenerator` 可独立从上下文生成 Markdown。
4. `StockAnalyzer.build_analysis_context` 与 `generate_report_from_context` 已委托新模块。
5. 现有 smoke tests、compileall、frontend lint 全部通过。
6. 工作区只剩用户自有未跟踪文件，不混入提交。
