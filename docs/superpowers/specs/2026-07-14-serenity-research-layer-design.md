# QuantTool Serenity Research Layer Design

> 日期: 2026-07-14
> 状态: 已批准进入第一波实施

## 背景

QuantTool 当前擅长回答“价格和量化信号是否支持现在行动”，但对“这家公司是否真的处在产业链稀缺环节、公开证据是否足够、市场叙事可能错在哪里”缺少稳定的数据结构和评分契约。Serenity 方法正好补足这一层，但它不应被包装成新的交易信号，也不应直接和技术评分相加。

本设计把两套能力明确分开：

- QuantTool 继续负责量化择时、趋势、突破、风险和回测。
- Serenity 研究层负责产业链位置、供给约束、证据质量、风险扣分和反方条件。
- 两者通过二维象限协作，不通过单一总分混合。

第一波只实现可重复、可测试的本地评分闭环。实时主题扫描、公告抓取和自动产业链建图留到第二波，它们需要独立的数据源与证据追踪设计。

## 目标

1. 将 Serenity 瓶颈评分规则迁入 QuantTool 自己的领域模型和应用服务。
2. 保持评分规则唯一，CLI、API 和前端共享同一份计算结果。
3. 保存证据、反方条件和风险扣分，不把研究评分简化成一个裸分数。
4. 可选接收 QuantTool 的量化择时分数，并返回二维象限，不计算混合总分。
5. 提供 CLI、FastAPI 和前端研究工作台三个入口。
6. 明确输出是研究优先级，不包含买卖指令、收益承诺或自动交易动作。

## 非目标

- 第一波不联网搜索公告、新闻、互动易或项目备案。
- 第一波不自动生成公司候选池或主题产业链地图。
- 不修改现有趋势、突破、经典评分权重和阈值。
- 不把 Serenity 分数写入现有 `final_recommendation`。
- 不根据研究分数触发下单、监控或调仓。
- 不新增 Python 运行依赖。

## 核心决策

### 1. 双层评分，不混分

`research_priority_score` 和 `timing_score` 分别保留。研究层只在两者都存在时返回象限：

| 研究优先级 | 量化择时 | 象限 | 含义 |
|---|---|---|---|
| 高 | 高 | `priority_now` | 研究逻辑较强，量化时机也支持继续验证 |
| 高 | 低 | `research_wait` | 研究逻辑较强，等待价格或技术确认 |
| 低 | 高 | `timing_only` | 价格表现强，但产业链和证据仍可能偏叙事 |
| 低 | 低 | `low_priority` | 暂不占用研究资源 |

高低阈值第一波固定为 70。它只用于界面分组，不改变两侧原始分数。

### 2. 评分规则兼容 Serenity.scorecard

八个正向因子使用 0 到 5 的评分，并按以下权重映射到 100 分：

- `demand_inflection`: 15
- `architecture_coupling`: 10
- `chokepoint_severity`: 15
- `supplier_concentration`: 12
- `expansion_difficulty`: 12
- `evidence_quality`: 15
- `valuation_disconnect`: 11
- `catalyst_timing`: 10

风险项同样使用 0 到 5，每级扣 2 分。第一波支持固定八类风险：融资稀释、治理、地缘政治、流动性、炒作、会计质量、周期性和替代设计。

最终研究优先级分数限制在 0 到 100：

```text
final_score = clamp(weighted_factor_points - penalty_points, 0, 100)
```

优先级文案：

- 85 及以上: `top_priority`
- 70 到 84.99: `high_priority`
- 55 到 69.99: `worth_tracking`
- 55 以下: `early_lead`

### 3. 证据是数据，不是备注尾巴

每条证据保存：

- `claim`: 被支持的具体判断
- `source`: 来源标题、路径或 URL
- `strength`: `strong`、`medium`、`weak`、`unverified`
- `published_at`: 可选日期

第一波不自动验证 URL，但应用服务会统计证据强弱和未验证数量。`evidence_quality` 因子仍由研究者评分，因为“来源等级”与“证据能否真正证明该结论”不是同一回事。

### 4. 反方条件必须进入结果

`what_could_weaken_view` 作为正式字段原样进入评分结果。CLI、API 和前端都展示它，避免只留下支持性论据。

## 目标结构

```text
quanttool/
├── domain/models/serenity.py          # 输入、结果、证据和枚举
├── application/serenity_service.py    # 纯评分、象限和 Markdown 输出
├── cli/commands/research_commands.py  # template / scorecard 命令
├── web/api/research.py                # Serenity API router
├── web/schemas/serenity.py            # HTTP 请求/响应契约
└── web/frontend/
    ├── app/research/page.tsx           # 研究评分工作台
    ├── lib/api/research.ts             # API client
    └── types/research.ts               # 前端类型
```

测试新增：

```text
tests/
├── test_serenity_service.py
├── test_serenity_cli.py
└── test_serenity_api.py
```

前端源代码契约测试继续放在 `tests/test_frontend_cli_optimization.py`，并用构建和浏览器检查补足运行验证。

## 领域模型

领域层使用 Pydantic 模型，与项目现有 Python 版本和序列化方式保持一致。

```python
class SerenityEvidence(BaseModel):
    claim: str
    source: str
    strength: EvidenceStrength
    published_at: Optional[date] = None


class SerenityScorecard(BaseModel):
    ticker: str = ""
    company: str = ""
    market: str = "A-share"
    theme: str = ""
    layer: str = ""
    role: str = ""
    factors: SerenityFactors
    penalties: SerenityPenalties = SerenityPenalties()
    evidence: List[SerenityEvidence] = []
    what_could_weaken_view: List[str] = []
    timing_score: Optional[float] = None


class SerenityScoreResult(BaseModel):
    research_priority_score: float
    raw_factor_points: float
    penalty_points: float
    verdict: ResearchVerdict
    quadrant: Optional[ResearchTimingQuadrant]
    factor_details: Dict[str, ScoreDetail]
    penalty_details: Dict[str, PenaltyDetail]
    evidence_summary: EvidenceSummary
    evidence: List[SerenityEvidence]
    what_could_weaken_view: List[str]
```

所有 0 到 5 的字段和可选 `timing_score` 都在模型边界校验，非法输入不会进入计算逻辑。

## 应用服务

`SerenityService` 是无状态纯服务：

```python
class SerenityService:
    def score(self, scorecard: SerenityScorecard) -> SerenityScoreResult:
        ...

    def template(self) -> SerenityScorecard:
        ...

    def to_markdown(self, result: SerenityScoreResult) -> str:
        ...
```

服务不导入行情、数据库、Web 或 CLI 模块。CLI 和 API 只负责输入输出转换。

## CLI 设计

新增顶级命令组：

```bash
quant research template
quant research scorecard thesis.json --format json
quant research scorecard thesis.json --format md
quant research scorecard - --format both
```

`-` 表示从标准输入读取。JSON 错误、字段越界和文件不存在都转换成清晰的 Click 错误，不输出 Python 堆栈。

## API 设计

第一波提供两个端点：

```text
GET  /api/research/serenity/template
POST /api/research/serenity/scorecard
```

响应遵循项目统一格式：

```json
{
  "success": true,
  "data": {},
  "error": null,
  "timestamp": "2026-07-14T00:00:00Z"
}
```

Pydantic 输入错误由 FastAPI 返回 422；业务层不吞掉验证错误。

## 前端动线

侧边栏“研究”分组新增“产业链研究”，路径为 `/research`。

页面采用单页工作台，不做介绍型落地页：

1. 顶部输入公司、代码、市场、主题、产业链层级和角色。
2. 中部用紧凑的 0 到 5 数值控件编辑八个正向因子和八个风险项。
3. 证据和反方条件是独立全宽区段，不嵌套卡片。
4. 提交后右侧或下方稳定展示研究分、风险扣分、证据摘要和二维象限。
5. 页面明确显示“研究优先级，不是交易指令”。

第一波不提供自动联网研究按钮，避免制造“已验证”的错觉。后续主题扫描会在同一路径增加独立模式，并明确数据更新时间和来源状态。

## 错误处理

- 领域模型拒绝区间外评分。
- CLI 对非法 JSON、读取失败和模型校验失败返回非零状态。
- API 对非法输入返回 422，对应用层意外错误返回统一失败响应。
- 前端保留用户已输入内容，展示字段级或请求级错误。
- 没有 `timing_score` 时结果不返回象限，不猜测量化信号。

## 测试策略

1. 领域/服务：锁定权重、扣分、上下界、优先级阈值、证据摘要和四象限。
2. CLI：用 Typer `CliRunner` 验证模板、JSON、Markdown、标准输入和错误路径。
3. API：验证路由注册、统一响应和 422 输入校验。
4. 前端：源代码契约测试、TypeScript 构建和浏览器交互检查。
5. 回归：运行现有 Python 全量测试，确保不改变旧评分和扫描逻辑。

## 风险与控制

### 研究分被误解为交易分

控制：字段名固定为 `research_priority_score`；结果保留研究/择时双轴；UI 和 Markdown 显示研究边界。

### 手工证据评分过度自信

控制：保存证据等级和未验证计数；没有强证据不会被系统隐式升级；反方条件必须可见。

### 与现有评分系统耦合

控制：第一波只接收可选 `timing_score` 数字，不导入 `StockAnalyzer` 或 `UnifiedScoringSystem`。后续自动接线由应用层适配器完成。

### 前端先于数据能力

控制：页面定位为结构化研究评分工作台，不宣称自动调研。实时抓取和主题扫描单独设计、单独标注数据状态。

## 完成边界

第一波完成需要同时满足：

1. 领域模型和 `SerenityService` 通过权重、扣分、阈值、证据和象限测试。
2. CLI 可生成模板并从文件或标准输入计算 JSON/Markdown。
3. FastAPI 端点已注册并使用统一响应结构。
4. `/research` 可完成一次评分并展示研究分、风险、证据摘要和反方条件。
5. 现有 56 个 Python 基线测试继续通过，新增测试通过。
6. 前端生产构建通过，并完成桌面与移动端浏览器检查。
7. 没有改动现有交易评分权重、阈值或推荐逻辑。
