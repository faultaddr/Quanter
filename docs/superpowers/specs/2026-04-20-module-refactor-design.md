# QuantTool 模块重构设计文档

> 日期: 2026-04-20
> 状态: 已完成

## 1. 问题背景

当前 QuantTool 项目存在以下架构混乱问题：

### 1.1 评分系统重复
| 文件 | 大小 | 功能 |
|------|------|------|
| `scoring_system.py` | 111KB | 多维度打分（右侧交易），基于 MyTT |
| `breakout_scoring_system.py` | 28KB | 低位盘整突破评分 |
| `trend_scoring_system.py` | 24KB | 趋势选股评分 |

问题：三套评分系统相互独立，代码重复，难以维护和扩展。

### 1.2 前端重复
| 前端 | 位置 | 技术栈 |
|------|------|--------|
| 单文件前端 | `quanttool/web/static/index.html` | Bootstrap + 原生 JS |
| Next.js 前端 | `quanttool/web/frontend/` | React + Next.js + Tailwind |

问题：两套前端并存，维护成本翻倍，功能不一致。

### 1.3 报告系统混乱
- `reports/daily_report_generator.py` - 每日投资报告
- `reports/signal_backtest_report.py` - 信号回测报告
- `reports/signal_attribution.py` - 信号归因
- 根目录 `stock_report_zh.py` - 硬编码的中文报告（废弃）

问题：报告生成逻辑分散，缺乏统一接口，难以扩展新报告类型。

### 1.4 数据提供者职责不清
共 14+ 个 Provider/Fetcher 类：
- 基础数据：`AShareProvider`, `TuShareProvider`, `CSVProvider`
- 增强数据：`EnhancedDataFetcher`, `AshareFetcher`, `RealAShareDataProvider`（重复）
- 实时数据：`SinaRealtimeProvider`, `PytdxRealtimeProvider`, `RealtimeDataProvider`
- 分钟数据：`AkShareMinuteProvider`, `IncrementalMinuteProvider`
- 异步/增量：`AsyncDataFetcher`, `IncrementalDataProvider`

问题：职责重叠，命名混乱，缺乏清晰的分层架构。

---

## 2. 新旧类映射表

### 2.1 数据提供者映射

| 现有类 | 目标位置 | 操作 |
|--------|----------|------|
| `AShareProvider` | `historical/ashare_provider.py` | 移动 |
| `TuShareProvider` | `historical/tushare_provider.py` | 移动 |
| `CSVProvider` | `historical/csv_provider.py` | 移动 |
| `RealAShareDataProvider` | - | **删除**（与 AShareProvider 重复） |
| `EnhancedDataFetcher` | `historical/enhanced_fetcher.py` | 移动 |
| `AshareFetcher` | `historical/ashare_fetcher.py` | 移动 |
| `SinaRealtimeProvider` | - | **合并**到 `realtime/sina_source.py` |
| `PytdxRealtimeProvider` | - | **合并**到 `realtime/pytdx_source.py` |
| `RealtimeDataProvider`（现有）| - | **重命名**为 `LegacyRealtimeDataProvider`，后续删除 |
| 新建 `RealtimeDataProvider` | `realtime/realtime_provider.py` | **新建**（统一入口） |
| `AkShareMinuteProvider` | `incremental/minute_provider.py` | 移动并合并 |
| `IncrementalMinuteProvider` | - | **合并**到 `minute_provider.py` |
| `IncrementalDataProvider` | `incremental/incremental_provider.py` | 移动 |
| `AsyncDataFetcher` | `incremental/async_fetcher.py` | 移动 |

### 2.2 评分系统映射

| 现有文件 | 目标位置 | 操作 |
|----------|----------|------|
| `scoring_system.py` | `scoring/strategies/multi_dimension.py` | 迁移为策略类 |
| `breakout_scoring_system.py` | `scoring/strategies/breakout.py` | 迁移为策略类 |
| `trend_scoring_system.py` | `scoring/strategies/trend.py` | 迁移为策略类 |
| 新建 | `scoring/base.py` | **新建**（策略接口） |
| 新建 | `scoring/unified_scoring_system.py` | **新建**（统一入口） |

### 2.3 报告系统映射

| 现有文件 | 目标位置 | 操作 |
|----------|----------|------|
| `daily_report_generator.py` | `generators/daily_report.py` | 迁移 |
| `signal_backtest_report.py` | `generators/backtest_report.py` | 迁移 |
| `signal_attribution.py` | `generators/attribution_report.py` | 迁移 |
| 新建 | `base.py` | **新建**（基类） |
| `stock_report_zh.py` | - | **删除**（硬编码废弃） |

---

## 3. 重构目标

1. **删除冗余前端**：保留 Next.js，删除单文件前端
2. **统一评分系统**：设计可配置的策略模式评分框架
3. **统一报告系统**：设计可扩展的报告生成器基类
4. **重构数据提供者**：按职责分层，清晰架构

---

## 4. 目标架构

### 4.1 目录结构

```
quanttool/
├── infrastructure/
│   └── data_providers/
│       ├── __init__.py
│       ├── base.py                    # IDataProvider 接口定义
│       │
│       ├── historical/                # 历史数据层
│       │   ├── __init__.py
│       │   ├── ashare_provider.py     # A股历史数据
│       │   ├── tushare_provider.py    # TuShare 数据
│       │   ├── csv_provider.py        # CSV 数据源
│       │   ├── enhanced_fetcher.py    # 增强数据获取
│       │   └── ashare_fetcher.py      # Ashare 数据获取
│       │
│       ├── realtime/                  # 实时数据层
│       │   ├── __init__.py
│       │   ├── realtime_provider.py   # 统一实时数据接口
│       │   ├── sina_source.py         # 新浪数据源
│       │   └── pytdx_source.py        # 通达信数据源
│       │
│       └── incremental/               # 增量数据层
│           ├── __init__.py
│           ├── minute_provider.py     # 分钟数据
│           ├── incremental_provider.py # 增量数据
│           ├── async_fetcher.py       # 异步数据获取
│           └── cache_manager.py       # 缓存管理
│
├── factors/
│   ├── scoring/                       # 评分系统（新）
│   │   ├── __init__.py
│   │   ├── base.py                    # ScoringStrategy 接口
│   │   ├── strategies/
│   │   │   ├── __init__.py
│   │   │   ├── trend.py               # 趋势评分策略
│   │   │   ├── breakout.py            # 突破评分策略
│   │   │   └── multi_dimension.py     # 多维度评分策略
│   │   └── unified_scoring_system.py  # 统一评分系统
│   │
│   └── ... (其他因子模块保留)
│
├── reports/                           # 报告系统（重构）
│   ├── __init__.py
│   ├── base.py                        # ReportGenerator 基类
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── daily_report.py            # 每日报告
│   │   ├── backtest_report.py         # 回测报告
│   │   └── attribution_report.py      # 归因报告
│   └── templates/
│       └── ...
│
└── web/
    ├── frontend/                      # Next.js 前端（保留）
    └── static/                        # 删除
```

### 4.2 评分系统架构

```python
# factors/scoring/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd

@dataclass
class ScoreResult:
    """评分结果"""
    final_score: float
    passed_filter: bool
    filter_reason: str = ""
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

class ScoringStrategy(ABC):
    """评分策略基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass

    @abstractmethod
    def calculate_score(self, df: pd.DataFrame, **kwargs) -> ScoreResult:
        """计算评分"""
        pass

    def validate_data(self, df: pd.DataFrame) -> bool:
        """验证数据是否满足要求"""
        return len(df) > 0

class UnifiedScoringSystem:
    """统一评分系统"""

    def __init__(self, strategies: list[ScoringStrategy] = None):
        self.strategies = strategies or []

    def add_strategy(self, strategy: ScoringStrategy):
        self.strategies.append(strategy)

    def calculate_scores(self, df: pd.DataFrame, **kwargs) -> Dict[str, ScoreResult]:
        """使用所有策略计算评分"""
        results = {}
        for strategy in self.strategies:
            if strategy.validate_data(df):
                results[strategy.name] = strategy.calculate_score(df, **kwargs)
        return results

    def get_best_strategy(self, df: pd.DataFrame, **kwargs) -> tuple[str, ScoreResult]:
        """获取最高评分的策略"""
        scores = self.calculate_scores(df, **kwargs)
        if not scores:
            return None, None
        return max(scores.items(), key=lambda x: x[1].final_score)
```

### 4.3 报告系统架构

```python
# reports/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Any, Optional
from pathlib import Path

@dataclass
class ReportContext:
    """报告上下文基类"""
    report_date: date
    generated_at: datetime = field(default_factory=datetime.now)

@dataclass
class DailyReportContext(ReportContext):
    """日报上下文"""
    pass

@dataclass
class BacktestReportContext(ReportContext):
    """回测报告上下文"""
    backtest_id: str = ""

@dataclass
class AttributionReportContext(ReportContext):
    """归因报告上下文"""
    portfolio_id: str = ""

class ReportGenerator(ABC):
    """报告生成器基类"""

    def __init__(self, template_dir: Optional[Path] = None):
        self.template_dir = template_dir

    @property
    @abstractmethod
    def report_type(self) -> str:
        """报告类型"""
        pass

    @abstractmethod
    def gather_data(self, context: ReportContext) -> Dict[str, Any]:
        """收集报告数据"""
        pass

    @abstractmethod
    def render(self, data: Dict[str, Any]) -> str:
        """渲染报告"""
        pass

    def generate(self, context: ReportContext) -> str:
        """生成报告（模板方法）"""
        data = self.gather_data(context)
        return self.render(data)

class ReportFactory:
    """报告工厂"""

    _generators: Dict[str, type[ReportGenerator]] = {}

    @classmethod
    def register(cls, generator_class: type[ReportGenerator]):
        cls._generators[generator_class.report_type] = generator_class

    @classmethod
    def create(cls, report_type: str, **kwargs) -> ReportGenerator:
        if report_type not in cls._generators:
            raise ValueError(f"Unknown report type: {report_type}")
        return cls._generators[report_type](**kwargs)
```

### 4.4 数据提供者架构

```python
# infrastructure/data_providers/base.py
from abc import ABC, abstractmethod
from typing import Optional
from datetime import date
import pandas as pd

class IDataProvider(ABC):
    """数据提供者接口"""

    @abstractmethod
    def get_daily_data(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """获取日线数据"""
        pass

# infrastructure/data_providers/realtime/base.py
class IRealtimeDataSource(ABC):
    """实时数据源接口"""

    @property
    @abstractmethod
    def name(self) -> str:
        """数据源名称"""
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> dict:
        """获取实时行情

        Returns:
            dict: {
                'symbol': str,
                'price': float,
                'change': float,
                'change_pct': float,
                'volume': int,
                'amount': float,
                'high': float,
                'low': float,
                'open': float,
                'prev_close': float,
                'timestamp': datetime,
            }
        """
        pass

    @abstractmethod
    def get_quotes(self, symbols: list[str]) -> list[dict]:
        """批量获取实时行情"""
        pass

    def is_available(self) -> bool:
        """检查数据源是否可用"""
        try:
            self.get_quote('000001.SZ')
            return True
        except Exception:
            return False

# infrastructure/data_providers/realtime/realtime_provider.py
class RealtimeDataProvider:
    """统一实时数据提供者"""

    def __init__(self, sources: list[IRealtimeDataSource] = None):
        self.sources = sources or []
        self._init_default_sources()

    def _init_default_sources(self):
        """初始化默认数据源"""
        if not self.sources:
            self.sources = [
                SinaRealtimeSource(),    # 优先新浪
                PytdxRealtimeSource(),   # 备用通达信
            ]

    def get_quote(self, symbol: str) -> dict:
        """获取实时行情（自动故障转移）"""
        for source in self.sources:
            try:
                return source.get_quote(symbol)
            except Exception:
                continue
        raise DataProviderError("所有数据源均不可用")

# infrastructure/data_providers/incremental/minute_provider.py
class MinuteProvider:
    """分钟数据提供者"""

    def __init__(self, cache_manager: Optional['CacheManager'] = None):
        self.cache_manager = cache_manager

    def get_minute_data(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        period: str = '1min'
    ) -> pd.DataFrame:
        """获取分钟数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            period: 周期 ('1min', '5min', '15min', '30min', '60min')

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume
        """
        pass
```

---

## 5. 实施计划

### Phase 1: 清理废弃代码（预估 1 天）

**目标**：删除明确不再使用的代码

**删除列表**：
```
quanttool/web/static/index.html          # 单文件前端
stock_report_zh.py                       # 硬编码报告
analyze_corrected_601777.py              # 测试脚本
analyze_sh601777.py                      # 测试脚本
check_stock_name.py                      # 测试脚本
test_all_enhanced_features.py            # 测试脚本（根目录）
```

**验证脚本**：
```bash
# 1. 运行单元测试
pytest tests/ -v

# 2. 检查 API 健康状态
curl http://localhost:8000/api/health

# 3. 启动前端开发服务器
cd quanttool/web/frontend && npm run dev

# 4. 访问前端页面确认正常
# 浏览器打开 http://localhost:3000
```

**验证清单**：
- [ ] `pytest tests/ -v` 全部通过
- [ ] `curl http://localhost:8000/api/health` 返回 `{"status": "ok"}`
- [ ] Next.js 前端正常运行（`npm run dev` 无报错）
- [ ] 所有页面可正常访问

---

### Phase 2: 数据提供者分层重构（预估 2 天）

**目标**：按职责重新组织数据提供者

**步骤**：
1. 创建新的目录结构
2. 移动现有代码到对应目录
3. **删除前验证功能对等**：
   ```bash
   # 验证 AShareProvider 包含 RealAShareDataProvider 的所有功能
   python -c "
   from quanttool.infrastructure.data_providers import AShareProvider, RealAShareDataProvider

   # 列出两个类的所有公开方法
   ashare_methods = set(m for m in dir(AShareProvider) if not m.startswith('_'))
   real_methods = set(m for m in dir(RealAShareDataProvider) if not m.startswith('_'))

   missing = real_methods - ashare_methods
   if missing:
       print(f'警告：AShareProvider 缺少以下方法: {missing}')
   else:
       print('AShareProvider 功能对等验证通过')
   "
   ```
4. 合并重复的类：
   - `RealAShareDataProvider` → 删除，使用 `AShareProvider`（验证对等后）
   - `SinaRealtimeProvider` + `PytdxRealtimeProvider` → 实现 `IRealtimeDataSource` 接口
   - `IncrementalDataProvider` + `IncrementalMinuteProvider` → `MinuteProvider`
5. 更新所有导入路径
6. 添加 `__init__.py` 导出

**验证脚本**：
```bash
# 1. 运行单元测试
pytest tests/ -v

# 2. 测试数据获取
python -c "
from quanttool.infrastructure.data_providers import AShareProvider
provider = AShareProvider()
df = provider.get_daily_data('000001.SZ', start_date='2024-01-01')
print(f'获取到 {len(df)} 条数据')
"

# 3. 测试实时行情
python -c "
from quanttool.infrastructure.data_providers.realtime import RealtimeDataProvider
provider = RealtimeDataProvider()
quote = provider.get_quote('000001.SZ')
print(f'实时价格: {quote[\"price\"]}')
"

# 4. 测试分钟数据
python -c "
from quanttool.infrastructure.data_providers.incremental import MinuteProvider
provider = MinuteProvider()
df = provider.get_minute_data('000001.SZ')
print(f'获取到 {len(df)} 条分钟数据')
"
```

**验证清单**：
- [ ] `pytest tests/ -v` 全部通过
- [ ] 历史数据获取正常
- [ ] 实时行情获取正常（自动故障转移）
- [ ] 分钟数据获取正常

---

### Phase 3: 评分系统统一（预估 2 天）

**目标**：设计统一的评分框架，迁移现有策略

**步骤**：
1. 创建 `factors/scoring/` 目录
2. 实现 `ScoringStrategy` 基类
3. 迁移三大评分系统为策略类：
   - `scoring_system.py` → `MultiDimensionScoringStrategy`
   - `breakout_scoring_system.py` → `BreakoutScoringStrategy`
   - `trend_scoring_system.py` → `TrendScoringStrategy`
4. 实现 `UnifiedScoringSystem`
5. 更新 API 端点使用新接口
6. 删除旧的评分文件

**验证脚本**：
```bash
# 1. 运行单元测试（包含评分对比测试）
pytest tests/test_scoring/ -v

# 2. 测试评分 API
curl -X POST http://localhost:8000/api/stock/000001.SZ/score

# 3. 测试各策略评分一致性
python -c "
from quanttool.factors.scoring import UnifiedScoringSystem
from quanttool.factors.scoring.strategies import (
    TrendScoringStrategy,
    BreakoutScoringStrategy,
    MultiDimensionScoringStrategy
)
from quanttool.infrastructure.data_providers import AShareProvider

provider = AShareProvider()
df = provider.get_daily_data('000001.SZ', start_date='2024-01-01')

scorer = UnifiedScoringSystem([
    TrendScoringStrategy(),
    BreakoutScoringStrategy(),
    MultiDimensionScoringStrategy(),
])

results = scorer.calculate_scores(df)
for name, result in results.items():
    print(f'{name}: {result.final_score:.2f}')
"
```

**验证清单**：
- [ ] `pytest tests/test_scoring/ -v` 全部通过
- [ ] 各策略评分结果与旧代码一致（误差 < 0.01）
- [ ] API 端点返回正确数据
- [ ] 前端评分展示正常

---

### Phase 4: 报告系统统一（预估 1 天）

**目标**：设计统一的报告框架

**步骤**：
1. 实现 `ReportGenerator` 基类
2. 迁移现有报告生成器：
   - `daily_report_generator.py` → `DailyReportGenerator`
   - `signal_backtest_report.py` → `BacktestReportGenerator`
   - `signal_attribution.py` → `AttributionReportGenerator`
3. 实现 `ReportFactory`
4. 更新 CLI 命令使用新接口

**验证脚本**：
```bash
# 1. 运行单元测试
pytest tests/test_reports/ -v

# 2. 测试日报生成
python -c "
from quanttool.reports import ReportFactory
from quanttool.reports.base import DailyReportContext
from datetime import date

generator = ReportFactory.create('daily')
context = DailyReportContext(report_date=date.today())
report = generator.generate(context)
print(report[:500])  # 打印前 500 字符
"

# 3. 测试回测报告生成
python -c "
from quanttool.reports import ReportFactory
from quanttool.reports.base import BacktestReportContext
from datetime import date

generator = ReportFactory.create('backtest')
context = BacktestReportContext(report_date=date.today(), backtest_id='test-run-001')
report = generator.generate(context)
print(report[:500])
"

# 4. 测试归因报告生成
python -c "
from quanttool.reports import ReportFactory
from quanttool.reports.base import AttributionReportContext
from datetime import date

generator = ReportFactory.create('attribution')
context = AttributionReportContext(report_date=date.today(), portfolio_id='test-portfolio')
report = generator.generate(context)
print(report[:500])
"
```

**验证清单**：
- [ ] `pytest tests/test_reports/ -v` 全部通过
- [ ] 日报生成正常（Markdown 格式）
- [ ] 回测报告生成正常
- [ ] 归因报告生成正常

---

## 6. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 导入路径变更导致外部调用失败 | 高 | 保留旧路径的 re-export，逐步迁移 |
| 评分计算结果不一致 | 高 | 迁移前记录测试用例，迁移后对比验证 |
| 数据获取失败 | 中 | 每阶段完成后运行完整测试 |
| 前端 API 调用失败 | 中 | 更新 API 文档，前端同步更新 |
| 循环导入问题 | 高 | 重构前分析依赖图，确保分层清晰（见 7.2） |
| 外部用户 API 兼容性 | 高 | 保留旧 API 端点并添加 deprecation warning，设置 2 版本过渡期 |
| 并发访问导致数据不一致 | 中 | 迁移期间使用 feature flag 控制功能切换 |
| 性能回归 | 中 | 迁移后运行性能基准测试对比 |

### 6.1 循环导入风险分析

重构前需分析以下潜在循环依赖：

```
factors/scoring/ → factors/tech_indicators.py → infrastructure/data_providers/
                              ↑
                              └── 可能的循环
```

**缓解措施**：
1. 使用依赖注入，而非直接导入
2. 将共享类型定义提取到 `domain/models/`
3. 使用 `TYPE_CHECKING` 进行类型提示的延迟导入

### 6.2 依赖管理

本次重构不新增外部依赖，仅重组现有代码。

需确保以下依赖关系正确：
- `factors/scoring/` 可依赖 `domain/models/`、`factors/tech_indicators.py`
- `infrastructure/data_providers/` 不应依赖 `factors/` 或 `application/`
- `reports/` 可依赖 `application/` 和 `infrastructure/`

---

## 7. 回滚策略

每个 Phase 完成后创建 Git 标签：
```bash
git tag -a refactor-phase1-done -m "Phase 1: 清理废弃代码完成"
git tag -a refactor-phase2-done -m "Phase 2: 数据提供者重构完成"
git tag -a refactor-phase3-done -m "Phase 3: 评分系统统一完成"
git tag -a refactor-phase4-done -m "Phase 4: 报告系统统一完成"
```

如遇严重问题，可回滚到上一阶段标签。

---

## 8. 测试迁移计划

### 8.1 现有测试

| 测试文件 | 需要更新 |
|----------|----------|
| `tests/test_score_enhancement.py` | 是，更新导入路径 |
| 其他测试文件 | 视导入路径变化而定 |

### 8.2 新增测试

| 测试文件 | 目的 |
|----------|------|
| `tests/test_scoring/test_strategies.py` | 测试各评分策略 |
| `tests/test_scoring/test_unified_scorer.py` | 测试统一评分器 |
| `tests/test_reports/test_generators.py` | 测试报告生成器 |
| `tests/test_reports/test_factory.py` | 测试报告工厂 |
| `tests/test_data_providers/test_realtime.py` | 测试实时数据提供者 |
| `tests/test_data_providers/test_incremental.py` | 测试增量数据提供者 |

### 8.3 测试覆盖率要求

每个 Phase 完成后，确保测试覆盖率不低于 80%：

```bash
pytest tests/ --cov=quanttool --cov-report=html
```

---

## 9. 验收标准

### 9.1 代码质量

- [ ] 删除所有废弃代码
- [ ] 数据提供者按职责分层，无重复类
- [ ] 评分系统使用策略模式，支持扩展
- [ ] 报告系统使用工厂模式，易于扩展
- [ ] 测试覆盖率 >= 80%

### 9.2 功能验证

- [ ] 所有测试通过 (`pytest tests/ -v`)
- [ ] API 端点正常响应
- [ ] 前端所有页面正常显示
- [ ] 无功能回退

### 9.3 文档更新

- [ ] API 文档更新
- [ ] README 更新（如需要）
- [ ] CLAUDE.md 更新（如需要）

### 9.4 性能验证

- [ ] 数据获取延迟不超过重构前的 110%
- [ ] 评分计算时间不超过重构前的 110%
- [ ] 内存使用不超过重构前的 120%

性能基准测试脚本：
```bash
# 数据获取性能测试
python -c "
import time
from quanttool.infrastructure.data_providers import AShareProvider

provider = AShareProvider()
start = time.time()
df = provider.get_daily_data('000001.SZ', start_date='2023-01-01')
elapsed = time.time() - start
print(f'数据获取耗时: {elapsed:.2f}s, 数据量: {len(df)} 条')
"

# 评分计算性能测试
python -c "
import time
from quanttool.factors.scoring import UnifiedScoringSystem
from quanttool.factors.scoring.strategies import TrendScoringStrategy
from quanttool.infrastructure.data_providers import AShareProvider

provider = AShareProvider()
df = provider.get_daily_data('000001.SZ', start_date='2023-01-01')

scorer = UnifiedScoringSystem([TrendScoringStrategy()])
start = time.time()
result = scorer.calculate_scores(df)
elapsed = time.time() - start
print(f'评分计算耗时: {elapsed:.4f}s')
"
```

---

## 10. 外部依赖影响分析

### 10.1 API 端点变更清单

| 端点 | 变更类型 | 影响范围 |
|------|----------|----------|
| `/api/stock/{symbol}/score` | 响应格式微调 | 前端 `scan/` 页面 |
| `/api/realtime/*` | 内部实现变更 | 前端 `monitor/` 页面 |
| `/api/reports/*` | 新增工厂接口 | CLI 命令 |

### 10.2 前端调用点清单

| 页面 | 调用模块 | 需要更新 |
|------|----------|----------|
| `/scan` | 评分 API | 可能需要调整响应解析 |
| `/picks` | 预测 API | 无需更新 |
| `/monitor` | 实时数据 API | 无需更新（内部变更） |

### 10.3 配置文件更新

| 文件 | 变更内容 |
|------|----------|
| 无 | 本次重构不涉及配置文件变更 |

---

## 11. 时间估算（含缓冲）

| 阶段 | 核心开发 | 缓冲(20%) | 总计 |
|------|----------|-----------|------|
| Phase 1 | 1 天 | 0.2 天 | 1.2 天 |
| Phase 2 | 2 天 | 0.4 天 | 2.4 天 |
| Phase 3 | 2 天 | 0.4 天 | 2.4 天 |
| Phase 4 | 1 天 | 0.2 天 | 1.2 天 |
| 代码审查 | 0.5 天 | - | 0.5 天 |
| 文档更新 | 0.5 天 | - | 0.5 天 |
| **总计** | **7 天** | **1.2 天** | **8.2 天** |
