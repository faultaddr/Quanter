# QuantTool 项目规范

## 项目概述

QuantTool 是一个专业 A 股量化交易平台，提供股票分析、策略回测、因子挖掘、实时监控、智能荐股等功能。

### 核心特性
- **免费数据源**：优先使用免费数据源（Ashare、EastMoney、AkShare），无需 API Token
- **实时数据**：支持分钟级实时行情获取
- **策略回测**：支持多种技术指标策略回测
- **智能选股**：基于技术指标和量化因子的智能选股
- **ML模型**：支持 GBM 模型训练和预测

## 架构设计

### 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Presentation Layer                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Web API   │  │  Frontend   │  │       CLI           │  │
│  │  (FastAPI)  │  │  (Next.js)  │  │      (Typer)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     Application Layer                        │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │ Analysis Svc  │  │ Backtest Svc  │  │  Monitor Svc  │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                       Domain Layer                           │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │   Models      │  │  Interfaces   │  │   Strategies  │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                       │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │Data Providers │  │   Database    │  │     Cache     │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 目录结构

```
quanttool/
├── core/                    # 核心功能
│   ├── errors.py           # 错误处理
│   ├── logging.py          # 日志
│   └── registry.py         # 组件注册
│
├── domain/                  # 领域层
│   ├── interfaces/         # 接口定义
│   │   └── data_provider.py
│   └── models/             # 数据模型
│       ├── __init__.py
│       └── ...
│
├── application/             # 应用服务层
│   ├── analysis_service.py
│   ├── backtest_service.py
│   ├── signal_service.py
│   ├── factor_service.py
│   └── ...
│
├── infrastructure/          # 基础设施层
│   ├── data_providers/     # 数据提供者
│   │   ├── ashare_provider.py
│   │   ├── akshare_minute_provider.py
│   │   ├── data_fetcher.py
│   │   └── ...
│   ├── database/           # 数据库
│   │   ├── connection.py
│   │   └── config.py
│   ├── cache/              # 缓存
│   │   └── local_cache.py
│   └── stores/             # 存储
│       ├── parquet_store.py
│       └── meta_db.py
│
├── strategies/              # 交易策略
│   ├── ma_cross.py
│   ├── breakout.py
│   ├── rsi.py
│   ├── bollinger.py
│   ├── qlib_strategy.py
│   └── ...
│
├── factors/                 # 因子库
│   ├── technical/          # 技术因子
│   ├── fundamental/        # 基本面因子
│   ├── stock_analyzer.py
│   ├── screening.py
│   └── ...
│
├── web/                     # Web 层
│   ├── api/                # API 路由
│   │   └── routes.py
│   ├── frontend/           # 前端代码
│   │   ├── app/
│   │   ├── components/
│   │   └── lib/
│   └── schemas/            # 请求/响应模型
│
├── cli/                     # 命令行工具
│   └── main.py
│
└── agent/                   # AI Agent
    └── server.py
```

## API 规范

### RESTful API 设计

#### 股票分析端点

```
GET /api/stock/{symbol}/analysis    # 获取完整分析数据
GET /api/stock/{symbol}/kline       # 获取 K 线数据
GET /api/stock/{symbol}/signals     # 获取交易信号
GET /api/stock/{symbol}/chip        # 获取筹码分布
GET /api/stock/{symbol}/flow        # 获取资金流向
GET /api/stock/{symbol}/risk        # 获取风险评估
```

#### 策略回测端点

```
POST /api/backtest/run              # 运行回测
GET  /api/backtest/runs/{run_id}    # 获取回测结果
GET  /api/strategies                 # 获取策略列表
```

#### 实时数据端点

```
GET  /api/realtime/search           # 搜索股票
GET  /api/realtime/quote/{symbol}   # 获取实时行情
WS   /ws/realtime                   # WebSocket 实时推送
```

#### 模型管理端点

```
POST /api/gbm/train                 # 训练模型
GET  /api/gbm/models                # 获取模型列表
GET  /api/gbm/predict/{model_id}    # 使用模型预测
```

### 响应格式

```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### 错误处理

```json
{
  "success": false,
  "data": null,
  "error": "错误信息",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## 数据提供者优先级

### 优先级顺序

1. **Ashare**（最高优先级）
   - 免费、无需 Token
   - 双核心数据源：新浪(主力) + 腾讯(备用)
   - 支持日线、周线、月线、分钟线

2. **EastMoney**
   - 免费、数据丰富
   - 支持实时行情

3. **AkShare**
   - 免费、接口丰富
   - 支持分钟级数据

4. **TuShare**
   - 需要 Token
   - 数据质量高

5. **BaoStock**（最低优先级）
   - 免费、稳定
   - 作为最后备选

### 性能要求

| 操作 | P50 | P95 | P99 |
|------|-----|-----|-----|
| 缓存命中 | < 10ms | < 50ms | < 100ms |
| 数据获取 | < 500ms | < 2s | < 5s |
| 完整分析 | < 2s | < 5s | < 10s |

### 重试策略

- **max_retries**: 1（最小化延迟）
- **base_delay**: 0.5s
- **max_delay**: 2s
- **timeout**: 10s

## 前端页面规范

### 页面列表

| 路由 | 页面名称 | 功能描述 |
|------|---------|---------|
| `/` | 盘面概览 | 市场指数、快速入口 |
| `/analyze` | 股票分析 | K线、技术指标、筹码、信号 |
| `/backtest` | 策略回测 | 回测配置、结果展示 |
| `/model` | ML模型 | 模型训练、预测、管理 |
| `/monitor` | 实时监控 | WebSocket 实时行情 |
| `/scan` | 智能选股 | 条件筛选、因子打分 |
| `/picks` | 智能荐股 | ML 模型推荐 |

### 股票分析页面功能模块

1. **实时行情卡片**
   - 当前价格、涨跌幅
   - 开高低收、成交量
   - 换手率、市盈率

2. **K 线图表**
   - 日K、周K、月K 切换
   - MA5/MA10/MA20/MA60 均线
   - 缩放、十字光标

3. **技术指标面板**
   - MACD（DIF、DEA、MACD柱）
   - KDJ（K、D、J 线）
   - RSI（相对强弱指数）
   - BOLL（布林带）

4. **筹码分布图**
   - 成本分布直方图
   - 获利比例
   - 平均成本

5. **资金流向**
   - 主力净流入
   - 散户流向
   - 大单、中单、小单

6. **回测对比**
   - 策略收益曲线
   - 基准对比
   - 收益统计

7. **风险评估**
   - 波动率
   - 最大回撤
   - 夏普比率

## 编码规范

### Python

- 遵循 PEP 8
- 使用类型注解
- 文档字符串使用 Google 风格
- 单文件不超过 800 行

### TypeScript/React

- 使用函数组件 + Hooks
- 遵循 Next.js App Router 规范
- 组件文件不超过 300 行
- 使用 Tailwind CSS

## 测试要求

### 单元测试
- 覆盖率 >= 80%
- 使用 pytest

### E2E 测试
- 使用 Playwright
- 覆盖核心用户流程

### 性能测试
- 使用 pytest-benchmark
- 定期运行基准测试

## 部署规范

### 环境变量

```bash
# 数据库
DATABASE_URL=postgresql://...

# 数据源 Token（可选）
TUSHARE_TOKEN=...

# 服务配置
API_HOST=0.0.0.0
API_PORT=8000
```

### 启动命令

```bash
# 后端
uvicorn quanttool.web.app:app --host 0.0.0.0 --port 8000

# 前端
cd quanttool/web/frontend && npm run build && npm run start
```

## 版本控制

### 分支策略

- `master` - 生产环境
- `feature/*` - 功能开发
- `hotfix/*` - 紧急修复

### 提交信息格式

```
<type>: <description>

[optional body]

Co-Authored-By: Codex Opus 4.6 <noreply@anthropic.com>
```

类型：feat, fix, refactor, docs, test, chore, perf, ci
