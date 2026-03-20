# QuantTool 项目规范 (SPEC)

> 版本: 1.0.0
> 最后更新: 2026-03-20
> 状态: 活跃开发中

## 1. 项目概述

QuantTool 是一个专业的 A 股量化分析平台，提供股票分析、策略回测、机器学习选股等核心功能。

### 1.1 核心特性

- **股票分析**: K线图、技术指标、筹码分布、交易信号
- **策略回测**: 多种内置策略，支持自定义参数
- **ML 选股**: GBM 模型训练和预测
- **实时监控**: WebSocket 实时行情推送

### 1.2 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python 3.9+, FastAPI |
| 前端 | Next.js 14, React 18, Tailwind CSS |
| 数据库 | PostgreSQL (缓存), Parquet (历史数据) |
| 数据源 | AkShare, Tushare, Sina, EastMoney |
| ML | LightGBM, Qlib |

## 2. 项目结构

```
Quanter/
├── quanttool/                 # 主包
│   ├── application/           # 应用服务层
│   │   ├── backtest_service.py
│   │   ├── data_service.py
│   │   ├── factor_service.py
│   │   └── experiment_service.py
│   ├── domain/               # 领域模型
│   │   ├── interfaces/       # 接口定义
│   │   └── models/           # 数据模型
│   ├── infrastructure/       # 基础设施
│   │   ├── cache/           # 缓存系统
│   │   ├── calendar/        # 交易日历
│   │   ├── data_providers/  # 数据提供者
│   │   ├── database/        # 数据库
│   │   └── stores/          # 数据存储
│   ├── strategies/           # 交易策略
│   │   ├── ma_cross.py
│   │   ├── rsi.py
│   │   ├── macd.py
│   │   ├── bollinger.py
│   │   ├── kdj.py
│   │   ├── gbm_strategy.py
│   │   └── qlib/           # Qlib 集成
│   ├── factors/              # 因子计算
│   ├── backtest/             # 回测引擎
│   ├── web/                  # Web 应用
│   │   ├── api/             # API 路由
│   │   ├── frontend/        # Next.js 前端
│   │   └── static/          # 静态文件
│   ├── cli/                  # 命令行工具
│   └── ml/                   # 机器学习
├── tests/                    # 测试
├── models/                   # 模型存储
│   ├── qlib/
│   └── gbm/
├── docs/                     # 文档
└── examples/                 # 示例代码
```

## 3. 架构设计

### 3.1 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                        表现层 (Web/CLI)                       │
│  Next.js Frontend  │  FastAPI REST API  │  WebSocket        │
├─────────────────────────────────────────────────────────────┤
│                        应用层 (Application)                   │
│  BacktestService   │  DataService  │  FactorService         │
├─────────────────────────────────────────────────────────────┤
│                        领域层 (Domain)                        │
│  策略接口  │  因子接口  │  数据提供者接口  │  实体模型         │
├─────────────────────────────────────────────────────────────┤
│                        基础设施层 (Infrastructure)            │
│  数据源适配器  │  缓存系统  │  数据库  │  外部服务            │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 数据流

```
数据获取流程:
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 数据源    │ -> │ 数据适配器 │ -> │ 缓存层    │ -> │ 应用服务  │
│ (AkShare) │    │ (Fetcher) │    │ (Cache)  │    │ (Service) │
└──────────┘    └──────────┘    └──────────┘    └──────────┘

数据源优先级 (实时行情):
1. Pytdx (通达信直连) - 最快，秒级
2. Sina (新浪财经) - 备用，支持批量

数据源优先级 (历史数据):
1. Ashare (新浪+腾讯) - 主力
2. AkShare - 备用
3. Tushare - 备用 (需要 Token)
```

## 4. API 规范

### 4.1 基础规范

- 基础路径: `/api`
- 数据格式: JSON
- 编码: UTF-8
- 时间格式: ISO 8601 (`YYYY-MM-DD HH:mm:ss`)

### 4.2 核心 API 端点

#### 股票分析

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/stock/{symbol}/analysis` | 完整分析数据 |
| GET | `/stock/{symbol}/kline` | K 线数据 |
| GET | `/stock/{symbol}/signals` | 交易信号 |
| GET | `/stock/{symbol}/chip` | 筹码分布 |

#### 实时数据

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/realtime/quote/{symbol}` | 实时行情 |
| POST | `/realtime/batch` | 批量行情 |
| GET | `/realtime/search` | 搜索股票 |

#### 回测

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/backtest/run` | 运行回测 |
| GET | `/backtest/strategies` | 策略列表 |

#### ML 模型

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/gbm/train` | 训练模型 |
| POST | `/gbm/predict` | 模型预测 |
| GET | `/gbm/models` | 模型列表 |

### 4.3 响应格式

```json
// 成功响应
{
  "symbol": "600519",
  "name": "贵州茅台",
  "data": { ... }
}

// 错误响应
{
  "detail": "错误描述"
}
```

## 5. 组件规范

### 5.1 策略接口

```python
class IStrategy(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass

    @property
    def params(self) -> Dict[str, Any]:
        """策略参数"""
        return {}
```

### 5.2 数据提供者接口

```python
class IDataProvider(ABC):
    @abstractmethod
    def get_bars(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """获取 K 线数据"""
        pass

    @abstractmethod
    def get_realtime_quote(self, symbol: str) -> Dict[str, Any]:
        """获取实时行情"""
        pass
```

### 5.3 因子接口

```python
class IFactor(ABC):
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """因子名称"""
        pass
```

## 6. 前端规范

### 6.1 目录结构

```
frontend/
├── app/                     # 页面路由
│   ├── page.tsx            # 首页
│   ├── analyze/            # 股票分析
│   ├── backtest/           # 策略回测
│   ├── model/              # ML 模型
│   ├── monitor/            # 实时监控
│   ├── scan/               # 智能选股
│   └── picks/              # 智能荐股
├── components/              # 组件
│   ├── ui/                 # 基础 UI 组件
│   ├── charts/             # 图表组件
│   ├── stock/              # 股票相关组件
│   ├── backtest/           # 回测组件
│   ├── model/              # 模型组件
│   └── layout/             # 布局组件
├── lib/                    # 工具库
│   ├── api/               # API 客户端
│   ├── utils.ts           # 工具函数
│   └── constants.ts       # 常量
├── stores/                 # 状态管理 (Zustand)
├── hooks/                  # 自定义 Hooks
└── types/                  # TypeScript 类型
```

### 6.2 命名规范

- 组件: PascalCase (`StockCard.tsx`)
- 函数: camelCase (`fetchStockData`)
- 常量: UPPER_SNAKE_CASE (`DEFAULT_BACKTEST_PARAMS`)
- 类型: PascalCase (`StockAnalysis`)

### 6.3 状态管理

使用 Zustand 进行状态管理:

```typescript
// stores/useAppStore.ts
interface AppState {
  activePage: string;
  setActivePage: (page: string) => void;
  theme: 'dark' | 'light';
  toggleTheme: () => void;
}
```

## 7. 数据规范

### 7.1 股票代码格式

- 标准: `600519.SH` (带交易所后缀)
- 简化: `600519` (无后缀)
- AkShare: `600519` (无后缀)

### 7.2 K 线数据格式

```typescript
interface KlineData {
  date: string;      // YYYY-MM-DD
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  amount?: number;
}
```

### 7.3 技术指标格式

```typescript
interface TechnicalIndicators {
  macd?: {
    dif: number[];
    dea: number[];
    macd: number[];
  };
  kdj?: {
    k: number[];
    d: number[];
    j: number[];
  };
  rsi?: {
    rsi6: number[];
    rsi12: number[];
    rsi24: number[];
  };
}
```

## 8. 性能规范

### 8.1 数据获取优化

- 重试次数: 1 (减少延迟)
- 缓存 TTL:
  - 实时行情: 3 秒
  - 分钟 K 线: 60 秒
  - 日线数据: 1 天

### 8.2 前端优化

- 使用 Next.js 静态生成
- 图表组件按需加载
- API 请求防抖处理

## 9. 错误处理

### 9.1 后端错误码

| HTTP 状态码 | 含义 |
|------------|------|
| 400 | 请求参数错误 |
| 404 | 资不存在 |
| 500 | 服务器内部错误 |

### 9.2 日志规范

```python
logger.info(f"Processing request for {symbol}")
logger.warning(f"Data provider fallback: {provider}")
logger.error(f"Failed to fetch data: {error}")
```

## 10. 安全规范

### 10.1 API 安全

- 输入验证: 使用 Pydantic 模型
- SQL 注入防护: 使用参数化查询
- XSS 防护: 前端自动转义

### 10.2 数据安全

- 敏感配置: 环境变量
- API Token: 不提交到版本控制

## 11. 测试规范

### 11.1 测试类型

- 单元测试: 测试单个函数/类
- 集成测试: 测试 API 端点
- E2E 测试: 测试关键用户流程

### 11.2 测试覆盖率

- 目标: 80%+
- 关键模块: 100%

## 12. 部署规范

### 12.1 环境变量

```bash
# 数据库
DATABASE_URL=postgresql://...

# Tushare (可选)
TUSHARE_TOKEN=xxx

# EastMoney (可选)
EASTMONEY_COOKIE=xxx
```

### 12.2 启动命令

```bash
# 后端
uvicorn quanttool.web.app:app --host 0.0.0.0 --port 8000

# 前端开发
cd quanttool/web/frontend && npm run dev

# 前端生产
cd quanttool/web/frontend && npm run build && npm start
```

## 13. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0.0 | 2026-03-20 | 初始版本 |

---

本文档是 QuantTool 项目的权威规范，所有开发工作应遵循此文档。
如有变更，需更新版本号和变更记录。
