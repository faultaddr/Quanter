# QuantTool - A股量化交易分析平台

[English](./README_EN.md) | [繁體中文](./README_ZH_TW.md)

QuantTool 是一个专业的 A 股量化交易分析平台，提供技术分析、因子研究、策略回测、风险控制等核心功能。

## 核心特性

- **免费数据源**：优先使用 Ashare、EastMoney、AkShare 等免费数据源
- **实时数据**：支持分钟级实时行情获取
- **Web 界面**：现代化的 Web 前端，支持全功能操作
- **策略回测**：支持多种技术指标策略回测，真实模拟 A 股交易规则
- **因子研究**：IC/IR 分析、因子优化、因子中性化
- **风控管理**：行业暴露监控、黑名单检查、仓位收缩

## 核心功能

### 1. 多维度技术分析评分系统

基于三大类因子的分层评分架构：

```
最终评分 = 趋势得分 × 位置修正系数
```

#### 趋势因子（权重 40%）
| 因子 | 权重 | 说明 |
|------|------|------|
| 趋势强度 | 20% | MA20 乖离率，DMI 状态修正 |
| 均线斜率 | 20% | MA5 斜率判断趋势方向 |
| MACD 动量 | 20% | MACD 柱状图变化 |
| 资金流向 | 20% | OBV 资金流评分 |
| 成交量 | 10% | 量价配合度 |
| K 线形态 | 10% | 位置敏感评分 |

### 2. K 线形态识别系统

支持识别 20+ 种经典 K 线形态，包括锤子线、吞没形态、晨星、暮星等。

### 3. 因子研究与优化

| 功能 | 说明 |
|------|------|
| 因子有效性检验 | IC/IR 分析，评估因子预测能力 |
| 因子权重优化 | IR 加权、IC 加权、等权、风险平价 |
| 因子中性化 | 行业中性、市值中性 |
| 因子流水线处理 | Winsorize、Standardize 处理 |

### 4. 组合风控管理

| 功能 | 说明 |
|------|------|
| 行业暴露监控 | 单行业仓位限制（默认 20%） |
| 黑名单检查 | 禁止持仓股票监控 |
| 仓位收缩 | 基于回撤动态调整仓位 |
| 风险评分 | 多维度风险评估 |

### 5. A 股交易约束

| 约束类型 | 说明 |
|----------|------|
| 涨跌停限制 | 涨停不能买入，跌停不能卖出 |
| ST 股限制 | 可配置排除 ST 股票 |
| 真实交易成本 | 佣金、印花税、滑点模拟 |

### 6. 内置交易策略

| 策略名称 | 类型 | 说明 |
|----------|------|------|
| `ma_cross` | 趋势跟踪 | 均线交叉策略 |
| `dual_ma` | 趋势跟踪 | 双均线策略 |
| `breakout` | 突破策略 | 价格突破 N 日高低点 |
| `turtle` | 趋势跟踪 | 海龟交易策略 |
| `ma_alignment` | 趋势跟踪 | 均线多头排列策略 |
| `rsi` | 震荡指标 | RSI 超买超卖策略 |
| `macd` | 趋势指标 | MACD 金叉死叉策略 |
| `kdj` | 震荡指标 | KDJ 金叉死叉策略 |
| `bollinger` | 震荡指标 | 布林带回归策略 |

## 技术架构

```
QuantTool/
├── quanttool/
│   ├── application/             # 应用服务层
│   ├── backtest/                # 回测引擎
│   ├── config/                  # 配置管理
│   ├── core/                    # 错误、日志、注册表
│   ├── domain/                  # 接口与领域模型
│   ├── factors/                 # 因子库
│   │   ├── scoring/             # 统一评分系统
│   │   └── technical/           # 技术因子
│   ├── infrastructure/          # 基础设施层
│   │   ├── cache/
│   │   ├── data_providers/
│   │   │   ├── historical/
│   │   │   ├── incremental/
│   │   │   └── realtime/
│   │   ├── database/
│   │   ├── notifiers/
│   │   └── stores/
│   ├── ml/                      # 模型训练与特征
│   ├── optimization/            # 优化器
│   ├── reports/                 # 报告系统
│   │   └── generators/
│   ├── risk/                    # 风险管理
│   ├── strategies/              # 交易策略
│   ├── validation/              # 评分/因子验证
│   ├── web/                     # Web 层
│   │   ├── api/                 # FastAPI 路由
│   │   ├── frontend/            # Next.js 前端
│   │   └── schemas/             # API 请求/响应模型
│   └── cli/                     # 命令行工具
└── tests/                       # smoke tests 与后续单元测试
```

## 安装

### 前置要求

- Python 3.9+
- Node.js 18+
- npm 或 yarn

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/faultaddr/Quanter.git
cd Quanter

# 安装 Python 依赖
python3 -m pip install -e .

# 安装前端依赖
cd quanttool/web/frontend
npm install
```

## 快速开始

### 启动服务

```bash
# 启动后端服务
uvicorn quanttool.web.app:app --host 0.0.0.0 --port 8000

# 启动前端开发服务器
cd quanttool/web/frontend
npm run dev
```

访问 http://localhost:3000 打开 Web 界面。

### 本地验证

```bash
# 后端 smoke tests
.venv-mcp/bin/python -m unittest discover -s tests -v

# Python 语法编译检查
.venv-mcp/bin/python -m compileall -q quanttool

# CLI 入口检查
.venv-mcp/bin/python -m quanttool --help

# 前端 lint
cd quanttool/web/frontend
npm run lint
```

### 使用 Web 界面

| 页面 | 功能 |
|------|------|
| `/` | 盘面概览、市场指数 |
| `/analyze` | 股票分析、K线、技术指标 |
| `/backtest` | 策略回测、收益对比 |
| `/factors` | 因子研究、IC/IR 分析 |
| `/risk` | 组合风控、风险检查 |
| `/scan` | 智能选股、条件筛选 |
| `/picks` | AI 推荐股票 |
| `/monitor` | 实时行情监控 |
| `/model` | ML 模型训练预测 |

### 使用 CLI

```bash
# 分析股票
quant analyze 600519 --days 360

# 回测策略
quant backtest run --strategy ma_cross --symbol 600519 \
  --start 2023-01-01 --end 2024-01-01 --cash 100000
```

### 使用 Python API

```python
from quanttool.factors.stock_analyzer import StockAnalyzer
from quanttool.backtest.engine import BacktestEngine

# 分析股票
analyzer = StockAnalyzer()
report = analyzer.analyze_stock("600519", days=360)
print(report.summary)

# 回测策略
engine = BacktestEngine()
result = engine.run(
    symbol="600519",
    strategy="ma_cross",
    start_date="2023-01-01",
    end_date="2024-01-01",
    initial_capital=1000000,
)
print(f"收益率: {result.total_return:.2%}")
```

## 数据源优先级

1. **Ashare** - 免费、无需 Token，主力数据源
2. **EastMoney** - 免费、数据丰富
3. **AkShare** - 免费、接口丰富
4. **TuShare** - 需要 Token，作为备选

## 性能指标

| 操作 | P50 | P95 |
|------|-----|-----|
| 缓存命中 | < 10ms | < 50ms |
| 数据获取 | < 500ms | < 2s |
| 完整分析 | < 2s | < 5s |

## 测试

```bash
# 当前最小测试基线（无需 pytest）
python3 -m unittest discover -s tests -v

# 如果安装了 dev 依赖，也可以使用 pytest
python3 -m pytest tests/ -v

# 覆盖率测试
pytest tests/ --cov=quanttool --cov-report=html
```

当前仓库提供 smoke tests 作为基础质量闸门，后续功能应继续补充 pytest 单元测试和 Playwright E2E 测试。

## 运行产物

以下目录和文件属于本地运行或构建产物，不应提交到版本库：

- `reports/`、`.cache/`、`quanttool.log`
- `models/`、`mlruns/`、`qlib_data/`
- `quanttool/web/frontend/.next/`、`node_modules/`、`*.tsbuildinfo`

## 许可证

MIT License
