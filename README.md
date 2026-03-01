# QuantTool - A股量化交易分析平台

QuantTool 是一个专为A股市场设计的量化分析平台，提供技术分析、形态识别、多因子评分、风险熔断等核心功能。

## 核心功能

### 1. 多维度技术分析评分系统

基于三大类因子的分层评分架构：

```
最终评分 = 趋势得分 × 位置修正系数
```

#### 趋势因子（权重40%）
| 因子 | 权重 | 说明 |
|------|------|------|
| 趋势强度 | 20% | MA20乖离率，DMI状态修正 |
| 均线斜率 | 20% | MA5斜率判断趋势方向 |
| MACD动量 | 20% | MACD柱状图变化 |
| 资金流向 | 20% | OBV资金流评分 |
| 成交量 | 10% | 量价配合度 |
| K线形态 | 10% | 位置敏感评分 |

#### 动能因子（权重35%）
| 因子 | 权重 | 说明 |
|------|------|------|
| KDJ位置 | 30% | KDJ超买超卖判断 |
| RSI强度 | 35% | RSI相对强弱 |
| MTM动量 | 20% | 动量指标 |
| ROC变动率 | 15% | 价格变动速率 |

#### 资金因子（权重25%）
| 因子 | 权重 | 说明 |
|------|------|------|
| OBV流向 | 40% | 能量潮指标 |
| MFI强度 | 35% | 资金流量指标 |
| 量价关系 | 25% | 成交量与价格配合 |

### 2. K线形态识别系统

支持识别20+种经典K线形态：

#### 单根形态
- **看涨形态**：锤子线、倒锤子、大阳线
- **看跌形态**：吊颈线、流星线、大阴线
- **中性形态**：十字星、长上影线、长下影线、纺锤线

#### 组合形态
- **看涨组合**：看涨吞没、晨星、穿刺线
- **看跌组合**：看跌吞没、暮星、乌云盖顶

#### 形态可视化
- ASCII艺术K线图（彩色显示）
- 形态示意图解说明

```
#### 📖 「锤子线」形态说明

    ┌─────────────────┐
    │      │          │  上影线短或无
    │    ┌───┐        │
    │    │   │ 小实体 │  实体在上方
    │    └───┘        │
    │       │         │
    │       │         │  长下影线
    │       │         │  (>=实体2倍)
    └─────────────────┘
    出现在低位 = 看涨反转信号
```

### 3. 智能熔断机制

多层风险控制体系：

| 熔断规则 | 触发条件 | 效果 |
|----------|----------|------|
| 强力看跌熔断 | 大阴线+高位 | 评分降至≤25分，仓位0% |
| 高位看跌熔断 | 高位+看跌形态 | 评分降至≤25分，仓位0% |
| 诱多陷阱熔断 | 高位+看涨+极端超买 | 强制观望 |
| 极端超买熔断 | 3+指标同时爆表 | 位置系数≤0.30 |

### 4. 位置修正系数

根据趋势方向动态调整风险：

| 状态 | 位置系数 | 触发条件 |
|------|----------|----------|
| 安全区 | 1.00 | 布林中下轨，RSI 30-50 |
| 适中区 | 0.75~0.95 | 布林中上轨，RSI 50-65 |
| 警戒区 | 0.50~0.75 | 布林上轨附近，RSI 65-75 |
| 危险区 | <0.50 | 极端超买，多指标爆表 |

#### 趋势敏感修正
- **下跌趋势+超卖**：系数0.50（接飞刀风险）
- **下跌趋势+阻力位**：系数0.45（反弹遇阻）
- **上升趋势+超买**：系数0.75（追高风险）
- **长期高位（>70%分位）**：系数0.65-0.80

### 5. 四部分报告架构

```
第一部分：核心结论区
├── 操作指令（买入/卖出/观望）
├── 技术评分（0-100分）
├── 置信度评估
└── 关键理由

第二部分：多维信号共振分析
├── 趋势状态（DMI+均线排列）
├── 动能状态（MACD+RSI）
├── 位置状态（布林带+CCI+WR）
├── 形态特权区（K线形态定性）
└── K线可视化图表

第三部分：量化评分与因子拆解
├── 综合评分计算公式
├── 各因子得分明细
├── 位置修正系数
└── 红黑榜

第四部分：交易执行计划
├── 策略类型判定
├── 入场/止损/目标位
├── 仓位建议
└── 风险提示
```

### 6. 技术指标计算

内置30+种技术指标：

| 类别 | 指标 |
|------|------|
| 趋势类 | MA、EMA、MACD、DMI、BOLL |
| 动量类 | RSI、KDJ、MTM、ROC、CCI |
| 成交量 | OBV、VOL、MFI、VWAP |
| 波动率 | ATR、BIAS、WR |

## 安装

### 从源码安装

```bash
git clone https://github.com/yourusername/quanttool.git
cd quanttool
pip install -e .
```

### 开发模式安装

```bash
pip install -e ".[dev]"
```

## 快速开始

### 1. 配置数据源

```bash
# TuShare配置
export TUSHARE_TOKEN="your_tushare_token"

# 或创建.env文件
echo "TUSHARE_TOKEN=your_tushare_token" > .env
```

### 2. 分析单只股票

```bash
# 基本分析
quant analyze 000001.SZ

# 指定分析周期
quant analyze 000001.SZ --days 360

# 保存报告到文件
quant analyze 000001.SZ --days 360 --output report.md
```

### 3. 市场扫描

```bash
# 扫描全市场
quant analyze scan --market all --days 360 --top 10

# 扫描特定市场
quant analyze scan --market sh --days 180 --top 20
```

### 4. 回测策略

```bash
quant backtest run \
  --strategy ma_cross \
  --symbol 000001.SZ \
  --start 2023-01-01 \
  --end 2023-06-01 \
  --cash 100000
```

## 项目架构

```
quanttool/
├── factors/                    # 因子分析模块
│   ├── scoring_system.py       # 多维度评分系统
│   ├── candlestick_patterns.py # K线形态识别
│   ├── stock_analyzer.py       # 股票综合分析
│   ├── tech_indicators.py      # 技术指标计算
│   └── trading_strategies.py   # 交易策略
├── application/                # 应用服务层
│   ├── analysis_service.py     # 分析服务
│   ├── backtest_service.py     # 回测服务
│   ├── signal_service.py       # 信号服务
│   └── portfolio_backtest_service.py
├── infrastructure/             # 基础设施层
│   ├── data_providers/         # 数据提供者
│   │   ├── tushare_provider.py
│   │   ├── ashare_provider.py
│   │   └── data_fetcher.py
│   ├── stores/                 # 存储层
│   ├── scheduler/              # 任务调度
│   └── notification/           # 通知服务
├── reports/                    # 报告生成
│   ├── generators.py
│   ├── daily_report_generator.py
│   └── templates/
├── cli/                        # 命令行接口
│   ├── main.py
│   └── commands/
├── web/                        # Web API
│   ├── app.py
│   ├── api/routes.py
│   └── websockets.py
└── ml/                         # 机器学习模块
    ├── models.py
    ├── trainer.py
    └── features.py
```

## 使用示例

### Python API

```python
from quanttool.factors.stock_analyzer import StockAnalyzer
from quanttool.factors.scoring_system import ScoringSystem

# 创建分析器
analyzer = StockAnalyzer()

# 分析股票
report = analyzer.analyze_stock("000001.SZ", days=360)
print(report)

# 获取评分结果
scoring = ScoringSystem()
result = scoring.calculate_score("000001.SZ", df)
print(f"综合评分: {result['score']}")
print(f"趋势方向: {result['trend_direction']}")
print(f"位置系数: {result['position_modifier']}")
print(f"操作建议: {result['execution']['action_guide']}")
```

### K线形态识别

```python
from quanttool.factors.candlestick_patterns import (
    CandlestickPatternRecognizer,
    draw_candlestick_chart,
    draw_pattern_illustration
)

# 识别形态
recognizer = CandlestickPatternRecognizer()
patterns = recognizer.recognize_all_patterns(df, lookback=5)

# 绘制K线图
chart = draw_candlestick_chart(df, num_candles=10)
print(chart)

# 获取形态示意图
illustration = draw_pattern_illustration("锤子线")
print(illustration)
```

## 报告示例

```markdown
## 第一部分：核心结论

### 🟢 操作指令：买入

**技术评分：70.5分（良好）**

**评分构成：** 趋势分 70.5 × 位置系数 1.00 = 70.5

**置信度：中高**（多数因子同向）

### 💡 关键理由

入场位置安全，趋势强势确立

---

## 第二部分：多维信号共振分析

### 📊 趋势状态

- **状态**：上升趋势（强）
- **说明**：均线多头排列+DMI多头占优，ADX=27.18

### 🕯️ 形态特权区

- **形态**：长上影线（强度：中）
- **定性影响**：➖ 中性信号 - 上方抛压

#### 📊 近期K线图

```
最高: ¥63.77
   │        │    │  ████ │     │
   │        │    │  ████ │     │
   │   │    ████ │  ████ ▓▓▓▓  │
   │   │    ████ │  ████ ▓▓▓▓  │
   │   │    ████ │       ▓▓▓▓  │
   │   │         │             │
最低: ¥60.27
```
> 图例：🟢 绿色 = 阳线（涨） | 🔴 红色 = 阴线（跌） | │ = 影线
```

## 数据源

| 数据源 | 类型 | 配置 | 用途 |
|--------|------|------|------|
| TuShare | 历史 | TUSHARE_TOKEN | 回测、分析 |
| AShare | 实时 | ASHARE_ENDPOINT | 实时信号 |
| CSV Mock | 模拟 | 数据目录 | 测试开发 |

## 内置策略

### MA Cross（均线交叉）
```python
strategy_params = {
    "short_window": 5,
    "long_window": 10
}
```

### Breakout（突破策略）
```python
strategy_params = {
    "lookback_period": 20,
    "entry_threshold": 0.02
}
```

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v

# 运行测试覆盖率
pytest --cov=quanttool tests/
```

## 许可证

MIT License