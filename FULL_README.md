# A股市场量化分析系统

这是一个全面的A股市场量化分析工具，包含股票筛选、策略分析和信号通知功能。

## 功能特性

### 1. 股票筛选器
- 自动筛选符合特定条件的优质股票
- 基于价格、成交量、波动率等多个维度进行筛选
- 计算股票潜力分数，帮助识别机会

### 2. 策略工具集
- **移动平均线交叉策略**: 短期均线上穿/下穿长期均线时产生买卖信号
- **RSI策略**: 基于相对强弱指标的超买超卖信号
- **MACD策略**: 基于MACD线和信号线的交叉信号
- **布林带策略**: 基于价格触及布林带上下轨的信号
- **均值回归策略**: 当价格显著偏离均值时的反转信号
- **突破策略**: 基于价格突破近期高低点的信号

### 3. 信号通知系统
- 实时生成买卖信号通知
- 支持邮件、短信、Telegram等多种通知方式
- 信号优先级管理
- 历史信号查询

### 4. 可视化仪表板
- 交互式股票筛选界面
- 策略分析图表
- 实时信号监控
- 技术指标可视化

## 安装指南

```bash
# 克隆项目
git clone <repository-url>
cd quant_trade_a_share

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

## 使用方法

### 方式一：运行完整仪表板
```bash
cd quant_trade_a_share
python run_app.py
```
然后选择选项1启动Web仪表板，通过浏览器访问 `http://127.0.0.1:8050`

### 方式二：命令行模式
```bash
# 筛选股票
python -m quant_trade_a_share.main_app --mode screen

# 分析特定股票
python -m quant_trade_a_share.main_app --mode analyze --symbol sh600519 --strategy ma_crossover

# 查看信号摘要
python -m quant_trade_a_share.main_app --mode summary

# 运行仪表板
python -m quant_trade_a_share.main_app --mode dashboard
```

### 方式三：编程接口
```python
from quant_trade_a_share.main_app import AShareAnalyzer

# 初始化分析器
analyzer = AShareAnalyzer()

# 筛选股票
filters = {
    'min_price': 10,
    'max_price': 150,
    'min_volume': 5000000,
    'days_back': 60,
    'min_return': 0.02,
    'max_volatility': 0.04
}
results = analyzer.screen_stocks(filters)

# 分析股票
analysis = analyzer.analyze_stock('sh600519', 'ma_crossover')

# 查看信号摘要
analyzer.generate_signals_summary()
```

## 组件说明

- `screeners/stock_screener.py`: 股票筛选器
- `strategies/strategy_tools.py`: 策略工具集
- `signals/signal_notifier.py`: 信号通知系统
- `viz/dashboard.py`: 可视化仪表板
- `main_app.py`: 主应用程序集成
- `run_app.py`: 用户入口脚本

## 注意事项

1. **数据源**: 系统会尝试从网络获取实时数据，如无法连接则自动生成模拟数据进行演示
2. **网络连接**: 部分功能需要网络连接以获取最新市场数据
3. **风险提示**: 量化策略仅供参考，实际投资需谨慎
4. **参数调整**: 可根据市场情况调整策略参数以获得更好的效果

## 扩展功能

- 添加新的技术指标
- 集成更多数据源
- 实现机器学习预测模型
- 添加风险管理模块
- 集成实盘交易接口（需合规性考虑）

## 许可证

MIT License