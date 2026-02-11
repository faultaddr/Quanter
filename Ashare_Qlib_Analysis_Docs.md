# Ashare-Qlib 综合分析指南

本指南介绍如何使用 Ashare 数据源进行 Qlib 增强分析，以当日之前 180 天的数据作为输入。

## 功能概述

Ashare-Qlib 综合分析模块提供了一套完整的量化分析工具，包括：

1. **数据获取**：从 Ashare 数据源获取高质量的 A 股历史数据
2. **因子分析**：结合 Qlib Alpha 因子和 MyTT 技术指标
3. **模型融合**：集成传统技术指标与机器学习模型
4. **风险管理**：基于 Qlib 的风险模型和投资组合优化
5. **自动调参**：多种参数优化算法

## 主要特性

- **使用 Ashare 数据**：直接从腾讯/新浪获取实时 A 股数据
- **180 天数据窗口**：使用当日之前 180 天的历史数据进行分析
- **综合分析功能**：一次性执行多项分析任务
- **智能信号生成**：自适应生成交易信号
- **风险控制**：内置风险管理机制

## 使用方法

### 1. 基本使用

```bash
python ashare_qlib_analyzer.py --symbol sh600023 --days 180 --analysis-type all
```

参数说明：
- `--symbol`: 股票代码（必需），例如 sh600023
- `--days`: 回溯天数（可选，默认 180）
- `--analysis-type`: 分析类型（可选）：
  - `comprehensive`: 综合性分析
  - `factor`: 高级因子分析
  - `signal`: 自适应信号生成
  - `portfolio`: 智能投资组合优化
  - `all`: 完整分析（默认）

### 2. 编程接口使用

```python
from quant_trade_a_share.integration.ashare_qlib_integration import AshareQlibIntegration

# 创建集成实例
integration = AshareQlibIntegration()

# 运行完整分析
results = integration.run_all_analysis_with_ashare("sh600023", days=180)

# 运行单项分析
comp_results = integration.run_comprehensive_qlib_analysis_with_ashare("sh600023", days=180)
factor_results = integration.run_advanced_factor_analysis_with_ashare("sh600023", days=180)
signal_results = integration.run_adaptive_signal_generation_with_ashare("sh600023", days=180)
```

### 3. 运行示例

运行演示脚本：
```bash
python demo_ashare_qlib_analysis.py
```

## 内部工作流程

1. **数据获取阶段**：
   - 使用 `AshareDataFetcher` 从腾讯/新浪获取指定股票的 180 天历史数据
   - 标准化数据格式，确保列名符合 Qlib 要求
   - 数据清洗和预处理

2. **数据分析阶段**：
   - 因子库扩充：结合 Qlib Alpha 因子和 MyTT 技术指标
   - 模型融合：传统技术指标与 ML 模型的结果融合
   - 风险管理：基于 Qlib 风险模型的投资组合风险分析
   - 自动调参：使用多种算法优化策略参数

3. **结果生成阶段**：
   - 综合分析报告
   - 因子有效性分析
   - 交易信号生成
   - 投资组合优化建议

## 优势

1. **数据质量高**：直接从 Ashare 数据源获取实时数据，避免数据延迟或不一致
2. **分析全面**：一次运行涵盖因子分析、模型融合、风险管理和参数优化
3. **适应性强**：支持多种分析类型和自定义参数
4. **风险控制**：内置风险管理机制，确保策略的稳健性
5. **易于扩展**：模块化设计，便于添加新的分析功能

## 注意事项

- 确保网络连接正常以便从 Ashare 数据源获取数据
- 180 天数据窗口可根据需求调整
- 如遇到数据获取问题，系统会自动尝试备用数据源
- 建议在生产环境中设置适当的数据缓存机制

## 错误处理

- 网络错误：自动重试机制
- 数据缺失：返回空数据集或默认值
- 模型错误：使用备用分析方法