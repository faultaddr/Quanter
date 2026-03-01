# 定时任务与投资组合回测验证系统 - 使用指南

## 系统概述

这个系统实现了：
1. **每日自动扫描**：收盘后自动运行 scan，选出沪深300中 top 5 股票
2. **投资组合回测**：用 50 万虚拟资金自动创建投资组合
3. **净值跟踪**：每日更新组合净值，跟踪 20 个交易日
4. **报告生成**：每日生成报告并通过邮件发送

## 安装依赖

```bash
pip install apscheduler aiosmtplib jinja2
```

## 配置环境变量

```bash
export TUSHARE_TOKEN="your_tushare_token"
```

## 使用方式

### 1. 手动执行完整流程（一键操作）

```bash
# 立即执行 scan 并创建投资组合
quant portfolio auto-create --capital 500000 --top 5
```

### 2. 分步操作

```bash
# Step 1: 执行 scan
quant analysis scan --market csi300 --top 10 --save

# Step 2: 从 scan 创建投资组合
quant portfolio create <scan_id> --capital 500000

# Step 3: 查看组合详情
quant portfolio view <backtest_id>

# Step 4: 每日更新净值
quant portfolio update

# Step 5: 生成报告
quant report daily
```

### 3. 定时任务（自动运行）

```bash
# 配置邮件（用于接收每日报告）
quant report config \
  --smtp-host smtp.gmail.com \
  --smtp-port 587 \
  --username your_email@gmail.com \
  --password your_app_password \
  --recipient your_email@gmail.com \
  --test

# 启动定时任务调度器（前台运行）
quant scheduler start

# 或后台运行
quant scheduler start --daemon

# 查看状态
quant scheduler status

# 停止调度器
quant scheduler stop
```

### 4. 定时任务默认时间表

| 时间 | 任务 |
|------|------|
| 15:30 | 执行每日 scan，选出 top 5 股票 |
| 15:35 | 自动创建投资组合（50万初始资金） |
| 18:00 | 更新所有活跃组合净值 |
| 19:00 | 生成报告并发送邮件 |

### 5. 查看历史记录

```bash
# 查看所有投资组合
quant portfolio list

# 查看活跃组合
quant portfolio list --status active

# 查看历史报告
quant report history --days 30

# 查看 scan 历史
quant analysis history --limit 10
```

## 数据库表结构

### portfolio_backtests - 投资组合回测记录
- id, scan_id, portfolio_name, initial_capital
- start_date, end_date, status
- total_return, annualized_return, sharpe_ratio, max_drawdown

### portfolio_holdings - 持仓明细
- backtest_id, symbol, name, entry_date, entry_price
- shares, weight, status, exit_date, exit_price, realized_return

### portfolio_daily_values - 每日净值
- backtest_id, date, total_value, cash_value, market_value, daily_return

## 投资组合逻辑

1. **选股**：取当日 scan 评分最高的 5 只股票
2. **资金分配**：等权重，每只股票 10 万元
3. **买入规则**：按收盘价买入，100 股整数倍
4. **持仓周期**：20 个交易日
5. **平仓规则**：到期日按收盘价卖出

## 报告内容

每日报告包含：
- 当日 scan 结果（top 5 股票、评分、操作建议）
- 活跃投资组合表现（当前市值、收益率、持仓明细）
- 近期平仓组合回顾
- 策略有效性分析（胜率、平均收益、夏普比率等）

## 注意事项

1. **交易日判断**：系统自动跳过周末和节假日
2. **数据源**：需要有效的 Tushare token
3. **邮件配置**：建议使用 Gmail 应用专用密码
4. **后台运行**：使用 `--daemon` 参数或配合 systemd/supervisor
