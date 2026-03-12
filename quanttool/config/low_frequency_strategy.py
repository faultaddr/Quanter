# 极低频交易策略配置
#
# 基于回测结果优化的最佳配置
# 目标：降低交易频率，减少成本侵蚀，提高收益
#

# ============================================
# 策略名称
# ============================================
STRATEGY_NAME = "极低频 LightGBM 策略"

# ============================================
# 核心参数 (优化后)
# ============================================

# 信号阈值
BUY_THRESHOLD = 0.70      # 买入阈值 (提高，减少交易)
SELL_THRESHOLD = 0.50     # 卖出阈值 (提高，减少交易)

# 风控参数
STOP_LOSS_PCT = 0.10      # 止损 10% (放宽)
TAKE_PROFIT_PCT = 0.25    # 止盈 25% (放宽)

# 模型参数
MODEL_TYPE = "lgb"        # LightGBM
FEATURE_SET = "Alpha158"  # 特征集
EPOCHS = 50               # 训练轮数

# ============================================
# 交易成本设置 (真实市场)
# ============================================
COMMISSION_RATE = 0.0003  # 手续费 0.03%
SLIPPAGE_RATE = 0.001     # 滑点 0.1% (单向)
STAMP_DUTY_RATE = 0.001   # 印花税 0.1% (卖出)

# ============================================
# 预期表现 (基于回测)
# ============================================
EXPECTED_ANNUAL_RETURN = 0.158   # 预期年化收益 15.8%
EXPECTED_SHARPE = 1.69           # 预期夏普比
EXPECTED_MAX_DRAWDOWN = 0.065    # 预期最大回撤 6.5%
EXPECTED_TRADES_PER_STOCK = 108  # 每只股票年交易次数

# ============================================
# 完整配置字典
# ============================================
STRATEGY_CONFIG = {
    # 策略参数
    'model_type': MODEL_TYPE,
    'feature_set': FEATURE_SET,
    'buy_threshold': BUY_THRESHOLD,
    'sell_threshold': SELL_THRESHOLD,
    'stop_loss_pct': STOP_LOSS_PCT,
    'take_profit_pct': TAKE_PROFIT_PCT,
    'epochs': EPOCHS,

    # 成本设置
    'commission_rate': COMMISSION_RATE,
    'slippage_rate': SLIPPAGE_RATE,
    'stamp_duty_rate': STAMP_DUTY_RATE,

    # 预期表现
    'expected_annual_return': EXPECTED_ANNUAL_RETURN,
    'expected_sharpe': EXPECTED_SHARPE,
    'expected_max_drawdown': EXPECTED_MAX_DRAWDOWN,
}

# ============================================
# 与高频策略对比
# ============================================
COMPARISON = {
    '高频交易': {
        'buy_threshold': 0.53,
        'sell_threshold': 0.30,
        'stop_loss_pct': 0.05,
        'take_profit_pct': 0.12,
        'annual_return': 0.123,
        'sharpe': 1.39,
        'trades': 700,
    },
    '极低频交易': {
        'buy_threshold': 0.70,
        'sell_threshold': 0.50,
        'stop_loss_pct': 0.10,
        'take_profit_pct': 0.25,
        'annual_return': 0.158,
        'sharpe': 1.69,
        'trades': 541,
    },
}


def get_strategy_config():
    """获取策略配置"""
    return STRATEGY_CONFIG.copy()


def print_strategy_summary():
    """打印策略摘要"""
    print(f"""
{'='*60}
{STRATEGY_NAME}
{'='*60}

核心参数:
  买入阈值: {BUY_THRESHOLD}
  卖出阈值: {SELL_THRESHOLD}
  止损比例: {STOP_LOSS_PCT*100:.0f}%
  止盈比例: {TAKE_PROFIT_PCT*100:.0f}%

交易成本:
  手续费: {COMMISSION_RATE*100:.3f}%
  滑点: {SLIPPAGE_RATE*100:.2f}% (单向)
  印花税: {STAMP_DUTY_RATE*100:.2f}% (卖出)

预期表现:
  年化收益: {EXPECTED_ANNUAL_RETURN*100:.2f}%
  夏普比: {EXPECTED_SHARPE:.2f}
  最大回撤: {EXPECTED_MAX_DRAWDOWN*100:.2f}%

收益提升: +{(EXPECTED_ANNUAL_RETURN - 0.123)*100:.2f}% (vs 高频策略)
{'='*60}
""")


if __name__ == "__main__":
    print_strategy_summary()