# 真实交易成本配置
#
# 基于实际 A股市场环境设置
#

# ============================================
# 交易成本设置
# ============================================

# 手续费 (双向)
COMMISSION_RATE = 0.0003  # 0.03%

# 滑点 (单向，双向需 x2)
SLIPPAGE_RATE = 0.001  # 0.1% 单向, 0.2% 双向

# 印花税 (仅卖出)
STAMP_DUTY_RATE = 0.001  # 0.1%

# 最小手续费
MIN_COMMISSION = 5.0  # 5元

# ============================================
# 成本影响估算
# ============================================

# 每次交易总成本 (买入 + 卖出)
# 手续费: 0.06% (双向)
# 滑点: 0.2% (双向)
# 印花税: 0.1% (卖出)
# 总计: 约 0.36%

TOTAL_COST_PER_TRADE = 0.0036  # 0.36%

# 年化影响 (假设每年交易 100 次)
# 100 x 0.36% = 36% 成本
# 实际侵蚀收益约 5-6%

# ============================================
# 回测折扣系数
# ============================================

# 真实收益约为回测结果的 70%
BACKTEST_DISCOUNT_FACTOR = 0.70

# ============================================
# 成本配置字典
# ============================================

REALISTIC_COSTS = {
    'commission': COMMISSION_RATE,
    'slippage': SLIPPAGE_RATE,
    'stamp_duty': STAMP_DUTY_RATE,
    'min_commission': MIN_COMMISSION,
}

# 获取回测引擎设置
def get_backtest_costs():
    """获取用于 BacktestEngine 的成本设置"""
    return {
        'commission_rate': COMMISSION_RATE,
        'slippage_rate': SLIPPAGE_RATE,
        'min_commission': MIN_COMMISSION,
    }