// 常量定义

// 信号类型映射
export const SIGNAL_TYPES = {
  buy: { label: '买入', color: 'success' },
  sell: { label: '卖出', color: 'danger' },
  hold: { label: '持有', color: 'warning' },
} as const;

// 策略分类
export const STRATEGY_CATEGORIES = {
  trend: '趋势跟踪',
  mean_reversion: '均值回归',
  momentum: '动量策略',
  factor: '因子策略',
} as const;

// 市场指数
export const MARKET_INDICES = [
  { code: '000001', name: '上证指数' },
  { code: '399001', name: '深证成指' },
  { code: '399006', name: '创业板指' },
  { code: '000016', name: '上证50' },
  { code: '000300', name: '沪深300' },
  { code: '000905', name: '中证500' },
] as const;

// 默认回测参数
export const DEFAULT_BACKTEST_PARAMS = {
  initial_capital: 1000000,
  commission: 0.0003,
  slippage: 0.001,
} as const;

// 主题色
export const THEME_COLORS = {
  primary: '#3B82F6',
  success: '#10B981',
  danger: '#EF4444',
  warning: '#F59E0B',
} as const;

// 图表颜色
export const CHART_COLORS = {
  up: '#10B981',
  down: '#EF4444',
  volume: '#3B82F6',
  ma5: '#F59E0B',
  ma10: '#8B5CF6',
  ma20: '#EC4899',
  ma60: '#06B6D4',
} as const;
