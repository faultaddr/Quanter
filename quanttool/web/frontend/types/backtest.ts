// 回测相关类型定义

export interface BacktestParams {
  symbols: string[];
  strategy_name: string;
  start_date?: string;
  end_date?: string;
  initial_cash: number;
  commission_rate?: number;
  strategy_params?: Record<string, any>;
}

export interface TradeRecord {
  date: string;
  type: 'buy' | 'sell';
  price: number;
  shares: number;
  amount?: number;
  commission?: number;
  profit?: number;
  reason?: string;
}

export interface BacktestMetrics {
  total_return: number;
  annual_return: number;
  max_drawdown: number;
  sharpe_ratio: number;
  win_rate: number;
  profit_factor?: number;
  total_trades: number;
  profit_trades?: number;
  loss_trades?: number;
  benchmark_return?: number;
  excess_return?: number;
}

// 匹配后端返回格式
export interface BacktestResult {
  strategy: string;
  symbols?: string[];
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_capital: number;
  total_return: number;
  annual_return: number;
  benchmark_return?: number;  // 基准收益
  excess_return?: number;      // 超额收益
  max_drawdown: number;
  sharpe_ratio: number;
  total_trades: number;
  win_rate: number;
  profit_factor?: number;
  trades: TradeRecord[];
  equity_curve?: { date: string; value: number }[];
  benchmark_curve?: { date: string; value: number }[];
  // 前端兼容字段
  symbol?: string;
  metrics?: BacktestMetrics;
}

export interface Strategy {
  id: string;
  name: string;
  display_name?: string;
  description?: string;
  category?: string;
  params?: Record<string, any>;
}
