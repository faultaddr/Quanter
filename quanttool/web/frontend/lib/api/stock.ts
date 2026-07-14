import { api } from './index';
import type { StockAnalysis, KlineData, ChipDistribution, Signal } from '@/types/stock';

// 资金流向数据类型
export interface FlowData {
  date: string;
  main_inflow: number;
  main_outflow: number;
  retail_inflow: number;
  retail_outflow: number;
  net_main: number;
  net_retail: number;
}

// 风险评估数据类型
export interface RiskMetrics {
  volatility: number;
  max_drawdown: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  win_rate: number;
  profit_loss_ratio: number;
  avg_holding_days: number;
  beta: number;
  alpha: number;
}

// 回测结果类型
export interface BacktestResult {
  strategy_name: string;
  total_return: number;
  annual_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  equity_curve: { date: string; value: number }[];
}

export const stockApi = {
  // 获取股票信息
  getInfo: (symbol: string) =>
    api.get<any, { symbol: string; name: string; industry?: string }>(`/stock/${symbol}/info`),

  // 获取 K 线数据
  getKline: (symbol: string, days: number = 120) =>
    api.get<any, KlineData[]>(`/stock/${symbol}/kline`, { params: { days } }),

  // 获取筹码分布
  getChip: (symbol: string) =>
    api.get<any, ChipDistribution[]>(`/stock/${symbol}/chip`),

  // 获取交易信号
  getSignals: (symbol: string) =>
    api.get<any, Signal[]>(`/stock/${symbol}/signals`),

  // 获取完整分析数据
  getAnalysis: (symbol: string, days: number = 120) =>
    api.get<any, StockAnalysis>(`/stock/${symbol}/analysis`, { params: { days } }),

  // 获取资金流向数据
  getFlow: (symbol: string, days: number = 30) =>
    api.get<any, { symbol: string; data: FlowData[] }>(`/stock/${symbol}/flow`, { params: { days } }),

  // 获取风险评估数据
  getRisk: (symbol: string, days: number = 250) =>
    api.get<any, { symbol: string; period_days: number; metrics: RiskMetrics }>(`/stock/${symbol}/risk`, { params: { days } }),

  // 获取回测对比数据
  getBacktestCompare: (symbol: string, days: number = 250) =>
    api.get<any, {
      symbol: string;
      period_days: number;
      results: BacktestResult[];
      benchmark: { name: string; total_return: number; equity_curve: { date: string; value: number }[] };
    }>(`/stock/${symbol}/backtest-compare`, { params: { days } }),

  // 搜索股票 (使用 realtime/search 端点)
  search: (keyword: string) =>
    api.get<any, { symbol: string; name: string; price?: number }[]>('/realtime/search', { params: { query: keyword } }),

  // 获取指数数据
  getIndexData: (indexCode: string = '000001') =>
    api.get<any, { date: string; value: number }[]>(`/index/${indexCode}/data`),
};
