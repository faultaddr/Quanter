// 股票相关类型定义

export interface StockInfo {
  symbol: string;
  name: string;
  industry?: string;
  market?: string;
}

export interface KlineData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  amount?: number;
  turnover?: number;
}

export interface ChipDistribution {
  price: number;
  percent: number;
  cost?: number;
}

export interface Signal {
  type: 'buy' | 'sell' | 'hold';
  name: string;
  description: string;
  confidence?: number;
  date?: string;
}

export interface TechnicalIndicators {
  macd?: {
    dif: number[];
    dea: number[];
    macd: number[];
  };
  kdj?: {
    k: number[];
    d: number[];
    j: number[];
  };
  rsi?: {
    rsi6: number[];
    rsi12: number[];
    rsi24: number[];
  };
  ma?: {
    ma5: number[];
    ma10: number[];
    ma20: number[];
    ma60: number[];
  };
}

export interface StockAnalysis {
  symbol: string;
  name: string;
  kline: KlineData[];
  chip: ChipDistribution[];
  signals: Signal[];
  indicators: TechnicalIndicators;
}
