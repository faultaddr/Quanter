import { api } from './index';
import type { RealtimeQuote, ScanResult } from '@/types/api';
import type { PredictionResult } from '@/types/model';

type BackendScanSignal = string | {
  type?: string;
  name?: string;
  description?: string;
  confidence?: number;
  date?: string;
};

type BackendScanResult = Partial<Omit<ScanResult, 'signals'>> & {
  code?: string;
  stock_code?: string;
  stock_name?: string;
  current_price?: number;
  close?: number;
  change_percent?: number;
  daily_return?: number;
  final_score?: number;
  signal?: string;
  signals?: BackendScanSignal[];
  trigger_type?: string;
  trigger_detail?: string;
  action_display?: string;
  score_grade?: string;
};

type BackendPickResult = {
  symbol?: string;
  code?: string;
  name?: string;
  score?: number;
  percentile?: number;
  probability?: number;
  predicted_return?: number;
  pred_return?: number;
  confidence?: number;
};

interface ScanApiResponse {
  market: string;
  total_stocks: number;
  analyzed_stocks: number;
  top_n: number;
  results?: BackendScanResult[];
}

interface PicksApiResponse {
  success: boolean;
  date?: string;
  total_stocks?: number;
  valid_stocks?: number;
  top_stocks?: BackendPickResult[];
  model_info?: Record<string, unknown>;
}

function normalizeScanSignal(signal: BackendScanSignal): ScanResult['signals'][number] {
  if (typeof signal === 'string') {
    return {
      type: 'buy',
      name: signal,
      description: signal,
    };
  }

  const type = signal.type === 'sell' || signal.type === 'hold' ? signal.type : 'buy';
  const name = signal.name || signal.description || '选股信号';

  return {
    type,
    name,
    description: signal.description || name,
    confidence: signal.confidence,
    date: signal.date,
  };
}

function scanSignalTypeFromText(value: string): ScanResult['signals'][number]['type'] {
  const normalized = value.toLowerCase();
  if (value.includes('回避') || value.includes('卖') || normalized.includes('avoid') || normalized.includes('sell')) {
    return 'sell';
  }
  if (value.includes('观望') || value.includes('等待') || normalized.includes('hold') || normalized.includes('wait')) {
    return 'hold';
  }
  return 'buy';
}

function buildFallbackScanSignals(result: BackendScanResult): BackendScanSignal[] {
  const label = result.action_display || result.trigger_detail || result.trigger_type;
  if (!label) {
    return [];
  }

  return [{
    type: scanSignalTypeFromText(label),
    name: label,
    description: result.score_grade ? `${label} · ${result.score_grade}` : label,
  }];
}

function normalizeScanResult(result: BackendScanResult): ScanResult {
  const rawSignals = Array.isArray(result.signals)
    ? result.signals
    : result.signal
      ? [result.signal]
      : buildFallbackScanSignals(result);

  return {
    symbol: result.symbol || result.code || result.stock_code || '',
    name: result.name || result.stock_name || '',
    price: Number(result.price ?? result.current_price ?? result.close ?? 0),
    change_pct: Number(result.change_pct ?? result.change_percent ?? result.daily_return ?? 0),
    signals: rawSignals.map(normalizeScanSignal),
    score: Number(result.score ?? result.final_score ?? 0),
  };
}

function normalizePickResult(result: BackendPickResult, index: number): PredictionResult {
  return {
    symbol: result.symbol || result.code || '',
    name: result.name || '',
    score: Number(result.score ?? result.percentile ?? result.probability ?? 0),
    rank: index + 1,
    predicted_return: Number(result.predicted_return ?? result.pred_return ?? 0),
    confidence: Number(result.confidence ?? result.probability ?? 0),
  };
}

export const monitorApi = {
  // 获取实时行情
  getQuote: (symbol: string) =>
    api.get<any, RealtimeQuote>(`/realtime/quote/${symbol}`),

  // 批量获取实时行情
  getQuotes: (symbols: string[]) =>
    api.post<any, RealtimeQuote[]>('/realtime/batch', { symbols }),

  // 智能选股扫描
  scan: (params: {
    market?: string;
    use_unified_score?: boolean;
    use_trend_score?: boolean;
    use_breakout_score?: boolean;
    use_momentum_score?: boolean;
    include_fundamentals?: boolean;
    include_market_state?: boolean;
    min_price?: number;
    max_price?: number;
    min_volume?: number;
    signals?: string[];
    // 因子筛选
    factor_type?: string;
    min_factor_score?: number;
    enable_factor_filter?: boolean;
    // 风控过滤
    exclude_st?: boolean;
    exclude_suspended?: boolean;
    exclude_limit?: boolean;
  }) => api.post<any, ScanApiResponse>('/scan', params).then((response) => {
    return (response.results || []).map(normalizeScanResult);
  }),

  // 获取推荐股票 (POST 端点)
  getPicks: (modelId: string, topK: number = 10) =>
    api.post<any, PicksApiResponse>('/gbm/picks', {
      top_n: topK,
      model_path: modelId || undefined,
    }).then((response) => {
      return (response.top_stocks || []).map(normalizePickResult);
    }),
};
