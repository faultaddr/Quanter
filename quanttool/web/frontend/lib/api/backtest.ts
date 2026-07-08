import { api, getApiUrl } from './index';
import type { BacktestParams, BacktestResult, Strategy } from '@/types/backtest';

export const backtestApi = {
  // 运行单个策略回测
  run: (params: {
    symbol: string;
    strategy: string;
    start_date: string;
    end_date: string;
    initial_capital: number;
    // A股约束选项
    enable_constraints?: boolean;
    exclude_st?: boolean;
    exclude_limit?: boolean;
    commission_rate?: number;
  }) =>
    api.post<any, BacktestResult>('/backtest/run', {
      symbols: [params.symbol],
      strategy_name: params.strategy,
      start_date: params.start_date,
      end_date: params.end_date,
      initial_cash: params.initial_capital,
      // A股约束
      enable_constraints: params.enable_constraints,
      exclude_st: params.exclude_st,
      exclude_limit: params.exclude_limit,
      commission_rate: params.commission_rate,
    }),

  // 运行所有策略回测对比（流式版本）
  runAllStream: (params: {
    symbol: string;
    start_date: string;
    end_date: string;
    initial_capital: number;
    enable_constraints?: boolean;
    exclude_st?: boolean;
    exclude_limit?: boolean;
    commission_rate?: number;
  }, onStrategyComplete: (result: any) => void, onDone: (results: any[]) => void, onError: (error: string) => void) => {
    const controller = new AbortController();

    fetch(getApiUrl('/backtest/run-all-stream'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        symbols: [params.symbol],
        start_date: params.start_date,
        end_date: params.end_date,
        initial_cash: params.initial_capital,
        enable_constraints: params.enable_constraints,
        exclude_st: params.exclude_st,
        exclude_limit: params.exclude_limit,
        commission_rate: params.commission_rate,
      }),
      signal: controller.signal,
    }).then(response => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      function read() {
        reader?.read().then(({ done, value }) => {
          if (done) {
            return;
          }
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                console.log('[SSE] Received:', data.type, data.result?.strategy || '');
                if (data.type === 'strategy_complete') {
                  onStrategyComplete(data.result);
                } else if (data.type === 'done') {
                  onDone(data.results);
                }
              } catch (e) {
                console.error('Parse SSE error:', e);
              }
            }
          }
          read();
        });
      }
      read();
    }).catch(err => {
      if (err.name !== 'AbortError') {
        onError(err.message);
      }
    });

    return controller;
  },

  // 获取可用策略列表
  getStrategies: () =>
    api.get<any, Strategy[]>('/backtest/strategies'),

  // 获取历史回测记录
  getHistory: (symbol?: string) =>
    api.get<any, BacktestResult[]>('/backtest/history', { params: { symbol } }),
};
