import { create } from 'zustand';
import type { RealtimeQuote } from '@/types/api';

interface MonitorState {
  // 监控股票列表
  symbols: string[];
  addSymbol: (symbol: string) => void;
  removeSymbol: (symbol: string) => void;
  setSymbols: (symbols: string[]) => void;

  // 实时行情数据
  quotes: Record<string, RealtimeQuote>;
  updateQuote: (symbol: string, quote: RealtimeQuote) => void;
  updateQuotes: (quotes: RealtimeQuote[]) => void;

  // WebSocket 连接状态
  wsConnected: boolean;
  setWsConnected: (connected: boolean) => void;

  // 自动刷新
  autoRefresh: boolean;
  toggleAutoRefresh: () => void;
}

export const useMonitorStore = create<MonitorState>()((set, get) => ({
  symbols: [],
  addSymbol: (symbol) =>
    set((state) => ({
      symbols: state.symbols.includes(symbol)
        ? state.symbols
        : [...state.symbols, symbol],
    })),
  removeSymbol: (symbol) =>
    set((state) => ({
      symbols: state.symbols.filter((s) => s !== symbol),
    })),
  setSymbols: (symbols) => set({ symbols }),

  quotes: {},
  updateQuote: (symbol, quote) =>
    set((state) => ({
      quotes: { ...state.quotes, [symbol]: quote },
    })),
  updateQuotes: (quotes) =>
    set(() => {
      const quotesMap: Record<string, RealtimeQuote> = {};
      quotes.forEach((q) => {
        quotesMap[q.symbol] = q;
      });
      return { quotes: quotesMap };
    }),

  wsConnected: false,
  setWsConnected: (connected) => set({ wsConnected: connected }),

  autoRefresh: true,
  toggleAutoRefresh: () =>
    set((state) => ({ autoRefresh: !state.autoRefresh })),
}));
