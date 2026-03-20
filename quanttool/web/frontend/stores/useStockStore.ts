import { create } from 'zustand';
import type { StockAnalysis, KlineData, Signal } from '@/types/stock';

interface StockState {
  // 当前股票
  currentSymbol: string;
  currentName: string;
  setStock: (symbol: string, name: string) => void;

  // 分析数据缓存
  analysisCache: Record<string, StockAnalysis>;
  setAnalysis: (symbol: string, analysis: StockAnalysis) => void;
  getAnalysis: (symbol: string) => StockAnalysis | undefined;

  // 自选股列表
  watchlist: { symbol: string; name: string }[];
  addToWatchlist: (symbol: string, name: string) => void;
  removeFromWatchlist: (symbol: string) => void;
  isInWatchlist: (symbol: string) => boolean;
}

export const useStockStore = create<StockState>()(
  (set, get) => ({
    currentSymbol: '',
    currentName: '',
    setStock: (symbol, name) => set({ currentSymbol: symbol, currentName: name }),

    analysisCache: {},
    setAnalysis: (symbol, analysis) =>
      set((state) => ({
        analysisCache: { ...state.analysisCache, [symbol]: analysis },
      })),
    getAnalysis: (symbol) => get().analysisCache[symbol],

    watchlist: [],
    addToWatchlist: (symbol, name) =>
      set((state) => {
        if (state.watchlist.some((w) => w.symbol === symbol)) {
          return state;
        }
        return {
          watchlist: [...state.watchlist, { symbol, name }],
        };
      }),
    removeFromWatchlist: (symbol) =>
      set((state) => ({
        watchlist: state.watchlist.filter((w) => w.symbol !== symbol),
      })),
    isInWatchlist: (symbol) => get().watchlist.some((w) => w.symbol === symbol),
  })
);
