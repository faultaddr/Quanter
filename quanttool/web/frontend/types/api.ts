// API 响应类型定义
import type { Signal } from './stock';

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface RealtimeQuote {
  symbol: string;
  name: string;
  price: number;
  change: number;
  change_pct: number;
  open: number;
  high: number;
  low: number;
  volume: number;
  amount: number;
  timestamp: string;
  time?: string;  // 兼容旧字段
  turnover?: number;
  source?: string;
  error?: string;
}

export interface ScanResult {
  symbol: string;
  name: string;
  price: number;
  change_pct: number;
  signals: Signal[];
  score?: number;
}
