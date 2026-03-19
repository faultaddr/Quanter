'use client';

import { useState, useEffect, useCallback } from 'react';
import PageContainer from '@/components/layout/PageContainer';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import Badge from '@/components/ui/Badge';
import QuoteCard from '@/components/stock/QuoteCard';
import { useAppStore } from '@/stores/useAppStore';
import { useMonitorStore } from '@/stores/useMonitorStore';
import { useWebSocket } from '@/hooks/useWebSocket';
import { monitorApi } from '@/lib/api/monitor';
import { useToast } from '@/hooks/useToast';
import type { RealtimeQuote } from '@/types/api';

export default function MonitorPage() {
  const setActivePage = useAppStore((state) => state.setActivePage);
  const toast = useToast();

  const symbols = useMonitorStore((state) => state.symbols);
  const quotes = useMonitorStore((state) => state.quotes);
  const addSymbol = useMonitorStore((state) => state.addSymbol);
  const removeSymbol = useMonitorStore((state) => state.removeSymbol);
  const updateQuotes = useMonitorStore((state) => state.updateQuotes);
  const wsConnected = useMonitorStore((state) => state.wsConnected);
  const setWsConnected = useMonitorStore((state) => state.setWsConnected);
  const autoRefresh = useMonitorStore((state) => state.autoRefresh);
  const toggleAutoRefresh = useMonitorStore((state) => state.toggleAutoRefresh);

  const [newSymbol, setNewSymbol] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);

  // WebSocket connection
  const { isConnected } = useWebSocket(
    process.env.NODE_ENV === 'production'
      ? `wss://${window.location.host}/ws`
      : 'ws://localhost:8000/ws',
    {
      onMessage: (data) => {
        if (data.type === 'quotes' && Array.isArray(data.quotes)) {
          updateQuotes(data.quotes);
        }
      },
      onOpen: () => {
        setWsConnected(true);
        toast.success('实时连接已建立');
      },
      onClose: () => {
        setWsConnected(false);
        toast.warning('实时连接已断开');
      },
    }
  );

  // Manual refresh
  const refreshQuotes = useCallback(async () => {
    if (symbols.length === 0) return;

    try {
      const data = await monitorApi.getQuotes(symbols);
      if (data) {
        updateQuotes(data);
      }
    } catch (error) {
      toast.error('刷新失败');
    }
  }, [symbols, updateQuotes, toast]);

  // Auto refresh interval
  useEffect(() => {
    if (!autoRefresh || symbols.length === 0) return;

    const interval = setInterval(refreshQuotes, 5000);
    return () => clearInterval(interval);
  }, [autoRefresh, symbols.length, refreshQuotes]);

  useEffect(() => {
    setActivePage('monitor');
    // Load initial quotes
    if (symbols.length > 0) {
      refreshQuotes();
    }
  }, [setActivePage, refreshQuotes, symbols.length]);

  const handleAddSymbol = () => {
    const symbol = newSymbol.trim().toUpperCase();
    if (!symbol) return;

    if (symbols.includes(symbol)) {
      toast.warning('该股票已在监控列表中');
      return;
    }

    addSymbol(symbol);
    setNewSymbol('');
    toast.success(`已添加 ${symbol}`);
  };

  const quoteList = Object.values(quotes).sort((a, b) =>
    symbols.indexOf(a.symbol) - symbols.indexOf(b.symbol)
  );

  return (
    <PageContainer>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-text-primary">实时监控</h1>
            <p className="text-text-muted mt-1">WebSocket实时行情，自定义监控列表</p>
          </div>
          <div className="flex items-center gap-3">
            <Badge variant={wsConnected ? 'success' : 'danger'}>
              {wsConnected ? '已连接' : '未连接'}
            </Badge>
            <Button
              variant={autoRefresh ? 'primary' : 'secondary'}
              size="sm"
              onClick={toggleAutoRefresh}
            >
              {autoRefresh ? '自动刷新中' : '自动刷新'}
            </Button>
            <Button size="sm" onClick={refreshQuotes}>
              手动刷新
            </Button>
          </div>
        </div>

        {/* Add Symbol */}
        <Card>
          <div className="flex gap-3">
            <Input
              value={newSymbol}
              onChange={(e) => setNewSymbol(e.target.value)}
              placeholder="输入股票代码，如：000001"
              className="flex-1"
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleAddSymbol();
              }}
            />
            <Button onClick={handleAddSymbol}>添加</Button>
          </div>
        </Card>

        {/* Quote Grid */}
        {quoteList.length === 0 ? (
          <Card className="text-center py-20">
            <svg className="w-16 h-16 mx-auto text-text-muted mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
            <h3 className="text-lg font-medium text-text-primary mb-2">添加监控股票</h3>
            <p className="text-text-muted">在上方输入框添加股票代码开始监控</p>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {quoteList.map((quote) => (
              <div key={quote.symbol} className="relative">
                <QuoteCard
                  quote={quote}
                  selected={selectedSymbol === quote.symbol}
                  onClick={() => setSelectedSymbol(
                    selectedSymbol === quote.symbol ? null : quote.symbol
                  )}
                />
                <button
                  onClick={() => {
                    removeSymbol(quote.symbol);
                    toast.info(`已移除 ${quote.symbol}`);
                  }}
                  className="absolute top-2 right-2 p-1 text-text-muted hover:text-danger transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </PageContainer>
  );
}
