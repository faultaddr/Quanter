'use client';

import type { Signal } from '@/types/stock';
import SignalCard from './SignalCard';

interface SignalPanelProps {
  signals: Signal[];
  title?: string;
  emptyText?: string;
}

export default function SignalPanel({
  signals,
  title = '交易信号',
  emptyText = '暂无交易信号',
}: SignalPanelProps) {
  // 按类型分组
  const buySignals = signals.filter((s) => s.type === 'buy');
  const sellSignals = signals.filter((s) => s.type === 'sell');
  const holdSignals = signals.filter((s) => s.type === 'hold');

  if (signals.length === 0) {
    return (
      <div className="text-center py-8 text-text-muted">
        {emptyText}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* 买入信号 */}
      {buySignals.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-success mb-2 flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
            </svg>
            买入信号 ({buySignals.length})
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {buySignals.map((signal, index) => (
              <SignalCard key={index} signal={signal} />
            ))}
          </div>
        </div>
      )}

      {/* 卖出信号 */}
      {sellSignals.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-danger mb-2 flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
            卖出信号 ({sellSignals.length})
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {sellSignals.map((signal, index) => (
              <SignalCard key={index} signal={signal} />
            ))}
          </div>
        </div>
      )}

      {/* 持有信号 */}
      {holdSignals.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-warning mb-2 flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
            </svg>
            持有信号 ({holdSignals.length})
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {holdSignals.map((signal, index) => (
              <SignalCard key={index} signal={signal} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
