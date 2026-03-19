'use client';

import Card from '@/components/ui/Card';
import Badge from '@/components/ui/Badge';
import { formatNumber, formatAmount, getChangeColorClass } from '@/lib/utils';
import type { RealtimeQuote } from '@/types/api';

interface QuoteCardProps {
  quote: RealtimeQuote;
  onClick?: () => void;
  selected?: boolean;
}

export default function QuoteCard({ quote, onClick, selected }: QuoteCardProps) {
  const changeColorClass = getChangeColorClass(quote.change_pct);

  return (
    <Card
      className={`cursor-pointer transition-all hover:border-primary ${selected ? 'border-primary ring-1 ring-primary' : ''}`}
      noPadding
      onClick={onClick}
    >
      <div className="p-4">
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div>
            <div className="font-medium text-text-primary">{quote.name}</div>
            <div className="text-xs text-text-muted mt-0.5">{quote.symbol}</div>
          </div>
          <Badge variant={quote.change_pct >= 0 ? 'success' : 'danger'}>
            {quote.change_pct >= 0 ? '↑' : '↓'}
          </Badge>
        </div>

        {/* Price */}
        <div className={`text-2xl font-bold ${changeColorClass}`}>
          {formatNumber(quote.price)}
        </div>
        <div className={`text-sm mt-1 ${changeColorClass}`}>
          {quote.change_pct >= 0 ? '+' : ''}{formatNumber(quote.change)} ({(quote.change_pct * 100).toFixed(2)}%)
        </div>

        {/* Details Grid */}
        <div className="grid grid-cols-2 gap-x-4 gap-y-2 mt-4 text-sm">
          <div className="flex justify-between">
            <span className="text-text-muted">今开</span>
            <span className="text-text-primary">{formatNumber(quote.open)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-text-muted">最高</span>
            <span className="text-success">{formatNumber(quote.high)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-text-muted">最低</span>
            <span className="text-danger">{formatNumber(quote.low)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-text-muted">成交量</span>
            <span className="text-text-primary">{formatAmount(quote.volume)}</span>
          </div>
        </div>

        {/* Time */}
        <div className="text-xs text-text-muted mt-3 text-right">
          {quote.timestamp || quote.time || ''}
        </div>
      </div>
    </Card>
  );
}
