'use client';

import Card from '@/components/ui/Card';
import Badge from '@/components/ui/Badge';
import type { Signal } from '@/types/stock';

interface SignalCardProps {
  signal: Signal;
}

export default function SignalCard({ signal }: SignalCardProps) {
  const getBadgeVariant = () => {
    switch (signal.type) {
      case 'buy':
        return 'success';
      case 'sell':
        return 'danger';
      case 'hold':
        return 'warning';
      default:
        return 'default';
    }
  };

  const getBadgeLabel = () => {
    switch (signal.type) {
      case 'buy':
        return '买入';
      case 'sell':
        return '卖出';
      case 'hold':
        return '持有';
      default:
        return '未知';
    }
  };

  return (
    <Card className="card-hover">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <Badge variant={getBadgeVariant()}>{getBadgeLabel()}</Badge>
            <span className="font-medium text-text-primary">{signal.name}</span>
          </div>
          <p className="text-sm text-text-muted">{signal.description}</p>
          {signal.confidence && (
            <div className="mt-2">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-text-muted">置信度</span>
                <span className="text-text-secondary">{(signal.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="h-1.5 bg-bg-tertiary rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${
                    signal.confidence >= 0.7 ? 'bg-success' : signal.confidence >= 0.4 ? 'bg-warning' : 'bg-danger'
                  }`}
                  style={{ width: `${signal.confidence * 100}%` }}
                />
              </div>
            </div>
          )}
        </div>
        {signal.date && (
          <span className="text-xs text-text-muted">{signal.date}</span>
        )}
      </div>
    </Card>
  );
}
