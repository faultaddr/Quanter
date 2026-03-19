'use client';

import { cn } from '@/lib/utils';
import type { Strategy } from '@/types/backtest';

interface StrategySelectorProps {
  strategies: Strategy[];
  selected: string | null;
  onSelect: (strategyId: string) => void;
  className?: string;
}

export default function StrategySelector({
  strategies,
  selected,
  onSelect,
  className,
}: StrategySelectorProps) {
  // 按分类分组
  const groupedStrategies = strategies.reduce((acc, strategy) => {
    const category = strategy.category || 'other';
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(strategy);
    return acc;
  }, {} as Record<string, Strategy[]>);

  const categoryLabels: Record<string, string> = {
    trend: '趋势跟踪',
    mean_reversion: '均值回归',
    momentum: '动量策略',
    factor: '因子策略',
    other: '其他策略',
  };

  return (
    <div className={cn('space-y-4', className)}>
      {Object.entries(groupedStrategies).map(([category, items]) => (
        <div key={category}>
          <h4 className="text-sm font-medium text-text-muted mb-2">
            {categoryLabels[category] || category}
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {items.map((strategy) => (
              <button
                key={strategy.id}
                onClick={() => onSelect(strategy.id)}
                className={cn(
                  'px-3 py-2 rounded-lg text-sm text-left transition-all',
                  'border',
                  selected === strategy.id
                    ? 'bg-primary/20 border-primary text-primary'
                    : 'bg-bg-tertiary border-border-primary text-text-secondary hover:border-primary hover:text-text-primary'
                )}
              >
                <div className="font-medium">{strategy.name}</div>
                {strategy.description && (
                  <div className="text-xs text-text-muted mt-0.5 truncate">
                    {strategy.description}
                  </div>
                )}
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
