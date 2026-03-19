'use client';

import MetricsCard from './MetricsCard';
import type { BacktestMetrics } from '@/types/backtest';
import { formatPercent, formatNumber } from '@/lib/utils';

interface MetricsGridProps {
  metrics: BacktestMetrics;
  excessReturn?: number;
}

export default function MetricsGrid({ metrics, excessReturn }: MetricsGridProps) {
  const profitRatio = metrics.profit_factor ?? metrics.profit_loss_ratio ?? 0;
  const profitTrades = metrics.profit_trades ?? Math.floor(metrics.total_trades * metrics.win_rate);

  const metricItems = [
    {
      title: '总收益率',
      value: formatPercent(metrics.total_return),
      trend: metrics.total_return >= 0 ? 'up' as const : 'down' as const,
    },
    {
      title: '年化收益率',
      value: formatPercent(metrics.annual_return),
      trend: metrics.annual_return >= 0 ? 'up' as const : 'down' as const,
    },
    {
      title: '最大回撤',
      value: formatPercent(metrics.max_drawdown),
      trend: 'down' as const,
    },
    {
      title: '夏普比率',
      value: formatNumber(metrics.sharpe_ratio, 2),
      trend: metrics.sharpe_ratio >= 1 ? 'up' as const : metrics.sharpe_ratio >= 0 ? 'neutral' as const : 'down' as const,
    },
    {
      title: '胜率',
      value: formatPercent(metrics.win_rate),
      trend: metrics.win_rate >= 0.5 ? 'up' as const : 'down' as const,
    },
    {
      title: '盈亏比',
      value: formatNumber(profitRatio, 2),
      trend: profitRatio >= 1 ? 'up' as const : 'down' as const,
    },
    {
      title: '总交易次数',
      value: metrics.total_trades,
    },
    {
      title: '盈利次数',
      value: profitTrades,
      trend: 'up' as const,
    },
  ];

  if (excessReturn !== undefined) {
    metricItems.push({
      title: '超额收益',
      value: formatPercent(excessReturn),
      trend: excessReturn >= 0 ? 'up' as const : 'down' as const,
    });
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
      {metricItems.map((item, index) => (
        <MetricsCard
          key={index}
          title={item.title}
          value={item.value}
          trend={item.trend}
          size="sm"
        />
      ))}
    </div>
  );
}
