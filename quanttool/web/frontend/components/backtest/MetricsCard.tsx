'use client';

import Card from '@/components/ui/Card';
import { formatNumber, formatPercent, getChangeColorClass } from '@/lib/utils';
import { cn } from '@/lib/utils';

interface MetricsCardProps {
  title: string;
  value: number | string;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
  format?: 'number' | 'percent' | 'raw';
  size?: 'sm' | 'md' | 'lg';
}

export default function MetricsCard({
  title,
  value,
  subtitle,
  trend,
  format = 'raw',
  size = 'md',
}: MetricsCardProps) {
  const formattedValue = typeof value === 'number'
    ? format === 'percent'
      ? formatPercent(value)
      : format === 'number'
        ? formatNumber(value)
        : value.toString()
    : value;

  const sizeClasses = {
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-5',
  };

  const valueSizeClasses = {
    sm: 'text-lg',
    md: 'text-xl',
    lg: 'text-2xl',
  };

  const getTrendColor = () => {
    if (!trend) return 'text-text-primary';
    return trend === 'up' ? 'text-success' : trend === 'down' ? 'text-danger' : 'text-text-primary';
  };

  return (
    <Card className={sizeClasses[size]} noPadding>
      <div className="p-4">
        <div className="text-sm text-text-muted mb-1">{title}</div>
        <div className={cn(valueSizeClasses[size], 'font-bold', getTrendColor())}>
          {formattedValue}
        </div>
        {subtitle && (
          <div className="text-xs text-text-muted mt-1">{subtitle}</div>
        )}
      </div>
    </Card>
  );
}
