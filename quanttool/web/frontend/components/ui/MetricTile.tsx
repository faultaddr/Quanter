'use client';

import { cn } from '@/lib/utils';

interface MetricTileProps {
  label: string;
  value: React.ReactNode;
  detail?: React.ReactNode;
  tone?: 'default' | 'positive' | 'negative' | 'warning' | 'accent' | 'muted';
  className?: string;
}

const toneClasses: Record<NonNullable<MetricTileProps['tone']>, string> = {
  default: 'text-text-primary',
  positive: 'text-success',
  negative: 'text-danger',
  warning: 'text-warning',
  accent: 'text-primary',
  muted: 'text-text-secondary',
};

export default function MetricTile({
  label,
  value,
  detail,
  tone = 'default',
  className,
}: MetricTileProps) {
  return (
    <div className={cn('rounded-lg border border-border-primary bg-bg-secondary px-4 py-3', className)}>
      <div className="text-xs text-text-muted">{label}</div>
      <div className={cn('mt-1 text-xl font-semibold tabular-nums', toneClasses[tone])}>
        {value}
      </div>
      {detail && <div className="mt-1 text-xs text-text-muted">{detail}</div>}
    </div>
  );
}
