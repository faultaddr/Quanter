'use client';

import { cn } from '@/lib/utils';

interface StatusBadgeProps {
  children: React.ReactNode;
  tone?: 'default' | 'success' | 'danger' | 'warning' | 'primary' | 'muted';
  size?: 'sm' | 'md';
  className?: string;
}

const toneClasses: Record<NonNullable<StatusBadgeProps['tone']>, string> = {
  default: 'bg-bg-tertiary text-text-secondary',
  success: 'bg-success/15 text-success',
  danger: 'bg-danger/15 text-danger',
  warning: 'bg-warning/15 text-warning',
  primary: 'bg-primary/15 text-primary',
  muted: 'bg-slate-500/15 text-slate-300',
};

const sizeClasses: Record<NonNullable<StatusBadgeProps['size']>, string> = {
  sm: 'px-2 py-0.5 text-xs',
  md: 'px-2.5 py-1 text-sm',
};

export default function StatusBadge({
  children,
  tone = 'default',
  size = 'sm',
  className,
}: StatusBadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-md font-medium',
        toneClasses[tone],
        sizeClasses[size],
        className
      )}
    >
      {children}
    </span>
  );
}
