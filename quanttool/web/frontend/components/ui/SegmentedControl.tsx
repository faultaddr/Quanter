'use client';

import { cn } from '@/lib/utils';

export interface SegmentedControlOption<T extends string> {
  value: T;
  label: string;
  description?: string;
}

interface SegmentedControlProps<T extends string> {
  options: Array<SegmentedControlOption<T>>;
  value: T;
  onChange: (value: T) => void;
  className?: string;
  compact?: boolean;
}

export default function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  className,
  compact = false,
}: SegmentedControlProps<T>) {
  return (
    <div className={cn('inline-flex flex-wrap gap-1 rounded-lg border border-border-primary bg-bg-tertiary p-1', className)}>
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            className={cn(
              'rounded-md font-medium transition-colors',
              compact ? 'px-2.5 py-1 text-xs' : 'px-3 py-1.5 text-sm',
              active
                ? 'bg-primary text-white'
                : 'text-text-secondary hover:bg-bg-secondary hover:text-text-primary'
            )}
          >
            <span>{option.label}</span>
            {option.description && !compact && (
              <span className="ml-2 text-xs opacity-80">{option.description}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}
