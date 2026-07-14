'use client';

import { cn } from '@/lib/utils';

interface SectionProps {
  title?: string;
  description?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
  framed?: boolean;
}

export default function Section({
  title,
  description,
  action,
  children,
  className,
  bodyClassName,
  framed = false,
}: SectionProps) {
  return (
    <section
      className={cn(
        framed && 'rounded-lg border border-border-primary bg-bg-secondary',
        className
      )}
    >
      {(title || description || action) && (
        <div
          className={cn(
            'flex items-start justify-between gap-4',
            framed ? 'border-b border-border-primary px-4 py-3' : 'mb-3'
          )}
        >
          <div>
            {title && <h2 className="text-base font-semibold text-text-primary">{title}</h2>}
            {description && <p className="mt-1 text-sm text-text-muted">{description}</p>}
          </div>
          {action && <div className="shrink-0">{action}</div>}
        </div>
      )}
      <div className={cn(framed && 'p-4', bodyClassName)}>{children}</div>
    </section>
  );
}
