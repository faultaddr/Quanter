'use client';

import { cn } from '@/lib/utils';
import { HTMLAttributes, forwardRef } from 'react';

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  title?: string;
  subtitle?: string;
  action?: React.ReactNode;
  noPadding?: boolean;
}

const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className, title, subtitle, action, noPadding, children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          'bg-bg-secondary rounded-lg border border-border-primary',
          className
        )}
        {...props}
      >
        {(title || subtitle || action) && (
          <div className="flex items-center justify-between px-4 py-3 border-b border-border-primary">
            <div>
              {title && (
                <h3 className="text-base font-medium text-text-primary">{title}</h3>
              )}
              {subtitle && (
                <p className="text-sm text-text-muted mt-0.5">{subtitle}</p>
              )}
            </div>
            {action && <div>{action}</div>}
          </div>
        )}
        <div className={noPadding ? '' : 'p-4'}>{children}</div>
      </div>
    );
  }
);

Card.displayName = 'Card';

export default Card;
