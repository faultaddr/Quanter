'use client';

import { cn } from '@/lib/utils';

interface PageContainerProps {
  children: React.ReactNode;
  className?: string;
}

export default function PageContainer({ children, className }: PageContainerProps) {
  return (
    <main
      className={cn(
        'flex-1 overflow-auto bg-bg-primary p-6',
        className
      )}
    >
      {children}
    </main>
  );
}
