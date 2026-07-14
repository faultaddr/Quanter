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
        'flex-1 overflow-auto bg-bg-primary px-4 py-5 md:px-6',
        className
      )}
    >
      <div className="mx-auto w-full max-w-[1440px]">
        {children}
      </div>
    </main>
  );
}
