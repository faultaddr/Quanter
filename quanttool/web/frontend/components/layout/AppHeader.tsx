'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import StockSearch from '@/components/stock/StockSearch';
import { StatusBadge } from '@/components/ui';
import { useAppStore } from '@/stores/useAppStore';
import { cn } from '@/lib/utils';

export default function AppHeader() {
  const router = useRouter();
  const setActivePage = useAppStore((state) => state.setActivePage);
  const addHistory = useAppStore((state) => state.addHistory);
  const sidebarCollapsed = useAppStore((state) => state.sidebarCollapsed);
  const mobileSidebarOpen = useAppStore((state) => state.mobileSidebarOpen);
  const toggleSidebar = useAppStore((state) => state.toggleSidebar);
  const toggleMobileSidebar = useAppStore((state) => state.toggleMobileSidebar);
  const theme = useAppStore((state) => state.theme);
  const toggleTheme = useAppStore((state) => state.toggleTheme);

  const handleStockSelect = (symbol: string, name: string) => {
    setActivePage('analyze');
    addHistory({
      type: 'stock',
      title: `${name} (${symbol})`,
      path: `/analyze?symbol=${symbol}&name=${encodeURIComponent(name)}`,
    });
    router.push(`/analyze?symbol=${symbol}&name=${encodeURIComponent(name)}`);
  };

  const handleNavigationToggle = () => {
    if (window.matchMedia('(max-width: 767px)').matches) {
      toggleMobileSidebar();
      return;
    }
    toggleSidebar();
  };

  return (
    <header className="sticky top-0 z-40 flex h-14 items-center justify-between border-b border-border-primary bg-bg-secondary px-4">
      <div className="flex min-w-0 items-center gap-3">
        <button
          id="app-navigation-toggle"
          type="button"
          onClick={handleNavigationToggle}
          className="rounded-lg p-2 text-text-secondary transition-colors hover:bg-bg-tertiary hover:text-text-primary"
          title={sidebarCollapsed ? '展开导航' : '切换导航'}
          aria-label="切换导航"
          aria-controls="app-navigation"
          aria-expanded={mobileSidebarOpen}
        >
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>

        <Link
          href="/"
          className="flex items-center gap-2"
          onClick={() => setActivePage('overview')}
        >
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
            <svg className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          </div>
          <div className="hidden leading-tight sm:block">
            <div className="text-sm font-semibold text-text-primary">QuantTool</div>
            <div className="text-xs text-text-muted">A股研究工作台</div>
          </div>
        </Link>
      </div>

      <div className="mx-4 hidden min-w-[280px] max-w-xl flex-1 md:block">
        <StockSearch
          onSelect={handleStockSelect}
          placeholder="搜索股票代码或名称"
          className="[&>button]:px-3 [&>button]:py-2 [&>button]:text-sm"
        />
      </div>

      <div className="flex items-center gap-2">
        <StatusBadge tone="muted" className="hidden sm:inline-flex">本地模式</StatusBadge>

        <Link
          href="/analyze"
          className="rounded-lg p-2 text-text-secondary transition-colors hover:bg-bg-tertiary hover:text-text-primary md:hidden"
          title="搜索股票"
          onClick={() => setActivePage('analyze')}
        >
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </Link>

        <button
          onClick={toggleTheme}
          className={cn(
            'rounded-lg p-2 text-text-secondary transition-colors hover:bg-bg-tertiary hover:text-text-primary',
            theme === 'light' && 'text-warning'
          )}
          title={theme === 'dark' ? '切换到浅色模式' : '切换到深色模式'}
        >
          {theme === 'dark' ? (
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
            </svg>
          ) : (
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
            </svg>
          )}
        </button>

        <a
          href="/docs"
          target="_blank"
          className="rounded-lg p-2 text-text-secondary transition-colors hover:bg-bg-tertiary hover:text-text-primary"
          title="API 文档"
        >
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </a>
      </div>
    </header>
  );
}
