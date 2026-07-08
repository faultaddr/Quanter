'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useAppStore } from '@/stores/useAppStore';
import { getPageKeyFromPath } from '@/lib/navigation';
import { cn } from '@/lib/utils';
import { formatDate } from '@/lib/utils';

const coreItems = [
  {
    key: 'analyze',
    label: '股票分析',
    href: '/analyze',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
  },
  {
    key: 'backtest',
    label: '策略回测',
    href: '/backtest',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    ),
  },
  {
    key: 'monitor',
    label: '实时监控',
    href: '/monitor',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
      </svg>
    ),
  },
  {
    key: 'model',
    label: 'ML模型',
    href: '/model',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
  },
];

const smartItems = [
  {
    key: 'scan',
    label: '智能选股',
    href: '/scan',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    ),
  },
  {
    key: 'picks',
    label: '智能荐股',
    href: '/picks',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
      </svg>
    ),
  },
  {
    key: 'factors',
    label: '因子研究',
    href: '/factors',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
  },
  {
    key: 'risk',
    label: '组合风控',
    href: '/risk',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
      </svg>
    ),
  },
];

export default function AppSidebar() {
  const pathname = usePathname() ?? '/';
  const activePage = getPageKeyFromPath(pathname);
  const setActivePage = useAppStore((state) => state.setActivePage);
  const history = useAppStore((state) => state.history);
  const sidebarCollapsed = useAppStore((state) => state.sidebarCollapsed);

  return (
    <aside
      className={cn(
        'bg-bg-secondary border-r border-border-primary flex flex-col transition-all duration-300',
        sidebarCollapsed ? 'w-16' : 'w-56'
      )}
    >
      {/* Core Functions */}
      {!sidebarCollapsed && (
        <div className="p-3 pt-4">
          <h3 className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
            核心功能
          </h3>
        </div>
      )}
      <nav className="px-3 space-y-1">
        {coreItems.map((item) => (
          <Link
            key={item.key}
            href={item.href}
            onClick={() => setActivePage(item.key)}
            className={cn(
              'flex items-center gap-3 px-3 py-2 rounded-lg transition-colors',
              activePage === item.key
                ? 'bg-primary/20 text-primary'
                : 'text-text-secondary hover:text-text-primary hover:bg-bg-tertiary'
            )}
            title={sidebarCollapsed ? item.label : undefined}
          >
            {item.icon}
            {!sidebarCollapsed && <span className="text-sm">{item.label}</span>}
          </Link>
        ))}
      </nav>

      {/* Smart Features */}
      {!sidebarCollapsed && (
        <div className="p-3 pt-4 border-t border-border-primary mt-2">
          <h3 className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
            智能功能
          </h3>
        </div>
      )}
      <nav className={cn('px-3 space-y-1', sidebarCollapsed && 'mt-2')}>
        {smartItems.map((item) => (
          <Link
            key={item.key}
            href={item.href}
            onClick={() => setActivePage(item.key)}
            className={cn(
              'flex items-center gap-3 px-3 py-2 rounded-lg transition-colors',
              activePage === item.key
                ? 'bg-primary/20 text-primary'
                : 'text-text-secondary hover:text-text-primary hover:bg-bg-tertiary'
            )}
            title={sidebarCollapsed ? item.label : undefined}
          >
            {item.icon}
            {!sidebarCollapsed && <span className="text-sm">{item.label}</span>}
          </Link>
        ))}
      </nav>

      {/* Recent History */}
      {!sidebarCollapsed && history.length > 0 && (
        <div className="flex-1 p-3 border-t border-border-primary mt-2">
          <h3 className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">
            最近记录
          </h3>
          <nav className="space-y-1">
            {history.slice(0, 10).map((item) => (
              <Link
                key={item.id}
                href={item.path}
                onClick={() => setActivePage(item.type)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-text-secondary hover:text-text-primary hover:bg-bg-tertiary transition-colors"
              >
                <span className="text-sm truncate">{item.title}</span>
              </Link>
            ))}
          </nav>
        </div>
      )}

      {/* System Status */}
      <div className={cn(
        'p-3 border-t border-border-primary mt-auto',
        sidebarCollapsed && 'flex justify-center'
      )}>
        <div className={cn(
          'flex items-center gap-2',
          sidebarCollapsed && 'flex-col'
        )}>
          <div className="w-2 h-2 bg-success rounded-full animate-pulse" />
          {!sidebarCollapsed && (
            <span className="text-xs text-text-muted">系统运行正常</span>
          )}
        </div>
      </div>
    </aside>
  );
}
