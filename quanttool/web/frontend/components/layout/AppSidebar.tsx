'use client';

import Link from 'next/link';
import { useAppStore } from '@/stores/useAppStore';
import { cn } from '@/lib/utils';
import { formatDate } from '@/lib/utils';

const sidebarItems = [
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
];

export default function AppSidebar() {
  const activePage = useAppStore((state) => state.activePage);
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
      {/* Quick Access */}
      <div className="p-3">
        <h3
          className={cn(
            'text-xs font-medium text-text-muted uppercase tracking-wider mb-2',
            sidebarCollapsed && 'hidden'
          )}
        >
          快速入口
        </h3>
        <nav className="space-y-1">
          {sidebarItems.map((item) => (
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
      </div>

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
