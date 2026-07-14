'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { StatusBadge } from '@/components/ui';
import { useAppStore } from '@/stores/useAppStore';
import { getPageKeyFromPath } from '@/lib/navigation';
import { cn } from '@/lib/utils';

interface NavItem {
  key: string;
  label: string;
  href: string;
  icon: React.ReactNode;
}

interface NavGroup {
  title: string;
  items: NavItem[];
}

const iconClass = 'h-5 w-5 shrink-0';

const navigationGroups: NavGroup[] = [
  {
    title: '市场',
    items: [
      {
        key: 'overview',
        label: '盘面概览',
        href: '/',
        icon: (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 13.5h4l3-7 4 11 3-4h4" />
          </svg>
        ),
      },
      {
        key: 'monitor',
        label: '实时监控',
        href: '/monitor',
        icon: (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
          </svg>
        ),
      },
    ],
  },
  {
    title: '研究',
    items: [
      {
        key: 'scan',
        label: '智能选股',
        href: '/scan',
        icon: (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        ),
      },
      {
        key: 'research',
        label: '产业链研究',
        href: '/research',
        icon: (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h10" />
          </svg>
        ),
      },
      {
        key: 'analyze',
        label: '股票分析',
        href: '/analyze',
        icon: (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-6m4 6V7m4 10v-3M5 21h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v14a2 2 0 002 2z" />
          </svg>
        ),
      },
      {
        key: 'factors',
        label: '因子研究',
        href: '/factors',
        icon: (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        ),
      },
      {
        key: 'picks',
        label: '智能荐股',
        href: '/picks',
        icon: (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
          </svg>
        ),
      },
    ],
  },
  {
    title: '验证',
    items: [
      {
        key: 'backtest',
        label: '策略回测',
        href: '/backtest',
        icon: (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
          </svg>
        ),
      },
      {
        key: 'risk',
        label: '组合风控',
        href: '/risk',
        icon: (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
        ),
      },
    ],
  },
  {
    title: '模型',
    items: [
      {
        key: 'model',
        label: 'ML模型',
        href: '/model',
        icon: (
          <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
        ),
      },
    ],
  },
];

function historyPageKey(type: string): string {
  return type === 'stock' ? 'analyze' : type;
}

export default function AppSidebar() {
  const pathname = usePathname() ?? '/';
  const activePage = getPageKeyFromPath(pathname);
  const setActivePage = useAppStore((state) => state.setActivePage);
  const history = useAppStore((state) => state.history);
  const sidebarCollapsed = useAppStore((state) => state.sidebarCollapsed);
  const mobileSidebarOpen = useAppStore((state) => state.mobileSidebarOpen);
  const closeMobileSidebar = useAppStore((state) => state.closeMobileSidebar);
  const sidebarRef = useRef<HTMLElement>(null);
  const [isMobileViewport, setIsMobileViewport] = useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(max-width: 767px)');
    const syncViewport = () => {
      setIsMobileViewport(mediaQuery.matches);
      if (!mediaQuery.matches) {
        closeMobileSidebar();
      }
    };

    syncViewport();
    mediaQuery.addEventListener('change', syncViewport);
    return () => mediaQuery.removeEventListener('change', syncViewport);
  }, [closeMobileSidebar]);

  useEffect(() => {
    if (!isMobileViewport || !mobileSidebarOpen) {
      return;
    }

    sidebarRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        closeMobileSidebar();
        window.requestAnimationFrame(() => {
          document.getElementById('app-navigation-toggle')?.focus();
        });
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [closeMobileSidebar, isMobileViewport, mobileSidebarOpen]);

  const mobileDrawerHidden = isMobileViewport && !mobileSidebarOpen;
  const showLabels = isMobileViewport || !sidebarCollapsed;

  return (
    <>
      {mobileSidebarOpen && (
        <button
          type="button"
          aria-label="关闭导航"
          className="fixed inset-x-0 bottom-0 top-14 z-20 bg-black/40 md:hidden"
          onClick={closeMobileSidebar}
        />
      )}
      <aside
        ref={sidebarRef}
        id="app-navigation"
        aria-hidden={mobileDrawerHidden}
        tabIndex={mobileSidebarOpen ? -1 : undefined}
        className={cn(
          'fixed bottom-0 left-0 top-14 z-30 flex w-60 shrink-0 flex-col border-r border-border-primary bg-bg-secondary transition-[transform,width] duration-200 md:static md:z-auto md:h-auto md:translate-x-0',
          mobileSidebarOpen ? 'translate-x-0' : '-translate-x-full',
          sidebarCollapsed ? 'md:w-16' : 'md:w-60'
        )}
      >
      <div className="flex-1 overflow-y-auto py-3">
        {navigationGroups.map((group, groupIndex) => (
          <div
            key={group.title}
            className={cn(groupIndex > 0 && 'mt-3 border-t border-border-primary pt-3')}
          >
            {showLabels && (
              <div className="px-4 pb-2 text-xs font-medium uppercase tracking-wider text-text-muted">
                {group.title}
              </div>
            )}
            <nav className="space-y-1 px-2">
              {group.items.map((item) => (
                <Link
                  key={item.key}
                  href={item.href}
                  tabIndex={mobileDrawerHidden ? -1 : undefined}
                  onClick={() => {
                    setActivePage(item.key);
                    closeMobileSidebar();
                  }}
                  className={cn(
                    'flex h-9 items-center gap-3 rounded-lg px-3 text-sm transition-colors',
                    activePage === item.key
                      ? 'bg-primary/15 text-primary'
                      : 'text-text-secondary hover:bg-bg-tertiary hover:text-text-primary',
                    !showLabels && 'justify-center px-0'
                  )}
                  title={!showLabels ? item.label : undefined}
                >
                  {item.icon}
                  {showLabels && <span className="truncate">{item.label}</span>}
                </Link>
              ))}
            </nav>
          </div>
        ))}

        {showLabels && history.length > 0 && (
          <div className="mt-3 border-t border-border-primary px-2 pt-3">
            <div className="px-2 pb-2 text-xs font-medium uppercase tracking-wider text-text-muted">
              最近记录
            </div>
            <nav className="space-y-1">
              {history.slice(0, 6).map((item) => (
                <Link
                  key={item.id}
                  href={item.path}
                  tabIndex={mobileDrawerHidden ? -1 : undefined}
                  onClick={() => {
                    setActivePage(historyPageKey(item.type));
                    closeMobileSidebar();
                  }}
                  className="flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm text-text-secondary transition-colors hover:bg-bg-tertiary hover:text-text-primary"
                >
                  <span className="truncate">{item.title}</span>
                </Link>
              ))}
            </nav>
          </div>
        )}
      </div>

      <div className={cn('border-t border-border-primary p-3', !showLabels && 'flex justify-center')}>
        {!showLabels ? (
          <span className="h-2 w-2 rounded-full bg-success" title="系统运行正常" />
        ) : (
          <div className="flex items-center justify-between gap-2">
            <StatusBadge tone="success">系统运行正常</StatusBadge>
            <span className="text-xs text-text-muted">本地</span>
          </div>
        )}
      </div>
      </aside>
    </>
  );
}
