import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface Toast {
  id: string;
  message: string;
  type: 'success' | 'error' | 'warning' | 'info';
}

interface HistoryItem {
  id: string;
  type: 'stock' | 'backtest' | 'model';
  title: string;
  path: string;
  timestamp: number;
}

interface AppState {
  // 当前活跃页面
  activePage: string;
  setActivePage: (page: string) => void;

  // 主题
  theme: 'dark' | 'light';
  toggleTheme: () => void;

  // 侧边栏折叠
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;

  // 历史记录
  history: HistoryItem[];
  addHistory: (item: Omit<HistoryItem, 'id' | 'timestamp'>) => void;
  clearHistory: () => void;

  // Toast 消息
  toasts: Toast[];
  showToast: (message: string, type?: Toast['type']) => void;
  removeToast: (id: string) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      activePage: 'overview',
      setActivePage: (page) => set({ activePage: page }),

      theme: 'dark',
      toggleTheme: () =>
        set((state) => ({
          theme: state.theme === 'dark' ? 'light' : 'dark',
        })),

      sidebarCollapsed: false,
      toggleSidebar: () =>
        set((state) => ({
          sidebarCollapsed: !state.sidebarCollapsed,
        })),

      history: [],
      addHistory: (item) =>
        set((state) => ({
          history: [
            { ...item, id: Date.now().toString(), timestamp: Date.now() },
            ...state.history,
          ].slice(0, 20),
        })),
      clearHistory: () => set({ history: [] }),

      toasts: [],
      showToast: (message, type = 'info') => {
        const id = Date.now().toString();
        set((state) => ({
          toasts: [...state.toasts, { id, message, type }],
        }));
        setTimeout(() => {
          set((state) => ({
            toasts: state.toasts.filter((t) => t.id !== id),
          }));
        }, 3000);
      },
      removeToast: (id) =>
        set((state) => ({
          toasts: state.toasts.filter((t) => t.id !== id),
        })),
    }),
    {
      name: 'quanttool-app-storage',
      partialize: (state) => ({
        theme: state.theme,
        sidebarCollapsed: state.sidebarCollapsed,
        history: state.history,
      }),
    }
  )
);
