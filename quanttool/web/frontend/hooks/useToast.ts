import { useCallback, useMemo } from 'react';
import { useAppStore } from '@/stores/useAppStore';

export function useToast() {
  const showToast = useAppStore((state) => state.showToast);

  const success = useCallback(
    (message: string) => showToast(message, 'success'),
    [showToast]
  );

  const error = useCallback(
    (message: string) => showToast(message, 'error'),
    [showToast]
  );

  const warning = useCallback(
    (message: string) => showToast(message, 'warning'),
    [showToast]
  );

  const info = useCallback(
    (message: string) => showToast(message, 'info'),
    [showToast]
  );

  // 使用 useMemo 返回稳定引用的对象
  return useMemo(
    () => ({
      success,
      error,
      warning,
      info,
      show: showToast,
    }),
    [success, error, warning, info, showToast]
  );
}
