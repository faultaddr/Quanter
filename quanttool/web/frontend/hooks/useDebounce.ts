import { useState, useEffect } from 'react';

/**
 * 防抖 Hook
 */
export function useDebounce<T>(value: T, delay: number = 300): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(timer);
    };
  }, [value, delay]);

  return debouncedValue;
}

/**
 * 节流 Hook
 */
export function useThrottle<T>(value: T, limit: number = 300): T {
  const [throttledValue, setThrottledValue] = useState<T>(value);
  const [lastRun, setLastRun] = useState<number>(Date.now());

  useEffect(() => {
    const now = Date.now();
    if (now - lastRun >= limit) {
      setThrottledValue(value);
      setLastRun(now);
    }
  }, [value, limit, lastRun]);

  return throttledValue;
}
