'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import Input from '@/components/ui/Input';
import { useDebounce } from '@/hooks/useDebounce';
import { stockApi } from '@/lib/api/stock';
import { cn } from '@/lib/utils';

interface StockSearchProps {
  onSelect: (symbol: string, name: string) => void;
  placeholder?: string;
  className?: string;
}

interface SearchResult {
  symbol: string;
  name: string;
  market?: string;
}

export default function StockSearch({
  onSelect,
  placeholder = '输入股票代码或名称搜索',
  className,
}: StockSearchProps) {
  const [keyword, setKeyword] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const debouncedKeyword = useDebounce(keyword, 300);
  const containerRef = useRef<HTMLDivElement>(null);

  const searchStocks = useCallback(async (query: string) => {
    if (!query.trim()) {
      setResults([]);
      return;
    }

    setLoading(true);
    try {
      const data = await stockApi.search(query);
      setResults(data || []);
    } catch (error) {
      console.error('Search error:', error);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    searchStocks(debouncedKeyword);
  }, [debouncedKeyword, searchStocks]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelect = (result: SearchResult) => {
    setKeyword('');
    setResults([]);
    setShowDropdown(false);
    onSelect(result.symbol, result.name);
  };

  return (
    <div ref={containerRef} className={cn('relative', className)}>
      <Input
        value={keyword}
        onChange={(e) => {
          setKeyword(e.target.value);
          setShowDropdown(true);
        }}
        onFocus={() => setShowDropdown(true)}
        placeholder={placeholder}
        icon={
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        }
      />

      {/* Dropdown */}
      {showDropdown && (keyword.trim() || results.length > 0) && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-bg-secondary border border-border-primary rounded-lg shadow-lg z-50 max-h-80 overflow-auto">
          {loading ? (
            <div className="p-4 text-center text-text-muted">
              <svg className="animate-spin w-5 h-5 mx-auto" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            </div>
          ) : results.length > 0 ? (
            <ul>
              {results.map((result) => (
                <li
                  key={result.symbol}
                  className="px-4 py-2 hover:bg-bg-tertiary cursor-pointer transition-colors"
                  onClick={() => handleSelect(result)}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-text-primary">{result.name}</span>
                    <span className="text-text-muted text-sm">{result.symbol}</span>
                  </div>
                  {result.market && (
                    <div className="text-xs text-text-muted mt-0.5">{result.market}</div>
                  )}
                </li>
              ))}
            </ul>
          ) : keyword.trim() ? (
            <div className="p-4 text-center text-text-muted">
              未找到匹配的股票
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}
