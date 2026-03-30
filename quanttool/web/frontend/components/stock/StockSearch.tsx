'use client';

import { useState, useCallback, useRef } from 'react';
import Input from '@/components/ui/Input';
import Button from '@/components/ui/Button';
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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const searchStock = useCallback(async (query: string) => {
    if (!query.trim()) {
      setError('请输入股票代码');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const results = await stockApi.search(query);
      if (!results || results.length === 0) {
        setError('未找到匹配的股票');
        return;
      }
      // 直接选择第一个结果
      const result = results[0];
      onSelect(result.symbol, result.name);
      setKeyword('');
    } catch (err) {
      console.error('Search error:', err);
      setError('搜索失败，请重试');
    } finally {
      setLoading(false);
    }
  }, [onSelect]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      searchStock(keyword);
    }
  };

  return (
    <div className={cn('relative flex gap-2', className)}>
      <Input
        ref={inputRef}
        value={keyword}
        onChange={(e) => {
          setKeyword(e.target.value);
          setError(null);
        }}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        className="flex-1"
        icon={
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        }
      />
      <Button
        onClick={() => searchStock(keyword)}
        loading={loading}
        disabled={!keyword.trim()}
      >
        搜索
      </Button>
      {error && (
        <span className="absolute -bottom-6 left-0 text-sm text-danger whitespace-nowrap">{error}</span>
      )}
    </div>
  );
}
