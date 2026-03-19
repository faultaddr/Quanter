'use client';

import { useState, useEffect } from 'react';
import PageContainer from '@/components/layout/PageContainer';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import Select from '@/components/ui/Select';
import Badge from '@/components/ui/Badge';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '@/components/ui/Table';
import { useAppStore } from '@/stores/useAppStore';
import { monitorApi } from '@/lib/api/monitor';
import { useApi } from '@/hooks/useApi';
import { useToast } from '@/hooks/useToast';
import Loading from '@/components/ui/Loading';
import { formatDate, formatNumber, formatPercent, getChangeColorClass } from '@/lib/utils';
import type { ScanResult } from '@/types/api';

export default function ScanPage() {
  const setActivePage = useAppStore((state) => state.setActivePage);
  const addHistory = useAppStore((state) => state.addHistory);
  const toast = useToast();

  const [market, setMarket] = useState('all');
  const [minPrice, setMinPrice] = useState('');
  const [maxPrice, setMaxPrice] = useState('');
  const [results, setResults] = useState<ScanResult[]>([]);

  const { loading, execute: runScan } = useApi(monitorApi.scan);

  useEffect(() => {
    setActivePage('scan');
  }, [setActivePage]);

  const handleScan = async () => {
    const params: Record<string, any> = {};

    if (market !== 'all') params.market = market;
    if (minPrice) params.min_price = Number(minPrice);
    if (maxPrice) params.max_price = Number(maxPrice);

    const data = await runScan(params);
    if (data) {
      setResults(data);
      toast.success(`扫描完成，共发现 ${data.length} 只符合条件的股票`);
    }
  };

  const marketOptions = [
    { value: 'all', label: '全部市场' },
    { value: 'sh', label: '上海证券交易所' },
    { value: 'sz', label: '深圳证券交易所' },
    { value: 'bj', label: '北京证券交易所' },
  ];

  return (
    <PageContainer>
      <div className="space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold text-text-primary">智能选股</h1>
          <p className="text-text-muted mt-1">基于技术指标和量化因子，全市场扫描符合条件的股票</p>
        </div>

        {/* Scan Parameters */}
        <Card title="扫描条件">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Select
              label="市场"
              options={marketOptions}
              value={market}
              onChange={(e) => setMarket(e.target.value)}
            />
            <Input
              label="最低价格"
              type="number"
              value={minPrice}
              onChange={(e) => setMinPrice(e.target.value)}
              placeholder="不限"
            />
            <Input
              label="最高价格"
              type="number"
              value={maxPrice}
              onChange={(e) => setMaxPrice(e.target.value)}
              placeholder="不限"
            />
            <div className="flex items-end">
              <Button onClick={handleScan} loading={loading} className="w-full">
                开始扫描
              </Button>
            </div>
          </div>

          {/* Signal Filters */}
          <div className="mt-4">
            <label className="block text-sm font-medium text-text-secondary mb-2">
              信号筛选
            </label>
            <div className="flex flex-wrap gap-2">
              {['MACD金叉', 'KDJ超卖', '放量突破', '均线多头', 'RSI超卖', '突破新高'].map((signal) => (
                <Button key={signal} size="sm" variant="secondary">
                  {signal}
                </Button>
              ))}
            </div>
          </div>
        </Card>

        {/* Results */}
        {results.length > 0 && (
          <Card title={`扫描结果 (${results.length})`}>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>代码</TableHead>
                  <TableHead>名称</TableHead>
                  <TableHead className="text-right">价格</TableHead>
                  <TableHead className="text-right">涨跌幅</TableHead>
                  <TableHead>信号</TableHead>
                  <TableHead className="text-right">评分</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {results.map((item) => (
                  <TableRow key={item.symbol} hoverable>
                    <TableCell className="font-mono">{item.symbol}</TableCell>
                    <TableCell>{item.name}</TableCell>
                    <TableCell className="text-right">{formatNumber(item.price)}</TableCell>
                    <TableCell className={`text-right ${getChangeColorClass(item.change_pct)}`}>
                      {formatPercent(item.change_pct)}
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-wrap gap-1">
                        {item.signals.slice(0, 3).map((signal, index) => (
                          <Badge
                            key={index}
                            variant={signal.type === 'buy' ? 'success' : signal.type === 'sell' ? 'danger' : 'warning'}
                            size="sm"
                          >
                            {signal.name}
                          </Badge>
                        ))}
                        {item.signals.length > 3 && (
                          <Badge variant="default" size="sm">
                            +{item.signals.length - 3}
                          </Badge>
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="text-right">
                      {item.score !== undefined ? (
                        <span className={item.score >= 70 ? 'text-success' : item.score >= 40 ? 'text-warning' : 'text-text-secondary'}>
                          {item.score}
                        </span>
                      ) : '-'}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>
        )}
      </div>
    </PageContainer>
  );
}
