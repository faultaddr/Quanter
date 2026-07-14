'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import PageContainer from '@/components/layout/PageContainer';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import Select from '@/components/ui/Select';
import Badge from '@/components/ui/Badge';
import { PageHeader, Section, SegmentedControl, StatusBadge } from '@/components/ui';
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
import EmptyState from '@/components/ui/EmptyState';
import { formatNumber, formatPercent, getChangeColorClass } from '@/lib/utils';
import type { ScanResult } from '@/types/api';

type ScoringMode = 'unified' | 'trend' | 'classic' | 'breakout' | 'momentum';
type ScanPreset = 'csi300-fast' | 'trend-full' | 'low-risk' | 'deep-fundamental';

export default function ScanPage() {
  const router = useRouter();
  const setActivePage = useAppStore((state) => state.setActivePage);
  const addHistory = useAppStore((state) => state.addHistory);
  const toast = useToast();

  const [market, setMarket] = useState('csi300');
  const [scanPreset, setScanPreset] = useState<ScanPreset>('csi300-fast');
  const [scoringMode, setScoringMode] = useState<ScoringMode>('unified');
  const [includeFundamentals, setIncludeFundamentals] = useState(false);
  const [selectedSignals, setSelectedSignals] = useState<string[]>([]);
  const [minPrice, setMinPrice] = useState('');
  const [maxPrice, setMaxPrice] = useState('');
  const [results, setResults] = useState<ScanResult[]>([]);
  const [hasScanned, setHasScanned] = useState(false);

  // 因子筛选选项
  const [factorType, setFactorType] = useState('momentum');
  const [minFactorScore, setMinFactorScore] = useState('60');
  const [enableFactorFilter, setEnableFactorFilter] = useState(false);

  // 风控过滤选项
  const [excludeST, setExcludeST] = useState(true);
  const [excludeSuspended, setExcludeSuspended] = useState(true);
  const [excludeLimit, setExcludeLimit] = useState(true);

  const { loading, execute: runScan } = useApi(monitorApi.scan);

  useEffect(() => {
    setActivePage('scan');
  }, [setActivePage]);

  const handleScan = async () => {
    const params: Record<string, any> = {
      market,
      use_unified_score: scoringMode === 'unified',
      use_trend_score: scoringMode === 'trend',
      use_breakout_score: scoringMode === 'breakout',
      use_momentum_score: scoringMode === 'momentum',
      include_fundamentals: includeFundamentals,
      include_market_state: includeFundamentals,
    };

    // 基本筛选
    if (minPrice) params.min_price = Number(minPrice);
    if (maxPrice) params.max_price = Number(maxPrice);
    if (selectedSignals.length > 0) params.signals = selectedSignals;

    // 因子筛选
    if (enableFactorFilter) {
      params.enable_factor_filter = true;
      params.factor_type = factorType;
      if (minFactorScore) params.min_factor_score = Number(minFactorScore);
    }

    // 风控过滤
    params.exclude_st = excludeST;
    params.exclude_suspended = excludeSuspended;
    params.exclude_limit = excludeLimit;

    setHasScanned(false);
    const data = await runScan(params);
    if (data) {
      setResults(data);
      setHasScanned(true);
      toast.success(`扫描完成，共发现 ${data.length} 只符合条件的股票`);
    }
  };

  const marketOptions = [
    { value: 'csi300', label: '沪深300' },
    { value: 'csi1000', label: '中证1000' },
  ];

  const scoringModes: Array<{ value: ScoringMode; label: string }> = [
    { value: 'unified', label: '统一' },
    { value: 'trend', label: '趋势' },
    { value: 'classic', label: '经典' },
    { value: 'breakout', label: '突破' },
    { value: 'momentum', label: '动量' },
  ];

  const scanPresets: Array<{ value: ScanPreset; label: string; description: string }> = [
    { value: 'csi300-fast', label: '沪深300快扫', description: '约30秒' },
    { value: 'trend-full', label: '趋势扩展', description: '中证1000' },
    { value: 'low-risk', label: '低风险观察', description: '风控优先' },
    { value: 'deep-fundamental', label: '深度基本面', description: '更完整' },
  ];

  const applyPreset = (preset: ScanPreset) => {
    setScanPreset(preset);
    if (preset === 'csi300-fast') {
      setMarket('csi300');
      setScoringMode('unified');
      setIncludeFundamentals(false);
      setExcludeST(true);
      setExcludeSuspended(true);
      setExcludeLimit(true);
      setSelectedSignals([]);
    }
    if (preset === 'trend-full') {
      setMarket('csi1000');
      setScoringMode('trend');
      setIncludeFundamentals(false);
      setSelectedSignals(['均线多头']);
    }
    if (preset === 'low-risk') {
      setMarket('csi300');
      setScoringMode('unified');
      setIncludeFundamentals(false);
      setExcludeST(true);
      setExcludeSuspended(true);
      setExcludeLimit(true);
      setSelectedSignals([]);
    }
    if (preset === 'deep-fundamental') {
      setMarket('csi300');
      setScoringMode('unified');
      setIncludeFundamentals(true);
      setSelectedSignals([]);
    }
  };

  return (
    <PageContainer>
      <div className="space-y-6">
        <PageHeader
          eyebrow="Candidate Scan"
          title="智能选股"
          description="先选扫描意图，再用统一评分和风控过滤生成候选池。"
          meta={
            <>
              <StatusBadge tone={includeFundamentals ? 'warning' : 'success'}>
                {includeFundamentals ? '深度数据' : '快速模式'}
              </StatusBadge>
              <StatusBadge tone="muted">{market === 'csi300' ? '沪深300' : '中证1000'}</StatusBadge>
            </>
          }
          actions={
            <Button onClick={handleScan} loading={loading}>
              开始扫描
            </Button>
          }
        />

        {/* Scan Parameters */}
        <Section
          title="扫描条件"
          description="推荐先从预设开始，再展开细节过滤。"
          framed
        >
          <div className="mb-4">
            <label className="mb-2 block text-sm font-medium text-text-secondary">
              扫描预设
            </label>
            <SegmentedControl
              options={scanPresets}
              value={scanPreset}
              onChange={applyPreset}
            />
          </div>

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

          <div className="mt-4">
            <label className="block text-sm font-medium text-text-secondary mb-2">
              评分模式
            </label>
            <SegmentedControl
              options={scoringModes}
              value={scoringMode}
              onChange={setScoringMode}
              compact
            />
          </div>

          <div className="mt-4">
            <label htmlFor="includeFundamentals" className="flex items-center gap-2 text-sm text-text-secondary">
              <input
                type="checkbox"
                id="includeFundamentals"
                checked={includeFundamentals}
                onChange={(e) => setIncludeFundamentals(e.target.checked)}
                className="rounded"
              />
              <span>深度数据</span>
            </label>
          </div>

          {/* Signal Filters */}
          <div className="mt-4">
            <label className="block text-sm font-medium text-text-secondary mb-2">
              信号筛选
            </label>
            <div className="flex flex-wrap gap-2">
              {['MACD金叉', 'KDJ超卖', '放量突破', '均线多头', 'RSI超卖', '突破新高'].map((signal) => (
                <Button
                  key={signal}
                  size="sm"
                  variant={selectedSignals.includes(signal) ? 'primary' : 'secondary'}
                  onClick={() => {
                    setSelectedSignals(prev =>
                      prev.includes(signal)
                        ? prev.filter(s => s !== signal)
                        : [...prev, signal]
                    );
                  }}
                >
                  {signal}
                </Button>
              ))}
              {selectedSignals.length > 0 && (
                <Button size="sm" variant="ghost" onClick={() => setSelectedSignals([])}>
                  清除
                </Button>
              )}
            </div>
          </div>

          {/* Factor Filters */}
          <div className="mt-4 pt-4 border-t border-border-primary">
            <div className="flex items-center gap-2 mb-2">
              <input
                type="checkbox"
                id="enableFactorFilter"
                checked={enableFactorFilter}
                onChange={(e) => setEnableFactorFilter(e.target.checked)}
                className="rounded"
              />
              <label htmlFor="enableFactorFilter" className="text-sm font-medium text-text-secondary">
                启用因子筛选
              </label>
            </div>
            {enableFactorFilter && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 ml-6">
                <Select
                  label="因子类型"
                  options={[
                    { value: 'momentum', label: '动量因子' },
                    { value: 'value', label: '价值因子' },
                    { value: 'quality', label: '质量因子' },
                    { value: 'growth', label: '成长因子' },
                  ]}
                  value={factorType}
                  onChange={(e) => setFactorType(e.target.value)}
                />
                <Input
                  label="最低因子评分"
                  type="number"
                  value={minFactorScore}
                  onChange={(e) => setMinFactorScore(e.target.value)}
                  placeholder="60"
                />
              </div>
            )}
          </div>

          {/* Risk Control Filters */}
          <div className="mt-4 pt-4 border-t border-border-primary">
            <label className="block text-sm font-medium text-text-secondary mb-2">
              风控过滤
            </label>
            <div className="flex flex-wrap gap-4">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={excludeST}
                  onChange={(e) => setExcludeST(e.target.checked)}
                  className="rounded"
                />
                <span>排除ST股</span>
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={excludeSuspended}
                  onChange={(e) => setExcludeSuspended(e.target.checked)}
                  className="rounded"
                />
                <span>排除停牌股</span>
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={excludeLimit}
                  onChange={(e) => setExcludeLimit(e.target.checked)}
                  className="rounded"
                />
                <span>排除涨跌停股</span>
              </label>
            </div>
          </div>
        </Section>

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
                  <TableHead className="text-right">下一步</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {results.map((item) => (
                  <TableRow
                    key={item.symbol}
                    hoverable
                    className="cursor-pointer"
                    onClick={() => {
                      addHistory({
                        type: 'stock',
                        title: `${item.name} (${item.symbol})`,
                        path: `/analyze?symbol=${item.symbol}`
                      });
                      router.push(`/analyze?symbol=${item.symbol}`);
                    }}
                  >
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
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-2">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={(event) => {
                            event.stopPropagation();
                            addHistory({
                              type: 'stock',
                              title: `${item.name} (${item.symbol})`,
                              path: `/analyze?symbol=${item.symbol}`,
                            });
                            router.push(`/analyze?symbol=${item.symbol}`);
                          }}
                        >
                          分析
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={(event) => {
                            event.stopPropagation();
                            router.push(`/backtest?symbol=${item.symbol}`);
                          }}
                        >
                          回测
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>
        )}

        {hasScanned && !loading && results.length === 0 && (
          <EmptyState
            title="暂无符合条件股票"
            description="可以调整市场范围、评分方式或过滤条件后重新扫描"
          />
        )}
      </div>
    </PageContainer>
  );
}
