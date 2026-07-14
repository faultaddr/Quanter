'use client';

import { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import PageContainer from '@/components/layout/PageContainer';
import Button from '@/components/ui/Button';
import {
  MetricTile,
  PageHeader,
  Section,
  StatusBadge,
} from '@/components/ui';
import { useAppStore } from '@/stores/useAppStore';
import { monitorApi } from '@/lib/api/monitor';
import { formatNumber, formatPercent } from '@/lib/utils';
import type { RealtimeQuote } from '@/types/api';

const INDEX_SYMBOLS = ['000001', '399001', '399006', '000300'];

const INDEX_NAMES: Record<string, string> = {
  '000001': '上证指数',
  '399001': '深证成指',
  '399006': '创业板指',
  '000300': '沪深300',
};

const quickActions = [
  {
    title: '运行沪深300快扫',
    description: '用默认统一评分快速找候选',
    href: '/scan',
    page: 'scan',
  },
  {
    title: '打开单股分析',
    description: '搜索代码后进入工作台',
    href: '/analyze',
    page: 'analyze',
  },
  {
    title: '查看组合风控',
    description: '检查风险和黑名单暴露',
    href: '/risk',
    page: 'risk',
  },
];

const candidatePrompts = [
  { label: '趋势延续', detail: '统一评分 + 均线多头', href: '/scan' },
  { label: '放量突破', detail: '突破评分 + 成交量确认', href: '/scan' },
  { label: '低风险观察', detail: '排除ST/停牌/涨跌停', href: '/scan' },
];

export default function HomePage() {
  const setActivePage = useAppStore((state) => state.setActivePage);
  const history = useAppStore((state) => state.history);
  const [marketIndices, setMarketIndices] = useState<RealtimeQuote[]>([]);
  const [loading, setLoading] = useState(true);
  const [marketError, setMarketError] = useState<string | null>(null);

  const fetchMarketIndices = useCallback(async () => {
    setLoading(true);
    setMarketError(null);
    try {
      const data = await monitorApi.getQuotes(INDEX_SYMBOLS);
      const validData = (data || [])
        .filter((quote) => !quote.error && quote.price > 0)
        .map((quote) => ({
          ...quote,
          name: quote.name || INDEX_NAMES[quote.symbol] || quote.symbol,
        }));
      setMarketIndices(validData);
      if (validData.length === 0) {
        setMarketError('暂未拿到有效市场指数');
      }
    } catch (error) {
      console.error('Failed to fetch market indices:', error);
      setMarketError('市场指数加载失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setActivePage('overview');
    fetchMarketIndices();
    const interval = setInterval(fetchMarketIndices, 30000);
    return () => clearInterval(interval);
  }, [setActivePage, fetchMarketIndices]);

  const freshnessTone = marketError ? 'warning' : 'success';
  const recentItems = history.slice(0, 5);

  return (
    <PageContainer>
      <div className="space-y-6">
        <PageHeader
          eyebrow="Market Workspace"
          title="盘面概览"
          description="从市场状态进入候选扫描，再把结果推进到单股分析、回测和风控。"
          meta={
            <>
              <StatusBadge tone={freshnessTone}>{marketError ? '数据异常' : '30秒刷新'}</StatusBadge>
              <StatusBadge tone="muted">免费数据源优先</StatusBadge>
            </>
          }
          actions={
            <Button size="sm" variant="ghost" onClick={fetchMarketIndices} loading={loading}>
              刷新指数
            </Button>
          }
        />

        <Section
          title="市场快照"
          description={marketError || '主要指数与盘面温度，作为扫描前的第一眼判断。'}
          framed
        >
          {loading && marketIndices.length === 0 ? (
            <div className="py-8 text-center text-sm text-text-muted">加载市场指数...</div>
          ) : marketIndices.length === 0 ? (
            <div className="flex items-center justify-between rounded-lg border border-warning/30 bg-warning/10 px-4 py-3">
              <span className="text-sm text-warning">{marketError || '暂无市场数据'}</span>
              <Button size="sm" variant="ghost" onClick={fetchMarketIndices}>
                重新加载
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              {marketIndices.map((index) => {
                const positive = index.change_pct > 0;
                return (
                  <MetricTile
                    key={index.symbol}
                    label={index.name || index.symbol}
                    value={formatNumber(index.price)}
                    detail={`${positive ? '+' : ''}${formatNumber(index.change)} / ${formatPercent(index.change_pct)}`}
                    tone={positive ? 'positive' : index.change_pct < 0 ? 'negative' : 'muted'}
                  />
                );
              })}
            </div>
          )}
        </Section>

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
          <Section title="今日动作" framed>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
              {quickActions.map((action) => (
                <Link
                  key={action.href}
                  href={action.href}
                  onClick={() => setActivePage(action.page)}
                  className="group rounded-lg border border-border-primary bg-bg-tertiary px-4 py-3 transition-colors hover:border-primary hover:bg-bg-secondary"
                >
                  <span className="text-sm font-semibold text-text-primary group-hover:text-primary">
                    {action.title}
                  </span>
                  <p className="mt-1 text-sm text-text-muted">{action.description}</p>
                </Link>
              ))}
            </div>
          </Section>

          <Section title="系统状态" framed>
            <div className="space-y-3 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-text-muted">行情接口</span>
                <StatusBadge tone={marketError ? 'warning' : 'success'}>
                  {marketError ? '部分异常' : '正常'}
                </StatusBadge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-text-muted">扫描模式</span>
                <StatusBadge tone="primary">默认快扫</StatusBadge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-text-muted">候选流向</span>
                <StatusBadge tone="muted">选股 / 分析 / 回测</StatusBadge>
              </div>
            </div>
          </Section>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Section
            title="候选入口"
            description="还没有扫描结果时，先从常用预设进入。"
            action={
              <Link href="/scan" onClick={() => setActivePage('scan')} className="text-sm text-primary hover:text-primary-light">
                打开智能选股
              </Link>
            }
            framed
          >
            <div className="space-y-2">
              {candidatePrompts.map((candidate) => (
                <Link
                  key={candidate.label}
                  href={candidate.href}
                  onClick={() => setActivePage('scan')}
                  className="flex items-center justify-between rounded-lg border border-border-primary bg-bg-tertiary px-3 py-2 transition-colors hover:border-primary"
                >
                  <div>
                    <div className="text-sm font-medium text-text-primary">{candidate.label}</div>
                    <div className="text-xs text-text-muted">{candidate.detail}</div>
                  </div>
                  <StatusBadge tone="primary">扫描</StatusBadge>
                </Link>
              ))}
            </div>
          </Section>

          <Section title="最近工作" description="继续上次看过的股票、回测或模型。" framed>
            {recentItems.length === 0 ? (
              <div className="rounded-lg border border-dashed border-border-primary px-4 py-8 text-center">
                <div className="text-sm font-medium text-text-primary">暂无最近记录</div>
                <p className="mt-1 text-sm text-text-muted">先运行扫描或搜索股票，记录会出现在这里。</p>
              </div>
            ) : (
              <div className="space-y-2">
                {recentItems.map((item) => (
                  <Link
                    key={item.id}
                    href={item.path}
                    onClick={() => setActivePage(item.type === 'stock' ? 'analyze' : item.type)}
                    className="flex items-center justify-between rounded-lg border border-border-primary bg-bg-tertiary px-3 py-2 transition-colors hover:border-primary"
                  >
                    <span className="truncate text-sm text-text-primary">{item.title}</span>
                    <StatusBadge tone="muted">{item.type}</StatusBadge>
                  </Link>
                ))}
              </div>
            )}
          </Section>
        </div>
      </div>
    </PageContainer>
  );
}
