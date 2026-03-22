'use client';

import { useEffect, useState, useCallback } from 'react';
import Link from 'next/link';
import PageContainer from '@/components/layout/PageContainer';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import { useAppStore } from '@/stores/useAppStore';
import { monitorApi } from '@/lib/api/monitor';
import type { RealtimeQuote } from '@/types/api';

const coreActions = [
  {
    title: '股票分析',
    description: 'K线图、技术指标、筹码分布和交易信号',
    href: '/analyze',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
    color: 'primary',
  },
  {
    title: '策略回测',
    description: '验证交易策略的历史表现',
    href: '/backtest',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    ),
    color: 'success',
  },
  {
    title: '实时监控',
    description: 'WebSocket实时行情推送',
    href: '/monitor',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
      </svg>
    ),
    color: 'danger',
  },
  {
    title: 'ML模型',
    description: '训练GBM模型，管理生命周期',
    href: '/model',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
    color: 'warning',
  },
];

const smartActions = [
  {
    title: '智能选股',
    description: '基于技术指标和因子筛选',
    href: '/scan',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    ),
    color: 'info',
  },
  {
    title: '智能荐股',
    description: 'GBM模型预测未来收益',
    href: '/picks',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
      </svg>
    ),
    color: 'primary',
  },
  {
    title: '因子研究',
    description: '因子有效性检验与IC/IR分析',
    href: '/factors',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    color: 'success',
  },
  {
    title: '组合风控',
    description: '行业暴露与黑名单监控',
    href: '/risk',
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
      </svg>
    ),
    color: 'secondary',
  },
];

// 市场指数代码
const INDEX_SYMBOLS = ['000001', '399001', '399006', '000300'];
const INDEX_NAMES: Record<string, string> = {
  '000001': '上证指数',
  '399001': '深证成指',
  '399006': '创业板指',
  '000300': '沪深300',
};

export default function HomePage() {
  const setActivePage = useAppStore((state) => state.setActivePage);
  const [marketIndices, setMarketIndices] = useState<RealtimeQuote[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchMarketIndices = useCallback(async () => {
    try {
      const data = await monitorApi.getQuotes(INDEX_SYMBOLS);
      if (data) {
        // 过滤掉错误的记录，并补充名称
        const validData = data
          .filter(q => !q.error && q.price > 0)
          .map(q => ({
            ...q,
            name: q.name || INDEX_NAMES[q.symbol] || q.symbol,
          }));
        setMarketIndices(validData);
      }
    } catch (error) {
      console.error('Failed to fetch market indices:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setActivePage('overview');
    fetchMarketIndices();
    // 每30秒刷新一次
    const interval = setInterval(fetchMarketIndices, 30000);
    return () => clearInterval(interval);
  }, [setActivePage, fetchMarketIndices]);

  return (
    <PageContainer>
      <div className="space-y-6">
        {/* Page Header */}
        <div>
          <h1 className="text-2xl font-bold text-text-primary">盘面概览</h1>
          <p className="text-text-muted mt-1">快速查看市场动态和常用功能入口</p>
        </div>

        {/* Core Functions */}
        <div>
          <h2 className="text-lg font-semibold text-text-primary mb-3">核心功能</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {coreActions.map((action) => (
              <Link
                key={action.href}
                href={action.href}
                className="block group"
              >
                <Card className="h-full card-hover cursor-pointer">
                  <div className="flex items-start gap-4">
                    <div className={`p-3 rounded-lg bg-${action.color}/20 text-${action.color}`}>
                      {action.icon}
                    </div>
                    <div className="flex-1">
                      <h3 className="font-medium text-text-primary group-hover:text-primary transition-colors">
                        {action.title}
                      </h3>
                      <p className="text-sm text-text-muted mt-1">{action.description}</p>
                    </div>
                  </div>
                </Card>
              </Link>
            ))}
          </div>
        </div>

        {/* Smart Features */}
        <div>
          <h2 className="text-lg font-semibold text-text-primary mb-3">智能功能</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {smartActions.map((action) => (
              <Link
                key={action.href}
                href={action.href}
                className="block group"
              >
                <Card className="h-full card-hover cursor-pointer">
                  <div className="flex items-start gap-4">
                    <div className={`p-3 rounded-lg bg-${action.color}/20 text-${action.color}`}>
                      {action.icon}
                    </div>
                    <div className="flex-1">
                      <h3 className="font-medium text-text-primary group-hover:text-primary transition-colors">
                        {action.title}
                      </h3>
                      <p className="text-sm text-text-muted mt-1">{action.description}</p>
                    </div>
                  </div>
                </Card>
              </Link>
            ))}
          </div>
        </div>

        {/* Market Indices */}
        <Card title="市场指数" action={<Badge variant="success">实时</Badge>}>
          {loading ? (
            <div className="text-center py-8 text-text-muted">加载中...</div>
          ) : marketIndices.length === 0 ? (
            <div className="text-center py-8 text-text-muted">暂无数据</div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {marketIndices.map((index) => (
                <div key={index.symbol} className="p-4 bg-bg-tertiary rounded-lg">
                  <div className="text-sm text-text-muted">{index.name}</div>
                  <div className="text-xl font-bold text-text-primary mt-1">
                    {index.price.toLocaleString()}
                  </div>
                  <div className={`text-sm mt-1 ${index.change_pct >= 0 ? 'text-success' : 'text-danger'}`}>
                    {index.change_pct >= 0 ? '+' : ''}{index.change.toFixed(2)} ({(index.change_pct * 100).toFixed(2)}%)
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* Quick Access to Smart Features */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card
            title="智能选股"
            action={
              <Link href="/scan">
                <Button size="sm" variant="ghost">查看全部</Button>
              </Link>
            }
          >
            <p className="text-text-muted text-sm">
              基于技术指标和量化因子，全市场扫描符合条件的股票
            </p>
            <div className="mt-4 flex gap-2">
              <Badge variant="primary">MACD金叉</Badge>
              <Badge variant="success">放量突破</Badge>
              <Badge variant="warning">均线多头</Badge>
            </div>
          </Card>

          <Card
            title="智能荐股"
            action={
              <Link href="/picks">
                <Button size="sm" variant="ghost">查看全部</Button>
              </Link>
            }
          >
            <p className="text-text-muted text-sm">
              利用GBM机器学习模型，预测股票未来收益表现
            </p>
            <div className="mt-4 flex gap-2">
              <Badge variant="success">高胜率</Badge>
              <Badge variant="primary">沪深300</Badge>
              <Badge variant="warning">Top 10</Badge>
            </div>
          </Card>
        </div>

        {/* Getting Started */}
        <Card title="快速开始">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-center gap-3 p-4 bg-bg-tertiary rounded-lg">
              <div className="w-8 h-8 bg-primary/20 text-primary rounded-full flex items-center justify-center font-bold">
                1
              </div>
              <div>
                <div className="font-medium text-text-primary">分析股票</div>
                <div className="text-sm text-text-muted">输入股票代码查看详情</div>
              </div>
            </div>
            <div className="flex items-center gap-3 p-4 bg-bg-tertiary rounded-lg">
              <div className="w-8 h-8 bg-primary/20 text-primary rounded-full flex items-center justify-center font-bold">
                2
              </div>
              <div>
                <div className="font-medium text-text-primary">回测验证</div>
                <div className="text-sm text-text-muted">测试策略历史表现</div>
              </div>
            </div>
            <div className="flex items-center gap-3 p-4 bg-bg-tertiary rounded-lg">
              <div className="w-8 h-8 bg-primary/20 text-primary rounded-full flex items-center justify-center font-bold">
                3
              </div>
              <div>
                <div className="font-medium text-text-primary">实时监控</div>
                <div className="text-sm text-text-muted">跟踪关注股票动态</div>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </PageContainer>
  );
}
