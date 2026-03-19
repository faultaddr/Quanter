'use client';

import { useState, useEffect } from 'react';
import PageContainer from '@/components/layout/PageContainer';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import StockSearch from '@/components/stock/StockSearch';
import SignalPanel from '@/components/stock/SignalPanel';
import KlineChart from '@/components/charts/KlineChart';
import MACDChart from '@/components/charts/MACDChart';
import KDJChart from '@/components/charts/KDJChart';
import RSIChart from '@/components/charts/RSIChart';
import ChipChart from '@/components/charts/ChipChart';
import { useAppStore } from '@/stores/useAppStore';
import { useStockStore } from '@/stores/useStockStore';
import { stockApi } from '@/lib/api/stock';
import { useApi } from '@/hooks/useApi';
import Loading from '@/components/ui/Loading';
import type { StockAnalysis, KlineData } from '@/types/stock';

export default function AnalyzePage() {
  const setActivePage = useAppStore((state) => state.setActivePage);
  const addHistory = useAppStore((state) => state.addHistory);
  const setStock = useStockStore((state) => state.setStock);

  const [symbol, setSymbol] = useState('');
  const [stockName, setStockName] = useState('');
  const [analysis, setAnalysis] = useState<StockAnalysis | null>(null);
  const [activeTab, setActiveTab] = useState<'kline' | 'chip' | 'signals'>('kline');
  const [days, setDays] = useState(120);

  const { loading, execute: fetchAnalysis } = useApi(stockApi.getAnalysis);

  useEffect(() => {
    setActivePage('analyze');
  }, [setActivePage]);

  const handleStockSelect = async (selectedSymbol: string, name: string) => {
    setSymbol(selectedSymbol);
    setStockName(name);
    setStock(selectedSymbol, name);
    addHistory({ type: 'stock', title: `${name} (${selectedSymbol})`, path: `/analyze?symbol=${selectedSymbol}` });

    const result = await fetchAnalysis(selectedSymbol, days);
    if (result) {
      setAnalysis(result);
    }
  };

  const dates = analysis?.kline.map((k) => k.date) || [];

  return (
    <PageContainer>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-text-primary">股票分析</h1>
            <p className="text-text-muted mt-1">查看K线图、技术指标、筹码分布和交易信号</p>
          </div>
          <StockSearch onSelect={handleStockSelect} className="w-72" />
        </div>

        {loading ? (
          <div className="flex justify-center py-20">
            <Loading size="lg" text="加载中..." />
          </div>
        ) : !symbol ? (
          <Card className="text-center py-20">
            <svg className="w-16 h-16 mx-auto text-text-muted mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <h3 className="text-lg font-medium text-text-primary mb-2">选择股票开始分析</h3>
            <p className="text-text-muted">在上方搜索框输入股票代码或名称</p>
          </Card>
        ) : (
          <>
            {/* Stock Info */}
            <div className="flex items-center gap-4">
              <h2 className="text-xl font-semibold text-text-primary">
                {stockName} <span className="text-text-muted">({symbol})</span>
              </h2>
              <div className="flex gap-2">
                {[60, 120, 250].map((d) => (
                  <Button
                    key={d}
                    size="sm"
                    variant={days === d ? 'primary' : 'secondary'}
                    onClick={() => {
                      setDays(d);
                      if (symbol) fetchAnalysis(symbol, d);
                    }}
                  >
                    {d}天
                  </Button>
                ))}
              </div>
            </div>

            {/* Tabs */}
            <div className="flex gap-2 border-b border-border-primary pb-2">
              {[
                { key: 'kline', label: 'K线图' },
                { key: 'chip', label: '筹码分布' },
                { key: 'signals', label: '交易信号' },
              ].map((tab) => (
                <Button
                  key={tab.key}
                  variant={activeTab === tab.key ? 'primary' : 'ghost'}
                  onClick={() => setActiveTab(tab.key as typeof activeTab)}
                >
                  {tab.label}
                </Button>
              ))}
            </div>

            {/* Content */}
            {activeTab === 'kline' && analysis && (
              <div className="space-y-4">
                <Card title="K线图" noPadding>
                  <div className="p-4">
                    <KlineChart data={analysis.kline} height={400} />
                  </div>
                </Card>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                  <Card title="MACD" noPadding>
                    <div className="p-4">
                      {analysis.indicators.macd ? (
                        <MACDChart data={analysis.indicators.macd} dates={dates} />
                      ) : (
                        <div className="text-center py-8 text-text-muted">暂无MACD数据</div>
                      )}
                    </div>
                  </Card>

                  <Card title="KDJ" noPadding>
                    <div className="p-4">
                      {analysis.indicators.kdj ? (
                        <KDJChart data={analysis.indicators.kdj} dates={dates} />
                      ) : (
                        <div className="text-center py-8 text-text-muted">暂无KDJ数据</div>
                      )}
                    </div>
                  </Card>

                  <Card title="RSI" noPadding>
                    <div className="p-4">
                      {analysis.indicators.rsi ? (
                        <RSIChart data={analysis.indicators.rsi} dates={dates} />
                      ) : (
                        <div className="text-center py-8 text-text-muted">暂无RSI数据</div>
                      )}
                    </div>
                  </Card>
                </div>
              </div>
            )}

            {activeTab === 'chip' && analysis && (
              <Card title="筹码分布" noPadding>
                <div className="p-4">
                  <ChipChart
                    data={analysis.chip}
                    currentPrice={analysis.kline[analysis.kline.length - 1]?.close}
                    height={400}
                  />
                </div>
              </Card>
            )}

            {activeTab === 'signals' && analysis && (
              <Card title="交易信号">
                <SignalPanel signals={analysis.signals} />
              </Card>
            )}
          </>
        )}
      </div>
    </PageContainer>
  );
}
