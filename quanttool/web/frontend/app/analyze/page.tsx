'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
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
import FlowChart from '@/components/charts/FlowChart';
import RiskAssessment from '@/components/stock/RiskAssessment';
import BacktestCompare from '@/components/stock/BacktestCompare';
import QuoteCard from '@/components/stock/QuoteCard';
import { useAppStore } from '@/stores/useAppStore';
import { useStockStore } from '@/stores/useStockStore';
import { stockApi, FlowData, RiskMetrics, BacktestResult } from '@/lib/api/stock';
import { monitorApi } from '@/lib/api/monitor';
import { useApi } from '@/hooks/useApi';
import Loading from '@/components/ui/Loading';
import { formatNumber, formatAmount, formatPercent, getChangeColorClass } from '@/lib/utils';
import type { StockAnalysis, KlineData } from '@/types/stock';
import type { RealtimeQuote } from '@/types/api';

type TabType = 'overview' | 'kline' | 'indicators' | 'chip' | 'flow' | 'risk' | 'backtest' | 'signals';

export default function AnalyzePage() {
  const setActivePage = useAppStore((state) => state.setActivePage);
  const addHistory = useAppStore((state) => state.addHistory);
  const setStock = useStockStore((state) => state.setStock);

  const [symbol, setSymbol] = useState('');
  const [stockName, setStockName] = useState('');
  const [analysis, setAnalysis] = useState<StockAnalysis | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [days, setDays] = useState(120);
  const [realtimeQuote, setRealtimeQuote] = useState<RealtimeQuote | null>(null);

  // 新增数据状态
  const [flowData, setFlowData] = useState<FlowData[]>([]);
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics | null>(null);
  const [backtestResults, setBacktestResults] = useState<BacktestResult[]>([]);
  const [benchmark, setBenchmark] = useState<{ name: string; total_return: number; equity_curve: { date: string; value: number }[] } | null>(null);

  const { loading, execute: fetchAnalysis } = useApi(stockApi.getAnalysis);
  const { loading: loadingFlow, execute: fetchFlow } = useApi(stockApi.getFlow);
  const { loading: loadingRisk, execute: fetchRisk } = useApi(stockApi.getRisk);
  const { loading: loadingBacktest, execute: fetchBacktest } = useApi(stockApi.getBacktestCompare);

  // 获取实时行情
  const fetchRealtimeQuote = useCallback(async (sym: string) => {
    try {
      const quote = await monitorApi.getQuote(sym);
      if (quote) {
        setRealtimeQuote(quote);
      }
    } catch (error) {
      console.error('Failed to fetch realtime quote:', error);
    }
  }, []);

  useEffect(() => {
    setActivePage('analyze');
  }, [setActivePage]);

  // 自动刷新实时行情
  useEffect(() => {
    if (!symbol) return;

    fetchRealtimeQuote(symbol);
    const interval = setInterval(() => {
      fetchRealtimeQuote(symbol);
    }, 10000);

    return () => clearInterval(interval);
  }, [symbol, fetchRealtimeQuote]);

  // 根据激活的 tab 加载数据
  useEffect(() => {
    if (!symbol) return;

    if (activeTab === 'flow' && flowData.length === 0) {
      fetchFlow(symbol, 30).then((data) => {
        if (data?.data) setFlowData(data.data);
      });
    }

    if (activeTab === 'risk' && !riskMetrics) {
      fetchRisk(symbol, 250).then((data) => {
        if (data?.metrics) setRiskMetrics(data.metrics);
      });
    }

    if (activeTab === 'backtest' && backtestResults.length === 0) {
      fetchBacktest(symbol, 250).then((data) => {
        if (data?.results) setBacktestResults(data.results);
        if (data?.benchmark) setBenchmark(data.benchmark);
      });
    }
  }, [symbol, activeTab, flowData.length, riskMetrics, backtestResults.length, fetchFlow, fetchRisk, fetchBacktest]);

  const handleStockSelect = async (selectedSymbol: string, name: string) => {
    setSymbol(selectedSymbol);
    setStockName(name);
    setStock(selectedSymbol, name);
    addHistory({
      type: 'stock',
      title: `${name} (${selectedSymbol})`,
      path: `/analyze?symbol=${selectedSymbol}`
    });

    const result = await fetchAnalysis(selectedSymbol, days);
    if (result) {
      setAnalysis(result);
    }

    // 重置其他数据
    setFlowData([]);
    setRiskMetrics(null);
    setBacktestResults([]);
    setBenchmark(null);
  };

  const handleDaysChange = async (newDays: number) => {
    setDays(newDays);
    if (symbol) {
      const result = await fetchAnalysis(symbol, newDays);
      if (result) {
        setAnalysis(result);
      }
    }
  };

  const dates = analysis?.kline.map((k) => k.date) || [];

  // 计算统计信息
  const stats = useMemo(() => {
    if (!analysis?.kline || analysis.kline.length === 0) return null;

    const kline = analysis.kline;
    const latest = kline[kline.length - 1];
    const first = kline[0];
    const periodReturn = ((latest.close - first.close) / first.close) * 100;

    const highs = kline.map(k => k.high);
    const lows = kline.map(k => k.low);
    const periodHigh = Math.max(...highs);
    const periodLow = Math.min(...lows);

    const volumes = kline.map(k => k.volume);
    const avgVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length;

    return {
      latestClose: latest.close,
      periodReturn,
      periodHigh,
      periodLow,
      avgVolume,
      rangeFromHigh: ((latest.close - periodHigh) / periodHigh) * 100,
      rangeFromLow: ((latest.close - periodLow) / periodLow) * 100,
    };
  }, [analysis?.kline]);

  const tabs: { key: TabType; label: string; icon: string }[] = [
    { key: 'overview', label: '概览', icon: '📊' },
    { key: 'kline', label: 'K线图', icon: '📈' },
    { key: 'indicators', label: '技术指标', icon: '📉' },
    { key: 'chip', label: '筹码分布', icon: '🎯' },
    { key: 'flow', label: '资金流向', icon: '💰' },
    { key: 'risk', label: '风险评估', icon: '⚠️' },
    { key: 'backtest', label: '回测对比', icon: '🔄' },
    { key: 'signals', label: '交易信号', icon: '📡' },
  ];

  return (
    <PageContainer>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-text-primary">股票分析</h1>
            <p className="text-text-muted mt-1">全方位股票分析：K线、技术指标、筹码分布、资金流向、风险评估</p>
          </div>
          <div className="flex items-center gap-4">
            <StockSearch onSelect={handleStockSelect} className="w-72" />
            <div className="flex gap-2">
              {[60, 120, 250].map((d) => (
                <Button
                  key={d}
                  size="sm"
                  variant={days === d ? 'primary' : 'secondary'}
                  onClick={() => handleDaysChange(d)}
                >
                  {d}天
                </Button>
              ))}
            </div>
          </div>
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
            {/* Stock Header */}
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-6">
                <div>
                  <h2 className="text-2xl font-bold text-text-primary">
                    {stockName} <span className="text-text-muted text-lg">({symbol})</span>
                  </h2>
                  <div className="flex items-center gap-4 mt-2">
                    {realtimeQuote && (
                      <>
                        <span className={`text-3xl font-bold ${getChangeColorClass(realtimeQuote.change_pct)}`}>
                          {formatNumber(realtimeQuote.price)}
                        </span>
                        <span className={`text-lg ${getChangeColorClass(realtimeQuote.change_pct)}`}>
                          {formatNumber(realtimeQuote.change)} ({(realtimeQuote.change_pct * 100).toFixed(2)}%)
                        </span>
                      </>
                    )}
                  </div>
                </div>
              </div>
              {realtimeQuote && (
                <div className="grid grid-cols-4 gap-4">
                  <div className="text-right">
                    <div className="text-text-muted text-sm">今开</div>
                    <div className="text-text-primary font-medium">{formatNumber(realtimeQuote.open)}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-text-muted text-sm">最高</div>
                    <div className="text-success font-medium">{formatNumber(realtimeQuote.high)}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-text-muted text-sm">最低</div>
                    <div className="text-danger font-medium">{formatNumber(realtimeQuote.low)}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-text-muted text-sm">成交量</div>
                    <div className="text-text-primary font-medium">{formatAmount(realtimeQuote.volume)}</div>
                  </div>
                </div>
              )}
            </div>

            {/* Tabs */}
            <div className="flex flex-wrap gap-2 border-b border-border-primary pb-2">
              {tabs.map((tab) => (
                <Button
                  key={tab.key}
                  size="sm"
                  variant={activeTab === tab.key ? 'primary' : 'ghost'}
                  onClick={() => setActiveTab(tab.key)}
                  className="flex items-center gap-1"
                >
                  <span>{tab.icon}</span>
                  {tab.label}
                </Button>
              ))}
            </div>

            {/* Content */}
            {activeTab === 'overview' && analysis && (
              <div className="space-y-6">
                {/* Stats Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                  <Card>
                    <div className="text-text-muted text-sm">区间涨跌</div>
                    <div className={`text-xl font-bold ${getChangeColorClass(stats?.periodReturn || 0)}`}>
                      {formatPercent(stats?.periodReturn || 0)}
                    </div>
                  </Card>
                  <Card>
                    <div className="text-text-muted text-sm">区间最高</div>
                    <div className="text-xl font-bold text-text-primary">{formatNumber(stats?.periodHigh || 0)}</div>
                  </Card>
                  <Card>
                    <div className="text-text-muted text-sm">区间最低</div>
                    <div className="text-xl font-bold text-text-primary">{formatNumber(stats?.periodLow || 0)}</div>
                  </Card>
                  <Card>
                    <div className="text-text-muted text-sm">平均成交量</div>
                    <div className="text-xl font-bold text-text-primary">{formatAmount(stats?.avgVolume || 0)}</div>
                  </Card>
                  <Card>
                    <div className="text-text-muted text-sm">距高点</div>
                    <div className="text-xl font-bold text-danger">{formatPercent(stats?.rangeFromHigh || 0)}</div>
                  </Card>
                  <Card>
                    <div className="text-text-muted text-sm">距低点</div>
                    <div className="text-xl font-bold text-success">{formatPercent(stats?.rangeFromLow || 0)}</div>
                  </Card>
                </div>

                {/* K线图 + 信号 */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div className="lg:col-span-2">
                    <Card title="K线图" noPadding>
                      <div className="p-4">
                        <KlineChart data={analysis.kline} height={400} />
                      </div>
                    </Card>
                  </div>
                  <div>
                    <Card title="交易信号">
                      <SignalPanel signals={analysis.signals} />
                    </Card>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'kline' && analysis && (
              <Card title="K线图" noPadding>
                <div className="p-4">
                  <KlineChart data={analysis.kline} height={500} />
                </div>
              </Card>
            )}

            {activeTab === 'indicators' && analysis && (
              <div className="space-y-6">
                <Card title="MACD" noPadding>
                  <div className="p-4">
                    {analysis.indicators.macd ? (
                      <MACDChart data={analysis.indicators.macd} dates={dates} />
                    ) : (
                      <div className="text-center py-8 text-text-muted">暂无MACD数据</div>
                    )}
                  </div>
                </Card>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
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

            {activeTab === 'flow' && (
              <Card title="资金流向" noPadding>
                <div className="p-4">
                  {loadingFlow ? (
                    <div className="flex justify-center py-10">
                      <Loading text="加载资金流向数据..." />
                    </div>
                  ) : flowData.length > 0 ? (
                    <FlowChart data={flowData} height={400} />
                  ) : (
                    <div className="text-center py-10 text-text-muted">暂无资金流向数据</div>
                  )}
                </div>
              </Card>
            )}

            {activeTab === 'risk' && (
              <div className="p-4">
                {loadingRisk ? (
                  <div className="flex justify-center py-10">
                    <Loading text="计算风险评估..." />
                  </div>
                ) : riskMetrics ? (
                  <RiskAssessment metrics={riskMetrics} period="近250日" />
                ) : (
                  <Card className="text-center py-10">
                    <p className="text-text-muted">暂无风险评估数据</p>
                  </Card>
                )}
              </div>
            )}

            {activeTab === 'backtest' && (
              <div className="p-4">
                {loadingBacktest ? (
                  <div className="flex justify-center py-10">
                    <Loading text="运行策略回测..." />
                  </div>
                ) : backtestResults.length > 0 ? (
                  <BacktestCompare results={backtestResults} benchmark={benchmark || undefined} />
                ) : (
                  <Card className="text-center py-10">
                    <p className="text-text-muted">暂无回测数据</p>
                  </Card>
                )}
              </div>
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
