'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import PageContainer from '@/components/layout/PageContainer';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import { MetricTile, PageHeader, Section, SegmentedControl, StatusBadge } from '@/components/ui';
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

type TabType = 'overview' | 'kline' | 'indicators' | 'chip' | 'flow' | 'risk' | 'backtest' | 'signals' | 'factors';

// 因子评分数据
interface FactorScores {
  momentum: number;
  value: number;
  quality: number;
  growth: number;
  overall: number;
}

// 交易可行性数据
interface TradingFeasibility {
  can_buy: boolean;
  can_sell: boolean;
  limit_status: 'normal' | 'limit_up' | 'limit_down';
  is_st: boolean;
  is_suspended: boolean;
  slippage_rate: number;
  commission_rate: number;
  reason: string;
}

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

  // 因子评分和交易可行性
  const [factorScores, setFactorScores] = useState<FactorScores | null>(null);
  const [tradingFeasibility, setTradingFeasibility] = useState<TradingFeasibility | null>(null);

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

  const fetchFactorScores = useCallback(async (sym: string) => {
    try {
      const response = await fetch(`/api/stock/${sym}/factors`);
      if (response.ok) {
        const data = await response.json();
        setFactorScores(data);
      }
    } catch (error) {
      console.error('Failed to fetch factor scores:', error);
    }
  }, []);

  const checkTradingFeasibility = useCallback(async (sym: string) => {
    try {
      const response = await fetch(`/api/stock/${sym}/feasibility`);
      if (response.ok) {
        const data = await response.json();
        setTradingFeasibility(data);
      }
    } catch (error) {
      console.error('Failed to check trading feasibility:', error);
    }
  }, []);

  const handleStockSelect = useCallback(async (selectedSymbol: string, name: string) => {
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
    setFactorScores(null);
    setTradingFeasibility(null);

    // 获取因子评分和交易可行性
    fetchFactorScores(selectedSymbol);
    checkTradingFeasibility(selectedSymbol);
  }, [addHistory, checkTradingFeasibility, days, fetchAnalysis, fetchFactorScores, setStock]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const requestedSymbol = params.get('symbol') || '';
    const requestedName = params.get('name') || requestedSymbol;
    if (!requestedSymbol || requestedSymbol === symbol) return;
    void handleStockSelect(requestedSymbol, requestedName);
  }, [handleStockSelect, symbol]);

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
    // 不乘以100，让 formatPercent 函数处理
    const periodReturn = (latest.close - first.close) / first.close;

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
      rangeFromHigh: (latest.close - periodHigh) / periodHigh,
      rangeFromLow: (latest.close - periodLow) / periodLow,
    };
  }, [analysis?.kline]);

  const tabs: { value: TabType; label: string }[] = [
    { value: 'overview', label: '概览' },
    { value: 'kline', label: 'K线图' },
    { value: 'indicators', label: '技术指标' },
    { value: 'chip', label: '筹码分布' },
    { value: 'flow', label: '资金流向' },
    { value: 'risk', label: '风险评估' },
    { value: 'backtest', label: '回测对比' },
  ];

  return (
    <PageContainer>
      <div className="space-y-6">
        <PageHeader
          eyebrow="Stock Workbench"
          title="股票分析"
          description="从候选股进入单股工作台，集中查看走势、信号、筹码资金和风险。"
          meta={
            <>
              <StatusBadge tone={symbol ? 'success' : 'muted'}>
                {symbol ? `${stockName || symbol} 已加载` : '等待选择股票'}
              </StatusBadge>
              <StatusBadge tone="primary">{days}天窗口</StatusBadge>
            </>
          }
          actions={
            <>
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
            </>
          }
        />

        {loading ? (
          <div className="flex justify-center py-20">
            <Loading size="lg" text="加载中..." />
          </div>
        ) : !symbol ? (
          <Section framed>
            <div className="mx-auto max-w-3xl py-12 text-center">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-lg border border-border-primary bg-bg-tertiary text-text-secondary">
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 17v-6m4 6V7m4 10v-3M5 21h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-text-primary">选择股票开始分析</h3>
              <p className="mt-2 text-sm text-text-muted">
                在顶部搜索框输入代码或名称，也可以从智能选股结果直接进入。
              </p>
              <div className="mt-5 flex flex-wrap justify-center gap-2">
                {[
                  ['000001', '平安银行'],
                  ['600519', '贵州茅台'],
                  ['000300', '沪深300'],
                ].map(([code, name]) => (
                  <Button
                    key={code}
                    size="sm"
                    variant="secondary"
                    onClick={() => handleStockSelect(code, name)}
                  >
                    {name}
                  </Button>
                ))}
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => window.location.href = '/scan'}
                >
                  去智能选股
                </Button>
              </div>
            </div>
          </Section>
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

            <SegmentedControl
              options={tabs}
              value={activeTab}
              onChange={setActiveTab}
              compact
            />

            {/* Content */}
            {activeTab === 'overview' && analysis && (
              <div className="space-y-6">
                {/* Stats Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                  <MetricTile
                    label="区间涨跌"
                    value={formatPercent(stats?.periodReturn || 0)}
                    tone={(stats?.periodReturn || 0) >= 0 ? 'positive' : 'negative'}
                  />
                  <MetricTile label="区间最高" value={formatNumber(stats?.periodHigh || 0)} />
                  <MetricTile label="区间最低" value={formatNumber(stats?.periodLow || 0)} />
                  <MetricTile label="平均成交量" value={formatAmount(stats?.avgVolume || 0)} />
                  <MetricTile label="距高点" value={formatPercent(stats?.rangeFromHigh || 0)} tone="negative" />
                  <MetricTile label="距低点" value={formatPercent(stats?.rangeFromLow || 0)} tone="positive" />

                  {/* 因子评分 */}
                  <Card>
                    <div className="text-text-muted text-sm mb-2">因子评分</div>
                    {factorScores ? (
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>动量</span>
                          <span className={factorScores.momentum >= 60 ? 'text-success' : 'text-warning'}>
                            {factorScores.momentum}
                          </span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>价值</span>
                          <span className={factorScores.value >= 60 ? 'text-success' : 'text-warning'}>
                            {factorScores.value}
                          </span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>质量</span>
                          <span className={factorScores.quality >= 60 ? 'text-success' : 'text-warning'}>
                            {factorScores.quality}
                          </span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span>成长</span>
                          <span className={factorScores.growth >= 60 ? 'text-success' : 'text-warning'}>
                            {factorScores.growth}
                          </span>
                        </div>
                        <div className="border-t pt-2 mt-2">
                          <div className="flex justify-between font-bold">
                            <span>综合</span>
                            <span className={factorScores.overall >= 60 ? 'text-success' : 'text-warning'}>
                              {factorScores.overall}
                            </span>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="text-text-muted text-sm">加载中...</div>
                    )}
                  </Card>

                  {/* 交易可行性 */}
                  <Card>
                    <div className="text-text-muted text-sm mb-2">交易可行性</div>
                    {tradingFeasibility ? (
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <span className={tradingFeasibility.can_buy ? 'text-success' : 'text-danger'}>
                            {tradingFeasibility.can_buy ? '✓' : '✗'}
                          </span>
                          <span className="text-sm">可买入</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className={tradingFeasibility.can_sell ? 'text-success' : 'text-danger'}>
                            {tradingFeasibility.can_sell ? '✓' : '✗'}
                          </span>
                          <span className="text-sm">可卖出</span>
                        </div>
                        {tradingFeasibility.limit_status !== 'normal' && (
                          <Badge variant={tradingFeasibility.limit_status === 'limit_up' ? 'success' : 'danger'}>
                            {tradingFeasibility.limit_status === 'limit_up' ? '涨停' : '跌停'}
                          </Badge>
                        )}
                        {tradingFeasibility.is_st && (
                          <Badge variant="danger">ST股</Badge>
                        )}
                        {tradingFeasibility.reason && (
                          <div className="text-xs text-text-muted mt-1">{tradingFeasibility.reason}</div>
                        )}
                      </div>
                    ) : (
                      <div className="text-text-muted text-sm">加载中...</div>
                    )}
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

                {/* 前往因子研究 */}
                <div className="flex justify-center">
                  <Button variant="secondary" onClick={() => window.location.href = '/factors'}>
                    详细因子分析
                  </Button>
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

          </>
        )}
      </div>
    </PageContainer>
  );
}
