'use client';

import { useState, useEffect } from 'react';
import PageContainer from '@/components/layout/PageContainer';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import StrategySelector from '@/components/backtest/StrategySelector';
import MetricsGrid from '@/components/backtest/MetricsGrid';
import EquityCurve from '@/components/charts/EquityCurve';
import TradeHistory from '@/components/backtest/TradeHistory';
import { useAppStore } from '@/stores/useAppStore';
import { backtestApi } from '@/lib/api/backtest';
import { useApi } from '@/hooks/useApi';
import Loading from '@/components/ui/Loading';
import type { Strategy, BacktestResult } from '@/types/backtest';

export default function BacktestPage() {
  const setActivePage = useAppStore((state) => state.setActivePage);

  const [symbol, setSymbol] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [initialCapital, setInitialCapital] = useState(1000000);
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [results, setResults] = useState<BacktestResult[]>([]);
  const [activeResultIndex, setActiveResultIndex] = useState(0);

  const { loading: loadingStrategies, execute: fetchStrategies } = useApi(backtestApi.getStrategies);
  const { loading: runningBacktest, execute: runBacktest } = useApi(backtestApi.run);
  const { loading: runningAllBacktest, execute: runAllBacktest } = useApi(backtestApi.runAll);

  useEffect(() => {
    setActivePage('backtest');
    fetchStrategies().then((data) => {
      if (data) {
        // 转换策略格式
        const formatted = data.map((s: any) => ({
          id: s.name,
          name: s.display_name || s.name,
          display_name: s.display_name,
          description: s.description,
          category: s.category,
        }));
        setStrategies(formatted);
      }
    });
  }, [setActivePage, fetchStrategies]);

  const handleRunSingle = async () => {
    if (!symbol || !selectedStrategy || !startDate || !endDate) {
      return;
    }

    const result = await runBacktest({
      symbol,
      strategy: selectedStrategy,
      start_date: startDate,
      end_date: endDate,
      initial_capital: initialCapital,
    });
    if (result) {
      setResults([result]);
      setActiveResultIndex(0);
    }
  };

  const handleRunAll = async () => {
    if (!symbol || !startDate || !endDate) {
      return;
    }

    const data = await runAllBacktest({
      symbol,
      start_date: startDate,
      end_date: endDate,
      initial_capital: initialCapital,
    });
    if (data) {
      setResults(data);
      setActiveResultIndex(0);
    }
  };

  const currentResult = results[activeResultIndex];

  // 从结果中提取指标
  const getMetrics = (result: BacktestResult) => ({
    total_return: result.total_return,
    annual_return: result.annual_return,
    max_drawdown: result.max_drawdown,
    sharpe_ratio: result.sharpe_ratio,
    win_rate: result.win_rate,
    profit_factor: result.profit_factor || 0,
    total_trades: result.total_trades,
    benchmark_return: result.benchmark_return,
    excess_return: result.excess_return,
  });

  return (
    <PageContainer>
      <div className="space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold text-text-primary">策略回测</h1>
          <p className="text-text-muted mt-1">验证交易策略的历史表现，对比多种策略收益</p>
        </div>

        {/* Parameters */}
        <Card title="回测参数">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Input
              label="股票代码"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              placeholder="如：600519"
            />
            <Input
              label="开始日期"
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
            <Input
              label="结束日期"
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
            <Input
              label="初始资金"
              type="number"
              value={initialCapital}
              onChange={(e) => setInitialCapital(Number(e.target.value))}
            />
          </div>

          <div className="mt-4">
            <label className="block text-sm font-medium text-text-secondary mb-2">
              选择策略
            </label>
            {loadingStrategies ? (
              <Loading size="sm" />
            ) : (
              <StrategySelector
                strategies={strategies}
                selected={selectedStrategy}
                onSelect={setSelectedStrategy}
              />
            )}
          </div>

          <div className="flex gap-3 mt-6">
            <Button onClick={handleRunSingle} loading={runningBacktest} disabled={!selectedStrategy}>
              运行选中策略
            </Button>
            <Button variant="secondary" onClick={handleRunAll} loading={runningAllBacktest}>
              运行所有策略对比
            </Button>
          </div>
        </Card>

        {/* Results */}
        {results.length > 0 && (
          <>
            {/* Strategy Tabs */}
            {results.length > 1 && (
              <div className="flex gap-2 overflow-x-auto pb-2">
                {results.map((result, index) => (
                  <Button
                    key={result.strategy}
                    variant={activeResultIndex === index ? 'primary' : 'secondary'}
                    size="sm"
                    onClick={() => setActiveResultIndex(index)}
                  >
                    {result.strategy}
                  </Button>
                ))}
              </div>
            )}

            {currentResult && (
              <>
                {/* Metrics */}
                <MetricsGrid
                  metrics={getMetrics(currentResult)}
                  excessReturn={currentResult.excess_return}
                />

                {/* Equity Curve */}
                {currentResult.equity_curve && currentResult.equity_curve.length > 0 && (
                  <Card title="收益曲线" noPadding>
                    <div className="p-4">
                      <EquityCurve
                        data={currentResult.equity_curve}
                        benchmarkData={currentResult.benchmark_curve}
                        height={350}
                      />
                    </div>
                  </Card>
                )}

                {/* Trade History */}
                {currentResult.trades && currentResult.trades.length > 0 && (
                  <Card title="交易记录">
                    <TradeHistory trades={currentResult.trades} />
                  </Card>
                )}
              </>
            )}
          </>
        )}
      </div>
    </PageContainer>
  );
}
