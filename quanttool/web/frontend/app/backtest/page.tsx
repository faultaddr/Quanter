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
import { formatPercent, formatNumber } from '@/lib/utils';
import { useApi } from '@/hooks/useApi';
import Loading from '@/components/ui/Loading';
import type { Strategy, BacktestResult } from '@/types/backtest';

export default function BacktestPage() {
  const setActivePage = useAppStore((state) => state.setActivePage);

  const [symbol, setSymbol] = useState('300750');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [initialCapital, setInitialCapital] = useState(1000000);
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [results, setResults] = useState<BacktestResult[]>([]);
  const [activeResultIndex, setActiveResultIndex] = useState(0);

  // A股约束选项
  const [enableConstraints, setEnableConstraints] = useState(true);
  const [excludeST, setExcludeST] = useState(true);
  const [excludeLimit, setExcludeLimit] = useState(true);
  const [commissionRate, setCommissionRate] = useState(0.0003);

  const { loading: loadingStrategies, execute: fetchStrategies } = useApi(backtestApi.getStrategies);
  const { loading: runningBacktest, execute: runBacktest } = useApi(backtestApi.run);

  useEffect(() => {
    setActivePage('backtest');

    // 设置默认日期：最近一年
    const now = new Date();
    const oneYearAgo = new Date(now);
    oneYearAgo.setFullYear(now.getFullYear() - 1);

    const formatDate = (d: Date) => d.toISOString().split('T')[0];
    setEndDate(formatDate(now));
    setStartDate(formatDate(oneYearAgo));

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
      // A股约束
      enable_constraints: enableConstraints,
      exclude_st: excludeST,
      exclude_limit: excludeLimit,
      commission_rate: commissionRate,
    });
    if (result) {
      setResults([result]);
      setActiveResultIndex(0);
    }
  };

  const [streaming, setStreaming] = useState(false);
  const [progress, setProgress] = useState({ completed: 0, total: 0 });

  const handleRunAll = () => {
    if (!symbol || !startDate || !endDate) {
      alert('请输入股票代码和日期范围');
      return;
    }

    setResults([]);
    setActiveResultIndex(0);
    setStreaming(true);
    setProgress({ completed: 0, total: 9 });

    // 使用流式 API
    backtestApi.runAllStream(
      {
        symbol,
        start_date: startDate,
        end_date: endDate,
        initial_capital: initialCapital,
        enable_constraints: enableConstraints,
        exclude_st: excludeST,
        exclude_limit: excludeLimit,
        commission_rate: commissionRate,
      },
      // 策略完成回调
      (result) => {
        console.log('[Backtest] Strategy complete:', result.strategy);
        setResults(prev => {
          // 检查是否已存在
          const exists = prev.find(r => r.strategy === result.strategy);
          if (exists) return prev;
          // 添加新结果并按年化收益排序
          const newResults = [...prev, result].sort((a, b) =>
            (b.annual_return || 0) - (a.annual_return || 0)
          );
          console.log('[Backtest] Updated results:', newResults.map(r => r.strategy));
          return newResults;
        });
        // 强制立即更新，禁用 React 批量更新
        setProgress(p => {
          const newProgress = { ...p, completed: p.completed + 1 };
          return newProgress;
        });
      },
      // 完成回调
      (finalResults) => {
        console.log('All done:', finalResults);
        setResults(finalResults);
        setStreaming(false);
      },
      // 错误回调
      (error) => {
        console.error('Stream error:', error);
        alert('回测失败: ' + error);
        setStreaming(false);
      }
    );
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

          {/* A股约束选项 */}
          <div className="mt-4 pt-4 border-t border-gray-200">
            <div className="flex items-center gap-2 mb-3">
              <input
                type="checkbox"
                id="enableConstraints"
                checked={enableConstraints}
                onChange={(e) => setEnableConstraints(e.target.checked)}
                className="rounded"
              />
              <label htmlFor="enableConstraints" className="text-sm font-medium text-text-secondary">
                启用A股交易约束
              </label>
              <span className="text-xs text-text-muted">(涨跌停、ST股限制、真实交易成本)</span>
            </div>

            {enableConstraints && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 ml-6">
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
                    checked={excludeLimit}
                    onChange={(e) => setExcludeLimit(e.target.checked)}
                    className="rounded"
                  />
                  <span>涨跌停不能交易</span>
                </label>
                <Input
                  label="佣金费率"
                  type="number"
                  step="0.0001"
                  value={commissionRate}
                  onChange={(e) => setCommissionRate(Number(e.target.value))}
                />
              </div>
            )}
          </div>

          <div className="flex gap-3 mt-6">
            <Button onClick={handleRunSingle} loading={runningBacktest} disabled={!selectedStrategy}>
              运行选中策略
            </Button>
            <Button variant="secondary" onClick={handleRunAll} disabled={streaming}>
              {streaming ? `运行中 ${progress.completed}/${progress.total}...` : '运行所有策略对比'}
            </Button>
          </div>
        </Card>

        {/* Results - Strategy Cards Grid */}
        {results.length > 0 && (
          <>
            {/* 策略卡片网格 */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {results.map((result, index) => (
                <div
                  key={result.strategy}
                  onClick={() => setActiveResultIndex(index)}
                  className={`
                    relative p-5 rounded-xl cursor-pointer transition-all duration-200
                    ${activeResultIndex === index
                      ? 'ring-2 ring-primary bg-primary/5 shadow-lg transform scale-[1.02]'
                      : 'bg-bg-secondary hover:bg-bg-tertiary hover:shadow-md border border-border-primary'}
                  `}
                >
                  {/* 排名徽章 */}
                  {index < 3 && (
                    <div className={`absolute -top-2 -right-2 w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold shadow-lg ${
                      index === 0 ? 'bg-yellow-500' : index === 1 ? 'bg-gray-400' : 'bg-amber-700'
                    }`}>
                      {index + 1}
                    </div>
                  )}

                  {/* 策略名称 */}
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-bold text-lg text-text-primary">
                      {result.strategy_display || result.strategy}
                    </h3>
                    {result.error && (
                      <span className="text-xs text-red-500">失败</span>
                    )}
                  </div>

                  {/* 收益率 */}
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-text-muted">总收益</span>
                      <span className={`text-xl font-bold ${
                        result.total_return > 0 ? 'text-green-600' : result.total_return < 0 ? 'text-red-600' : 'text-gray-600'
                      }`}>
                        {formatPercent(result.total_return || 0)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-text-muted">年化收益</span>
                      <span className={`text-lg font-semibold ${
                        result.annual_return > 0 ? 'text-green-600' : result.annual_return < 0 ? 'text-red-600' : 'text-gray-600'
                      }`}>
                        {formatPercent(result.annual_return || 0)}
                      </span>
                    </div>
                  </div>

                  {/* 分隔线 */}
                  <div className="my-3 border-t border-border-primary"></div>

                  {/* 详细指标 */}
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-text-muted">夏普比率</span>
                      <div className="font-medium">{result.sharpe_ratio?.toFixed(2) || '-'}</div>
                    </div>
                    <div>
                      <span className="text-text-muted">最大回撤</span>
                      <div className="font-medium text-red-500">{formatPercent(result.max_drawdown || 0)}</div>
                    </div>
                    <div>
                      <span className="text-text-muted">胜率</span>
                      <div className="font-medium">{formatPercent(result.win_rate || 0)}</div>
                    </div>
                    <div>
                      <span className="text-text-muted">交易次数</span>
                      <div className="font-medium">{result.total_trades || 0}</div>
                    </div>
                  </div>

                  {/* 超额收益 */}
                  {result.excess_return !== undefined && (
                    <div className="mt-3 pt-3 border-t border-border-primary">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-text-muted">超额收益</span>
                        <span className={`text-sm font-medium ${
                          result.excess_return > 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {formatPercent(result.excess_return)}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* 选中策略的详情图表 */}
            {currentResult && (
              <>
                {/* Equity Curve */}
                {currentResult.equity_curve && currentResult.equity_curve.length > 0 && (
                  <Card title={`${currentResult.strategy_display || currentResult.strategy} - 收益曲线`} noPadding>
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
