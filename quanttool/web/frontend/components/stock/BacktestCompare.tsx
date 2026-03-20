'use client';

import { useState } from 'react';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import EquityCurve from '@/components/charts/EquityCurve';
import { formatNumber, formatPercent, formatAmount, getChangeColorClass } from '@/lib/utils';

interface BacktestResult {
  strategy_name: string;
  total_return: number;
  annual_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  equity_curve: { date: string; value: number }[];
}

interface BacktestCompareProps {
  results: BacktestResult[];
  benchmark?: {
    name: string;
    total_return: number;
    equity_curve: { date: string; value: number }[];
  };
  onSelectStrategy?: (strategyName: string) => void;
}

export default function BacktestCompare({
  results,
  benchmark,
  onSelectStrategy,
}: BacktestCompareProps) {
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(
    results[0]?.strategy_name || null
  );

  const selectedResult = results.find(r => r.strategy_name === selectedStrategy);

  const getReturnColor = (value: number) => {
    if (value > 0.1) return 'success';
    if (value > 0) return 'primary';
    return 'danger';
  };

  const getSharpeBadge = (sharpe: number) => {
    if (sharpe >= 2) return <Badge variant="success">优秀</Badge>;
    if (sharpe >= 1) return <Badge variant="primary">良好</Badge>;
    if (sharpe >= 0.5) return <Badge variant="warning">一般</Badge>;
    return <Badge variant="danger">较差</Badge>;
  };

  if (results.length === 0) {
    return (
      <Card className="text-center py-10">
        <p className="text-text-muted">暂无回测结果</p>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* 策略选择器 */}
      <div className="flex flex-wrap gap-2">
        {results.map((result) => (
          <Button
            key={result.strategy_name}
            size="sm"
            variant={selectedStrategy === result.strategy_name ? 'primary' : 'secondary'}
            onClick={() => {
              setSelectedStrategy(result.strategy_name);
              onSelectStrategy?.(result.strategy_name);
            }}
          >
            {result.strategy_name}
          </Button>
        ))}
        {benchmark && (
          <Button
            size="sm"
            variant="ghost"
            disabled
          >
            📊 {benchmark.name}（基准）
          </Button>
        )}
      </div>

      {/* 收益对比图表 */}
      {selectedResult && (
        <Card title="收益曲线对比" noPadding>
          <div className="p-4">
            <EquityCurve
              data={selectedResult.equity_curve}
              benchmarkData={benchmark?.equity_curve}
              height={300}
            />
          </div>
        </Card>
      )}

      {/* 策略对比表格 */}
      <Card title="策略表现对比">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border-primary">
                <th className="text-left py-3 px-2 text-text-muted text-sm">策略</th>
                <th className="text-right py-3 px-2 text-text-muted text-sm">总收益</th>
                <th className="text-right py-3 px-2 text-text-muted text-sm">年化收益</th>
                <th className="text-right py-3 px-2 text-text-muted text-sm">夏普比率</th>
                <th className="text-right py-3 px-2 text-text-muted text-sm">最大回撤</th>
                <th className="text-right py-3 px-2 text-text-muted text-sm">胜率</th>
                <th className="text-right py-3 px-2 text-text-muted text-sm">交易次数</th>
              </tr>
            </thead>
            <tbody>
              {results.map((result) => (
                <tr
                  key={result.strategy_name}
                  className={`border-b border-border-primary cursor-pointer hover:bg-bg-tertiary ${
                    selectedStrategy === result.strategy_name ? 'bg-primary/10' : ''
                  }`}
                  onClick={() => setSelectedStrategy(result.strategy_name)}
                >
                  <td className="py-3 px-2 font-medium">{result.strategy_name}</td>
                  <td className={`py-3 px-2 text-right ${getChangeColorClass(result.total_return)}`}>
                    {formatPercent(result.total_return)}
                  </td>
                  <td className={`py-3 px-2 text-right ${getChangeColorClass(result.annual_return)}`}>
                    {formatPercent(result.annual_return)}
                  </td>
                  <td className="py-3 px-2 text-right">
                    <div className="flex items-center justify-end gap-2">
                      {formatNumber(result.sharpe_ratio, 2)}
                      {getSharpeBadge(result.sharpe_ratio)}
                    </div>
                  </td>
                  <td className="py-3 px-2 text-right text-danger">
                    {formatPercent(result.max_drawdown)}
                  </td>
                  <td className="py-3 px-2 text-right">
                    {formatPercent(result.win_rate)}
                  </td>
                  <td className="py-3 px-2 text-right">
                    {result.total_trades}
                  </td>
                </tr>
              ))}
              {benchmark && (
                <tr className="border-b border-border-primary bg-bg-tertiary">
                  <td className="py-3 px-2 font-medium">📊 {benchmark.name}</td>
                  <td className={`py-3 px-2 text-right ${getChangeColorClass(benchmark.total_return)}`}>
                    {formatPercent(benchmark.total_return)}
                  </td>
                  <td className="py-3 px-2 text-right">-</td>
                  <td className="py-3 px-2 text-right">-</td>
                  <td className="py-3 px-2 text-right">-</td>
                  <td className="py-3 px-2 text-right">-</td>
                  <td className="py-3 px-2 text-right">-</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* 详细指标 */}
      {selectedResult && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <div className="text-text-muted text-sm mb-1">总收益率</div>
            <div className={`text-2xl font-bold ${getChangeColorClass(selectedResult.total_return)}`}>
              {formatPercent(selectedResult.total_return)}
            </div>
          </Card>
          <Card>
            <div className="text-text-muted text-sm mb-1">年化收益</div>
            <div className={`text-2xl font-bold ${getChangeColorClass(selectedResult.annual_return)}`}>
              {formatPercent(selectedResult.annual_return)}
            </div>
          </Card>
          <Card>
            <div className="text-text-muted text-sm mb-1">夏普比率</div>
            <div className="text-2xl font-bold">
              {formatNumber(selectedResult.sharpe_ratio, 2)}
            </div>
          </Card>
          <Card>
            <div className="text-text-muted text-sm mb-1">最大回撤</div>
            <div className="text-2xl font-bold text-danger">
              {formatPercent(selectedResult.max_drawdown)}
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
