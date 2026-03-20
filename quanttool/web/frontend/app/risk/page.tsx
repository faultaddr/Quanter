'use client';

import { useState } from 'react';
import PageContainer from '@/components/layout/PageContainer';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import Badge from '@/components/ui/Badge';
import { useAppStore } from '@/stores/useAppStore';
import { formatPercent, formatNumber, getChangeColorClass } from '@/lib/utils';

interface IndustryViolation {
  industry: string;
  exposure: number;
  limit: number;
}

interface RiskReport {
  risk_score: number;
  industry_violations: IndustryViolation[];
  blacklist_violations: string[];
  position_shrink_factor: number;
  recommendations: string[];
  error?: string;
}

export default function RiskPage() {
  const setActivePage = useAppStore((state) => state.setActivePage);

  // Position input
  const [positionsInput, setPositionsInput] = useState('');
  const [industryMapInput, setIndustryMapInput] = useState('');
  const [portfolioValue, setPortfolioValue] = useState('1000000');
  const [peakValue, setPeakValue] = useState('1200000');

  const [result, setResult] = useState<RiskReport | null>(null);
  const [loading, setLoading] = useState(false);

  const handleCheck = async () => {
    setLoading(true);

    try {
      // Parse positions
      const positions: Record<string, { industry: string; value: number }> = {};
      try {
        const posArray = JSON.parse(positionsInput);
        for (const p of posArray) {
          if (p.symbol) {
            positions[p.symbol] = { industry: p.industry || '未知', value: p.value || 0 };
          }
        }
      } catch (e) {
        // Try line-by-line format
        const lines = positionsInput.trim().split('\n');
        for (const line of lines) {
          const [symbol, value, industry] = line.split(',').map(s => s.trim());
          if (symbol && value) {
            positions[symbol] = { industry: industry || '未知', value: parseFloat(value) };
          }
        }
      }

      // Parse industry map
      const industryMap: Record<string, string> = {};
      try {
        const mapObj = JSON.parse(industryMapInput);
        Object.assign(industryMap, mapObj);
      } catch (e) {
        const lines = industryMapInput.trim().split('\n');
        for (const line of lines) {
          const [symbol, industry] = line.split(',').map(s => s.trim());
          if (symbol && industry) {
            industryMap[symbol] = industry;
          }
        }
      }

      const portfolioValueNum = parseFloat(portfolioValue) || 1000000;
      const peakValueNum = parseFloat(peakValue) || portfolioValueNum;

      // Call the actual API
      const response = await fetch('/api/risk/portfolio/check', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          positions: positions,
          industry_map: industryMap,
          portfolio_value: portfolioValueNum,
          peak_value: peakValueNum,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Risk check error:', error);
    } finally {
      setLoading(false);
    }
  };

  const getRiskLevel = (score: number) => {
    if (score >= 80) return { label: '低风险', variant: 'success' as const };
    if (score >= 60) return { label: '中等风险', variant: 'warning' as const };
    if (score >= 40) return { label: '较高风险', variant: 'warning' as const };
    return { label: '高风险', variant: 'danger' as const };
  };

  return (
    <PageContainer>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900">组合风控</h1>
          <Badge variant="primary">行业暴露 & 黑名单监控</Badge>
        </div>

        <Card>
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">组合风险检查</h2>
            <p className="text-sm text-gray-600">
              检查组合的行业暴露、黑名单持仓、动态计算仓位收缩系数
            </p>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  持仓数据
                </label>
                <textarea
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={5}
                  value={positionsInput}
                  onChange={(e) => setPositionsInput(e.target.value)}
                  placeholder={'格式1: [{"symbol": "600000", "value": 100000, "industry": "银行"}]\n格式2:\n600000,100000,银行\n000001,80000,地产'}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  行业映射（可选）
                </label>
                <textarea
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={5}
                  value={industryMapInput}
                  onChange={(e) => setIndustryMapInput(e.target.value)}
                  placeholder={'格式1: {"600000": "银行", "000001": "地产"}\n格式2:\n600000,银行\n000001,地产'}
                />
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <Input
                label="当前组合价值"
                type="number"
                value={portfolioValue}
                onChange={(e) => setPortfolioValue(e.target.value)}
                placeholder="1000000"
              />
              <Input
                label="历史最高价值"
                type="number"
                value={peakValue}
                onChange={(e) => setPeakValue(e.target.value)}
                placeholder="1200000"
              />
              <div className="flex items-end">
                <Button
                  onClick={handleCheck}
                  loading={loading}
                >
                  检查风险
                </Button>
              </div>
            </div>

            {result && (
              <div className="mt-4 space-y-4">
                {/* 风险评分 */}
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-lg font-semibold">风险评分</span>
                    <Badge variant={getRiskLevel(result.risk_score).variant}>
                      {getRiskLevel(result.risk_score).label}
                    </Badge>
                  </div>
                  <div className="text-3xl font-bold text-blue-600">
                    {result.risk_score.toFixed(0)} / 100
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                    <div
                      className={`h-2 rounded-full ${
                        result.risk_score >= 80 ? 'bg-green-500' :
                        result.risk_score >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${result.risk_score}%` }}
                    />
                  </div>
                </div>

                {/* 行业暴露 */}
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h3 className="font-semibold mb-2">行业暴露检查</h3>
                  {result.industry_violations.length === 0 && (
                    <p className="text-green-600">OK 所有行业暴露在限制范围内</p>
                  )}
                  {result.industry_violations.length > 0 && (
                    <div className="space-y-2">
                      {result.industry_violations.map((v, i) => (
                        <div key={i} className="flex justify-between items-center text-sm">
                          <span className="text-red-600">{v.industry}</span>
                          <span>
                            <span className="text-red-600 font-semibold">
                              {formatPercent(v.exposure)}
                            </span>
                            <span className="text-gray-500"> / 限制 {formatPercent(v.limit)}</span>
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* 仓位收缩 */}
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h3 className="font-semibold mb-2">仓位收缩系数</h3>
                  <div className="text-2xl font-bold text-blue-600">
                    {formatPercent(result.position_shrink_factor)}
                  </div>
                  <p className="text-sm text-gray-600 mt-1">
                    基于当前回撤，建议的仓位收缩比例
                  </p>
                </div>

                {/* 黑名单 */}
                {result.blacklist_violations.length > 0 && (
                  <div className="p-4 bg-red-50 rounded-lg">
                    <h3 className="font-semibold mb-2 text-red-600">黑名单违规</h3>
                    <div className="flex gap-2">
                      {result.blacklist_violations.map((symbol, i) => (
                        <Badge key={i} variant="danger">{symbol}</Badge>
                      ))}
                    </div>
                    <p className="text-sm text-red-600 mt-1">
                      以上股票在黑名单中，建议卖出
                    </p>
                  </div>
                )}

                {/* 建议 */}
                {result.recommendations.length > 0 && (
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <h3 className="font-semibold mb-2">风险建议</h3>
                    <ul className="space-y-1">
                      {result.recommendations.map((r, i) => (
                        <li key={i} className="text-sm text-blue-700">
                          • {r}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </Card>

        {/* 说明 */}
        <Card>
          <h2 className="text-lg font-semibold mb-2">风控说明</h2>
          <ul className="text-sm text-gray-600 space-y-1">
            <li>• 行业暴露限制：默认单行业不超过20%</li>
            <li>• 黑名单：可配置禁止持仓的股票列表</li>
            <li>• 仓位收缩：回撤超过10%时建议减仓，超过20%建议大幅减仓</li>
            <li>• 动态调整：根据市场环境自动调整风控参数</li>
          </ul>
        </Card>
      </div>
    </PageContainer>
  );
}
