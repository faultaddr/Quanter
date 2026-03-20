'use client';

import { useState } from 'react';
import PageContainer from '@/components/layout/PageContainer';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import Badge from '@/components/ui/Badge';
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from '@/components/ui/Table';
import { useAppStore } from '@/stores/useAppStore';
import { formatPercent, getChangeColorClass } from '@/lib/utils';

interface FactorValidationResult {
  factor_name: string;
  ic_mean: number;
  ic_std: number;
  ir: number;
  long_short_return: number;
  overall_score: number;
  is_effective: boolean;
  recommendations: string[];
}

export default function FactorsPage() {
  const setActivePage = useAppStore((state) => state.setActivePage);

  const [factorName, setFactorName] = useState('');
  const [factorData, setFactorData] = useState('');
  const [returnsData, setReturnsData] = useState('');
  const [result, setResult] = useState<FactorValidationResult | null>(null);
  const [loading, setLoading] = useState(false);

  // Also manage optimization state
  const [factorNames, setFactorNames] = useState('factor1,factor2,factor3');
  const [icHistory, setIcHistory] = useState('');
  const [optimizeMethod, setOptimizeMethod] = useState('ir_weighted');
  const [optimizeResult, setOptimizeResult] = useState<any>(null);

  const handleValidate = async () => {
    if (!factorData || !returnsData) {
      return;
    }

    setLoading(true);

    try {
      // Parse input data
      const factorValues = factorData.split(',').map(Number);
      const returns = returnsData.split(',').map(Number);

      // Call the API
      const response = await fetch('/api/factors/validate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          factor_values: factorValues,
          returns: returns,
          factor_name: factorName || 'factor',
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Factor validation error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleOptimize = async () => {
    setLoading(true);

    try {
      const names = factorNames.split(',').map(s => s.trim());

      // Generate mock IC history for each factor
      const icHistory: Record<string, number[]> = {};
      names.forEach(name => {
        icHistory[name] = Array.from({ length: 20 }, () => (Math.random() - 0.3) * 0.1);
      });

      // Call the API
      const response = await fetch('/api/factors/optimize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          factor_names: names,
          ic_history: icHistory,
          method: optimizeMethod,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      setOptimizeResult(data);
    } catch (error) {
      console.error('Optimization error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageContainer>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900">因子研究</h1>
          <Badge variant="primary">因子有效性检验 & 权重优化</Badge>
        </div>

        {/* 因子有效性检验 */}
        <Card>
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">因子有效性检验</h2>
            <p className="text-sm text-gray-600">
              输入因子值和对应收益率，检验因子的预测能力（IC/IR分析）
            </p>

            <div className="grid grid-cols-2 gap-4">
              <Input
                label="因子名称"
                value={factorName}
                onChange={(e) => setFactorName(e.target.value)}
                placeholder="例如: pe_ratio"
              />
              <div className="flex items-end">
                <Button
                  onClick={handleValidate}
                  loading={loading}
                  disabled={!factorData || !returnsData}
                >
                  检验因子有效性
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  因子值序列（逗号分隔）
                </label>
                <textarea
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={3}
                  value={factorData}
                  onChange={(e) => setFactorData(e.target.value)}
                  placeholder="1.2, 3.4, 2.1, ..."
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  收益率序列（逗号分隔）
                </label>
                <textarea
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={3}
                  value={returnsData}
                  onChange={(e) => setReturnsData(e.target.value)}
                  placeholder="0.01, -0.02, 0.03, ..."
                />
              </div>
            </div>

            {result && (
              <div className="mt-4 p-4 bg-gray-50 rounded-lg space-y-3">
                <h3 className="font-semibold">检验结果</h3>
                <div className="grid grid-cols-4 gap-4">
                  <div>
                    <div className="text-sm text-gray-500">IC均值</div>
                    <div className={`text-lg font-semibold ${getChangeColorClass(result.ic_mean)}`}>
                      {formatPercent(result.ic_mean)}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-500">IC标准差</div>
                    <div className="text-lg font-semibold">
                      {formatPercent(result.ic_std)}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-500">IR</div>
                    <div className={`text-lg font-semibold ${getChangeColorClass(result.ir)}`}>
                      {result.ir.toFixed(2)}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-500">综合评分</div>
                    <div className="text-lg font-semibold">
                      {result.overall_score.toFixed(1)}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <Badge variant={result.is_effective ? 'success' : 'warning'}>
                    {result.is_effective ? '有效因子' : '弱因子'}
                  </Badge>
                  <span className="text-sm text-gray-600">
                    多空收益: {formatPercent(result.long_short_return)}
                  </span>
                </div>

                {result.recommendations.length > 0 && (
                  <div className="text-sm text-gray-600">
                    <strong>建议：</strong>
                    {result.recommendations.map((r, i) => (
                      <span key={i} className="block">• {r}</span>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </Card>

        {/* 因子权重优化 */}
        <Card>
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">因子权重优化</h2>
            <p className="text-sm text-gray-600">
              基于IC/IR历史数据，动态优化因子权重
            </p>

            <div className="grid grid-cols-3 gap-4">
              <Input
                label="因子名称（逗号分隔）"
                value={factorNames}
                onChange={(e) => setFactorNames(e.target.value)}
                placeholder="factor1,factor2,factor3"
              />
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  优化方法
                </label>
                <select
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={optimizeMethod}
                  onChange={(e) => setOptimizeMethod(e.target.value)}
                >
                  <option value="ir_weighted">IR加权</option>
                  <option value="ic_weighted">IC加权</option>
                  <option value="equal">等权</option>
                  <option value="risk_parity">风险平价</option>
                </select>
              </div>
              <div className="flex items-end">
                <Button
                  onClick={handleOptimize}
                  loading={loading}
                >
                  优化权重
                </Button>
              </div>
            </div>

            {optimizeResult && (
              <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                <h3 className="font-semibold mb-2">优化结果</h3>
                <div className="grid grid-cols-3 gap-4">
                  {Object.entries(optimizeResult.weights).map(([name, weight]) => (
                    <div key={name} className="text-center">
                      <div className="text-sm text-gray-500">{name}</div>
                      <div className="text-xl font-bold text-blue-600">
                        {formatPercent(Number(weight))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Card>
      </div>
    </PageContainer>
  );
}
