'use client';

import { useState, useEffect } from 'react';
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
import { formatPercent, formatNumber, getChangeColorClass } from '@/lib/utils';

// 因子库数据（实际应从API获取）
const FACTOR_LIBRARY = [
  { name: 'pe_ttm', label: '市盈率(PE-TTM)', category: '估值', ic_mean: 0.05, ir: 0.42, description: '股票价格与每股收益的比率' },
  { name: 'pb', label: '市净率(PB)', category: '估值', ic_mean: 0.03, ir: 0.28, description: '股票价格与账面价值的比率' },
  { name: 'roe', label: 'ROE', category: '盈利', ic_mean: 0.08, ir: 0.55, description: '净资产收益率' },
  { name: 'gross_margin', label: '毛利率', category: '盈利', ic_mean: 0.06, ir: 0.48, description: '毛利润与营业收入的比率' },
  { name: 'revenue_growth', label: '营收增长', category: '成长', ic_mean: 0.07, ir: 0.52, description: '营业收入同比增长率' },
  { name: 'profit_growth', label: '利润增长', category: '成长', ic_mean: 0.06, ir: 0.45, description: '净利润同比增长率' },
  { name: 'volume_ratio', label: '量比', category: '技术', ic_mean: 0.04, ir: 0.35, description: '当日每分钟平均成交量与过去5日每分钟平均成交量之比' },
  { name: 'turnover', label: '换手率', category: '技术', ic_mean: 0.02, ir: 0.18, description: '股票在一定时期内的交易频率' },
  { name: 'momentum_20d', label: '20日动量', category: '技术', ic_mean: 0.06, ir: 0.50, description: '过去20个交易日的收益率' },
  { name: 'volatility', label: '波动率', category: '风险', ic_mean: -0.03, ir: -0.25, description: '收益率的标准差（负IC，低波动更优）' },
];

// 因子评分接口
interface FactorScore {
  factor: string;
  label: string;
  value: number;
  zscore: number;
  percentile: number;
  signal: 'buy' | 'sell' | 'neutral';
}

interface StockFactorAnalysis {
  symbol: string;
  name: string;
  overall_score: number;
  factor_scores: FactorScore[];
}

export default function FactorsPage() {
  const setActivePage = useAppStore((state) => state.setActivePage);

  // 搜索状态
  const [keyword, setKeyword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 分析结果
  const [analysis, setAnalysis] = useState<StockFactorAnalysis | null>(null);

  // 选中的因子详情
  const [selectedFactor, setSelectedFactor] = useState<typeof FACTOR_LIBRARY[0] | null>(null);

  const handleSearch = async () => {
    if (!keyword.trim()) {
      setError('请输入股票代码');
      return;
    }

    setLoading(true);
    setError(null);
    setAnalysis(null);

    try {
      // 调用API获取因子评分
      const response = await fetch(`/api/stock/${keyword.trim()}/factors`);
      if (!response.ok) {
        throw new Error('获取因子数据失败');
      }
      const data = await response.json();

      // 构造分析结果
      setAnalysis({
        symbol: keyword.trim(),
        name: data.name || keyword.trim(),
        overall_score: data.overall || 70,
        factor_scores: data.scores || [],
      });
    } catch (err) {
      console.error('Search error:', err);
      setError('未找到该股票的因子数据');
    } finally {
      setLoading(false);
    }
  };

  const getScoreLevel = (score: number) => {
    if (score >= 80) return { label: '优秀', color: 'text-green-600', bg: 'bg-green-50' };
    if (score >= 60) return { label: '良好', color: 'text-blue-600', bg: 'bg-blue-50' };
    if (score >= 40) return { label: '一般', color: 'text-yellow-600', bg: 'bg-yellow-50' };
    return { label: '较弱', color: 'text-red-600', bg: 'bg-red-50' };
  };

  const getSignalBadge = (signal: string) => {
    switch (signal) {
      case 'buy':
        return <Badge variant="success">买入信号</Badge>;
      case 'sell':
        return <Badge variant="danger">卖出信号</Badge>;
      default:
        return <Badge variant="default">中性</Badge>;
    }
  };

  return (
    <PageContainer>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-text-primary">因子研究</h1>
            <p className="text-text-muted mt-1">分析个股的因子表现，识别投资机会</p>
          </div>
          <Badge variant="primary">因子评分 & 有效性分析</Badge>
        </div>

        {/* 搜索区域 */}
        <Card>
          <div className="flex gap-4 items-end">
            <div className="flex-1">
              <Input
                label="股票代码"
                value={keyword}
                onChange={(e) => setKeyword(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="输入股票代码（如：600519）"
              />
              {error && <p className="text-sm text-danger mt-1">{error}</p>}
            </div>
            <Button onClick={handleSearch} loading={loading} disabled={!keyword.trim()}>
              分析
            </Button>
          </div>
        </Card>

        {/* 分析结果 */}
        {analysis && (
          <div className="space-y-6">
            {/* 综合评分 */}
            <Card>
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-bold">{analysis.name}</h2>
                  <p className="text-text-muted">{analysis.symbol}</p>
                </div>
                <div className="text-center">
                  <div className="text-4xl font-bold text-primary">{analysis.overall_score}</div>
                  <div className="text-text-muted">综合因子评分</div>
                </div>
                <Badge variant={analysis.overall_score >= 60 ? 'success' : 'warning'}>
                  {getScoreLevel(analysis.overall_score).label}
                </Badge>
              </div>
            </Card>

            {/* 因子明细 */}
            <Card title="因子评分明细">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>因子</TableHead>
                    <TableHead>当前值</TableHead>
                    <TableHead>Z-Score</TableHead>
                    <TableHead>历史分位</TableHead>
                    <TableHead>信号</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {analysis.factor_scores.map((score, idx) => (
                    <TableRow key={idx}>
                      <TableCell className="font-medium">{score.label}</TableCell>
                      <TableCell>{score.value.toFixed(2)}</TableCell>
                      <TableCell className={getChangeColorClass(score.zscore)}>
                        {score.zscore > 0 ? '+' : ''}{score.zscore.toFixed(2)}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-primary rounded-full"
                              style={{ width: `${score.percentile}%` }}
                            />
                          </div>
                          <span className="text-sm">{score.percentile}%</span>
                        </div>
                      </TableCell>
                      <TableCell>{getSignalBadge(score.signal)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Card>
          </div>
        )}

        {/* 因子库概览 */}
        <Card title="因子库概览">
          <p className="text-text-muted mb-4">
            点击因子查看详情，或在上方输入股票代码分析其因子表现
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {FACTOR_LIBRARY.map((factor) => (
              <div
                key={factor.name}
                className="p-4 border border-border-primary rounded-lg hover:border-primary cursor-pointer transition-colors"
                onClick={() => setSelectedFactor(factor)}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">{factor.label}</span>
                  <Badge variant="default">{factor.category}</Badge>
                </div>
                <p className="text-xs text-text-muted mb-3">{factor.description}</p>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <span className="text-text-muted">IC均值: </span>
                    <span className={getChangeColorClass(factor.ic_mean)}>
                      {formatPercent(factor.ic_mean)}
                    </span>
                  </div>
                  <div>
                    <span className="text-text-muted">IR: </span>
                    <span className={factor.ir > 0.3 ? 'text-success' : 'text-warning'}>
                      {factor.ir.toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* 因子详情弹窗 */}
        {selectedFactor && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setSelectedFactor(null)}>
            <div className="bg-bg-secondary rounded-lg p-6 max-w-md w-full mx-4" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold">{selectedFactor.label}</h3>
                <button onClick={() => setSelectedFactor(null)} className="text-text-muted hover:text-text-primary">
                  ✕
                </button>
              </div>
              <p className="text-text-muted mb-4">{selectedFactor.description}</p>
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="p-3 bg-bg-tertiary rounded-lg">
                  <div className="text-sm text-text-muted">IC均值</div>
                  <div className={`text-xl font-bold ${getChangeColorClass(selectedFactor.ic_mean)}`}>
                    {formatPercent(selectedFactor.ic_mean)}
                  </div>
                </div>
                <div className="p-3 bg-bg-tertiary rounded-lg">
                  <div className="text-sm text-text-muted">IR</div>
                  <div className={`text-xl font-bold ${selectedFactor.ir > 0.3 ? 'text-success' : 'text-warning'}`}>
                    {selectedFactor.ir.toFixed(2)}
                  </div>
                </div>
              </div>
              <div className="p-3 bg-bg-tertiary rounded-lg">
                <div className="text-sm text-text-muted mb-1">有效性评级</div>
                <div className="font-medium">
                  {selectedFactor.ir >= 0.5 ? '⭐⭐⭐ 强有效因子' :
                   selectedFactor.ir >= 0.3 ? '⭐⭐  中等有效因子' :
                   '⭐  弱有效因子'}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </PageContainer>
  );
}
