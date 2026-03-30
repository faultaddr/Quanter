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
import { formatPercent, formatNumber, getChangeColorClass } from '@/lib/utils';

interface RiskItem {
  type: 'industry' | 'liquidity' | 'volatility' | 'blacklist' | 'concentration';
  severity: 'high' | 'medium' | 'low';
  title: string;
  description: string;
  suggestion: string;
}

interface RiskReport {
  overall_score: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  risk_items: RiskItem[];
  portfolio_metrics: {
    total_value: number;
    position_count: number;
    industry_count: number;
    top_holding_ratio: number;
    avg_volatility: number;
  };
}

// 模拟风控检查API
const mockRiskCheck = async (symbols: string[]): Promise<RiskReport> => {
  await new Promise(resolve => setTimeout(resolve, 1500));

  // 模拟生成风控报告
  const riskItems: RiskItem[] = [];

  // 模拟行业集中度风险
  if (symbols.length > 3) {
    riskItems.push({
      type: 'industry',
      severity: 'medium',
      title: '行业集中度偏高',
      description: '持仓中银行+金融类占比超过40%',
      suggestion: '建议分散至消费、科技等其他行业',
    });
  }

  // 模拟流动性风险
  if (symbols.length < 3) {
    riskItems.push({
      type: 'liquidity',
      severity: 'high',
      title: '持仓过于集中',
      description: '持仓股票数量少于3只',
      suggestion: '建议分散至5-10只股票',
    });
  }

  // 模拟波动率风险
  riskItems.push({
    type: 'volatility',
    severity: 'low',
    title: '波动率正常',
    description: '组合年化波动率在合理范围内',
    suggestion: '保持当前配置',
  });

  // 模拟黑名单
  const blacklisted = symbols.filter(s => s === '000001');
  if (blacklisted.length > 0) {
    riskItems.push({
      type: 'blacklist',
      severity: 'high',
      title: '持仓黑名单股票',
      description: `${blacklisted.join(', ')} 存在于黑名单中`,
      suggestion: '建议立即卖出',
    });
  }

  // 计算风险评分
  const severityScore = {
    high: 30,
    medium: 15,
    low: 5,
  };
  const deductedScore = riskItems.reduce((sum, item) => sum + severityScore[item.severity], 0);
  const overallScore = Math.max(0, 100 - deductedScore);

  let riskLevel: RiskReport['risk_level'] = 'low';
  if (overallScore < 40) riskLevel = 'critical';
  else if (overallScore < 60) riskLevel = 'high';
  else if (overallScore < 80) riskLevel = 'medium';

  return {
    overall_score: overallScore,
    risk_level: riskLevel,
    risk_items: riskItems,
    portfolio_metrics: {
      total_value: symbols.length * 100000,
      position_count: symbols.length,
      industry_count: Math.min(symbols.length, 5),
      top_holding_ratio: symbols.length > 0 ? 0.4 : 0,
      avg_volatility: 0.18,
    },
  };
};

export default function RiskPage() {
  const setActivePage = useAppStore((state) => state.setActivePage);

  // 输入状态
  const [inputMode, setInputMode] = useState<'text' | 'table'>('text');
  const [textInput, setTextInput] = useState('');
  const [positions, setPositions] = useState<{symbol: string; name: string; value: number}[]>([]);

  // 状态
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RiskReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  // 解析输入
  const parseInput = (): string[] => {
    if (inputMode === 'text') {
      // 支持多种格式：600519,000001 或 600519 000001 或 换行分隔
      return textInput
        .split(/[,，\s\n]+/)
        .map(s => s.trim().toUpperCase())
        .filter(s => /^\d{6}$/.test(s));
    } else {
      return positions.map(p => p.symbol).filter(s => s);
    }
  };

  // 搜索添加股票
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<{symbol: string; name: string}[]>([]);
  const [searching, setSearching] = useState(false);

  const handleSearch = async (query: string) => {
    if (!query || query.length < 1) {
      setSearchResults([]);
      return;
    }
    setSearching(true);
    try {
      const response = await fetch(`/api/realtime/search?query=${encodeURIComponent(query)}&limit=5`);
      const data = await response.json();
      setSearchResults(data.slice(0, 5));
    } catch (e) {
      console.error('Search error:', e);
    } finally {
      setSearching(false);
    }
  };

  const addPosition = (stock: {symbol: string; name: string}) => {
    if (!positions.find(p => p.symbol === stock.symbol)) {
      setPositions([...positions, { ...stock, value: 100000 }]);
    }
    setSearchQuery('');
    setSearchResults([]);
  };

  const updatePositionValue = (index: number, value: number) => {
    const newPositions = [...positions];
    newPositions[index].value = value;
    setPositions(newPositions);
  };

  const removePosition = (index: number) => {
    setPositions(positions.filter((_, i) => i !== index));
  };

  // 执行风控检查
  const handleCheck = async () => {
    const symbols = parseInput();
    if (symbols.length === 0) {
      setError('请输入至少一只股票代码');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const report = await mockRiskCheck(symbols);
      setResult(report);
    } catch (err) {
      setError('风控检查失败，请重试');
    } finally {
      setLoading(false);
    }
  };

  const getRiskLevelConfig = (level: RiskReport['risk_level']) => {
    switch (level) {
      case 'low':
        return { label: '低风险', color: 'text-green-600', bg: 'bg-green-50', badge: 'success' as const };
      case 'medium':
        return { label: '中等风险', color: 'text-yellow-600', bg: 'bg-yellow-50', badge: 'warning' as const };
      case 'high':
        return { label: '较高风险', color: 'text-orange-600', bg: 'bg-orange-50', badge: 'danger' as const };
      case 'critical':
        return { label: '高风险', color: 'text-red-600', bg: 'bg-red-50', badge: 'danger' as const };
    }
  };

  const getSeverityIcon = (severity: RiskItem['severity']) => {
    switch (severity) {
      case 'high':
        return <span className="text-red-500 text-xl">⚠️</span>;
      case 'medium':
        return <span className="text-yellow-500 text-xl">⚡</span>;
      case 'low':
        return <span className="text-green-500 text-xl">✓</span>;
    }
  };

  const totalValue = positions.reduce((sum, p) => sum + p.value, 0);

  return (
    <PageContainer>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-text-primary">组合风控</h1>
            <p className="text-text-muted mt-1">一键扫描持仓风险，获取调仓建议</p>
          </div>
          <Badge variant="primary">智能风控检测</Badge>
        </div>

        {/* 输入方式切换 */}
        <Card>
          <div className="flex gap-4 mb-4">
            <button
              className={`px-4 py-2 rounded-lg transition-colors ${inputMode === 'text' ? 'bg-primary text-white' : 'bg-bg-tertiary text-text-muted'}`}
              onClick={() => setInputMode('text')}
            >
              文本输入
            </button>
            <button
              className={`px-4 py-2 rounded-lg transition-colors ${inputMode === 'table' ? 'bg-primary text-white' : 'bg-bg-tertiary text-text-muted'}`}
              onClick={() => setInputMode('table')}
            >
              持仓表格
            </button>
          </div>

          {inputMode === 'text' ? (
            <div className="space-y-4">
              <Input
                label="股票代码"
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleCheck()}
                placeholder="输入股票代码，多个用逗号或空格分隔，如：600519,000001"
              />
              <p className="text-sm text-text-muted">
                支持格式：600519,000001 或 600519 000001
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {/* 搜索添加 */}
              <div className="relative">
                <Input
                  label="添加股票"
                  value={searchQuery}
                  onChange={(e) => {
                    setSearchQuery(e.target.value);
                    handleSearch(e.target.value);
                  }}
                  placeholder="输入股票代码或名称搜索..."
                />
                {searchResults.length > 0 && (
                  <div className="absolute z-10 w-full mt-1 bg-bg-secondary border border-border-primary rounded-lg shadow-lg">
                    {searchResults.map(stock => (
                      <button
                        key={stock.symbol}
                        className="w-full px-4 py-2 text-left hover:bg-bg-tertiary flex justify-between"
                        onClick={() => addPosition(stock)}
                      >
                        <span className="font-medium">{stock.symbol}</span>
                        <span className="text-text-muted">{stock.name}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* 持仓表格 */}
              {positions.length > 0 && (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>股票</TableHead>
                      <TableHead>持仓金额(元)</TableHead>
                      <TableHead>占比</TableHead>
                      <TableHead></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {positions.map((pos, idx) => (
                      <TableRow key={pos.symbol}>
                        <TableCell className="font-medium">{pos.symbol} {pos.name}</TableCell>
                        <TableCell>
                          <input
                            type="number"
                            className="w-32 px-2 py-1 border border-border-primary rounded bg-bg-tertiary"
                            value={pos.value}
                            onChange={(e) => updatePositionValue(idx, Number(e.target.value))}
                          />
                        </TableCell>
                        <TableCell>{formatPercent(totalValue > 0 ? pos.value / totalValue : 0)}</TableCell>
                        <TableCell>
                          <button
                            onClick={() => removePosition(idx)}
                            className="text-danger hover:text-red-700"
                          >
                            删除
                          </button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}

              <div className="flex justify-between text-sm text-text-muted">
                <span>共 {positions.length} 只股票</span>
                <span>总市值: {formatNumber(totalValue)} 元</span>
              </div>
            </div>
          )}

          {error && <p className="text-sm text-danger mt-2">{error}</p>}

          <Button onClick={handleCheck} loading={loading} className="w-full mt-4">
            开始风控扫描
          </Button>
        </Card>

        {/* 风控结果 */}
        {result && (
          <div className="space-y-6">
            {/* 风险仪表盘 */}
            <Card>
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-bold">风险评分</h2>
                  <p className="text-text-muted">综合评估结果</p>
                </div>
                <div className="text-right">
                  <div className={`text-5xl font-bold ${getRiskLevelConfig(result.risk_level).color}`}>
                    {result.overall_score}
                  </div>
                  <Badge variant={getRiskLevelConfig(result.risk_level).badge}>
                    {getRiskLevelConfig(result.risk_level).label}
                  </Badge>
                </div>
              </div>

              {/* 风险指标 */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
                <div className="p-3 bg-bg-tertiary rounded-lg text-center">
                  <div className="text-2xl font-bold">{result.portfolio_metrics.position_count}</div>
                  <div className="text-sm text-text-muted">持仓数量</div>
                </div>
                <div className="p-3 bg-bg-tertiary rounded-lg text-center">
                  <div className="text-2xl font-bold">{result.portfolio_metrics.industry_count}</div>
                  <div className="text-sm text-text-muted">涉及行业</div>
                </div>
                <div className="p-3 bg-bg-tertiary rounded-lg text-center">
                  <div className="text-2xl font-bold">{formatPercent(result.portfolio_metrics.top_holding_ratio)}</div>
                  <div className="text-sm text-text-muted">持仓集中度</div>
                </div>
                <div className="p-3 bg-bg-tertiary rounded-lg text-center">
                  <div className="text-2xl font-bold">{formatPercent(result.portfolio_metrics.avg_volatility)}</div>
                  <div className="text-sm text-text-muted">平均波动率</div>
                </div>
              </div>
            </Card>

            {/* 风险项详情 */}
            <Card title="风险提示">
              <div className="space-y-3">
                {result.risk_items.map((item, idx) => (
                  <div
                    key={idx}
                    className={`p-4 rounded-lg border ${item.severity === 'high' ? 'border-red-200 bg-red-50' : item.severity === 'medium' ? 'border-yellow-200 bg-yellow-50' : 'border-green-200 bg-green-50'}`}
                  >
                    <div className="flex items-start gap-3">
                      {getSeverityIcon(item.severity)}
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-medium">{item.title}</span>
                          <Badge variant={item.severity === 'high' ? 'danger' : item.severity === 'medium' ? 'warning' : 'success'}>
                            {item.severity === 'high' ? '高' : item.severity === 'medium' ? '中' : '低'}
                          </Badge>
                        </div>
                        <p className="text-sm text-text-muted">{item.description}</p>
                        <p className="text-sm text-primary mt-1">💡 {item.suggestion}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            {/* 无风险提示 */}
            {result.risk_items.length === 0 && (
              <Card>
                <div className="text-center py-8">
                  <div className="text-4xl mb-4">🎉</div>
                  <h3 className="text-xl font-bold text-success">风险检查通过</h3>
                  <p className="text-text-muted mt-2">您的持仓组合暂无明显风险点</p>
                </div>
              </Card>
            )}
          </div>
        )}
      </div>
    </PageContainer>
  );
}
