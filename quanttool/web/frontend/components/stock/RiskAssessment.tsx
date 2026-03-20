'use client';

import Card from '@/components/ui/Card';
import Badge from '@/components/ui/Badge';
import { formatNumber, formatPercent, getChangeColorClass } from '@/lib/utils';

interface RiskMetrics {
  volatility: number;         // 波动率（年化）
  max_drawdown: number;       // 最大回撤
  sharpe_ratio: number;       // 夏普比率
  sortino_ratio: number;      // 索提诺比率
  win_rate: number;           // 胜率
  profit_loss_ratio: number;  // 盈亏比
  avg_holding_days: number;   // 平均持仓天数
  beta: number;               // Beta 系数
  alpha: number;              // Alpha 系数
}

interface RiskAssessmentProps {
  metrics: RiskMetrics;
  period?: string;
}

function getRiskLevel(volatility: number, maxDrawdown: number): {
  level: string;
  color: string;
  description: string;
} {
  const riskScore = volatility * 100 + Math.abs(maxDrawdown) * 100;

  if (riskScore < 30) {
    return { level: '低风险', color: 'success', description: '该股票波动较小，风险较低' };
  } else if (riskScore < 60) {
    return { level: '中等风险', color: 'warning', description: '该股票波动适中，需注意风险控制' };
  } else {
    return { level: '高风险', color: 'danger', description: '该股票波动较大，建议谨慎投资' };
  }
}

function getSharpeRating(sharpe: number): { rating: string; color: string } {
  if (sharpe >= 2) return { rating: '优秀', color: 'success' };
  if (sharpe >= 1) return { rating: '良好', color: 'primary' };
  if (sharpe >= 0.5) return { rating: '一般', color: 'warning' };
  return { rating: '较差', color: 'danger' };
}

export default function RiskAssessment({ metrics, period = '近60日' }: RiskAssessmentProps) {
  const risk = getRiskLevel(metrics.volatility, metrics.max_drawdown);
  const sharpeRating = getSharpeRating(metrics.sharpe_ratio);

  return (
    <div className="space-y-4">
      {/* 风险等级概览 */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-text-primary">风险评估</h3>
          <Badge variant={risk.color as any}>{risk.level}</Badge>
        </div>
        <p className="text-sm text-text-muted">{risk.description}</p>
        <p className="text-xs text-text-muted mt-2">评估周期：{period}</p>
      </Card>

      {/* 核心指标网格 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Card>
          <div className="text-text-muted text-xs mb-1">年化波动率</div>
          <div className={`text-xl font-bold ${getChangeColorClass(metrics.volatility)}`}>
            {formatPercent(metrics.volatility)}
          </div>
          <div className="text-xs text-text-muted mt-1">风险度量</div>
        </Card>

        <Card>
          <div className="text-text-muted text-xs mb-1">最大回撤</div>
          <div className="text-xl font-bold text-danger">
            {formatPercent(metrics.max_drawdown)}
          </div>
          <div className="text-xs text-text-muted mt-1">最大亏损幅度</div>
        </Card>

        <Card>
          <div className="text-text-muted text-xs mb-1">夏普比率</div>
          <div className={`text-xl font-bold ${metrics.sharpe_ratio >= 1 ? 'text-success' : 'text-warning'}`}>
            {formatNumber(metrics.sharpe_ratio, 2)}
          </div>
          <Badge variant={sharpeRating.color as any} size="sm">{sharpeRating.rating}</Badge>
        </Card>

        <Card>
          <div className="text-text-muted text-xs mb-1">胜率</div>
          <div className={`text-xl font-bold ${metrics.win_rate >= 0.5 ? 'text-success' : 'text-warning'}`}>
            {formatPercent(metrics.win_rate)}
          </div>
          <div className="text-xs text-text-muted mt-1">盈利交易占比</div>
        </Card>
      </div>

      {/* 详细指标 */}
      <Card title="详细风险指标">
        <div className="space-y-3">
          <div className="flex justify-between items-center py-2 border-b border-border-primary">
            <span className="text-text-muted">索提诺比率</span>
            <span className="font-medium">{formatNumber(metrics.sortino_ratio, 2)}</span>
          </div>
          <div className="flex justify-between items-center py-2 border-b border-border-primary">
            <span className="text-text-muted">盈亏比</span>
            <span className="font-medium">{formatNumber(metrics.profit_loss_ratio, 2)}</span>
          </div>
          <div className="flex justify-between items-center py-2 border-b border-border-primary">
            <span className="text-text-muted">平均持仓天数</span>
            <span className="font-medium">{formatNumber(metrics.avg_holding_days, 1)}天</span>
          </div>
          <div className="flex justify-between items-center py-2 border-b border-border-primary">
            <span className="text-text-muted">Beta 系数</span>
            <span className={`font-medium ${metrics.beta > 1 ? 'text-danger' : 'text-success'}`}>
              {formatNumber(metrics.beta, 2)}
              {metrics.beta > 1 && <span className="text-xs ml-1">(高弹性)</span>}
              {metrics.beta < 1 && <span className="text-xs ml-1">(低弹性)</span>}
            </span>
          </div>
          <div className="flex justify-between items-center py-2">
            <span className="text-text-muted">Alpha 系数</span>
            <span className={`font-medium ${metrics.alpha > 0 ? 'text-success' : 'text-danger'}`}>
              {metrics.alpha > 0 ? '+' : ''}{formatPercent(metrics.alpha)}
              {metrics.alpha > 0 && <span className="text-xs ml-1">(超额收益)</span>}
            </span>
          </div>
        </div>
      </Card>

      {/* 风险提示 */}
      <Card className="bg-bg-tertiary">
        <div className="flex items-start gap-3">
          <span className="text-warning text-xl">⚠️</span>
          <div>
            <h4 className="font-medium text-text-primary mb-1">风险提示</h4>
            <ul className="text-sm text-text-muted space-y-1">
              <li>• 历史表现不代表未来收益</li>
              <li>• 高波动股票需要更强的风险承受能力</li>
              <li>• 建议设置止损位控制风险</li>
              <li>• 分散投资可降低组合风险</li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
}
