import { MetricTile, Section, StatusBadge } from '@/components/ui';
import type {
  EvidenceStrength,
  ResearchTimingQuadrant,
  ResearchVerdict,
  SerenityScoreDetail,
  SerenityScoreResult,
} from '@/types/research';

interface ResearchResultProps {
  result: SerenityScoreResult;
}

const verdictMeta: Record<ResearchVerdict, { label: string; tone: 'success' | 'primary' | 'warning' | 'muted' }> = {
  top_priority: { label: '最高研究优先级', tone: 'success' },
  high_priority: { label: '高研究优先级', tone: 'primary' },
  worth_tracking: { label: '值得跟踪', tone: 'warning' },
  early_lead: { label: '早期线索', tone: 'muted' },
};

const quadrantMeta: Record<ResearchTimingQuadrant, { label: string; detail: string }> = {
  priority_now: {
    label: '优先研究，时机较强',
    detail: '研究优先级与交易时机均处于高位，但仍需分别验证。',
  },
  research_wait: {
    label: '优先研究，等待时机',
    detail: '研究价值较高，当前交易时机尚未同步。',
  },
  timing_only: {
    label: '时机较强，研究优先级较低',
    detail: '交易时机信号较强，不代表产业链研究结论充分。',
  },
  low_priority: {
    label: '研究与时机均偏低',
    detail: '当前信息不足以支持更高研究顺位或更强时机判断。',
  },
};

const detailLabels: Record<string, string> = {
  demand_inflection: '需求拐点',
  architecture_coupling: '架构耦合',
  chokepoint_severity: '瓶颈强度',
  supplier_concentration: '供应集中度',
  expansion_difficulty: '扩产难度',
  evidence_quality: '证据质量',
  valuation_disconnect: '估值错配',
  catalyst_timing: '催化临近度',
  dilution_financing: '融资稀释',
  governance: '治理风险',
  geopolitics: '地缘风险',
  liquidity: '流动性风险',
  hype_risk: '叙事过热',
  accounting_quality: '会计质量',
  cyclicality: '周期波动',
  alternative_design_risk: '替代方案风险',
};

const evidenceLabels: Record<EvidenceStrength, string> = {
  strong: '强证据',
  medium: '中等证据',
  weak: '弱证据',
  unverified: '待核验',
};

function DetailRows({ details }: { details: Record<string, SerenityScoreDetail> }) {
  return (
    <div className="border-t border-border-primary">
      {Object.entries(details).map(([key, detail]) => (
        <div
          key={key}
          className="grid grid-cols-[minmax(0,1fr)_64px_72px] items-center gap-3 border-b border-border-primary py-2.5 text-sm"
        >
          <div className="min-w-0">
            <div className="truncate font-medium text-text-primary">{detailLabels[key] || key}</div>
            <div className="text-xs text-text-muted">权重 {detail.weight.toFixed(1)}</div>
          </div>
          <div className="text-right tabular-nums text-text-secondary">{detail.rating.toFixed(1)} / 5</div>
          <div className="text-right font-medium tabular-nums text-text-primary">{detail.points.toFixed(1)} 分</div>
        </div>
      ))}
    </div>
  );
}

export default function ResearchResult({ result }: ResearchResultProps) {
  const verdict = verdictMeta[result.verdict];
  const quadrant = result.quadrant ? quadrantMeta[result.quadrant] : null;
  const identity = [result.company, result.ticker].filter(Boolean).join(' / ') || '未命名研究对象';
  const evidenceCounts = [
    ['全部', result.evidence_summary.total],
    ['强', result.evidence_summary.strong],
    ['中等', result.evidence_summary.medium],
    ['弱', result.evidence_summary.weak],
    ['待核验', result.evidence_summary.unverified],
  ];

  return (
    <Section
      title="研究结果"
      description={`${identity} · ${result.market}`}
      action={<StatusBadge tone={verdict.tone}>{verdict.label}</StatusBadge>}
      className="border-t border-border-primary pt-6"
    >
      <div className="space-y-6">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <MetricTile
            label="研究优先级"
            value={`${result.research_priority_score.toFixed(1)} / 100`}
            detail="产业链研究轴"
            tone="accent"
          />
          <MetricTile
            label="交易时机"
            value={result.timing_score === null ? '未提供' : `${result.timing_score.toFixed(1)} / 100`}
            detail="独立时机轴"
            tone={result.timing_score === null ? 'muted' : 'default'}
          />
          <MetricTile
            label="因子原始分"
            value={result.raw_factor_points.toFixed(1)}
            detail="扣减风险前"
          />
          <MetricTile
            label="风险扣分"
            value={`-${result.penalty_points.toFixed(1)}`}
            detail="不改变独立时机分"
            tone={result.penalty_points > 0 ? 'warning' : 'muted'}
          />
        </div>

        <div className="border-y border-border-primary py-4">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-text-primary">研究 / 时机象限</span>
            <StatusBadge tone={quadrant ? 'primary' : 'muted'}>
              {quadrant?.label || '未提供交易时机'}
            </StatusBadge>
          </div>
          <p className="mt-2 text-sm text-text-muted">
            {quadrant?.detail || '研究优先级已计算；未提供独立交易时机分，因此不生成象限判断。'}
          </p>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div>
            <h3 className="mb-3 text-sm font-semibold text-text-primary">正向因子贡献</h3>
            <DetailRows details={result.factor_details} />
          </div>
          <div>
            <h3 className="mb-3 text-sm font-semibold text-text-primary">风险扣减明细</h3>
            <DetailRows details={result.penalty_details} />
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-text-primary">证据结构</h3>
          <div className="mt-3 grid grid-cols-2 border-y border-border-primary py-3 sm:grid-cols-5">
            {evidenceCounts.map(([label, count], index) => (
              <div
                key={label}
                className={`px-3 py-1 ${index > 0 ? 'border-l border-border-primary' : ''}`}
              >
                <div className="text-xs text-text-muted">{label}</div>
                <div className="mt-1 text-lg font-semibold tabular-nums text-text-primary">{count}</div>
              </div>
            ))}
          </div>
          {result.evidence.length > 0 && (
            <div className="mt-3 border-t border-border-primary">
              {result.evidence.map((item, index) => (
                <div key={`${item.source}-${index}`} className="border-b border-border-primary py-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <StatusBadge tone={item.strength === 'strong' ? 'success' : 'muted'}>
                      {evidenceLabels[item.strength]}
                    </StatusBadge>
                    {item.published_at && <span className="text-xs text-text-muted">{item.published_at}</span>}
                  </div>
                  <p className="mt-2 text-sm text-text-primary">{item.claim}</p>
                  <p className="mt-1 break-all text-xs text-text-muted">{item.source}</p>
                </div>
              ))}
            </div>
          )}
        </div>

        <div>
          <h3 className="text-sm font-semibold text-text-primary">可能削弱观点的条件</h3>
          {result.what_could_weaken_view.length > 0 ? (
            <ul className="mt-3 divide-y divide-border-primary border-y border-border-primary text-sm text-text-secondary">
              {result.what_could_weaken_view.map((condition, index) => (
                <li key={`${condition}-${index}`} className="py-3">{condition}</li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-sm text-text-muted">本次未记录削弱条件。</p>
          )}
        </div>
      </div>
    </Section>
  );
}
