'use client';

import { FormEvent, useRef, useState } from 'react';
import PageContainer from '@/components/layout/PageContainer';
import { ResearchResult, ScoreField } from '@/components/research';
import { Button, Input, PageHeader, Section, StatusBadge } from '@/components/ui';
import { researchApi } from '@/lib/api/research';
import type {
  EvidenceStrength,
  SerenityEvidence,
  SerenityFactors,
  SerenityPenalties,
  SerenityScorecardInput,
  SerenityScoreResult,
} from '@/types/research';

const factorFields: Array<{ key: keyof SerenityFactors; label: string; description: string }> = [
  { key: 'demand_inflection', label: '需求拐点', description: '终端需求是否出现可验证的增量或结构变化。' },
  { key: 'architecture_coupling', label: '架构耦合', description: '供给是否深度嵌入系统架构与客户流程。' },
  { key: 'chokepoint_severity', label: '瓶颈强度', description: '短缺或替代困难对产业链形成的实际约束。' },
  { key: 'supplier_concentration', label: '供应集中度', description: '关键供给是否集中于少数合格厂商。' },
  { key: 'expansion_difficulty', label: '扩产难度', description: '新增产能受资本、工艺、认证或周期制约的程度。' },
  { key: 'evidence_quality', label: '证据质量', description: '判断是否由多源、直接且近期的证据支持。' },
  { key: 'valuation_disconnect', label: '估值错配', description: '市场定价是否尚未反映瓶颈位置与兑现能力。' },
  { key: 'catalyst_timing', label: '催化临近度', description: '研究催化是否进入能够持续验证的窗口。' },
];

const penaltyFields: Array<{ key: keyof SerenityPenalties; label: string; description: string }> = [
  { key: 'dilution_financing', label: '融资稀释', description: '再融资、股权激励或资本开支带来的股东稀释风险。' },
  { key: 'governance', label: '治理风险', description: '控制权、关联交易与治理透明度的不确定性。' },
  { key: 'geopolitics', label: '地缘风险', description: '贸易、制裁、出口管制或区域冲突的影响。' },
  { key: 'liquidity', label: '流动性风险', description: '成交深度、股权结构与退出约束。' },
  { key: 'hype_risk', label: '叙事过热', description: '市场预期领先于订单、产能或利润兑现的程度。' },
  { key: 'accounting_quality', label: '会计质量', description: '收入确认、现金流与资产质量的可信度。' },
  { key: 'cyclicality', label: '周期波动', description: '行业周期对需求、价格和盈利稳定性的影响。' },
  { key: 'alternative_design_risk', label: '替代方案风险', description: '新架构、替代材料或竞争路线削弱瓶颈价值的可能。' },
];

const identityFields: Array<{
  key: 'ticker' | 'company' | 'market' | 'theme' | 'layer' | 'role';
  label: string;
  placeholder: string;
}> = [
  { key: 'ticker', label: '标的代码', placeholder: '如 688012.SH' },
  { key: 'company', label: '公司名称', placeholder: '研究对象' },
  { key: 'market', label: '市场', placeholder: '如 A-share' },
  { key: 'theme', label: '研究主题', placeholder: '如 AI 算力基础设施' },
  { key: 'layer', label: '产业链层级', placeholder: '如 半导体设备' },
  { key: 'role', label: '瓶颈角色', placeholder: '如 先进制程关键设备' },
];

const strengthOptions: Array<{ value: EvidenceStrength; label: string }> = [
  { value: 'strong', label: '强证据' },
  { value: 'medium', label: '中等证据' },
  { value: 'weak', label: '弱证据' },
  { value: 'unverified', label: '待核验' },
];

const inputClass = 'h-10 w-full rounded-lg border border-border-primary bg-bg-tertiary px-3 text-sm text-text-primary focus:border-transparent focus:outline-none focus:ring-2 focus:ring-primary';

function createInitialScorecard(): SerenityScorecardInput {
  return {
    ticker: '',
    company: '',
    market: 'A-share',
    theme: '',
    layer: '',
    role: '',
    factors: Object.fromEntries(factorFields.map(({ key }) => [key, 0])) as unknown as SerenityFactors,
    penalties: Object.fromEntries(penaltyFields.map(({ key }) => [key, 0])) as unknown as SerenityPenalties,
    evidence: [{ claim: '', source: '', strength: 'unverified', published_at: null }],
    what_could_weaken_view: [''],
    timing_score: null,
  };
}

export default function ResearchPage() {
  const [scorecard, setScorecard] = useState<SerenityScorecardInput>(createInitialScorecard);
  const [result, setResult] = useState<SerenityScoreResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nextRowId = useRef(1);
  const evidenceRowIds = useRef(['evidence-0']);
  const weakeningRowIds = useRef(['weakening-0']);

  const addEvidence = () => {
    evidenceRowIds.current.push(`evidence-${nextRowId.current++}`);
    setScorecard((current) => ({
      ...current,
      evidence: [...current.evidence, { claim: '', source: '', strength: 'unverified', published_at: null }],
    }));
  };

  const removeEvidence = (index: number) => {
    evidenceRowIds.current = evidenceRowIds.current.filter((_, itemIndex) => itemIndex !== index);
    setScorecard((current) => ({
      ...current,
      evidence: current.evidence.filter((_, itemIndex) => itemIndex !== index),
    }));
  };

  const addWeakeningCondition = () => {
    weakeningRowIds.current.push(`weakening-${nextRowId.current++}`);
    setScorecard((current) => ({
      ...current,
      what_could_weaken_view: [...current.what_could_weaken_view, ''],
    }));
  };

  const removeWeakeningCondition = (index: number) => {
    weakeningRowIds.current = weakeningRowIds.current.filter((_, itemIndex) => itemIndex !== index);
    setScorecard((current) => ({
      ...current,
      what_could_weaken_view: current.what_could_weaken_view.filter((_, itemIndex) => itemIndex !== index),
    }));
  };

  const updateEvidence = (index: number, key: keyof SerenityEvidence, value: string) => {
    setScorecard((current) => ({
      ...current,
      evidence: current.evidence.map((item, itemIndex) =>
        itemIndex === index ? { ...item, [key]: value || (key === 'published_at' ? null : value) } : item
      ),
    }));
  };

  const updateWeakeningCondition = (index: number, value: string) => {
    setScorecard((current) => ({
      ...current,
      what_could_weaken_view: current.what_could_weaken_view.map((item, itemIndex) =>
        itemIndex === index ? value : item
      ),
    }));
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    const payload: SerenityScorecardInput = {
      ...scorecard,
      evidence: scorecard.evidence.filter((item) => item.claim.trim() || item.source.trim()),
      what_could_weaken_view: scorecard.what_could_weaken_view.filter((item) => item.trim()),
    };

    try {
      setResult(await researchApi.scorecard(payload));
    } catch (requestError) {
      console.error('Serenity research request failed:', requestError);
      setError(requestError instanceof Error ? requestError.message : '研究评分请求失败，请保留当前输入并重试。');
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageContainer>
      <form className="space-y-7" onSubmit={handleSubmit}>
        <PageHeader
          title="产业链研究"
          description="以瓶颈、证据与估值错配确定研究顺序。研究优先级不是交易建议。"
          meta={
            <>
              <StatusBadge tone="primary">Serenity 手工评分</StatusBadge>
              <span className="text-xs text-text-muted">研究优先级与交易时机独立计算</span>
            </>
          }
        />

        <Section title="研究对象" description="标的身份、产业链位置与瓶颈角色。" className="border-t border-border-primary pt-6">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {identityFields.map((field) => (
              <Input
                key={field.key}
                label={field.label}
                placeholder={field.placeholder}
                value={scorecard[field.key]}
                onChange={(event) => setScorecard((current) => ({ ...current, [field.key]: event.target.value }))}
              />
            ))}
          </div>
          <div className="mt-5 max-w-sm">
            <Input
              label="独立交易时机分（可选）"
              type="number"
              min={0}
              max={100}
              step={1}
              placeholder="0 - 100"
              value={scorecard.timing_score ?? ''}
              onChange={(event) => setScorecard((current) => ({
                ...current,
                timing_score: event.target.value === '' ? null : Number(event.target.value),
              }))}
            />
            <p className="mt-1.5 text-xs text-text-muted">该分数只用于交易时机轴，不参与研究优先级计算。</p>
          </div>
        </Section>

        <Section title="研究因子" description="从需求、瓶颈与证据强度评估研究价值。" className="border-t border-border-primary pt-6">
          <div className="grid grid-cols-1 gap-x-8 lg:grid-cols-2">
            {factorFields.map((field) => (
              <ScoreField
                key={field.key}
                name={field.key}
                label={field.label}
                description={field.description}
                value={scorecard.factors[field.key]}
                onChange={(value) => setScorecard((current) => ({
                  ...current,
                  factors: { ...current.factors, [field.key]: value },
                }))}
              />
            ))}
          </div>
        </Section>

        <Section title="风险扣减" description="记录会削弱研究优先级的结构性风险。" className="border-t border-border-primary pt-6">
          <div className="grid grid-cols-1 gap-x-8 lg:grid-cols-2">
            {penaltyFields.map((field) => (
              <ScoreField
                key={field.key}
                name={field.key}
                label={field.label}
                description={field.description}
                value={scorecard.penalties[field.key]}
                onChange={(value) => setScorecard((current) => ({
                  ...current,
                  penalties: { ...current.penalties, [field.key]: value },
                }))}
              />
            ))}
          </div>
        </Section>

        <Section
          title="研究证据"
          description="保留结论、来源、强度与发布日期。"
          action={<Button type="button" size="sm" variant="secondary" onClick={addEvidence}>添加证据</Button>}
          className="border-t border-border-primary pt-6"
        >
          <div className="divide-y divide-border-primary border-y border-border-primary">
            {scorecard.evidence.map((item, index) => (
              <div key={evidenceRowIds.current[index]} className="grid grid-cols-1 gap-3 py-4 md:grid-cols-2 lg:grid-cols-[minmax(0,2fr)_minmax(0,2fr)_140px_160px_auto] lg:items-end">
                <Input label="研究结论" value={item.claim} onChange={(event) => updateEvidence(index, 'claim', event.target.value)} />
                <Input label="来源" value={item.source} onChange={(event) => updateEvidence(index, 'source', event.target.value)} />
                <label className="text-sm font-medium text-text-secondary">
                  证据强度
                  <select className={`${inputClass} mt-1.5`} value={item.strength} onChange={(event) => updateEvidence(index, 'strength', event.target.value)}>
                    {strengthOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                  </select>
                </label>
                <Input label="发布日期" type="date" value={item.published_at ?? ''} onChange={(event) => updateEvidence(index, 'published_at', event.target.value)} />
                <Button type="button" size="sm" variant="ghost" disabled={scorecard.evidence.length === 1} onClick={() => removeEvidence(index)}>删除</Button>
              </div>
            ))}
          </div>
        </Section>

        <Section
          title="观点削弱条件"
          description="明确哪些事实变化会推翻或降低当前研究结论。"
          action={<Button type="button" size="sm" variant="secondary" onClick={addWeakeningCondition}>添加条件</Button>}
          className="border-t border-border-primary pt-6"
        >
          <div className="divide-y divide-border-primary border-y border-border-primary">
            {scorecard.what_could_weaken_view.map((condition, index) => (
              <div key={weakeningRowIds.current[index]} className="flex items-center gap-3 py-3">
                <input aria-label={`削弱条件 ${index + 1}`} className={inputClass} value={condition} onChange={(event) => updateWeakeningCondition(index, event.target.value)} placeholder="例如：客户切换到替代架构" />
                <Button type="button" size="sm" variant="ghost" disabled={scorecard.what_could_weaken_view.length === 1} onClick={() => removeWeakeningCondition(index)}>删除</Button>
              </div>
            ))}
          </div>
        </Section>

        {error && <div role="alert" className="border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger">{error}</div>}

        <div className="flex justify-end border-t border-border-primary pt-5">
          <Button type="submit" loading={loading}>生成研究评分</Button>
        </div>

        <div aria-live="polite">
          {result && <ResearchResult result={result} />}
        </div>
      </form>
    </PageContainer>
  );
}
