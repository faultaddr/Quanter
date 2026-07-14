export type EvidenceStrength = 'strong' | 'medium' | 'weak' | 'unverified';

export type ResearchVerdict =
  | 'top_priority'
  | 'high_priority'
  | 'worth_tracking'
  | 'early_lead';

export type ResearchTimingQuadrant =
  | 'priority_now'
  | 'research_wait'
  | 'timing_only'
  | 'low_priority';

export interface SerenityFactors {
  demand_inflection: number;
  architecture_coupling: number;
  chokepoint_severity: number;
  supplier_concentration: number;
  expansion_difficulty: number;
  evidence_quality: number;
  valuation_disconnect: number;
  catalyst_timing: number;
}

export interface SerenityPenalties {
  dilution_financing: number;
  governance: number;
  geopolitics: number;
  liquidity: number;
  hype_risk: number;
  accounting_quality: number;
  cyclicality: number;
  alternative_design_risk: number;
}

export interface SerenityEvidence {
  claim: string;
  source: string;
  strength: EvidenceStrength;
  published_at?: string | null;
}

export interface SerenityScorecardInput {
  ticker: string;
  company: string;
  market: string;
  theme: string;
  layer: string;
  role: string;
  factors: SerenityFactors;
  penalties: SerenityPenalties;
  evidence: SerenityEvidence[];
  what_could_weaken_view: string[];
  timing_score: number | null;
}

export interface SerenityScoreDetail {
  rating: number;
  weight: number;
  points: number;
}

export interface SerenityEvidenceSummary {
  total: number;
  strong: number;
  medium: number;
  weak: number;
  unverified: number;
}

export interface SerenityScoreResult {
  ticker: string;
  company: string;
  market: string;
  theme: string;
  layer: string;
  role: string;
  research_priority_score: number;
  raw_factor_points: number;
  penalty_points: number;
  verdict: ResearchVerdict;
  timing_score: number | null;
  quadrant: ResearchTimingQuadrant | null;
  factor_details: Record<keyof SerenityFactors, SerenityScoreDetail>;
  penalty_details: Record<keyof SerenityPenalties, SerenityScoreDetail>;
  evidence_summary: SerenityEvidenceSummary;
  evidence: SerenityEvidence[];
  what_could_weaken_view: string[];
}

export interface SerenityResponse<T> {
  success: boolean;
  data: T | null;
  error: string | null;
  timestamp: string;
}
