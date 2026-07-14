"""
基本面评级器 - 四维评级体系

评分维度：
1. 盈利能力 (25分): ROE、毛利率、净利率
2. 成长性 (25分): 营收增速、净利增速
3. 估值 (25分): PE、PB 合理性
4. 财务安全 (25分): 负债率、扣非EPS质量
"""
from typing import Dict, Tuple, List
from dataclasses import dataclass, field


@dataclass
class DimensionRating:
    """单维度评级"""
    score: float = 0.0          # 0-25
    stars: int = 0              # 1-5
    label: str = ""             # 一句话描述
    detail: str = ""            # 详细说明


@dataclass
class FundamentalRatingResult:
    """基本面评级结果"""
    total_score: float = 0.0    # 0-100
    total_label: str = ""       # 总评
    profitability: DimensionRating = field(default_factory=DimensionRating)
    growth: DimensionRating = field(default_factory=DimensionRating)
    valuation: DimensionRating = field(default_factory=DimensionRating)
    safety: DimensionRating = field(default_factory=DimensionRating)

    def to_dict(self) -> Dict:
        return {
            'total_score': self.total_score,
            'total_label': self.total_label,
            'profitability': {'score': self.profitability.score, 'stars': self.profitability.stars,
                               'label': self.profitability.label},
            'growth': {'score': self.growth.score, 'stars': self.growth.stars,
                       'label': self.growth.label},
            'valuation': {'score': self.valuation.score, 'stars': self.valuation.stars,
                          'label': self.valuation.label},
            'safety': {'score': self.safety.score, 'stars': self.safety.stars,
                       'label': self.safety.label},
        }


class FundamentalRating:
    """基本面评级器"""

    def rate(self, data: Dict) -> FundamentalRatingResult:
        """对基本面数据进行四维评级"""
        result = FundamentalRatingResult()

        result.profitability = self._rate_profitability(data)
        result.growth = self._rate_growth(data)
        result.valuation = self._rate_valuation(data)
        result.safety = self._rate_safety(data)

        result.total_score = (
            result.profitability.score + result.growth.score +
            result.valuation.score + result.safety.score
        )
        result.total_label = self._get_total_label(result.total_score)

        return result

    def _rate_profitability(self, data: Dict) -> DimensionRating:
        """盈利能力评级 (0-25)"""
        score = 0.0
        reasons = []

        roe = data.get('roe')
        gross_margin = data.get('gross_margin')
        profit_margin = data.get('profit_margin')

        # ROE (0-12分)
        if roe is not None:
            if roe >= 20:
                score += 12
                reasons.append("ROE优秀")
            elif roe >= 15:
                score += 10
                reasons.append("ROE良好")
            elif roe >= 10:
                score += 7
                reasons.append("ROE中等")
            elif roe >= 5:
                score += 4
            else:
                reasons.append("ROE偏低")

        # 毛利率 (0-8分)
        if gross_margin is not None:
            if gross_margin >= 60:
                score += 8
                reasons.append("毛利率极高")
            elif gross_margin >= 40:
                score += 6
                reasons.append("毛利率良好")
            elif gross_margin >= 25:
                score += 4
                reasons.append("毛利率中等")
            elif gross_margin < 15:
                score += 1
                reasons.append("毛利率偏低")

        # 净利率 (0-5分)
        if profit_margin is not None:
            if profit_margin >= 20:
                score += 5
                reasons.append("净利率优秀")
            elif profit_margin >= 10:
                score += 3
            elif profit_margin >= 5:
                score += 2

        return DimensionRating(
            score=score,
            stars=self._score_to_stars(score, 25),
            label="、".join(reasons) if reasons else "数据不足",
        )

    def _rate_growth(self, data: Dict) -> DimensionRating:
        """成长性评级 (0-25)"""
        score = 0.0
        reasons = []

        rev_yoy = data.get('revenue_yoy')
        profit_yoy = data.get('profit_yoy')

        # 营收增速 (0-13分)
        if rev_yoy is not None:
            if rev_yoy >= 30:
                score += 13
                reasons.append("营收高增长")
            elif rev_yoy >= 15:
                score += 10
                reasons.append("营收稳健增长")
            elif rev_yoy >= 5:
                score += 7
                reasons.append("营收温和增长")
            elif rev_yoy >= 0:
                score += 4
                reasons.append("营收停滞")
            elif rev_yoy >= -10:
                score += 2
                reasons.append("营收下滑")
            else:
                reasons.append("营收大幅下滑")

        # 净利增速 (0-12分)
        if profit_yoy is not None:
            if profit_yoy >= 30:
                score += 12
                reasons.append("净利高增长")
            elif profit_yoy >= 15:
                score += 9
            elif profit_yoy >= 0:
                score += 5
            elif profit_yoy >= -20:
                score += 2
                reasons.append("净利下滑")
            else:
                reasons.append("净利大幅下滑")

        # 从历史数据看趋势
        history = data.get('annual_history', [])
        if len(history) >= 3:
            recent_rev = [h.get('revenue', 0) for h in history[:3] if h.get('revenue')]
            if len(recent_rev) >= 3:
                if all(recent_rev[i] < recent_rev[i + 1] for i in range(len(recent_rev) - 1)):
                    reasons.append("营收连续下滑")
                    score = max(0, score - 3)

        return DimensionRating(
            score=score,
            stars=self._score_to_stars(score, 25),
            label="、".join(reasons) if reasons else "数据不足",
        )

    def _rate_valuation(self, data: Dict) -> DimensionRating:
        """估值评级 (0-25)"""
        score = 0.0
        reasons = []

        pe = data.get('pe_ttm')
        pb = data.get('pb')

        # PE (0-15分)
        if pe is not None and pe > 0:
            if pe <= 10:
                score += 15
                reasons.append("PE极低(低估)")
            elif pe <= 15:
                score += 13
                reasons.append("PE合理偏低")
            elif pe <= 25:
                score += 10
                reasons.append("PE合理")
            elif pe <= 35:
                score += 7
                reasons.append("PE偏高")
            elif pe <= 50:
                score += 4
                reasons.append("PE较高")
            else:
                score += 2
                reasons.append("PE极高")

        # PB (0-10分)
        if pb is not None and pb > 0:
            if pb <= 1:
                score += 10
                reasons.append("破净")
            elif pb <= 1.5:
                score += 8
            elif pb <= 3:
                score += 5
                reasons.append("PB中等")
            elif pb <= 5:
                score += 3
                reasons.append("PB偏高")
            else:
                score += 1
                reasons.append("PB极高")

        # 没有估值数据时给中间分
        if pe is None and pb is None:
            score = 12

        return DimensionRating(
            score=score,
            stars=self._score_to_stars(score, 25),
            label="、".join(reasons) if reasons else "数据不足",
        )

    def _rate_safety(self, data: Dict) -> DimensionRating:
        """财务安全评级 (0-25)"""
        score = 0.0
        reasons = []

        debt_ratio = data.get('debt_ratio')
        eps = data.get('eps')
        deduct_eps = data.get('deduct_eps')

        # 负债率 (0-15分)
        if debt_ratio is not None:
            if debt_ratio <= 20:
                score += 15
                reasons.append("负债率极低")
            elif debt_ratio <= 40:
                score += 12
                reasons.append("负债率低")
            elif debt_ratio <= 60:
                score += 8
                reasons.append("负债率中等")
            elif debt_ratio <= 70:
                score += 5
                reasons.append("负债率偏高")
            else:
                score += 2
                reasons.append("高负债风险")

        # 扣非EPS质量 (0-10分)
        if eps is not None and deduct_eps is not None and eps > 0:
            ratio = deduct_eps / eps
            if ratio >= 0.9:
                score += 10
                reasons.append("利润质量高")
            elif ratio >= 0.7:
                score += 7
                reasons.append("利润质量良好")
            elif ratio >= 0.5:
                score += 4
                reasons.append("利润质量一般")
            else:
                score += 2
                reasons.append("非经常性损益占比大")

        return DimensionRating(
            score=score,
            stars=self._score_to_stars(score, 25),
            label="、".join(reasons) if reasons else "数据不足",
        )

    @staticmethod
    def _score_to_stars(score: float, max_score: float) -> int:
        """将分数转为星级"""
        ratio = score / max_score if max_score > 0 else 0
        if ratio >= 0.8:
            return 5
        elif ratio >= 0.6:
            return 4
        elif ratio >= 0.4:
            return 3
        elif ratio >= 0.2:
            return 2
        else:
            return 1

    @staticmethod
    def _get_total_label(score: float) -> str:
        """获取总评标签"""
        if score >= 85:
            return "优质低估 — 盈利强、成长好、估值低、安全高"
        elif score >= 70:
            return "好公司 — 基本面扎实，需关注估值"
        elif score >= 55:
            return "中等 — 部分指标优秀，但有明显短板"
        elif score >= 40:
            return "偏弱 — 基本面有较大隐患"
        else:
            return "风险较高 — 多项指标不佳"
