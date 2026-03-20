"""
评分解释生成器模块

生成可解释的评分分析报告：
- 评分因素分解
- 主要加减分项说明
- 交易理由生成
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ScoreExplanation:
    """评分解释"""
    total_score: float
    grade: str
    main_positive_factors: List[Dict]   # 主要加分项
    main_negative_factors: List[Dict]   # 主要减分项
    signal_suggestion: str               # 信号建议
    risk_warning: str                    # 风险警告
    stop_loss_price: Optional[float]     # 止损价
    confidence: float                    # 置信度


class ScoreExplainer:
    """
    评分解释生成器

    将评分转化为可理解的投资建议
    """

    # 评分等级
    GRADE_THRESHOLDS = {
        'A+': 90,
        'A': 80,
        'B+': 70,
        'B': 60,
        'C+': 50,
        'C': 40,
        'D': 30,
        'F': 0,
    }

    # 因子解释模板
    FACTOR_EXPLANATIONS = {
        # 趋势因子
        'trend_strength': {
            'name': '趋势强度',
            'positive': '股价在MA20上方运行，趋势向上',
            'negative': '股价跌破MA20，趋势走弱',
            'weight': 0.30,
        },
        'ma_slope': {
            'name': '均线斜率',
            'positive': '均线向上发散，多头排列',
            'negative': '均线向下发散，空头排列',
            'weight': 0.10,
        },
        'macd_momentum': {
            'name': 'MACD动量',
            'positive': 'MACD金叉，动量向上',
            'negative': 'MACD死叉，动量向下',
            'weight': 0.30,
        },
        'money_flow': {
            'name': '资金流向',
            'positive': 'OBV上升，资金流入',
            'negative': 'OBV下降，资金流出',
            'weight': 0.20,
        },
        'volume_ratio': {
            'name': '成交量比',
            'positive': '成交量放大，市场活跃',
            'negative': '成交量萎缩，市场冷淡',
            'weight': 0.10,
        },
        # 动能因子
        'kdj_position': {
            'name': 'KDJ位置',
            'positive': 'KDJ金叉，动能向上',
            'negative': 'KDJ死叉，动能向下',
            'weight': 0.25,
        },
        'rsi_strength': {
            'name': 'RSI强度',
            'positive': 'RSI处于健康区间，未超买',
            'negative': 'RSI超买或超卖，注意风险',
            'weight': 0.35,
        },
        'mtm_momentum': {
            'name': '动量指标',
            'positive': '动量向上，上涨加速',
            'negative': '动量向下，上涨乏力',
            'weight': 0.20,
        },
        'roc_rate': {
            'name': '变动率',
            'positive': 'ROC为正，价格上升',
            'negative': 'ROC为负，价格下降',
            'weight': 0.20,
        },
        # 资金因子
        'obv_flow': {
            'name': 'OBV流向',
            'positive': 'OBV上升，资金持续流入',
            'negative': 'OBV下降，资金持续流出',
            'weight': 0.40,
        },
        'mfi_strength': {
            'name': 'MFI强度',
            'positive': 'MFI上升，量价配合良好',
            'negative': 'MFI下降，量价背离',
            'weight': 0.35,
        },
        'volume_price': {
            'name': '量价关系',
            'positive': '价涨量增，健康上涨',
            'negative': '价涨量缩，上涨乏力',
            'weight': 0.25,
        },
    }

    # 信号建议模板
    SIGNAL_TEMPLATES = {
        'strong_buy': {
            'condition': lambda s: s >= 80,
            'message': '强烈买入信号：多个因子共振，市场情绪积极',
            'action': '建议积极建仓，分批买入'
        },
        'buy': {
            'condition': lambda s: 70 <= s < 80,
            'message': '买入信号：技术形态良好，具备上涨潜力',
            'action': '建议逢低建仓，控制仓位'
        },
        'hold_buy': {
            'condition': lambda s: 60 <= s < 70,
            'message': '偏多信号：趋势偏强，可适度参与',
            'action': '建议轻仓试探，设好止损'
        },
        'neutral': {
            'condition': lambda s: 40 <= s < 60,
            'message': '中性信号：方向不明，建议观望',
            'action': '等待更明确的买入或卖出信号'
        },
        'hold_sell': {
            'condition': lambda s: 30 <= s < 40,
            'message': '偏空信号：趋势走弱，注意风险',
            'action': '建议减仓，控制风险'
        },
        'sell': {
            'condition': lambda s: s < 30,
            'message': '卖出信号：多个因子走弱，下跌风险较大',
            'action': '建议清仓或大幅减仓'
        },
    }

    def __init__(
        self,
        positive_threshold: float = 60.0,
        negative_threshold: float = 40.0,
        max_factors_to_show: int = 3
    ):
        """
        初始化评分解释器

        Args:
            positive_threshold: 正向因子阈值
            negative_threshold: 负向因子阈值
            max_factors_to_show: 最大显示因子数
        """
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.max_factors_to_show = max_factors_to_show

    def explain_score(
        self,
        score_result: Dict,
        price_data: Optional[pd.DataFrame] = None
    ) -> ScoreExplanation:
        """
        生成评分解释

        Args:
            score_result: 评分结果字典
            price_data: 价格数据（用于计算止损）

        Returns:
            ScoreExplanation: 评分解释
        """
        total_score = score_result.get('final_score', 50)
        factor_scores = score_result.get('factor_scores', {})
        factors_raw = score_result.get('factors_raw', {})

        # 获取评分等级
        grade = self._get_grade(total_score)

        # 提取主要加减分项
        positive_factors = self._extract_positive_factors(factor_scores, factors_raw)
        negative_factors = self._extract_negative_factors(factor_scores, factors_raw)

        # 生成信号建议
        signal_suggestion = self._generate_signal_suggestion(total_score)

        # 生成风险警告
        risk_warning = self._generate_risk_warning(factor_scores, factors_raw)

        # 计算止损价
        stop_loss_price = self._calculate_stop_loss(price_data, total_score)

        # 计算置信度
        confidence = self._calculate_confidence(factor_scores, factors_raw)

        return ScoreExplanation(
            total_score=total_score,
            grade=grade,
            main_positive_factors=positive_factors[:self.max_factors_to_show],
            main_negative_factors=negative_factors[:self.max_factors_to_show],
            signal_suggestion=signal_suggestion,
            risk_warning=risk_warning,
            stop_loss_price=stop_loss_price,
            confidence=confidence
        )

    def generate_trade_rationale(
        self,
        signal: Dict,
        score_explanation: ScoreExplanation
    ) -> str:
        """
        生成交易理由

        Args:
            signal: 信号字典
            score_explanation: 评分解释

        Returns:
            str: 交易理由文本
        """
        direction = signal.get('direction', 'hold')
        score = score_explanation.total_score

        # 构建交易理由
        lines = []
        lines.append("=" * 50)
        lines.append("📊 评分分析报告")
        lines.append("=" * 50)
        lines.append("")

        # 总分
        lines.append(f"总分: {score:.1f}分 ({score_explanation.grade}级)")
        lines.append("")

        # 主要加分项
        if score_explanation.main_positive_factors:
            lines.append("✅ 主要加分项:")
            for factor in score_explanation.main_positive_factors:
                lines.append(f"  • {factor['name']}: {factor['explanation']} ({factor['contribution']:+.1f}分)")
            lines.append("")

        # 主要减分项
        if score_explanation.main_negative_factors:
            lines.append("⚠️ 主要减分项:")
            for factor in score_explanation.main_negative_factors:
                lines.append(f"  • {factor['name']}: {factor['explanation']} ({factor['contribution']:+.1f}分)")
            lines.append("")

        # 信号建议
        lines.append("💡 信号建议:")
        lines.append(f"  {score_explanation.signal_suggestion}")
        lines.append("")

        # 风险警告
        if score_explanation.risk_warning:
            lines.append("⚠️ 风险警告:")
            lines.append(f"  {score_explanation.risk_warning}")
            lines.append("")

        # 止损建议
        if score_explanation.stop_loss_price:
            lines.append(f"🛡️ 建议止损位: {score_explanation.stop_loss_price:.2f}元")

        lines.append("=" * 50)

        return "\n".join(lines)

    def _get_grade(self, score: float) -> str:
        """获取评分等级"""
        for grade, threshold in sorted(
            self.GRADE_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if score >= threshold:
                return grade
        return 'F'

    def _extract_positive_factors(
        self,
        factor_scores: Dict,
        factors_raw: Dict
    ) -> List[Dict]:
        """提取正向因子"""
        positive = []

        for factor_name, score in factor_scores.items():
            if score >= self.positive_threshold:
                explanation = self._get_factor_explanation(
                    factor_name, score, factors_raw
                )
                contribution = self._calculate_contribution(factor_name, score)

                positive.append({
                    'name': self.FACTOR_EXPLANATIONS.get(
                        factor_name, {}
                    ).get('name', factor_name),
                    'score': score,
                    'explanation': explanation,
                    'contribution': contribution
                })

        # 按贡献排序
        return sorted(positive, key=lambda x: x['contribution'], reverse=True)

    def _extract_negative_factors(
        self,
        factor_scores: Dict,
        factors_raw: Dict
    ) -> List[Dict]:
        """提取负向因子"""
        negative = []

        for factor_name, score in factor_scores.items():
            if score <= self.negative_threshold:
                explanation = self._get_factor_explanation(
                    factor_name, score, factors_raw
                )
                contribution = self._calculate_contribution(factor_name, score)

                negative.append({
                    'name': self.FACTOR_EXPLANATIONS.get(
                        factor_name, {}
                    ).get('name', factor_name),
                    'score': score,
                    'explanation': explanation,
                    'contribution': contribution
                })

        # 按贡献排序（最负的在前）
        return sorted(negative, key=lambda x: x['contribution'])

    def _get_factor_explanation(
        self,
        factor_name: str,
        score: float,
        factors_raw: Dict
    ) -> str:
        """获取因子解释"""
        template = self.FACTOR_EXPLANATIONS.get(factor_name, {})

        if score >= 60:
            return template.get('positive', f'{factor_name}表现良好')
        else:
            return template.get('negative', f'{factor_name}表现不佳')

    def _calculate_contribution(self, factor_name: str, score: float) -> float:
        """计算因子贡献"""
        weight = self.FACTOR_EXPLANATIONS.get(factor_name, {}).get('weight', 0.1)
        return (score - 50) * weight

    def _generate_signal_suggestion(self, score: float) -> str:
        """生成信号建议"""
        for signal_type, template in self.SIGNAL_TEMPLATES.items():
            if template['condition'](score):
                return f"{template['message']}。{template['action']}"
        return "建议观望，等待更明确信号"

    def _generate_risk_warning(
        self,
        factor_scores: Dict,
        factors_raw: Dict
    ) -> str:
        """生成风险警告"""
        warnings = []

        # 检查极端指标
        aux_factors = factors_raw.get('aux_factors', {})
        momentum_factors = factors_raw.get('momentum_factors', {})

        # RSI超买
        rsi = momentum_factors.get('rsi', 50)
        if rsi > 75:
            warnings.append(f"RSI严重超买({rsi:.1f})，短期回调风险高")
        elif rsi > 70:
            warnings.append(f"RSI超买({rsi:.1f})，注意回调风险")

        # 乖离率过大
        bias20 = aux_factors.get('bias20', 0)
        if bias20 > 0.08:
            warnings.append(f"BIAS20={bias20*100:.1f}%，乖离过大，回调风险极高")
        elif bias20 > 0.05:
            warnings.append(f"BIAS20={bias20*100:.1f}%，乖离偏大")

        # 布林带位置
        pctb = aux_factors.get('pctb', 0.5)
        if pctb > 0.9:
            warnings.append("股价接近布林上轨，短期超买")

        return "; ".join(warnings) if warnings else ""

    def _calculate_stop_loss(
        self,
        price_data: Optional[pd.DataFrame],
        score: float
    ) -> Optional[float]:
        """计算止损价"""
        if price_data is None or price_data.empty:
            return None

        close = price_data['close'].iloc[-1]

        # 根据评分调整止损幅度
        if score >= 70:
            stop_pct = 0.05  # 高评分，5%止损
        elif score >= 50:
            stop_pct = 0.07  # 中等评分，7%止损
        else:
            stop_pct = 0.03  # 低评分，3%止损

        return close * (1 - stop_pct)

    def _calculate_confidence(
        self,
        factor_scores: Dict,
        factors_raw: Dict
    ) -> float:
        """计算置信度"""
        if not factor_scores:
            return 0.5

        # 因子一致性
        scores = list(factor_scores.values())
        if not scores:
            return 0.5

        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # 标准差越小，置信度越高
        consistency = max(0, 1 - std_score / 30)

        # 评分偏离50度越大，置信度越高
        deviation = abs(mean_score - 50) / 50

        confidence = 0.5 + consistency * 0.3 + deviation * 0.2

        return min(1.0, confidence)


def explain_score(score_result: Dict, price_data: Optional[pd.DataFrame] = None) -> str:
    """
    便捷函数：生成评分解释

    Args:
        score_result: 评分结果
        price_data: 价格数据

    Returns:
        str: 解释文本
    """
    explainer = ScoreExplainer()
    explanation = explainer.explain_score(score_result, price_data)

    signal = {'direction': 'buy' if explanation.total_score >= 60 else 'hold'}
    return explainer.generate_trade_rationale(signal, explanation)