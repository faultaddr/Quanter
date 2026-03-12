#!/usr/bin/env python
"""
低位超卖保护机制单元测试

测试三系统评分中的低位超卖保护逻辑：
1. 强保护：60日分位≤10% + RSI≤30
2. 标准保护：60日分位≤20% + RSI≤30 或 60日分位≤20% + WR≥80/布林带位置≤15%
3. 弱保护：60日分位≤30% + RSI≤35
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict

from quanttool.factors.scoring_system import ScoringSystem
from quanttool.factors.recommendation_engine import RecommendationEngine, PROTECTION_POSITION_SIZES
from quanttool.factors.analysis_context import (
    AnalysisContext,
    ActionType,
    ClassicScore,
    TrendScore,
    BreakoutScore,
    PositionAssessment,
    UnifiedMarketState,
    MarketState,
    StopLossConfig,
    ScoringSystemType,
)


class TestLowOversoldProtection:
    """低位超卖保护测试类"""

    @pytest.fixture
    def scoring_system(self):
        """创建评分系统实例"""
        return ScoringSystem()

    @pytest.fixture
    def recommendation_engine(self):
        """创建推荐引擎实例"""
        return RecommendationEngine()

    def _create_test_factors(
        self,
        position_ratio: float,
        rsi: float,
        wr: float = 50,
        pctb: float = 0.5,
        is_downtrend: bool = True
    ) -> Dict:
        """
        创建测试用的因子字典

        Args:
            position_ratio: 60日分位 (0-1)
            rsi: RSI值
            wr: WR值 (WR越大越超卖)
            pctb: 布林带位置 (0-1)
            is_downtrend: 是否下跌趋势
        """
        # 设置均线以模拟下跌趋势
        if is_downtrend:
            # 空头排列：MA5 < MA20 < MA50
            ma5 = 10.0
            ma20 = 11.0
            ma50 = 12.0
            close = 9.5  # 股价在均线下方
        else:
            # 多头排列
            ma5 = 12.0
            ma20 = 11.0
            ma50 = 10.0
            close = 12.5  # 股价在均线上方

        return {
            'aux_factors': {
                'position_ratio': position_ratio,
                'pctb': pctb,
                'wr': wr,
                'bias6': -0.02,
                'bias20': -0.03,
                'cci': -50,
                'close': close,
                'ma5': ma5,
                'ma10': ma5,
                'ma20': ma20,
                'ma50': ma50,
            },
            'momentum_factors': {
                'rsi': rsi,
            },
            'trend_factors': {},
        }

    def test_strong_protection(self, scoring_system):
        """测试强保护：60日分位≤10% + RSI≤30"""
        factors = self._create_test_factors(
            position_ratio=0.08,  # 8%分位，极度低位
            rsi=25,               # RSI=25，极端超卖
            wr=85,
            pctb=0.10,
            is_downtrend=True
        )

        # 调用保护检测方法
        is_protected, protection_level = scoring_system._check_low_oversold_protection(
            factors['aux_factors'],
            factors['momentum_factors']
        )

        assert is_protected == True, "应该触发保护"
        assert protection_level == "strong", "应该是强保护"

    def test_standard_protection_rsi(self, scoring_system):
        """测试标准保护：60日分位≤20% + RSI≤30"""
        factors = self._create_test_factors(
            position_ratio=0.18,  # 18%分位
            rsi=28,               # RSI=28
            wr=75,
            pctb=0.20,
            is_downtrend=True
        )

        is_protected, protection_level = scoring_system._check_low_oversold_protection(
            factors['aux_factors'],
            factors['momentum_factors']
        )

        assert is_protected == True, "应该触发保护"
        assert protection_level == "standard", "应该是标准保护"

    def test_standard_protection_wr(self, scoring_system):
        """测试标准保护：60日分位≤20% + WR≥80"""
        factors = self._create_test_factors(
            position_ratio=0.15,  # 15%分位
            rsi=35,               # RSI不是极端超卖
            wr=84,                # WR=84，超卖
            pctb=0.20,
            is_downtrend=True
        )

        is_protected, protection_level = scoring_system._check_low_oversold_protection(
            factors['aux_factors'],
            factors['momentum_factors']
        )

        assert is_protected == True, "应该触发保护（WR条件）"
        assert protection_level == "standard", "应该是标准保护"

    def test_standard_protection_pctb(self, scoring_system):
        """测试标准保护：60日分位≤20% + 布林带位置≤15%"""
        factors = self._create_test_factors(
            position_ratio=0.18,
            rsi=40,               # RSI不超卖
            wr=60,                # WR不极端
            pctb=0.12,            # 布林带位置12%
            is_downtrend=True
        )

        is_protected, protection_level = scoring_system._check_low_oversold_protection(
            factors['aux_factors'],
            factors['momentum_factors']
        )

        assert is_protected == True, "应该触发保护（布林带条件）"
        assert protection_level == "standard", "应该是标准保护"

    def test_weak_protection(self, scoring_system):
        """测试弱保护：60日分位≤30% + RSI≤35"""
        factors = self._create_test_factors(
            position_ratio=0.25,  # 25%分位
            rsi=32,               # RSI=32
            wr=70,
            pctb=0.25,
            is_downtrend=True
        )

        is_protected, protection_level = scoring_system._check_low_oversold_protection(
            factors['aux_factors'],
            factors['momentum_factors']
        )

        assert is_protected == True, "应该触发保护"
        assert protection_level == "weak", "应该是弱保护"

    def test_no_protection_high_position(self, scoring_system):
        """测试高位不触发保护"""
        factors = self._create_test_factors(
            position_ratio=0.50,  # 50%分位，非低位
            rsi=28,               # RSI超卖
            wr=85,
            pctb=0.10,
            is_downtrend=True
        )

        is_protected, protection_level = scoring_system._check_low_oversold_protection(
            factors['aux_factors'],
            factors['momentum_factors']
        )

        assert is_protected == False, "高位不应触发保护"
        assert protection_level == "", "保护级别应为空"

    def test_no_protection_not_oversold(self, scoring_system):
        """测试非超卖不触发保护"""
        factors = self._create_test_factors(
            position_ratio=0.15,  # 低位
            rsi=50,               # RSI不超卖
            wr=50,                # WR不极端
            pctb=0.40,            # 布林带位置正常
            is_downtrend=True
        )

        is_protected, protection_level = scoring_system._check_low_oversold_protection(
            factors['aux_factors'],
            factors['momentum_factors']
        )

        assert is_protected == False, "非超卖不应触发保护"
        assert protection_level == "", "保护级别应为空"

    def test_position_modifier_with_protection(self, scoring_system):
        """测试保护机制对位置修正系数的影响"""
        # 创建测试 DataFrame
        df = pd.DataFrame({
            'close': [10.0] * 5,
            'open': [10.0] * 5,
            'high': [10.5] * 5,
            'low': [9.5] * 5,
            'volume': [100000] * 5,
        })
        latest = pd.Series({'close': 10.0})

        # 有保护的因子（强保护）
        factors_with_protection = self._create_test_factors(
            position_ratio=0.08,
            rsi=25,
            wr=85,
            pctb=0.10,
            is_downtrend=True
        )

        modifier_protected, warnings_protected = scoring_system._calculate_position_modifier(
            latest, factors_with_protection
        )

        # 无保护的因子
        factors_no_protection = self._create_test_factors(
            position_ratio=0.50,  # 高位
            rsi=25,
            wr=85,
            pctb=0.10,
            is_downtrend=True
        )

        modifier_no_protection, warnings_no_protection = scoring_system._calculate_position_modifier(
            latest, factors_no_protection
        )

        # 有保护的位置修正系数应该更高
        assert modifier_protected > modifier_no_protection, \
            f"有保护的修正系数({modifier_protected})应该高于无保护的({modifier_no_protection})"

        # 检查警告信息
        assert any("强保护" in w for w in warnings_protected), "应该包含强保护警告"

    def test_recommendation_engine_protection_override(self, recommendation_engine):
        """测试推荐引擎中保护机制覆盖回避信号"""
        # 创建分析上下文
        context = AnalysisContext(
            symbol="600460",
            current_price=10.0,
            analysis_date=datetime.now()
        )

        # 设置经典评分（包含保护警告）
        context.classic_score = ClassicScore(
            score=35.0,  # 低评分
            trend_score=30.0,  # 趋势弱
            position_modifier=0.8,
            score_grade="较差",
            warnings=["🟢【强保护】极度低位+极端超卖(60日分位≤10%+RSI≤30)，反弹概率较高"]
        )

        # 设置位置评估
        context.position_assessment = PositionAssessment(
            position="low",
            long_term_position="low",
            short_term_position="low",
            is_oversold=True,
            position_modifier=0.8
        )

        # 设置市场状态
        context.market_state = UnifiedMarketState(
            combined_regime=MarketState.BEAR
        )

        # 设置趋势评分
        context.trend_score = TrendScore(
            final_score=0.0,
            passed_hard_filter=False
        )

        # 设置突破评分
        context.breakout_score = BreakoutScore(
            final_score=0.0,
            passed_filter=False
        )

        # 设置止损配置
        context.stop_loss_config = StopLossConfig()

        # 生成推荐
        recommendation = recommendation_engine.generate_recommendation(context)

        # 验证：由于强保护，回避信号应该被覆盖为轻仓试探
        assert recommendation.action == ActionType.LIGHT_POSITION, \
            f"强保护应该将回避转为轻仓试探，实际是{recommendation.action}"
        assert recommendation.position_size == PROTECTION_POSITION_SIZES["strong"], \
            f"仓位应该是{PROTECTION_POSITION_SIZES['strong']}"

    def test_recommendation_engine_standard_protection(self, recommendation_engine):
        """测试推荐引擎中标准保护"""
        context = AnalysisContext(
            symbol="600460",
            current_price=10.0,
            analysis_date=datetime.now()
        )

        context.classic_score = ClassicScore(
            score=35.0,
            trend_score=30.0,
            position_modifier=0.7,
            score_grade="较差",
            warnings=["📊【标准保护】低位超卖(60日分位≤20%+RSI≤30或WR≥80)，关注反转信号"]
        )

        context.position_assessment = PositionAssessment(
            position="low",
            long_term_position="low",
            short_term_position="low",
            is_oversold=True,
            position_modifier=0.7
        )

        context.market_state = UnifiedMarketState(
            combined_regime=MarketState.BEAR
        )

        context.trend_score = TrendScore(
            final_score=0.0,
            passed_hard_filter=False
        )

        context.breakout_score = BreakoutScore(
            final_score=0.0,
            passed_filter=False
        )

        context.stop_loss_config = StopLossConfig()

        recommendation = recommendation_engine.generate_recommendation(context)

        assert recommendation.action == ActionType.LIGHT_POSITION, \
            f"标准保护应该将回避转为轻仓试探，实际是{recommendation.action}"
        assert recommendation.position_size == PROTECTION_POSITION_SIZES["standard"]

    def test_600460_case(self, scoring_system, recommendation_engine):
        """
        测试 600460 个案

        当前数据:
        - 位置评估: 长短期双低位
        - WR: 84.1 (超卖)
        - KDJ-J: -6.83 (极端超卖)
        - MACD: -0.89 (空头)

        预期修复后结果:
        - 保护级别: 标准/强保护
        - 建议: 轻仓观察 (20-30%仓位)
        """
        # 模拟 600460 的数据
        factors = self._create_test_factors(
            position_ratio=0.15,  # 假设低位
            rsi=28,               # RSI超卖
            wr=84.1,              # WR超卖
            pctb=0.12,            # 布林带下轨附近
            is_downtrend=True
        )

        # 测试保护检测
        is_protected, protection_level = scoring_system._check_low_oversold_protection(
            factors['aux_factors'],
            factors['momentum_factors']
        )

        assert is_protected == True, "600460 应该触发保护"
        assert protection_level in ["standard", "strong"], \
            f"保护级别应该是standard或strong，实际是{protection_level}"

        # 测试位置修正系数
        latest = pd.Series({'close': 9.5})
        modifier, warnings = scoring_system._calculate_position_modifier(latest, factors)

        # 验证：有保护时，modifier 不应该被过度惩罚
        assert modifier >= 0.7, f"有保护时modifier应该>=0.7，实际是{modifier}"

        # 验证警告包含保护信息
        protection_warnings = [w for w in warnings if "保护" in w]
        assert len(protection_warnings) > 0, "应该包含保护警告"


class TestProtectionThresholds:
    """保护阈值边界测试"""

    @pytest.fixture
    def scoring_system(self):
        return ScoringSystem()

    def test_strong_protection_boundary(self, scoring_system):
        """测试强保护边界条件"""
        # 刚好满足强保护
        factors = {
            'aux_factors': {'position_ratio': 0.10, 'wr': 50, 'pctb': 0.5},
            'momentum_factors': {'rsi': 30}
        }
        is_protected, level = scoring_system._check_low_oversold_protection(
            factors['aux_factors'], factors['momentum_factors']
        )
        assert is_protected == True and level == "strong"

        # 刚好不满足强保护（position_ratio > 10%）
        factors = {
            'aux_factors': {'position_ratio': 0.11, 'wr': 50, 'pctb': 0.5},
            'momentum_factors': {'rsi': 30}
        }
        is_protected, level = scoring_system._check_low_oversold_protection(
            factors['aux_factors'], factors['momentum_factors']
        )
        # 应该触发标准保护
        assert is_protected == True and level == "standard"

    def test_standard_protection_boundary(self, scoring_system):
        """测试标准保护边界条件"""
        # 刚好满足标准保护
        factors = {
            'aux_factors': {'position_ratio': 0.20, 'wr': 50, 'pctb': 0.5},
            'momentum_factors': {'rsi': 30}
        }
        is_protected, level = scoring_system._check_low_oversold_protection(
            factors['aux_factors'], factors['momentum_factors']
        )
        assert is_protected == True and level == "standard"

        # 刚好不满足标准保护（position_ratio > 20%）
        factors = {
            'aux_factors': {'position_ratio': 0.21, 'wr': 50, 'pctb': 0.5},
            'momentum_factors': {'rsi': 30}
        }
        is_protected, level = scoring_system._check_low_oversold_protection(
            factors['aux_factors'], factors['momentum_factors']
        )
        # 应该触发弱保护
        assert is_protected == True and level == "weak"

    def test_weak_protection_boundary(self, scoring_system):
        """测试弱保护边界条件"""
        # 刚好满足弱保护
        factors = {
            'aux_factors': {'position_ratio': 0.30, 'wr': 50, 'pctb': 0.5},
            'momentum_factors': {'rsi': 35}
        }
        is_protected, level = scoring_system._check_low_oversold_protection(
            factors['aux_factors'], factors['momentum_factors']
        )
        assert is_protected == True and level == "weak"

        # 刚好不满足弱保护（position_ratio > 30%）
        factors = {
            'aux_factors': {'position_ratio': 0.31, 'wr': 50, 'pctb': 0.5},
            'momentum_factors': {'rsi': 35}
        }
        is_protected, level = scoring_system._check_low_oversold_protection(
            factors['aux_factors'], factors['momentum_factors']
        )
        assert is_protected == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])