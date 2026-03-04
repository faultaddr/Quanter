"""
股票多维度打分系统（百分制版本 - 增强版）

基于 MyTT 指标库设计更精准的多因子组合

核心改进：
1. 分层评分架构：趋势分 × 位置修正系数
2. 三大类因子组：
   - 趋势因子（确认方向）：均线系统、DMI、MACD
   - 动能因子（确认强度）：MTM、ROC、KDJ、RSI
   - 资金因子（确认真实性）：OBV、MFI、成交量
3. 多维度交叉验证，减少假信号
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from datetime import datetime

from .candlestick_patterns import analyze_candlestick_patterns


# ==================== MyTT 核心函数移植 ====================
# 基于 https://github.com/mpquant/MyTT

def REF(S, N=1):
    """对序列整体下移动N"""
    return pd.Series(S).shift(N).values

def MA(S, N):
    """简单移动平均"""
    return pd.Series(S).rolling(N).mean().values

def EMA(S, N):
    """指数移动平均"""
    return pd.Series(S).ewm(span=N, adjust=False).mean().values

def HHV(S, N):
    """N日内最高值"""
    return pd.Series(S).rolling(N).max().values

def LLV(S, N):
    """N日内最低值"""
    return pd.Series(S).rolling(N).min().values

def SUM(S, N):
    """N日累加"""
    return pd.Series(S).rolling(N).sum().values if N > 0 else pd.Series(S).cumsum().values

def STD(S, N):
    """N日标准差"""
    return pd.Series(S).rolling(N).std(ddof=0).values

def MAX(S1, S2):
    """序列最大值"""
    return np.maximum(S1, S2)

def MIN(S1, S2):
    """序列最小值"""
    return np.minimum(S1, S2)

def ABS(S):
    """绝对值"""
    return np.abs(S)

def IF(S, A, B):
    """条件判断"""
    return np.where(S, A, B)

def CROSS(S1, S2):
    """金叉判断"""
    return np.concatenate(([False], np.logical_not((S1 > S2)[:-1]) & (S1 > S2)[1:]))

def COUNT(S, N):
    """N日内满足条件的天数"""
    return SUM(S, N)

def AVEDEV(S, N):
    """平均绝对偏差"""
    return pd.Series(S).rolling(N).apply(lambda x: (np.abs(x - x.mean())).mean()).values

# ==================== MyTT 指标函数 ====================

def MACD(CLOSE, SHORT=12, LONG=26, M=9):
    """MACD指标"""
    DIF = EMA(CLOSE, SHORT) - EMA(CLOSE, LONG)
    DEA = EMA(DIF, M)
    MACD_VAL = (DIF - DEA) * 2
    return DIF, DEA, MACD_VAL

def KDJ(CLOSE, HIGH, LOW, N=9, M1=3, M2=3):
    """KDJ指标"""
    RSV = (CLOSE - LLV(LOW, N)) / (HHV(HIGH, N) - LLV(LOW, N) + 1e-10) * 100
    K = EMA(RSV, (M1 * 2 - 1))
    D = EMA(K, (M2 * 2 - 1))
    J = K * 3 - D * 2
    return K, D, J

def RSI(CLOSE, N=24):
    """RSI指标"""
    DIF = CLOSE - REF(CLOSE, 1)
    return np.round(SMA(MAX(DIF, 0), N) / SMA(ABS(DIF), N) * 100, 2)

def SMA(S, N, M=1):
    """中国式SMA"""
    return pd.Series(S).ewm(alpha=M/N, adjust=False).mean().values

def WR(CLOSE, HIGH, LOW, N=10):
    """威廉指标"""
    return (HHV(HIGH, N) - CLOSE) / (HHV(HIGH, N) - LLV(LOW, N) + 1e-10) * 100

def CCI(CLOSE, HIGH, LOW, N=14):
    """CCI指标"""
    TP = (HIGH + LOW + CLOSE) / 3
    return (TP - MA(TP, N)) / (0.015 * AVEDEV(TP, N) + 1e-10)

def DMI(CLOSE, HIGH, LOW, M1=14, M2=6):
    """DMI指标"""
    TR = SUM(MAX(MAX(HIGH - LOW, ABS(HIGH - REF(CLOSE, 1))), ABS(LOW - REF(CLOSE, 1))), M1)
    HD = HIGH - REF(HIGH, 1)
    LD = REF(LOW, 1) - LOW
    DMP = SUM(IF((HD > 0) & (HD > LD), HD, 0), M1)
    DMM = SUM(IF((LD > 0) & (LD > HD), LD, 0), M1)
    PDI = DMP * 100 / (TR + 1e-10)
    MDI = DMM * 100 / (TR + 1e-10)
    ADX = MA(ABS(MDI - PDI) / (PDI + MDI + 1e-10) * 100, M2)
    return PDI, MDI, ADX

def MTM(CLOSE, N=12):
    """动量指标"""
    return CLOSE - REF(CLOSE, N)

def ROC(CLOSE, N=12):
    """变动率指标"""
    return 100 * (CLOSE - REF(CLOSE, N)) / (REF(CLOSE, N) + 1e-10)

def OBV(CLOSE, VOL):
    """能量潮指标"""
    return SUM(IF(CLOSE > REF(CLOSE, 1), VOL, IF(CLOSE < REF(CLOSE, 1), -VOL, 0)), 0) / 10000

def MFI(CLOSE, HIGH, LOW, VOL, N=14):
    """MFI指标（成交量的RSI）"""
    TYP = (HIGH + LOW + CLOSE) / 3
    V1 = SUM(IF(TYP > REF(TYP, 1), TYP * VOL, 0), N) / (SUM(IF(TYP < REF(TYP, 1), TYP * VOL, 0), N) + 1e-10)
    return 100 - (100 / (1 + V1))

def BOLL(CLOSE, N=20, P=2):
    """布林带"""
    MID = MA(CLOSE, N)
    UPPER = MID + STD(CLOSE, N) * P
    LOWER = MID - STD(CLOSE, N) * P
    return UPPER, MID, LOWER

def PSY(CLOSE, N=12):
    """心理线指标"""
    return COUNT(CLOSE > REF(CLOSE, 1), N) / N * 100

def ATR(CLOSE, HIGH, LOW, N=20):
    """真实波动幅度"""
    TR = MAX(MAX(HIGH - LOW, ABS(REF(CLOSE, 1) - HIGH)), ABS(REF(CLOSE, 1) - LOW))
    return MA(TR, N)


class ScoringSystem:
    """
    股票多维度打分系统（百分制 - 增强版）

    架构：最终评分 = 趋势得分 × 位置修正系数

    三大类因子组：
    1. 趋势因子（权重40%）：确认方向
       - 均线系统（MA5/MA10/MA20排列）
       - DMI趋势强度（PDI/MDI/ADX）
       - MACD趋势方向（DIF/DEA方向）

    2. 动能因子（权重35%）：确认强度
       - KDJ超买超卖（K/D/J位置）
       - RSI强度（RSI位置和斜率）
       - MTM动量（价格动量）
       - ROC变动率（价格变化速度）

    3. 资金因子（权重25%）：确认真实性
       - OBV资金流（量价配合）
       - MFI资金流量（成交量RSI）
       - 量价关系（价涨量增/价跌量缩）

    位置修正系数（风险控制）：
    - 安全区: 1.0
    - 适中区: 0.85
    - 警戒区: 0.6
    - 危险区: 0.35
    """

    # 三大类因子权重配置（优化版：降低趋势追高权重，增加均值回归考量）
    FACTOR_GROUP_WEIGHTS = {
        'trend': 0.35,    # 趋势因子权重（降低）
        'momentum': 0.40, # 动能因子权重（提高）
        'money': 0.25,    # 资金因子权重
    }

    # 趋势因子权重（组内）- 优化版：降低追高风险
    # 注意：K线形态已移至独立筛选层，不再参与评分计算
    # 分析发现 ma_slope 与收益负相关，需要降低权重
    TREND_FACTOR_WEIGHTS = {
        'trend_strength': 0.30,      # 趋势强度（MA20乖离率）- 提高
        'ma_slope': 0.10,            # 均线斜率 - 降低（负相关因子）
        'macd_momentum': 0.30,       # MACD动量 - 提高（正相关因子）
        'money_flow': 0.20,          # 资金流向
        'volume_ratio': 0.10,        # 成交量比率
    }

    # 动能因子权重（组内）- 增加 RSI 均值回归
    MOMENTUM_FACTOR_WEIGHTS = {
        'kdj_position': 0.25,   # KDJ位置
        'rsi_strength': 0.35,   # RSI强度（关注超卖反弹）
        'mtm_momentum': 0.20,   # MTM动量
        'roc_rate': 0.20,       # ROC变动率（提高）
    }

    # 资金因子权重（组内）
    MONEY_FACTOR_WEIGHTS = {
        'obv_flow': 0.40,       # OBV资金流
        'mfi_strength': 0.35,   # MFI强度
        'volume_price': 0.25,   # 量价关系
    }

    # 乖离率阈值配置（8%档位）
    BIAS_THRESHOLDS = {
        'hard_filter_max': 0.08,    # 硬过滤上限 +8%
        'score_threshold': 0.08,    # 评分阈值 8%
    }

    # 流动性过滤阈值
    LIQUIDITY_THRESHOLDS = {
        'min_amt_ma20': 100000,  # 千元单位，即1亿元
        'min_atrp': 0.015,       # ATR% >= 1.5%
    }

    # 动态止损阈值配置
    STOP_LOSS_CONFIG = {
        'atr_low_threshold': 0.02,      # ATR% < 2% 视为低波动
        'boll_bandwidth_threshold': 0.05,  # 布林带带宽 < 5% 视为窄幅
        'tight_stop_loss_pct': 0.02,    # 紧止损 2%
        'normal_stop_loss_pct': 0.05,   # 正常止损 5%
        'ma50_stop_loss': True,         # 允许使用MA50作为止损参考
    }

    # K线形态基础权重
    CANDLESTICK_PATTERN_WEIGHTS = {
        'strong_bullish': {
            'patterns': ['晨星', '看涨吞没', '白色三兵'],
            'base_weight': 15.0,
        },
        'medium_bullish': {
            'patterns': ['锤子线', '倒锤子', '穿刺线', '大阳线'],
            'base_weight': 10.0,
        },
        'strong_bearish': {
            'patterns': ['暮星', '看跌吞没', '黑色三鸦'],
            'base_weight': -15.0,
        },
        'medium_bearish': {
            'patterns': ['流星线', '吊颈线', '乌云盖顶', '大阴线'],
            'base_weight': -10.0,
        },
    }

    # 位置修正系数（位置决定形态意义）
    POSITION_PATTERN_MODIFIERS = {
        'low_position': {'bullish': 1.5, 'bearish': 0.3},   # 低位: 看涨加强，看跌减弱
        'high_position': {'bullish': -0.5, 'bearish': 1.5}, # 高位: 看涨警惕诱多，看跌加强
        'mid_position': {'bullish': 1.0, 'bearish': 1.0},   # 中位: 正常权重
    }

    def __init__(self, stop_loss_pct: float = 0.05, use_dynamic_weights: bool = False):
        """
        初始化评分系统

        Args:
            stop_loss_pct: 止损比例，默认5%（买入价×0.95）
            use_dynamic_weights: 是否使用动态权重
        """
        self.stop_loss_pct = stop_loss_pct
        self.use_dynamic_weights = use_dynamic_weights
        self.score_breakdown = {}

        # 动态权重相关
        self._dynamic_weights = None
        self._market_regime = None

        # 验证钩子
        self._validation_callback = None

    def set_dynamic_weights(self, weights: Dict[str, float]) -> None:
        """
        设置动态因子组权重

        Args:
            weights: 包含 'trend', 'momentum', 'money' 的权重字典
        """
        if weights:
            # 验证权重
            total = weights.get('trend', 0) + weights.get('momentum', 0) + weights.get('money', 0)
            if abs(total - 1.0) > 0.01:
                # 归一化权重
                if total > 0:
                    weights = {
                        'trend': weights.get('trend', 0) / total,
                        'momentum': weights.get('momentum', 0) / total,
                        'money': weights.get('money', 0) / total
                    }
            self._dynamic_weights = weights
            self.use_dynamic_weights = True

    def set_market_regime(self, regime: str) -> None:
        """
        设置市场状态

        Args:
            regime: 市场状态 ('bull', 'bear', 'sideway', 'volatile')
        """
        self._market_regime = regime

        # 根据市场状态设置默认权重
        regime_weights = {
            'bull': {'trend': 0.50, 'momentum': 0.30, 'money': 0.20},
            'bear': {'trend': 0.30, 'momentum': 0.25, 'money': 0.45},
            'sideway': {'trend': 0.25, 'momentum': 0.45, 'money': 0.30},
            'volatile': {'trend': 0.35, 'momentum': 0.30, 'money': 0.35},
        }
        if regime in regime_weights:
            self.set_dynamic_weights(regime_weights[regime])

    def set_validation_callback(self, callback) -> None:
        """
        设置验证回调函数

        Args:
            callback: 回调函数，接收评分结果作为参数
        """
        self._validation_callback = callback

    def get_current_weights(self) -> Dict[str, float]:
        """获取当前使用的权重"""
        if self.use_dynamic_weights and self._dynamic_weights:
            return self._dynamic_weights.copy()
        return self.FACTOR_GROUP_WEIGHTS.copy()

    def calculate_comprehensive_score(self, df: pd.DataFrame) -> Dict:
        """
        计算综合评分（简化版，用于回测）

        Args:
            df: 股票数据DataFrame

        Returns:
            Dict: 包含 final_score 和各因子评分
        """
        if df.empty or len(df) < 30:
            return {'final_score': 50, 'trend_score': 50, 'momentum_score': 50, 'money_score': 50}

        latest = df.iloc[-1]

        # 计算因子
        factors = self._calculate_trend_factors(df, latest)
        trend_score, factor_scores = self._calculate_trend_score(factors, df)

        # 计算位置修正系数
        position_modifier, _ = self._calculate_position_modifier(latest, factors)

        # 最终评分
        final_score = trend_score * position_modifier

        # 获取各因子组评分
        momentum_factors = factors.get('momentum_factors', {})
        money_factors = factors.get('money_factors', {})
        aux_factors = factors.get('aux_factors', {})

        momentum_score = sum(
            momentum_factors.get(k, 50) * self.MOMENTUM_FACTOR_WEIGHTS.get(k, 0.25)
            for k in self.MOMENTUM_FACTOR_WEIGHTS
        )
        money_score = sum(
            money_factors.get(k, 50) * self.MONEY_FACTOR_WEIGHTS.get(k, 0.33)
            for k in self.MONEY_FACTOR_WEIGHTS
        )

        # ========== 均值回归信号增强 ==========
        # 分析发现：超卖信号（低BIAS + 低RSI）有显著正收益
        mean_reversion_bonus = 0.0
        bias20 = aux_factors.get('bias20', 0)
        rsi = momentum_factors.get('rsi', 50)

        # 超卖反转信号：BIAS20 < -3% 且 RSI < 40
        if bias20 < -0.03 and rsi < 40:
            mean_reversion_bonus = 15.0  # 强超卖反转信号
        elif bias20 < -0.02 and rsi < 50:
            mean_reversion_bonus = 8.0   # 中等超卖信号
        elif bias20 < -0.01:
            mean_reversion_bonus = 3.0   # 轻微超卖
        # 超买风险警示：BIAS20 > +3% 且 RSI > 60
        elif bias20 > 0.03 and rsi > 60:
            mean_reversion_bonus = -10.0  # 强超买，有回调风险
        elif bias20 > 0.02 and rsi > 55:
            mean_reversion_bonus = -5.0   # 中等超买

        # 应用均值回归修正
        final_score = final_score + mean_reversion_bonus

        result = {
            'final_score': round(max(0, min(100, final_score)), 2),
            'trend_score': round(trend_score, 2),
            'momentum_score': round(momentum_score, 2),
            'money_score': round(money_score, 2),
            'position_modifier': round(position_modifier, 2),
            'mean_reversion_bonus': mean_reversion_bonus,
            'factor_scores': factor_scores,
            'factors_raw': factors
        }

        # 验证钩子
        if self._validation_callback:
            try:
                self._validation_callback(result)
            except Exception:
                pass

        return result

    def calculate_all_scores(self, df: pd.DataFrame,
                            stock_code: str = "",
                            trade_date_T: Optional[str] = None,
                            trade_date_T1: Optional[str] = None,
                            open_T1: Optional[float] = None) -> Dict:
        """
        计算股票的综合评分（百分制 - 重构版）

        架构：最终评分 = 趋势得分 × 位置修正系数

        Args:
            df: 股票数据DataFrame（需包含至少40日数据）
            stock_code: 股票代码
            trade_date_T: T日（信号计算日）日期字符串
            trade_date_T1: T+1日（计划买入日）日期字符串
            open_T1: T+1日开盘价（用于计算实际买入价和止损）

        Returns:
            Dict: 包含评分、因子值、交易计划的字典
        """
        if df.empty or len(df) < 40:
            return {"error": "数据不足，至少需要40个交易日数据"}

        latest = df.iloc[-1]

        # 1. 乖离率硬过滤检查（8%档位）
        bias_passed, bias_reason = self._check_bias_filter(latest)
        if not bias_passed:
            return {
                "error": f"乖离率过滤未通过: {bias_reason}",
                "bias_passed": False,
                "stock_code": stock_code,
            }

        # 2. 计算趋势因子得分
        trend_factors = self._calculate_trend_factors(df, latest)
        trend_score, factor_scores = self._calculate_trend_score(trend_factors, df)

        # 3. 计算位置修正系数
        position_modifier, position_warnings = self._calculate_position_modifier(latest, trend_factors)

        # 4. 最终评分 = 趋势得分 × 位置修正系数
        final_score = trend_score * position_modifier

        # 5. 检测双触发信号
        trigger_type, trigger_detail = self._detect_triggers(df, latest)

        # 5.5 将K线形态详情添加到trend_factors中（用于熔断机制）
        if 'candlestick_detail' in trend_factors:
            trend_factors['candlestick_detail'] = trend_factors['candlestick_detail']

        # 6. 计算交易执行信息（传入完整的trend_factors用于熔断判断）
        execution = self._calculate_execution_info(
            latest, open_T1, final_score, trigger_type, trend_factors
        )

        # 7. 关键修复：熔断机制前置修正评分
        # 如果熔断触发，大幅降低最终评分以反映真实风险
        action_guide = execution.get('action_guide', '')
        if '熔断' in action_guide:
            # 熔断触发：评分强制降至30分以下
            final_score = min(final_score, 25)
        elif '风险警告' in action_guide:
            # 风险警告：评分打7折
            final_score = final_score * 0.7

        # 8. 收集警告信息
        warnings = self._collect_warnings(trend_factors, latest)
        warnings.extend(position_warnings)

        return {
            "stock_code": stock_code,
            "trade_date_T": trade_date_T or latest.get('trade_date', ''),
            "trade_date_T1": trade_date_T1,
            "close_T": float(latest.get('close', 0)),
            "score": round(final_score, 2),
            "score_grade": self._get_score_grade(final_score),
            "trend_score": round(trend_score, 2),  # 新增：趋势得分
            "position_modifier": round(position_modifier, 2),  # 新增：位置修正系数
            "trigger_type": trigger_type,
            "trigger_detail": trigger_detail,
            "factors_raw": trend_factors,      # 原始因子值
            "factors_score": factor_scores,    # 各因子得分(0-100)
            "liquidity_passed": True,
            "execution": execution,
            "warnings": warnings,
        }

    def calculate_portfolio_scores(self, stock_data_dict: Dict[str, pd.DataFrame],
                                   trade_date_T: str,
                                   trade_date_T1: str,
                                   open_prices_T1: Dict[str, float]) -> pd.DataFrame:
        """
        对多只股票进行评分和排名（推荐用法）

        Args:
            stock_data_dict: {stock_code: df} 字典
            trade_date_T: T日日期
            trade_date_T1: T+1日日期
            open_prices_T1: {stock_code: open_price} T+1日开盘价字典

        Returns:
            pd.DataFrame: 评分排名表
        """
        results = []

        for stock_code, df in stock_data_dict.items():
            open_T1 = open_prices_T1.get(stock_code)

            result = self.calculate_all_scores(
                df=df,
                stock_code=stock_code,
                trade_date_T=trade_date_T,
                trade_date_T1=trade_date_T1,
                open_T1=open_T1
            )

            if 'error' not in result:
                results.append(result)

        if not results:
            return pd.DataFrame()

        # 在股票池中进行因子归一化（rank_pct）
        results = self._normalize_factors_in_pool(results)

        # 重新计算得分
        for r in results:
            score, _ = self._calculate_weighted_score(r['factors_score'])
            r['score'] = round(score * 100, 2)
            r['score_grade'] = self._get_score_grade(r['score'])
            # 更新仓位建议
            r['execution']['position_suggest'] = self._get_position_suggestion(
                r['score'], r['trigger_type']
            )

        # 创建DataFrame并排序
        df_result = pd.DataFrame(results)

        # 优先：突破型 > 回踩型 > 普通高分，其次按分数排序
        trigger_order = {'breakout': 0, 'pullback': 1, 'none': 2}
        df_result['trigger_order'] = df_result['trigger_type'].map(trigger_order)
        df_result = df_result.sort_values(['trigger_order', 'score'], ascending=[True, False])

        # 选择输出列
        output_cols = [
            'stock_code', 'score', 'score_grade', 'trigger_type',
            'close_T', 'buy_price', 'stop_price', 'position_suggest',
            'trade_date_T', 'trade_date_T1'
        ]

        # 从execution中提取交易信息
        df_result['buy_price'] = df_result['execution'].apply(lambda x: x.get('buy_price'))
        df_result['stop_price'] = df_result['execution'].apply(lambda x: x.get('stop_price'))
        df_result['position_suggest'] = df_result['execution'].apply(lambda x: x.get('position_suggest'))

        return df_result[output_cols].reset_index(drop=True)

    # ==================== 新增：趋势因子计算方法 ====================

    def _calculate_trend_factors(self, df: pd.DataFrame, latest: pd.Series) -> Dict:
        """
        计算三类因子组（增强版）

        基于 MyTT 指标库实现更精准的多因子组合

        返回:
            - trend_factors: 趋势因子组
            - momentum_factors: 动能因子组
            - money_factors: 资金因子组
            - aux_factors: 辅助指标
        """
        # 提取基础数据序列
        CLOSE = df['close'].values
        HIGH = df['high'].values
        LOW = df['low'].values
        VOL = df['volume'].values
        OPEN = df['open'].values if 'open' in df.columns else CLOSE

        # ==================== 趋势因子组 ====================
        trend_factors = {}

        # 1. 均线排列评分 (MA5 > MA10 > MA20 > MA50)
        ma5 = MA(CLOSE, 5)[-1]
        ma10 = MA(CLOSE, 10)[-1]
        ma20 = MA(CLOSE, 20)[-1]
        ma50 = MA(CLOSE, 50)[-1] if len(CLOSE) >= 50 else ma20

        # 均线多头排列得分
        ma_alignment = 0
        if ma5 > ma10:
            ma_alignment += 25
        if ma10 > ma20:
            ma_alignment += 25
        if ma20 > ma50:
            ma_alignment += 25
        # 斜率方向
        ma5_slope = (MA(CLOSE, 5)[-1] - MA(CLOSE, 5)[-5]) / MA(CLOSE, 5)[-5] if MA(CLOSE, 5)[-5] > 0 else 0
        if ma5_slope > 0:
            ma_alignment += 25
        trend_factors['ma_alignment'] = ma_alignment

        # 2. DMI趋势强度
        pdi, mdi, adx = DMI(CLOSE, HIGH, LOW, M1=14, M2=6)
        pdi_val = pdi[-1]
        mdi_val = mdi[-1]
        adx_val = adx[-1]

        # DMI评分：PDI>MDI 且 ADX>20 表示上升趋势确立
        dmi_score = 50
        if pdi_val > mdi_val:
            dmi_score += 20
            if adx_val > 25:
                dmi_score += 20  # 趋势强
        elif mdi_val > pdi_val:
            dmi_score -= 20
        if adx_val > 30:
            dmi_score += 10  # 趋势明确
        trend_factors['dmi_strength'] = max(0, min(100, dmi_score))
        trend_factors['pdi'] = pdi_val
        trend_factors['mdi'] = mdi_val
        trend_factors['adx'] = adx_val

        # 3. MACD方向
        dif, dea, macd_val = MACD(CLOSE)
        dif_val = dif[-1]
        dea_val = dea[-1]
        macd_hist = macd_val[-1]

        # 保存原始MACD值供评分使用
        trend_factors['macd_dif'] = dif_val
        trend_factors['macd_dea'] = dea_val
        trend_factors['macd_hist'] = macd_hist

        macd_score = 50
        if dif_val > dea_val:
            macd_score += 25  # DIF在DEA上方
        if macd_hist > 0:
            macd_score += 15  # 红柱
        # MACD动量（柱状图变化）
        if len(macd_val) >= 4:
            macd_momentum = macd_val[-1] - macd_val[-4]
            if macd_momentum > 0:
                macd_score += 10  # 动量向上
        trend_factors['macd_direction'] = max(0, min(100, macd_score))
        trend_factors['macd_momentum'] = float(macd_momentum) if len(macd_val) >= 4 else 0

        # ==================== 动能因子组 ====================
        momentum_factors = {}

        # 4. KDJ位置
        k, d, j = KDJ(CLOSE, HIGH, LOW)
        k_val, d_val, j_val = k[-1], d[-1], j[-1]

        kdj_score = 50
        if 20 < k_val < 80:  # 正常区域
            if k_val > d_val:
                kdj_score += 20  # K在D上方
            if j_val > k_val and j_val > d_val:
                kdj_score += 15  # J最强
        elif k_val <= 20:
            kdj_score = 70  # 超卖，有机会
        elif k_val >= 80:
            kdj_score = 30  # 超买，风险
        momentum_factors['kdj_position'] = kdj_score
        momentum_factors['k'] = k_val
        momentum_factors['d'] = d_val
        momentum_factors['j'] = j_val

        # 5. RSI强度
        rsi_val = RSI(CLOSE, N=24)[-1]

        rsi_score = 50
        if 50 <= rsi_val <= 70:  # 多头强势区
            rsi_score = 80 + (70 - rsi_val)  # 80-100分
        elif 40 <= rsi_val < 50:
            rsi_score = 60
        elif 30 <= rsi_val < 40:
            rsi_score = 50  # 弱势但有机会
        elif rsi_val < 30:
            rsi_score = 65  # 超卖反弹机会
        elif rsi_val > 70:
            rsi_score = 35  # 超买风险
        momentum_factors['rsi_strength'] = rsi_score
        momentum_factors['rsi'] = rsi_val

        # 6. MTM动量
        mtm_val = MTM(CLOSE, N=12)[-1]
        mtm_score = 50 + (mtm_val / CLOSE[-1]) * 500 if CLOSE[-1] > 0 else 50
        momentum_factors['mtm_momentum'] = max(0, min(100, mtm_score))
        momentum_factors['mtm'] = mtm_val

        # 7. ROC变动率
        roc_val = ROC(CLOSE, N=12)[-1]
        roc_score = 50 + roc_val * 2  # ROC>0 加分，<0 减分
        momentum_factors['roc_rate'] = max(0, min(100, roc_score))
        momentum_factors['roc'] = roc_val

        # ==================== 资金因子组 ====================
        money_factors = {}

        # 8. OBV资金流
        obv_val = OBV(CLOSE, VOL)[-1]
        obv_ma5 = MA(OBV(CLOSE, VOL), 5)[-1] if len(CLOSE) >= 5 else obv_val
        obv_score = 70 if obv_val > obv_ma5 else 40  # OBV在均线上方为好
        money_factors['obv_flow'] = obv_score
        money_factors['obv'] = obv_val

        # 9. MFI资金流量（成交量的RSI）
        mfi_val = MFI(CLOSE, HIGH, LOW, VOL, N=14)[-1]
        mfi_score = 50
        if 20 <= mfi_val <= 80:
            if mfi_val > 50:
                mfi_score += 25
        elif mfi_val < 20:
            mfi_score = 70  # 超卖
        elif mfi_val > 80:
            mfi_score = 30  # 超买
        money_factors['mfi_strength'] = mfi_score
        money_factors['mfi'] = mfi_val

        # 10. 量价关系
        close_change = (CLOSE[-1] - CLOSE[-2]) / CLOSE[-2] if CLOSE[-2] > 0 else 0
        vol_change = (VOL[-1] - VOL[-2]) / VOL[-2] if VOL[-2] > 0 else 0

        vp_score = 50
        if close_change > 0 and vol_change > 0:
            vp_score = 80  # 价涨量增，健康
        elif close_change > 0 and vol_change < 0:
            vp_score = 55  # 价涨量缩，观望
        elif close_change < 0 and vol_change > 0:
            vp_score = 35  # 价跌量增，恐慌
        elif close_change < 0 and vol_change < 0:
            vp_score = 50  # 价跌量缩，正常回调
        money_factors['volume_price'] = vp_score

        # 当日量/5日均量
        vol_ma5 = np.mean(VOL[-5:]) if len(VOL) >= 5 else VOL[-1]
        volume_ratio = VOL[-1] / vol_ma5 if vol_ma5 > 0 else 1.0
        money_factors['volume_ratio'] = volume_ratio

        # ==================== 辅助指标 ====================
        aux_factors = {}

        # 布林带位置
        upper, mid, lower = BOLL(CLOSE)
        boll_upper, boll_mid, boll_lower = upper[-1], mid[-1], lower[-1]
        if boll_upper != boll_lower:
            pctb = (CLOSE[-1] - boll_lower) / (boll_upper - boll_lower)
        else:
            pctb = 0.5
        aux_factors['pctb'] = pctb
        aux_factors['boll_upper'] = boll_upper
        aux_factors['boll_lower'] = boll_lower

        # WR
        wr_val = WR(CLOSE, HIGH, LOW, N=10)[-1]
        aux_factors['wr'] = wr_val

        # CCI
        cci_val = CCI(CLOSE, HIGH, LOW, N=14)[-1]
        aux_factors['cci'] = cci_val

        # PSY心理线
        psy_val = PSY(CLOSE, N=12)[-1]
        aux_factors['psy'] = psy_val

        # BIAS
        bias6 = (CLOSE[-1] - MA(CLOSE, 6)[-1]) / MA(CLOSE, 6)[-1] if MA(CLOSE, 6)[-1] > 0 else 0
        bias20 = (CLOSE[-1] - MA(CLOSE, 20)[-1]) / MA(CLOSE, 20)[-1] if MA(CLOSE, 20)[-1] > 0 else 0
        aux_factors['bias6'] = bias6
        aux_factors['bias20'] = bias20

        # 60日高低点位置比率（用于K线形态位置判断）
        if len(df) >= 60:
            high_60 = df['high'].iloc[-60:].max()
            low_60 = df['low'].iloc[-60:].min()
            position_ratio = (CLOSE[-1] - low_60) / (high_60 - low_60) if high_60 > low_60 else 0.5
        else:
            position_ratio = 0.5
        aux_factors['position_ratio'] = position_ratio

        # 趋势强度
        aux_factors['trend_strength'] = bias20

        # MA值
        aux_factors['ma5'] = ma5
        aux_factors['ma10'] = ma10
        aux_factors['ma20'] = ma20
        aux_factors['ma50'] = ma50
        aux_factors['close'] = CLOSE[-1]

        return {
            'trend_factors': trend_factors,
            'momentum_factors': momentum_factors,
            'money_factors': money_factors,
            'aux_factors': aux_factors,
        }

    def _calculate_trend_score(self, factors: Dict, df: pd.DataFrame = None) -> Tuple[float, Dict]:
        """
        计算趋势得分（重构版 - 集成K线形态）

        Args:
            factors: 因子字典，包含 trend_factors, momentum_factors, money_factors, aux_factors
            df: 股票数据DataFrame（用于K线形态识别）

        Returns:
            Tuple[float, Dict]: (趋势得分 0-100, 各因子得分字典)
        """
        factor_scores = {}

        # 从嵌套结构中提取各类因子
        trend_factors = factors.get('trend_factors', {})
        momentum_factors = factors.get('momentum_factors', {})
        money_factors = factors.get('money_factors', {})
        aux_factors = factors.get('aux_factors', {})

        # 1. 趋势强度评分（使用 aux_factors 中的 bias20，并考虑DMI状态）
        bias20 = aux_factors.get('bias20', 0)
        pdi = trend_factors.get('pdi', 0)
        mdi = trend_factors.get('mdi', 0)
        adx = trend_factors.get('adx', 0)
        factor_scores['trend_strength'] = self._score_trend_strength(bias20, pdi, mdi, adx)

        # 2. 均线斜率评分
        ma5_slope = trend_factors.get('ma5_slope', 0)
        ma10 = aux_factors.get('ma10', 0)
        ma20 = aux_factors.get('ma20', 0)
        factor_scores['ma_slope'] = self._score_ma_slope(ma5_slope, ma10, ma20)

        # 3. MACD综合评分（修复注水问题）
        macd_momentum = trend_factors.get('macd_momentum', 0)
        macd_dif = trend_factors.get('macd_dif', 0)
        macd_dea = trend_factors.get('macd_dea', 0)
        macd_hist = trend_factors.get('macd_hist', 0)
        factor_scores['macd_momentum'] = self._score_macd_comprehensive(
            macd_momentum, macd_dif, macd_dea, macd_hist
        )

        # 4. 资金流向评分（使用 OBV 评分）
        obv_flow = money_factors.get('obv_flow', 50)  # 使用已计算的OBV评分
        factor_scores['money_flow'] = obv_flow

        # 5. 成交量评分（结合趋势方向，加入均线排列判断和DMI）
        volume_ratio = money_factors.get('volume_ratio', 1.0)
        trend_direction = self._determine_trend_direction({
            'trend_strength': bias20,
            'ma5_slope': ma5_slope,
            'macd_momentum': macd_momentum,
            'pdi': trend_factors.get('pdi', 0),
            'mdi': trend_factors.get('mdi', 0),
            'adx': trend_factors.get('adx', 0),
        }, aux_factors)  # 传入 aux_factors 用于均线排列判断
        factor_scores['volume_ratio'] = self._score_volume_ratio(volume_ratio, trend_direction)

        # K线形态已移至独立筛选层，不再参与评分计算
        # 如需获取K线形态筛选结果，请在评分后调用 screening.CandlestickPatternScreener

        # 加权计算总分
        total_score = sum(
            factor_scores[k] * self.TREND_FACTOR_WEIGHTS[k]
            for k in self.TREND_FACTOR_WEIGHTS
        )

        return total_score, factor_scores

    def _determine_trend_direction(self, factors: Dict, aux_factors: Dict = None) -> str:
        """
        判断趋势方向（优化版：均线排列 + DMI + MACD 综合判断）

        核心逻辑：
        1. DMI判断多空力量对比（PDI vs MDI）
        2. ADX判断趋势强度（>25有趋势，<20震荡）
        3. 均线排列判断长期方向
        4. 三者必须一致才能判定明确趋势

        Returns:
            'up': 上涨趋势
            'down': 下跌趋势
            'sideways': 横盘震荡
        """
        trend_strength = factors.get('trend_strength', 0)
        ma5_slope = factors.get('ma5_slope', 0)
        macd_momentum = factors.get('macd_momentum', 0)

        # DMI指标
        pdi = factors.get('pdi', 0)
        mdi = factors.get('mdi', 0)
        adx = factors.get('adx', 0)

        # 新增：均线排列判断
        ma_alignment = 'neutral'  # 默认中性
        if aux_factors:
            close = aux_factors.get('close', 0)
            ma5 = aux_factors.get('ma5', 0)
            ma20 = aux_factors.get('ma20', 0)
            ma50 = aux_factors.get('ma50', 0)

            # 多头排列：MA5 > MA20 > MA50 且 股价在均线上方
            if ma5 > 0 and ma20 > 0 and ma50 > 0:
                if ma5 > ma20 > ma50 and close > ma5:
                    ma_alignment = 'bullish'  # 多头排列
                elif ma5 < ma20 < ma50 and close < ma5:
                    ma_alignment = 'bearish'  # 空头排列

        # ==================== DMI趋势判断 ====================
        # DMI核心逻辑：
        # - PDI > MDI 且 ADX > 25 = 上升趋势确立
        # - MDI > PDI 且 ADX > 25 = 下降趋势确立
        # - ADX < 20 = 无趋势震荡
        dmi_trend = 'neutral'
        dmi_strength = 0  # DMI信号强度

        if adx < 20:
            # ADX < 20 表示无明确趋势，震荡市
            dmi_trend = 'sideways'
            dmi_strength = 1
        elif pdi > mdi and adx > 25:
            # 多头力量占优且趋势明显
            dmi_trend = 'up'
            dmi_strength = 2 if adx > 30 else 1
        elif mdi > pdi and adx > 25:
            # 空头力量占优且趋势明显
            dmi_trend = 'down'
            dmi_strength = 2 if adx > 30 else 1
        elif pdi > mdi:
            # 多头略强但趋势不明显
            dmi_trend = 'weak_up'
            dmi_strength = 0.5
        elif mdi > pdi:
            # 空头略强但趋势不明显
            dmi_trend = 'weak_down'
            dmi_strength = 0.5

        # ==================== 综合判断 ====================
        up_signals = 0
        down_signals = 0

        # 1. DMI信号（权重最高）
        if dmi_trend == 'up':
            up_signals += 2
        elif dmi_trend == 'down':
            down_signals += 2
        elif dmi_trend == 'weak_up':
            up_signals += 0.5
        elif dmi_trend == 'weak_down':
            down_signals += 0.5
        elif dmi_trend == 'sideways':
            # 震荡市，降低信号权重
            pass

        # 2. 趋势强度（BIAS20）
        if trend_strength > 0.02:
            up_signals += 1
        elif trend_strength < -0.02:
            down_signals += 1

        # 3. 均线斜率
        if ma5_slope > 0.005:
            up_signals += 0.5
        elif ma5_slope < -0.005:
            down_signals += 0.5

        # 4. MACD动量
        if macd_momentum > 0:
            up_signals += 0.5
        elif macd_momentum < 0:
            down_signals += 0.5

        # ==================== 关键判断逻辑 ====================

        # 优先看DMI：如果DMI明确显示空头占优，即使均线金叉也要警惕
        if dmi_trend == 'down' or dmi_trend == 'weak_down':
            # DMI空头占优
            if ma_alignment == 'bullish':
                # 均线多头但DMI空头 = 假突破/诱多
                return 'sideways'  # 判定为震荡，不追高
            return 'down'

        if dmi_trend == 'up':
            # DMI多头占优
            if ma_alignment == 'bearish':
                # 均线空头但DMI多头 = 反弹而非反转
                return 'sideways'
            return 'up'

        # DMI震荡时（ADX < 20），看均线排列
        if dmi_trend == 'sideways':
            if ma_alignment == 'bullish':
                return 'sideways'  # 均线多头但无趋势，判定震荡
            elif ma_alignment == 'bearish':
                return 'down'  # 均线空头且无趋势，偏弱
            return 'sideways'

        # 默认震荡
        if up_signals > down_signals + 1:
            return 'up'
        elif down_signals > up_signals + 1:
            return 'down'
        else:
            return 'sideways'

    def _determine_position_zone(self, position_ratio: float,
                                  bias20: float,
                                  boll_pctb: float) -> Tuple[str, str]:
        """
        判断当前位置区域（用于K线形态评分）

        核心逻辑：位置决定形态意义
        - 低位 + 看涨形态 = 强力底部信号
        - 高位 + 看涨形态 = 警惕诱多/力竭
        - 高位 + 看跌形态 = 强力顶部信号
        - 低位 + 看跌形态 = 可能是最后洗盘

        Args:
            position_ratio: 股价相对60日高低点位置 (0-1)
            bias20: MA20乖离率
            boll_pctb: 布林带百分比位置 (0-1)

        Returns:
            Tuple[str, str, str, str]: (综合位置区域key, 综合位置描述, 长期位置, 短期位置)
        """
        # 长期位置：基于60日高低点分位（反映大周期位置）
        if position_ratio < 0.35:
            long_term_position = "low"  # 长期低位
        elif position_ratio > 0.70:
            long_term_position = "high"  # 长期高位
        else:
            long_term_position = "mid"  # 长期中位

        # 短期位置：基于乖离率和布林带位置（反映短线超买超卖）
        is_short_high = bias20 > 0.05 or boll_pctb > 0.8
        is_short_low = bias20 < -0.05 or boll_pctb < 0.2

        if is_short_low:
            short_term_position = "low"  # 短期超卖
        elif is_short_high:
            short_term_position = "high"  # 短期超买
        else:
            short_term_position = "mid"  # 短期正常

        # 综合位置判断（优先考虑短期风险）
        # 如果长期低位但短期超买，视为"短线高位"（建议等待回踩）
        # 如果长期高位但短期超卖，视为"可能反弹"（但不改变大趋势判断）
        if short_term_position == "high":
            # 短期超买，无论长期位置如何，都视为风险区
            return "high_position", f"短期超买（长期{long_term_position}位）", long_term_position, short_term_position
        elif short_term_position == "low":
            # 短期超卖，入场机会
            return "low_position", f"短期超卖（长期{long_term_position}位）", long_term_position, short_term_position
        else:
            # 短期正常，使用长期位置
            if long_term_position == "low":
                return "low_position", "长期低位（短期正常）", long_term_position, short_term_position
            elif long_term_position == "high":
                return "high_position", "长期高位（短期正常）", long_term_position, short_term_position
            else:
                return "mid_position", "中位区域", long_term_position, short_term_position

    def _get_pattern_weight(self, pattern_name: str, pattern_type: str) -> float:
        """
        获取K线形态的基础权重

        Args:
            pattern_name: 形态名称
            pattern_type: 形态类型 ('bullish' / 'bearish')

        Returns:
            float: 基础权重值
        """
        for category, config in self.CANDLESTICK_PATTERN_WEIGHTS.items():
            if pattern_name in config['patterns']:
                return config['base_weight']

        # 默认权重（根据类型）
        if pattern_type == 'bullish':
            return 5.0
        elif pattern_type == 'bearish':
            return -5.0
        else:
            return 0.0

    def _calculate_candlestick_score(self, df: pd.DataFrame,
                                      position_ratio: float,
                                      bias20: float,
                                      boll_pctb: float) -> Tuple[float, Dict]:
        """
        计算K线形态评分（位置敏感）

        .. deprecated::
            此方法已废弃，K线形态已移至独立筛选层。
            请使用 quanttool.factors.screening.CandlestickPatternScreener 替代。
            保留此方法仅为向后兼容，将在未来版本中移除。

        核心逻辑：先判位置，再判形态
        - 低位 + 看涨形态 = 强力加分
        - 高位 + 看涨形态 = 减分（警惕诱多）
        - 高位 + 看跌形态 = 强力减分
        - 低位 + 看跌形态 = 中性（可能洗盘）

        Args:
            df: 股票数据DataFrame
            position_ratio: 股价相对60日高低点位置
            bias20: MA20乖离率
            boll_pctb: 布林带百分比位置

        Returns:
            Tuple[float, Dict]: (评分 -30~+30, 详情字典)
        """
        import warnings
        warnings.warn(
            "_calculate_candlestick_score 已废弃，请使用 screening.CandlestickPatternScreener",
            DeprecationWarning,
            stacklevel=2
        )

        # 调用形态识别
        patterns_result = analyze_candlestick_patterns(df, lookback=5)

        if "error" in patterns_result or not patterns_result.get("patterns"):
            return 0.0, {"patterns": [], "position_zone": "unknown", "assessment": ""}

        # 判断位置区域（区分长期和短期）
        position_zone, position_desc, long_term, short_term = self._determine_position_zone(
            position_ratio, bias20, boll_pctb
        )
        modifiers = self.POSITION_PATTERN_MODIFIERS[position_zone]

        # 计算评分
        total_score = 0.0
        pattern_details = []

        for pattern in patterns_result["patterns"]:
            p_type = pattern.get("type", "neutral")
            if p_type == "neutral":
                continue

            pattern_name = pattern.get("name", "")
            # 获取基础权重
            base_weight = self._get_pattern_weight(pattern_name, p_type)

            # 应用位置修正系数
            modifier = modifiers["bullish" if p_type == "bullish" else "bearish"]

            # 强度调整系数
            strength = pattern.get("strength", "中")
            strength_mult = {"强": 1.0, "中": 0.7, "弱": 0.4}.get(strength, 0.5)

            # 计算该形态的最终得分
            pattern_score = base_weight * modifier * strength_mult
            total_score += pattern_score

            # 记录详情
            pattern_details.append({
                "name": pattern_name,
                "type": p_type,
                "strength": strength,
                "base_weight": base_weight,
                "modifier": modifier,
                "final_score": round(pattern_score, 2),
            })

        # 限制评分范围 [-30, +30]
        final_score = max(-30.0, min(30.0, total_score))

        # 生成评估描述
        assessment = self._generate_pattern_assessment(
            patterns_result, position_zone, position_desc
        )

        return final_score, {
            "patterns": pattern_details,
            "position_zone": position_zone,
            "position_desc": position_desc,
            "long_term_position": long_term,
            "short_term_position": short_term,
            "assessment": assessment,
        }

    def _generate_pattern_assessment(self, patterns_result: Dict,
                                      position_zone: str,
                                      position_desc: str) -> str:
        """
        生成K线形态评估描述

        Args:
            patterns_result: 形态识别结果
            position_zone: 位置区域
            position_desc: 位置描述

        Returns:
            str: 评估描述
        """
        patterns = patterns_result.get("patterns", [])
        if not patterns:
            return ""

        # 分类形态
        bullish_patterns = [p for p in patterns if p.get("type") == "bullish"]
        bearish_patterns = [p for p in patterns if p.get("type") == "bearish"]

        assessments = []

        # 根据位置和形态组合生成评估
        if position_zone == "low_position":
            if bullish_patterns:
                strong = [p for p in bullish_patterns if p.get("strength") == "强"]
                if strong:
                    assessments.append(f"【强力底部信号】{position_desc}出现强看涨形态，底部反转概率高")
                else:
                    assessments.append(f"【底部信号】{position_desc}出现看涨形态，关注反弹机会")
            if bearish_patterns:
                assessments.append(f"【洗盘信号】{position_desc}出现看跌形态，可能是最后恐慌洗盘")

        elif position_zone == "high_position":
            if bearish_patterns:
                strong = [p for p in bearish_patterns if p.get("strength") == "强"]
                if strong:
                    assessments.append(f"【强力顶部信号】{position_desc}出现强看跌形态，顶部确认")
                else:
                    assessments.append(f"【顶部信号】{position_desc}出现看跌形态，警惕回调")
            if bullish_patterns:
                assessments.append(f"【警惕】{position_desc}出现看涨形态，可能是诱多或力竭")

        else:  # mid_position
            if bullish_patterns:
                assessments.append(f"【延续信号】{position_desc}出现看涨形态，趋势可能延续")
            if bearish_patterns:
                assessments.append(f"【调整信号】{position_desc}出现看跌形态，关注调整深度")

        return "; ".join(assessments) if assessments else ""

    def _score_trend_strength(self, bias: float, pdi: float = 0, mdi: float = 0, adx: float = 0) -> float:
        """
        趋势强度评分（修复版：考虑DMI多空状态）

        逻辑：
        - 股价在MA20上方3-8%：高分（趋势确立且不过度偏离）
        - 股价在MA20上方0-3%：中等分（趋势刚形成）
        - 股价在MA20上方>8%：低分（过度偏离，回调风险）
        - 股价在MA20下方：极低分（趋势未确立）
        - **新增**：DMI空头占优时，上限封顶60分（趋势不可靠）
        - **新增**：DMI多头占优时，可正常评分
        """
        # 基础分数（基于乖离率）
        if bias > 0.08:
            base_score = 30  # 过度偏离
        elif bias > 0.03:
            base_score = 90 + (bias - 0.03) * 200  # 90-100分，最佳区域
        elif bias > 0:
            base_score = 60 + bias * 1000  # 60-90分，趋势刚形成
        elif bias > -0.05:
            base_score = 40 + bias * 400  # 40-60分，弱势
        else:
            base_score = max(10, 40 + bias * 200)  # 10-40分，趋势未确立

        # 关键修复：DMI状态修正
        dmi_diff = abs(pdi - mdi)
        DMI_THRESHOLD = 3.0  # DMI多空差异阈值

        if mdi > pdi and dmi_diff >= DMI_THRESHOLD:
            # DMI空头占优 - 趋势不可靠，封顶60分
            # 即使股价在MA20上方，也可能是诱多或反弹受阻
            return min(base_score, 55)
        elif pdi > mdi and dmi_diff >= DMI_THRESHOLD and adx > 25:
            # DMI多头占优且趋势明确 - 可信度高
            return base_score
        elif adx < 20:
            # 无明确趋势 - 降低评分
            return min(base_score, 65)
        else:
            return base_score

    def _score_ma_slope(self, slope: float, ma10: float, ma20: float) -> float:
        """
        均线斜率评分（优化版：考虑均值回归）

        核心逻辑：
        - 温和上涨的斜率最好（可持续）
        - 过于陡峭的斜率可能面临回调风险
        - 下跌斜率在低位可能是买入机会
        """
        score = 50  # 基础分

        # 斜率贡献（优化：温和上涨最优，陡峭上涨风险大）
        if slope > 0.03:  # 陡峭向上 - 可能过热，降低评分
            score += 5  # 降低加分（原30）
        elif slope > 0.01:  # 温和向上 - 最佳区间
            score += 25
        elif slope > 0.003:  # 微弱向上
            score += 15
        elif slope > -0.005:  # 平稳
            score += 10
        elif slope > -0.02:  # 温和下跌 - 可能是买点
            score += 5
        else:  # 陡峭下跌
            score -= 10  # 降低减分（原20）

        # 金叉加分（降低权重）
        if ma10 > 0 and ma20 > 0 and ma10 > ma20:
            score += 10  # 降低加分（原20）

        return max(0, min(100, score))

    def _score_macd_momentum(self, momentum: float) -> float:
        """
        MACD动量评分（已废弃，改用 _score_macd_comprehensive）
        """
        return self._score_macd_comprehensive(momentum, 0, 0, 0)

    def _score_macd_comprehensive(self, momentum: float, dif_val: float,
                                   dea_val: float, macd_hist: float) -> float:
        """
        MACD综合评分（修复注水问题）

        核心逻辑：
        1. MACD在零轴下方时，基础分不及格（<=50分）
        2. MACD在零轴上方且金叉时，才可能得高分
        3. 柱状图颜色和动量变化作为加分项

        Args:
            momentum: 柱状图变化（macd[-1] - macd[-4]）
            dif_val: DIF值
            dea_val: DEA值
            macd_hist: 柱状图值（MACD柱）

        Returns:
            float: 0-100分
        """
        # 基础分：根据MACD位置决定
        if dif_val < 0 and dea_val < 0:
            # 零轴下方，空头市场，基础分低
            base_score = 35
            # 但如果绿柱缩短（收敛），说明空头力量衰竭
            if macd_hist < 0 and momentum > 0:
                base_score += 15  # 绿柱缩短，有机会反弹
            elif macd_hist > 0:
                base_score += 10  # 在零轴下但出红柱（金叉）
        elif dif_val < 0 or dea_val < 0:
            # 一线在零轴下方，震荡市
            base_score = 45
            if macd_hist > 0:
                base_score += 15  # 红柱
        else:
            # 零轴上方，多头市场
            base_score = 60
            if macd_hist > 0:
                base_score += 20  # 红柱
            elif macd_hist < 0:
                base_score -= 10  # 绿柱（死叉风险）

        # 金叉/死叉调整
        if dif_val > dea_val:
            # 金叉状态
            if macd_hist > 0:
                base_score += 10  # 金叉且有红柱
        else:
            # 死叉状态
            base_score -= 10

        # 动量变化调整
        if momentum > 0.05:
            base_score += 10  # 动量强劲向上
        elif momentum > 0:
            base_score += 5  # 动量向上
        elif momentum < -0.05:
            base_score -= 10  # 动量快速向下
        elif momentum < 0:
            base_score -= 5  # 动量向下

        # 限制范围 0-100
        # 关键：零轴下方最高不能超过65分
        if dif_val < 0 and dea_val < 0:
            base_score = min(65, base_score)

        return max(0, min(100, base_score))

    def _score_money_flow(self, flow: float) -> float:
        """
        资金流向评分（OBV变化率）
        """
        if flow > 20:
            return 100
        elif flow > 10:
            return 90
        elif flow > 5:
            return 80
        elif flow > 0:
            return 60 + flow * 4  # 60-80
        elif flow > -10:
            return 40 + flow * 2  # 20-40
        else:
            return max(10, 40 + flow)

    def _score_volume_ratio(self, ratio: float, trend_direction: str = 'up') -> float:
        """
        成交量评分（优化版：结合趋势方向）

        Args:
            ratio: 量比（当日量/5日均量）
            trend_direction: 趋势方向 ('up'/'down'/'sideways')

        逻辑：
        - 上涨趋势 + 放量 = 高分（资金抢筹）
        - 上涨趋势 + 缩量 = 中等分（上涨乏力）
        - 下跌趋势 + 放量 = 低分（恐慌出逃）
        - 下跌趋势 + 缩量 = 中等分（抛压衰竭）
        """
        if trend_direction == 'up':
            # 上涨趋势中的成交量评分
            if 1.2 <= ratio <= 2.5:
                return 95  # 温和放量，资金抢筹
            elif 1.0 <= ratio < 1.2:
                return 80  # 量能不足但尚可
            elif ratio > 2.5:
                return 60  # 放量过大，可能见顶
            elif 0.7 <= ratio < 1.0:
                return 50  # 缩量上涨，动力不足
            else:
                return 35  # 严重缩量，上涨乏力

        elif trend_direction == 'down':
            # 下跌趋势中的成交量评分
            if ratio > 2.0:
                return 25  # 放量下跌，恐慌出逃
            elif ratio > 1.2:
                return 35  # 温和放量下跌，仍有抛压
            elif 0.6 <= ratio <= 1.2:
                return 50  # 缩量下跌，抛压衰竭（可能有反弹）
            else:
                return 60  # 严重缩量下跌，抛压枯竭（机会）

        else:
            # 横盘震荡
            if 0.8 <= ratio <= 1.5:
                return 70  # 量能适中
            elif ratio > 1.5:
                return 55  # 放量但方向不明
            else:
                return 60  # 缩量震荡

    def _calculate_position_modifier(self, latest: pd.Series, factors: Dict) -> Tuple[float, List[str]]:
        """
        计算位置修正系数（风险控制）- 趋势敏感版

        核心改进：位置风险必须结合趋势方向判断
        - 下跌趋势 + 布林带下轨附近 = 危险区（接飞刀风险）
        - 下跌趋势 + 布林带上轨附近 = 反弹到阻力位，危险
        - 上升趋势 + 布林带下轨附近 = 回调机会，相对安全
        - 上升趋势 + 布林带上轨附近 = 超买，但趋势可能延续

        用于风险控制，不奖励强势，只惩罚风险
        """
        modifier = 1.0
        warnings = []

        # 从 aux_factors 中获取指标值
        aux_factors = factors.get('aux_factors', {})
        momentum_factors = factors.get('momentum_factors', {})
        trend_factors = factors.get('trend_factors', {})

        pctb = aux_factors.get('pctb', 0.5)
        rsi = momentum_factors.get('rsi', 50)
        wr = aux_factors.get('wr', 50)
        bias6 = aux_factors.get('bias6', 0)
        bias20 = aux_factors.get('bias20', 0)
        cci = aux_factors.get('cci', 0)
        position_ratio = aux_factors.get('position_ratio', 0.5)  # 新增：60日分位

        # 获取均线数据用于趋势判断
        close = aux_factors.get('close', 0)
        ma5 = aux_factors.get('ma5', 0)
        ma10 = aux_factors.get('ma10', 0)
        ma20 = aux_factors.get('ma20', 0)
        ma50 = aux_factors.get('ma50', 0)

        # ==================== 新增：长期位置惩罚 ====================
        # 如果股价处于60日高位（前30%），给予惩罚
        if position_ratio > 0.85:
            modifier *= 0.65
            warnings.append(f"🔴【长期高位】股价处于60日{position_ratio*100:.0f}%分位，追高风险极大")
        elif position_ratio > 0.70:
            modifier *= 0.80
            warnings.append(f"🟠【相对高位】股价处于60日{position_ratio*100:.0f}%分位，需谨慎")

        # ==================== 核心修复：趋势方向判断 ====================
        # 判断趋势方向（不能只看DMI，必须看均线排列）
        is_downtrend = False
        is_uptrend = False
        trend_desc = ""

        if ma5 > 0 and ma20 > 0 and ma50 > 0:
            # 空头排列：MA5 < MA20 < MA50 且 股价在均线下方
            if ma5 < ma20 < ma50 and close < ma5:
                is_downtrend = True
                trend_desc = "空头排列"
            # 多头排列：MA5 > MA20 > MA50 且 股价在均线上方
            elif ma5 > ma20 > ma50 and close > ma5:
                is_uptrend = True
                trend_desc = "多头排列"
            # 股价在MA20和MA50下方（即使没有完美空头排列）
            elif close < ma20 and close < ma50:
                is_downtrend = True
                trend_desc = "均线压制"
            # 股价在MA20和MA50上方
            elif close > ma20 and close > ma50:
                is_uptrend = True
                trend_desc = "均线支撑"

        # ==================== 关键修复：下跌趋势风险惩罚 ====================
        # 在下跌趋势中，任何位置都是高风险
        if is_downtrend:
            # 下跌趋势 + 布林带下轨附近 = 接飞刀风险
            if pctb < 0.2:
                modifier *= 0.5
                warnings.append(f"🔴【下跌趋势+超卖】布林下轨附近({pctb*100:.1f}%)，接飞刀风险极高！")
            # 下跌趋势 + 布林带上轨附近 = 反弹到阻力位
            elif pctb > 0.7:
                modifier *= 0.45
                warnings.append(f"🔴【下跌趋势+阻力位】布林上轨附近({pctb*100:.1f}%)，反弹遇阻风险！")
            # 下跌趋势 + 中位 = 下跌中继
            else:
                modifier *= 0.6
                warnings.append(f"🟠【下跌趋势】{trend_desc}，不建议买入")

            # 下跌趋势中，MA20和MA50是强阻力
            if close < ma20 and ma20 > 0:
                warnings.append(f"⚠️ MA20({ma20:.2f})为强阻力位，买入即被套风险大")

        # ==================== 上升趋势风险调整 ====================
        elif is_uptrend:
            # 上升趋势 + 布林带下轨附近 = 回调机会（相对安全）
            if pctb < 0.2:
                # 不惩罚，甚至可以稍微加分，但保持谨慎
                warnings.append(f"📊【上升趋势+超卖】布林下轨附近({pctb*100:.1f}%)，回调机会")
            # 上升趋势 + 布林带上轨附近 = 超买但趋势可能延续
            elif pctb > 0.85:
                modifier *= 0.75
                warnings.append(f"⚠️【上升趋势+超买】布林上轨附近({pctb*100:.1f}%)，追高风险")
            elif pctb > 0.7:
                modifier *= 0.85
                warnings.append(f"📈 上升趋势中，布林带偏高({pctb*100:.1f}%)")

            # ==================== 关键修复：上升趋势中的极端超买惩罚 ====================
            # BIAS24 > 8% = 极端超买，即使上升趋势也要惩罚
            if bias20 > 0.08:
                modifier *= 0.4
                warnings.append(f"🔴【极端超买】BIAS20={bias20*100:.1f}%，乖离过大，回调风险极高！")
            elif bias20 > 0.06:
                modifier *= 0.6
                warnings.append(f"⚠️【超买警告】BIAS20={bias20*100:.1f}%，乖离偏大")
            elif bias20 > 0.04:
                modifier *= 0.8
                warnings.append(f"📈 BIAS20={bias20*100:.1f}%，注意回调风险")
        # ==================== 无明确趋势 ====================
        else:
            # 震荡行情，按原逻辑处理
            if pctb > 0.85:
                modifier *= 0.7
                warnings.append(f"震荡行情，布林带上轨附近({pctb*100:.1f}%)，超买风险")

        # ==================== 极端行情熔断机制 ====================
        extreme_overbought_count = 0
        if pctb > 1.0:
            extreme_overbought_count += 1
        if wr < 5:
            extreme_overbought_count += 1
        if cci > 200:
            extreme_overbought_count += 1
        if rsi > 80:
            extreme_overbought_count += 1

        if extreme_overbought_count >= 3:
            modifier = min(modifier, 0.3)
            warnings.append(f"🔴【极端超买熔断】{extreme_overbought_count}个指标同时爆表！")

        # ==================== 其他指标修正（仅在非下跌趋势时应用） ====================
        if not is_downtrend:
            # RSI修正
            if rsi > 80:
                modifier *= 0.6
                warnings.append(f"RSI严重超买({rsi:.0f})")
            elif rsi > 75:
                modifier *= 0.75
                warnings.append(f"RSI超买({rsi:.0f})")
            elif rsi > 70:
                modifier *= 0.85

            # WR修正（WR越小越超买）
            if wr < 5:
                modifier *= 0.5
                warnings.append(f"⚠️ WR极度超买({wr:.1f})，见顶信号极强")
            elif wr < 10:
                modifier *= 0.6
                warnings.append(f"WR严重超买({wr:.1f})")
            elif wr < 20:
                modifier *= 0.8

            # CCI修正
            if cci > 200:
                modifier *= 0.5
                warnings.append(f"⚠️ CCI极度超买({cci:.0f})")
            elif cci > 150:
                modifier *= 0.6
                warnings.append(f"CCI严重超买({cci:.0f})")
            elif cci > 100:
                modifier *= 0.8

            # BIAS修正
            if bias6 > 0.08:
                modifier *= 0.5
                warnings.append(f"乖离率过大({bias6*100:.1f}%)")
            elif bias6 > 0.06:
                modifier *= 0.7
                warnings.append(f"乖离率偏高({bias6*100:.1f}%)")
            elif bias6 > 0.05:
                modifier *= 0.85

        return max(0.3, modifier), warnings

    # ==================== 原有方法 ====================

    def _calculate_factors(self, df: pd.DataFrame, latest: pd.Series) -> Dict:
        """
        计算9个因子的原始值

        因子设计采用"目标值偏离"思路：
        - 每个因子都有一个"最优值"
        - 偏离最优值越远，得分越低
        - 最终在股票池中进行rank_pct归一化
        """
        close = latest.get('close', np.nan)
        ma10 = latest.get('ma_10', np.nan)
        ma20 = latest.get('ma_20', np.nan)
        ma50 = latest.get('ma_50', np.nan)

        # 1. trend1: close/ma20 - 1，衡量偏离均线的程度
        # 最优值：0~5%（略微偏离），过高或过低都不好
        trend1 = (close / ma20 - 1) if not pd.isna(ma20) and ma20 != 0 else 0

        # 2. trend2: ma10/ma20 - 1，衡量短期均线斜率
        # 最优值：0~3%（多头排列初期）
        trend2 = (ma10 / ma20 - 1) if not pd.isna(ma10) and not pd.isna(ma20) and ma20 != 0 else 0

        # 3. mom1: MACD histogram 3日变化
        # 最优值：正值且加速（ MACD hist 持续扩大）
        mom1 = 0
        if len(df) >= 4 and 'macd' in df.columns:
            macd_hist = df['macd']
            mom1 = macd_hist.iloc[-1] - macd_hist.iloc[-4]  # 3日变化

        # 4. mom2: -|RSI-60|，RSI偏离60的程度
        # 最优值：60（温和偏多），偏离扣分
        rsi = latest.get('rsi_24', 50)
        mom2 = -abs(rsi - 60)

        # 5. flow1: log(vol_ratio)，成交量对数比
        # 最优值：1.0-1.5（温和放量）
        vol = latest.get('volume', 0)
        vol_ma5 = df['volume'].tail(5).mean() if len(df) >= 5 else vol
        vol_ratio = vol / vol_ma5 if vol_ma5 > 0 else 1.0
        flow1 = np.log(vol_ratio + 1e-12)

        # 6. flow2: OBV 5日变化率（%）
        # 最优值：正值（资金流入）
        # 改为相对变化率，避免大盘股垄断高分
        flow2 = 0
        if 'obv' in df.columns and len(df) >= 6:
            obv_current = latest.get('obv', 0)
            obv_prev = df['obv'].iloc[-6]
            if abs(obv_prev) > 0:
                flow2 = (obv_current - obv_prev) / abs(obv_prev) * 100  # 百分比变化

        # 7. pos1: -|pctB - 0.65|，布林带位置偏离0.65的程度
        # 最优值：0.65（中轨偏上），偏离扣分
        boll_upper = latest.get('boll_upper', close)
        boll_lower = latest.get('boll_lower', close)
        boll_mid = latest.get('boll_mid', close)
        if boll_upper != boll_lower:
            pctb = (close - boll_lower) / (boll_upper - boll_lower)
        else:
            pctb = 0.5
        pos1 = -abs(pctb - 0.65)

        # 8. pos2: -|ATR% - 0.03|，ATR偏离3%的程度
        # 最优值：3%波动率，偏离扣分
        atr = latest.get('atr_14', 0)
        atrp = atr / close if close > 0 else 0
        pos2 = -abs(atrp - 0.03)

        # 9. bias20: MA20乖离率（close/ma20 - 1）
        # 最优值：0（越接近0越好，超过±8%评分为0）
        bias20 = (close / ma20 - 1) if not pd.isna(ma20) and ma20 != 0 else 0

        # 10. bias6: BIAS(6)乖离率，用于超买惩罚
        ma6 = latest.get('ma_6', np.nan)
        bias6 = (close / ma6 - 1) if not pd.isna(ma6) and ma6 != 0 else 0

        return {
            'trend1': float(trend1),
            'trend2': float(trend2),
            'mom1': float(mom1),
            'mom2': float(mom2),
            'flow1': float(flow1),
            'flow2': float(flow2),
            'pos1': float(pos1),
            'pos2': float(pos2),
            'bias20': float(bias20),
            'bias6': float(bias6),
        }

    def _calculate_score_percentile(self, factors: Dict) -> Tuple[float, Dict]:
        """
        计算百分制得分

        由于单只股票无法进行rank_pct，这里先计算原始因子值的加权得分
        实际使用时应在股票池中进行rank_pct归一化
        """
        # 将每个因子值映射到0-1区间（基于经验阈值）
        factor_norm = {}

        # trend1: -0.05~0.15 映射到 0-1
        factor_norm['trend1'] = self._map_range(factors['trend1'], -0.05, 0.15, 0, 1)

        # trend2: -0.05~0.10 映射到 0-1
        factor_norm['trend2'] = self._map_range(factors['trend2'], -0.05, 0.10, 0, 1)

        # mom1: -0.5~0.5 映射到 0-1
        factor_norm['mom1'] = self._map_range(factors['mom1'], -0.5, 0.5, 0, 1)

        # mom2: -40~0 映射到 0-1（-|RSI-60|，范围-40到0）
        factor_norm['mom2'] = self._map_range(factors['mom2'], -40, 0, 0, 1)

        # flow1: -0.5~0.5 映射到 0-1
        factor_norm['flow1'] = self._map_range(factors['flow1'], -0.5, 0.5, 0, 1)

        # flow2: OBV 5日变化率(%), -50%~50% 映射到 0-1
        factor_norm['flow2'] = self._map_range(factors['flow2'], -50, 50, 0, 1)

        # pos1: -0.65~-0.05 映射到 0-1（-|pctB-0.65|，范围-0.65到0）
        factor_norm['pos1'] = self._map_range(factors['pos1'], -0.65, 0, 0, 1)

        # pos2: -0.03~-0.01 映射到 0-1（-|ATR%-0.03|，范围-0.03到0）
        factor_norm['pos2'] = self._map_range(factors['pos2'], -0.03, 0, 0, 1)

        # bias20: 越接近0越好，超过8%归零
        # 评分公式: score = max(0, 1 - abs(bias20) / 0.08)
        bias_raw = factors['bias20']
        factor_norm['bias20'] = max(0.0, 1.0 - abs(bias_raw) / 0.08)

        # 加权求和
        score = sum(factor_norm[k] * self.FACTOR_WEIGHTS[k] for k in factor_norm)

        # 应用BIAS6超买惩罚
        score = self._apply_bias6_penalty(score, factors)

        return round(score * 100, 2), factor_norm

    def _apply_bias6_penalty(self, score: float, factors: Dict) -> float:
        """
        应用BIAS(6)超买非线性惩罚

        当 BIAS(6) > 5% 时，应用指数惩罚函数：
        penalty = 1 - exp(-3 * (bias6 - 0.05))
        最终得分 = 原始得分 * (1 - penalty * 0.5)
        """
        bias6 = factors.get('bias6', 0)

        if bias6 > 0.05:  # BIAS6 > 5%
            # 非线性惩罚函数，越偏离5%惩罚越重
            excess = bias6 - 0.05
            penalty = 1 - np.exp(-10 * excess)
            # 最大惩罚50%
            score = score * (1 - penalty * 0.5)

        return score

    def _apply_overbought_oversold_penalty(self, score: float,
                                             latest: pd.Series) -> Tuple[float, List[str]]:
        """
        根据超买超卖程度调整评分

        解决问题：原有评分系统对超买超卖状态惩罚不足

        Args:
            score: 原始评分（0-100）
            latest: 最新数据Series

        Returns:
            (调整后分数, [警告信息列表])
        """
        warnings = []

        wr = latest.get('wr_14', latest.get('wr', 50))
        cci = latest.get('cci', 0)
        rsi = latest.get('rsi_24', 50)
        close = latest.get('close', 0)
        boll_upper = latest.get('boll_upper', 0)
        boll_lower = latest.get('boll_lower', 0)
        boll_pctb = (close - boll_lower) / (boll_upper - boll_lower) if boll_upper > boll_lower else 0.5

        # 极端超买判断（WR<10, CCI>200, RSI>80, 布林上轨外）
        is_extreme_overbought = (
            wr < 10 or
            cci > 200 or
            rsi > 80 or
            boll_pctb > 0.95
        )

        # 普通超买判断
        is_overbought = (
            wr < 20 or
            cci > 100 or
            rsi > 70 or
            boll_pctb > 0.85
        )

        # 极端超卖判断
        is_extreme_oversold = (
            wr > 90 or
            cci < -200 or
            rsi < 20 or
            boll_pctb < 0.05
        )

        # 普通超卖判断
        is_oversold = (
            wr > 80 or
            cci < -100 or
            rsi < 30 or
            boll_pctb < 0.15
        )

        # 应用惩罚/加成
        if is_extreme_overbought:
            score *= 0.5  # 极端超买打5折
            warnings.append(f"⚠️ 极端超买状态（WR={wr:.1f}, CCI={cci:.1f}, RSI={rsi:.1f}），评分已大幅下调")
        elif is_overbought:
            score *= 0.75  # 普通超买打75折
            warnings.append(f"⚠️ 技术指标超买（WR={wr:.1f}, CCI={cci:.1f}, RSI={rsi:.1f}），评分已下调")
        elif is_extreme_oversold:
            # 极端超卖可能存在反弹机会，但需要谨慎
            warnings.append(f"💡 极端超卖状态（WR={wr:.1f}, CCI={cci:.1f}, RSI={rsi:.1f}），可能存在反弹机会")
            # 不调整分数，让用户自行判断
        elif is_oversold:
            warnings.append(f"📊 技术指标超卖（WR={wr:.1f}, CCI={cci:.1f}, RSI={rsi:.1f}），关注反弹信号")

        return round(score, 2), warnings

    def _normalize_factors_in_pool(self, results: List[Dict]) -> List[Dict]:
        """
        在股票池中对因子进行rank_pct归一化
        """
        if len(results) < 2:
            return results

        # 收集所有股票的因子值
        factor_names = ['trend1', 'trend2', 'mom1', 'mom2', 'flow1', 'flow2', 'pos1', 'pos2', 'bias20', 'bias6']
        factor_values = {name: [] for name in factor_names}

        for r in results:
            for name in factor_names:
                factor_values[name].append(r['factors_raw'][name])

        # 对每个因子进行rank_pct
        factor_ranks = {}
        for name in factor_names:
            s = pd.Series(factor_values[name])
            factor_ranks[name] = s.rank(pct=True).values

        # 更新每个结果的因子得分
        for i, r in enumerate(results):
            r['factors_score'] = {
                name: float(factor_ranks[name][i]) for name in factor_names
            }

        return results

    def _calculate_weighted_score(self, factor_scores: Dict) -> Tuple[float, Dict]:
        """计算加权得分"""
        score = sum(factor_scores[k] * self.FACTOR_WEIGHTS[k] for k in factor_scores)
        return score, factor_scores

    def _map_range(self, value: float, in_min: float, in_max: float,
                   out_min: float, out_max: float) -> float:
        """将值从一个范围映射到另一个范围"""
        if in_max == in_min:
            return (out_min + out_max) / 2
        mapped = (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
        return max(out_min, min(out_max, mapped))  # 裁剪到输出范围

    def _check_liquidity(self, df: pd.DataFrame, latest: pd.Series) -> Tuple[bool, str]:
        """
        检查流动性条件

        Returns:
            (是否通过, 未通过原因)
        """
        close = latest.get('close', 0)

        # 1. 20日平均成交额 >= 1亿元（千元单位）
        amt_ma20 = 0
        if 'amount' in df.columns and len(df) >= 20:
            amt_ma20 = df['amount'].tail(20).mean()
        elif 'amt' in df.columns and len(df) >= 20:
            amt_ma20 = df['amt'].tail(20).mean()

        if amt_ma20 < self.LIQUIDITY_THRESHOLDS['min_amt_ma20']:
            return False, f"20日日均成交额{amt_ma20:.0f}千元低于{self.LIQUIDITY_THRESHOLDS['min_amt_ma20']}千元"

        # 2. ATR% >= 1.5%
        atr = latest.get('atr_14', 0)
        atrp = atr / close if close > 0 else 0
        if atrp < self.LIQUIDITY_THRESHOLDS['min_atrp']:
            return False, f"ATR%{atrp*100:.2f}%低于{self.LIQUIDITY_THRESHOLDS['min_atrp']*100}%"

        return True, ""

    def _check_bias_filter(self, latest: pd.Series) -> Tuple[bool, str]:
        """
        乖离率硬过滤检查（调整版：提高到15%，让位置修正系数处理超买）

        Returns:
            (是否通过, 未通过原因)
        """
        close = latest.get('close', 0)
        ma20 = latest.get('ma_20', 0)

        if ma20 <= 0:
            return False, "MA20数据无效"

        bias20 = close / ma20 - 1

        # 硬过滤: bias20 > +15% 剔除（极端追高风险）
        # 将阈值从8%提高到15%，让位置修正系数处理8-15%区间的超买惩罚
        if bias20 > 0.15:
            return False, f"乖离率+{bias20*100:.2f}%超过+15%阈值，极端追高风险"

        return True, ""

    def _calculate_dynamic_stop_loss(self, latest: pd.Series,
                                      buy_price: float) -> Tuple[float, float, str]:
        """
        计算动态止损价和止损比例

        当检测到ATR较低（<2%）且布林带带宽较窄（<5%）时，
        自动切换为紧止损模式（2%或跌破MA50）。

        Returns:
            (stop_price, stop_loss_pct, stop_loss_type)
            stop_loss_type: 'tight' | 'normal' | 'ma50'
        """
        close = latest.get('close', 0)
        atr = latest.get('atr_14', 0)
        atrp = atr / close if close > 0 else 0

        # 计算布林带带宽
        boll_upper = latest.get('boll_upper', close)
        boll_lower = latest.get('boll_lower', close)
        boll_mid = latest.get('boll_mid', close)
        boll_bandwidth = (boll_upper - boll_lower) / boll_mid if boll_mid > 0 else 0

        # 低波动压缩判断
        is_low_volatility = (atrp < self.STOP_LOSS_CONFIG['atr_low_threshold'] and
                             boll_bandwidth < self.STOP_LOSS_CONFIG['boll_bandwidth_threshold'])

        if is_low_volatility:
            # 紧止损模式
            # 方案1: 2%固定止损
            tight_stop_pct = self.STOP_LOSS_CONFIG['tight_stop_loss_pct']
            tight_stop = buy_price * (1 - tight_stop_pct)

            # 方案2: MA50止损（如果MA50高于2%止损价，使用更紧的止损）
            ma50 = latest.get('ma_50', 0)
            if ma50 > 0 and self.STOP_LOSS_CONFIG['ma50_stop_loss']:
                ma50_stop = ma50 * 0.98  # MA50下方2%
                # 选择较高的止损价（更紧的止损，保护更多资金）
                if ma50_stop > tight_stop:
                    stop_loss_pct = (buy_price - ma50_stop) / buy_price
                    return ma50_stop, stop_loss_pct, 'ma50'

            return tight_stop, tight_stop_pct, 'tight'

        # 正常止损模式（使用默认止损比例）
        normal_stop_pct = getattr(self, 'stop_loss_pct', 0.05)
        normal_stop = buy_price * (1 - normal_stop_pct)
        return normal_stop, normal_stop_pct, 'normal'

    def _analyze_volatility_compression(self, latest: pd.Series) -> Optional[str]:
        """
        分析波动率压缩状态，提供变盘预警

        Returns:
            波动率预警描述，如果不是压缩状态则返回 None
        """
        close = latest.get('close', 0)
        atr = latest.get('atr_14', 0)
        atrp = atr / close if close > 0 else 0

        # 计算布林带带宽和位置
        boll_upper = latest.get('boll_upper', close)
        boll_lower = latest.get('boll_lower', close)
        boll_mid = latest.get('boll_mid', close)

        if boll_mid <= 0:
            return None

        boll_bandwidth = (boll_upper - boll_lower) / boll_mid
        boll_position = (close - boll_lower) / (boll_upper - boll_lower) if boll_upper != boll_lower else 0.5

        # 低波动压缩判断
        is_low_atr = atrp < 0.02  # ATR% < 2%
        is_narrow_boll = boll_bandwidth < 0.05  # 布林带带宽 < 5%

        if is_low_atr and is_narrow_boll:
            # 计算关键突破位（上下轨外1%）
            breakout_up = boll_upper * 1.01
            breakout_down = boll_lower * 0.99

            # 股价位置描述
            if boll_position > 0.7:
                position_desc = "上轨附近"
            elif boll_position < 0.3:
                position_desc = "下轨附近"
            else:
                position_desc = "中轨附近"

            return (f"📊 **波动率预警**：当前处于低波动压缩期（ATR%{atrp*100:.1f}%，带宽{boll_bandwidth*100:.1f}%），"
                    f"预计即将变盘。股价位于{position_desc}，"
                    f"建议等待放量突破¥{breakout_up:.2f}或跌破¥{breakout_down:.2f}后再跟随方向。")

        return None

    def _detect_triggers(self, df: pd.DataFrame, latest: pd.Series) -> Tuple[str, str]:
        """
        检测双触发信号（改进版：增加位置过滤）

        Returns:
            (触发类型: breakout/pullback/none, 触发描述)
        """
        close = latest.get('close', 0)
        low = latest.get('low', 0)
        open_price = latest.get('open', close)
        ma10 = latest.get('ma_10', 0)
        ma20 = latest.get('ma_20', 0)

        # 计算20日最高收盘价（不含当天）
        hh_close20 = 0
        if len(df) >= 21:
            hh_close20 = df['close'].iloc[-21:-1].max()

        # 计算成交量比
        vol = latest.get('volume', 0)
        vol_ma5 = df['volume'].tail(5).mean() if len(df) >= 5 else vol
        vol_ratio = vol / vol_ma5 if vol_ma5 > 0 else 1.0

        # 计算股价相对于60日高低点的位置（用于位置过滤）
        position_ratio = 0.5  # 默认中间位置
        if len(df) >= 60:
            high_60 = df['high'].iloc[-60:].max()
            low_60 = df['low'].iloc[-60:].min()
            if high_60 > low_60:
                position_ratio = (close - low_60) / (high_60 - low_60)

        # 突破型信号（增加位置过滤）
        # 条件1: close > hh_close20（突破20日最高收盘价）
        # 条件2: vol_ratio >= 1.2（放量）- 高位突破需要更强量能
        # 条件3: close > ma20（趋势向上）
        # 条件4（新增）: 位置过滤
        breakout_cond1 = close > hh_close20 if hh_close20 > 0 else False
        breakout_cond3 = close > ma20 if ma20 > 0 else False

        # 根据位置调整量能要求
        if position_ratio > 0.7:  # 高位突破（前30%）
            breakout_cond2 = vol_ratio >= 1.5  # 需要更强的量能确认
            position_desc = f"高位突破（位置{position_ratio:.0%}）"
        elif position_ratio > 0.4:  # 中位突破
            breakout_cond2 = vol_ratio >= 1.3
            position_desc = f"中位突破（位置{position_ratio:.0%}）"
        else:  # 低位突破
            breakout_cond2 = vol_ratio >= 1.2
            position_desc = f"低位突破（位置{position_ratio:.0%}）"

        if breakout_cond1 and breakout_cond2 and breakout_cond3:
            return 'breakout', f"{position_desc}，突破20日新高（{close:.2f}>{hh_close20:.2f}）+ 放量（{vol_ratio:.2f}倍）"

        # 回踩型信号
        # 条件1: low <= ma10（回踩MA10）
        # 条件2: close >= ma10（收在MA10上方）
        # 条件3: vol < vol_ma5（缩量）
        # 条件4: close > open（阳线）
        pullback_cond1 = low <= ma10 if ma10 > 0 else False
        pullback_cond2 = close >= ma10 if ma10 > 0 else False
        pullback_cond3 = vol < vol_ma5
        pullback_cond4 = close > open_price

        if pullback_cond1 and pullback_cond2 and pullback_cond3 and pullback_cond4:
            return 'pullback', f"回踩MA10（最低{low:.2f}<=MA10:{ma10:.2f}）+ 缩量收阳"

        return 'none', "无触发信号"

    def _calculate_execution_info(self, latest: pd.Series, open_T1: Optional[float],
                                  score: float, trigger_type: str, factors: Dict = None) -> Dict:
        """
        计算交易执行信息（改进版：动态止损 + 熔断前置修正）
        """
        close_T = float(latest.get('close', 0))

        # 买入价：优先使用T+1开盘价，否则用T日收盘价估算
        if open_T1 is not None and open_T1 > 0:
            buy_price = open_T1
        else:
            # 如果没有T+1开盘价，使用T日收盘价作为估计
            buy_price = close_T

        # 止损价：使用动态止损计算
        stop_price, stop_loss_pct, stop_loss_type = self._calculate_dynamic_stop_loss(
            latest, buy_price
        )

        # 根据止损类型生成止损说明
        stop_loss_desc = f"{stop_loss_pct*100:.0f}%"
        if stop_loss_type == 'tight':
            stop_loss_desc = f"紧止损{stop_loss_pct*100:.0f}%（低波动模式）"
        elif stop_loss_type == 'ma50':
            stop_loss_desc = f"MA50止损（约{stop_loss_pct*100:.1f}%）"

        # 关键修复：先生成操作指引（包含熔断判断）
        action_guide = self._generate_action_guide(score, trigger_type, factors, latest)

        # 关键修复：仓位建议考虑熔断状态
        position_suggest = self._get_position_suggestion(score, trigger_type, action_guide)

        return {
            'buy_price': round(buy_price, 2),
            'stop_price': round(stop_price, 2),
            'stop_loss_pct': stop_loss_pct,  # 使用动态计算的止损比例
            'stop_loss_type': stop_loss_type,  # 新增：止损类型
            'stop_loss_desc': stop_loss_desc,  # 新增：止损说明
            'position_suggest': position_suggest,
            'expected_return_5pct': round((close_T * 1.05 - buy_price) / buy_price * 100, 2) if buy_price > 0 else 0,
            'action_guide': action_guide,
        }

    def _generate_action_guide(self, score: float, trigger_type: str,
                               factors: Dict, latest: pd.Series) -> str:
        """
        生成操作指引文本（增强版：高位形态熔断机制）

        核心规则：
        1. 高位形态熔断：高位 + 看跌K线形态 = 强制观望/卖出
        2. 超买状态：限制追涨建议，提示回调风险
        3. 超卖状态：提示反弹机会
        4. 正常状态：按评分阈值判断
        """
        # 获取BIAS6
        bias6 = 0
        if factors and 'bias6' in factors:
            bias6 = factors['bias6'] * 100  # 转换为百分比
        else:
            bias6 = latest.get('bias_6', 0)

        # 获取bias20用于高位判断
        bias20 = 0
        if factors and 'bias20' in factors:
            bias20 = factors.get('bias20', 0)
        elif factors and 'aux_factors' in factors:
            bias20 = factors['aux_factors'].get('bias20', 0)

        # ========== 新增：判断超买超卖状态 ==========
        wr = latest.get('wr_14', latest.get('wr', 50))
        cci = latest.get('cci', 0)
        rsi = latest.get('rsi_24', 50)
        close = latest.get('close', 0)
        boll_upper = latest.get('boll_upper', 0)
        boll_lower = latest.get('boll_lower', 0)
        boll_pctb = (close - boll_lower) / (boll_upper - boll_lower) if boll_upper > boll_lower else 0.5

        # 极端超买判断
        is_extreme_overbought = (
            wr < 10 or
            cci > 200 or
            rsi > 80 or
            boll_pctb > 0.95 or
            bias20 > 0.08  # BIAS20 > 8%
        )

        # 普通超买判断
        is_overbought = (
            wr < 20 or
            cci > 100 or
            rsi > 70 or
            boll_pctb > 0.85 or
            bias20 > 0.05  # BIAS20 > 5%
        )

        # 极端超卖判断
        is_extreme_oversold = (
            wr > 90 or
            cci < -200 or
            rsi < 20 or
            boll_pctb < 0.05
        )

        # 普通超卖判断
        is_oversold = (
            wr > 80 or
            cci < -100 or
            rsi < 30 or
            boll_pctb < 0.15
        )

        # ========== 核心新增：高位形态熔断机制 ==========
        # 获取K线形态详情
        candlestick_detail = factors.get('candlestick_detail', {}) if factors else {}

        # 判断是否有看跌形态
        has_bearish_pattern = False
        has_strong_bearish = False  # 强力看跌形态（如大阴线、看跌吞没）
        bearish_pattern_names = []
        # 明确的看跌形态列表
        strong_bearish_patterns = ['大阴线', '看跌吞没', '暮星', '黑色三鸦']
        medium_bearish_patterns = ['流星线', '吊颈线', '乌云盖顶']

        patterns = candlestick_detail.get('patterns', [])
        position_zone = candlestick_detail.get('position_zone', '')

        for p in patterns:
            pattern_name = p.get('name', '')
            pattern_type = p.get('type', '')
            pattern_strength = p.get('strength', '中')

            # 只有type=bearish的才是看跌形态
            if pattern_type == 'bearish':
                has_bearish_pattern = True
                bearish_pattern_names.append(pattern_name)

                # 判断是否强力看跌
                if pattern_name in strong_bearish_patterns:
                    has_strong_bearish = True

        # 判断是否高位
        is_high_position = (
            bias20 > 0.05 or  # BIAS20 > 5%
            boll_pctb > 0.75 or  # 布林带上轨区域
            position_zone == 'high_position'
        )

        # 判断是否有看涨形态（用于诱多判断）
        has_bullish_pattern = False
        for p in patterns:
            if p.get('type') == 'bullish':
                has_bullish_pattern = True
                break

        # ========== 熔断规则（优先级从高到低）==========

        # 规则1：强力看跌形态（大阴线等）- 无视位置，直接警告
        if has_strong_bearish:
            pattern_str = '、'.join(bearish_pattern_names[:2])
            return f"🚫【熔断-回避】出现{pattern_str}等强力看跌形态，无视趋势评分，建议观望或卖出"

        # 规则2：高位 + 看跌形态 = 强制回避
        if is_high_position and has_bearish_pattern:
            pattern_str = '、'.join(bearish_pattern_names[:2]) if bearish_pattern_names else '看跌形态'
            return f"🚫【熔断-回避】高位出现{pattern_str}，无视趋势评分，建议观望或卖出"

        # 规则3：高位 + 看涨形态 + 极端超买 = 诱多陷阱
        if is_extreme_overbought and has_bullish_pattern:
            assessment = candlestick_detail.get('assessment', '')
            if '诱多' in assessment or '力竭' in assessment:
                return f"🚫【熔断-诱多】高位看涨形态可能是诱多陷阱，建议回避"

        # 规则4：有看跌形态（非强力）+ 评分较高 = 警告
        if has_bearish_pattern and score >= 60:
            pattern_str = '、'.join(bearish_pattern_names[:2])
            return f"⚠️【风险警告】出现{pattern_str}，趋势评分可能失效，建议谨慎"

        # ========== 超买状态下的特殊处理（优先级最高）==========
        if is_extreme_overbought:
            if score >= 80:
                return f"⚠️ 谨慎持有 - 评分优秀但极端超买（WR={wr:.1f},BIAS20={bias20*100:.1f}%），建议逢高减仓"
            elif score >= 60:
                return f"🚫 不宜追高 - 技术指标极端超买（WR={wr:.1f},BIAS20={bias20*100:.1f}%），等待回调"
            else:
                return "❌ 建议规避 - 评分一般且极端超买，风险极大"

        if is_overbought:
            if score >= 80:
                return f"⏳ 不宜追涨 - 评分优秀但超买（WR={wr:.1f},RSI={rsi:.1f}），等待回调再介入"
            elif score >= 60:
                return f"👀 观望等待 - 技术指标超买（WR={wr:.1f},RSI={rsi:.1f}），不宜追高"
            else:
                return "❌ 不建议操作 - 评分一般且超买，建议观望"

        # ========== 超卖状态下的特殊处理 ==========
        if is_extreme_oversold:
            if score >= 60:
                return f"💡 关注低吸 - 极端超卖（WR={wr:.1f},CCI={cci:.1f}），可能存在反弹机会"
            elif score >= 40:
                return f"👀 观察等待 - 极端超卖但评分一般，等待确认信号"

        if is_oversold:
            if score >= 70:
                return f"📉 分批低吸 - 超卖区域评分良好，可考虑轻仓布局"
            elif score >= 50:
                return f"👀 关注反弹 - 超卖状态，等待企稳信号"

        # ========== 正常状态下的原有逻辑 ==========
        # 计算调整后的评分（与仓位建议逻辑一致）
        trigger_bonus = 0
        if trigger_type == 'breakout':
            trigger_bonus = 10
        elif trigger_type == 'pullback':
            trigger_bonus = 5

        adjusted_score = score + trigger_bonus

        # 根据调整后的评分和操作指引逻辑生成建议
        if adjusted_score >= 80:
            # 高评分 + 触发信号 = 积极买入
            if bias6 < 3:
                return "✅ 积极买入 - 评分优秀且乖离率适中，建议建仓"
            else:
                return "⏳ 分批建仓 - 评分优秀但乖离率偏高，建议分批买入"

        elif adjusted_score >= 70:
            # 中高评分 = 试探性买入
            if trigger_type == 'breakout':
                return "📈 轻仓跟进 - 突破信号确认，可小仓位参与"
            elif trigger_type == 'pullback':
                return "📉 关注低吸 - 回踩确认，观察支撑位后轻仓介入"
            elif bias6 < 3:
                return "👀 轻仓试探 - 评分良好且乖离率适中，可小仓位试探"
            else:
                return "⏳ 观望等待 - 评分良好但无明确信号且乖离率偏高，建议等待"

        elif adjusted_score >= 60:
            # 及格评分 = 观望为主
            return "👀 观望为主 - 评分及格但信号不够明确，建议保持关注"

        elif score >= 40:
            # 低评分 = 不建议操作
            return "❌ 不建议操作 - 评分一般，不符合买入条件"

        else:
            # 很差评分 = 明确回避
            return "🚫 回避 - 评分较低，建议规避或减仓"

    def _get_position_suggestion(self, score: float, trigger_type: str, action_guide: str = None) -> str:
        """
        根据得分、触发类型和熔断状态给出仓位建议

        关键修复：熔断触发时，强制返回空仓建议
        """
        # 关键修复：如果熔断触发，强制返回空仓建议
        if action_guide and '熔断' in action_guide:
            return "0%（建议空仓观望）"

        # 关键修复：如果风险警告，限制仓位
        if action_guide and '风险警告' in action_guide:
            return "0-10%（高风险，谨慎参与）"

        # 触发类型有加分
        trigger_bonus = 0
        if trigger_type == 'breakout':
            trigger_bonus = 10
        elif trigger_type == 'pullback':
            trigger_bonus = 5

        adjusted_score = score + trigger_bonus

        if adjusted_score >= 80:
            return "50-70%"
        elif adjusted_score >= 60:
            return "30-50%"
        elif adjusted_score >= 40:
            return "10-30%"
        else:
            return "不建议买入"

    def _get_score_grade(self, score: float) -> str:
        """
        将分数转换为等级
        """
        if score >= 80:
            return "优秀"
        elif score >= 60:
            return "良好"
        elif score >= 40:
            return "一般"
        elif score >= 20:
            return "较差"
        else:
            return "很差"

    def _collect_warnings(self, factors: Dict, latest: pd.Series) -> List[str]:
        """收集警告信息（改进版：增加波动率预警和K线形态评估）"""
        warnings = []

        # RSI过高或过低警告
        rsi = latest.get('rsi_24', 50)
        if rsi > 70:
            warnings.append(f"RSI超买（{rsi:.1f}）")
        elif rsi < 30:
            warnings.append(f"RSI超卖（{rsi:.1f}）")

        # 乖离率警告 - 使用factors中的bias6（小数形式）和latest中的bias_6（百分比形式）
        # factors['bias6'] 是小数形式 (如 0.027 表示 2.7%)
        # latest['bias_6'] 是百分比形式 (如 2.7 表示 2.7%)
        bias6_factor = factors.get('bias6', 0)  # 小数形式
        bias6_pct = bias6_factor * 100  # 转换为百分比

        if bias6_pct > 5:
            warnings.append(f"正乖离过大（BIAS6: {bias6_pct:.1f}%）")
        elif bias6_pct < -5:
            warnings.append(f"负乖离过大（BIAS6: {bias6_pct:.1f}%）")

        # 同时检查BIAS(20)
        bias20_factor = factors.get('bias20', 0)
        bias20_pct = bias20_factor * 100
        if abs(bias20_pct) > 8:
            warnings.append(f"MA20乖离率异常（{bias20_pct:+.1f}%）")

        # 成交量异常（兼容新旧因子名）
        vol_ratio = factors.get('volume_ratio', factors.get('flow1', 1.0))
        if vol_ratio > 2.0:
            warnings.append("成交量异常放大")

        # === 新增：波动率压缩预警 ===
        volatility_warning = self._analyze_volatility_compression(latest)
        if volatility_warning:
            warnings.append(volatility_warning)

        # === 新增：K线形态评估警告 ===
        candlestick_detail = factors.get('candlestick_detail', {})
        if candlestick_detail:
            assessment = candlestick_detail.get('assessment', '')
            if assessment:
                warnings.append(f"📊 K线形态：{assessment}")
            # 如果形态评分很低（<40），添加额外警告
            patterns = candlestick_detail.get('patterns', [])
            position_zone = candlestick_detail.get('position_zone', '')
            for p in patterns:
                if p.get('final_score', 0) < 0:
                    warnings.append(f"⚠️ 高位出现{p['name']}，可能是诱多信号")

        return warnings

    def format_score_report(self, score_result: Dict) -> str:
        """
        格式化打分报告为 Markdown 格式
        """
        if "error" in score_result:
            return f"\n**打分失败：** {score_result['error']}\n"

        lines = []
        lines.append("\n### 股票评分报告（百分制）")
        lines.append("")

        # 基本信息
        lines.append(f"**股票代码：** {score_result.get('stock_code', 'N/A')}")
        lines.append(f"**交易日期：** T日 {score_result.get('trade_date_T', 'N/A')} → T+1日 {score_result.get('trade_date_T1', 'N/A')}")
        lines.append("")

        # 评分
        score = score_result['score']
        grade = score_result['score_grade']
        trigger_type = score_result['trigger_type']
        trigger_detail = score_result['trigger_detail']

        lines.append(f"**综合评分：{score:.1f} 分（{grade}）**")
        lines.append("")

        # 触发信号
        if trigger_type != 'none':
            trigger_icon = "📈" if trigger_type == 'breakout' else "📉"
            lines.append(f"{trigger_icon} **触发信号：** {trigger_detail}")
            lines.append("")

        # 交易执行计划
        execution = score_result.get('execution', {})
        lines.append("#### 交易执行计划")
        lines.append("")
        lines.append(f"| 项目 | 数值 |")
        lines.append(f"|------|------|")
        lines.append(f"| T日收盘价 | ¥{score_result.get('close_T', 0):.2f} |")
        lines.append(f"| T+1买入价 | ¥{execution.get('buy_price', 0):.2f} |")
        # 改进：使用动态止损描述
        stop_loss_pct = execution.get('stop_loss_pct', 0.05)
        stop_loss_type = execution.get('stop_loss_type', 'normal')
        stop_loss_desc = execution.get('stop_loss_desc', f"{stop_loss_pct*100:.0f}%")

        # 根据止损类型显示不同的格式
        if stop_loss_type == 'tight':
            stop_display = f"紧止损 {stop_loss_pct*100:.0f}%"
        elif stop_loss_type == 'ma50':
            stop_display = f"MA50止损 {stop_loss_pct*100:.1f}%"
        else:
            stop_display = f"{stop_loss_pct*100:.0f}%"

        lines.append(f"| 止损价（{stop_display}） | ¥{execution.get('stop_price', 0):.2f} |")
        lines.append(f"| 建议仓位 | {execution.get('position_suggest', 'N/A')} |")
        lines.append("")

        # 操作指引
        action_guide = execution.get('action_guide', '')
        if action_guide:
            lines.append(f"**🎯 操作指引：** {action_guide}")
            lines.append("")

        # 因子得分表
        lines.append("#### 因子得分详情")
        lines.append("")
        lines.append("| 因子 | 权重 | 原始值 | 得分 | 说明 |")
        lines.append("|------|------|--------|------|------|")

        factor_info = {
            'trend1': ('趋势偏离', '股价相对MA20位置'),
            'trend2': ('均线斜率', '短期趋势强度'),
            'mom1': ('MACD动量', 'MACD柱状图变化'),
            'mom2': ('RSI位置', '负值代表向上空间大'),
            'flow1': ('成交量比', '成交量相对5日均值'),
            'flow2': ('OBV资金流', 'OBV相对5日变化率%'),
            'pos1': ('布林带位置', '价格在布林带中的位置'),
            'pos2': ('波动率位置', 'ATR%偏离3%的程度'),
            'bias20': ('MA20乖离率', '越接近0越好'),
        }

        factors_raw = score_result.get('factors_raw', {})
        factors_score = score_result.get('factors_score', {})

        for key in self.FACTOR_WEIGHTS:
            weight = self.FACTOR_WEIGHTS[key] * 100
            raw = factors_raw.get(key, 0)
            score_val = factors_score.get(key, 0)
            name, desc = factor_info.get(key, (key, '-'))
            lines.append(f"| {name} | {weight:.0f}% | {raw:.4f} | {score_val*100:.1f} | {desc} |")

        lines.append("")

        # 警告信息
        warnings = score_result.get('warnings', [])
        if warnings:
            lines.append("⚠️ **风险提示：**")
            for warning in warnings:
                lines.append(f"- {warning}")
            lines.append("")

        return "\n".join(lines)
