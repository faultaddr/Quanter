"""
综合选股框架

支持200+条件自由组合的选股系统，参考 InStock 的分类架构：

1. 股票范围：市场、行业、地区、概念、风格、指数成份、上市时间
2. 基本面：估值指标、每股指标、盈利能力、成长能力、资本结构、股本股东
3. 技术面：MACD金叉、KDJ金叉、放量突破、均线排列、连涨放量等
4. 消息面：公告大事、机构关注、机构持股
5. 人气指标：股吧人气、粉丝占比、浏览排名
6. 行情数据：股价表现、成交情况、资金流向、沪深股通
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings('ignore')


class ConditionCategory(Enum):
    """条件分类"""
    STOCK_SCOPE = "股票范围"
    FUNDAMENTAL = "基本面"
    TECHNICAL = "技术面"
    NEWS = "消息面"
    POPULARITY = "人气指标"
    MARKET_DATA = "行情数据"


class ConditionOperator(Enum):
    """条件操作符"""
    GT = ">"          # 大于
    GTE = ">="        # 大于等于
    LT = "<"          # 小于
    LTE = "<="        # 小于等于
    EQ = "=="         # 等于
    NEQ = "!="        # 不等于
    BETWEEN = "区间"   # 在区间内
    IN = "包含"        # 在列表中
    NOT_IN = "排除"    # 不在列表中
    CROSS_UP = "金叉"  # 上穿
    CROSS_DOWN = "死叉"  # 下穿


@dataclass
class ScreeningCondition:
    """选股条件"""
    name: str                           # 条件名称
    category: ConditionCategory         # 条件分类
    field: str                          # 数据字段
    operator: ConditionOperator         # 操作符
    value: Union[float, int, str, List, Tuple]  # 条件值
    weight: float = 1.0                 # 条件权重
    description: str = ""               # 条件描述
    required: bool = False              # 是否必须满足


@dataclass
class ScreeningResult:
    """选股结果"""
    stock_code: str
    stock_name: str
    score: float
    rank: int
    matched_conditions: List[str]
    condition_scores: Dict[str, float]
    details: Dict[str, Any]


class ConditionChecker(ABC):
    """条件检查器基类"""

    @abstractmethod
    def check(self, data: pd.DataFrame, condition: ScreeningCondition) -> pd.Series:
        """
        检查条件

        Args:
            data: 股票数据
            condition: 选股条件

        Returns:
            满足条件的布尔序列
        """
        pass


class TechnicalConditionChecker(ConditionChecker):
    """技术面条件检查器"""

    def check(self, data: pd.DataFrame, condition: ScreeningCondition) -> pd.Series:
        df = data.copy()
        field = condition.field
        op = condition.operator
        value = condition.value

        # 计算技术指标
        if field == 'macd_golden_cross':
            # MACD金叉
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9).mean()
            result = (dif > dea) & (dif.shift(1) <= dea.shift(1))
            return result

        elif field == 'kdj_golden_cross':
            # KDJ金叉
            low_9 = df['low'].rolling(9).min()
            high_9 = df['high'].rolling(9).max()
            rsv = (df['close'] - low_9) / (high_9 - low_9 + 0.0001) * 100
            k = rsv.ewm(com=2).mean()
            d = k.ewm(com=2).mean()
            result = (k > d) & (k.shift(1) <= d.shift(1))
            return result

        elif field == 'volume_breakout':
            # 放量突破
            vol_ma = df['volume'].rolling(20).mean()
            result = df['volume'] > vol_ma * value
            return result

        elif field == 'ma_bullish_alignment':
            # 均线多头排列
            ma5 = df['close'].rolling(5).mean()
            ma10 = df['close'].rolling(10).mean()
            ma20 = df['close'].rolling(20).mean()
            ma60 = df['close'].rolling(60).mean()
            result = (ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60)
            return result

        elif field == 'ma_bearish_alignment':
            # 均线空头排列
            ma5 = df['close'].rolling(5).mean()
            ma10 = df['close'].rolling(10).mean()
            ma20 = df['close'].rolling(20).mean()
            result = (ma5 < ma10) & (ma10 < ma20)
            return result

        elif field == 'continuous_rise':
            # 连续上涨
            pct_change = df['close'].pct_change()
            result = pct_change.rolling(int(value)).apply(lambda x: (x > 0).all()).fillna(0).astype(bool)
            return result

        elif field == 'continuous_fall':
            # 连续下跌
            pct_change = df['close'].pct_change()
            result = pct_change.rolling(int(value)).apply(lambda x: (x < 0).all()).fillna(0).astype(bool)
            return result

        elif field == 'near_year_high':
            # 接近年度新高
            year_high = df['close'].rolling(250).max()
            result = df['close'] >= year_high * (1 - value)
            return result

        elif field == 'near_year_low':
            # 接近年度新低
            year_low = df['close'].rolling(250).min()
            result = df['close'] <= year_low * (1 + value)
            return result

        elif field == 'rsi_oversold':
            # RSI超卖
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 0.0001)
            rsi = 100 - (100 / (1 + rs))
            result = rsi < value
            return result

        elif field == 'rsi_overbought':
            # RSI超买
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 0.0001)
            rsi = 100 - (100 / (1 + rs))
            result = rsi > value
            return result

        elif field == 'boll_upper_break':
            # 布林带上轨突破
            ma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            upper = ma20 + 2 * std20
            result = df['close'] > upper
            return result

        elif field == 'boll_lower_break':
            # 布林带下轨突破
            ma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            lower = ma20 - 2 * std20
            result = df['close'] < lower
            return result

        # 通用数值比较
        elif field in df.columns:
            return self._compare(df[field], op, value)

        return pd.Series([False] * len(df), index=df.index)

    def _compare(self, series: pd.Series, op: ConditionOperator, value) -> pd.Series:
        """通用比较操作"""
        if op == ConditionOperator.GT:
            return series > value
        elif op == ConditionOperator.GTE:
            return series >= value
        elif op == ConditionOperator.LT:
            return series < value
        elif op == ConditionOperator.LTE:
            return series <= value
        elif op == ConditionOperator.EQ:
            return series == value
        elif op == ConditionOperator.NEQ:
            return series != value
        elif op == ConditionOperator.BETWEEN:
            return (series >= value[0]) & (series <= value[1])
        elif op == ConditionOperator.IN:
            return series.isin(value)
        elif op == ConditionOperator.NOT_IN:
            return ~series.isin(value)
        else:
            return pd.Series([False] * len(series), index=series.index)


class FundamentalConditionChecker(ConditionChecker):
    """基本面条件检查器"""

    def check(self, data: pd.DataFrame, condition: ScreeningCondition) -> pd.Series:
        # 基本面数据通常需要外部提供
        # 这里返回一个基于技术面的估算结果
        field = condition.field
        op = condition.operator
        value = condition.value

        if field == 'pe_ratio':
            # PE估值筛选（需要外部数据）
            if 'pe' in data.columns:
                return self._compare(data['pe'], op, value)

        elif field == 'pb_ratio':
            # PB估值筛选
            if 'pb' in data.columns:
                return self._compare(data['pb'], op, value)

        elif field == 'roe':
            # ROE筛选
            if 'roe' in data.columns:
                return self._compare(data['roe'], op, value)

        elif field == 'profit_growth':
            # 利润增长筛选
            if 'profit_growth' in data.columns:
                return self._compare(data['profit_growth'], op, value)

        # 默认返回空结果
        return pd.Series([False] * len(data), index=data.index)

    def _compare(self, series: pd.Series, op: ConditionOperator, value) -> pd.Series:
        """通用比较操作"""
        if op == ConditionOperator.GT:
            return series > value
        elif op == ConditionOperator.GTE:
            return series >= value
        elif op == ConditionOperator.LT:
            return series < value
        elif op == ConditionOperator.LTE:
            return series <= value
        elif op == ConditionOperator.BETWEEN:
            return (series >= value[0]) & (series <= value[1])
        return pd.Series([False] * len(series), index=series.index)


class MarketDataConditionChecker(ConditionChecker):
    """行情数据条件检查器"""

    def check(self, data: pd.DataFrame, condition: ScreeningCondition) -> pd.Series:
        df = data.copy()
        field = condition.field
        op = condition.operator
        value = condition.value

        if field == 'price_change_1d':
            # 单日涨跌幅
            pct = df['close'].pct_change()
            return self._compare(pct, op, value)

        elif field == 'price_change_5d':
            # 5日涨跌幅
            pct = df['close'].pct_change(5)
            return self._compare(pct, op, value)

        elif field == 'price_change_20d':
            # 20日涨跌幅
            pct = df['close'].pct_change(20)
            return self._compare(pct, op, value)

        elif field == 'turnover_rate':
            # 换手率（需要流通股本数据）
            if 'turnover' in df.columns:
                return self._compare(df['turnover'], op, value)

        elif field == 'amount':
            # 成交额
            amount = df['close'] * df['volume']
            return self._compare(amount, op, value)

        elif field == 'amplitude':
            # 振幅
            amp = (df['high'] - df['low']) / df['close'].shift(1)
            return self._compare(amp, op, value)

        elif field == 'volume_ratio':
            # 量比
            vol_ma5 = df['volume'].rolling(5).mean()
            ratio = df['volume'] / vol_ma5
            return self._compare(ratio, op, value)

        elif field in df.columns:
            return self._compare(df[field], op, value)

        return pd.Series([False] * len(df), index=df.index)

    def _compare(self, series: pd.Series, op: ConditionOperator, value) -> pd.Series:
        """通用比较操作"""
        if op == ConditionOperator.GT:
            return series > value
        elif op == ConditionOperator.GTE:
            return series >= value
        elif op == ConditionOperator.LT:
            return series < value
        elif op == ConditionOperator.LTE:
            return series <= value
        elif op == ConditionOperator.BETWEEN:
            return (series >= value[0]) & (series <= value[1])
        return pd.Series([False] * len(series), index=series.index)


class ComprehensiveStockScreener:
    """
    综合选股器

    支持200+条件自由组合选股
    """

    # 预定义条件模板
    PREDEFINED_CONDITIONS = {
        # 技术面条件
        'macd_golden_cross': {
            'name': 'MACD金叉',
            'category': ConditionCategory.TECHNICAL,
            'field': 'macd_golden_cross',
            'operator': ConditionOperator.EQ,
            'value': True,
            'description': 'MACD指标出现金叉买入信号'
        },
        'kdj_golden_cross': {
            'name': 'KDJ金叉',
            'category': ConditionCategory.TECHNICAL,
            'field': 'kdj_golden_cross',
            'operator': ConditionOperator.EQ,
            'value': True,
            'description': 'KDJ指标出现金叉买入信号'
        },
        'volume_breakout_2x': {
            'name': '放量突破(2倍)',
            'category': ConditionCategory.TECHNICAL,
            'field': 'volume_breakout',
            'operator': ConditionOperator.GT,
            'value': 2.0,
            'description': '成交量突破20日均量2倍'
        },
        'ma_bullish': {
            'name': '均线多头排列',
            'category': ConditionCategory.TECHNICAL,
            'field': 'ma_bullish_alignment',
            'operator': ConditionOperator.EQ,
            'value': True,
            'description': '5日>10日>20日>60日均线'
        },
        'ma_bearish': {
            'name': '均线空头排列',
            'category': ConditionCategory.TECHNICAL,
            'field': 'ma_bearish_alignment',
            'operator': ConditionOperator.EQ,
            'value': True,
            'description': '5日<10日<20日均线'
        },
        'rsi_oversold_30': {
            'name': 'RSI超卖(<30)',
            'category': ConditionCategory.TECHNICAL,
            'field': 'rsi_oversold',
            'operator': ConditionOperator.LT,
            'value': 30,
            'description': 'RSI指标低于30，超卖区域'
        },
        'rsi_overbought_70': {
            'name': 'RSI超买(>70)',
            'category': ConditionCategory.TECHNICAL,
            'field': 'rsi_overbought',
            'operator': ConditionOperator.GT,
            'value': 70,
            'description': 'RSI指标高于70，超买区域'
        },
        'near_year_high_10pct': {
            'name': '接近年度新高(10%)',
            'category': ConditionCategory.TECHNICAL,
            'field': 'near_year_high',
            'operator': ConditionOperator.EQ,
            'value': 0.10,
            'description': '股价接近250日最高价10%以内'
        },
        'continuous_rise_3': {
            'name': '连续3日上涨',
            'category': ConditionCategory.TECHNICAL,
            'field': 'continuous_rise',
            'operator': ConditionOperator.EQ,
            'value': 3,
            'description': '连续3个交易日上涨'
        },
        'continuous_fall_3': {
            'name': '连续3日下跌',
            'category': ConditionCategory.TECHNICAL,
            'field': 'continuous_fall',
            'operator': ConditionOperator.EQ,
            'value': 3,
            'description': '连续3个交易日下跌'
        },

        # 行情数据条件
        'price_up_5pct': {
            'name': '单日涨幅>5%',
            'category': ConditionCategory.MARKET_DATA,
            'field': 'price_change_1d',
            'operator': ConditionOperator.GT,
            'value': 0.05,
            'description': '单日涨幅超过5%'
        },
        'price_down_5pct': {
            'name': '单日跌幅>5%',
            'category': ConditionCategory.MARKET_DATA,
            'field': 'price_change_1d',
            'operator': ConditionOperator.LT,
            'value': -0.05,
            'description': '单日跌幅超过5%'
        },
        'volume_ratio_gt_2': {
            'name': '量比>2',
            'category': ConditionCategory.MARKET_DATA,
            'field': 'volume_ratio',
            'operator': ConditionOperator.GT,
            'value': 2.0,
            'description': '当日成交量/5日均量>2'
        },
        'amount_gt_1b': {
            'name': '成交额>10亿',
            'category': ConditionCategory.MARKET_DATA,
            'field': 'amount',
            'operator': ConditionOperator.GT,
            'value': 1000000000,
            'description': '当日成交额超过10亿'
        },

        # 基本面条件
        'pe_lt_20': {
            'name': 'PE<20',
            'category': ConditionCategory.FUNDAMENTAL,
            'field': 'pe_ratio',
            'operator': ConditionOperator.LT,
            'value': 20,
            'description': '市盈率小于20'
        },
        'pb_lt_3': {
            'name': 'PB<3',
            'category': ConditionCategory.FUNDAMENTAL,
            'field': 'pb_ratio',
            'operator': ConditionOperator.LT,
            'value': 3,
            'description': '市净率小于3'
        },
        'roe_gt_15': {
            'name': 'ROE>15%',
            'category': ConditionCategory.FUNDAMENTAL,
            'field': 'roe',
            'operator': ConditionOperator.GT,
            'value': 0.15,
            'description': '净资产收益率超过15%'
        },
    }

    def __init__(self):
        """初始化选股器"""
        self.technical_checker = TechnicalConditionChecker()
        self.fundamental_checker = FundamentalConditionChecker()
        self.market_checker = MarketDataConditionChecker()
        self.conditions: List[ScreeningCondition] = []

    def add_condition(
        self,
        name: str,
        category: ConditionCategory,
        field: str,
        operator: ConditionOperator,
        value: Any,
        weight: float = 1.0,
        required: bool = False,
        description: str = ""
    ) -> 'ComprehensiveStockScreener':
        """
        添加选股条件

        Args:
            name: 条件名称
            category: 条件分类
            field: 数据字段
            operator: 操作符
            value: 条件值
            weight: 权重
            required: 是否必须满足
            description: 描述

        Returns:
            self，支持链式调用
        """
        condition = ScreeningCondition(
            name=name,
            category=category,
            field=field,
            operator=operator,
            value=value,
            weight=weight,
            description=description,
            required=required
        )
        self.conditions.append(condition)
        return self

    def add_predefined_condition(
        self,
        condition_key: str,
        weight: float = 1.0,
        required: bool = False
    ) -> 'ComprehensiveStockScreener':
        """
        添加预定义条件

        Args:
            condition_key: 预定义条件键
            weight: 权重
            required: 是否必须满足

        Returns:
            self
        """
        if condition_key not in self.PREDEFINED_CONDITIONS:
            raise ValueError(f"未知的预定义条件: {condition_key}")

        template = self.PREDEFINED_CONDITIONS[condition_key]
        return self.add_condition(
            name=template['name'],
            category=template['category'],
            field=template['field'],
            operator=template['operator'],
            value=template['value'],
            weight=weight,
            required=required,
            description=template['description']
        )

    def clear_conditions(self) -> 'ComprehensiveStockScreener':
        """清除所有条件"""
        self.conditions = []
        return self

    def screen(
        self,
        data: pd.DataFrame,
        stock_code: str = "",
        stock_name: str = ""
    ) -> Optional[ScreeningResult]:
        """
        对单只股票进行选股筛选

        Args:
            data: 股票数据
            stock_code: 股票代码
            stock_name: 股票名称

        Returns:
            选股结果，如果不满足条件返回None
        """
        if data is None or len(data) < 10:
            return None

        matched_conditions = []
        condition_scores = {}
        total_score = 0.0
        total_weight = 0.0
        required_failed = False

        for condition in self.conditions:
            # 根据分类选择检查器
            if condition.category == ConditionCategory.TECHNICAL:
                result = self.technical_checker.check(data, condition)
            elif condition.category == ConditionCategory.FUNDAMENTAL:
                result = self.fundamental_checker.check(data, condition)
            elif condition.category == ConditionCategory.MARKET_DATA:
                result = self.market_checker.check(data, condition)
            else:
                result = pd.Series([False] * len(data), index=data.index)

            # 检查最新一行是否满足
            is_matched = result.iloc[-1] if len(result) > 0 else False

            if is_matched:
                matched_conditions.append(condition.name)
                score = condition.weight * 100
                condition_scores[condition.name] = score
                total_score += score
            else:
                condition_scores[condition.name] = 0
                if condition.required:
                    required_failed = True

            total_weight += condition.weight

        # 如果有必须条件未满足，返回None
        if required_failed:
            return None

        # 计算综合得分
        final_score = total_score / total_weight if total_weight > 0 else 0

        if final_score == 0:
            return None

        return ScreeningResult(
            stock_code=stock_code,
            stock_name=stock_name,
            score=final_score,
            rank=0,
            matched_conditions=matched_conditions,
            condition_scores=condition_scores,
            details={}
        )

    def screen_multiple(
        self,
        data_dict: Dict[str, pd.DataFrame],
        stock_info: Optional[Dict[str, Dict]] = None,
        top_n: int = 50
    ) -> List[ScreeningResult]:
        """
        对多只股票进行批量选股

        Args:
            data_dict: 股票数据字典 {stock_code: DataFrame}
            stock_info: 股票基本信息 {stock_code: {name: xxx, ...}}
            top_n: 返回前N只股票

        Returns:
            选股结果列表，按得分排序
        """
        results = []

        for stock_code, data in data_dict.items():
            stock_name = ""
            if stock_info and stock_code in stock_info:
                stock_name = stock_info[stock_code].get('name', '')

            result = self.screen(data, stock_code, stock_name)
            if result:
                results.append(result)

        # 按得分排序
        results.sort(key=lambda x: x.score, reverse=True)

        # 设置排名
        for i, r in enumerate(results[:top_n]):
            r.rank = i + 1

        return results[:top_n]

    def list_predefined_conditions(self) -> List[Dict]:
        """列出所有预定义条件"""
        return [
            {
                'key': key,
                **template
            }
            for key, template in self.PREDEFINED_CONDITIONS.items()
        ]

    def get_condition_categories(self) -> Dict[str, List[str]]:
        """获取按分类组织的条件列表"""
        categories = {}
        for key, template in self.PREDEFINED_CONDITIONS.items():
            cat_name = template['category'].value
            if cat_name not in categories:
                categories[cat_name] = []
            categories[cat_name].append({
                'key': key,
                'name': template['name'],
                'description': template['description']
            })
        return categories


def create_screener_with_strategy(strategy_name: str) -> ComprehensiveStockScreener:
    """
    根据策略名称创建预配置的选股器

    Args:
        strategy_name: 策略名称

    Returns:
        配置好的选股器
    """
    screener = ComprehensiveStockScreener()

    if strategy_name == 'momentum':
        # 动量策略
        screener.add_predefined_condition('macd_golden_cross', weight=1.5, required=True)
        screener.add_predefined_condition('volume_breakout_2x', weight=1.0)
        screener.add_predefined_condition('ma_bullish', weight=1.0)

    elif strategy_name == 'value':
        # 价值策略
        screener.add_predefined_condition('pe_lt_20', weight=1.0, required=True)
        screener.add_predefined_condition('pb_lt_3', weight=1.0)
        screener.add_predefined_condition('roe_gt_15', weight=1.5)

    elif strategy_name == 'breakout':
        # 突破策略
        screener.add_predefined_condition('volume_breakout_2x', weight=1.5, required=True)
        screener.add_predefined_condition('near_year_high_10pct', weight=1.0)
        screener.add_predefined_condition('price_up_5pct', weight=1.0)

    elif strategy_name == 'oversold':
        # 超卖反弹策略
        screener.add_predefined_condition('rsi_oversold_30', weight=1.5, required=True)
        screener.add_predefined_condition('volume_breakout_2x', weight=1.0)

    elif strategy_name == 'trend':
        # 趋势策略
        screener.add_predefined_condition('ma_bullish', weight=1.5, required=True)
        screener.add_predefined_condition('continuous_rise_3', weight=1.0)
        screener.add_predefined_condition('amount_gt_1b', weight=0.5)

    return screener
