"""
测试增强功能模块

覆盖以下模块：
- 筹码分布 (chip_distribution)
- K线形态扩展 (talib_patterns)
- 经典选股策略 (classic_strategies)
- 综合选股框架 (comprehensive_screening)
- 批量时间处理 (batch_time_processor)
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


# ==================== 测试数据生成 ====================

def generate_sample_ohlcv(days: int = 300, seed: int = 42) -> pd.DataFrame:
    """生成测试用的OHLCV数据"""
    np.random.seed(seed)
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')

    # 生成价格数据
    close = 10 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, days)))
    high = close * (1 + np.abs(np.random.normal(0, 0.02, days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.02, days)))
    open_price = close * (1 + np.random.normal(0, 0.01, days))
    volume = np.random.randint(1000000, 10000000, days)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    return df


# ==================== 筹码分布测试 ====================

class TestChipDistribution:
    """筹码分布模块测试"""

    @pytest.fixture
    def sample_data(self):
        return generate_sample_ohlcv(300)

    def test_calculator_initialization(self):
        """测试计算器初始化"""
        from quanttool.factors.chip_distribution import ChipDistributionCalculator

        calculator = ChipDistributionCalculator(
            lookback_days=210,
            price_bins=100
        )

        assert calculator.lookback_days == 210
        assert calculator.price_bins == 100

    def test_calculate_distribution(self, sample_data):
        """测试筹码分布计算"""
        from quanttool.factors.chip_distribution import ChipDistributionCalculator

        calculator = ChipDistributionCalculator()
        result = calculator.calculate(sample_data)

        # 验证结果结构
        assert result.price_levels is not None
        assert result.chip_distribution is not None
        assert 0 <= result.concentration_ratio <= 100
        assert 0 <= result.profit_ratio <= 100
        assert 0 <= result.upper_pressure <= 100
        assert 0 <= result.lower_support <= 100
        assert 0 <= result.score <= 100

    def test_concentration_calculation(self, sample_data):
        """测试集中度计算"""
        from quanttool.factors.chip_distribution import ChipDistributionCalculator

        calculator = ChipDistributionCalculator()
        result = calculator.calculate(sample_data)

        # 集中度应该在合理范围内
        assert 0 <= result.concentration_ratio <= 100

    def test_support_resistance_levels(self, sample_data):
        """测试支撑阻力位识别"""
        from quanttool.factors.chip_distribution import ChipDistributionCalculator

        calculator = ChipDistributionCalculator()
        result = calculator.calculate(sample_data)

        # 支撑位应该低于当前价格
        for level in result.support_levels:
            assert level < sample_data['close'].iloc[-1]

        # 阻力位应该高于当前价格
        for level in result.resistance_levels:
            assert level > sample_data['close'].iloc[-1]

    def test_empty_data_handling(self):
        """测试空数据处理"""
        from quanttool.factors.chip_distribution import ChipDistributionCalculator

        calculator = ChipDistributionCalculator()
        result = calculator.calculate(pd.DataFrame())

        # 应该返回空结果而不是报错
        assert result.score == 0.0

    def test_get_chip_assessment(self, sample_data):
        """测试筹码评估"""
        from quanttool.factors.chip_distribution import (
            ChipDistributionCalculator,
            get_chip_assessment
        )

        calculator = ChipDistributionCalculator()
        result = calculator.calculate(sample_data)
        assessment = get_chip_assessment(result)

        assert isinstance(assessment, str)
        assert len(assessment) > 0


# ==================== K线形态扩展测试 ====================

class TestTalibPatterns:
    """TA-Lib K线形态测试"""

    @pytest.fixture
    def sample_data(self):
        return generate_sample_ohlcv(100)

    def test_recognizer_initialization(self):
        """测试识别器初始化"""
        from quanttool.factors.talib_patterns import TalibPatternRecognizer

        recognizer = TalibPatternRecognizer()
        assert recognizer.patterns_config is not None

    def test_recognize_all_patterns(self, sample_data):
        """测试形态识别"""
        try:
            import talib  # noqa: F401
        except ImportError:
            pytest.skip("TA-Lib not installed")

        from quanttool.factors.talib_patterns import TalibPatternRecognizer, TALIB_AVAILABLE

        if not TALIB_AVAILABLE:
            pytest.skip("TA-Lib not installed")

        recognizer = TalibPatternRecognizer()
        result = recognizer.recognize_all(sample_data, lookback=5)

        assert result.total_patterns >= 0
        assert result.bullish_count >= 0
        assert result.bearish_count >= 0
        assert -100 <= result.composite_signal <= 100

    def test_list_all_patterns(self):
        """测试列出所有形态"""
        from quanttool.factors.talib_patterns import TalibPatternRecognizer

        recognizer = TalibPatternRecognizer()
        patterns = recognizer.list_all_patterns()

        # 应该有61种形态
        assert len(patterns) == 61

    def test_get_pattern_description(self):
        """测试获取形态描述"""
        from quanttool.factors.talib_patterns import TalibPatternRecognizer

        recognizer = TalibPatternRecognizer()
        desc = recognizer.get_pattern_description('CDLHAMMER')

        assert desc['name'] == 'CDLHAMMER'
        assert desc['name_cn'] == '锤头'
        assert desc['type'] == 'bullish'


# ==================== 经典选股策略测试 ====================

class TestClassicStrategies:
    """经典选股策略测试"""

    @pytest.fixture
    def sample_data(self):
        return generate_sample_ohlcv(300)

    def test_volume_breakout_strategy(self, sample_data):
        """测试放量上涨策略"""
        from quanttool.strategies.classic_strategies import VolumeBreakoutStrategy

        strategy = VolumeBreakoutStrategy()
        strategy.initialize({})

        signals = strategy.calculate_signals(sample_data)

        assert 'signal' in signals.columns
        assert signals['signal'].isin([0, 1]).all()

    def test_ma_alignment_strategy(self, sample_data):
        """测试均线多头策略"""
        from quanttool.strategies.classic_strategies import MADAlignmentStrategy

        strategy = MADAlignmentStrategy()
        strategy.initialize({})

        signals = strategy.calculate_signals(sample_data)

        assert 'signal' in signals.columns
        assert 'ma' in signals.columns

    def test_apron_strategy(self, sample_data):
        """测试停机坪策略"""
        from quanttool.strategies.classic_strategies import ApronStrategy

        strategy = ApronStrategy()
        strategy.initialize({})

        signals = strategy.calculate_signals(sample_data)

        assert 'signal' in signals.columns

    def test_platform_breakout_strategy(self, sample_data):
        """测试突破平台策略"""
        from quanttool.strategies.classic_strategies import PlatformBreakoutStrategy

        strategy = PlatformBreakoutStrategy()
        strategy.initialize({})

        signals = strategy.calculate_signals(sample_data)

        assert 'signal' in signals.columns
        assert 'ma60' in signals.columns

    def test_no_large_drawdown_strategy(self, sample_data):
        """测试无大幅回撤策略"""
        from quanttool.strategies.classic_strategies import NoLargeDrawdownStrategy

        strategy = NoLargeDrawdownStrategy()
        strategy.initialize({})

        signals = strategy.calculate_signals(sample_data)

        assert 'signal' in signals.columns

    def test_strategy_interface(self, sample_data):
        """测试策略接口"""
        from quanttool.strategies.classic_strategies import VolumeBreakoutStrategy

        strategy = VolumeBreakoutStrategy()
        strategy.initialize({})

        # 测试接口方法
        assert strategy.get_name() == "VolumeBreakout"
        assert isinstance(strategy.get_parameters(), dict)
        assert isinstance(strategy.get_description(), str)

    def test_get_signal_method(self, sample_data):
        """测试get_signal方法"""
        from quanttool.strategies.classic_strategies import VolumeBreakoutStrategy

        strategy = VolumeBreakoutStrategy()
        strategy.initialize({})

        current_bar = sample_data.iloc[-1]
        historical_bars = sample_data

        signal = strategy.get_signal(current_bar, historical_bars)

        assert 'direction' in signal
        assert signal['direction'] in ['buy', 'sell', 'hold']


# ==================== 综合选股框架测试 ====================

class TestComprehensiveScreening:
    """综合选股框架测试"""

    @pytest.fixture
    def sample_data(self):
        return generate_sample_ohlcv(300)

    def test_screener_initialization(self):
        """测试选股器初始化"""
        from quanttool.factors.comprehensive_screening import ComprehensiveStockScreener

        screener = ComprehensiveStockScreener()
        assert screener.conditions == []

    def test_add_condition(self):
        """测试添加条件"""
        from quanttool.factors.comprehensive_screening import (
            ComprehensiveStockScreener,
            ConditionCategory,
            ConditionOperator
        )

        screener = ComprehensiveStockScreener()
        screener.add_condition(
            name="PE低估值",
            category=ConditionCategory.FUNDAMENTAL,
            field="pe_ratio",
            operator=ConditionOperator.LT,
            value=20,
            weight=1.0
        )

        assert len(screener.conditions) == 1
        assert screener.conditions[0].name == "PE低估值"

    def test_add_predefined_condition(self):
        """测试添加预定义条件"""
        from quanttool.factors.comprehensive_screening import ComprehensiveStockScreener

        screener = ComprehensiveStockScreener()
        screener.add_predefined_condition('macd_golden_cross')

        assert len(screener.conditions) == 1

    def test_screen_single_stock(self, sample_data):
        """测试单股票筛选"""
        from quanttool.factors.comprehensive_screening import ComprehensiveStockScreener

        screener = ComprehensiveStockScreener()
        screener.add_predefined_condition('ma_bullish')

        result = screener.screen(sample_data, "000001", "平安银行")

        # 结果可能为None（如果不满足条件）
        if result:
            assert result.stock_code == "000001"
            assert result.score >= 0

    def test_list_predefined_conditions(self):
        """测试列出预定义条件"""
        from quanttool.factors.comprehensive_screening import ComprehensiveStockScreener

        screener = ComprehensiveStockScreener()
        conditions = screener.list_predefined_conditions()

        assert len(conditions) > 0

    def test_get_condition_categories(self):
        """测试获取条件分类"""
        from quanttool.factors.comprehensive_screening import ComprehensiveStockScreener

        screener = ComprehensiveStockScreener()
        categories = screener.get_condition_categories()

        assert '技术面' in categories
        assert '基本面' in categories

    def test_create_screener_with_strategy(self, sample_data):
        """测试创建策略选股器"""
        from quanttool.factors.comprehensive_screening import create_screener_with_strategy

        screener = create_screener_with_strategy('momentum')
        assert len(screener.conditions) > 0

        screener = create_screener_with_strategy('value')
        assert len(screener.conditions) > 0


# ==================== 批量时间处理测试 ====================

class TestBatchTimeProcessor:
    """批量时间处理测试"""

    def test_trading_calendar_is_trading_day(self):
        """测试交易日判断"""
        from quanttool.infrastructure.batch_time_processor import TradingCalendar

        # 工作日应该是交易日（排除节假日）
        monday = datetime(2024, 1, 8)  # 周一
        assert TradingCalendar.is_trading_day(monday) == True

        # 周末不是交易日
        saturday = datetime(2024, 1, 6)  # 周六
        assert TradingCalendar.is_trading_day(saturday) == False

        sunday = datetime(2024, 1, 7)  # 周日
        assert TradingCalendar.is_trading_day(sunday) == False

    def test_trading_calendar_get_trading_days(self):
        """测试获取交易日列表"""
        from quanttool.infrastructure.batch_time_processor import TradingCalendar

        start = datetime(2024, 1, 8)
        end = datetime(2024, 1, 12)

        trading_days = TradingCalendar.get_trading_days(start, end)

        # 应该有5个交易日（周一到周五）
        assert len(trading_days) == 5

    def test_time_parser_parse(self):
        """测试时间解析"""
        from quanttool.infrastructure.batch_time_processor import TimeParser

        # 测试不同格式
        assert TimeParser.parse('2024-01-15') == datetime(2024, 1, 15)
        assert TimeParser.parse('20240115') == datetime(2024, 1, 15)
        assert TimeParser.parse('2024/01/15') == datetime(2024, 1, 15)

    def test_time_parser_parse_list(self):
        """测试日期列表解析"""
        from quanttool.infrastructure.batch_time_processor import TimeParser

        dates = TimeParser.parse_list('2024-01-01,2024-01-02,2024-01-03')

        assert len(dates) == 3
        assert dates[0] == datetime(2024, 1, 1)

    def test_time_parser_parse_range(self):
        """测试日期范围解析"""
        from quanttool.infrastructure.batch_time_processor import TimeParser

        dates = TimeParser.parse_range('2024-01-08', '2024-01-12', trading_days_only=False)

        assert len(dates) == 5

    def test_batch_processor_parse_time_args_current(self):
        """测试当前时间模式"""
        from quanttool.infrastructure.batch_time_processor import BatchTimeProcessor

        processor = BatchTimeProcessor()
        config = processor.parse_time_args(None)

        assert config.mode.value == 'current'
        assert len(config.dates) == 1

    def test_batch_processor_parse_time_args_single(self):
        """测试单个时间模式"""
        from quanttool.infrastructure.batch_time_processor import BatchTimeProcessor

        processor = BatchTimeProcessor()
        config = processor.parse_time_args(['2024-01-15'])

        assert config.mode.value == 'single'
        assert len(config.dates) == 1

    def test_batch_processor_parse_time_args_range(self):
        """测试区间时间模式"""
        from quanttool.infrastructure.batch_time_processor import BatchTimeProcessor

        processor = BatchTimeProcessor()
        config = processor.parse_time_args(['2024-01-08', '2024-01-12'])

        assert config.mode.value == 'range'
        assert len(config.dates) == 5  # 5个交易日

    def test_batch_processor_run_job(self):
        """测试运行批量作业"""
        from quanttool.infrastructure.batch_time_processor import BatchTimeProcessor

        results = []

        def job_func(date):
            results.append(date)
            return {"status": "success"}

        processor = BatchTimeProcessor(show_progress=False)
        result = processor.run_job(['2024-01-08', '2024-01-12'], job_func)

        assert result.success_count == 5
        assert result.failure_count == 0

    def test_run_batch_job_convenience(self):
        """测试便捷函数"""
        from quanttool.infrastructure.batch_time_processor import run_batch_job

        result = run_batch_job(
            lambda d: {"date": str(d)},
            ['2024-01-15'],
            show_progress=False
        )

        assert result.success_count == 1


# ==================== 集成测试 ====================

class TestIntegration:
    """集成测试"""

    def test_full_analysis_workflow(self):
        """测试完整分析流程"""
        # 生成测试数据
        df = generate_sample_ohlcv(300)

        # 1. 筹码分布分析
        from quanttool.factors.chip_distribution import ChipDistributionCalculator
        chip_calc = ChipDistributionCalculator()
        chip_result = chip_calc.calculate(df)

        assert chip_result.score >= 0

        # 2. 策略信号
        from quanttool.strategies.classic_strategies import VolumeBreakoutStrategy
        strategy = VolumeBreakoutStrategy()
        signals = strategy.calculate_signals(df)

        assert 'signal' in signals.columns

        # 3. 综合选股
        from quanttool.factors.comprehensive_screening import create_screener_with_strategy
        screener = create_screener_with_strategy('momentum')
        result = screener.screen(df, "000001", "测试股票")

        # 结果可能为None（如果不满足条件）
        if result:
            assert result.score >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
