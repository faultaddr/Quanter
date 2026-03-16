"""
Stock Analysis Module - Performs comprehensive technical analysis and generates buy/sell signals
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from typing import Dict, List, Tuple, Optional

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from quanttool.factors.tech_indicators import *
from quanttool.factors.trading_strategies import TradingStrategies
from quanttool.factors.scoring_system import ScoringSystem
from quanttool.factors.trend_scoring_system import TrendScoringSystem, analyze_trend_quality
from quanttool.factors.breakout_scoring_system import BreakoutScoringSystem, analyze_breakout_quality
from quanttool.factors.talib_patterns import (
    TalibPatternRecognizer, recognize_talib_patterns, format_patterns_report,
    draw_candlestick_chart, draw_pattern_illustration
)
from quanttool.factors.screening import StockScreener, ScreenResult
from quanttool.infrastructure.data_providers.data_fetcher import create_data_fetcher_with_credentials
from quanttool.infrastructure.data_providers.incremental_data_manager import IncrementalDataManager, DataType
from quanttool.strategies.adaptive_threshold import (
    AdaptiveThresholdManager, MarketRegime, VolatilityLevel,
    IndexMarketDetector, DualMarketState, CombinedSignal
)
from quanttool.risk.risk_controller import RiskController, StopLossType

# 新增：统一分析上下文相关导入
from quanttool.factors.analysis_context import (
    AnalysisContext, ActionType, MarketState, PositionAssessment,
    StopLossConfig, StopLossType as UnifiedStopLossType,
    UnifiedMarketState, FinalRecommendation, ScoringSystemType,
    ClassicScore, TrendScore, BreakoutScore
)
from quanttool.factors.recommendation_engine import RecommendationEngine
from quanttool.factors.unified_stop_loss import UnifiedStopLossCalculator


class StockAnalyzer:
    """
    A comprehensive stock analyzer that calculates technical indicators and evaluates trading strategies
    """

    def __init__(self, use_cache: bool = True, max_workers: int = 10, use_incremental: bool = True):
        """Initialize the stock analyzer with data fetcher

        Args:
            use_cache: Whether to use local cache for data (default: True)
            max_workers: Maximum parallel workers for batch fetching (default: 10)
            use_incremental: Whether to use incremental data fetching (default: True)
        """
        self.fetcher = create_data_fetcher_with_credentials()
        self.fetcher.initialize()
        self.use_cache = use_cache
        self.max_workers = max_workers
        self.use_incremental = use_incremental

        # 增量数据管理器
        self._incremental_manager: Optional[IncrementalDataManager] = None
        if use_incremental:
            try:
                self._incremental_manager = IncrementalDataManager()
            except Exception as e:
                print(f"⚠️ 增量数据管理器初始化失败: {e}")
                self._incremental_manager = None

        # Cache for batch-fetched data
        self._batch_data_cache: Dict[str, pd.DataFrame] = {}

    def get_stock_data(
        self,
        symbol: str,
        days: int = 360,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch stock data for the specified symbol and time period.

        增量数据拉取策略：
        1. 优先使用增量数据管理器（只拉取缺失日期）
        2. 如果禁用增量模式，使用传统方式拉取完整数据
        3. 支持强制刷新缓存

        Args:
            symbol: Stock symbol (e.g., '600519.SH' or '600519')
            days: Number of days of data to fetch (used only if start_date/end_date not provided)
            start_date: Optional start date (if provided, days parameter is ignored)
            end_date: Optional end date (if provided, days parameter is ignored)
            force_refresh: Force refresh cache (default: False)

        Returns:
            DataFrame with stock data, empty DataFrame if failed
        """
        # Use provided dates or calculate from days
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=days)

        # Normalize symbol format for different data providers
        normalized_symbol = self._normalize_symbol(symbol)

        # Check batch cache first
        if normalized_symbol in self._batch_data_cache and not force_refresh:
            print(f"✅ 从内存缓存获取 {normalized_symbol} 的数据")
            return self._batch_data_cache[normalized_symbol].copy()

        # 使用增量数据管理器
        if self._incremental_manager and not force_refresh:
            print(f"⏳ 增量获取 {normalized_symbol} 数据 ({start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})...")
            try:
                df = self._incremental_manager.get_data(
                    normalized_symbol,
                    start_date,
                    end_date,
                    self.fetcher,
                    data_type=DataType.STOCK_BAR,
                    force_refresh=False
                )
                if not df.empty:
                    print(f"✅ 成功获取 {len(df)} 条记录")
                    return df
            except Exception as e:
                print(f"⚠️ 增量获取失败，回退到传统方式: {e}")

        # 传统方式：完整拉取
        print(f"⏳ 获取 {normalized_symbol} 数据 ({start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})...")
        symbols = [normalized_symbol]
        data = self.fetcher.get_bars(symbols, start_date, end_date)

        if normalized_symbol in data and not data[normalized_symbol].empty:
            df = data[normalized_symbol].copy()
            print(f"成功获取 {len(df)} 条记录")
            return df
        else:
            print(f"未能获取 {normalized_symbol} 的数据")
            return pd.DataFrame()

    def update_cache_latest(
        self,
        symbols: List[str],
        days_back: int = 30
    ) -> Dict[str, int]:
        """
        批量更新最新数据（用于定时任务）

        只拉取每只股票缺失的最新数据，而不是完整历史数据

        Args:
            symbols: 股票列表
            days_back: 回溯天数（防止停牌股票漏数据）

        Returns:
            Dict[symbol, rows_added]: 每只股票新增的行数
        """
        if self._incremental_manager:
            print(f"📦 增量更新 {len(symbols)} 只股票的最新数据...")
            results = self._incremental_manager.update_latest(
                symbols, self.fetcher, days_back
            )
            total_added = sum(results.values())
            print(f"✅ 更新完成，共新增 {total_added} 条记录")
            return results
        else:
            print("⚠️ 增量数据管理器未启用")
            return {}

    def get_cache_stats(self) -> Dict:
        """
        获取缓存统计信息

        Returns:
            Dict: 包含 symbol_count, total_rows, total_size_mb 等
        """
        if self._incremental_manager:
            return self._incremental_manager.get_cache_stats()
        return {
            "symbol_count": len(self._batch_data_cache),
            "total_rows": sum(len(df) for df in self._batch_data_cache.values()),
            "total_size_mb": 0,
            "cache_dir": "memory only"
        }

    def list_cached_symbols(self) -> List[Dict]:
        """
        列出所有已缓存的股票

        Returns:
            List of Dict: 每个包含 symbol, earliest_date, latest_date, row_count 等
        """
        if self._incremental_manager:
            return self._incremental_manager.list_symbols()
        return [
            {"symbol": s, "row_count": len(df)}
            for s, df in self._batch_data_cache.items()
        ]

    def clear_expired_cache(self) -> int:
        """
        清理过期的缓存数据

        Returns:
            int: 清理的条目数
        """
        if self._incremental_manager:
            count = self._incremental_manager.clear_expired()
            print(f"🗑️ 清理 {count} 条过期缓存")
            return count
        return 0

    def get_stock_data_batch(
        self,
        symbols: List[str],
        days: int = 360,
        use_parallel: bool = True,
        show_progress: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data for multiple symbols in parallel with caching.

        This method is significantly faster than calling get_stock_data()
        repeatedly for each symbol, as it:
        1. Fetches all symbols in parallel (10 workers by default)
        2. Uses local cache to avoid redundant network requests
        3. Stores results in memory for subsequent analysis

        Args:
            symbols: List of stock symbols
            days: Number of days of data to fetch
            use_parallel: Whether to use parallel fetching (default: True)
            show_progress: Whether to show progress logs

        Returns:
            Dictionary mapping normalized symbols to DataFrames
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Normalize all symbols
        normalized_symbols = [self._normalize_symbol(s) for s in symbols]

        # Filter out already cached symbols
        symbols_to_fetch = [
            s for s in normalized_symbols
            if s not in self._batch_data_cache
        ]

        if symbols_to_fetch:
            if show_progress:
                print(f"正在批量获取 {len(symbols_to_fetch)} 只股票数据...")

            if use_parallel:
                # Use parallel fetching with cache
                data = self.fetcher.get_bars_cached(
                    symbols_to_fetch, start_date, end_date
                )
            else:
                data = self.fetcher.get_bars(
                    symbols_to_fetch, start_date, end_date
                )

            # Store in batch cache
            for symbol, df in data.items():
                if not df.empty:
                    self._batch_data_cache[symbol] = df

            if show_progress:
                print(f"成功获取 {len(data)} 只股票数据")

        # Return all requested data (from batch cache)
        result = {}
        for symbol in normalized_symbols:
            if symbol in self._batch_data_cache:
                result[symbol] = self._batch_data_cache[symbol].copy()

        return result

    def clear_batch_cache(self):
        """Clear the in-memory batch data cache."""
        self._batch_data_cache.clear()

    def preload_data_for_scan(
        self,
        stock_list: List[Dict[str, str]],
        days: int = 360
    ) -> int:
        """
        Preload all stock data for a scan operation.

        Call this before analyzing multiple stocks to significantly
        speed up the scan by fetching all data in parallel.

        Args:
            stock_list: List of stock info dicts with 'code' key
            days: Number of days of data to fetch

        Returns:
            Number of stocks successfully loaded
        """
        symbols = [
            stock['code'] if isinstance(stock, dict) else stock
            for stock in stock_list
        ]

        data = self.get_stock_data_batch(symbols, days, use_parallel=True)
        return len(data)

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize the symbol format for different data providers
        """
        symbol = symbol.upper().strip()

        # Handle various input formats
        if symbol.endswith('.SH') or symbol.endswith('.SZ'):
            return symbol
        elif len(symbol) == 6:
            if symbol.startswith(('5', '6', '9')):
                return f"{symbol}.SH"  # Shanghai
            else:
                return f"{symbol}.SZ"  # Shenzhen
        else:
            return symbol

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various technical indicators for the given data
        """
        if df.empty or 'close' not in df.columns:
            print("Error: DataFrame is empty or missing 'close' column")
            return df

        df = df.sort_values('timestamp').reset_index(drop=True)

        # Ensure all required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                if col in ['open', 'high', 'low']:
                    # Use close price for missing OHLC data
                    df[col] = df['close']
                elif col == 'volume':
                    # Create a default volume column if missing
                    df[col] = 1000000  # Default volume

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        open_price = df['open'].values

        # Calculate technical indicators
        print("正在计算技术指标...")

        # Moving Averages
        df['ma_5'] = MA(close, 5)
        df['ma_6'] = MA(close, 6)  # 用于BIAS(6)计算
        df['ma_10'] = MA(close, 10)
        df['ma_20'] = MA(close, 20)
        df['ma_30'] = MA(close, 30)
        df['ma_50'] = MA(close, 50)
        df['ma_100'] = MA(close, 100)
        df['ma_200'] = MA(close, 200)

        # RSI
        df['rsi_6'] = RSI(close, 6)
        df['rsi_12'] = RSI(close, 12)
        df['rsi_24'] = RSI(close, 24)

        # MACD
        dif, dea, macd = MACD(close)
        df['macd_dif'] = dif
        df['macd_dea'] = dea
        df['macd'] = macd

        # KDJ
        k, d, j = KDJ(close, high, low)
        df['kdj_k'] = k
        df['kdj_d'] = d
        df['kdj_j'] = j

        # Bollinger Bands
        upper, mid, lower = BOLL(close)
        df['boll_upper'] = upper
        df['boll_mid'] = mid
        df['boll_lower'] = lower

        # BIAS
        bias_6, bias_12, bias_24 = BIAS(close)
        df['bias_6'] = bias_6
        df['bias_12'] = bias_12
        df['bias_24'] = bias_24

        # CCI
        df['cci'] = CCI(close, high, low)

        # ATR and True Range
        df['atr_14'] = ATR(close, high, low, 14)

        # DMI/DMI
        pdi, mdi, adx, adxr = DMI(close, high, low)
        df['dmi_pdi'] = pdi
        df['dmi_mdi'] = mdi
        df['dmi_adx'] = adx
        df['dmi_adxr'] = adxr

        # TRIX
        trix, trma = TRIX(close)
        df['trix'] = trix
        df['trix_ma'] = trma

        # VR
        df['vr'] = VR(close, volume)

        # CR
        df['cr'] = CR(close, high, low)

        # Williams %R
        wr, wr_6 = WR(close, high, low)
        df['wr'] = wr
        df['wr_6'] = wr_6

        # BBI
        df['bbi'] = BBI(close)

        # PSY
        psy, psyma = PSY(close)
        df['psy'] = psy
        df['psyma'] = psyma

        # OBV 能量潮
        df['obv'] = OBV(close, volume)
        df['obv_ma'] = MA(df['obv'].values, 20)  # OBV 20日均线

        # 成交量均线
        df['vol_ma5'] = MA(volume, 5)
        df['vol_ma10'] = MA(volume, 10)
        df['vol_ma20'] = MA(volume, 20)

        # MFI 资金流量指标
        df['mfi'] = MFI(close, high, low, volume)

        # Daily return
        df['daily_return'] = df['close'].pct_change()

        # Price position in the range
        df['price_position'] = (df['close'] - LLV(low, 20)) / (HHV(high, 20) - LLV(low, 20)) * 100

        print("技术指标计算完成。")
        return df

    def run_trading_strategies(self, df: pd.DataFrame, symbol: str = "") -> Dict:
        """
        Run various trading strategies and scoring system on the given data
        """
        if df.empty:
            return {}

        print("正在运行交易策略...")

        # Initialize strategy evaluator (传统策略，保留兼容性)
        strategies = TradingStrategies()

        results = {}

        # Individual strategies
        results['rsi_strategy'] = strategies.rsi_strategy(df)
        results['macd_strategy'] = strategies.macd_strategy(df)
        results['ma_crossover_strategy'] = strategies.ma_crossover_strategy(df)
        results['bollinger_bands_strategy'] = strategies.bollinger_bands_strategy(df)
        results['combined_strategy'] = strategies.combined_strategy(df)

        # Evaluate current signals
        results['evaluations'] = {}
        results['evaluations']['rsi'] = strategies.evaluate_current_signal(results['rsi_strategy'], 'RSI Strategy')
        results['evaluations']['macd'] = strategies.evaluate_current_signal(results['macd_strategy'], 'MACD Strategy')
        results['evaluations']['ma_crossover'] = strategies.evaluate_current_signal(results['ma_crossover_strategy'], 'MA Crossover Strategy')
        results['evaluations']['bollinger_bands'] = strategies.evaluate_current_signal(results['bollinger_bands_strategy'], 'Bollinger Bands Strategy')
        results['evaluations']['combined'] = strategies.evaluate_current_signal(results['combined_strategy'], 'Combined Strategy')

        # Run NEW scoring system (百分制)
        print("正在计算多维度评分（百分制）...")
        scoring = ScoringSystem(stop_loss_pct=0.05)

        # 新增：双重市场状态检测
        print("正在检测双重市场状态...")
        market_detector = IndexMarketDetector(default_index='hs300')
        dual_market_state = market_detector.get_dual_market_state(df)
        results['dual_market_state'] = {
            'index_regime': dual_market_state.index_regime.value,
            'stock_regime': dual_market_state.stock_regime.value,
            'combined_signal': dual_market_state.combined_signal.value,
            'confidence': dual_market_state.confidence,
            'index_code': dual_market_state.index_code,
            'index_name': dual_market_state.index_name,
        }

        # 根据综合信号调整评分系统的市场状态
        if dual_market_state.combined_signal == CombinedSignal.CASH:
            # 空仓信号，设置熊市权重
            scoring.set_market_regime('bear')
        elif dual_market_state.combined_signal == CombinedSignal.STRONG_BUY:
            # 强买入信号，设置牛市权重
            scoring.set_market_regime('bull')
        elif dual_market_state.combined_signal == CombinedSignal.WAIT:
            # 观望信号，设置震荡市权重
            scoring.set_market_regime('sideway')

        # 获取日期信息
        latest_date = df['timestamp'].iloc[-1].strftime('%Y-%m-%d') if 'timestamp' in df.columns else ''

        # K线形态分析（已集成到评分系统，此处仅用于报告展示）
        candlestick_result = recognize_talib_patterns(df, lookback=5)

        # 计算股价位置（用于形态定性判断）
        latest = df.iloc[-1]
        close = latest.get('close', 0)
        ma20 = latest.get('ma_20', close)
        bias20 = (close / ma20 - 1) if ma20 > 0 else 0

        # 计算60日高低点位置
        position_ratio = 0.5
        if len(df) >= 60:
            high_60 = df['high'].iloc[-60:].max()
            low_60 = df['low'].iloc[-60:].min()
            if high_60 > low_60:
                position_ratio = (close - low_60) / (high_60 - low_60)

        # 计算布林带百分比位置
        boll_pctb = 0.5
        boll_upper = latest.get('boll_upper', close)
        boll_lower = latest.get('boll_lower', close)
        if boll_upper != boll_lower:
            boll_pctb = (close - boll_lower) / (boll_upper - boll_lower)

        # 评分计算（纯因子评分，不含K线形态）
        results['scoring'] = scoring.calculate_all_scores(
            df=df,
            stock_code=symbol,
            trade_date_T=latest_date,
            trade_date_T1=None,
            open_T1=None
        )

        # 新增：应用筛选层（K线形态筛选）
        screener = StockScreener()
        screening_result = screener.screen(df, results['scoring'])
        results['screening'] = {
            'result': screening_result.result.value,
            'score_modifier': screening_result.score_modifier,
            'reasons': screening_result.reasons,
            'details': screening_result.details,
        }

        # 根据筛选结果调整推荐
        if screening_result.result == ScreenResult.FILTER:
            results['scoring']['screening_action'] = 'reject'
            results['scoring']['screening_reason'] = '; '.join(screening_result.reasons)
        elif screening_result.result == ScreenResult.WARNING:
            results['scoring']['screening_action'] = 'caution'
            results['scoring']['screening_reason'] = '; '.join(screening_result.reasons)
            # 应用评分修正
            if 'score' in results['scoring']:
                original_score = results['scoring']['score']
                results['scoring']['original_score'] = original_score
                results['scoring']['score'] = original_score * screening_result.score_modifier
        else:
            results['scoring']['screening_action'] = 'pass'
            results['scoring']['screening_reason'] = '; '.join(screening_result.reasons) if screening_result.reasons else '通过筛选'

        # 保存形态和位置信息供报告使用
        results['candlestick_patterns'] = candlestick_result
        results['position_info'] = {
            'position_ratio': position_ratio,
            'bias20': bias20,
            'boll_pctb': boll_pctb,
            'is_low': position_ratio < 0.35 or bias20 < -0.05 or boll_pctb < 0.2,
            'is_high': position_ratio > 0.70 or bias20 > 0.05 or boll_pctb > 0.8,
        }

        # 新增：趋势评分系统（纯趋势强度评分）
        print("正在计算趋势评分...")
        try:
            trend_system = TrendScoringSystem()
            trend_result = trend_system.calculate_score(df)
            results['trend_scoring'] = {
                'final_score': trend_result.final_score,
                'trend_total_score': trend_result.trend_total_score,
                'timing_coefficient': trend_result.timing_coefficient,
                'ma_structure_score': trend_result.ma_structure_score,
                'price_momentum_score': trend_result.price_momentum_score,
                'volume_score': trend_result.volume_score,
                'relative_strength_score': trend_result.relative_strength_score,
                'timing_type': trend_result.timing_type,
                'passed_hard_filter': trend_result.passed_hard_filter,
                'hard_filter_reason': trend_result.hard_filter_reason,
                'details': trend_result.details,
            }
            print(f"趋势评分: {trend_result.final_score:.1f}分 (时机系数: {trend_result.timing_coefficient:.2f})")
        except Exception as e:
            print(f"趋势评分计算失败: {e}")
            results['trend_scoring'] = None

        # 新增：市场状态检测和自适应阈值
        try:
            threshold_manager = AdaptiveThresholdManager()
            adaptive_config = threshold_manager.get_adaptive_thresholds(df)
            results['market_regime'] = {
                'regime': adaptive_config.market_regime.value,
                'volatility_level': adaptive_config.volatility_level.value,
                'buy_threshold': adaptive_config.buy_threshold,
                'sell_threshold': adaptive_config.sell_threshold,
                'confidence': adaptive_config.confidence,
            }
            print(f"市场状态: {adaptive_config.market_regime.value} | 波动率: {adaptive_config.volatility_level.value}")
        except Exception as e:
            print(f"市场状态检测失败: {e}")
            results['market_regime'] = None

        # 新增：风险控制建议
        try:
            risk_controller = RiskController()
            signal_strength = results['scoring'].get('score', 50) / 100
            stop_loss_result = risk_controller.calculate_dynamic_stop_loss(
                df, close, signal_strength=signal_strength
            )
            results['risk_control'] = {
                'stop_price': stop_loss_result.stop_price,
                'stop_type': stop_loss_result.stop_type.value,
                'distance_percent': stop_loss_result.distance_percent,
                'confidence': stop_loss_result.confidence,
            }
            print(f"风险控制: 建议止损位 ¥{stop_loss_result.stop_price:.2f} ({stop_loss_result.stop_type.value})")
        except Exception as e:
            print(f"风险控制计算失败: {e}")
            results['risk_control'] = None

        print(f"评分计算完成。综合评分: {results['scoring'].get('score', 0):.1f}分")
        print(f"筛选结果: {results['scoring'].get('screening_action', 'unknown')} - {results['scoring'].get('screening_reason', '')}")
        print("交易策略运行完成。")
        return results

    def build_analysis_context(
        self,
        df: pd.DataFrame,
        symbol: str,
        primary_system: ScoringSystemType = ScoringSystemType.AUTO
    ) -> AnalysisContext:
        """
        构建统一分析上下文

        整合三套评分系统、市场状态、位置评估、止损计算，
        生成最终的推荐结果。

        Args:
            df: 股票数据（已计算技术指标）
            symbol: 股票代码
            primary_system: 主评分系统类型

        Returns:
            AnalysisContext: 统一分析上下文
        """
        if df.empty:
            return AnalysisContext(
                symbol=symbol,
                current_price=0,
                analysis_date=datetime.now()
            )

        latest = df.iloc[-1]
        close = latest.get('close', 0)

        # 初始化上下文
        context = AnalysisContext(
            symbol=symbol,
            current_price=close,
            analysis_date=datetime.now()
        )

        # 1. 运行三套评分系统
        print("正在计算三套评分系统...")

        # 经典评分
        context.classic_score = self._run_classic_scoring(df, symbol)

        # 趋势评分
        context.trend_score = self._run_trend_scoring(df)

        # 突破评分
        context.breakout_score = self._run_breakout_scoring(df)

        # 2. 构建统一市场状态
        context.market_state = self._build_unified_market_state(df)

        # 3. 构建位置评估
        context.position_assessment = self._build_position_assessment(df, context.classic_score)

        # 4. 计算统一止损
        stop_loss_calculator = UnifiedStopLossCalculator()
        context.stop_loss_config = stop_loss_calculator.calculate(df, context)

        # 5. 存储K线形态
        candlestick_result = recognize_talib_patterns(df, lookback=5)
        context.candlestick_patterns = candlestick_result.get('patterns', [])

        # 6. 存储筛选结果
        context.screening_result = {
            'result': 'pass',
            'reasons': [],
        }

        # 7. 存储最后一行数据
        context.df_last_row = latest.to_dict() if hasattr(latest, 'to_dict') else dict(latest)

        # 8. 生成最终推荐（单点决策）
        recommendation_engine = RecommendationEngine()
        context.final_recommendation = recommendation_engine.generate_recommendation(context)

        print(f"评分完成: 经典={context.classic_score.score:.1f}分, "
              f"趋势={context.trend_score.final_score:.1f}分, "
              f"突破={context.breakout_score.final_score:.1f}分")
        print(f"最终推荐: {context.final_recommendation.get_action_display()} "
              f"(主系统: {context.final_recommendation.primary_system.value})")

        return context

    def _run_classic_scoring(self, df: pd.DataFrame, symbol: str) -> ClassicScore:
        """运行经典评分系统"""
        scoring = ScoringSystem(stop_loss_pct=0.05)

        latest_date = df['timestamp'].iloc[-1].strftime('%Y-%m-%d') if 'timestamp' in df.columns else ''

        try:
            result = scoring.calculate_all_scores(
                df=df,
                stock_code=symbol,
                trade_date_T=latest_date,
                trade_date_T1=None,
                open_T1=None
            )

            return ClassicScore(
                score=result.get('score', 50),
                trend_score=result.get('trend_score', 50),
                position_modifier=result.get('position_modifier', 1.0),
                score_grade=result.get('score_grade', '一般'),
                factors_score=result.get('factors_score', {}),
                factors_raw=result.get('factors_raw', {}),
                execution=result.get('execution', {}),
                warnings=result.get('warnings', []),
            )
        except Exception as e:
            print(f"经典评分计算失败: {e}")
            return ClassicScore()

    def _run_trend_scoring(self, df: pd.DataFrame) -> TrendScore:
        """运行趋势评分系统"""
        try:
            system = TrendScoringSystem()
            result = system.calculate_score(df)

            return TrendScore(
                final_score=result.final_score,
                trend_total_score=result.trend_total_score,
                timing_coefficient=result.timing_coefficient,
                timing_type=result.timing_type,
                passed_hard_filter=result.passed_hard_filter,
                hard_filter_reason=result.hard_filter_reason,
                ma_structure_score=result.ma_structure_score,
                price_momentum_score=result.price_momentum_score,
                volume_score=result.volume_score,
                relative_strength_score=result.relative_strength_score,
                details=result.details,
            )
        except Exception as e:
            print(f"趋势评分计算失败: {e}")
            return TrendScore()

    def _run_breakout_scoring(self, df: pd.DataFrame) -> BreakoutScore:
        """运行突破评分系统"""
        try:
            system = BreakoutScoringSystem()
            result = system.calculate_score(df)

            return BreakoutScore(
                final_score=result.final_score,
                is_low_position=result.is_low_position,
                is_consolidating=result.is_consolidating,
                has_breakout=result.has_breakout,
                passed_filter=result.passed_filter,
                filter_reason=result.filter_reason,
                quality_score=result.quality_score,
                growth_score=result.growth_score,
                value_score=result.value_score,
                momentum_score=result.momentum_score,
                flow_score=result.flow_score,
                risk_score=result.risk_score,
                consolidation_days=result.consolidation_days,
                price_range=result.price_range,
                volume_ratio=result.volume_ratio,
                breakout_strength=result.breakout_strength,
                stop_loss_price=result.stop_loss_price,
                take_profit_price=result.take_profit_price,
                details=result.details,
            )
        except Exception as e:
            print(f"突破评分计算失败: {e}")
            return BreakoutScore()

    def _build_unified_market_state(self, df: pd.DataFrame) -> UnifiedMarketState:
        """构建统一市场状态"""
        try:
            market_detector = IndexMarketDetector(default_index='hs300')
            dual_state = market_detector.get_dual_market_state(df)

            # 映射市场状态
            regime_map = {
                MarketRegime.BULL: MarketState.BULL,
                MarketRegime.BEAR: MarketState.BEAR,
                MarketRegime.SIDEWAY: MarketState.SIDEWAY,
                MarketRegime.VOLATILE: MarketState.VOLATILE,
            }

            return UnifiedMarketState(
                index_regime=regime_map.get(dual_state.index_regime, MarketState.SIDEWAY),
                stock_regime=regime_map.get(dual_state.stock_regime, MarketState.SIDEWAY),
                combined_regime=regime_map.get(dual_state.index_regime, MarketState.SIDEWAY),
                confidence=dual_state.confidence,
                volatility_level='normal',
                combined_signal=dual_state.combined_signal.value,
                index_code=dual_state.index_code,
                index_name=dual_state.index_name,
            )
        except Exception as e:
            print(f"市场状态检测失败: {e}")
            return UnifiedMarketState()

    def _build_position_assessment(
        self,
        df: pd.DataFrame,
        classic_score: ClassicScore
    ) -> PositionAssessment:
        """构建位置评估"""
        if df.empty:
            return PositionAssessment()

        latest = df.iloc[-1]
        close = latest.get('close', 0)
        ma20 = latest.get('ma_20', close)
        ma50 = latest.get('ma_50', close)
        ma200 = latest.get('ma_200', 0)

        # 计算位置修正系数（从经典评分获取）
        position_modifier = classic_score.position_modifier

        # 技术指标状态
        wr = latest.get('wr', 50)
        cci = latest.get('cci', 0)
        rsi = latest.get('rsi_24', 50)
        boll_upper = latest.get('boll_upper', close)
        boll_lower = latest.get('boll_lower', close)

        boll_pctb = (close - boll_lower) / (boll_upper - boll_lower) if boll_upper > boll_lower else 0.5

        # 判断超买超卖
        is_extreme_overbought = wr < 10 or cci > 200 or rsi > 80 or boll_pctb > 0.95
        is_extreme_oversold = wr > 90 or cci < -200 or rsi < 20 or boll_pctb < 0.05
        is_overbought = (wr < 20 or cci > 100 or rsi > 70 or boll_pctb > 0.85) and not is_extreme_overbought
        is_oversold = (wr > 80 or cci < -100 or rsi < 30 or boll_pctb < 0.15) and not is_extreme_oversold

        # 计算价格位置
        avg_ma = (ma20 + ma50) / 2 if ma20 > 0 and ma50 > 0 else close
        price_ratio = close / avg_ma if avg_ma > 0 else 1.0

        # 计算MA20乖离率
        bias20 = (close / ma20 - 1) if ma20 > 0 else 0

        # 计算长短期位置
        long_term_position = 'mid'
        short_term_position = 'mid'

        if len(df) >= 60:
            high_60 = df['high'].iloc[-60:].max()
            low_60 = df['low'].iloc[-60:].min()
            if high_60 > low_60:
                position_ratio = (close - low_60) / (high_60 - low_60)
                if position_ratio < 0.35:
                    long_term_position = 'low'
                elif position_ratio > 0.65:
                    long_term_position = 'high'

        # 短期位置基于技术指标
        if is_extreme_oversold or is_oversold:
            short_term_position = 'low'
        elif is_extreme_overbought or is_overbought:
            short_term_position = 'high'

        # 综合位置判断
        if is_extreme_oversold or is_oversold:
            position = 'low'
            reason = f"技术指标超卖（WR={wr:.1f}, RSI={rsi:.1f}）"
        elif is_extreme_overbought or is_overbought:
            position = 'high'
            reason = f"技术指标超买（WR={wr:.1f}, RSI={rsi:.1f}）"
        elif boll_pctb < 0.25:
            position = 'low'
            reason = f"布林带下轨附近（位置={boll_pctb*100:.0f}%）"
        elif boll_pctb > 0.75:
            position = 'high'
            reason = f"布林带上轨附近（位置={boll_pctb*100:.0f}%）"
        elif price_ratio < 0.95:
            position = 'low'
            reason = f"均线位置偏低（价格/均价={price_ratio:.2f}）"
        elif price_ratio > 1.05:
            position = 'high'
            reason = f"均线位置偏高（价格/均价={price_ratio:.2f}）"
        else:
            position = 'middle'
            reason = f"位置适中（修正系数={position_modifier:.2f}）"

        return PositionAssessment(
            position=position,
            long_term_position=long_term_position,
            short_term_position=short_term_position,
            is_overbought=is_overbought or is_extreme_overbought,
            is_oversold=is_oversold or is_extreme_oversold,
            is_extreme_overbought=is_extreme_overbought,
            is_extreme_oversold=is_extreme_oversold,
            position_modifier=position_modifier,
            price_ratio=price_ratio,
            boll_pctb=boll_pctb,
            bias20=bias20,
            close=close,
            ma20=ma20,
            ma50=ma50,
            ma200=ma200 if not pd.isna(ma200) else 0,
            reason=reason,
        )

    def analyze_stock_with_context(
        self,
        symbol: str,
        days: int = 360,
        primary_system: ScoringSystemType = ScoringSystemType.AUTO
    ) -> Tuple[AnalysisContext, str]:
        """
        使用统一分析上下文分析股票

        Args:
            symbol: 股票代码
            days: 分析天数
            primary_system: 主评分系统类型

        Returns:
            Tuple[AnalysisContext, str]: (分析上下文, 报告文本)
        """
        print(f"开始分析 {symbol}（使用统一上下文）...")

        # 获取数据
        df = self.get_stock_data(symbol, days)
        if df.empty:
            return AnalysisContext(
                symbol=symbol,
                current_price=0,
                analysis_date=datetime.now()
            ), f"无法获取 {symbol} 的数据"

        # 计算技术指标
        df_with_indicators = self.calculate_technical_indicators(df)

        # 构建统一分析上下文
        context = self.build_analysis_context(df_with_indicators, symbol, primary_system)

        # 生成报告
        report = self.generate_report_from_context(df_with_indicators, context, symbol)

        return context, report

    def generate_report_from_context(
        self,
        df: pd.DataFrame,
        context: AnalysisContext,
        symbol: str
    ) -> str:
        """
        从统一分析上下文生成报告

        使用同一个 FinalRecommendation 对象，确保报告各部分一致
        """
        if df.empty:
            return "No data available for report generation"

        report = []
        rec = context.final_recommendation

        # 基本信息
        report.append(f"# 股票技术分析报告：{symbol}")
        report.append("")
        report.append(f"**分析日期：** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"**主评分系统：** {rec.primary_system.value}")
        report.append("")

        # ========== 第一部分：核心结论 ==========
        report.extend(self._generate_core_conclusion_v2(context))

        # ========== 第二部分：三系统评分对比 ==========
        report.extend(self._generate_three_system_analysis(context))

        # ========== 第三部分：市场状态与风险 ==========
        report.extend(self._generate_market_risk_section(context))

        # ========== 第四部分：交易执行计划 ==========
        report.extend(self._generate_trading_plan_v2(context))

        # 附录：原始技术指标
        report.append("---")
        report.append("")
        report.append("## 附录：原始技术指标")
        report.append("")

        latest = df.iloc[-1]
        report.extend(self._generate_technical_indicators_table(latest, context))

        report.append("")
        report.append("> **免责声明：** 本分析仅供学习参考，不构成投资建议。投资决策应基于全面研究和独立判断。")
        report.append("")

        return "\n".join(report)

    def _generate_core_conclusion_v2(self, context: AnalysisContext) -> List[str]:
        """生成核心结论（使用统一推荐）"""
        report = []
        rec = context.final_recommendation

        report.append("## 第一部分：核心结论")
        report.append("")

        # 操作指令
        report.append(f"### {rec.get_action_emoji()} 操作指令：{rec.get_action_display()}")
        report.append("")

        # 评分展示
        report.append(f"**技术评分：{rec.final_score:.1f}分（{rec.score_grade}）**")
        report.append("")

        # 主评分系统说明
        system_names = {
            ScoringSystemType.CLASSIC: "经典多因子评分",
            ScoringSystemType.TREND: "趋势强度评分",
            ScoringSystemType.BREAKOUT: "低位盘整突破评分",
        }
        report.append(f"**主评分系统：{system_names.get(rec.primary_system, '未知')}**")
        report.append("")

        # 熔断原因（如果有）
        if rec.fuse_reason:
            report.append(f"**⚠️ 熔断原因：{rec.fuse_reason}**")
            report.append("")

        # 形态覆盖原因（如果有）
        if rec.pattern_override:
            report.append(f"**🔍 形态覆盖：{rec.pattern_override}**")
            report.append("")

        # 置信度
        report.append(f"**置信度：{rec.confidence}**")
        report.append("")

        # 关键理由
        report.append("### 💡 关键理由")
        report.append("")
        for reason in rec.reasons:
            report.append(f"- {reason}")

        # 警告
        if rec.warnings:
            report.append("")
            report.append("### ⚠️ 风险警告")
            report.append("")
            for warning in rec.warnings:
                report.append(f"- {warning}")

        report.append("")
        return report

    def _generate_three_system_analysis(self, context: AnalysisContext) -> List[str]:
        """生成三系统评分对比"""
        report = []

        report.append("## 第二部分：三系统评分对比")
        report.append("")

        report.append("| 评分系统 | 最终评分 | 状态 | 说明 |")
        report.append("|----------|----------|------|------|")

        # 经典评分
        classic = context.classic_score
        classic_status = "✅ 通过" if classic.score >= 50 else "❌ 未通过"
        report.append(f"| 经典评分 | {classic.score:.1f}分 | {classic_status} | 趋势分{classic.trend_score:.1f} × 位置系数{classic.position_modifier:.2f} |")

        # 趋势评分
        trend = context.trend_score
        if trend.passed_hard_filter:
            trend_status = "✅ 通过"
            trend_note = f"时机系数{trend.timing_coefficient:.2f}({trend.timing_type})"
        else:
            trend_status = "❌ 未通过"
            trend_note = trend.hard_filter_reason
        report.append(f"| 趋势评分 | {trend.final_score:.1f}分 | {trend_status} | {trend_note} |")

        # 突破评分
        breakout = context.breakout_score
        if breakout.passed_filter:
            breakout_status = "✅ 通过"
            if breakout.has_breakout:
                breakout_note = f"盘整{breakout.consolidation_days}天后突破"
            elif breakout.is_consolidating:
                breakout_note = "盘整蓄势中"
            else:
                breakout_note = "形态不完整"
        else:
            breakout_status = "❌ 未通过"
            breakout_note = breakout.filter_reason
        report.append(f"| 突破评分 | {breakout.final_score:.1f}分 | {breakout_status} | {breakout_note} |")

        report.append("")

        # 主系统选择说明
        report.append(f"**主系统选择原因：** {context.final_recommendation.primary_system.value}")
        market = context.market_state
        regime_names = {
            MarketState.BULL: "牛市",
            MarketState.BEAR: "熊市",
            MarketState.SIDEWAY: "震荡市",
            MarketState.VOLATILE: "剧烈波动",
        }
        report.append(f"- 市场状态：{regime_names.get(market.combined_regime, '未知')}")
        report.append(f"- 综合信号：{market.combined_signal}")
        report.append("")

        return report

    def _generate_market_risk_section(self, context: AnalysisContext) -> List[str]:
        """生成市场状态与风险部分"""
        report = []
        market = context.market_state
        position = context.position_assessment
        stop_loss = context.stop_loss_config

        report.append("## 第三部分：市场状态与风险控制")
        report.append("")

        # 市场状态
        report.append("### 🌡️ 市场状态")
        report.append("")

        regime_emoji = {
            MarketState.BULL: '📈',
            MarketState.BEAR: '📉',
            MarketState.SIDEWAY: '↔️',
            MarketState.VOLATILE: '⚡'
        }
        regime_cn = {
            MarketState.BULL: '牛市',
            MarketState.BEAR: '熊市',
            MarketState.SIDEWAY: '震荡',
            MarketState.VOLATILE: '剧烈波动'
        }

        report.append(f"- **综合状态**: {regime_cn.get(market.combined_regime, '未知')} {regime_emoji.get(market.combined_regime, '➖')}")
        report.append(f"- **综合信号**: {market.combined_signal}")
        report.append(f"- **置信度**: {market.confidence*100:.0f}%")
        report.append("")

        # 位置评估
        report.append("### 📍 位置评估")
        report.append("")

        position_emoji = {'high': '🔴', 'middle': '🟡', 'low': '🟢'}
        report.append(f"- **价格位置**: {position_emoji.get(position.position, '⚪')} {position.position}")
        report.append(f"- **长期位置**: {position_emoji.get(position.long_term_position, '⚪')} {position.long_term_position}")
        report.append(f"- **短期位置**: {position_emoji.get(position.short_term_position, '⚪')} {position.short_term_position}")
        report.append(f"- **位置修正系数**: {position.position_modifier:.2f}")
        report.append(f"- **原因**: {position.reason}")
        report.append("")

        # 风险控制
        report.append("### 🛡️ 风险控制")
        report.append("")

        stop_type_cn = {
            UnifiedStopLossType.PATTERN: '形态止损',
            UnifiedStopLossType.SUPPORT: '支撑位止损',
            UnifiedStopLossType.ATR: 'ATR止损',
            UnifiedStopLossType.MA: '均线止损',
            UnifiedStopLossType.PERCENTAGE: '固定比例止损',
        }

        report.append(f"- **建议止损位**: ¥{stop_loss.stop_price:.2f} ({stop_type_cn.get(stop_loss.stop_type, '止损')})")
        report.append(f"- **止损幅度**: {stop_loss.distance_percent*100:.1f}%")
        if stop_loss.take_profit_price > 0:
            report.append(f"- **建议止盈位**: ¥{stop_loss.take_profit_price:.2f}")
            report.append(f"- **盈亏比**: {stop_loss.risk_reward_ratio:.1f}:1")
        report.append("")

        return report

    def _generate_trading_plan_v2(self, context: AnalysisContext) -> List[str]:
        """生成交易执行计划（使用统一推荐）"""
        report = []
        rec = context.final_recommendation
        stop_loss = context.stop_loss_config

        report.append("## 第四部分：交易执行计划")
        report.append("")

        # 策略类型
        report.append("### 📈 策略类型")
        report.append("")

        if rec.is_actionable():
            if rec.primary_system == ScoringSystemType.TREND:
                strategy_type = "✅ 右侧交易（趋势跟随）"
                strategy_desc = "趋势确立，跟随趋势操作"
            elif rec.primary_system == ScoringSystemType.BREAKOUT:
                strategy_type = "✅ 突破交易（形态驱动）"
                strategy_desc = "低位盘整突破，形态确认"
            else:
                strategy_type = "✅ 信号驱动（逢低布局）"
                strategy_desc = "综合评分良好，当前位置适合布局"
        else:
            if rec.action == ActionType.SELL:
                strategy_type = "🛡️ 防御型（减仓/清仓）"
                strategy_desc = "风险信号明确，建议规避"
            else:
                strategy_type = "🔍 观望型（等待机会）"
                strategy_desc = "信号不明确，等待更好的入场时机"

        report.append(f"- **类型**：{strategy_type}")
        report.append(f"- **说明**：{strategy_desc}")
        report.append("")

        # 具体点位
        report.append("### 📍 具体点位")
        report.append("")

        if rec.is_actionable():
            report.append("| 项目 | 建议数值 | 说明 |")
            report.append("|------|----------|------|")
            report.append(f"| 入场区间 | ¥{rec.entry_low:.2f} ~ ¥{rec.entry_high:.2f} | 当前价格附近 |")
            report.append(f"| 止损位 | ¥{stop_loss.stop_price:.2f} | {stop_loss.stop_type.value} |")
            if stop_loss.take_profit_price > 0:
                report.append(f"| 目标位 | ¥{stop_loss.take_profit_price:.2f} | 盈亏比{stop_loss.risk_reward_ratio:.1f}:1 |")
        else:
            report.append(f"| ⚠️ 操作建议 | **{rec.get_action_display()}** | {rec.fuse_reason or '等待更明确信号'} |")

        report.append("")

        # 仓位建议
        report.append("### 💰 仓位建议")
        report.append("")
        report.append(f"- **建议仓位**：{rec.position_size}")
        report.append("")

        # 风险提示
        if rec.warnings:
            report.append("### ⚠️ 风险提示")
            report.append("")
            for warning in rec.warnings:
                report.append(f"- {warning}")
            report.append("")

        return report

    def _generate_technical_indicators_table(
        self,
        latest: pd.Series,
        context: AnalysisContext
    ) -> List[str]:
        """生成技术指标表格"""
        report = []

        report.append("| 指标 | 数值 | 状态 |")
        report.append("|------|------|------|")

        close = latest.get('close', 0)

        # RSI
        rsi_val = latest.get('rsi_24', 50)
        if rsi_val > 70:
            rsi_desc = "超买区"
        elif rsi_val < 30:
            rsi_desc = "超卖区"
        else:
            rsi_desc = "中性"
        report.append(f"| RSI(24) | {rsi_val:.2f} | {rsi_desc} |")

        # MACD
        macd_val = latest.get('macd', 0)
        macd_desc = "多头" if macd_val > 0 else "空头"
        report.append(f"| MACD | {macd_val:.2f} | {macd_desc} |")

        # KDJ
        k = latest.get('kdj_k', 0)
        d = latest.get('kdj_d', 0)
        j = latest.get('kdj_j', 0)
        kdj_desc = "J值偏高" if j > 80 else "J值偏低" if j < 20 else "正常"
        report.append(f"| KDJ | K: {k:.2f} / D: {d:.2f} / J: {j:.2f} | {kdj_desc} |")

        # 均线
        ma20 = latest.get('ma_20', 0)
        ma50 = latest.get('ma_50', 0)
        ma200 = latest.get('ma_200', 0)
        ma200_str = f"{ma200:.2f}" if not pd.isna(ma200) else "无数据"
        report.append(f"| 移动平均线 | MA20: ¥{ma20:.2f} / MA50: ¥{ma50:.2f} / MA200: {ma200_str} | 趋势参考 |")

        # 布林带
        boll_upper = latest.get('boll_upper', close)
        boll_mid = latest.get('boll_mid', close)
        boll_lower = latest.get('boll_lower', close)
        boll_pctb = (close - boll_lower) / (boll_upper - boll_lower) if boll_upper > boll_lower else 0.5
        boll_desc = "触及上轨" if close >= boll_upper else "触及下轨" if close <= boll_lower else "中轨附近"
        report.append(f"| 布林带 | 上轨: ¥{boll_upper:.2f} / 中轨: ¥{boll_mid:.2f} / 下轨: ¥{boll_lower:.2f} | {boll_desc} |")

        # BIAS 乖离率
        bias_6 = latest.get('bias_6', 0)
        bias_12 = latest.get('bias_12', 0)
        bias_24 = latest.get('bias_24', 0)
        bias_desc = "超买" if bias_6 > 5 else "超卖" if bias_6 < -5 else "正常"
        report.append(f"| BIAS(乖离率) | BIAS6: {bias_6:.2f}% / BIAS12: {bias_12:.2f}% / BIAS24: {bias_24:.2f}% | {bias_desc} |")

        # DMI 动向指标
        pdi = latest.get('dmi_pdi', 0)
        mdi = latest.get('dmi_mdi', 0)
        adx = latest.get('dmi_adx', 0)
        if pdi > mdi and adx > 25:
            dmi_desc = "多头趋势强"
        elif mdi > pdi and adx > 25:
            dmi_desc = "空头趋势强"
        elif adx < 20:
            dmi_desc = "趋势不明"
        else:
            dmi_desc = "多空平衡"
        report.append(f"| DMI(动向指标) | PDI: {pdi:.2f} / MDI: {mdi:.2f} / ADX: {adx:.2f} | {dmi_desc} |")

        # ATR 真实波幅
        atr = latest.get('atr_14', 0)
        atr_pct = (atr / close * 100) if close > 0 else 0
        atr_desc = "高波动" if atr_pct > 3 else "低波动" if atr_pct < 1 else "正常波动"
        report.append(f"| ATR(真实波幅) | {atr:.2f} ({atr_pct:.2f}%) | {atr_desc} |")

        # CCI 顺势指标
        cci = latest.get('cci', 0)
        if cci > 100:
            cci_desc = "超买区"
        elif cci < -100:
            cci_desc = "超卖区"
        else:
            cci_desc = "正常区间"
        report.append(f"| CCI(顺势指标) | {cci:.2f} | {cci_desc} |")

        # WR 威廉指标
        wr = latest.get('wr', 0)
        wr_6 = latest.get('wr_6', 0)
        if wr < -80:
            wr_desc = "超卖区"
        elif wr > -20:
            wr_desc = "超买区"
        else:
            wr_desc = "正常区间"
        report.append(f"| WR(威廉指标) | WR14: {wr:.2f} / WR6: {wr_6:.2f} | {wr_desc} |")

        # TRIX 三重平滑移动平均
        trix = latest.get('trix', 0)
        trix_ma = latest.get('trix_ma', 0)
        trix_desc = "多头" if trix > trix_ma else "空头"
        report.append(f"| TRIX | TRIX: {trix:.4f} / TRMA: {trix_ma:.4f} | {trix_desc} |")

        # OBV 能量潮
        obv = latest.get('obv', 0)
        obv_desc = "资金流入" if obv > 0 else "资金流出"
        report.append(f"| OBV(能量潮) | {obv:,.0f} | {obv_desc} |")

        # MFI 资金流量指标
        mfi = latest.get('mfi', 50)
        if mfi > 80:
            mfi_desc = "资金超买"
        elif mfi < 20:
            mfi_desc = "资金超卖"
        else:
            mfi_desc = "正常"
        report.append(f"| MFI(资金流量) | {mfi:.2f} | {mfi_desc} |")

        # VR 容量比率
        vr = latest.get('vr', 100)
        if vr > 350:
            vr_desc = "超买区"
        elif vr < 70:
            vr_desc = "超卖区"
        else:
            vr_desc = "正常区间"
        report.append(f"| VR(容量比率) | {vr:.2f} | {vr_desc} |")

        # PSY 心理线
        psy = latest.get('psy', 50)
        if psy > 75:
            psy_desc = "超买区"
        elif psy < 25:
            psy_desc = "超卖区"
        else:
            psy_desc = "正常区间"
        report.append(f"| PSY(心理线) | {psy:.2f}% | {psy_desc} |")

        # BBI 多空指标
        bbi = latest.get('bbi', close)
        bbi_desc = "多头排列" if close > bbi else "空头排列"
        report.append(f"| BBI(多空指标) | ¥{bbi:.2f} | {bbi_desc} |")

        report.append("")
        return report

    def _analyze_signal_conflicts(self, latest_data: pd.Series, strategies_results: Dict,
                                   df: pd.DataFrame = None,
                                   candlestick_result: Dict = None) -> List[str]:
        """
        分析指标信号冲突情况（改进版：加入动量因子和K线形态）

        检测MACD、TRIX、RSI、DMI等指标之间的方向一致性，
        同时考虑MACD动量变化和成交量配合，
        当存在矛盾信号时给出解释。

        修复：
        1. DMI指标当PDI≈MDI时显示"多空平衡"而非强行归类
        2. 增加信号明细表格，列出每个指标的判断结果
        3. K线形态纳入信号统计（强晨星等计入看多信号）
        """
        conflicts = []

        # DMI多空平衡阈值
        DMI_BALANCE_THRESHOLD = 0.5

        # 获取各指标当前值
        macd = latest_data.get('macd', 0)
        macd_dif = latest_data.get('macd_dif', 0)
        macd_dea = latest_data.get('macd_dea', 0)
        trix = latest_data.get('trix', 0)
        rsi = latest_data.get('rsi_24', 50)
        pdi = latest_data.get('dmi_pdi', 0)
        mdi = latest_data.get('dmi_mdi', 0)
        adx = latest_data.get('dmi_adx', 0)
        cci = latest_data.get('cci', 0)

        # 判断各指标方向（多头/空头/中性）
        macd_bull = macd > 0 and macd_dif > macd_dea  # MACD柱状图为正且DIF在DEA上方
        trix_bull = trix > 0  # TRIX为正表示向上
        rsi_bull = rsi > 50  # RSI大于50偏多
        rsi_neutral = 45 <= rsi <= 55  # RSI接近50视为中性

        # 修复：DMI增加多空平衡判断
        dmi_diff = abs(pdi - mdi)
        dmi_bull = pdi > mdi and dmi_diff >= DMI_BALANCE_THRESHOLD
        dmi_bear = mdi > pdi and dmi_diff >= DMI_BALANCE_THRESHOLD
        dmi_neutral = dmi_diff < DMI_BALANCE_THRESHOLD  # PDI≈MDI时视为平衡

        cci_bull = cci > 0  # CCI为正表示强势

        # === 新增：动量衰减和量能分析 ===
        macd_declining = False
        volume_weak = False

        if df is not None and len(df) >= 2:
            # MACD动量衰减检测：当前MACD柱状图 < 前一日
            macd_hist_current = df['macd'].iloc[-1] if 'macd' in df.columns else macd
            macd_hist_prev = df['macd'].iloc[-2] if 'macd' in df.columns else macd_hist_current
            macd_declining = macd_hist_current < macd_hist_prev

            # 量能不足检测：当前成交量 < 5日均量
            vol_current = latest_data.get('volume', 0)
            vol_ma5 = df['volume'].tail(5).mean() if 'volume' in df.columns and len(df) >= 5 else vol_current
            volume_weak = vol_current < vol_ma5

        # === 新增：K线形态信号统计 ===
        candlestick_bull = False
        candlestick_bear = False
        candlestick_neutral = False
        pattern_names = []

        if candlestick_result and 'patterns' in candlestick_result:
            patterns = candlestick_result['patterns']
            strong_bullish = ["晨星", "看涨吞没", "白色三兵"]
            medium_bullish = ["锤子线", "倒锤子", "穿刺线", "大阳线"]
            strong_bearish = ["暮星", "看跌吞没", "黑色三鸦"]
            medium_bearish = ["流星线", "吊颈线", "乌云盖顶", "大阴线"]

            for p in patterns:
                name = p.get('name', '')
                strength = p.get('strength', '')
                ptype = p.get('type', '')
                pattern_names.append(f"{name}({strength})")

                # 强形态计入信号统计
                if name in strong_bullish or (name in medium_bullish and strength == '强'):
                    candlestick_bull = True
                elif name in strong_bearish or (name in medium_bearish and strength == '强'):
                    candlestick_bear = True
                elif ptype == 'neutral':
                    candlestick_neutral = True

        # 统计原始信号数量（5个技术指标 + K线形态）
        technical_bull = sum([macd_bull, trix_bull, rsi_bull, dmi_bull, cci_bull])
        technical_bear = sum([not macd_bull, not trix_bull, not rsi_bull, dmi_bear, not cci_bull])

        # 总信号统计（包含K线形态）
        bull_signals = technical_bull + (1 if candlestick_bull else 0)
        bear_signals = technical_bear + (1 if candlestick_bear else 0)
        neutral_signals = (1 if dmi_neutral else 0) + (1 if rsi_neutral else 0) + (1 if candlestick_neutral else 0)

        # === 新增：信号明细表格 ===
        conflicts.append("📊 **指标信号明细：**")
        conflicts.append("")
        conflicts.append("| 指标 | 当前值 | 判断 | 说明 |")
        conflicts.append("|------|--------|------|------|")

        # MACD
        macd_status = "看多✅" if macd_bull else "看空❌"
        macd_note = f"柱状图{macd:.3f}"
        conflicts.append(f"| MACD | DIF:{macd_dif:.2f} DEA:{macd_dea:.2f} | {macd_status} | {macd_note} |")

        # TRIX
        trix_status = "看多✅" if trix_bull else "看空❌"
        conflicts.append(f"| TRIX | {trix:.3f} | {trix_status} | {'向上' if trix_bull else '向下'} |")

        # RSI
        if rsi_neutral:
            rsi_status = "中性➖"
        else:
            rsi_status = "看多✅" if rsi_bull else "看空❌"
        conflicts.append(f"| RSI(24) | {rsi:.1f} | {rsi_status} | {'偏多' if rsi > 60 else '偏空' if rsi < 40 else '中性'} |")

        # DMI（修复：显示多空平衡）
        if dmi_neutral:
            dmi_status = "平衡⚖️"
            dmi_note = f"PDI≈MDI({dmi_diff:.2f}差值)，多空力量均衡"
        elif dmi_bull:
            dmi_status = "看多✅"
            dmi_note = f"PDI({pdi:.1f}) > MDI({mdi:.1f})"
        else:
            dmi_status = "看空❌"
            dmi_note = f"MDI({mdi:.1f}) > PDI({pdi:.1f})"
        conflicts.append(f"| DMI | PDI:{pdi:.1f} MDI:{mdi:.1f} ADX:{adx:.1f} | {dmi_status} | {dmi_note} |")

        # CCI
        cci_status = "看多✅" if cci_bull else "看空❌"
        cci_note = "强势" if cci > 100 else "弱势" if cci < -100 else "正常"
        conflicts.append(f"| CCI | {cci:.1f} | {cci_status} | {cci_note} |")

        # K线形态
        if pattern_names:
            pattern_str = ", ".join(pattern_names[:3])  # 最多显示3个
            if candlestick_bull:
                pattern_status = "看多✅"
            elif candlestick_bear:
                pattern_status = "看空❌"
            else:
                pattern_status = "中性➖"
            conflicts.append(f"| K线形态 | - | {pattern_status} | {pattern_str} |")

        conflicts.append("")

        # 统计原始信号数量
        bull_signals = sum([macd_bull, trix_bull, rsi_bull, dmi_bull, cci_bull])
        bear_signals = 5 - bull_signals

        # === 新增：信号强度调整 ===
        # 如果MACD看多但动能衰减且量能不足，视为"弱势偏多"或"中性"
        signal_strength = "normal"
        if macd_bull and macd_declining and volume_weak:
            signal_strength = "weak_bullish"
            bull_signals = max(0, bull_signals - 1)  # 降低多头信号计数

        # 检测主要冲突
        if macd_bull and not trix_bull:
            conflicts.append("⚠️ **MACD与TRIX矛盾**：")
            conflicts.append(f"   - MACD为正（{macd:.2f}），显示短期动量偏多")
            conflicts.append(f"   - TRIX为负（{trix:.2f}），显示中长期趋势仍向下")
            conflicts.append("   - **解读**：MACD反弹可能是短期技术性修复，需观察TRIX是否拐头向上确认趋势反转")
            conflicts.append("")

        if not macd_bull and trix_bull:
            conflicts.append("⚠️ **MACD与TRIX矛盾**：")
            conflicts.append(f"   - MACD为负（{macd:.2f}），显示短期动量偏弱")
            conflicts.append(f"   - TRIX为正（{trix:.2f}），显示中长期趋势向上")
            conflicts.append("   - **解读**：短期回调不改中长期上升趋势，可关注支撑位低吸机会")
            conflicts.append("")

        # 检测RSI与价格趋势背离
        if rsi > 70 and not macd_bull:
            conflicts.append("⚠️ **RSI超买与MACD背离**：")
            conflicts.append(f"   - RSI处于超买区（{rsi:.1f}）")
            conflicts.append(f"   - 但MACD未同步走强（{macd:.2f}）")
            conflicts.append("   - **解读**：上涨动能可能衰竭，警惕冲高回落风险")
            conflicts.append("")

        # === 新增：检测MACD动量衰减 ===
        if macd > 0 and macd_declining and volume_weak:
            conflicts.append("⚠️ **信号混合：趋势虽在均线上方，但动能衰减，量能不足**：")
            conflicts.append(f"   - MACD为正（{macd:.2f}），显示短期趋势偏多")
            conflicts.append(f"   - 但MACD柱状图下降（{macd_hist_current:.2f} < {macd_hist_prev:.2f}），动能减弱")
            conflicts.append(f"   - 成交量萎缩（{vol_current:,.0f} < {vol_ma5:,.0f}），市场参与度下降")
            conflicts.append("   - **解读**：属于弱势整理，建议观望等待量能配合")
            conflicts.append("")

        # 检测DMI与MACD一致性
        if dmi_bull and not macd_bull:
            conflicts.append("⚠️ **DMI与MACD矛盾**：")
            conflicts.append(f"   - DMI显示多头占优（PDI{pdi:.1f} > MDI{mdi:.1f}）")
            conflicts.append(f"   - 但MACD显示空头（{macd:.2f}）")
            conflicts.append("   - **解读**：多方力量占优但动能不足，可能进入盘整阶段")
            conflicts.append("")

        # === 新增：动量衰减与量能不足检测 ===
        if signal_strength == "weak_bullish":
            conflicts.append("⚠️ **信号混合：趋势与动能背离**：")
            conflicts.append(f"   - MACD为正（{macd:.2f}），显示趋势在均线上方")
            conflicts.append(f"   - 但MACD柱状图下降（{macd_hist_current:.3f} < {macd_hist_prev:.3f}），动能衰减")
            conflicts.append(f"   - 成交量不足（{vol_current:,.0f} < 均量{vol_ma5:,.0f}），资金参与度低")
            conflicts.append("   - **解读**：属于弱势整理状态，不宜盲目追多")
            conflicts.append("")

        # 信号一致性总结（改进版：包含K线形态和DMI平衡状态）
        total_signals = 6 if candlestick_result and pattern_names else 5  # 5技术指标 + 可选K线形态

        if dmi_neutral:
            dmi_balance_note = f"（注：DMI显示多空平衡，PDI≈MDI差值仅{dmi_diff:.2f}）"
        else:
            dmi_balance_note = ""

        if signal_strength == "weak_bullish":
            conflicts.append(f"📊 **信号一致性：中性偏弱**（技术看多{technical_bull} vs 看空{technical_bear}，K线{'看多' if candlestick_bull else '看空' if candlestick_bear else '中性'}）")
            if dmi_balance_note:
                conflicts.append(f"   {dmi_balance_note}")
            conflicts.append("   趋势虽在均线上方，但动能衰减，量能不足，建议观望")
            conflicts.append("")
        elif bull_signals >= 4:
            conflicts.append(f"✅ **信号一致性：偏多**（{bull_signals}/{total_signals}个信号看多，含K线形态）")
            if pattern_names and candlestick_bull:
                conflicts.append(f"   K线形态强力支持：{', '.join(pattern_names[:2])}")
            conflicts.append("   多数指标方向一致，趋势较为明确")
            conflicts.append("")
        elif bear_signals >= 4:
            conflicts.append(f"⚠️ **信号一致性：偏空**（{bear_signals}/{total_signals}个信号看空）")
            if pattern_names and candlestick_bear:
                conflicts.append(f"   K线形态看跌：{', '.join(pattern_names[:2])}")
            conflicts.append("   多数指标方向一致，需警惕下行风险")
            conflicts.append("")
        elif dmi_neutral and neutral_signals >= 2:
            conflicts.append(f"📊 **信号一致性：中性震荡**（看多{bull_signals} vs 看空{bear_signals}，中性信号{neutral_signals}个）")
            conflicts.append(f"   {dmi_balance_note}")
            conflicts.append("   市场处于震荡整理阶段，建议观望等待方向明确")
            conflicts.append("")
        else:
            conflicts.append(f"📊 **信号一致性：混合**（看多{bull_signals} vs 看空{bear_signals}）")
            if dmi_balance_note:
                conflicts.append(f"   {dmi_balance_note}")
            conflicts.append("   指标信号分歧较大，建议观望等待方向明确")
            conflicts.append("")

        return conflicts

    def generate_report(self, df: pd.DataFrame, strategies_results: Dict, symbol: str) -> str:
        """
        Generate comprehensive analysis report using four-section architecture
        """
        if df.empty:
            return "No data available for report generation"

        latest_data = df.iloc[-1]
        report = []

        # 基本信息
        report.append(f"# 股票技术分析报告：{symbol}")
        report.append("")
        report.append(f"**分析日期：** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("")

        # 获取评分数据
        scoring = strategies_results.get('scoring', {})
        if not scoring or 'error' in scoring:
            report.append("*评分系统未返回有效结果*")
            return "\n".join(report)

        score = scoring.get('score', 50)

        # 使用增强的位置判断方法（综合均线位置 + 技术超买超卖指标）
        # 优先使用 scoring 中的位置修正系数，避免双重判断
        position_modifier = scoring.get('position_modifier', 1.0)
        position_info = self._determine_position_status(latest_data, position_modifier)

        # 保留原有的 price_ratio 转换逻辑（兼容后续代码）
        position_info['price_ratio_display'] = position_info['price_ratio'] * 50

        # 获取K线形态评估（定性）
        candlestick_recognizer = TalibPatternRecognizer()
        candlestick_result = candlestick_recognizer.recognize_all(df, lookback=5)

        # 将形态结果转换为评估字典
        pattern_assessment = self._convert_pattern_to_assessment(candlestick_result, df)

        # ============ 四部分报告架构 ============

        # 第一部分：核心结论区
        report.extend(self._generate_core_conclusion(scoring, position_info, pattern_assessment))

        # 新增：三因子组评分表格
        report.extend(self._generate_factor_groups_score(scoring, strategies_results))

        # 新增：市场状态与风险控制
        report.extend(self._generate_market_and_risk_info(strategies_results))

        # 新增：趋势评分系统报告
        report.extend(self._generate_trend_scoring_report(strategies_results))

        # 第二部分：多维信号共振分析
        report.extend(self._generate_signal_resonance(latest_data, position_info, pattern_assessment, scoring, df))

        # 第三部分：量化评分与因子拆解
        report.extend(self._generate_score_breakdown(scoring))

        # 第四部分：交易执行计划
        report.extend(self._generate_trading_plan(scoring, position_info, pattern_assessment, latest_data))

        # 原始信号明细（保留在附录）
        report.append("---")
        report.append("")
        report.append("## 附录：原始技术指标")
        report.append("")
        report.append(f"| 指标 | 数值 | 状态 |")
        report.append(f"|------|------|------|")

        # RSI
        rsi_val = latest_data.get('rsi_24', 50)
        if rsi_val > 70:
            rsi_desc = "超买区"
        elif rsi_val < 30:
            rsi_desc = "超卖区"
        else:
            rsi_desc = "中性"
        report.append(f"| RSI(24) | {rsi_val:.2f} | {rsi_desc} |")

        # MACD
        macd_val = latest_data['macd']
        macd_desc = "多头" if macd_val > 0 else "空头"
        report.append(f"| MACD | {macd_val:.2f} | {macd_desc} |")

        # KDJ
        k, d, j = latest_data['kdj_k'], latest_data['kdj_d'], latest_data['kdj_j']
        kdj_desc = "J值偏高" if j > 80 else "J值偏低" if j < 20 else "正常"
        report.append(f"| KDJ | K: {k:.2f} / D: {d:.2f} / J: {j:.2f} | {kdj_desc} |")

        # MA - 从 position_info 获取值
        ma20 = position_info.get('ma20', 0)
        ma50 = position_info.get('ma50', 0)
        close = position_info.get('close', 0)
        ma200 = latest_data.get('ma_200', np.nan)
        ma200_str = f"{ma200:.2f}" if not pd.isna(ma200) else "无数据"
        report.append(f"| 移动平均线 | MA20: ¥{ma20:.2f} / MA50: ¥{ma50:.2f} / MA200: {ma200_str} | 趋势参考 |")

        # BOLL
        bu, bm, bl = latest_data['boll_upper'], latest_data['boll_mid'], latest_data['boll_lower']
        boll_desc = "触及上轨" if close >= bu else "触及下轨" if close <= bl else "中轨附近"
        report.append(f"| 布林带 | 上轨: ¥{bu:.2f} / 中轨: ¥{bm:.2f} / 下轨: ¥{bl:.2f} | {boll_desc} |")

        # CCI
        cci_val = latest_data['cci']
        if cci_val > 200:
            cci_desc = "严重超买区"
        elif cci_val > 100:
            cci_desc = "超买区"
        elif cci_val < -200:
            cci_desc = "严重超卖区"
        elif cci_val < -100:
            cci_desc = "超卖区"
        else:
            cci_desc = "正常区间"
        report.append(f"| CCI | {cci_val:.2f} | {cci_desc} |")

        # DMI
        pdi, mdi, adx = latest_data['dmi_pdi'], latest_data['dmi_mdi'], latest_data['dmi_adx']
        dmi_diff = abs(pdi - mdi)
        DMI_BALANCE_THRESHOLD = 0.5
        if dmi_diff < DMI_BALANCE_THRESHOLD:
            dmi_desc = f"多空平衡（差值{dmi_diff:.2f}）"
        elif pdi > mdi:
            dmi_desc = "多头占优"
        else:
            dmi_desc = "空头占优"
        report.append(f"| DMI | PDI: {pdi:.2f} / MDI: {mdi:.2f} / ADX: {adx:.2f} | {dmi_desc} |")

        # WR
        wr_val = latest_data['wr']
        if wr_val < 10:
            wr_desc = "严重超买"
        elif wr_val > 90:
            wr_desc = "严重超卖"
        else:
            wr_desc = "正常区间"
        report.append(f"| WR | {wr_val:.2f} | {wr_desc} |")

        # BIAS
        bias6 = latest_data['bias_6']
        bias12 = latest_data['bias_12']
        bias24 = latest_data['bias_24']
        bias_avg = (bias6 + bias12 + bias24) / 3
        if bias_avg > 5:
            bias_desc = "超买偏离"
        elif bias_avg < -5:
            bias_desc = "超卖偏离"
        else:
            bias_desc = "正常区间"
        report.append(f"| BIAS | BIAS6: {bias6:.2f}% / BIAS12: {bias12:.2f}% / BIAS24: {bias24:.2f}% | {bias_desc} |")

        report.append("")

        # 添加K线形态报告（如果有）
        if candlestick_result and candlestick_result.patterns:
            report.append("### K线形态识别")
            report.append("")
            candlestick_report = format_patterns_report(candlestick_result)
            report.append(candlestick_report)

        report.append("")
        report.append("> **免责声明：** 本分析仅供学习参考，不构成投资建议。投资决策应基于全面研究和独立判断。")
        report.append("")

        return "\n".join(report)

    def _determine_position_status(self, latest_data: pd.Series, position_modifier: float = 1.0) -> Dict:
        """
        综合判断位置状态（改进版：多指标交叉验证）

        核心改进：
        1. 区分"短线偏热"和"全面超买"
        2. 需要多个指标一致确认才算超买/超卖
        3. 均线乖离率作为重要参考

        返回: {
            'position': 'high'/'low'/'middle',
            'price_ratio': float,
            'is_overbought': bool,
            'is_oversold': bool,
            'is_extreme_overbought': bool,
            'is_extreme_oversold': bool,
            'is_short_term_hot': bool,  # 新增：短线偏热（但不一定是全面超买）
            'reason': str,
            'close': float,
            'ma20': float,
            'ma50': float
        }
        """
        close = latest_data.get('close', 0)
        ma20 = latest_data.get('ma_20', close)
        ma50 = latest_data.get('ma_50', close)

        # 1. 均线位置判断（核心参考）
        avg_ma = (ma20 + ma50) / 2 if ma20 > 0 and ma50 > 0 else close
        price_ratio = close / avg_ma if avg_ma > 0 else 1.0
        bias20 = (close / ma20 - 1) if ma20 > 0 else 0  # MA20乖离率
        bias50 = (close / ma50 - 1) if ma50 > 0 else 0  # MA50乖离率

        # 2. 技术位置判断（基于超买超卖指标）
        wr = latest_data.get('wr_14', latest_data.get('wr', 50))
        cci = latest_data.get('cci', 0)
        rsi = latest_data.get('rsi_24', 50)
        boll_upper = latest_data.get('boll_upper', 0)
        boll_lower = latest_data.get('boll_lower', 0)
        boll_pctb = (close - boll_lower) / (boll_upper - boll_lower) if boll_upper > boll_lower else 0.5

        # 3. 多指标交叉验证（改进：需要多个指标一致确认）
        # 计算超买信号数量
        overbought_signals = 0
        overbought_details = []
        if wr < 20:
            overbought_signals += 1
            overbought_details.append(f"WR={wr:.0f}")
        if rsi > 70:
            overbought_signals += 1
            overbought_details.append(f"RSI={rsi:.0f}")
        if cci > 100:
            overbought_signals += 1
            overbought_details.append(f"CCI={cci:.0f}")
        if boll_pctb > 0.85:
            overbought_signals += 1
            overbought_details.append(f"布林{boll_pctb*100:.0f}%")
        if bias20 > 0.08:
            overbought_signals += 1
            overbought_details.append(f"BIAS20={bias20*100:.1f}%")

        # 计算超卖信号数量
        oversold_signals = 0
        oversold_details = []
        if wr > 80:
            oversold_signals += 1
            oversold_details.append(f"WR={wr:.0f}")
        if rsi < 30:
            oversold_signals += 1
            oversold_details.append(f"RSI={rsi:.0f}")
        if cci < -100:
            oversold_signals += 1
            oversold_details.append(f"CCI={cci:.0f}")
        if boll_pctb < 0.15:
            oversold_signals += 1
            oversold_details.append(f"布林{boll_pctb*100:.0f}%")
        if bias20 < -0.08:
            oversold_signals += 1
            oversold_details.append(f"BIAS20={bias20*100:.1f}%")

        # 4. 综合判断（改进：需要至少2个指标一致确认）
        # 极端超买：至少3个指标确认 + MA20乖离率 > 5%
        is_extreme_overbought = (
            overbought_signals >= 3 and
            bias20 > 0.05
        )

        # 普通超买：至少2个指标确认 + 均线乖离率为正
        is_overbought = (
            overbought_signals >= 2 and
            bias20 > 0 and
            not is_extreme_overbought
        )

        # 短线偏热：只有1个指标显示超买（如WR<20），但其他指标不确认
        is_short_term_hot = (
            overbought_signals == 1 and
            not is_overbought and
            not is_extreme_overbought
        )

        # 极端超卖：至少3个指标确认 + MA20乖离率 < -5%
        is_extreme_oversold = (
            oversold_signals >= 3 and
            bias20 < -0.05
        )

        # 普通超卖：至少2个指标确认 + 均线乖离率为负
        is_oversold = (
            oversold_signals >= 2 and
            bias20 < 0 and
            not is_extreme_oversold
        )

        # 5. 位置判断（综合均线和技术指标）
        # 关键改进：均线位置优先，技术指标辅助
        if is_extreme_oversold:
            position = 'low'
            reason = f"全面超卖（{', '.join(oversold_details[:3])}）"
        elif is_extreme_overbought:
            position = 'high'
            reason = f"全面超买（{', '.join(overbought_details[:3])}）"
        elif is_oversold:
            position = 'low'
            reason = f"技术指标超卖（{', '.join(oversold_details[:2])}）"
        elif is_overbought:
            position = 'high'
            reason = f"技术指标超买（{', '.join(overbought_details[:2])}）"
        # 均线位置判断（更重要的参考）
        elif bias20 < -0.05:
            position = 'low'
            reason = f"均线位置偏低（MA20乖离率={bias20*100:.1f}%）"
        elif bias20 > 0.05:
            position = 'high'
            reason = f"均线位置偏高（MA20乖离率={bias20*100:.1f}%）"
        # 布林带位置判断
        elif boll_pctb < 0.25:
            position = 'low'
            reason = f"布林带下轨附近（位置={boll_pctb*100:.0f}%）"
        elif boll_pctb > 0.75:
            position = 'high'
            reason = f"布林带上轨附近（位置={boll_pctb*100:.0f}%）"
        else:
            position = 'middle'
            reason = f"位置适中（MA20乖离率={bias20*100:.1f}%）"

        return {
            'position': position,
            'price_ratio': price_ratio,
            'is_overbought': is_overbought or is_extreme_overbought,
            'is_oversold': is_oversold or is_extreme_oversold,
            'is_extreme_overbought': is_extreme_overbought,
            'is_extreme_oversold': is_extreme_oversold,
            'is_short_term_hot': is_short_term_hot,  # 新增
            'reason': reason,
            'close': close,
            'ma20': ma20,
            'ma50': ma50,
            'wr': wr,
            'cci': cci,
            'rsi': rsi,
            'boll_pctb': boll_pctb,
            'bias20': bias20,
            'overbought_signals': overbought_signals,
            'oversold_signals': oversold_signals
        }

    def _convert_pattern_to_assessment(self, candlestick_result, df: pd.DataFrame) -> Dict:
        """
        将K线形态识别结果转换为形态评估字典
        """
        # 兼容 AllPatternsResult 对象和字典
        if hasattr(candlestick_result, 'patterns'):
            patterns = candlestick_result.patterns
        else:
            patterns = candlestick_result.get('patterns', [])

        if not patterns:
            return {
                'type': 'none',
                'strength': 0,
                'patterns': [],
                'stop_loss': None
            }

        # 中文强度值映射为数字
        strength_map = {'极强': 5, '强': 4, '中': 2, '弱': 1}

        # 按强度排序（使用映射后的数字值）
        def get_strength_value(p):
            s = p.strength if hasattr(p, 'strength') else p.get('strength', 1)
            return strength_map.get(s, 1) if isinstance(s, str) else int(s)

        sorted_patterns = sorted(patterns, key=get_strength_value, reverse=True)
        strongest = sorted_patterns[0]

        pattern_name = strongest.name if hasattr(strongest, 'name') else strongest.get('name', '')
        pattern_type = strongest.type if hasattr(strongest, 'type') else strongest.get('type', '')

        # 中文强度值映射为数字
        strength_raw = strongest.strength if hasattr(strongest, 'strength') else strongest.get('strength', 1)
        strength = strength_map.get(strength_raw, 1) if isinstance(strength_raw, str) else int(strength_raw)

        # 计算止损位（基于形态最低点）
        stop_loss = None
        if len(df) >= 3:
            recent_low = df['low'].iloc[-3:].min()
            stop_loss = recent_low * 0.98  # 形态最低点下方2%

        # 映射类型
        if pattern_type == 'bullish':
            assessment_type = 'bullish_reversal'
        elif pattern_type == 'bearish':
            assessment_type = 'bearish_reversal'
        else:
            assessment_type = 'neutral'

        return {
            'type': assessment_type,
            'strength': strength,
            'patterns': [p.name if hasattr(p, 'name') else p.get('name', '') for p in sorted_patterns[:3]],
            'stop_loss': stop_loss,
            'description': strongest.description if hasattr(strongest, 'description') else strongest.get('description', '')
        }

    def analyze_stock(self, symbol: str, days: int = 360) -> str:
        """
        Main method to analyze a stock completely
        """
        print(f"开始分析 {symbol}...")

        # Get stock data
        df = self.get_stock_data(symbol, days)
        if df.empty:
            return f"无法获取 {symbol} 的数据"

        # Calculate technical indicators
        df_with_indicators = self.calculate_technical_indicators(df)

        # Run trading strategies
        strategies_results = self.run_trading_strategies(df_with_indicators, symbol)

        # Generate comprehensive report using four-section architecture
        report = self.generate_report(df_with_indicators, strategies_results, symbol)

        return report

    def _add_detailed_reasoning(self, report: List[str], score: float,
                                factors_raw: Dict, factors_score: Dict,
                                latest_data: pd.Series, trigger_type: str) -> None:
        """
        添加详细的结论理由说明

        Args:
            report: 报告内容列表
            score: 综合评分
            factors_raw: 因子原始值
            factors_score: 因子得分
            latest_data: 最新数据
            trigger_type: 触发类型
        """
        report.append("")
        report.append("**📋 理由说明：**")
        report.append("")

        # 1. 分析主要支撑和拖累因子
        factor_scores_list = [
            ('trend1', '趋势偏离', factors_score.get('trend1', 0)),
            ('trend2', '均线斜率', factors_score.get('trend2', 0)),
            ('mom1', 'MACD动量', factors_score.get('mom1', 0)),
            ('mom2', 'RSI位置', factors_score.get('mom2', 0)),
            ('flow1', '成交量比', factors_score.get('flow1', 0)),
            ('flow2', 'OBV资金流', factors_score.get('flow2', 0)),
            ('pos1', '布林带位置', factors_score.get('pos1', 0)),
            ('pos2', '波动率位置', factors_score.get('pos2', 0)),
            ('bias20', 'MA20乖离率', factors_score.get('bias20', 0)),
        ]

        # 排序找出最强和最弱因子
        factor_scores_list.sort(key=lambda x: x[2], reverse=True)

        strong_factors = [f for f in factor_scores_list if f[2] >= 0.7][:2]
        weak_factors = [f for f in factor_scores_list if f[2] <= 0.4][:2]

        if strong_factors:
            report.append("✅ **支撑因素：**")
            for key, name, val in strong_factors:
                raw_val = factors_raw.get(key, 0)
                report.append(f"   - {name}表现良好（得分{val*100:.0f}分）")
                # 添加具体说明
                if key == 'trend1' and raw_val > 0:
                    report.append(f"     股价在MA20上方{raw_val*100:.1f}%，短期趋势偏多")
                elif key == 'mom1' and raw_val > 0:
                    report.append(f"     MACD动量增强，柱状图3日变化+{raw_val:.2f}")
                elif key == 'flow2' and raw_val > 0:
                    report.append(f"     OBV资金流入，5日变化率+{raw_val:.1f}%")
                elif key == 'bias20' and abs(raw_val) < 0.05:
                    report.append(f"     乖离率适中（{raw_val*100:.2f}%），无超买超卖风险")
            report.append("")

        if weak_factors:
            report.append("⚠️ **拖累因素：**")
            for key, name, val in weak_factors:
                raw_val = factors_raw.get(key, 0)
                report.append(f"   - {name}表现较弱（得分{val*100:.0f}分）")
                # 添加具体说明
                if key == 'trend2' and raw_val < 0:
                    report.append(f"     短期均线斜率为负，上升趋势放缓")
                elif key == 'mom1' and raw_val < 0:
                    report.append(f"     MACD动量减弱，柱状图3日变化{raw_val:.2f}")
                elif key == 'mom2':
                    rsi_val = latest_data.get('rsi_24', 50)
                    if rsi_val > 70:
                        report.append(f"     RSI处于超买区（{rsi_val:.1f}），短期回调风险")
                    elif rsi_val < 40:
                        report.append(f"     RSI偏弱（{rsi_val:.1f}），动能不足")
                elif key == 'flow1' and raw_val < 0:
                    report.append(f"     成交量萎缩，市场参与度下降")
                elif key == 'bias20' and raw_val > 0.05:
                    report.append(f"     乖离率偏高（{raw_val*100:.2f}%），有回归均线需求")
            report.append("")

        # 2. 触发信号说明
        if trigger_type != 'none':
            report.append(f"📊 **触发信号：** 检测到{trigger_type}型信号，")
            if trigger_type == 'breakout':
                report.append("   股价放量突破近期高点，显示多方力量增强")
            elif trigger_type == 'pullback':
                report.append("   缩量回踩均线后收阳，显示支撑有效")
            report.append("")

        # 3. 关键价位提示
        close = latest_data.get('close', 0)
        ma20 = latest_data.get('ma_20', 0)
        boll_upper = latest_data.get('boll_upper', 0)
        boll_lower = latest_data.get('boll_lower', 0)

        report.append("📍 **关键价位：**")
        if ma20 > 0:
            report.append(f"   - MA20支撑位：¥{ma20:.2f}")
        if boll_upper > 0 and boll_lower > 0:
            report.append(f"   - 布林带上轨压力位：¥{boll_upper:.2f}")
            report.append(f"   - 布林带下轨支撑位：¥{boll_lower:.2f}")
        report.append("")

        # 4. 综合判断
        report.append("🔍 **综合判断：**")
        if score >= 75:
            report.append(f"   综合评分{score:.1f}分，各因子整体表现良好，")
            report.append("   主要指标方向一致，建议积极参与。")
        elif score <= 40:
            report.append(f"   综合评分{score:.1f}分，存在较多不利因素，")
            report.append("   建议控制仓位，等待信号改善。")
        else:
            report.append(f"   综合评分{score:.1f}分，多空因素交织，")
            report.append("   建议观望等待更明确的方向信号。")
        report.append("")

    def _generate_factor_groups_score(self, scoring: Dict, strategies_results: Dict) -> List[str]:
        """
        生成因子评分表格（右侧交易版）

        评分公式：最终评分 = 趋势分 + 趋势确认加成 + 成交量加成
        """
        report = []
        report.append("### 📊 因子组评分")
        report.append("")

        factors_score = scoring.get('factors_score', {})
        factors_raw = scoring.get('factors_raw', {})

        # 获取评分系统的核心数据
        trend_score = scoring.get('trend_score', 50)
        momentum_score = scoring.get('momentum_score', 50)
        money_score = scoring.get('money_score', 50)
        trend_bonus = scoring.get('trend_bonus', 0)
        volume_bonus = scoring.get('volume_bonus', 0)
        final_score = scoring.get('score', 50)

        # 解析嵌套的 factors_raw 结构
        trend_factors = factors_raw.get('trend_factors', {})
        momentum_factors = factors_raw.get('momentum_factors', {})
        money_factors = factors_raw.get('money_factors', {})
        aux_factors = factors_raw.get('aux_factors', {})

        # 三因子组评分
        report.append("| 因子组 | 因子得分 | 说明 |")
        report.append("|--------|----------|------|")
        report.append(f"| 趋势因子组 | {trend_score:.1f} | 综合评分（核心指标）|")
        report.append(f"| 动能因子组 | {momentum_score:.1f} | 参考指标 |")
        report.append(f"| 资金因子组 | {money_score:.1f} | 参考指标 |")
        report.append("")

        # 评分计算过程
        report.append("**评分计算过程（右侧交易版）：**")
        report.append("")
        report.append(f"| 步骤 | 计算 | 结果 |")
        report.append("|------|------|------|")
        report.append(f"| ① 趋势基础分 | 五因子加权得分 | **{trend_score:.1f}分** |")
        report.append(f"| ② 趋势确认加成 | 多头排列+DMI+MACD确认 | **+{trend_bonus:.1f}分** |")
        report.append(f"| ③ 成交量加成 | 放量确认 | **+{volume_bonus:.1f}分** |")
        report.append(f"| **最终评分** | 基础分 + 加成 | **{final_score:.1f}分** |")
        report.append("")

        # 检查是否有熔断或风险警告修正
        execution = scoring.get('execution', {})
        action_guide = execution.get('action_guide', '')
        has_circuit_breaker = '熔断' in action_guide
        has_risk_warning = '风险警告' in action_guide

        if has_circuit_breaker:
            report.append(f"| ③ 熔断修正 | 触发熔断，评分强制降低 | **{base_score:.1f} → {min(base_score, 25):.1f}分** |")
        elif has_risk_warning:
            report.append(f"| ④ 风险警告修正 | 评分打7折 | **{final_score * 0.7:.1f}分** |")
        else:
            report.append(f"| ④ 最终评分 | 基础分 + 加成 | **{final_score:.1f}分** |")

        report.append("")

        # 因子详情
        report.append("**因子详情：**")
        report.append("")

        # 定义因子信息：描述、原始值来源、格式化方式
        factor_info = {
            'trend_strength': {
                'desc': '股价相对MA20位置',
                'raw_key': ('aux_factors', 'trend_strength'),  # 从 aux_factors 获取
                'format': 'percent',  # 百分比格式
                'label': 'MA20乖离率'
            },
            'ma_slope': {
                'desc': '均线斜率方向',
                'raw_key': ('trend_factors', 'ma_alignment'),  # 从 trend_factors 获取
                'format': 'score',  # 评分格式
                'label': '均线排列得分'
            },
            'macd_momentum': {
                'desc': 'MACD柱状图动量',
                'raw_key': ('trend_factors', 'macd_momentum'),  # 从 trend_factors 获取
                'format': 'float',  # 浮点数格式
                'label': 'MACD动量变化'
            },
            'volume_ratio': {
                'desc': '成交量相对强度',
                'raw_key': ('money_factors', 'volume_ratio'),  # 从 money_factors 获取
                'format': 'ratio',  # 比率格式
                'label': '量比'
            },
            'money_flow': {
                'desc': 'OBV资金流向',
                'raw_key': ('money_factors', 'obv_flow'),  # 从 money_factors 获取
                'format': 'score',  # 评分格式
                'label': 'OBV强度'
            }
        }

        for factor_key, info in factor_info.items():
            score_val = factors_score.get(factor_key, 50)

            # 从嵌套结构中获取原始值
            group_name, raw_key = info['raw_key']
            if group_name == 'trend_factors':
                raw_val = trend_factors.get(raw_key, 'N/A')
            elif group_name == 'momentum_factors':
                raw_val = momentum_factors.get(raw_key, 'N/A')
            elif group_name == 'money_factors':
                raw_val = money_factors.get(raw_key, 'N/A')
            elif group_name == 'aux_factors':
                raw_val = aux_factors.get(raw_key, 'N/A')
            else:
                raw_val = 'N/A'

            # 状态判断
            if score_val >= 70:
                status = "✅ 良好"
            elif score_val >= 50:
                status = "➖ 一般"
            else:
                status = "⚠️ 偏弱"

            # 格式化原始值
            if raw_val == 'N/A':
                raw_str = 'N/A'
            elif info['format'] == 'percent':
                # 百分比格式
                raw_str = f"{raw_val*100:.2f}%" if isinstance(raw_val, (int, float)) else str(raw_val)
            elif info['format'] == 'ratio':
                # 比率格式
                raw_str = f"{raw_val:.2f}" if isinstance(raw_val, (int, float)) else str(raw_val)
            elif info['format'] == 'score':
                # 评分格式
                raw_str = f"{raw_val:.0f}分" if isinstance(raw_val, (int, float)) else str(raw_val)
            else:
                # 默认浮点数格式
                raw_str = f"{raw_val:.3f}" if isinstance(raw_val, (int, float)) else str(raw_val)

            report.append(f"- **{info['desc']}** ({factor_key}): {score_val:.1f}分 {status} ({info['label']}: {raw_str})")

        report.append("")
        return report

    def _generate_market_and_risk_info(self, strategies_results: Dict) -> List[str]:
        """
        生成市场状态与风险控制信息
        """
        report = []

        # 双重市场状态信息
        dual_market_state = strategies_results.get('dual_market_state')
        if dual_market_state:
            report.append("### 🌡️ 双重市场状态")
            report.append("")

            index_regime = dual_market_state.get('index_regime', 'sideway')
            stock_regime = dual_market_state.get('stock_regime', 'sideway')
            combined_signal = dual_market_state.get('combined_signal', '观望')
            index_name = dual_market_state.get('index_name', '沪深300')
            confidence = dual_market_state.get('confidence', 0.5)

            # 市场状态emoji
            regime_emoji = {
                'bull': '📈',
                'bear': '📉',
                'sideway': '↔️',
                'volatile': '⚡'
            }

            regime_cn = {
                'bull': '牛市',
                'bear': '熊市',
                'sideway': '震荡',
                'volatile': '剧烈波动'
            }

            # 综合信号emoji
            signal_emoji = {
                '强买入': '🚀',
                '关注': '👀',
                '回避': '⚠️',
                '轻仓': '💰',
                '观望': '➖',
                '空仓': '🛑'
            }.get(combined_signal, '➖')

            index_regime_cn = regime_cn.get(index_regime, '未知')
            stock_regime_cn = regime_cn.get(stock_regime, '未知')

            report.append(f"- **综合信号**: {combined_signal} {signal_emoji}")
            report.append(f"- **大盘({index_name})**: {index_regime_cn} {regime_emoji.get(index_regime, '➖')}")
            report.append(f"- **个股状态**: {stock_regime_cn} {regime_emoji.get(stock_regime, '➖')}")
            report.append(f"- **置信度**: {confidence*100:.0f}%")
            report.append("")

        # 市场状态信息（个股状态详情）
        market_regime = strategies_results.get('market_regime')
        if market_regime:
            report.append("### 📊 自适应阈值")
            report.append("")

            regime = market_regime.get('regime', 'sideway')
            volatility = market_regime.get('volatility_level', 'normal')
            buy_threshold = market_regime.get('buy_threshold', 50)
            sell_threshold = market_regime.get('sell_threshold', 25)
            confidence = market_regime.get('confidence', 0.5)

            volatility_cn = {
                'low': '低波动',
                'normal': '正常',
                'high': '高波动',
                'extreme': '极端波动'
            }.get(volatility, '正常')

            report.append(f"- **买入阈值**: {buy_threshold:.1f}")
            report.append(f"- **卖出阈值**: {sell_threshold:.1f}")
            report.append(f"- **波动率水平**: {volatility_cn}")
            report.append("")

        # 风险控制信息
        risk_control = strategies_results.get('risk_control')
        if risk_control:
            report.append("### 🛡️ 风险控制")
            report.append("")

            stop_price = risk_control.get('stop_price', 0)
            stop_type = risk_control.get('stop_type', 'atr')
            distance_pct = risk_control.get('distance_percent', 0.05)
            confidence = risk_control.get('confidence', 0.5)

            stop_type_cn = {
                'atr': 'ATR止损',
                'support': '支撑位止损',
                'mae': '历史MAE止损',
                'percentage': '固定比例止损'
            }.get(stop_type, '止损')

            report.append(f"- **建议止损位**: ¥{stop_price:.2f} ({stop_type_cn})")
            report.append(f"- **止损幅度**: {distance_pct*100:.1f}%")
            report.append(f"- **信号强度**: {confidence:.2f}")
            report.append("")

            # 风险提示
            if distance_pct > 0.10:
                report.append("⚠️ **风险提示**: 止损幅度较大，建议控制仓位")
            elif distance_pct > 0.08:
                report.append("⚠️ **注意**: 止损幅度适中，可考虑分批建仓")
            else:
                report.append("✅ **建议**: 止损幅度合理，风险可控")
            report.append("")

        return report

    def _generate_trend_scoring_report(self, strategies_results: Dict) -> List[str]:
        """
        生成趋势评分系统报告

        展示纯趋势强度评分、各因子得分和时机系数
        """
        report = []

        trend_scoring = strategies_results.get('trend_scoring')
        if not trend_scoring:
            return report

        report.append("### 📈 趋势评分系统")
        report.append("")

        # 硬过滤结果
        passed_filter = trend_scoring.get('passed_hard_filter', False)
        filter_reason = trend_scoring.get('hard_filter_reason', '')

        if not passed_filter:
            report.append(f"**❌ 未通过硬过滤**: {filter_reason}")
            report.append("")
            return report

        # 趋势总分
        final_score = trend_scoring.get('final_score', 0)
        trend_total = trend_scoring.get('trend_total_score', 0)
        timing_coef = trend_scoring.get('timing_coefficient', 1.0)
        timing_type = trend_scoring.get('timing_type', '标准')

        # 评分等级
        if final_score >= 90:
            grade = "极强趋势"
            grade_emoji = "🚀"
        elif final_score >= 75:
            grade = "强趋势"
            grade_emoji = "📈"
        elif final_score >= 60:
            grade = "趋势一般"
            grade_emoji = "➖"
        else:
            grade = "趋势弱"
            grade_emoji = "📉"

        report.append(f"**最终评分**: {final_score:.1f}分 {grade_emoji} ({grade})")
        report.append(f"- 趋势总分: {trend_total:.1f}分")
        report.append(f"- 时机系数: {timing_coef:.2f} ({timing_type})")
        report.append("")

        # 各因子得分表格
        report.append("| 因子 | 得分 | 权重 | 贡献分 |")
        report.append("|------|------|------|--------|")

        ma_score = trend_scoring.get('ma_structure_score', 0)
        momentum_score = trend_scoring.get('price_momentum_score', 0)
        volume_score = trend_scoring.get('volume_score', 0)
        rs_score = trend_scoring.get('relative_strength_score', 0)

        report.append(f"| 均线结构 | {ma_score:.1f} | 30% | {ma_score * 0.3:.1f} |")
        report.append(f"| 价格动能 | {momentum_score:.1f} | 30% | {momentum_score * 0.3:.1f} |")
        report.append(f"| 量能配合 | {volume_score:.1f} | 25% | {volume_score * 0.25:.1f} |")
        report.append(f"| 相对强度 | {rs_score:.1f} | 15% | {rs_score * 0.15:.1f} |")
        report.append(f"| **合计** | - | **100%** | **{trend_total:.1f}** |")
        report.append("")

        # 时机分析
        details = trend_scoring.get('details', {})
        timing_details = details.get('timing', {})

        if timing_details:
            report.append("**时机分析**:")
            timing_reason = timing_details.get('timing_reason', '')
            return_5d = timing_details.get('return_5d', 0)
            dist_to_ma20 = timing_details.get('dist_to_ma20', 0)

            report.append(f"- 时机类型: {timing_type}")
            if timing_reason:
                report.append(f"- 原因: {timing_reason}")
            report.append(f"- 5日涨幅: {return_5d:.2f}%")
            report.append(f"- 距MA20: {dist_to_ma20:.2f}%")
            report.append("")

        # 建议
        report.append("**操作建议**:")
        if final_score >= 75:
            report.append("- ✅ 趋势强劲，可考虑入场或加仓")
            if timing_coef >= 1.1:
                report.append("- 🎯 时机较好，建议积极参与")
            elif timing_coef <= 0.8:
                report.append("- ⚠️ 时机一般，建议控制仓位")
        elif final_score >= 60:
            report.append("- ➖ 趋势一般，建议观望或轻仓")
        else:
            report.append("- ❌ 趋势较弱，不建议入场")
        report.append("")

        return report

    def _generate_core_conclusion(self, scoring: Dict, position_info: Dict,
                                   pattern_assessment: Dict) -> List[str]:
        """
        生成第一部分：核心结论区（重构版）

        新架构：趋势得分 × 位置修正系数
        修复：优先使用评分系统的熔断机制结果
        """
        report = []
        report.append("## 第一部分：核心结论")
        report.append("")

        score = scoring.get('score', 50)
        score_grade = scoring.get('score_grade', 'N/A')
        trend_score = scoring.get('trend_score', score)
        trend_bonus = scoring.get('trend_bonus', 0)

        # 获取评分系统的操作指引（已包含熔断机制）
        execution = scoring.get('execution', {})
        action_guide = execution.get('action_guide', '')

        # 获取形态和位置信息
        pattern_type = pattern_assessment.get('type', 'none')
        pattern_strength = int(pattern_assessment.get('strength', 0))
        position = position_info.get('position', 'middle')
        is_overbought = position_info.get('is_overbought', False)
        is_oversold = position_info.get('is_oversold', False)

        # ========== 关键修复：优先使用熔断机制的结果 ==========
        action_override = None
        override_reason = None

        # 获取K线形态详情（区分长短期位置）- 提前定义
        factors_raw = scoring.get('factors_raw', {})
        candlestick_detail = factors_raw.get('candlestick_detail', {})
        long_term_pos = candlestick_detail.get('long_term_position', 'mid')
        short_term_pos = candlestick_detail.get('short_term_position', 'mid')

        # 如果 action_guide 包含熔断关键词，直接使用
        if '熔断' in action_guide or '回避' in action_guide:
            if '熔断-回避' in action_guide:
                action_emoji = "🔴"
                action_text = "回避/卖出"
                action_override = 'fuse'
                override_reason = action_guide
            elif '熔断-诱多' in action_guide:
                action_emoji = "🟡"
                action_text = "观望（警惕诱多）"
                action_override = 'fuse'
                override_reason = action_guide
            else:
                action_emoji = "🟡"
                action_text = "观望"
                action_override = 'fuse'
                override_reason = action_guide
        elif '风险警告' in action_guide:
            action_emoji = "🟡"
            action_text = "谨慎观望"
            action_override = 'warning'
            override_reason = action_guide
        else:
            # 形态覆盖规则（修正：区分长短期位置）
            if long_term_pos == 'low' and short_term_pos == 'low' and pattern_type in ['bullish_reversal', 'bullish_continuation'] and pattern_strength >= 3:
                action_override = 'buy'
                override_reason = f"长短期双低位+强底部形态({pattern_assessment.get('patterns', ['未知'])[0]})"
            elif short_term_pos == 'high' and pattern_type in ['bearish_reversal', 'bearish_continuation'] and pattern_strength >= 3:
                action_override = 'sell'
                override_reason = f"短期高位+强顶部形态({pattern_assessment.get('patterns', ['未知'])[0]})"
            elif short_term_pos == 'high' and pattern_type in ['bullish_reversal', 'bullish_continuation']:
                action_override = 'wait'
                override_reason = f"短期高位看涨形态，可能是诱多/力竭"

        # 根据评分调整操作建议
        if action_override in ['fuse', 'warning']:
            # 熔断机制或风险警告已确定操作，不做调整
            pass
        elif action_override == 'buy':
            action_emoji = "🟢"
            action_text = "试探性买入"
        elif action_override == 'sell':
            action_emoji = "🔴"
            action_text = "卖出/减仓"
        elif action_override == 'wait':
            action_emoji = "🟡"
            action_text = "观望（警惕诱多）"
        elif score >= 80:
            # 高评分：趋势确立，右侧交易机会
            action_emoji = "🟢"
            action_text = "买入"
        elif score >= 65:
            # 评分较高
            action_emoji = "🟢"
            action_text = "买入"
        elif score >= 50:
            action_emoji = "🟡"
            action_text = "观望"
        else:
            action_emoji = "🔴"
            action_text = "不建议"

        report.append(f"### {action_emoji} 操作指令：{action_text}")
        report.append("")

        # 评分展示（新架构）
        if override_reason:
            report.append(f"**技术评分：{score:.1f}分（{score_grade}）→ {action_text}**")
            report.append(f"**覆盖原因：{override_reason}**")
        else:
            report.append(f"**技术评分：{score:.1f}分（{score_grade}）**")
        report.append("")

        # 评分构成（修正：如果评分被熔断修改，显示原始计算和最终结果）
        # 评分构成
        report.append(f"**评分构成：** 趋势分 {trend_score:.1f} + 趋势加成 {trend_bonus:.1f} = **{score:.1f}分**")
        report.append("")

        # 置信度计算
        factors_score = scoring.get('factors_score', {})
        bullish_count = sum(1 for v in factors_score.values() if isinstance(v, (int, float)) and v > 60)
        bearish_count = sum(1 for v in factors_score.values() if isinstance(v, (int, float)) and v < 40)

        if bullish_count >= 3 or bearish_count >= 3:
            if pattern_strength >= 3:
                confidence = "高"
                confidence_reason = f"多数因子同向+形态支持"
            else:
                confidence = "中高"
                confidence_reason = f"多数因子同向"
        elif bullish_count >= 2 or bearish_count >= 2:
            confidence = "中"
            confidence_reason = f"因子方向较一致"
        else:
            confidence = "中低"
            confidence_reason = "因子分歧，需综合判断"

        report.append(f"**置信度：{confidence}**（{confidence_reason}）")
        report.append("")

        # 关键理由（区分长短期位置）
        report.append("### 💡 关键理由")
        report.append("")

        reason_parts = []

        # 长短期位置分析
        if long_term_pos == 'low' and short_term_pos == 'high':
            reason_parts.append("⚠️ 长期低位但短期超买，建议等待回踩后介入")
        elif long_term_pos == 'low' and short_term_pos == 'low':
            reason_parts.append("✅ 长期低位+短期超卖，黄金坑机会")
        elif long_term_pos == 'high' and short_term_pos == 'high':
            reason_parts.append("⚠️ 长短期双高位，风险较大")
        elif long_term_pos == 'high' and short_term_pos == 'low':
            reason_parts.append("⚠️ 长期高位短期超卖，可能是反弹但趋势未改")
        elif short_term_pos == 'high':
            reason_parts.append("⚠️ 短期超买，不宜追高")
        elif short_term_pos == 'low':
            reason_parts.append("✅ 短期超卖，可关注反弹机会")

        # 基于超买超卖状态
        is_oversold = position_info.get('is_oversold', False)
        is_overbought = position_info.get('is_overbought', False)

        if is_overbought:
            reason_parts.append("入场位置偏高（技术指标偏热）")
        elif is_oversold:
            reason_parts.append("入场位置安全（技术指标超卖）")

        # MA200压力判断
        close = position_info.get('close', 0)
        ma200 = position_info.get('ma200', 0)
        if ma200 > 0 and close > 0:
            ma200_distance = (ma200 - close) / close * 100
            if ma200 > close and ma200_distance > 3:
                reason_parts.append(f"⚠️ MA200(¥{ma200:.2f})在上方{ma200_distance:.1f}%处形成压力，需突破确认")

        # 基于趋势得分
        if trend_score >= 70:
            reason_parts.append("趋势强势确立")
        elif trend_score >= 55:
            reason_parts.append("趋势偏强")
        elif trend_score < 40:
            # 检测"接飞刀"行情：趋势极弱
            if is_oversold:
                reason_parts.append("⚠️【接飞刀风险】位置虽低但趋势极弱，切勿盲目抄底")
            else:
                reason_parts.append("趋势极弱")
        elif trend_score < 45:
            reason_parts.append("趋势偏弱")

        # 基于形态（结合位置判断）
        if pattern_strength >= 3:
            patterns = pattern_assessment.get('patterns', [])
            if patterns:
                # 区分形态在当前位置的含义
                if short_term_pos == 'high' and pattern_type in ['bullish_reversal', 'bullish_continuation']:
                    reason_parts.append(f"⚠️ 高位看涨形态「{patterns[0]}」可能是诱多/力竭")
                elif short_term_pos == 'low' and pattern_type in ['bullish_reversal', 'bullish_continuation']:
                    reason_parts.append(f"✅ 低位看涨形态「{patterns[0]}」支持反弹")
                else:
                    reason_parts.append(f"K线形态：{patterns[0]}")

        # 添加警告
        warnings = scoring.get('warnings', [])
        if warnings:
            reason_parts.append(f"注意：{warnings[0]}")

        report.append("，".join(reason_parts) if reason_parts else "综合分析，建议谨慎操作")
        report.append("")
        return report

    def _generate_signal_resonance(self, latest_data: pd.Series, position_info: Dict,
                                    pattern_assessment: Dict, scoring: Dict,
                                    df: pd.DataFrame = None) -> List[str]:
        """
        生成第二部分：多维信号共振分析
        """
        report = []
        report.append("## 第二部分：多维信号共振分析")
        report.append("")

        # 1. 趋势状态（综合考虑DMI和均线排列）
        report.append("### 📊 趋势状态")
        report.append("")

        pdi = latest_data.get('dmi_pdi', 50)
        mdi = latest_data.get('dmi_mdi', 20)
        adx = latest_data.get('dmi_adx', 20)
        ma20 = latest_data.get('ma_20', 0)
        ma50 = latest_data.get('ma_50', 0)
        close = latest_data.get('close', 0)

        dmi_diff = abs(pdi - mdi)
        DMI_BALANCE_THRESHOLD = 0.5

        # 判断均线排列和股价位置
        ma_bullish = False
        ma_bearish = False
        price_above_ma = False  # 股价在均线上方
        price_below_ma = False  # 股价在均线下方
        if ma20 > 0 and ma50 > 0 and close > 0:
            if close > ma20 > ma50:
                ma_bullish = True  # 完美多头排列
            elif close < ma20 < ma50:
                ma_bearish = True  # 完美空头排列
            elif close > ma20 and close > ma50:
                # 股价在MA20和MA50上方，即使MA20<MA50也是偏多
                price_above_ma = True
            elif close < ma20 and close < ma50:
                # 股价在MA20和MA50下方
                price_below_ma = True

        # 综合判断趋势状态（均线优先，DMI辅助）
        if ma_bearish or price_below_ma:
            # 均线空头排列或股价在均线下方
            if price_below_ma and not ma_bearish:
                trend_state = "下跌趋势（弱）"
                trend_desc = f"⚠️ 股价在MA20({ma20:.2f})和MA50({ma50:.2f})下方，均线压制"
            elif adx > 25:
                trend_state = "下跌趋势（强）"
                trend_desc = f"均线空头排列（股价<MA20<MA50），ADX={adx:.1f}"
            else:
                trend_state = "下跌趋势（弱）"
                trend_desc = f"均线空头排列（股价<MA20<MA50），PDI>MDI仅为短期反弹"
        elif ma_bullish:
            # 完美多头排列（股价>MA20>MA50）
            # 关键修复：必须结合DMI判断，避免"均线多头但DMI空头"的误判
            if mdi > pdi and dmi_diff >= DMI_BALANCE_THRESHOLD:
                # DMI空头占优 - 均线多头可能是诱多/假突破
                trend_state = "震荡偏空（警惕诱多）"
                trend_desc = f"⚠️ 均线虽多头排列，但MDI({mdi:.1f})>PDI({pdi:.1f})，空头力量更强，警惕回调"
            elif pdi > mdi and dmi_diff >= DMI_BALANCE_THRESHOLD and adx > 25:
                trend_state = "上升趋势（强）"
                trend_desc = f"✅ 均线多头排列+DMI多头占优，ADX={adx:.1f}"
            elif pdi > mdi and dmi_diff >= DMI_BALANCE_THRESHOLD:
                trend_state = "上升趋势（弱）"
                trend_desc = f"✅ 均线多头排列，PDI({pdi:.1f})>MDI({mdi:.1f})，但ADX={adx:.1f}偏低"
            else:
                # DMI平衡，均线多头但无明确趋势
                trend_state = "震荡偏多"
                trend_desc = f"均线多头排列，但PDI≈MDI（差值{dmi_diff:.2f}），趋势不明确"
        elif price_above_ma:
            # 股价在均线上方，但均线未多头排列 - 需要结合DMI判断
            # 关键修复：不能简单判定为上升，需看DMI多空力量
            if pdi > mdi and dmi_diff >= DMI_BALANCE_THRESHOLD:
                # DMI多头占优
                if adx > 25:
                    trend_state = "上升趋势（强）"
                else:
                    trend_state = "上升趋势（弱）"
                trend_desc = f"股价在MA20({ma20:.2f})和MA50({ma50:.2f})上方，PDI>MDI，偏多"
            elif mdi > pdi and dmi_diff >= DMI_BALANCE_THRESHOLD:
                # DMI空头占优 - 尽管股价在均线上方，但空方力量更强
                trend_state = "震荡偏空"
                trend_desc = f"⚠️ 股价虽在均线上方，但MDI({mdi:.1f})>PDI({pdi:.1f})，空头占优，警惕回调"
            else:
                # DMI平衡
                trend_state = "震荡/无趋势"
                trend_desc = f"股价在MA20({ma20:.2f})和MA50({ma50:.2f})上方，但PDI≈MDI（差值{dmi_diff:.2f}），多空力量均衡"
        elif dmi_diff < DMI_BALANCE_THRESHOLD:
            trend_state = "震荡/无趋势"
            trend_desc = f"PDI≈MDI（差值{dmi_diff:.2f}），多空力量均衡"
        elif pdi > mdi:
            # 关键修复：必须结合MACD状态判断，避免DMI与MACD矛盾时的误判
            macd = latest_data.get('macd', 0)
            if macd < 0:
                # DMI多头但MACD空头 = 矛盾信号，趋势不可靠
                if adx > 25:
                    trend_state = "震荡偏多（警惕背离）"
                    trend_desc = f"⚠️ PDI>MDI但MACD({macd:.1f})为空头，DMI与MACD背离，趋势存疑"
                else:
                    trend_state = "震荡/无趋势"
                    trend_desc = f"PDI>MDI但MACD({macd:.1f})为空头，ADX={adx:.1f}偏低，趋势不明确"
            else:
                if adx > 25:
                    trend_state = "上升趋势（强）"
                else:
                    trend_state = "上升趋势（弱）"
                trend_desc = f"PDI>MDI，多头占优，ADX={adx:.1f}（均线交织，需确认）"
        else:
            # MDI > PDI
            macd = latest_data.get('macd', 0)
            if macd > 0:
                # DMI空头但MACD多头 = 矛盾信号
                if adx > 25:
                    trend_state = "震荡偏空（警惕背离）"
                    trend_desc = f"⚠️ MDI>PDI但MACD({macd:.1f})为多头，可能反弹而非反转"
                else:
                    trend_state = "震荡/无趋势"
                    trend_desc = f"MDI>PDI但MACD({macd:.1f})为多头，趋势不明确"
            else:
                if adx > 25:
                    trend_state = "下跌趋势（强）"
                else:
                    trend_state = "下跌趋势（弱）"
                trend_desc = f"PDI<MDI，空头占优，ADX={adx:.1f}"

        report.append(f"- **状态**：{trend_state}")
        report.append(f"- **说明**：{trend_desc}")
        report.append("")

        # 2. 动能状态
        report.append("### ⚡ 动能状态")
        report.append("")

        macd = latest_data.get('macd', 0)
        rsi = latest_data.get('rsi_24', 50)

        # 从 scoring 中获取 MACD 动量原始值
        factors_raw = scoring.get('factors_raw', {})
        mom1 = factors_raw.get('macd_momentum', 0)

        if macd > 0 and mom1 > 0:
            momentum_state = "多头动能增强"
            momentum_desc = "MACD红柱放大，动能向上"
        elif macd > 0 and mom1 < 0:
            momentum_state = "多头动能减弱（警惕）"
            momentum_desc = "MACD红柱缩短，存在顶背离可能"
        elif macd < 0 and mom1 < 0:
            momentum_state = "空头动能增强"
            momentum_desc = "MACD绿柱放大，动能向下"
        elif macd < 0 and mom1 > 0:
            momentum_state = "空头动能减弱（关注）"
            momentum_desc = "MACD绿柱缩短，存在底背离可能"
        else:
            momentum_state = "动能中性"
            momentum_desc = "MACD动能变化不明显"

        report.append(f"- **状态**：{momentum_state}")
        report.append(f"- **说明**：{momentum_desc}")

        # RSI斜率判断
        if rsi > 70:
            rsi_state = "超买区"
        elif rsi < 30:
            rsi_state = "超卖区"
        elif rsi > 60:
            rsi_state = "偏强"
        elif rsi < 40:
            rsi_state = "偏弱"
        else:
            rsi_state = "中性"
        report.append(f"- **RSI(24)**：{rsi:.1f}（{rsi_state}）")
        report.append("")

        # 3. 位置状态
        report.append("### 📍 位置状态")
        report.append("")

        close = latest_data.get('close', 0)
        boll_upper = latest_data.get('boll_upper', 0)
        boll_lower = latest_data.get('boll_lower', 0)
        cci = latest_data.get('cci', 0)
        wr = latest_data.get('wr', 50)

        # 布林带位置 - 综合考虑CCI和WR指标
        if boll_upper > boll_lower:
            boll_pct = (close - boll_lower) / (boll_upper - boll_lower) * 100
        else:
            boll_pct = 50

        # 综合判断：布林带位置结合CCI/WR超买超卖状态
        # 解决矛盾：布林带中部但CCI/WR显示超卖时，应该描述为偏低
        is_oversold = cci < -100 or wr > 80
        is_overbought = cci > 100 or wr < 20

        if boll_pct > 90:
            boll_state = "极度超买（触及上轨）"
        elif boll_pct > 70:
            if is_overbought:
                boll_state = "偏高（超买确认）"
            else:
                boll_state = "偏高"
        elif boll_pct < 10:
            boll_state = "极度超卖（触及下轨）"
        elif boll_pct < 30:
            if is_oversold:
                boll_state = "偏低（超卖确认）"
            else:
                boll_state = "偏低"
        elif boll_pct < 50:
            # 30-50%区间，如果CCI/WR超卖，应该描述为偏低
            if is_oversold:
                boll_state = "偏低（超卖信号）"
            else:
                boll_state = "中下部"
        elif boll_pct > 50 and boll_pct <= 70:
            # 50-70%区间，如果CCI/WR超买，应该描述为偏高
            if is_overbought:
                boll_state = "偏高（超买信号）"
            else:
                boll_state = "中上部"
        else:
            boll_state = "中部"

        report.append(f"- **布林带**：{boll_state}（{boll_pct:.0f}%位置）")

        # CCI状态
        if cci > 200:
            cci_state = "严重超买"
        elif cci > 100:
            cci_state = "超买区"
        elif cci < -200:
            cci_state = "严重超卖"
        elif cci < -100:
            cci_state = "超卖区"
        else:
            cci_state = "正常区间"
        report.append(f"- **CCI(14)**：{cci:.1f}（{cci_state}）")

        # WR状态
        if wr < 10:
            wr_state = "严重超买"
        elif wr < 20:
            wr_state = "接近超买"
        elif wr > 90:
            wr_state = "严重超卖"
        elif wr > 80:
            wr_state = "接近超卖"
        else:
            wr_state = "正常区间"
        report.append(f"- **WR(14)**：{wr:.1f}（{wr_state}）")
        report.append("")

        # 4. 形态特权区
        report.append("### 🕯️ 形态特权区（定性分析）")
        report.append("")

        pattern_type = pattern_assessment.get('type', 'none')
        pattern_strength = pattern_assessment.get('strength', 0)
        patterns = pattern_assessment.get('patterns', [])
        position = position_info.get('position', 'middle')

        # 关键修复：降低显示阈值，强度>=1（弱形态）也显示
        # 之前的阈值>=3过高，导致"中"强度形态（如长上影线）被忽略
        if pattern_strength >= 1 and patterns:
            pattern_name = patterns[0] if patterns else '形态'

            # 强度描述
            strength_desc = "强" if pattern_strength >= 4 else "中" if pattern_strength >= 2 else "弱"

            if position == 'low' and pattern_type == 'bullish_reversal':
                report.append(f"- **形态**：{pattern_name}（强度：{strength_desc}）")
                report.append(f"- **位置权重**：低位（{position_info.get('price_ratio', 50):.0f}%分位）")
                if pattern_strength >= 3:
                    report.append(f"- **定性影响**：🔥 强力支撑信号 - 底部形态在低位形成，可靠性高")
                else:
                    report.append(f"- **定性影响**：✅ 偏多信号 - 低位出现看涨形态，值得关注")
            elif position == 'high' and pattern_type == 'bearish_reversal':
                report.append(f"- **形态**：{pattern_name}（强度：{strength_desc}）")
                report.append(f"- **位置权重**：高位（{position_info.get('price_ratio', 50):.0f}%分位）")
                if pattern_strength >= 3:
                    report.append(f"- **定性影响**：⚠️ 顶部警示信号 - 顶部形态在高位形成，需警惕回调")
                else:
                    report.append(f"- **定性影响**：⚠️ 偏空信号 - 高位出现看跌形态，保持谨慎")
            elif position == 'high' and pattern_type == 'bullish_reversal':
                report.append(f"- **形态**：{pattern_name}（强度：{strength_desc}）")
                report.append(f"- **位置权重**：高位（{position_info.get('price_ratio', 50):.0f}%分位）")
                report.append(f"- **定性影响**：⚠️ 中性偏警惕 - 高位出现看涨形态可能是诱多/力竭")
            elif position == 'low' and pattern_type == 'bearish_reversal':
                report.append(f"- **形态**：{pattern_name}（强度：{strength_desc}）")
                report.append(f"- **位置权重**：低位（{position_info.get('price_ratio', 50):.0f}%分位）")
                report.append(f"- **定性影响**：➖ 中性 - 低位看跌形态可能是洗盘/诱空")
            elif pattern_type == 'neutral':
                # 中性形态（如长上影线、长下影线等）
                report.append(f"- **形态**：{pattern_name}（强度：{strength_desc}）")
                report.append(f"- **位置权重**：{position}")
                report.append(f"- **定性影响**：➖ 中性信号 - {pattern_assessment.get('description', '需结合其他信号判断')}")
            else:
                report.append(f"- **形态**：{pattern_name}（强度：{strength_desc}）")
                report.append(f"- **位置权重**：{position}")
                report.append(f"- **定性影响**：形态存在但位置权重一般")
        else:
            report.append("- **形态**：无明显形态")
            report.append("- **定性影响**：当前无显著K线形态信号")

        # 新增：绘制近期K线图
        if df is not None and len(df) >= 5:
            report.append("")
            report.append("#### 📊 近期K线图")
            report.append("")
            report.append("```")
            report.append(draw_candlestick_chart(df, num_candles=10, width_per_candle=6, height=13))
            report.append("```")
            report.append("")
            report.append("> 图例：🟢 绿色 = 阳线（涨） | 🔴 红色 = 阴线（跌） | │ = 影线")

        # 新增：如果有识别到形态，显示形态示意图
        if pattern_strength >= 1 and patterns:
            pattern_name = patterns[0]
            illustration = draw_pattern_illustration(pattern_name)
            if illustration and len(illustration.strip()) > 10:
                report.append("")
                report.append(f"#### 📖 「{pattern_name}」形态说明")
                report.append("")
                report.append("```")
                report.append(illustration)
                report.append("```")

        report.append("")
        return report

    def _generate_score_breakdown(self, scoring: Dict) -> List[str]:
        """
        生成第三部分：量化评分与因子拆解（右侧交易版）

        评分公式：最终评分 = 趋势分 + 趋势确认加成 + 成交量加成
        """
        report = []
        report.append("## 第三部分：量化评分与因子拆解")
        report.append("")

        score = scoring.get('score', 0)
        score_grade = scoring.get('score_grade', 'N/A')
        trend_score = scoring.get('trend_score', score)
        trend_bonus = scoring.get('trend_bonus', 0)
        volume_bonus = scoring.get('volume_bonus', 0)
        factors_raw = scoring.get('factors_raw', {})
        factors_score = scoring.get('factors_score', {})

        # 综合评分
        report.append(f"### 📊 综合评分：{score:.1f} 分（{score_grade}）")
        report.append("")
        report.append(f"> **计算公式**：趋势分 + 趋势加成 + 成交量加成")
        report.append(f"> `{trend_score:.1f} + {trend_bonus:.1f} + {volume_bonus:.1f} = {score:.1f}`")
        report.append("")

        # 趋势得分分解
        report.append("### 趋势因子得分")
        report.append("")
        report.append("| 因子 | 权重 | 得分 | 说明 |")
        report.append("|------|------|------|------|")

        factor_info = {
            'trend_strength': ('趋势强度', 0.30, '股价相对MA20位置'),
            'ma_slope': ('均线斜率', 0.10, '斜率向上为佳'),
            'macd_momentum': ('MACD动量', 0.30, '金叉/柱状图扩大'),
            'money_flow': ('资金流向', 0.20, 'OBV持续流入'),
            'volume_ratio': ('成交量', 0.10, '放量确认'),
        }

        for key, (name, weight, desc) in factor_info.items():
            score_val = factors_score.get(key, 50)
            report.append(f"| {name} | {weight*100:.0f}% | {score_val:.1f} | {desc} |")

        report.append("")

        # 趋势确认加成说明
        report.append("### 📈 趋势确认加成")
        report.append("")
        report.append("| 加成项 | 条件 | 加分 |")
        report.append("|--------|------|------|")
        report.append(f"| 多头排列 | MA5 > MA10 > MA20 | +10分 |")
        report.append(f"| DMI趋势 | PDI > MDI 且 ADX > 25 | +5分 |")
        report.append(f"| MACD确认 | DIF > DEA 且 DIF > 0 | +5分 |")
        report.append(f"| 放量确认 | 成交量 > 1.5倍均量 | +3分 |")
        report.append("")
        report.append(f"**当前加成：** +{trend_bonus:.1f}分")
        report.append("")

        # 红黑榜
        report.append("### 🔴🟢 红黑榜")
        report.append("")

        factor_scores_list = [
            ('trend_strength', '趋势强度', factors_score.get('trend_strength', 50)),
            ('ma_slope', '均线斜率', factors_score.get('ma_slope', 50)),
            ('macd_momentum', 'MACD动量', factors_score.get('macd_momentum', 50)),
            ('money_flow', '资金流向', factors_score.get('money_flow', 50)),
            ('volume_ratio', '成交量', factors_score.get('volume_ratio', 50)),
        ]

        factor_scores_list = [(k, n, v*100 if v <= 1.0 else v) for k, n, v in factor_scores_list]
        factor_scores_list.sort(key=lambda x: x[2], reverse=True)

        # 红榜
        strong_factors = [f for f in factor_scores_list if f[2] >= 70]
        if strong_factors:
            report.append("**🟢 核心支撑：**")
            for key, name, val in strong_factors[:3]:
                report.append(f"- {name}（得分{val:.0f}分）")
            report.append("")

        # 黑榜
        weak_factors = [f for f in factor_scores_list if f[2] <= 40]
        if weak_factors:
            report.append("**🔴 主要拖累：**")
            for key, name, val in weak_factors[:3]:
                report.append(f"- {name}（得分{val:.0f}分）")
            report.append("")

        # K线形态详情展示（位置敏感评估）
        candlestick_detail = factors_raw.get('candlestick_detail', {})
        if candlestick_detail and candlestick_detail.get('patterns'):
            report.append("### 🕯️ K线形态分析（位置敏感）")
            report.append("")

            position_zone = candlestick_detail.get('position_zone', 'mid_position')
            position_desc = candlestick_detail.get('position_desc', '中位区域')
            long_term_pos = candlestick_detail.get('long_term_position', 'mid')
            short_term_pos = candlestick_detail.get('short_term_position', 'mid')
            assessment = candlestick_detail.get('assessment', '')

            # 长短期位置说明（关键修复：区分长短期）
            long_emoji = {'low': '🟢', 'mid': '🟡', 'high': '🔴'}.get(long_term_pos, '⚪')
            short_emoji = {'low': '🟢', 'mid': '🟡', 'high': '🔴'}.get(short_term_pos, '⚪')
            long_desc = {'low': '低位', 'mid': '中位', 'high': '高位'}.get(long_term_pos, '中位')
            short_desc = {'low': '超卖', 'mid': '正常', 'high': '超买'}.get(short_term_pos, '正常')

            report.append(f"**位置分析：**")
            report.append(f"- 长期位置：{long_emoji} {long_desc}（基于60日分位）")
            report.append(f"- 短期位置：{short_emoji} {short_desc}（基于乖离率/布林带）")
            report.append(f"- 综合判定：{position_desc}")
            report.append("")

            # 形态详情
            patterns = candlestick_detail.get('patterns', [])
            if patterns:
                report.append("| 形态名称 | 类型 | 强度 | 基础权重 | 位置修正 | 最终得分 |")
                report.append("|----------|------|------|----------|----------|----------|")
                for p in patterns[:5]:  # 最多显示5个形态
                    name = p.get('name', '未知')
                    p_type = '看涨' if p.get('type') == 'bullish' else '看跌' if p.get('type') == 'bearish' else '中性'
                    strength = p.get('strength', '中')
                    base_weight = p.get('base_weight', 0)
                    modifier = p.get('modifier', 1.0)
                    final_score = p.get('final_score', 0)
                    report.append(f"| {name} | {p_type} | {strength} | {base_weight:+.1f} | ×{modifier:.1f} | {final_score:+.1f} |")
                report.append("")

            # 评估结论
            if assessment:
                report.append(f"**评估结论：** {assessment}")
                report.append("")

            # 位置敏感逻辑说明
            report.append("> **位置敏感逻辑：**")
            report.append("> - 长期低位+短期低位+看涨 = 黄金坑机会")
            report.append("> - 长期低位+短期高位+看涨 = 警惕诱多，等待回踩")
            report.append("> - 高位+看跌 = 强力减分")
            report.append("> - 低位+看跌 = 可能洗盘")
            report.append("")

        return report

    def _generate_trading_plan(self, scoring: Dict, position_info: Dict,
                                pattern_assessment: Dict, latest_data: pd.Series) -> List[str]:
        """
        生成第四部分：交易执行计划
        """
        report = []
        report.append("## 第四部分：交易执行计划")
        report.append("")

        score = scoring.get('score', 50)
        trigger_type = scoring.get('trigger_type', 'none')
        execution = scoring.get('execution', {})
        confidence = scoring.get('confidence', '中')

        close = latest_data.get('close', 0)
        ma20 = latest_data.get('ma_20', 0)
        ma50 = latest_data.get('ma_50', 0)
        ma200 = latest_data.get('ma_200', 0)
        boll_upper = latest_data.get('boll_upper', 0)
        boll_lower = latest_data.get('boll_lower', 0)

        position = position_info.get('position', 'middle')
        pattern_strength = pattern_assessment.get('strength', 0)

        # 1. 策略类型判定
        report.append("### 📈 策略类型")
        report.append("")

        # 获取趋势方向和评分
        trend_score = scoring.get('trend_score', score)
        trend_direction = scoring.get('trend_direction', 'sideways')

        # 策略类型判断逻辑（重构版 - 与操作指令联动）
        # 左侧交易：主动预测拐点，在趋势反转前介入（高风险）
        # 右侧交易：跟随已确认的趋势，在趋势确立后介入（较安全）

        # 获取熔断状态 - 优先检查
        action_guide = execution.get('action_guide', '')
        is_fuse_triggered = '熔断' in action_guide

        # 关键修复：策略类型必须与第一部分的操作指令(action_text)保持一致
        # 需要重新计算 action_text（与 _generate_core_conclusion 保持一致）
        position_modifier = scoring.get('position_modifier', 1.0)
        factors_raw = scoring.get('factors_raw', {})
        candlestick_detail = factors_raw.get('candlestick_detail', {})
        long_term_pos = candlestick_detail.get('long_term_position', 'mid')
        short_term_pos = candlestick_detail.get('short_term_position', 'mid')
        pattern_type = pattern_assessment.get('type', 'none')
        pattern_strength_val = int(pattern_assessment.get('strength', 0))

        # 计算 action_text（与第一部分逻辑一致）
        if is_fuse_triggered:
            if '熔断-回避' in action_guide:
                action_text = "回避/卖出"
            elif '熔断-诱多' in action_guide:
                action_text = "观望（警惕诱多）"
            else:
                action_text = "观望"
        elif '风险警告' in action_guide:
            action_text = "谨慎观望"
        elif long_term_pos == 'low' and short_term_pos == 'low' and pattern_type in ['bullish_reversal', 'bullish_continuation'] and pattern_strength_val >= 3:
            action_text = "试探性买入"
        elif short_term_pos == 'high' and pattern_type in ['bearish_reversal', 'bearish_continuation'] and pattern_strength_val >= 3:
            action_text = "卖出/减仓"
        elif short_term_pos == 'high' and pattern_type in ['bullish_reversal', 'bullish_continuation']:
            action_text = "观望（警惕诱多）"
        elif position_modifier < 0.5:
            action_text = "不宜追高"
        elif position_modifier < 0.7:
            action_text = "谨慎观望"
        elif score >= 65:
            action_text = "买入"
        elif score >= 50:
            action_text = "观望"
        else:
            action_text = "不建议"

        if is_fuse_triggered:
            # 熔断触发 - 风控/持币观望
            if '熔断-回避' in action_guide:
                strategy_type = "🚫 风控（回避）"
                strategy_desc = "触发熔断机制，高位顶部信号，建议观望或卖出"
            elif '熔断-诱多' in action_guide:
                strategy_type = "⚠️ 风控（警惕诱多）"
                strategy_desc = "触发熔断机制，高位看涨形态可能是诱多陷阱"
            else:
                strategy_type = "🛡️ 风控（持币观望）"
                strategy_desc = "触发熔断机制，建议持币观望"
        elif action_text in ["买入", "试探性买入"]:
            # 操作指令是买入 - 必须显示买入相关策略
            if trigger_type == 'breakout':
                strategy_type = "✅ 右侧交易（突破买入）"
                strategy_desc = "价格突破关键位置，趋势启动信号明确"
            elif trigger_type == 'pullback':
                strategy_type = "✅ 右侧交易（回调买入）"
                strategy_desc = "上涨趋势中的回踩，逢低布局机会"
            elif position == 'low' and pattern_strength >= 3:
                strategy_type = "✅ 左侧交易（底部博弈）"
                strategy_desc = "低位出现底部形态，博弈趋势反转"
            elif trend_direction == 'up':
                strategy_type = "✅ 右侧交易（趋势跟随）"
                strategy_desc = "上涨趋势确立，跟随趋势买入"
            else:
                strategy_type = "✅ 信号驱动（逢低布局）"
                strategy_desc = "综合评分良好，当前位置适合布局"
        elif action_text in ["谨慎观望", "观望（警惕诱多）"]:
            strategy_type = "⚠️ 谨慎观望"
            strategy_desc = "存在风险信号，建议等待更明确的机会"
        elif action_text == "观望":
            strategy_type = "🔍 观望型"
            strategy_desc = "信号不明确，等待更好的入场机会"
        elif action_text in ["不宜追高", "卖出/减仓", "不建议"]:
            strategy_type = "🛡️ 防御型"
            strategy_desc = "当前风险较高，建议规避或减仓"
        elif position == 'high':
            # 高位区域 - 防御为主
            strategy_type = "🛡️ 防御型"
            strategy_desc = "当前位置偏高，以风险控制为主，不宜追高"
        elif trend_direction == 'up' and trend_score >= 55:
            # 上涨趋势确认 - 右侧交易
            if position == 'low':
                strategy_type = "✅ 右侧交易（回调买入）"
                strategy_desc = "上涨趋势中的回调，趋势确立后的低吸机会"
            else:
                strategy_type = "✅ 右侧交易（趋势跟随）"
                strategy_desc = "上涨趋势确立，跟随趋势操作"
        elif trend_direction == 'down' and trend_score < 45:
            # 下跌趋势 - 谨慎
            if position == 'low' and pattern_strength >= 3:
                strategy_type = "⚠️ 左侧交易（博反弹）"
                strategy_desc = "下跌趋势中的底部形态，尝试抄底博反弹（风险较高）"
            else:
                strategy_type = "🚫 观望型"
                strategy_desc = "下跌趋势未结束，等待趋势企稳信号"
        elif trigger_type == 'breakout':
            # 突破信号 - 右侧交易
            strategy_type = "✅ 右侧交易（突破买入）"
            strategy_desc = "价格突破关键位置，趋势可能启动"
        elif position == 'low' and pattern_strength >= 3:
            # 低位+形态 - 可能是左侧交易
            if trend_direction == 'sideways':
                strategy_type = "⚠️ 左侧交易（形态博弈）"
                strategy_desc = "震荡底部的形态信号，博弈趋势反转"
            else:
                strategy_type = "✅ 右侧交易（底部确认）"
                strategy_desc = "底部形态确认，趋势可能反转向上"
        else:
            # 默认观望
            strategy_type = "🔍 观望型"
            strategy_desc = "信号不明确，等待更好的入场机会"

        report.append(f"- **类型**：{strategy_type}")
        report.append(f"- **说明**：{strategy_desc}")
        report.append("")

        # 2. 具体点位（根据风险状态调整）
        report.append("### 📍 具体点位")
        report.append("")

        # 检查是否极端超买（高风险状态）
        # 注意：action_guide 在前面已定义，用于策略类型判断
        position_modifier = scoring.get('position_modifier', 1.0)
        warnings_list = scoring.get('warnings', [])
        is_extreme_risk = (
            position_modifier < 0.5 or
            any('极端' in w or '熔断' in w for w in warnings_list) or
            '熔断' in action_guide
        )

        if is_extreme_risk:
            # 极端风险：不显示入场区间，显示风险提示
            # 关键修复：根据实际风险类型显示不同的提示信息
            is_oversold = position_info.get('is_oversold', False)
            is_overbought = position_info.get('is_overbought', False)

            if '熔断' in action_guide:
                risk_desc = "触发熔断机制，形态信号强烈，建议回避"
            elif is_overbought:
                risk_desc = "指标严重超买，追高风险极高"
            elif is_oversold and position_modifier < 0.5:
                risk_desc = "下跌趋势风险，技术指标偏弱，抄底需谨慎"
            else:
                risk_desc = "风险较高，建议观望等待"

            report.append(f"| ⚠️ 风险提示 | **当前不宜入场** | {risk_desc} |")
            report.append("")
            report.append(f"| 建议操作 | 观望/止盈 | 等待回调至安全区间 |")
            if ma20 > 0 and ma20 < close:
                report.append(f"| 回调买点 | ¥{ma20:.2f}附近 | 等待回踩MA20 |")
            if boll_lower > 0:
                report.append(f"| 安全买点 | ¥{boll_lower:.2f}附近 | 等待布林下轨 |")

            # 止损位（如果持有仓位）
            stop_price = execution.get('stop_price', close * 0.95)
            report.append(f"| 止盈/止损 | ¥{stop_price:.2f} | 若持有，设止盈保护利润 |")
        else:
            # 正常情况：显示入场区间
            report.append("| 项目 | 建议数值 | 说明 |")
            report.append("|------|----------|------|")

            # 入场区间（使用ATR计算合理区间）
            atr = latest_data.get('atr', close * 0.02)  # 默认2%波动
            if atr <= 0 or pd.isna(atr):
                atr = close * 0.02

            # 根据趋势方向调整入场区间
            trend_direction = scoring.get('trend_direction', 'sideways')
            if trend_direction == 'up':
                # 上涨趋势：入场区间在现价下方
                entry_low = max(close - atr, close * 0.97)
                entry_high = close
                entry_desc = "回调买入区间"
            elif trend_direction == 'down':
                # 下跌趋势：谨慎，入场区间更宽
                entry_low = close - atr * 1.5
                entry_high = close + atr * 0.5
                entry_desc = "分批试探区间（谨慎）"
            else:
                # 横盘：围绕现价
                entry_low = close - atr * 0.5
                entry_high = close + atr * 0.5
                entry_desc = "震荡区间内分批建仓"

            report.append(f"| 入场区间 | ¥{entry_low:.2f} ~ ¥{entry_high:.2f} | {entry_desc} |")

            # 触发加仓
            if boll_upper > 0:
                report.append(f"| 触发加仓 | 突破¥{boll_upper:.2f} | 突破布林上轨且放量 |")
            else:
                report.append(f"| 触发加仓 | 突破¥{close*1.05:.2f} | 突破近期高点5% |")

            # 止损位
            stop_price = execution.get('stop_price', close * 0.95)
            if pattern_strength >= 3 and pattern_assessment.get('stop_loss'):
                stop_price = pattern_assessment.get('stop_loss', stop_price)
                stop_desc = "基于形态最低点"
            else:
                stop_desc = "固定5%止损"
            report.append(f"| 止损位 | ¥{stop_price:.2f} | {stop_desc} |")

            # 目标位（区分上涨趋势和下跌趋势）
            targets = []

            # 根据趋势方向决定显示逻辑
            if trend_direction == 'down':
                # 下跌趋势：显示需要突破的阻力位
                report.append("| 操作提示 | 下跌趋势中 | 需突破阻力确认反转 |")
                if boll_upper > 0 and boll_upper > close:
                    report.append(f"| 首要阻力 | ¥{boll_upper:.2f} | 突破布林上轨 |")
                if ma20 > 0 and ma20 > close:
                    report.append(f"| 次要阻力 | ¥{ma20:.2f} | 突破MA20确认 |")
                if ma50 > 0 and ma50 > close:
                    report.append(f"| 关键阻力 | ¥{ma50:.2f} | 突破MA50转强 |")
                # 支撑位
                if boll_lower > 0 and boll_lower < close:
                    report.append(f"| 下档支撑 | ¥{boll_lower:.2f} | 布林下轨支撑 |")
            else:
                # 上涨趋势或横盘：显示目标价
                if boll_upper > 0 and boll_upper > close:
                    targets.append(f"| 短期目标 | ¥{boll_upper:.2f} | 布林带上轨 |")
                if ma50 > 0 and ma50 > close:
                    targets.append(f"| 中期目标 | ¥{ma50:.2f} | MA50压力位 |")
                if ma200 > 0 and not pd.isna(ma200) and ma200 > close:
                    targets.append(f"| 长期目标 | ¥{ma200:.2f} | MA200趋势线 |")

                if targets:
                    for target in targets:
                        report.append(target)
                else:
                    # 股价已突破主要阻力
                    report.append("| 当前状态 | 股价强势 | 突破主要阻力 |")
                    # 显示支撑位
                    if ma20 > 0 and ma20 < close:
                        report.append(f"| 回调支撑 | ¥{ma20:.2f} | MA20支撑 |")
                    if ma50 > 0 and ma50 < close:
                        report.append(f"| 重要支撑 | ¥{ma50:.2f} | MA50支撑 |")

        report.append("")

        # 3. 仓位建议
        report.append("### 💰 仓位建议")
        report.append("")

        if confidence == "高":
            position_pct = "50-70%"
            position_desc = "高置信度信号，可积极建仓"
        elif confidence == "中高":
            position_pct = "30-50%"
            position_desc = "中高置信度，适度参与"
        elif confidence == "中":
            position_pct = "20-30%"
            position_desc = "中等置信度，轻仓试探"
        elif confidence == "中低":
            position_pct = "10-20%"
            position_desc = "置信度一般，极小仓位或观望"
        else:
            position_pct = "0-10%"
            position_desc = "低置信度，建议观望"

        report.append(f"- **建议仓位**：{position_pct}")
        report.append(f"- **说明**：{position_desc}")
        report.append("")

        # 4. 风险提示
        report.append("### ⚠️ 风险提示")
        report.append("")

        trend_score = scoring.get('trend_score', score)
        position_modifier = scoring.get('position_modifier', 1.0)
        factors_score = scoring.get('factors_score', {})

        # 风险分类收集
        high_risks = []  # 高风险
        medium_risks = []  # 中风险
        low_risks = []  # 低风险

        # ========== 高风险检测 ==========
        # 风险1：接飞刀风险（趋势极弱 + 位置安全）
        if trend_score < 40 and position_modifier >= 0.95:
            high_risks.append({
                'title': '接飞刀行情',
                'desc': '当前处于强下跌趋势，虽然位置极低，但属于"接飞刀"行情。资金持续流出，反弹难度极大。',
                'advice': '放弃或极小仓位试探，切勿重仓抄底'
            })

        # 风险2：追高风险（位置危险 + 技术指标超买）- 关键修复：需要真正超买
        # position_modifier低可能是因为下跌趋势风险，必须检查实际超买状态
        is_extreme_overbought = position_info.get('is_extreme_overbought', False)
        is_overbought = position_info.get('is_overbought', False)

        if is_extreme_overbought and position_modifier < 0.5:
            high_risks.append({
                'title': '追高被套',
                'desc': '当前位置严重超买，技术指标处于极端高位。短期回调概率大，追高风险极高。',
                'advice': '观望等待回调，或极小仓位试探'
            })
        elif position_modifier < 0.5 and not is_overbought:
            # 修正系数低但不是超买，可能是下跌趋势风险
            high_risks.append({
                'title': '下跌趋势风险',
                'desc': '当前处于下跌趋势中，技术指标偏弱。即使位置较低，也存在继续下跌风险。',
                'advice': '观望为主，等待趋势企稳信号'
            })

        # ========== 中风险检测 ==========
        # 风险3：高位诱多风险（高位 + 看涨形态）
        if position_modifier < 0.6 and pattern_strength >= 3:
            pattern_type = pattern_assessment.get('type', '')
            if pattern_type in ['bullish_reversal', 'bullish_continuation']:
                medium_risks.append({
                    'title': '高位诱多',
                    'desc': '高位出现看涨形态，需警惕主力诱多出货。如果成交量异常放大，可能是最后的拉升。',
                    'advice': '谨慎参与，设置紧止损'
                })

        # 风险4：假突破风险（突破但量能不足）
        volume_score = factors_score.get('volume_ratio', 50)
        if trigger_type == 'breakout' and volume_score < 50:
            medium_risks.append({
                'title': '假突破',
                'desc': '价格突破但成交量未能有效配合。假突破概率较大，可能快速回落。',
                'advice': '等待放量确认后再入场'
            })

        # 风险5：顶部背离风险（MACD背离）- 关键修复：需要真正在高位
        # position_modifier低可能是因为下跌趋势风险，而不是因为价格在高位
        is_overbought = position_info.get('is_overbought', False)
        boll_pctb = position_info.get('boll_pctb', 0.5)
        wr = position_info.get('wr', 50)

        # 真正的高位：布林带位置>70%或WR<30（超买）
        is_price_high = boll_pctb > 0.7 or wr < 30 or is_overbought

        # 获取MACD动量得分
        macd_momentum = factors_score.get('macd_momentum', 50)

        if is_price_high and macd_momentum < 40:
            medium_risks.append({
                'title': '顶部背离',
                'desc': '价格处于高位但MACD动能减弱。存在顶背离可能，趋势可能反转。',
                'advice': '减仓或设置止盈保护利润'
            })

        # 风险6：流动性风险（成交量过低）
        if volume_score < 35:
            medium_risks.append({
                'title': '流动性不足',
                'desc': '成交量严重萎缩，流动性差。买卖价差大，滑点成本高。',
                'advice': '小仓位或等待放量后再参与'
            })

        # ========== 低风险检测 ==========
        # 风险7：均线压制风险（重要均线在上方，仅当位置低时）
        if close < ma20 and close < ma50 and ma20 > 0 and ma50 > 0 and position_modifier >= 0.7:
            low_risks.append({
                'title': '均线压制',
                'desc': f'MA20(¥{ma20:.2f})和MA50(¥{ma50:.2f})在上方形成压力。突破需要放量配合，否则可能受阻回落。',
                'advice': '在均线附近观察量能变化'
            })

        # 风险8：弱势企稳（合并原"弱势反弹"和"下跌中继"）
        if 40 <= trend_score < 50 and position_modifier >= 0.85:
            low_risks.append({
                'title': '弱势企稳',
                'desc': '趋势偏弱但位置较低，可能是下跌中继或弱势反弹。真正的趋势反转尚未确认。',
                'advice': '轻仓试探或观望，严格止损'
            })

        # ========== 输出风险提示（按优先级）==========
        # 高风险：最多显示1个
        if high_risks:
            risk = high_risks[0]
            report.append(f"- **【高风险：{risk['title']}】**")
            report.append(f"  - {risk['desc']}")
            report.append(f"  - **建议**：{risk['advice']}")

        # 中风险：最多显示2个
        for risk in medium_risks[:2]:
            report.append(f"- **【中风险：{risk['title']}】**")
            report.append(f"  - {risk['desc']}")
            report.append(f"  - **建议**：{risk['advice']}")

        # 低风险：最多显示2个（仅在没有高风险时显示）
        if not high_risks:
            for risk in low_risks[:2]:
                report.append(f"- **【低风险：{risk['title']}】**")
                report.append(f"  - {risk['desc']}")
                report.append(f"  - **建议**：{risk['advice']}")

        # 添加系统级警告（来自scoring_system）
        warnings = scoring.get('warnings', [])
        if warnings:
            for warning in warnings[:2]:  # 最多显示2个系统警告
                report.append(f"- {warning}")

        # 如果没有任何风险提示
        if not high_risks and not medium_risks and not low_risks and not warnings:
            report.append("- 当前风险可控，建议按计划执行")
            if position_modifier < 0.7:
                report.append("- 注意：位置略高，建议设置好止损保护")

        report.append("")
        return report

    # ==================== 增强分析功能 ====================

    def analyze_stock_enhanced(
        self,
        symbol: str,
        days: int = 360,
        include_chip: bool = True,
        include_talib_patterns: bool = True,
        include_strategies: bool = True
    ) -> str:
        """
        增强版股票分析 - 整合筹码分布、K线形态、策略信号

        Args:
            symbol: 股票代码
            days: 分析天数
            include_chip: 是否包含筹码分布分析
            include_talib_patterns: 是否包含TA-Lib形态识别
            include_strategies: 是否包含策略信号

        Returns:
            增强版分析报告
        """
        print(f"开始增强分析 {symbol}...")

        # 获取股票数据
        df = self.get_stock_data(symbol, days)
        if df.empty:
            return f"无法获取 {symbol} 的数据"

        # 计算技术指标
        df_with_indicators = self.calculate_technical_indicators(df)

        # 运行交易策略（原有功能）
        strategies_results = self.run_trading_strategies(df_with_indicators, symbol)

        # 生成增强版报告
        report = self.generate_enhanced_report(
            df_with_indicators,
            strategies_results,
            symbol,
            include_chip=include_chip,
            include_talib_patterns=include_talib_patterns,
            include_strategies=include_strategies
        )

        return report

    def generate_enhanced_report(
        self,
        df: pd.DataFrame,
        strategies_results: Dict,
        symbol: str,
        include_chip: bool = True,
        include_talib_patterns: bool = True,
        include_strategies: bool = True
    ) -> str:
        """
        生成增强版分析报告

        整合：
        - 原有技术分析
        - 筹码分布分析
        - TA-Lib 61种K线形态
        - 10种经典策略信号
        """
        if df.empty:
            return "无数据可生成报告"

        latest_data = df.iloc[-1]
        report = []

        # 基本信息
        report.append(f"# 股票增强技术分析报告：{symbol}")
        report.append("")
        report.append(f"**分析日期：** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"**分析周期：** {len(df)} 个交易日")
        report.append("")

        # ============ 第一部分：核心结论区 ============
        scoring = strategies_results.get('scoring', {})
        score = scoring.get('score', 50) if scoring and 'error' not in scoring else 50
        position_modifier = scoring.get('position_modifier', 1.0) if scoring else 1.0

        # 使用增强的位置判断
        position_info = self._determine_position_status(latest_data, position_modifier)
        position_info['price_ratio_display'] = position_info['price_ratio'] * 50

        # 评分等级
        if score >= 80:
            rating = "🌟 优秀"
            action = "积极关注"
        elif score >= 65:
            rating = "👍 良好"
            action = "可以参与"
        elif score >= 50:
            rating = "➖ 中等"
            action = "谨慎观望"
        elif score >= 35:
            rating = "⚠️ 偏弱"
            action = "控制风险"
        else:
            rating = "❌ 较差"
            action = "暂时回避"

        report.append("## 📊 核心结论")
        report.append("")
        report.append("| 评分 | 等级 | 建议操作 |")
        report.append("|------|------|----------|")
        report.append(f"| **{score:.1f}分** | {rating} | {action} |")
        report.append("")

        # 位置状态
        position_desc = {
            'high': '高位区域',
            'low': '低位区域',
            'middle': '中位区域'
        }.get(position_info.get('position', 'middle'), '中位区域')

        report.append(f"**位置状态：** {position_desc}")

        # 改进：区分"全面超买"和"短线偏热"
        if position_info.get('is_extreme_overbought'):
            report.append("- 🔴 全面超买，多个指标共振，注意回调风险")
        elif position_info.get('is_overbought'):
            report.append("- ⚠️ 技术超买，注意回调风险")
        elif position_info.get('is_short_term_hot'):
            # 新增：短线偏热但不是全面超买
            wr = position_info.get('wr', 50)
            report.append(f"- 📊 短线偏热（WR={wr:.0f}），但整体并非全面超买")

        if position_info.get('is_extreme_oversold'):
            report.append("- 🟢 全面超卖，多个指标共振，存在反弹机会")
        elif position_info.get('is_oversold'):
            report.append("- 📉 技术超卖，可能存在反弹机会")

        # 显示均线乖离率信息
        bias20 = position_info.get('bias20', 0)
        if abs(bias20) > 0.01:
            report.append(f"- 📈 MA20乖离率：{bias20*100:+.2f}%")
        report.append("")

        # ============ 第二部分：筹码分布分析 ============
        if include_chip:
            report.extend(self._generate_chip_distribution_section(df, latest_data))

        # ============ 第三部分：K线形态分析 ============
        if include_talib_patterns:
            report.extend(self._generate_talib_patterns_section(df))

        # ============ 第四部分：经典策略信号 ============
        if include_strategies:
            report.extend(self._generate_strategies_signals_section(df, symbol))

        # ============ 第五部分：因子评分 ============
        if scoring and 'error' not in scoring:
            report.extend(self._generate_factor_groups_score(scoring, strategies_results))

        # ============ 第六部分：交易计划 ============
        report.extend(self._generate_trading_plan_enhanced(scoring, position_info, latest_data, df))

        # ============ 附录：技术指标详情 ============
        report.append("---")
        report.append("")
        report.append("## 📈 附录：技术指标详情")
        report.append("")
        report.extend(self._generate_indicators_table(latest_data, position_info))

        report.append("")
        report.append("> **免责声明：** 本分析仅供学习参考，不构成投资建议。投资决策应基于全面研究和独立判断。")
        report.append("")

        return "\n".join(report)

    def _generate_chip_distribution_section(self, df: pd.DataFrame, latest_data: pd.Series) -> List[str]:
        """生成筹码分布分析部分"""
        report = []

        try:
            from quanttool.factors.chip_distribution import (
                ChipDistributionCalculator,
                get_chip_assessment
            )

            calculator = ChipDistributionCalculator(lookback_days=210)
            result = calculator.calculate(df)

            report.append("## 🎯 筹码分布分析")
            report.append("")
            report.append("| 指标 | 数值 | 解读 |")
            report.append("|------|------|------|")
            report.append(f"| 筹码集中度 | {result.concentration_ratio:.1f}% | {'高度集中' if result.concentration_ratio >= 70 else '较为集中' if result.concentration_ratio >= 50 else '分散'} |")
            report.append(f"| 平均持仓成本 | ¥{result.avg_cost:.2f} | 主力成本参考 |")
            report.append(f"| 获利盘比例 | {result.profit_ratio:.1f}% | {'获利盘较多' if result.profit_ratio > 70 else '获利盘适中' if result.profit_ratio > 30 else '获利盘较少'} |")
            report.append(f"| 上方套牢压力 | {result.upper_pressure:.1f}% | {'压力大' if result.upper_pressure > 50 else '压力一般'} |")
            report.append(f"| 下方支撑强度 | {result.lower_support:.1f}% | {'支撑强' if result.lower_support > 50 else '支撑一般'} |")
            report.append(f"| 筹码评分 | {result.score:.1f}分 | {'优秀' if result.score >= 70 else '良好' if result.score >= 50 else '一般'} |")
            report.append("")

            # 支撑阻力位
            if result.support_levels:
                support_str = ', '.join([f'¥{p:.2f}' for p in result.support_levels[:3]])
                report.append(f"**支撑位：** {support_str}")

            if result.resistance_levels:
                resistance_str = ', '.join([f'¥{p:.2f}' for p in result.resistance_levels[:3]])
                report.append(f"**阻力位：** {resistance_str}")

            if result.peak_prices:
                peak_str = ', '.join([f'¥{p:.2f}' for p in result.peak_prices[:3]])
                report.append(f"**筹码峰：** {peak_str}")

            report.append("")

            # 定性评估
            assessment = get_chip_assessment(result)
            report.append(f"**综合评估：** {assessment}")
            report.append("")

        except Exception as e:
            report.append("## 🎯 筹码分布分析")
            report.append("")
            report.append(f"*筹码分布分析暂不可用：{str(e)}*")
            report.append("")

        return report

    def _generate_talib_patterns_section(self, df: pd.DataFrame) -> List[str]:
        """生成TA-Lib K线形态分析部分（改进版：区分最新和历史）"""
        report = []

        try:
            from quanttool.factors.talib_patterns import (
                TalibPatternRecognizer,
                get_pattern_assessment,
                TALIB_AVAILABLE
            )

            if not TALIB_AVAILABLE:
                report.append("## 📊 K线形态分析（TA-Lib 61种）")
                report.append("")
                report.append("*TA-Lib 未安装，无法使用此功能。请执行：pip install TA-Lib*")
                report.append("")
                return report

            recognizer = TalibPatternRecognizer()
            result = recognizer.recognize_all(df, lookback=5)

            report.append("## 📊 K线形态分析（TA-Lib 61种形态）")
            report.append("")

            if result.total_patterns == 0:
                report.append("最近5个交易日未识别到明显K线形态。")
                report.append("")
                return report

            # 按类型分组
            bullish = [p for p in result.patterns if p.type == 'bullish']
            bearish = [p for p in result.patterns if p.type == 'bearish']
            neutral = [p for p in result.patterns if p.type == 'neutral']

            # 区分最新形态和历史形态
            all_dates = sorted(set(p.date for p in result.patterns if p.date), reverse=True)
            latest_date = all_dates[0] if all_dates else None

            latest_patterns = [p for p in result.patterns if p.date == latest_date] if latest_date else []
            historical_patterns = [p for p in result.patterns if p.date != latest_date] if latest_date else result.patterns

            # 形态统计（改进：明确说明统计范围）
            report.append("**形态统计（最近5日扫描）：**")
            report.append(f"- 看涨形态：{result.bullish_count}个（方向明确）")
            report.append(f"- 看跌形态：{result.bearish_count}个（方向明确）")
            report.append(f"- 中性形态：{result.neutral_count}个（如十字星、纺锤等）")
            report.append(f"- 综合信号：{result.composite_signal:.1f}（仅统计方向明确形态）")
            report.append("")

            # 最新形态
            if latest_patterns:
                report.append(f"**🔥 最新形态（{latest_date}）：**")
                report.append("")

                latest_bullish = [p for p in latest_patterns if p.type == 'bullish']
                latest_bearish = [p for p in latest_patterns if p.type == 'bearish']
                latest_neutral = [p for p in latest_patterns if p.type == 'neutral']

                if latest_bullish:
                    report.append("📈 看涨：")
                    for p in latest_bullish:
                        report.append(f"  - **{p.name_cn}**（{p.strength}）")
                if latest_bearish:
                    report.append("📉 看跌：")
                    for p in latest_bearish:
                        report.append(f"  - **{p.name_cn}**（{p.strength}）")
                if latest_neutral:
                    report.append("➖ 中性：")
                    for p in latest_neutral[:3]:
                        report.append(f"  - **{p.name_cn}**（{p.strength}）")
                report.append("")

            # 历史形态
            if historical_patterns:
                report.append("**📅 历史命中形态：**")
                report.append("")

                hist_bullish = [p for p in historical_patterns if p.type == 'bullish']
                hist_bearish = [p for p in historical_patterns if p.type == 'bearish']

                if hist_bullish:
                    report.append("📈 看涨：")
                    for p in hist_bullish[:5]:
                        date_str = p.date if p.date else "未知"
                        report.append(f"  - **{p.name_cn}**（{date_str}）")
                if hist_bearish:
                    report.append("📉 看跌：")
                    for p in hist_bearish[:5]:
                        date_str = p.date if p.date else "未知"
                        report.append(f"  - **{p.name_cn}**（{date_str}）")
                report.append("")

            # 综合评估
            assessment = get_pattern_assessment(result)
            report.append(f"**综合评估：** {assessment}")

            # 说明
            if bullish and bearish:
                report.append("")
                report.append("📌 **说明：** 最新形态对当日交易参考价值更高，历史形态反映近期市场状态。")

            report.append("")

        except Exception as e:
            report.append("## 📊 K线形态分析（TA-Lib）")
            report.append("")
            report.append(f"*K线形态分析出错：{str(e)}*")
            report.append("")

        return report

    def _generate_strategies_signals_section(self, df: pd.DataFrame, symbol: str = None) -> List[str]:
        """生成经典策略信号部分（展示所有10个策略）"""
        report = []

        try:
            from quanttool.strategies.classic_strategies import (
                VolumeBreakoutStrategy,
                MADAlignmentStrategy,
                PlatformBreakoutStrategy,
                NoLargeDrawdownStrategy,
                YearLinePullbackStrategy,
                ApronStrategy,
                HighNarrowFlagStrategy,
                VolumeLimitDownStrategy,
                LowATRGrowthStrategy,
                FundamentalSelectionStrategy,
            )

            report.append("## 📋 经典策略信号")
            report.append("")
            report.append("| 策略名称 | 最新信号 | 信号说明 |")
            report.append("|----------|----------|----------|")

            # 所有10个策略
            strategies = [
                ("放量上涨", VolumeBreakoutStrategy()),
                ("均线多头", MADAlignmentStrategy()),
                ("突破平台", PlatformBreakoutStrategy()),
                ("无大幅回撤", NoLargeDrawdownStrategy()),
                ("回踩年线", YearLinePullbackStrategy()),
                ("停机坪", ApronStrategy()),
                ("高而窄旗形", HighNarrowFlagStrategy()),
                ("放量跌停", VolumeLimitDownStrategy()),
                ("低ATR成长", LowATRGrowthStrategy()),
                ("基本面选股", FundamentalSelectionStrategy()),
            ]

            buy_signals = 0
            sell_signals = 0
            hold_signals = 0

            for name, strategy in strategies:
                # 为基本面策略传递股票代码
                if name == "基本面选股" and symbol:
                    strategy.initialize({'symbol': symbol})
                else:
                    strategy.initialize({})

                try:
                    signal = strategy.get_signal(df.iloc[-1], df)
                    direction = signal.get('direction', 'hold')
                    reason = signal.get('reason', '')

                    if direction == 'buy':
                        signal_display = '🟢 买入'
                        buy_signals += 1
                    elif direction == 'sell':
                        signal_display = '🔴 卖出'
                        sell_signals += 1
                    else:
                        signal_display = '➖ 持有'
                        hold_signals += 1

                    # 简化信号说明
                    if reason == 'no_signal':
                        reason_desc = "未触发"
                    elif reason == 'insufficient_data':
                        reason_desc = "数据不足"
                    elif reason == 'requires_fundamental_data':
                        reason_desc = "需基本面数据"
                    else:
                        reason_desc = reason

                    report.append(f"| {name} | {signal_display} | {reason_desc} |")

                except Exception as e:
                    report.append(f"| {name} | ⚠️ 错误 | {str(e)[:20]} |")
                    hold_signals += 1

            report.append("")

            # 策略信号汇总
            total = len(strategies)
            report.append("**策略信号汇总：**")
            report.append(f"- 买入信号：{buy_signals}个")
            report.append(f"- 卖出信号：{sell_signals}个")
            report.append(f"- 持有/观望：{hold_signals}个")

            if buy_signals >= 5:
                report.append("- 🚀 **多策略共振看多**")
            elif sell_signals >= 5:
                report.append("- ⚠️ **多策略共振看空**")
            elif buy_signals >= 3:
                report.append(f"- 📈 {buy_signals}个买入信号，偏多格局")
            elif sell_signals >= 3:
                report.append(f"- 📉 {sell_signals}个卖出信号，偏空格局")
            elif buy_signals == sell_signals and buy_signals > 0:
                report.append("- ➖ 多空均衡，观望为主")
            elif buy_signals > 0:
                report.append(f"- 📊 {buy_signals}个买入信号，信号较弱")
            elif sell_signals > 0:
                report.append(f"- 📊 {sell_signals}个卖出信号，信号较弱")
            else:
                report.append("- ➖ 所有策略观望，无明显方向")

            report.append("")

        except Exception as e:
            report.append("## 📋 经典策略信号")
            report.append("")
            report.append(f"*策略信号分析出错：{str(e)}*")
            report.append("")

        return report

    def _generate_trading_plan_enhanced(
        self,
        scoring: Dict,
        position_info: Dict,
        latest_data: pd.Series,
        df: pd.DataFrame
    ) -> List[str]:
        """生成增强版交易计划（改进：根据具体情况给出精细建议）"""
        report = []

        report.append("## 📝 交易计划建议")
        report.append("")

        score = scoring.get('score', 50) if scoring else 50
        position_modifier = scoring.get('position_modifier', 1.0) if scoring else 1.0
        close = latest_data.get('close', 0)
        ma20 = latest_data.get('ma_20', 0)
        ma50 = latest_data.get('ma_50', 0)
        boll_lower = latest_data.get('boll_lower', 0)

        # 获取位置和趋势信息
        position = position_info.get('position', 'middle')
        is_overbought = position_info.get('is_overbought', False)
        is_oversold = position_info.get('is_oversold', False)
        warnings = scoring.get('warnings', []) if scoring else []

        # 判断趋势方向
        is_downtrend = ma20 < ma50 and close < ma20
        is_uptrend = ma20 > ma50 and close > ma20

        # 根据评分、位置、趋势综合给出建议
        if score >= 70:
            report.append("**操作建议：** 🟢 积极关注")
            report.append(f"- 入场参考：突破前高或回踩MA20（¥{ma20:.2f}）")
            report.append(f"- 止损位：¥{boll_lower:.2f}（布林带下轨）")
            report.append("- 仓位建议：可适当加仓")
        elif score >= 55:
            report.append("**操作建议：** 🟡 可以参与")
            report.append(f"- 入场参考：等待确认信号")
            report.append(f"- 止损位：¥{boll_lower:.2f}")
            report.append("- 仓位建议：正常仓位")
        elif score >= 40:
            report.append("**操作建议：** 🟠 谨慎观望")
            # 改进：根据具体情况给出建议
            if is_downtrend:
                report.append("- 中期趋势未转强，等待均线修复")
                report.append(f"- 关注MA20（¥{ma20:.2f}）能否有效突破")
            else:
                report.append("- 等待更明确的信号")
            report.append("- 仓位建议：轻仓或观望")
        else:
            # 改进：区分"评分低但可观察"和"评分低需回避"
            if is_downtrend and not is_oversold:
                report.append("**操作建议：** 🔴 暂时回避")
                report.append("- 中期趋势偏弱，不建议主动入场")
                report.append("- 若后续放量突破关键阻力，再考虑转积极")
            elif is_oversold:
                report.append("**操作建议：** 🟡 观察等待")
                report.append("- 技术面偏弱但已超卖，关注企稳信号")
                report.append(f"- 若放量站稳关键支撑，可轻仓试多")
            else:
                report.append("**操作建议：** 🟠 暂不入场")
                report.append("- 技术面偏弱，风险收益比不足")
                report.append("- 等待趋势或位置改善")

        report.append("")

        # 持仓者建议（改进：更精细）
        report.append("**已持仓者建议：**")
        if score >= 55:
            report.append("- 可继续持有，关注趋势是否延续")
        elif score >= 40:
            report.append(f"- 观察关键支撑位（如¥{boll_lower:.2f}）是否有效")
            report.append("- 跌破关键支撑再考虑减仓")
        else:
            if is_downtrend:
                report.append("- 建议减仓或止损，等待趋势修复")
            elif is_oversold:
                report.append("- 观察是否出现企稳反弹信号")
                report.append("- 若继续破位，再考虑止损")
            else:
                report.append("- 建议减仓，等待更好的入场时机")

        report.append("")

        # 风险提示（改进：更具体）
        if is_overbought:
            report.append("⚠️ **风险提示：** 技术指标超买，短期回调风险较高")
        if is_oversold:
            report.append("📉 **机会提示：** 技术指标超卖，关注企稳反弹信号")

        # 显示主要警告
        if warnings:
            critical_warnings = [w for w in warnings if '⚠️' in w or '🔴' in w or '风险' in w]
            if critical_warnings:
                report.append("")
                report.append("**关键风险：**")
                for w in critical_warnings[:2]:
                    report.append(f"- {w}")

        report.append("")

        return report

    def _generate_indicators_table(self, latest_data: pd.Series, position_info: Dict) -> List[str]:
        """生成技术指标表格（扩展版：包含所有用到的指标）"""
        report = []

        report.append("| 指标 | 数值 | 状态/解读 |")
        report.append("|------|------|----------|")

        close = latest_data.get('close', 0)

        # ============ 趋势指标 ============
        # 均线系统
        ma5 = latest_data.get('ma_5', 0)
        ma10 = latest_data.get('ma_10', 0)
        ma20 = latest_data.get('ma_20', 0)
        ma50 = latest_data.get('ma_50', 0)
        ma100 = latest_data.get('ma_100', 0)

        if ma20 > ma50:
            ma_status = "多头排列"
        elif ma20 < ma50:
            ma_status = "空头排列"
        else:
            ma_status = "震荡"
        report.append(f"| **均线系统** | MA5:¥{ma5:.2f} MA10:¥{ma10:.2f} MA20:¥{ma20:.2f} MA50:¥{ma50:.2f} | {ma_status} |")

        # BIAS 乖离率
        bias6 = latest_data.get('bias_6', 0)
        bias12 = latest_data.get('bias_12', 0)
        bias24 = latest_data.get('bias_24', 0)
        bias20 = (close / ma20 - 1) * 100 if ma20 > 0 else 0
        if abs(bias20) > 8:
            bias_status = "极端偏离"
        elif abs(bias20) > 5:
            bias_status = "偏离较大"
        else:
            bias_status = "正常区间"
        report.append(f"| BIAS乖离率 | BIAS6:{bias6:.2f}% BIAS12:{bias12:.2f}% BIAS20:{bias20:.2f}% | {bias_status} |")

        # MACD
        macd_val = latest_data.get('macd', 0)
        macd_signal = latest_data.get('macd_signal', 0)
        macd_hist = latest_data.get('macd_hist', 0)
        if macd_val > 0 and macd_hist > 0:
            macd_desc = "多头增强"
        elif macd_val > 0:
            macd_desc = "多头减弱"
        elif macd_val < 0 and macd_hist < 0:
            macd_desc = "空头增强"
        else:
            macd_desc = "空头减弱"
        report.append(f"| MACD | DIF:{macd_val:.4f} DEA:{macd_signal:.4f} 柱:{macd_hist:.4f} | {macd_desc} |")

        # DMI趋势指标
        pdi = latest_data.get('dmi_pdi', latest_data.get('pdi', 0))
        mdi = latest_data.get('dmi_mdi', latest_data.get('mdi', 0))
        adx = latest_data.get('dmi_adx', latest_data.get('adx', 0))
        dmi_diff = abs(pdi - mdi)
        if adx > 25:
            if dmi_diff > 0.5:
                dmi_desc = f"趋势强({'上涨' if pdi > mdi else '下跌'})"
            else:
                dmi_desc = "趋势强但多空平衡"
        elif adx > 20:
            dmi_desc = "趋势中等"
        else:
            dmi_desc = "趋势弱/震荡"
        report.append(f"| DMI趋势 | PDI:{pdi:.1f} MDI:{mdi:.1f} ADX:{adx:.1f} | {dmi_desc} |")

        # ============ 动能指标 ============
        # RSI
        rsi_val = latest_data.get('rsi_24', 50)
        if rsi_val > 80:
            rsi_desc = "严重超买"
        elif rsi_val > 70:
            rsi_desc = "超买区"
        elif rsi_val < 20:
            rsi_desc = "严重超卖"
        elif rsi_val < 30:
            rsi_desc = "超卖区"
        else:
            rsi_desc = "中性"
        report.append(f"| RSI(24) | {rsi_val:.2f} | {rsi_desc} |")

        # KDJ
        k = latest_data.get('kdj_k', 50)
        d = latest_data.get('kdj_d', 50)
        j = latest_data.get('kdj_j', 50)
        if j > 100:
            kdj_desc = "严重超买"
        elif j > 80:
            kdj_desc = "超买"
        elif j < 0:
            kdj_desc = "严重超卖"
        elif j < 20:
            kdj_desc = "超卖"
        else:
            kdj_desc = "正常"
        report.append(f"| KDJ | K:{k:.1f} D:{d:.1f} J:{j:.1f} | {kdj_desc} |")

        # WR威廉指标
        wr = latest_data.get('wr_14', latest_data.get('wr', 50))
        if wr < 10:
            wr_desc = "严重超买"
        elif wr < 20:
            wr_desc = "超买"
        elif wr > 90:
            wr_desc = "严重超卖"
        elif wr > 80:
            wr_desc = "超卖"
        else:
            wr_desc = "正常"
        report.append(f"| WR(14) | {wr:.2f} | {wr_desc} |")

        # CCI顺势指标
        cci = latest_data.get('cci', 0)
        if cci > 100:
            cci_desc = "超买/上涨趋势"
        elif cci < -100:
            cci_desc = "超卖/下跌趋势"
        else:
            cci_desc = "震荡区间"
        report.append(f"| CCI | {cci:.2f} | {cci_desc} |")

        # ============ 波动指标 ============
        # 布林带
        boll_upper = latest_data.get('boll_upper', close)
        boll_mid = latest_data.get('boll_mid', close)
        boll_lower = latest_data.get('boll_lower', close)
        boll_pct = (close - boll_lower) / (boll_upper - boll_lower) if boll_upper != boll_lower else 0.5
        boll_width = (boll_upper - boll_lower) / boll_mid * 100 if boll_mid > 0 else 0
        if boll_pct > 0.9:
            boll_desc = f"上轨附近(超买) 带宽{boll_width:.1f}%"
        elif boll_pct < 0.1:
            boll_desc = f"下轨附近(超卖) 带宽{boll_width:.1f}%"
        else:
            boll_desc = f"位置{boll_pct*100:.0f}% 带宽{boll_width:.1f}%"
        report.append(f"| 布林带 | 上:¥{boll_upper:.2f} 中:¥{boll_mid:.2f} 下:¥{boll_lower:.2f} | {boll_desc} |")

        # ATR真实波幅
        atr = latest_data.get('atr_14', 0)
        atr_pct = atr / close * 100 if close > 0 else 0
        if atr_pct < 1.5:
            atr_desc = "低波动(可能变盘)"
        elif atr_pct > 4:
            atr_desc = "高波动"
        else:
            atr_desc = "正常波动"
        report.append(f"| ATR(14) | ¥{atr:.3f} ({atr_pct:.2f}%) | {atr_desc} |")

        # ============ 成交量指标 ============
        # 成交量
        volume = latest_data.get('volume', 0)
        vol_ma5 = latest_data.get('vol_ma5', volume)
        vol_ma20 = latest_data.get('vol_ma20', volume)
        vol_ratio = volume / vol_ma5 if vol_ma5 > 0 else 1.0
        if vol_ratio > 2:
            vol_desc = f"放量({vol_ratio:.1f}倍)"
        elif vol_ratio < 0.5:
            vol_desc = f"缩量({vol_ratio:.1f}倍)"
        else:
            vol_desc = f"量比{vol_ratio:.2f}"
        report.append(f"| 成交量 | {volume/10000:.0f}手 MA5:{vol_ma5/10000:.0f} MA20:{vol_ma20/10000:.0f} | {vol_desc} |")

        # OBV能量潮
        obv = latest_data.get('obv', 0)
        obv_ma = latest_data.get('obv_ma', obv)
        obv_desc = "资金流入" if obv > obv_ma else "资金流出"
        report.append(f"| OBV | {obv/10000:.1f}万 MA:{obv_ma/10000:.1f}万 | {obv_desc} |")

        # ============ 价格信息 ============
        report.append(f"| 当日价格 | 开:¥{latest_data.get('open', 0):.2f} 高:¥{latest_data.get('high', 0):.2f} 低:¥{latest_data.get('low', 0):.2f} 收:¥{close:.2f} | - |")

        report.append("")

        return report

        report.append("")

        return report
if __name__ == "__main__":
    analyzer = StockAnalyzer()

    # Example: Analyze a stock
    symbol = input("Enter stock symbol (e.g., 601777): ").strip()
    if symbol:
        report = analyzer.analyze_stock(symbol)
        print(report)

        # Optionally save to file
        filename = f"{symbol}_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nAnalysis report saved to {filename}")