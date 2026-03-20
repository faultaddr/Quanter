"""
信号回测报告模块

生成带历史信号表现的报告，包含：
- 历史信号分析
- 胜率统计
- 收益分布
- MFE/MAE分析
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


@dataclass
class SignalPerformance:
    """信号表现统计"""
    signal_count: int           # 信号次数
    win_count: int              # 盈利次数
    loss_count: int             # 亏损次数
    win_rate: float             # 胜率
    avg_return_5d: float        # 5日平均收益
    avg_return_10d: float       # 10日平均收益
    avg_return_20d: float       # 20日平均收益
    max_return: float           # 最大收益
    max_loss: float             # 最大亏损
    mfe_avg: float              # 平均最大有利偏移
    mae_avg: float              # 平均最大不利偏移
    profit_factor: float        # 盈亏比
    sharpe: float               # 夏普比率


@dataclass
class HistoricalSignalAnalysis:
    """历史信号分析结果"""
    overall: SignalPerformance
    by_score_range: Dict[str, SignalPerformance]
    by_market_regime: Dict[str, SignalPerformance]
    recent_signals: List[Dict]
    timestamp: datetime


class SignalBacktestReporter:
    """
    信号回测报告器

    分析历史信号表现
    """

    def __init__(
        self,
        score_threshold_buy: float = 70.0,
        score_threshold_sell: float = 50.0,
        lookback_days: int = 250
    ):
        """
        初始化报告器

        Args:
            score_threshold_buy: 买入信号评分阈值
            score_threshold_sell: 卖出信号评分阈值
            lookback_days: 回看天数
        """
        self.score_threshold_buy = score_threshold_buy
        self.score_threshold_sell = score_threshold_sell
        self.lookback_days = lookback_days

    def analyze_historical_signals(
        self,
        df: pd.DataFrame,
        score_column: str = 'final_score'
    ) -> HistoricalSignalAnalysis:
        """
        分析历史信号表现

        Args:
            df: 包含评分和价格数据的DataFrame
            score_column: 评分列名

        Returns:
            HistoricalSignalAnalysis: 分析结果
        """
        if len(df) < 30:
            return self._create_empty_analysis()

        df = df.copy()

        # 确保有评分数据
        if score_column not in df.columns:
            # 尝试计算评分
            if 'close' in df.columns:
                df['final_score'] = self._calculate_simple_score(df)
                score_column = 'final_score'
            else:
                return self._create_empty_analysis()

        # 计算未来收益
        df['return_5d'] = df['close'].pct_change(5).shift(-5)
        df['return_10d'] = df['close'].pct_change(10).shift(-10)
        df['return_20d'] = df['close'].pct_change(20).shift(-20)

        # 识别买入信号
        df['buy_signal'] = df[score_column] >= self.score_threshold_buy

        # 计算MFE/MAE
        df['mfe_5d'] = self._calculate_mfe(df, 5)
        df['mae_5d'] = self._calculate_mae(df, 5)

        # 获取买入信号点
        buy_signals = df[df['buy_signal']].copy()

        if len(buy_signals) == 0:
            return self._create_empty_analysis()

        # 整体表现
        overall = self._calculate_performance(buy_signals)

        # 按评分区间分析
        by_score_range = self._analyze_by_score_range(buy_signals, score_column)

        # 按市场状态分析（简化版）
        by_market_regime = self._analyze_by_market_regime(buy_signals)

        # 最近信号
        recent_signals = self._get_recent_signals(buy_signals, n=10)

        return HistoricalSignalAnalysis(
            overall=overall,
            by_score_range=by_score_range,
            by_market_regime=by_market_regime,
            recent_signals=recent_signals,
            timestamp=datetime.now()
        )

    def _calculate_simple_score(self, df: pd.DataFrame) -> pd.Series:
        """简单评分计算"""
        # 基于RSI和均线位置
        close = df['close']
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # 评分 = 50 + RSI偏离 + 均线位置
        score = 50 + (rsi - 50) * 0.3

        # 均线排列加分
        ma_bullish = (close > ma5) & (ma5 > ma10) & (ma10 > ma20)
        ma_bearish = (close < ma5) & (ma5 < ma10) & (ma10 < ma20)

        score = np.where(ma_bullish, score + 15, score)
        score = np.where(ma_bearish, score - 15, score)

        return pd.Series(score, index=df.index).clip(0, 100)

    def _calculate_mfe(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        """计算最大有利偏移"""
        mfe = []
        for i in range(len(df)):
            if i + horizon >= len(df):
                mfe.append(np.nan)
                continue

            entry_price = df['close'].iloc[i]
            future_high = df['high'].iloc[i:i+horizon+1].max()
            mfe.append((future_high - entry_price) / entry_price)

        return pd.Series(mfe, index=df.index)

    def _calculate_mae(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        """计算最大不利偏移"""
        mae = []
        for i in range(len(df)):
            if i + horizon >= len(df):
                mae.append(np.nan)
                continue

            entry_price = df['close'].iloc[i]
            future_low = df['low'].iloc[i:i+horizon+1].min()
            mae.append((entry_price - future_low) / entry_price)

        return pd.Series(mae, index=df.index)

    def _calculate_performance(self, signals_df: pd.DataFrame) -> SignalPerformance:
        """计算信号表现"""
        returns_5d = signals_df['return_5d'].dropna()
        returns_10d = signals_df['return_10d'].dropna()
        returns_20d = signals_df['return_20d'].dropna()

        win_mask = returns_5d > 0
        win_count = win_mask.sum()
        loss_count = (~win_mask).sum()
        total = len(returns_5d)

        if total == 0:
            return self._create_empty_performance()

        win_rate = win_count / total
        avg_return_5d = returns_5d.mean()
        avg_return_10d = returns_10d.mean() if len(returns_10d) > 0 else 0
        avg_return_20d = returns_20d.mean() if len(returns_20d) > 0 else 0

        max_return = returns_5d.max()
        max_loss = returns_5d.min()

        # MFE/MAE
        mfe_avg = signals_df['mfe_5d'].mean() if 'mfe_5d' in signals_df.columns else 0
        mae_avg = signals_df['mae_5d'].mean() if 'mae_5d' in signals_df.columns else 0

        # 盈亏比
        gross_profit = returns_5d[win_mask].sum() if win_count > 0 else 0
        gross_loss = abs(returns_5d[~win_mask].sum()) if loss_count > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # 夏普比率
        sharpe = returns_5d.mean() / returns_5d.std() * np.sqrt(252/5) if returns_5d.std() > 0 else 0

        return SignalPerformance(
            signal_count=total,
            win_count=int(win_count),
            loss_count=int(loss_count),
            win_rate=win_rate,
            avg_return_5d=avg_return_5d,
            avg_return_10d=avg_return_10d,
            avg_return_20d=avg_return_20d,
            max_return=max_return,
            max_loss=max_loss,
            mfe_avg=mfe_avg,
            mae_avg=mae_avg,
            profit_factor=profit_factor,
            sharpe=sharpe
        )

    def _analyze_by_score_range(
        self,
        signals_df: pd.DataFrame,
        score_column: str
    ) -> Dict[str, SignalPerformance]:
        """按评分区间分析"""
        ranges = {
            '70-75': (70, 75),
            '75-80': (75, 80),
            '80-85': (80, 85),
            '85-90': (85, 90),
            '90-100': (90, 100)
        }

        results = {}
        for name, (low, high) in ranges.items():
            mask = (signals_df[score_column] >= low) & (signals_df[score_column] < high)
            if mask.sum() > 0:
                results[name] = self._calculate_performance(signals_df[mask])

        return results

    def _analyze_by_market_regime(
        self,
        signals_df: pd.DataFrame
    ) -> Dict[str, SignalPerformance]:
        """按市场状态分析"""
        # 简化版：使用收益方向判断市场状态
        if 'return_5d' not in signals_df.columns:
            return {}

        returns = signals_df['return_5d'].dropna()

        # 用前5日市场收益判断状态
        results = {
            'bull': self._calculate_performance(
                signals_df[signals_df['return_5d'] > 0.02]
            ),
            'bear': self._calculate_performance(
                signals_df[signals_df['return_5d'] < -0.02]
            ),
            'sideway': self._calculate_performance(
                signals_df[signals_df['return_5d'].abs() <= 0.02]
            )
        }

        return results

    def _get_recent_signals(
        self,
        signals_df: pd.DataFrame,
        n: int = 10
    ) -> List[Dict]:
        """获取最近N个信号"""
        recent = signals_df.tail(n)

        signals = []
        for idx, row in recent.iterrows():
            timestamp = row.get('timestamp', idx) if 'timestamp' in row.index else idx

            signal = {
                'timestamp': str(timestamp),
                'close': row.get('close', 0),
                'return_5d': row.get('return_5d', None),
                'return_10d': row.get('return_10d', None),
                'mfe_5d': row.get('mfe_5d', None),
                'mae_5d': row.get('mae_5d', None)
            }
            signals.append(signal)

        return signals

    def _create_empty_performance(self) -> SignalPerformance:
        """创建空的表现数据"""
        return SignalPerformance(
            signal_count=0,
            win_count=0,
            loss_count=0,
            win_rate=0.0,
            avg_return_5d=0.0,
            avg_return_10d=0.0,
            avg_return_20d=0.0,
            max_return=0.0,
            max_loss=0.0,
            mfe_avg=0.0,
            mae_avg=0.0,
            profit_factor=0.0,
            sharpe=0.0
        )

    def _create_empty_analysis(self) -> HistoricalSignalAnalysis:
        """创建空的分析结果"""
        return HistoricalSignalAnalysis(
            overall=self._create_empty_performance(),
            by_score_range={},
            by_market_regime={},
            recent_signals=[],
            timestamp=datetime.now()
        )

    def generate_report_markdown(
        self,
        analysis: HistoricalSignalAnalysis,
        symbol: str = "Unknown"
    ) -> str:
        """
        生成Markdown格式报告

        Args:
            analysis: 分析结果
            symbol: 股票代码

        Returns:
            str: Markdown报告
        """
        lines = []

        lines.append(f"# 历史信号分析报告 - {symbol}")
        lines.append(f"\n生成时间: {analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 整体表现
        lines.append("## 整体表现")
        lines.append("")
        overall = analysis.overall
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|------|------|")
        lines.append(f"| 信号次数 | {overall.signal_count} |")
        lines.append(f"| 盈利次数 | {overall.win_count} |")
        lines.append(f"| 亏损次数 | {overall.loss_count} |")
        lines.append(f"| 5日胜率 | {overall.win_rate:.2%} |")
        lines.append(f"| 平均5日收益 | {overall.avg_return_5d:.2%} |")
        lines.append(f"| 平均10日收益 | {overall.avg_return_10d:.2%} |")
        lines.append(f"| 最佳情况 (MFE) | {overall.mfe_avg:.2%} |")
        lines.append(f"| 最差情况 (MAE) | {overall.mae_avg:.2%} |")
        lines.append(f"| 盈亏比 | {overall.profit_factor:.2f} |")
        lines.append("")

        # 按评分区间
        if analysis.by_score_range:
            lines.append("## 按评分区间分析")
            lines.append("")
            lines.append(f"| 评分区间 | 信号数 | 5日胜率 | 平均收益 | 盈亏比 |")
            lines.append(f"|----------|--------|---------|----------|--------|")

            for range_name, perf in analysis.by_score_range.items():
                lines.append(
                    f"| {range_name} | {perf.signal_count} | "
                    f"{perf.win_rate:.1%} | {perf.avg_return_5d:.2%} | "
                    f"{perf.profit_factor:.2f} |"
                )
            lines.append("")

        # 最近信号
        if analysis.recent_signals:
            lines.append("## 最近信号")
            lines.append("")
            for signal in analysis.recent_signals[-5:]:
                ret_5d = signal.get('return_5d')
                ret_str = f"{ret_5d:.2%}" if ret_5d is not None else "N/A"
                lines.append(f"- {signal['timestamp']}: 5日收益 {ret_str}")

        lines.append("")
        lines.append("---")
        lines.append("*本报告由 QuantTool 信号回测系统生成*")

        return "\n".join(lines)