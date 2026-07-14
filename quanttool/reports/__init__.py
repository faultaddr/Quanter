"""Reports module for QuantTool.

This module provides a unified report generation framework with:

- Base classes: ReportGenerator, ReportContext, ReportFactory
- Generators: DailyReportGenerator, BacktestReportGenerator, AttributionReportGenerator
- Legacy support: SignalBacktestReporter, SignalPerformance

Usage:
    from quanttool.reports import ReportFactory

    # Create a daily report
    generator = ReportFactory.create('daily')
    report = generator.generate(context)

    # List available report types
    types = ReportFactory.list_types()
"""

# New unified framework
from .base import (
    ReportGenerator,
    ReportContext,
    DailyReportContext,
    BacktestReportContext,
    AttributionReportContext,
    ReportFactory,
)

# Generators
from .generators import (
    DailyReportGenerator,
    BacktestReportGenerator,
    AttributionReportGenerator,
)

# Legacy support
from .signal_backtest_report import (
    SignalBacktestReporter,
    SignalPerformance,
    HistoricalSignalAnalysis,
)

from .signal_attribution import SignalAttributor

# Legacy generators (for backward compatibility)
from .daily_report_generator import DailyReportGenerator as LegacyDailyReportGenerator
from .legacy_generators import (
    BaseReportGenerator,
    FactorReportGenerator,
    HTMLReportGenerator,
)

__all__ = [
    # New framework
    'ReportGenerator',
    'ReportContext',
    'DailyReportContext',
    'BacktestReportContext',
    'AttributionReportContext',
    'ReportFactory',
    # Generators
    'DailyReportGenerator',
    'BacktestReportGenerator',
    'AttributionReportGenerator',
    # Legacy
    'SignalBacktestReporter',
    'SignalPerformance',
    'HistoricalSignalAnalysis',
    'SignalAttributor',
    'LegacyDailyReportGenerator',
    'BaseReportGenerator',
    'FactorReportGenerator',
    'HTMLReportGenerator',
]
