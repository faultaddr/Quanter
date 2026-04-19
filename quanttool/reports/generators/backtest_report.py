"""Backtest report generator."""

from typing import Dict, Any

from ..base import ReportGenerator, ReportContext, BacktestReportContext

# Import the original generator for backward compatibility
from ..signal_backtest_report import SignalBacktestReporter as _LegacyBacktestReporter


class BacktestReportGenerator(ReportGenerator):
    """信号回测报告生成器

    生成带历史信号表现的报告
    """

    def __init__(self, db_path: str = "./quanttool.db"):
        """
        初始化回测报告生成器

        Args:
            db_path: 数据库路径
        """
        super().__init__()
        self.db_path = db_path
        self._legacy_reporter = _LegacyBacktestReporter()

    @property
    def report_type(self) -> str:
        return "backtest"

    @property
    def description(self) -> str:
        return "信号回测报告 - 包含历史信号分析、胜率统计、收益分布"

    def gather_data(self, context: ReportContext) -> Dict[str, Any]:
        """收集回测报告数据"""
        if isinstance(context, BacktestReportContext):
            backtest_id = context.backtest_id
        else:
            backtest_id = ""

        return {
            'backtest_id': backtest_id,
            'report_date': context.report_date,
        }

    def render(self, data: Dict[str, Any]) -> str:
        """渲染回测报告"""
        # 这里可以扩展为更详细的报告
        return f"""# 回测报告

**报告日期**: {data['report_date']}
**回测ID**: {data['backtest_id']}

> 此报告由统一报告框架生成
"""


# 注册到工厂
from ..base import ReportFactory
ReportFactory.register(BacktestReportGenerator)
