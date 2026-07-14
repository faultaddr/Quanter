"""Daily report generator."""

from datetime import date
from typing import Dict, Any, Optional

from ..base import ReportGenerator, ReportContext, DailyReportContext

# Import the original generator for backward compatibility
from ..daily_report_generator import DailyReportGenerator as _LegacyDailyReportGenerator


class DailyReportGenerator(ReportGenerator):
    """每日投资报告生成器

    汇总 scan 结果和投资组合表现
    """

    def __init__(self, db_path: str = "./quanttool.db"):
        """
        初始化日报生成器

        Args:
            db_path: 数据库路径
        """
        super().__init__()
        self.db_path = db_path
        self._legacy_generator = _LegacyDailyReportGenerator(db_path)

    @property
    def report_type(self) -> str:
        return "daily"

    @property
    def description(self) -> str:
        return "每日投资报告 - 汇总 scan 结果和投资组合表现"

    def gather_data(self, context: ReportContext) -> Dict[str, Any]:
        """收集日报数据"""
        report_date = context.report_date
        # 使用旧生成器生成报告
        report_content = self._legacy_generator.generate_daily_report(report_date)
        return {
            'report_content': report_content,
            'report_date': report_date,
        }

    def render(self, data: Dict[str, Any]) -> str:
        """渲染日报"""
        return data['report_content']

    def generate(self, context: ReportContext) -> str:
        """生成日报"""
        return self._legacy_generator.generate_daily_report(context.report_date)


# 注册到工厂
from ..base import ReportFactory
ReportFactory.register(DailyReportGenerator)
