"""Attribution report generator."""

from typing import Dict, Any

from ..base import ReportGenerator, ReportContext, AttributionReportContext

# Import the original generator for backward compatibility
from ..signal_attribution import SignalAttributor as _LegacyAttributor


class AttributionReportGenerator(ReportGenerator):
    """信号归因报告生成器

    分析信号表现归因
    """

    def __init__(self, db_path: str = "./quanttool.db"):
        """
        初始化归因报告生成器

        Args:
            db_path: 数据库路径
        """
        super().__init__()
        self.db_path = db_path
        self._legacy_attributor = _LegacyAttributor()

    @property
    def report_type(self) -> str:
        return "attribution"

    @property
    def description(self) -> str:
        return "信号归因报告 - 分析信号表现归因"

    def gather_data(self, context: ReportContext) -> Dict[str, Any]:
        """收集归因报告数据"""
        if isinstance(context, AttributionReportContext):
            portfolio_id = context.portfolio_id
        else:
            portfolio_id = ""

        return {
            'portfolio_id': portfolio_id,
            'report_date': context.report_date,
        }

    def render(self, data: Dict[str, Any]) -> str:
        """渲染归因报告"""
        return f"""# 归因报告

**报告日期**: {data['report_date']}
**组合ID**: {data['portfolio_id']}

> 此报告由统一报告框架生成
"""


# 注册到工厂
from ..base import ReportFactory
ReportFactory.register(AttributionReportGenerator)
