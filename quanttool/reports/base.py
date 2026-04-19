"""Base classes for the report system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class ReportContext:
    """报告上下文基类"""
    report_date: date
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class DailyReportContext(ReportContext):
    """日报上下文"""
    pass


@dataclass
class BacktestReportContext(ReportContext):
    """回测报告上下文"""
    backtest_id: str = ""


@dataclass
class AttributionReportContext(ReportContext):
    """归因报告上下文"""
    portfolio_id: str = ""


class ReportGenerator(ABC):
    """报告生成器基类

    所有报告生成器必须实现此接口。

    Usage:
        class MyReportGenerator(ReportGenerator):
            @property
            def report_type(self) -> str:
                return "my_report"

            def gather_data(self, context: ReportContext) -> Dict[str, Any]:
                # 收集数据
                return {"data": ...}

            def render(self, data: Dict[str, Any]) -> str:
                # 渲染报告
                return "Markdown content"
    """

    def __init__(self, template_dir: Optional[Path] = None):
        """
        初始化报告生成器

        Args:
            template_dir: 模板目录路径
        """
        self.template_dir = template_dir

    @property
    @abstractmethod
    def report_type(self) -> str:
        """报告类型（唯一标识）"""
        pass

    @property
    def description(self) -> str:
        """报告描述"""
        return ""

    @property
    def version(self) -> str:
        """报告版本"""
        return "1.0.0"

    @abstractmethod
    def gather_data(self, context: ReportContext) -> Dict[str, Any]:
        """收集报告数据

        Args:
            context: 报告上下文

        Returns:
            Dict: 报告数据
        """
        pass

    @abstractmethod
    def render(self, data: Dict[str, Any]) -> str:
        """渲染报告

        Args:
            data: 报告数据

        Returns:
            str: 渲染后的报告内容（通常是 Markdown）
        """
        pass

    def generate(self, context: ReportContext) -> str:
        """生成报告（模板方法）

        Args:
            context: 报告上下文

        Returns:
            str: 完整的报告内容
        """
        data = self.gather_data(context)
        return self.render(data)

    def get_default_context(self) -> ReportContext:
        """获取默认上下文"""
        return ReportContext(report_date=date.today())


class ReportFactory:
    """报告工厂

    用于注册和创建报告生成器。

    Usage:
        # 注册报告类型
        ReportFactory.register(DailyReportGenerator)
        ReportFactory.register(BacktestReportGenerator)

        # 创建报告生成器
        generator = ReportFactory.create('daily')
        report = generator.generate(context)
    """

    _generators: Dict[str, type[ReportGenerator]] = {}

    @classmethod
    def register(cls, generator_class: type[ReportGenerator]):
        """注册报告生成器

        Args:
            generator_class: 报告生成器类
        """
        # 创建临时实例获取 report_type
        temp_instance = generator_class()
        report_type = temp_instance.report_type
        cls._generators[report_type] = generator_class

    @classmethod
    def create(cls, report_type: str, **kwargs) -> ReportGenerator:
        """创建报告生成器

        Args:
            report_type: 报告类型
            **kwargs: 传递给生成器的参数

        Returns:
            ReportGenerator: 报告生成器实例

        Raises:
            ValueError: 如果报告类型不存在
        """
        if report_type not in cls._generators:
            available = list(cls._generators.keys())
            raise ValueError(f"Unknown report type: {report_type}. Available: {available}")
        return cls._generators[report_type](**kwargs)

    @classmethod
    def list_types(cls) -> list[str]:
        """列出所有可用的报告类型"""
        return list(cls._generators.keys())

    @classmethod
    def is_registered(cls, report_type: str) -> bool:
        """检查报告类型是否已注册"""
        return report_type in cls._generators
