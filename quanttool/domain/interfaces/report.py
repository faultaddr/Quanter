"""Abstract interface for reports in QuantTool."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import pandas as pd


class IReport(ABC):
    """Abstract interface for report generators."""

    @abstractmethod
    def generate(self, data: Dict[str, Any]) -> Union[str, bytes]:
        """
        Generate report from input data.

        Args:
            data: Input data for report generation

        Returns:
            Generated report (as string for text/html, bytes for binary formats)
        """
        pass

    @abstractmethod
    def save(self, data: Dict[str, Any], filepath: str) -> None:
        """
        Generate and save report to file.

        Args:
            data: Input data for report generation
            filepath: Path to save the report
        """
        pass

    @abstractmethod
    def get_format(self) -> str:
        """
        Get the format of the report.

        Returns:
            Report format (e.g., 'html', 'json', 'csv')
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the report type.

        Returns:
            Report name
        """
        pass
