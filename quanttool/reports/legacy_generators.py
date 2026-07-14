"""Report generators for QuantTool."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
from ..core.logging import get_logger
from ..domain.models import BacktestResult, FactorEvaluationResult


logger = get_logger(__name__)


class BaseReportGenerator(ABC):
    """Base class for report generators."""

    def __init__(self, output_dir: str = "./reports"):
        """Initialize the report generator.

        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def generate(self, data: Dict[str, Any], **kwargs) -> str:
        """Generate a report.

        Args:
            data: Data to include in the report
            **kwargs: Additional parameters for report generation

        Returns:
            Path to the generated report file
        """
        pass

    def _save_report(self, content: str, filename: str) -> str:
        """Save report content to file.

        Args:
            content: Report content
            filename: Filename for the report

        Returns:
            Path to the saved report
        """
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Report saved to {filepath}")
        return str(filepath)


class BacktestReportGenerator(BaseReportGenerator):
    """Generator for backtest reports."""

    def generate(
        self,
        result: BacktestResult,
        strategy_name: str = None,
        symbols: List[str] = None,
        **kwargs
    ) -> str:
        """Generate a backtest report.

        Args:
            result: Backtest result
            strategy_name: Name of the strategy used
            symbols: List of symbols traded
            **kwargs: Additional parameters

        Returns:
            Path to the generated report file
        """
        report_lines = []

        # Header
        report_lines.append("=" * 80)
        report_lines.append("BACKTEST REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Strategy Info
        if strategy_name:
            report_lines.append(f"Strategy: {strategy_name}")
        if symbols:
            report_lines.append(f"Symbols: {', '.join(symbols)}")
        report_lines.append(f"Period: {result.start_date.date()} to {result.end_date.date()}")
        report_lines.append("")

        # Performance Summary
        report_lines.append("-" * 40)
        report_lines.append("PERFORMANCE SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Initial Capital: ${result.initial_capital:,.2f}")
        report_lines.append(f"Final Capital: ${result.final_capital:,.2f}")
        report_lines.append(f"Total Return: {result.total_return:.2%}")
        report_lines.append(f"Annual Return: {result.annual_return:.2%}")
        report_lines.append(f"Volatility: {result.volatility:.2%}")
        report_lines.append(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        report_lines.append(f"Sortino Ratio: {result.sortino_ratio:.2f}")
        report_lines.append(f"Max Drawdown: {result.max_drawdown:.2%}")
        report_lines.append("")

        # Trade Statistics
        report_lines.append("-" * 40)
        report_lines.append("TRADE STATISTICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Trades: {result.total_trades}")
        report_lines.append(f"Winning Trades: {result.winning_trades}")
        report_lines.append(f"Losing Trades: {result.losing_trades}")
        report_lines.append(f"Win Rate: {result.win_rate:.2%}")
        report_lines.append(f"Profit Factor: {result.profit_factor:.2f}")
        report_lines.append("")

        # Metrics
        if result.metrics:
            report_lines.append("-" * 40)
            report_lines.append("DETAILED METRICS")
            report_lines.append("-" * 40)
            for metric in result.metrics:
                report_lines.append(f"{metric.name}: {metric.value:.4f}")
                if metric.description:
                    report_lines.append(f"  ({metric.description})")
            report_lines.append("")

        # Trades
        if result.trades:
            report_lines.append("-" * 40)
            report_lines.append("TRADE LIST")
            report_lines.append("-" * 40)
            report_lines.append(f"{'ID':<10} {'Symbol':<12} {'Side':<6} {'Qty':<10} {'Price':<12} {'PnL':<12}")
            report_lines.append("-" * 62)
            for trade in result.trades[:50]:  # Show first 50 trades
                pnl_str = f"${trade.pnl:,.2f}" if trade.pnl else "N/A"
                report_lines.append(
                    f"{trade.id:<10} {trade.symbol:<12} {trade.side:<6} "
                    f"{trade.quantity:<10.2f} ${trade.price:<11.2f} {pnl_str:<12}"
                )
            if len(result.trades) > 50:
                report_lines.append(f"... and {len(result.trades) - 50} more trades")
            report_lines.append("")

        # Footer
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)

        # Save report
        content = "\n".join(report_lines)
        filename = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        return self._save_report(content, filename)


class FactorReportGenerator(BaseReportGenerator):
    """Generator for factor analysis reports."""

    def generate(
        self,
        results: Dict[str, FactorEvaluationResult],
        factor_name: str = None,
        **kwargs
    ) -> str:
        """Generate a factor analysis report.

        Args:
            results: Dictionary mapping symbols to factor evaluation results
            factor_name: Name of the factor analyzed
            **kwargs: Additional parameters

        Returns:
            Path to the generated report file
        """
        report_lines = []

        # Header
        report_lines.append("=" * 80)
        report_lines.append("FACTOR ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if factor_name:
            report_lines.append(f"Factor: {factor_name}")
        report_lines.append(f"Number of Symbols: {len(results)}")
        report_lines.append("")

        if not results:
            report_lines.append("No results to report.")
            content = "\n".join(report_lines)
            filename = f"factor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            return self._save_report(content, filename)

        # Summary Statistics
        report_lines.append("-" * 40)
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 40)

        ics = [r.ic for r in results.values()]
        rank_ics = [r.rank_ic for r in results.values()]
        win_rates = [r.win_rate for r in results.values()]
        sharpe_ratios = [r.sharpe_ratio for r in results.values()]

        report_lines.append(f"IC (mean): {pd.Series(ics).mean():.4f}")
        report_lines.append(f"IC (std): {pd.Series(ics).std():.4f}")
        report_lines.append(f"Rank IC (mean): {pd.Series(rank_ics).mean():.4f}")
        report_lines.append(f"Win Rate (mean): {pd.Series(win_rates).mean():.2%}")
        report_lines.append(f"Sharpe Ratio (mean): {pd.Series(sharpe_ratios).mean():.4f}")
        report_lines.append("")

        # Individual Results
        report_lines.append("-" * 40)
        report_lines.append("INDIVIDUAL SYMBOL RESULTS")
        report_lines.append("-" * 40)
        report_lines.append(
            f"{'Symbol':<12} {'IC':<10} {'Rank IC':<10} {'Win Rate':<10} {'Sharpe':<10} {'Avg Ret':<10}"
        )
        report_lines.append("-" * 62)

        for symbol, result in results.items():
            report_lines.append(
                f"{symbol:<12} {result.ic:<10.4f} {result.rank_ic:<10.4f} "
                f"{result.win_rate:<10.2%} {result.sharpe_ratio:<10.4f} {result.avg_return:<10.4f}"
            )
        report_lines.append("")

        # Top/Bottom Performers
        report_lines.append("-" * 40)
        report_lines.append("TOP 5 BY IC")
        report_lines.append("-" * 40)
        sorted_by_ic = sorted(results.items(), key=lambda x: x[1].ic, reverse=True)
        for symbol, result in sorted_by_ic[:5]:
            report_lines.append(f"{symbol}: IC={result.ic:.4f}, Sharpe={result.sharpe_ratio:.4f}")
        report_lines.append("")

        # Footer
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)

        # Save report
        content = "\n".join(report_lines)
        filename = f"factor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        return self._save_report(content, filename)


class HTMLReportGenerator(BaseReportGenerator):
    """Generator for HTML reports with charts."""

    def generate_backtest_html(
        self,
        result: BacktestResult,
        strategy_name: str = None,
        symbols: List[str] = None,
        **kwargs
    ) -> str:
        """Generate an HTML backtest report.

        Args:
            result: Backtest result
            strategy_name: Name of the strategy
            symbols: List of symbols
            **kwargs: Additional parameters

        Returns:
            Path to the generated HTML report
        """
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report - {strategy_name or 'Strategy'}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; margin-top: 5px; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Report</h1>
        <p><strong>Strategy:</strong> {strategy_name or 'N/A'}</p>
        <p><strong>Period:</strong> {result.start_date.date()} to {result.end_date.date()}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Performance Summary</h2>
        <div class="summary">
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {'positive' if result.total_return >= 0 else 'negative'}">{result.total_return:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Annual Return</div>
                <div class="metric-value {'positive' if result.annual_return >= 0 else 'negative'}">{result.annual_return:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{result.sharpe_ratio:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">{result.max_drawdown:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{result.win_rate:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{result.total_trades}</div>
            </div>
        </div>

        <h2>Trade Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr><td>Initial Capital</td><td>${result.initial_capital:,.2f}</td></tr>
            <tr><td>Final Capital</td><td>${result.final_capital:,.2f}</td></tr>
            <tr><td>Volatility</td><td>{result.volatility:.2%}</td></tr>
            <tr><td>Sortino Ratio</td><td>{result.sortino_ratio:.2f}</td></tr>
            <tr><td>Profit Factor</td><td>{result.profit_factor:.2f}</td></tr>
            <tr><td>Winning Trades</td><td>{result.winning_trades}</td></tr>
            <tr><td>Losing Trades</td><td>{result.losing_trades}</td></tr>
        </table>

        <div class="footer">
            <p>Generated by QuantTool Backtest Engine</p>
            <p>Disclaimer: This report is for informational purposes only. Past performance does not guarantee future results.</p>
        </div>
    </div>
</body>
</html>
"""
        filename = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        return self._save_report(html_content, filename)

    def generate(self, data: Dict[str, Any], **kwargs) -> str:
        """Generate an HTML report (delegates to specific methods)."""
        if "backtest_result" in data:
            return self.generate_backtest_html(data["backtest_result"], **kwargs)
        else:
            raise ValueError("Unknown report type. Use generate_backtest_html for backtest reports.")


class ReportFactory:
    """Factory for creating report generators."""

    _generators = {
        "backtest": BacktestReportGenerator,
        "factor": FactorReportGenerator,
        "html": HTMLReportGenerator,
    }

    @classmethod
    def create(cls, report_type: str, output_dir: str = "./reports") -> BaseReportGenerator:
        """Create a report generator.

        Args:
            report_type: Type of report generator to create
            output_dir: Directory for saving reports

        Returns:
            Report generator instance
        """
        if report_type not in cls._generators:
            raise ValueError(f"Unknown report type: {report_type}. Available: {list(cls._generators.keys())}")

        return cls._generators[report_type](output_dir)

    @classmethod
    def register(cls, report_type: str, generator_class: type):
        """Register a custom report generator.

        Args:
            report_type: Type name for the generator
            generator_class: Generator class to register
        """
        cls._generators[report_type] = generator_class
