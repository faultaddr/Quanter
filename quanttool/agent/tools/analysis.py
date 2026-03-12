"""
Stock analysis tools for MCP Agent.
"""

from typing import Dict, Any
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from ..schemas.tools import (
    AnalyzeStockInput,
    AnalyzeStockOutput,
    GetStockScoreInput,
    GetStockScoreOutput,
    ScoreBreakdown,
)


def analyze_stock(input_data: AnalyzeStockInput) -> AnalyzeStockOutput:
    """
    Analyze a stock with technical indicators and generate recommendations.

    Args:
        input_data: Analysis input parameters

    Returns:
        AnalyzeStockOutput with analysis results
    """
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer(use_cache=True)
        report = analyzer.analyze_stock(input_data.symbol, input_data.days)

        # Parse key information from report
        lines = report.split('\n')
        recommendation = "未知"
        score = None
        price = None
        change_pct = None
        indicators = {}

        for line in lines:
            if "操作建议" in line or "推荐" in line:
                if "买入" in line:
                    recommendation = "买入"
                elif "卖出" in line or "清仓" in line:
                    recommendation = "卖出"
                elif "观望" in line or "等待" in line:
                    recommendation = "观望"
                elif "轻仓" in line:
                    recommendation = "轻仓买入"

            if "综合评分" in line:
                try:
                    score = int(''.join(filter(str.isdigit, line.split(':')[1].strip())))
                except (ValueError, IndexError):
                    pass

            if "最新价" in line or "当前价格" in line:
                try:
                    price_str = line.split(':')[1].strip().split()[0]
                    price = float(price_str.replace('¥', '').replace(',', ''))
                except (ValueError, IndexError):
                    pass

            if "涨跌幅" in line or "涨幅" in line:
                try:
                    pct_str = line.split(':')[1].strip().replace('%', '')
                    change_pct = float(pct_str)
                except (ValueError, IndexError):
                    pass

        return AnalyzeStockOutput(
            symbol=input_data.symbol,
            analysis_report=report,
            recommendation=recommendation,
            score=score,
            price=price,
            change_pct=change_pct,
            indicators=indicators if indicators else None,
        )

    except Exception as e:
        return AnalyzeStockOutput(
            symbol=input_data.symbol,
            analysis_report="",
            recommendation="分析失败",
            error=str(e),
        )


def get_stock_score(input_data: GetStockScoreInput) -> GetStockScoreOutput:
    """
    Get multi-dimensional score for a stock.

    Args:
        input_data: Input with stock symbol

    Returns:
        GetStockScoreOutput with score breakdown
    """
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer
        from quanttool.factors.scoring_system import ScoringSystem

        analyzer = StockAnalyzer(use_cache=True)
        df = analyzer.get_stock_data(input_data.symbol, days=120)

        if df.empty:
            return GetStockScoreOutput(
                symbol=input_data.symbol,
                total_score=0,
                recommendation="无数据",
                error="无法获取股票数据",
            )

        # Calculate scores using ScoringSystem
        scoring = ScoringSystem()
        score_result = scoring.calculate_score(df)

        total_score = score_result.get('total_score', 0)
        recommendation = "观望"

        if total_score >= 75:
            recommendation = "强烈买入"
        elif total_score >= 65:
            recommendation = "买入"
        elif total_score >= 50:
            recommendation = "观望"
        elif total_score >= 35:
            recommendation = "减仓"
        else:
            recommendation = "卖出"

        breakdown = ScoreBreakdown(
            trend_score=score_result.get('trend_score'),
            momentum_score=score_result.get('momentum_score'),
            capital_flow_score=score_result.get('capital_flow_score'),
            position_modifier=score_result.get('position_modifier'),
        )

        latest = df.iloc[-1]
        price = float(latest.get('close', 0))

        return GetStockScoreOutput(
            symbol=input_data.symbol,
            total_score=total_score,
            score_breakdown=breakdown,
            recommendation=recommendation,
            confidence=score_result.get('confidence'),
            price=price if price > 0 else None,
        )

    except Exception as e:
        return GetStockScoreOutput(
            symbol=input_data.symbol,
            total_score=0,
            recommendation="分析失败",
            error=str(e),
        )
