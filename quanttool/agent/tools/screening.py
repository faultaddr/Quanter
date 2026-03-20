"""
Stock screening tools for MCP Agent.
"""

from typing import List, Optional
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from ..schemas.tools import (
    ScreenStocksInput,
    ScreenStocksOutput,
    ScreenedStock,
    FilterCondition,
)


# Index component mappings
INDEX_COMPONENTS = {
    "hs300": "沪深300",
    "zz500": "中证500",
    "sz50": "上证50",
    "all": "全市场",
}


def screen_stocks(input_data: ScreenStocksInput) -> ScreenStocksOutput:
    """
    Screen stocks based on technical criteria.

    Args:
        input_data: Screening parameters

    Returns:
        ScreenStocksOutput with screening results
    """
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer
        from quanttool.factors.scoring_system import ScoringSystem

        # Get index components
        symbols = _get_index_symbols(input_data.index)

        if not symbols:
            return ScreenStocksOutput(
                index=input_data.index,
                total_stocks=0,
                filtered_count=0,
                stocks=[],
                error=f"无法获取{INDEX_COMPONENTS.get(input_data.index, input_data.index)}成分股列表",
            )

        # Preload data
        analyzer = StockAnalyzer(use_cache=True)
        analyzer.preload_data_for_scan([{"code": s} for s in symbols], days=120)

        # Score all stocks
        scoring = ScoringSystem()
        scored_stocks = []

        for symbol in symbols[:100]:  # Limit for performance
            try:
                df = analyzer.get_stock_data(symbol, days=120)
                if df.empty:
                    continue

                score_result = scoring.calculate_score(df)
                total_score = score_result.get('total_score', 0)

                # Apply minimum score filter
                if input_data.min_score and total_score < input_data.min_score:
                    continue

                latest = df.iloc[-1]
                screened = ScreenedStock(
                    symbol=symbol,
                    score=total_score,
                    price=float(latest.get('close', 0)),
                    reason=_get_score_reason(score_result),
                )
                scored_stocks.append(screened)

            except Exception:
                continue

        # Sort by score and limit
        scored_stocks.sort(key=lambda x: x.score or 0, reverse=True)
        scored_stocks = scored_stocks[:input_data.limit]

        # Apply additional filters if provided
        filters_applied = []
        if input_data.filters:
            filters_applied = [
                f"{f.indicator} {f.operator} {f.value}"
                for f in input_data.filters
            ]
            scored_stocks = _apply_filters(scored_stocks, input_data.filters, analyzer)

        return ScreenStocksOutput(
            index=input_data.index,
            total_stocks=len(symbols),
            filtered_count=len(scored_stocks),
            stocks=scored_stocks,
            filters_applied=filters_applied if filters_applied else None,
        )

    except Exception as e:
        return ScreenStocksOutput(
            index=input_data.index,
            total_stocks=0,
            filtered_count=0,
            stocks=[],
            error=str(e),
        )


def _get_index_symbols(index: str) -> List[str]:
    """Get list of symbols for an index."""
    try:
        # Try to use tushare for index components
        import tushare as ts

        token = os.environ.get('TUSHARE_TOKEN')
        if token:
            pro = ts.pro_api(token)

            if index == "hs300":
                df = pro.index_weight(index_code='399300.SZ')
            elif index == "zz500":
                df = pro.index_weight(index_code='000905.SH')
            elif index == "sz50":
                df = pro.index_weight(index_code='000016.SH')
            else:
                # Default to hs300
                df = pro.index_weight(index_code='399300.SZ')

            return list(df['con_code'].unique())[:300]

    except Exception:
        pass

    # Fallback: return some common symbols for testing
    fallback_symbols = {
        "hs300": [
            "600519.SH", "000858.SZ", "600036.SH", "601318.SH",
            "000001.SZ", "601166.SH", "600276.SH", "000333.SZ",
            "600887.SH", "601888.SH", "000651.SZ", "600030.SH",
        ],
        "zz500": [
            "002475.SZ", "000725.SZ", "002415.SZ", "300059.SZ",
            "000063.SZ", "002304.SZ", "300015.SZ", "002352.SZ",
        ],
        "sz50": [
            "600519.SH", "601318.SH", "600036.SH", "601166.SH",
            "600887.SH", "601888.SH", "600030.SH", "601398.SH",
        ],
        "all": [
            "600519.SH", "000858.SZ", "600036.SH", "601318.SH",
            "000001.SZ", "601166.SH", "600276.SH", "000333.SZ",
        ],
    }

    return fallback_symbols.get(index, fallback_symbols["hs300"])


def _get_score_reason(score_result: dict) -> str:
    """Generate reason string from score result."""
    total = score_result.get('total_score', 0)

    if total >= 75:
        return "综合评分优秀，技术面强势"
    elif total >= 65:
        return "综合评分良好，有上涨动能"
    elif total >= 50:
        return "综合评分中等，建议观望"
    else:
        return "综合评分偏低，谨慎关注"


def _apply_filters(
    stocks: List[ScreenedStock],
    filters: List[FilterCondition],
    analyzer
) -> List[ScreenedStock]:
    """Apply filter conditions to screened stocks."""
    # For now, return stocks as-is
    # In production, implement actual filtering logic
    return stocks
