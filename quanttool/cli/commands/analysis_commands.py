"""Commands for stock analysis."""
import typer
import sys
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from quanttool.factors.stock_analyzer import StockAnalyzer
from quanttool.factors.scoring_system import ScoringSystem
from quanttool.infrastructure.data_providers.data_fetcher import create_data_fetcher_with_credentials
import pandas as pd

app = typer.Typer()


@app.command(name="single")
def analyze_single(
    symbol: str = typer.Argument(..., help="Stock symbol to analyze (e.g., 601777, 000001.SZ)"),
    days: int = typer.Option(360, "--days", "-d", help="Number of days to analyze (default: 360)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file to save the analysis report")
):
    """Analyze a single stock with technical indicators and trading strategies."""
    _run_analysis(symbol, days, output)


def _run_analysis(symbol: str, days: int, output: Optional[str]):
    """Internal function to run the analysis."""
    typer.echo(f"正在分析股票：{symbol}")
    typer.echo(f"分析周期：{days} 天")
    typer.echo("-" * 50)

    # Create analyzer instance
    analyzer = StockAnalyzer()

    # Run analysis
    report = analyzer.analyze_stock(symbol, days)

    # Print report
    typer.echo(report)

    # Save to file if requested
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(report)
        typer.echo(f"\n分析报告已保存至：{output}")


def get_csi300_constituents() -> List[Dict[str, str]]:
    """Get CSI 300 (沪深300) index constituents with names using DataFetcher."""
    try:
        fetcher = create_data_fetcher_with_credentials()
        fetcher.initialize()

        # Use include_names=True to get both code and name
        constituents = fetcher.get_csi300_constituents(include_names=True)

        if constituents:
            return constituents

        # If both failed, log warning and return empty list
        typer.echo("警告：无法从任何数据源获取沪深300成分股")
        return []

    except Exception as e:
        typer.echo(f"获取沪深300成分股失败: {e}")
        return []


def analyze_stock_score(stock_info: Dict[str, str], days: int, analyzer: StockAnalyzer) -> Optional[Dict[str, Any]]:
    """Analyze a single stock and return its score data."""
    try:
        symbol = stock_info['code'] if isinstance(stock_info, dict) else stock_info
        name = stock_info.get('name', symbol) if isinstance(stock_info, dict) else symbol

        # Get stock data
        df = analyzer.get_stock_data(symbol, days)
        if df.empty or len(df) < 20:
            return None

        # Calculate technical indicators
        df_with_indicators = analyzer.calculate_technical_indicators(df)

        # Run scoring system
        scoring = ScoringSystem()
        score_result = scoring.calculate_all_scores(df_with_indicators)

        if "error" in score_result:
            return None

        # Get latest data
        latest = df_with_indicators.iloc[-1]

        return {
            "symbol": symbol,
            "name": name,
            "close": latest['close'],
            "daily_return": latest.get('daily_return', 0),
            "total_score": score_result['total_score'],
            "rating": score_result['rating'],
            "action": score_result['action'],
            "risk_level": score_result['risk_level'],
            "dimensions": score_result['dimensions'],
            "warnings": score_result['warnings']
        }
    except Exception as e:
        return None


@app.command()
def scan(
    market: str = typer.Option("csi300", "--market", "-m", help="Market to scan: csi300, sh, sz, or all"),
    days: int = typer.Option(360, "--days", "-d", help="Number of days to analyze (default: 360)"),
    top_n: int = typer.Option(10, "--top", "-n", help="Number of top stocks to return (default: 10)")
):
    """Scan the market for potential opportunities based on technical indicators."""

    # Get stock list based on market parameter
    if market.lower() == "csi300":
        typer.echo("正在获取沪深300成分股...")
        stock_list = get_csi300_constituents()
    else:
        typer.echo(f"不支持的扫描市场: {market}")
        typer.echo("当前支持的市场: csi300 (沪深300)")
        return

    if not stock_list:
        typer.echo("无法获取股票列表，请检查数据接口配置。")
        return

    typer.echo(f"共获取 {len(stock_list)} 只股票，开始分析...")
    typer.echo(f"分析周期：{days} 天")
    typer.echo("-" * 60)

    # Create analyzer instance
    analyzer = StockAnalyzer()

    # Analyze each stock
    results = []
    total = len(stock_list)

    for i, stock_info in enumerate(stock_list, 1):
        symbol = stock_info['code'] if isinstance(stock_info, dict) else stock_info
        name = stock_info.get('name', symbol) if isinstance(stock_info, dict) else symbol
        typer.echo(f"[{i}/{total}] 分析 {symbol} ({name})...", nl=False)
        result = analyze_stock_score(stock_info, days, analyzer)
        if result:
            results.append(result)
            typer.echo(f" 评分: {result['total_score']:+d}")
        else:
            typer.echo(" 跳过 (数据不足)")

    if not results:
        typer.echo("\n没有成功分析任何股票，请检查数据接口配置。")
        return

    # Sort by total score (descending)
    results.sort(key=lambda x: x['total_score'], reverse=True)

    # Display top N results
    typer.echo("\n" + "=" * 95)
    typer.echo(f"📊 沪深300成分股评分排名 - Top {top_n}")
    typer.echo("=" * 95)
    typer.echo(f"{'排名':<4} {'代码':<10} {'股票名称':<10} {'收盘价':<10} {'总评分':<8} {'评级':<12} {'操作建议':<20}")
    typer.echo("-" * 95)

    for i, r in enumerate(results[:top_n], 1):
        symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
        name_short = r['name'][:8] if len(r['name']) > 8 else r['name']
        typer.echo(f"{i:<4} {symbol_short:<10} {name_short:<10} ¥{r['close']:<9.2f} {r['total_score']:+d}{'':<6} {r['rating']:<12} {r['action']:<20}")

    # Display detailed breakdown for top 5
    typer.echo("\n" + "=" * 80)
    typer.echo("📈 Top 5 详细评分维度")
    typer.echo("=" * 80)

    for i, r in enumerate(results[:5], 1):
        symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
        typer.echo(f"\n【#{i}】{symbol_short} {r['name']} - 总分: {r['total_score']:+d} | {r['rating']}")
        typer.echo(f"    收盘价: ¥{r['close']:.2f} | 风险等级: {r['risk_level']}")
        typer.echo(f"    维度评分:")

        dims = r['dimensions']
        typer.echo(f"      • 趋势维度 (MA均线): {dims['trend']['score']:+d} 分 - {dims['trend']['desc']}")
        typer.echo(f"      • 动量维度 (MACD+RSI): {dims['momentum']['score']:+d} 分 - {dims['momentum']['desc']}")
        typer.echo(f"      • 波动维度 (布林带): {dims['volatility']['score']:+d} 分 - {dims['volatility']['desc']}")
        typer.echo(f"      • 资金维度 (OBV+VR): {dims['capital']['score']:+d} 分 - {dims['capital']['desc']}")
        typer.echo(f"      • 结构维度 (DMI): {dims['structure']['score']:+d} 分 - {dims['structure']['desc']}")

        if r['warnings']:
            typer.echo(f"    ⚠️ 风险提示:")
            for warning in r['warnings']:
                typer.echo(f"      - {warning}")

    typer.echo("\n" + "=" * 80)
    typer.echo("📋 评分说明:")
    typer.echo("  • 总分范围: -10 ~ +10 分")
    typer.echo("  • > +3 分: 强烈看多，可考虑买入")
    typer.echo("  • 0 ~ +3 分: 偏多观望")
    typer.echo("  • -3 ~ 0 分: 中性观望")
    typer.echo("  • < -3 分: 看空，考虑减仓")
    typer.echo("=" * 80)


if __name__ == "__main__":
    app()