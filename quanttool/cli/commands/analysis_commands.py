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
from quanttool.infrastructure.stores.meta_db import MetaDB
import pandas as pd
import json

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


def analyze_stock_score(stock_info: Dict[str, str], days: int, analyzer: StockAnalyzer,
                       bias_min: Optional[float] = None, bias_max: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Analyze a single stock and return its score data with BIAS filtering."""
    try:
        symbol = stock_info['code'] if isinstance(stock_info, dict) else stock_info
        name = stock_info.get('name', symbol) if isinstance(stock_info, dict) else symbol

        # Get stock data
        df = analyzer.get_stock_data(symbol, days)
        if df.empty or len(df) < 20:
            return None

        # Calculate technical indicators
        df_with_indicators = analyzer.calculate_technical_indicators(df)

        # Get latest data for BIAS check
        latest = df_with_indicators.iloc[-1]

        # BIAS filtering (乖离率过滤)
        bias_6 = latest.get('bias_6', 0)
        bias_12 = latest.get('bias_12', 0)
        bias_24 = latest.get('bias_24', 0)

        # Apply BIAS filter if specified
        if bias_min is not None and bias_6 < bias_min:
            return None
        if bias_max is not None and bias_6 > bias_max:
            return None

        # Run scoring system
        scoring = ScoringSystem()
        score_result = scoring.calculate_all_scores(df_with_indicators)

        if "error" in score_result:
            return None

        # Get dimension scores
        dims = score_result.get('dimensions', {})

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
            "warnings": score_result['warnings'],
            # BIAS data
            "bias_6": bias_6,
            "bias_12": bias_12,
            "bias_24": bias_24,
            # Individual dimension scores for tracking
            "trend_score": dims.get('trend', {}).get('score', 0),
            "momentum_score": dims.get('momentum', {}).get('score', 0),
            "volatility_score": dims.get('volatility', {}).get('score', 0),
            "capital_score": dims.get('capital', {}).get('score', 0),
            "structure_score": dims.get('structure', {}).get('score', 0),
        }
    except Exception as e:
        return None


@app.command()
def scan(
    market: str = typer.Option("csi300", "--market", "-m", help="Market to scan: csi300, sh, sz, or all"),
    days: int = typer.Option(360, "--days", "-d", help="Number of days to analyze (default: 360)"),
    top_n: int = typer.Option(10, "--top", "-n", help="Number of top stocks to return (default: 10)"),
    # BIAS filter options (乖离率过滤)
    bias_min: Optional[float] = typer.Option(None, "--bias-min", help="Minimum BIAS(6) value to include stock (e.g., -5.0)"),
    bias_max: Optional[float] = typer.Option(None, "--bias-max", help="Maximum BIAS(6) value to include stock (e.g., 5.0)"),
    # Record saving options
    save_record: bool = typer.Option(True, "--save/--no-save", help="Save scan record to database"),
    db_path: str = typer.Option("./quanttool.db", "--db-path", help="Path to the database file"),
):
    """Scan the market for potential opportunities based on technical indicators.

    Examples:
        # Basic scan
        quanttool analysis scan

        # Scan with BIAS filter (乖离率过滤)
        quanttool analysis scan --bias-min -5.0 --bias-max 5.0

        # Scan without saving record
        quanttool analysis scan --no-save
    """

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

    # Display BIAS filter info
    if bias_min is not None or bias_max is not None:
        typer.echo(f"乖离率过滤：BIAS(6) ", nl=False)
        if bias_min is not None:
            typer.echo(f">= {bias_min}% ", nl=False)
        if bias_max is not None:
            typer.echo(f"<= {bias_max}%", nl=False)
        typer.echo()

    typer.echo("-" * 60)

    # Create analyzer instance
    analyzer = StockAnalyzer()

    # Analyze each stock
    results = []
    filtered_count = 0
    total = len(stock_list)

    for i, stock_info in enumerate(stock_list, 1):
        symbol = stock_info['code'] if isinstance(stock_info, dict) else stock_info
        name = stock_info.get('name', symbol) if isinstance(stock_info, dict) else symbol
        typer.echo(f"[{i}/{total}] 分析 {symbol} ({name})...", nl=False)
        result = analyze_stock_score(stock_info, days, analyzer, bias_min, bias_max)
        if result:
            results.append(result)
            typer.echo(f" 评分: {result['total_score']:+d} | BIAS(6): {result.get('bias_6', 0):.2f}%")
        else:
            # Check if filtered by BIAS
            if bias_min is not None or bias_max is not None:
                filtered_count += 1
                typer.echo(f" 跳过 (乖离率过滤)")
            else:
                typer.echo(" 跳过 (数据不足)")

    if not results:
        typer.echo("\n没有成功分析任何股票，请检查数据接口配置。")
        return

    # Sort by total score (descending)
    results.sort(key=lambda x: x['total_score'], reverse=True)

    # Display filter summary
    if bias_min is not None or bias_max is not None:
        typer.echo(f"\n📊 乖离率过滤：{filtered_count} 只股票被过滤")

    # Display top N results
    typer.echo("\n" + "=" * 110)
    typer.echo(f"📊 沪深300成分股评分排名 - Top {top_n}")
    typer.echo("=" * 110)
    typer.echo(f"{'排名':<4} {'代码':<10} {'股票名称':<10} {'收盘价':<10} {'总评分':<8} {'BIAS(6)':<10} {'评级':<12} {'操作建议':<20}")
    typer.echo("-" * 110)

    for i, r in enumerate(results[:top_n], 1):
        symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
        name_short = r['name'][:8] if len(r['name']) > 8 else r['name']
        bias_str = f"{r.get('bias_6', 0):+.2f}%"
        typer.echo(f"{i:<4} {symbol_short:<10} {name_short:<10} ¥{r['close']:<9.2f} {r['total_score']:+d}{'':<6} {bias_str:<10} {r['rating']:<12} {r['action']:<20}")

    # Display detailed breakdown for top 5
    typer.echo("\n" + "=" * 80)
    typer.echo("📈 Top 5 详细评分维度")
    typer.echo("=" * 80)

    for i, r in enumerate(results[:5], 1):
        symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
        typer.echo(f"\n【#{i}】{symbol_short} {r['name']} - 总分: {r['total_score']:+d} | {r['rating']}")
        typer.echo(f"    收盘价: ¥{r['close']:.2f} | 风险等级: {r['risk_level']}")
        typer.echo(f"    乖离率: BIAS(6)={r.get('bias_6', 0):+.2f}% | BIAS(12)={r.get('bias_12', 0):+.2f}% | BIAS(24)={r.get('bias_24', 0):+.2f}%")
        typer.echo(f"    维度评分:")

        dims = r['dimensions']
        typer.echo(f"      • 趋势维度 (MA均线): {dims['trend']['score']:+d} 分 - {dims['trend']['desc']}")
        typer.echo(f"      • 动量维度 (MACD+RSI): {dims['momentum']['score']:+d} 分 - {dims['momentum']['desc']}")
        typer.echo(f"      • 波动维度 (布林带): {dims['volatility']['score']:+d} 分 - {dims['volatility']['desc']}")
        typer.echo(f"      • 资金维度 (OBV+VR): {dims['capital']['score']:+d} 分 - {dims['capital']['desc']}")
        typer.echo(f"      • 结构维度 (DMI): {dims['structure']['score']:+d} 分 - {dims['structure']['desc']}")
        typer.echo(f"      • 乖离率维度 (BIAS): {dims['bias']['score']:+d} 分 - {dims['bias']['desc']}")

        if r['warnings']:
            typer.echo(f"    ⚠️ 风险提示:")
            for warning in r['warnings']:
                typer.echo(f"      - {warning}")

    typer.echo("\n" + "=" * 80)
    typer.echo("📋 评分说明:")
    typer.echo("  • 总分范围: -12 ~ +12 分 (6个维度，每个维度 ±2 分)")
    typer.echo("  • > +4 分: 强烈看多，可考虑买入")
    typer.echo("  • 0 ~ +4 分: 偏多观望")
    typer.echo("  • -4 ~ 0 分: 中性观望")
    typer.echo("  • < -4 分: 看空，考虑减仓")
    typer.echo("=" * 80)

    # Save scan record to database
    if save_record:
        try:
            meta_db = MetaDB(db_path)
            scan_date = datetime.now().isoformat()

            scan_data = {
                "scan_date": scan_date,
                "market": market,
                "days_analyzed": days,
                "total_stocks": len(stock_list),
                "bias_filter_min": bias_min,
                "bias_filter_max": bias_max,
                "results": results[:top_n],  # Only save top N results
            }

            scan_id = meta_db.save_scan_record(scan_data)
            typer.echo(f"\n💾 扫描记录已保存 (ID: {scan_id})")
            typer.echo(f"   数据库: {db_path}")
        except Exception as e:
            typer.echo(f"\n⚠️ 保存扫描记录失败: {e}")


@app.command(name="history")
def scan_history(
    market: str = typer.Option(None, "--market", "-m", help="Filter by market"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of records to show"),
    db_path: str = typer.Option("./quanttool.db", "--db-path", help="Path to the database file"),
):
    """查看扫描历史记录。"""
    try:
        meta_db = MetaDB(db_path)
        history = meta_db.get_scan_history(market=market, limit=limit)

        if not history:
            typer.echo("暂无扫描历史记录")
            return

        typer.echo("\n" + "=" * 100)
        typer.echo("📊 扫描历史记录")
        typer.echo("=" * 100)
        typer.echo(f"{'ID':<36} {'日期':<20} {'市场':<10} {'股票数':<8} {'乖离率过滤':<20}")
        typer.echo("-" * 100)

        for record in history:
            scan_id_short = record['id'][:8] + "..."
            scan_date = record['scan_date'][:19] if record['scan_date'] else "N/A"
            market_str = record['market']
            total_stocks = record['total_stocks']

            bias_filter_str = "无"
            if record['bias_filter_min'] is not None or record['bias_filter_max'] is not None:
                bias_filter_str = ""
                if record['bias_filter_min'] is not None:
                    bias_filter_str += f">={record['bias_filter_min']}%"
                if record['bias_filter_max'] is not None:
                    if bias_filter_str:
                        bias_filter_str += ","
                    bias_filter_str += f"<={record['bias_filter_max']}%"

            typer.echo(f"{scan_id_short:<36} {scan_date:<20} {market_str:<10} {total_stocks:<8} {bias_filter_str:<20}")

        typer.echo("=" * 100)
        typer.echo(f"\n共 {len(history)} 条记录")
        typer.echo(f"使用 `quanttool analysis view <scan_id>` 查看详细结果")
        typer.echo(f"使用 `quanttool analysis compare <scan_id1> <scan_id2>` 对比两次扫描")

    except Exception as e:
        typer.echo(f"查询扫描历史失败: {e}")


@app.command(name="view")
def view_scan(
    scan_id: str = typer.Argument(..., help="Scan record ID (or first 8 characters)"),
    top_n: int = typer.Option(10, "--top", "-n", help="Number of top stocks to show"),
    db_path: str = typer.Option("./quanttool.db", "--db-path", help="Path to the database file"),
):
    """查看指定扫描记录的详细结果。"""
    try:
        meta_db = MetaDB(db_path)

        # Try to find scan by full ID or partial ID
        if len(scan_id) < 36:
            # Search for partial ID match
            history = meta_db.get_scan_history(limit=100)
            matching_scans = [h for h in history if h['id'].startswith(scan_id)]
            if not matching_scans:
                typer.echo(f"未找到匹配的扫描记录: {scan_id}")
                return
            if len(matching_scans) > 1:
                typer.echo(f"找到多个匹配的扫描记录，请提供更完整的ID:")
                for scan in matching_scans:
                    typer.echo(f"  {scan['id']}")
                return
            scan_id = matching_scans[0]['id']

        scan_record = meta_db.get_scan_record(scan_id)

        if not scan_record:
            typer.echo(f"未找到扫描记录: {scan_id}")
            return

        typer.echo("\n" + "=" * 110)
        typer.echo(f"📊 扫描详情 - {scan_record['scan_date'][:19]}")
        typer.echo("=" * 110)
        typer.echo(f"扫描ID: {scan_record['id']}")
        typer.echo(f"市场: {scan_record['market']}")
        typer.echo(f"分析天数: {scan_record['days_analyzed']}")
        typer.echo(f"扫描股票数: {scan_record['total_stocks']}")

        if scan_record['bias_filter_min'] is not None or scan_record['bias_filter_max'] is not None:
            typer.echo(f"乖离率过滤: ", nl=False)
            if scan_record['bias_filter_min'] is not None:
                typer.echo(f"BIAS >= {scan_record['bias_filter_min']}% ", nl=False)
            if scan_record['bias_filter_max'] is not None:
                typer.echo(f"BIAS <= {scan_record['bias_filter_max']}%", nl=False)
            typer.echo()

        results = scan_record.get('results', [])

        typer.echo("\n" + "-" * 110)
        typer.echo(f"{'排名':<4} {'代码':<10} {'股票名称':<10} {'收盘价':<10} {'总评分':<8} {'BIAS(6)':<10} {'评级':<12} {'操作建议':<20}")
        typer.echo("-" * 110)

        for r in results[:top_n]:
            symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
            name_short = r['name'][:8] if len(r['name']) > 8 else r['name']
            bias_str = f"{r.get('bias_6', 0):+.2f}%" if r.get('bias_6') is not None else "N/A"
            typer.echo(f"{r['rank']:<4} {symbol_short:<10} {name_short:<10} ¥{r['close']:<9.2f} {r['total_score']:+d}{'':<6} {bias_str:<10} {r['rating']:<12} {r['action']:<20}")

        typer.echo("=" * 110)

    except Exception as e:
        typer.echo(f"查看扫描记录失败: {e}")


@app.command(name="compare")
def compare_scans(
    scan_id_1: str = typer.Argument(..., help="First scan record ID"),
    scan_id_2: str = typer.Argument(..., help="Second scan record ID"),
    db_path: str = typer.Option("./quanttool.db", "--db-path", help="Path to the database file"),
):
    """对比两次扫描的结果。"""
    try:
        meta_db = MetaDB(db_path)

        # Resolve partial IDs
        def resolve_scan_id(partial_id: str) -> str:
            if len(partial_id) >= 36:
                return partial_id
            history = meta_db.get_scan_history(limit=100)
            matching = [h for h in history if h['id'].startswith(partial_id)]
            if not matching:
                raise ValueError(f"未找到匹配的扫描记录: {partial_id}")
            if len(matching) > 1:
                raise ValueError(f"ID '{partial_id}' 匹配到多个记录，请提供更完整的ID")
            return matching[0]['id']

        scan_id_1 = resolve_scan_id(scan_id_1)
        scan_id_2 = resolve_scan_id(scan_id_2)

        comparison = meta_db.compare_scans(scan_id_1, scan_id_2)

        typer.echo("\n" + "=" * 100)
        typer.echo("📊 扫描结果对比")
        typer.echo("=" * 100)
        typer.echo(f"扫描 1: {comparison['scan_id_1'][:8]}...")
        typer.echo(f"扫描 2: {comparison['scan_id_2'][:8]}...")
        typer.echo()

        # Common stocks
        typer.echo(f"📈 共同关注的股票: {comparison['common_count']} 只")
        if comparison['common_stocks']:
            typer.echo(f"{'股票':<10} {'扫描1排名':<10} {'扫描2排名':<10} {'排名变化':<10} {'扫描1评分':<10} {'扫描2评分':<10} {'评分变化':<10}")
            typer.echo("-" * 80)
            for stock in comparison['common_stocks'][:10]:  # Show top 10
                rank_change = f"+{stock['rank_change']}" if stock['rank_change'] > 0 else str(stock['rank_change'])
                score_change = f"+{stock['score_change']}" if stock['score_change'] > 0 else str(stock['score_change'])
                symbol_short = stock['symbol'].replace('.SH', '').replace('.SZ', '')
                typer.echo(f"{symbol_short:<10} {stock['scan_1_rank']:<10} {stock['scan_2_rank']:<10} {rank_change:<10} {stock['scan_1_score']:<10} {stock['scan_2_score']:<10} {score_change:<10}")

        typer.echo()

        # Only in scan 1
        if comparison['only_in_scan_1']:
            typer.echo(f"📉 仅在扫描1中出现: {comparison['only_in_scan_1_count']} 只")
            typer.echo(f"   {', '.join([s.replace('.SH', '').replace('.SZ', '') for s in comparison['only_in_scan_1'][:10]])}")
            if len(comparison['only_in_scan_1']) > 10:
                typer.echo(f"   ... 还有 {len(comparison['only_in_scan_1']) - 10} 只")
            typer.echo()

        # Only in scan 2
        if comparison['only_in_scan_2']:
            typer.echo(f"📈 仅在扫描2中出现: {comparison['only_in_scan_2_count']} 只")
            typer.echo(f"   {', '.join([s.replace('.SH', '').replace('.SZ', '') for s in comparison['only_in_scan_2'][:10]])}")
            if len(comparison['only_in_scan_2']) > 10:
                typer.echo(f"   ... 还有 {len(comparison['only_in_scan_2']) - 10} 只")

        typer.echo("=" * 100)

    except Exception as e:
        typer.echo(f"对比扫描记录失败: {e}")


if __name__ == "__main__":
    app()