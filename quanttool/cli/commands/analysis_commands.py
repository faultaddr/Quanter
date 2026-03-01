"""Commands for stock analysis."""
import typer
import sys
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

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


def _get_reason_type(skip_reason: Optional[str]) -> str:
    """Extract reason type from skip reason for grouping.

    Examples:
        "数据获取失败" -> "数据获取失败"
        "新股/数据不足 (15条/需20条, 最早2025-01-15)" -> "新股/数据不足"
        "乖离率过滤 (BIAS6=5.5% > 5%)" -> "乖离率过滤"
        "评分错误: xxx" -> "评分错误"
        "分析异常: xxx" -> "分析异常"
    """
    if not skip_reason:
        return "未知原因"
    if skip_reason.startswith("数据获取失败"):
        return "数据获取失败"
    if skip_reason.startswith("新股/数据不足"):
        return "新股/数据不足"
    if skip_reason.startswith("乖离率过滤"):
        return "乖离率过滤"
    if skip_reason.startswith("评分错误"):
        return "评分错误"
    if skip_reason.startswith("分析异常"):
        return "分析异常"
    return skip_reason


def analyze_stock_score(stock_info: Dict[str, str], days: int, analyzer: StockAnalyzer,
                       bias_min: Optional[float] = None, bias_max: Optional[float] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Analyze a single stock and return its score data with BIAS filtering.

    Returns:
        Tuple of (result_dict, skip_reason). If analysis succeeds, result_dict contains
        the analysis data and skip_reason is None. If analysis fails, result_dict is None
        and skip_reason contains the reason for skipping.
    """
    try:
        symbol = stock_info['code'] if isinstance(stock_info, dict) else stock_info
        name = stock_info.get('name', symbol) if isinstance(stock_info, dict) else symbol

        # Get stock data
        df = analyzer.get_stock_data(symbol, days)
        if df.empty:
            return None, "数据获取失败"
        if len(df) < 20:
            # Check if it's a new stock by looking at the date range
            if 'timestamp' in df.columns and len(df) > 0:
                earliest_date = pd.to_datetime(df['timestamp'].min())
                trading_days = len(df)
                days_since_listing = (datetime.now() - earliest_date).days
                return None, f"新股/数据不足 ({trading_days}条/需20条, 最早{earliest_date.strftime('%Y-%m-%d')})"
            else:
                return None, f"数据条数不足 ({len(df)}条/需20条)"

        # Calculate technical indicators
        df_with_indicators = analyzer.calculate_technical_indicators(df)

        # Get latest data for BIAS check
        latest = df_with_indicators.iloc[-1]

        # BIAS filtering (乖离率过滤)
        bias_6 = latest.get('bias_6', 0)
        bias_12 = latest.get('bias_12', 0)
        bias_24 = latest.get('bias_24', 0)

        # Calculate BIAS(20) for display
        close = latest.get('close', 0)
        ma20 = latest.get('ma_20', 0)
        bias_20 = (close / ma20 - 1) * 100 if ma20 > 0 else 0

        # Apply BIAS filter if specified
        if bias_min is not None and bias_6 < bias_min:
            return None, f"乖离率过滤 (BIAS6={bias_6:.2f}% < {bias_min}%)"
        if bias_max is not None and bias_6 > bias_max:
            return None, f"乖离率过滤 (BIAS6={bias_6:.2f}% > {bias_max}%)"

        # Run scoring system
        scoring = ScoringSystem()
        score_result = scoring.calculate_all_scores(df_with_indicators, stock_code=symbol)

        if "error" in score_result:
            return None, f"评分错误: {score_result.get('error', '未知错误')}"

        return {
            "symbol": symbol,
            "name": name,
            "close": latest['close'],
            "daily_return": latest.get('daily_return', 0),
            "score": score_result['score'],
            "score_grade": score_result['score_grade'],
            "trigger_type": score_result['trigger_type'],
            "trigger_detail": score_result['trigger_detail'],
            "factors_raw": score_result['factors_raw'],
            "factors_score": score_result['factors_score'],
            "execution": score_result['execution'],
            "warnings": score_result['warnings'],
            # BIAS data
            "bias_6": bias_6,
            "bias_12": bias_12,
            "bias_24": bias_24,
            "bias_20": bias_20,
        }, None
    except Exception as e:
        return None, f"分析异常: {str(e)}"


@app.command()
def scan(
    market: str = typer.Option("csi300", "--market", "-m", help="Market to scan: csi300, sh, sz, or all"),
    days: int = typer.Option(360, "--days", "-d", help="Number of days to analyze (default: 360)"),
    top_n: int = typer.Option(10, "--top", "-n", help="Number of top stocks to return (default: 10)"),
    # BIAS filter options (乖离率过滤)
    bias_min: Optional[float] = typer.Option(None, "--bias-min", help="Minimum BIAS(6) value to include stock (e.g., -5.0) - Note: Hard filter uses BIAS(20) > +8%"),
    bias_max: Optional[float] = typer.Option(None, "--bias-max", help="Maximum BIAS(6) value to include stock (e.g., 5.0) - Note: Hard filter uses BIAS(20) > +8%"),
    # Record saving options
    save_record: bool = typer.Option(True, "--save/--no-save", help="Save scan record to database"),
    db_path: str = typer.Option("./quanttool.db", "--db-path", help="Path to the database file"),
    # Report generation options
    generate_reports: bool = typer.Option(True, "--reports/--no-reports", help="Generate detailed markdown reports for top N stocks"),
    output_dir: str = typer.Option("./reports", "--output-dir", "-o", help="Directory to save detailed reports"),
):
    """Scan the market for potential opportunities based on technical indicators.

    Examples:
        # Basic scan
        quanttool analysis scan

        # Scan with BIAS filter (乖离率过滤)
        quanttool analysis scan --bias-min -5.0 --bias-max 5.0

        # Scan without saving record and reports
        quanttool analysis scan --no-save --no-reports

        # Scan with custom output directory
        quanttool analysis scan --output-dir ./my_reports
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

    typer.echo("注意：系统已启用 BIAS(20) > +8% 硬过滤（自动剔除追高风险股票）")
    typer.echo("-" * 60)

    # Create analyzer instance
    analyzer = StockAnalyzer()

    # Analyze each stock
    results = []
    skipped_stocks = []  # Track skipped stocks with reasons
    total = len(stock_list)

    for i, stock_info in enumerate(stock_list, 1):
        symbol = stock_info['code'] if isinstance(stock_info, dict) else stock_info
        name = stock_info.get('name', symbol) if isinstance(stock_info, dict) else symbol
        typer.echo(f"[{i}/{total}] 分析 {symbol} ({name})...", nl=False)
        result, skip_reason = analyze_stock_score(stock_info, days, analyzer, bias_min, bias_max)
        if result:
            results.append(result)
            typer.echo(f" 评分: {result['score']:.1f}/100 | 触发: {result['trigger_type']} | BIAS(20): {result.get('bias_20', 0):.2f}%")
        else:
            skipped_stocks.append({
                'symbol': symbol,
                'name': name,
                'reason': skip_reason or "未知原因",
                'reason_type': _get_reason_type(skip_reason)
            })
            typer.echo(f" 跳过 ({skip_reason})")

    if not results:
        typer.echo("\n没有成功分析任何股票，请检查数据接口配置。")
        return

    # Sort by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)

    # Display filter summary
    if bias_min is not None or bias_max is not None:
        typer.echo(f"\n📊 乖离率过滤：{filtered_count} 只股票被过滤")

    # Display top N results
    typer.echo("\n" + "=" * 110)
    typer.echo(f"📊 沪深300成分股评分排名 - Top {top_n}")
    typer.echo("=" * 110)
    typer.echo(f"{'排名':<4} {'代码':<10} {'股票名称':<10} {'收盘价':<10} {'评分':<10} {'BIAS(20)':<10} {'等级':<8} {'触发类型':<12}")
    typer.echo("-" * 110)

    for i, r in enumerate(results[:top_n], 1):
        symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
        name_short = r['name'][:8] if len(r['name']) > 8 else r['name']
        bias_str = f"{r.get('bias_20', 0):+.2f}%"
        typer.echo(f"{i:<4} {symbol_short:<10} {name_short:<10} ¥{r['close']:<9.2f} {r['score']:<10.1f} {bias_str:<10} {r['score_grade']:<8} {r['trigger_type']:<12}")

    # Display detailed breakdown for top N
    typer.echo("\n" + "=" * 80)
    typer.echo(f"📈 Top {top_n} 详细评分维度")
    typer.echo("=" * 80)

    for i, r in enumerate(results[:top_n], 1):
        symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
        typer.echo(f"\n【#{i}】{symbol_short} {r['name']} - 评分: {r['score']:.1f}/100 | 等级: {r['score_grade']} | 触发: {r['trigger_type']}")
        typer.echo(f"    收盘价: ¥{r['close']:.2f}")
        typer.echo(f"    乖离率: BIAS(20)={r.get('bias_20', 0):+.2f}% | BIAS(6)={r.get('bias_6', 0):+.2f}% | BIAS(12)={r.get('bias_12', 0):+.2f}%")

        # 显示触发详情
        if r['trigger_detail']:
            typer.echo(f"    触发详情: {r['trigger_detail']}")

        # 显示因子得分
        typer.echo(f"    因子得分:")
        factors_score = r.get('factors_score', {})
        factors_raw = r.get('factors_raw', {})
        for factor_name, score_val in factors_score.items():
            raw_val = factors_raw.get(factor_name, 'N/A')
            typer.echo(f"      • {factor_name}: {score_val:.2f} (原始值: {raw_val})")

        # 显示交易执行计划
        execution = r.get('execution', {})
        if execution:
            typer.echo(f"    交易计划:")
            if 'position_suggest' in execution:
                typer.echo(f"      • 建议仓位: {execution['position_suggest']}")
            if 'buy_price' in execution:
                typer.echo(f"      • 买入价位: ¥{execution['buy_price']:.2f}")
            if 'stop_price' in execution:
                stop_desc = execution.get('stop_loss_desc', f"{execution.get('stop_loss_pct', 0.05)*100:.0f}%")
                typer.echo(f"      • 止损价位: ¥{execution['stop_price']:.2f} ({stop_desc})")
            if 'action_guide' in execution:
                typer.echo(f"      • 操作指引: {execution['action_guide']}")

        if r['warnings']:
            typer.echo(f"    ⚠️ 风险提示:")
            for warning in r['warnings']:
                typer.echo(f"      - {warning}")

    typer.echo("\n" + "=" * 80)
    typer.echo("📋 评分说明:")
    typer.echo("  • 评分范围: 0-100 分")
    typer.echo("  • >= 85 分: 优秀，强烈看多")
    typer.echo("  • 70-84 分: 良好，偏多")
    typer.echo("  • 50-69 分: 一般，观望")
    typer.echo("  • 35-49 分: 较差，偏空")
    typer.echo("  • < 35 分: 差，看空")
    typer.echo("=" * 80)

    # Display skipped stocks report
    if skipped_stocks:
        typer.echo("\n" + "=" * 80)
        typer.echo(f"⚠️  未纳入计算的股票报告 (共 {len(skipped_stocks)} 只)")
        typer.echo("=" * 80)

        # Group skipped stocks by reason_type
        from collections import defaultdict
        skipped_by_type = defaultdict(list)
        for stock in skipped_stocks:
            skipped_by_type[stock['reason_type']].append(stock)

        # Display each group
        for reason_type, stocks in sorted(skipped_by_type.items(), key=lambda x: -len(x[1])):
            # Show an example reason for this type
            example_reason = stocks[0]['reason']
            typer.echo(f"\n【{reason_type}】({len(stocks)} 只)")
            if len(stocks) <= 3:
                # For small groups, show detailed reasons for each
                for s in stocks:
                    symbol_short = s['symbol'].replace('.SH', '').replace('.SZ', '')
                    typer.echo(f"  {symbol_short}({s['name']}) - {s['reason']}")
            else:
                # For larger groups, show example and list stocks
                if '(' in example_reason and ')' in example_reason:
                    typer.echo(f"  示例: {example_reason}")
                # Show stocks in rows of 5
                stock_list_str = []
                for s in stocks:
                    symbol_short = s['symbol'].replace('.SH', '').replace('.SZ', '')
                    stock_list_str.append(f"{symbol_short}({s['name']})")

                # Print in chunks of 5
                for i in range(0, len(stock_list_str), 5):
                    chunk = stock_list_str[i:i+5]
                    typer.echo("  " + ", ".join(chunk))

        typer.echo("\n" + "=" * 80)
        typer.echo(f"统计: 共扫描 {total} 只股票，成功分析 {len(results)} 只，跳过 {len(skipped_stocks)} 只")
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

    # Generate detailed reports for top N stocks
    if generate_reports and results:
        try:
            import os

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            scan_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = os.path.join(output_dir, f"scan_top{top_n}_{scan_timestamp}.md")

            typer.echo(f"\n📝 正在为 Top {top_n} 股票生成详细分析报告...")

            with open(report_file, 'w', encoding='utf-8') as f:
                # Write report header
                f.write(f"# 股票扫描分析报告\n\n")
                f.write(f"**扫描时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**扫描市场：** {market.upper()}\n\n")
                f.write(f"**分析周期：** {days} 天\n\n")
                f.write(f"**股票总数：** {len(stock_list)} 只\n\n")
                f.write(f"**成功分析：** {len(results)} 只\n\n")
                if bias_min is not None or bias_max is not None:
                    f.write(f"**乖离率过滤：** BIAS(6) ")
                    if bias_min is not None:
                        f.write(f">= {bias_min}% ")
                    if bias_max is not None:
                        f.write(f"<= {bias_max}%")
                    f.write("\n\n")
                f.write("---\n\n")

                # Write summary table
                f.write("## 📊 Top {} 股票概览\n\n".format(top_n))
                f.write("| 排名 | 代码 | 名称 | 收盘价 | 评分 | 等级 | 触发类型 | BIAS(20) |\n")
                f.write("|------|------|------|--------|------|------|----------|----------|\n")

                for i, r in enumerate(results[:top_n], 1):
                    symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
                    name = r.get('name', symbol_short)
                    bias_str = f"{r.get('bias_20', 0):+.2f}%"
                    f.write(f"| {i} | {symbol_short} | {name} | ¥{r['close']:.2f} | {r['score']:.1f} | {r['score_grade']} | {r['trigger_type']} | {bias_str} |\n")

                f.write("\n---\n\n")

                # Generate detailed report for each top stock
                for i, r in enumerate(results[:top_n], 1):
                    symbol = r['symbol']
                    name = r.get('name', symbol)

                    typer.echo(f"  [{i}/{top_n}] 生成 {symbol} ({name}) 的详细报告...")

                    try:
                        # Get full analysis report
                        report = analyzer.analyze_stock(symbol, days)

                        # Write to file
                        f.write(f"## #{i} {symbol} - {name}\n\n")
                        f.write(report)
                        f.write("\n\n---\n\n")

                    except Exception as e:
                        f.write(f"## #{i} {symbol} - {name}\n\n")
                        f.write(f"生成报告时出错: {e}\n\n")
                        f.write("---\n\n")

            typer.echo(f"\n✅ 详细分析报告已保存: {report_file}")

        except Exception as e:
            typer.echo(f"\n⚠️ 生成详细报告失败: {e}")


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
        typer.echo(f"{'排名':<4} {'代码':<10} {'股票名称':<10} {'收盘价':<10} {'评分':<10} {'BIAS(20)':<10} {'等级':<8} {'触发类型':<12}")
        typer.echo("-" * 110)

        for i, r in enumerate(results[:top_n], 1):
            symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
            name_short = r['name'][:8] if len(r['name']) > 8 else r['name']
            bias_str = f"{r.get('bias_20', r.get('bias_6', 0)):+.2f}%"
            score = r.get('score', 0)
            score_grade = r.get('score_grade', 'N/A')
            trigger_type = r.get('trigger_type', 'N/A')
            typer.echo(f"{i:<4} {symbol_short:<10} {name_short:<10} ¥{r['close']:<9.2f} {score:<10.1f} {bias_str:<10} {score_grade:<8} {trigger_type:<12}")

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