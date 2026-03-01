"""Portfolio backtest CLI commands."""

import typer
from typing import Optional
from datetime import datetime

app = typer.Typer()


@app.command()
def create(
    scan_id: str = typer.Argument(..., help="Scan record ID"),
    capital: float = typer.Option(500000, "--capital", "-c", help="初始资金"),
    top_n: int = typer.Option(5, "--top", "-n", help="选取前 N 只股票"),
):
    """从 scan 结果创建投资组合回测."""
    from quanttool.application.portfolio_backtest_service import PortfolioBacktestService

    service = PortfolioBacktestService()

    try:
        backtest_id = service.create_portfolio_from_scan(
            scan_id=scan_id,
            initial_capital=capital,
            top_n=top_n
        )
        typer.echo(f"✓ 投资组合已创建")
        typer.echo(f"  回测 ID: {backtest_id}")
    except Exception as e:
        typer.echo(f"✗ 创建失败: {e}")


@app.command()
def list(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="筛选状态: active/closed"),
    limit: int = typer.Option(20, "--limit", "-l", help="最多显示数量"),
):
    """列出投资组合回测."""
    from quanttool.infrastructure.stores.meta_db import MetaDB

    db = MetaDB()

    # 查询数据库
    conn = db._connect()
    cursor = conn.cursor()

    query = "SELECT id, portfolio_name, initial_capital, status, start_date, end_date, total_return FROM portfolio_backtests"
    params = []

    if status:
        query += " WHERE status = ?"
        params.append(status)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        typer.echo("暂无投资组合回测记录")
        return

    typer.echo(f"{'ID':<36} {'名称':<15} {'状态':<8} {'初始资金':>12} {'总收益':>10} {'日期':<21}")
    typer.echo("-" * 110)

    for row in rows:
        backtest_id, name, initial, stat, start, end, ret = row
        name = (name or "-")[:15]
        ret_str = f"{ret:+.2f}%" if ret is not None else "N/A"
        date_str = f"{start} ~ {end}" if end else f"{start} ~"

        typer.echo(f"{backtest_id:<36} {name:<15} {stat:<8} {initial:>12,.0f} {ret_str:>10} {date_str:<21}")


@app.command()
def view(
    backtest_id: str = typer.Argument(..., help="回测 ID"),
):
    """查看投资组合详情."""
    from quanttool.infrastructure.stores.meta_db import MetaDB

    db = MetaDB()
    backtest = db.get_portfolio_backtest(backtest_id)

    if not backtest:
        typer.echo(f"未找到回测: {backtest_id}")
        return

    typer.echo(f"\n{'='*60}")
    typer.echo(f"投资组合: {backtest.get('portfolio_name', '-')}")
    typer.echo(f"{'='*60}")

    typer.echo(f"\n基本信息:")
    typer.echo(f"  初始资金: {backtest.get('initial_capital', 0):,.0f} 元")
    typer.echo(f"  状态: {backtest.get('status', '-')}")
    typer.echo(f"  开始日期: {backtest.get('start_date', '-')}")
    if backtest.get('end_date'):
        typer.echo(f"  结束日期: {backtest.get('end_date')}")

    if backtest.get('total_return') is not None:
        typer.echo(f"\n绩效指标:")
        typer.echo(f"  总收益率: {backtest.get('total_return', 0):+.2f}%")
        typer.echo(f"  年化收益率: {backtest.get('annualized_return', 0):+.2f}%")
        typer.echo(f"  夏普比率: {backtest.get('sharpe_ratio', 0):.2f}")
        typer.echo(f"  最大回撤: {backtest.get('max_drawdown', 0):.2f}%")

    # 持仓明细
    holdings = backtest.get('holdings', [])
    if holdings:
        typer.echo(f"\n持仓明细:")
        typer.echo(f"  {'代码':<10} {'名称':<10} {'买入价':>10} {'数量':>10} {'状态':<8}")
        typer.echo("  " + "-" * 55)
        for h in holdings:
            typer.echo(
                f"  {h.get('symbol', '-'):<10} "
                f"{h.get('name', '-')[:10]:<10} "
                f"{h.get('entry_price', 0):>10.2f} "
                f"{h.get('shares', 0):>10} "
                f"{h.get('status', '-'):<8}"
            )
            if h.get('status') == 'closed' and h.get('realized_return') is not None:
                typer.echo(f"    └─ 卖出价: {h.get('exit_price', 0):.2f}, 收益率: {h.get('realized_return', 0):+.2f}%")

    # 净值曲线
    daily_values = backtest.get('daily_values', [])
    if daily_values:
        typer.echo(f"\n净值走势（最近5天）:")
        typer.echo(f"  {'日期':<12} {'总市值':>15} {'日收益':>10}")
        typer.echo("  " + "-" * 40)
        for dv in daily_values[-5:]:
            ret_str = f"{dv.get('daily_return', 0):+.2f}%" if dv.get('daily_return') else "N/A"
            typer.echo(
                f"  {dv.get('date', '-'):<12} "
                f"{dv.get('total_value', 0):>15,.2f} "
                f"{ret_str:>10}"
            )

    typer.echo()


@app.command()
def update(
    backtest_id: Optional[str] = typer.Option(None, "--id", help="回测 ID，不指定则更新所有活跃组合"),
    date: Optional[str] = typer.Option(None, "--date", help="指定日期 (YYYY-MM-DD)，默认为今天"),
):
    """手动更新投资组合净值."""
    from quanttool.application.portfolio_backtest_service import PortfolioBacktestService
    from quanttool.infrastructure.stores.meta_db import MetaDB
    from datetime import date as dt_date

    service = PortfolioBacktestService()
    db = MetaDB()

    target_date = dt_date.today()
    if date:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()

    if backtest_id:
        # 更新指定组合
        typer.echo(f"更新组合 {backtest_id} 的净值...")
        service.update_portfolio_value(backtest_id, target_date)
        typer.echo("✓ 更新完成")
    else:
        # 更新所有活跃组合
        active_portfolios = db.get_active_portfolios()
        typer.echo(f"发现 {len(active_portfolios)} 个活跃组合")

        for portfolio in active_portfolios:
            pid = portfolio.get('id')
            typer.echo(f"  更新 {pid[:8]}...")
            service.update_portfolio_value(pid, target_date)

        typer.echo("✓ 所有组合已更新")


@app.command()
def close(
    backtest_id: str = typer.Argument(..., help="回测 ID"),
    date: Optional[str] = typer.Option(None, "--date", help="平仓日期 (YYYY-MM-DD)，默认为今天"),
):
    """平仓投资组合."""
    from quanttool.application.portfolio_backtest_service import PortfolioBacktestService
    from datetime import date as dt_date

    service = PortfolioBacktestService()

    exit_date = dt_date.today()
    if date:
        exit_date = datetime.strptime(date, "%Y-%m-%d").date()

    try:
        service.close_portfolio(backtest_id, exit_date)
        typer.echo(f"✓ 组合 {backtest_id[:8]} 已平仓")
    except Exception as e:
        typer.echo(f"✗ 平仓失败: {e}")


@app.command()
def auto_create(
    days: int = typer.Option(360, "--days", "-d", help="分析天数"),
    capital: float = typer.Option(500000, "--capital", "-c", help="初始资金"),
    top_n: int = typer.Option(5, "--top", "-n", help="选取前 N 只股票"),
):
    """立即执行 scan 并创建投资组合（一键操作）."""
    import asyncio
    from quanttool.infrastructure.scheduler.task_scheduler import DailyTaskScheduler

    scheduler = DailyTaskScheduler()

    async def run():
        typer.echo("执行每日 scan...")
        scan_id = await scheduler.run_daily_scan()

        if scan_id:
            typer.echo(f"✓ Scan 完成，ID: {scan_id}")
            typer.echo("创建投资组合...")
            from quanttool.application.portfolio_backtest_service import PortfolioBacktestService
            service = PortfolioBacktestService()
            backtest_id = service.create_portfolio_from_scan(scan_id, capital, top_n)
            typer.echo(f"✓ 组合已创建，ID: {backtest_id}")
        else:
            typer.echo("✗ Scan 失败")

    asyncio.run(run())
