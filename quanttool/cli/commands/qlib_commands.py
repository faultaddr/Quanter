"""
Qlib 模型回测命令

支持 23 种 Qlib 原生模型的回测:
- GBDT: lgb, xgboost, catboost, double_ensemble
- PyTorch 序列: lstm, gru, alstm, transformer, tcn, localformer
- PyTorch 高级: gats, sfm, tabnet, adarnn, add, hist, igmtf, krnn, tra, tcts, sandwich
"""

import typer
from typing import List, Optional
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import warnings
warnings.filterwarnings('ignore')

app = typer.Typer()
console = Console()


# 支持的模型类型
GBDT_MODELS = ['lgb', 'lightgbm', 'xgboost', 'xgb', 'catboost', 'double_ensemble']
PYTORCH_SEQUENCE_MODELS = ['lstm', 'gru', 'alstm', 'transformer', 'tcn', 'localformer']
PYTORCH_ADVANCED_MODELS = ['gats', 'sfm', 'tabnet', 'adarnn', 'add', 'hist', 'igmtf', 'krnn', 'tra', 'tcts', 'sandwich']
ALL_MODELS = GBDT_MODELS + PYTORCH_SEQUENCE_MODELS + PYTORCH_ADVANCED_MODELS


@app.command("list")
def list_models():
    """列出所有支持的 Qlib 模型"""
    table = Table(title="Qlib 支持的模型 (23种)")
    table.add_column("类别", style="cyan")
    table.add_column("模型", style="green")
    table.add_column("描述", style="white")

    model_info = {
        # GBDT
        'lgb': 'LightGBM 梯度提升模型',
        'lightgbm': 'LightGBM 梯度提升模型',
        'xgboost': 'XGBoost 梯度提升模型',
        'xgb': 'XGBoost 梯度提升模型',
        'catboost': 'CatBoost 梯度提升模型',
        'double_ensemble': 'Double Ensemble 模型',
        # PyTorch 序列
        'lstm': 'LSTM 长短期记忆网络',
        'gru': 'GRU 门控循环单元',
        'alstm': 'Attention LSTM',
        'transformer': 'Transformer 模型',
        'tcn': '时间卷积网络',
        'localformer': 'Local Transformer',
        # PyTorch 高级
        'gats': '图注意力网络',
        'sfm': '状态频域模型',
        'tabnet': 'TabNet 表格网络',
        'adarnn': '自适应 RNN',
        'add': 'ADD 模型',
        'hist': 'HIST 历史感知模型',
        'igmtf': 'IGMTF 模型',
        'krnn': 'KNN-RNN 混合模型',
        'tra': 'TRA 模型',
        'tcts': 'TCTS 模型',
        'sandwich': 'Sandwich 模型',
    }

    for model in GBDT_MODELS:
        table.add_row("GBDT", model, model_info.get(model, ''))
    for model in PYTORCH_SEQUENCE_MODELS:
        table.add_row("PyTorch序列", model, model_info.get(model, ''))
    for model in PYTORCH_ADVANCED_MODELS:
        table.add_row("PyTorch高级", model, model_info.get(model, ''))

    console.print(table)


@app.command("backtest")
def run_backtest(
    symbols: List[str] = typer.Option(..., "--symbol", "-s", help="股票代码 (可多个)"),
    model: str = typer.Option("lgb", "--model", "-m", help="模型类型"),
    start_date: str = typer.Option(None, "--start", "-sd", help="开始日期 (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, "--end", "-ed", help="结束日期 (YYYY-MM-DD)"),
    days: int = typer.Option(365, "--days", "-d", help="回测天数 (默认365天)"),
    initial_cash: float = typer.Option(100000.0, "--cash", "-c", help="初始资金"),
    horizon: int = typer.Option(5, "--horizon", "-h", help="预测周期 (天)"),
    buy_threshold: float = typer.Option(0.55, "--buy", "-b", help="买入阈值"),
    sell_threshold: float = typer.Option(0.45, "--sell", "-sell", help="卖出阈值"),
    # PyTorch 参数
    hidden_size: int = typer.Option(64, "--hidden", help="隐藏层大小 (PyTorch)"),
    num_layers: int = typer.Option(2, "--layers", help="层数 (PyTorch)"),
    epochs: int = typer.Option(100, "--epochs", help="训练轮数 (PyTorch)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细输出"),
):
    """
    使用 Qlib 模型进行回测

    示例:
        qlib backtest -s 000876 -s 600515 -m lgb -d 180
        qlib backtest -s 000876 -m transformer --hidden 128 --epochs 50
    """
    model = model.lower()

    if model not in ALL_MODELS:
        console.print(f"[red]错误: 不支持的模型类型 '{model}'[/red]")
        console.print(f"支持的模型: {ALL_MODELS}")
        raise typer.Exit(1)

    # 处理日期
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end_dt = datetime.now()

    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        start_dt = end_dt - timedelta(days=days)

    console.print(Panel.fit(
        f"[bold cyan]Qlib 模型回测[/bold cyan]\n"
        f"模型: [green]{model}[/green]\n"
        f"股票: [yellow]{', '.join(symbols)}[/yellow]\n"
        f"周期: {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}\n"
        f"初始资金: {initial_cash:,.0f}",
        title="回测配置"
    ))

    # 执行回测
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("正在获取数据并回测...", total=None)

        try:
            results = _run_qlib_backtest(
                symbols=symbols,
                model_type=model,
                start_date=start_dt,
                end_date=end_dt,
                initial_cash=initial_cash,
                horizon=horizon,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                hidden_size=hidden_size,
                num_layers=num_layers,
                epochs=epochs,
                verbose=verbose,
            )
            progress.remove_task(task)

            # 显示结果
            _display_results(results, model, symbols)

        except Exception as e:
            progress.remove_task(task)
            console.print(f"[red]回测失败: {e}[/red]")
            if verbose:
                import traceback
                traceback.print_exc()
            raise typer.Exit(1)


@app.command("compare")
def compare_models(
    symbols: List[str] = typer.Option(..., "--symbol", "-s", help="股票代码"),
    models: List[str] = typer.Option(None, "--model", "-m", help="要比较的模型 (默认全部GBDT)"),
    days: int = typer.Option(180, "--days", "-d", help="回测天数"),
    initial_cash: float = typer.Option(100000.0, "--cash", "-c", help="初始资金"),
):
    """
    比较不同模型的回测结果

    示例:
        qlib compare -s 000876 -m lgb -m xgboost -m catboost
        qlib compare -s 000876 -s 600515 -d 365
    """
    # 默认比较所有 GBDT 模型
    if not models:
        models = ['lgb', 'xgboost', 'catboost']

    # 过滤支持的模型
    models = [m.lower() for m in models if m.lower() in ALL_MODELS]

    if not models:
        console.print("[red]错误: 没有有效的模型[/red]")
        raise typer.Exit(1)

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)

    console.print(Panel.fit(
        f"[bold cyan]模型对比回测[/bold cyan]\n"
        f"模型: [green]{', '.join(models)}[/green]\n"
        f"股票: [yellow]{', '.join(symbols)}[/yellow]\n"
        f"周期: {days} 天",
        title="对比配置"
    ))

    all_results = {}

    for model in models:
        console.print(f"\n[cyan]正在测试模型: {model}...[/cyan]")
        try:
            results = _run_qlib_backtest(
                symbols=symbols,
                model_type=model,
                start_date=start_dt,
                end_date=end_dt,
                initial_cash=initial_cash,
            )
            all_results[model] = results
        except Exception as e:
            console.print(f"[red]模型 {model} 回测失败: {e}[/red]")
            all_results[model] = None

    # 显示对比结果
    _display_comparison(all_results, models)


def _run_qlib_backtest(
    symbols: List[str],
    model_type: str,
    start_date: datetime,
    end_date: datetime,
    initial_cash: float = 100000.0,
    horizon: int = 5,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
    hidden_size: int = 64,
    num_layers: int = 2,
    epochs: int = 100,
    verbose: bool = False,
) -> dict:
    """执行 Qlib 模型回测"""
    import pandas as pd
    import numpy as np
    from quanttool.strategies.qlib_strategy import QlibStrategy

    # 尝试使用多种数据源
    provider = None

    # 1. 尝试 baostock (免费)
    try:
        from quanttool.infrastructure.data_providers.real_data_provider import RealAShareDataProvider
        provider = RealAShareDataProvider(primary_source="baostock", use_fallback=True)
    except Exception:
        pass

    # 2. 尝试 TuShare
    if provider is None:
        try:
            from quanttool.infrastructure.data_providers.tushare_provider import TuShareProvider
            provider = TuShareProvider()
        except Exception:
            pass

    if provider is None:
        raise ValueError("没有可用的数据源，请安装 baostock 或设置 TUSHARE_TOKEN")

    # 获取数据
    all_data = {}
    for symbol in symbols:
        try:
            # 尝试多种方法获取数据
            if hasattr(provider, 'get_stock_daily'):
                df = provider.get_stock_daily(
                    symbol=symbol,
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                )
            elif hasattr(provider, 'get_daily_bars'):
                df = provider.get_daily_bars(
                    symbol=symbol,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                )
            else:
                # 直接使用 baostock
                import baostock as bs
                bs.login()
                rs = bs.query_history_k_data_plus(
                    symbol,
                    "date,code,open,high,low,close,volume",
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    frequency="d",
                    adjustflag="3",
                )
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                df = pd.DataFrame(data_list, columns=rs.fields)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                bs.logout()

            if df is not None and len(df) >= 120:
                # 确保列名正确
                if 'vol' in df.columns and 'volume' not in df.columns:
                    df['volume'] = df['vol']
                all_data[symbol] = df
        except Exception as e:
            if verbose:
                console.print(f"[yellow]获取 {symbol} 数据失败: {e}[/yellow]")

    if not all_data:
        raise ValueError("没有获取到有效数据")

    # 创建策略
    strategy = QlibStrategy(
        model_type=model_type,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        hidden_size=hidden_size,
        num_layers=num_layers,
        epochs=epochs,
    )

    # 合并所有股票数据训练模型
    all_features = []
    all_labels = []

    for symbol, df in all_data.items():
        try:
            features = strategy.feature_engineer.generate_features(df)
            close = df['close']
            labels = (close.shift(-horizon) / close - 1 > 0).astype(int)

            valid_idx = features.dropna().index.intersection(labels.dropna().index)
            all_features.append(features.loc[valid_idx])
            all_labels.append(labels.loc[valid_idx])
        except Exception as e:
            if verbose:
                console.print(f"[yellow]{symbol} 特征生成失败: {e}[/yellow]")

    if all_features:
        X = pd.concat(all_features)
        y = pd.concat(all_labels)
        strategy.model.fit(X, y)

    # 回测每个股票
    results = {
        'symbols': symbols,
        'model': model_type,
        'initial_cash': initial_cash,
        'stocks': {},
        'total_return': 0,
        'total_trades': 0,
        'winning_trades': 0,
    }

    total_return = 0
    total_trades = 0
    winning_trades = 0

    for symbol, df in all_data.items():
        try:
            # 计算信号
            signals = strategy.calculate_signals(df)

            # 简单回测
            cash = initial_cash / len(all_data)
            position = 0
            shares = 0
            trades = 0
            wins = 0

            buy_price = 0

            for i in range(len(signals)):
                signal = signals['signal'].iloc[i]
                close = df['close'].iloc[i]

                if signal == 'buy' and position == 0:
                    shares = cash / close
                    position = 1
                    buy_price = close
                    trades += 1
                elif signal == 'sell' and position == 1:
                    cash = shares * close
                    position = 0
                    trades += 1
                    if close > buy_price:
                        wins += 1
                    buy_price = 0

            # 最后平仓
            if position == 1:
                cash = shares * df['close'].iloc[-1]
                trades += 1
                if df['close'].iloc[-1] > buy_price:
                    wins += 1

            stock_return = (cash - initial_cash / len(all_data)) / (initial_cash / len(all_data))
            total_return += stock_return
            total_trades += trades
            winning_trades += wins

            results['stocks'][symbol] = {
                'final_cash': cash,
                'return': stock_return,
                'trades': trades,
                'wins': wins,
            }

        except Exception as e:
            if verbose:
                console.print(f"[yellow]{symbol} 回测失败: {e}[/yellow]")

    n_stocks = len(all_data)
    results['total_return'] = total_return / n_stocks if n_stocks > 0 else 0
    results['total_trades'] = total_trades
    results['winning_trades'] = winning_trades
    results['win_rate'] = winning_trades / total_trades if total_trades > 0 else 0

    return results


def _display_results(results: dict, model: str, symbols: List[str]):
    """显示回测结果"""
    # 总体结果
    total_table = Table(title="回测结果汇总")
    total_table.add_column("指标", style="cyan")
    total_table.add_column("值", style="green")

    total_table.add_row("模型", model)
    total_table.add_row("股票数", str(len(results['stocks'])))
    total_table.add_row("总收益率", f"{results['total_return']*100:.2f}%")
    total_table.add_row("总交易次数", str(results['total_trades']))
    total_table.add_row("盈利次数", str(results['winning_trades']))
    total_table.add_row("胜率", f"{results['win_rate']*100:.1f}%")

    console.print(total_table)

    # 各股票结果
    if results['stocks']:
        stock_table = Table(title="各股票回测详情")
        stock_table.add_column("股票", style="cyan")
        stock_table.add_column("收益率", style="green")
        stock_table.add_column("交易次数", style="yellow")
        stock_table.add_column("盈利次数", style="magenta")

        for symbol, data in results['stocks'].items():
            return_color = "green" if data['return'] > 0 else "red"
            stock_table.add_row(
                symbol,
                f"[{return_color}]{data['return']*100:.2f}%[/{return_color}]",
                str(data['trades']),
                str(data['wins']),
            )

        console.print(stock_table)


def _display_comparison(all_results: dict, models: List[str]):
    """显示模型对比结果"""
    table = Table(title="模型对比结果")
    table.add_column("模型", style="cyan")
    table.add_column("总收益率", style="green")
    table.add_column("交易次数", style="yellow")
    table.add_column("胜率", style="magenta")

    best_model = None
    best_return = -float('inf')

    for model in models:
        results = all_results.get(model)
        if results:
            return_color = "green" if results['total_return'] > 0 else "red"
            table.add_row(
                model,
                f"[{return_color}]{results['total_return']*100:.2f}%[/{return_color}]",
                str(results['total_trades']),
                f"{results['win_rate']*100:.1f}%",
            )
            if results['total_return'] > best_return:
                best_return = results['total_return']
                best_model = model
        else:
            table.add_row(model, "[red]失败[/red]", "-", "-")

    console.print(table)

    if best_model:
        console.print(f"\n[bold green]最佳模型: {best_model} ({best_return*100:.2f}%)[/bold green]")


if __name__ == "__main__":
    app()