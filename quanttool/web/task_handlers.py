"""
任务处理函数

所有耗时操作的处理函数，由 TaskManager 调度执行。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import traceback

from quanttool.web.task_manager import TaskContext, get_task_manager
from quanttool.core.logging import get_logger

logger = get_logger(__name__)


def qlib_train_handler(ctx: TaskContext, **params) -> Dict[str, Any]:
    """
    Qlib 模型训练任务

    Args:
        ctx: 任务上下文
        params: 训练参数

    Returns:
        训练结果
    """
    from quanttool.strategies.qlib import create_model
    from quanttool.factors.stock_analyzer import StockAnalyzer
    from quanttool.cli.commands.analysis_commands import get_csi300_constituents
    import os
    import uuid

    ctx.update_progress(0, 100, "初始化训练环境...", "init")
    ctx.log("开始 Qlib 模型训练")

    try:
        # 解析参数
        model_type = params.get('model_type', 'lgb')
        train_start = params.get('train_start', '2019-01-01')
        train_end = params.get('train_end', '2023-12-31')
        valid_start = params.get('valid_start', '2024-01-01')
        valid_end = params.get('valid_end', '2024-12-31')
        test_start = params.get('test_start', '2025-01-01')
        test_end = params.get('test_end', '2026-03-17')
        use_rich_features = params.get('use_rich_features', True)
        feature_set = params.get('feature_set', 'Alpha158')
        max_train_stocks = params.get('max_train_stocks', 0)

        # 模型参数
        n_estimators = params.get('n_estimators', 200)
        max_depth = params.get('max_depth', 6)
        learning_rate = params.get('learning_rate', 0.01)

        # 获取沪深300成分股
        ctx.update_progress(5, 100, "获取沪深300成分股列表...", "init")
        csi300_stocks = get_csi300_constituents()
        train_symbols = [s['code'] if isinstance(s, dict) else s for s in csi300_stocks]

        if max_train_stocks > 0:
            train_symbols = train_symbols[:max_train_stocks]

        ctx.log(f"训练股票数量: {len(train_symbols)}")

        # 初始化
        analyzer = StockAnalyzer()
        train_data = []
        valid_data = []
        test_data = []

        train_start_dt = datetime.fromisoformat(train_start)
        train_end_dt = datetime.fromisoformat(train_end)
        valid_start_dt = datetime.fromisoformat(valid_start)
        valid_end_dt = datetime.fromisoformat(valid_end)
        test_start_dt = datetime.fromisoformat(test_start)
        test_end_dt = datetime.fromisoformat(test_end)

        # 特征工程器
        if use_rich_features:
            from quanttool.strategies.qlib_strategy import QlibFeatureEngineer
            feature_engineer = QlibFeatureEngineer(feature_set=feature_set)
            ctx.log(f"使用 {feature_set} 特征集")

        # 数据收集（并行化）
        total_stocks = len(train_symbols)
        success_count = 0
        first_symbol_features = None

        # 并行获取数据的函数
        def fetch_single_stock(symbol: str) -> Optional[Dict]:
            """获取单只股票的数据和特征"""
            try:
                df = analyzer.get_stock_data(symbol, 2500)
                if df.empty or len(df) < 120:
                    return None

                # 确定日期列
                date_column = None
                for col in ['trade_date', 'timestamp']:
                    if col in df.columns:
                        date_column = col
                        break

                if not date_column:
                    return None

                df['_date'] = pd.to_datetime(df[date_column])

                # 特征工程
                if use_rich_features:
                    try:
                        feature_df = feature_engineer.generate_features(df)
                        available_features = list(feature_df.columns)
                        df = pd.concat([df, feature_df], axis=1)
                    except Exception:
                        return None
                else:
                    df = analyzer.calculate_technical_indicators(df)
                    available_features = ['close', 'volume', 'ma_5', 'ma_10', 'ma_20']

                if not available_features:
                    return None

                # 计算标签：未来5天收益率，转换为二分类标签
                df['return_5d'] = df['close'].pct_change(5).shift(-5)
                # 二分类标签：收益 > 0 为 1，否则为 0
                df['label'] = (df['return_5d'] > 0).astype(int)

                return {
                    'symbol': symbol,
                    'df': df,
                    'features': available_features
                }
            except Exception:
                return None

        # 使用线程池并行获取数据
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        stock_results = {}
        completed_count = 0
        progress_lock = threading.Lock()

        ctx.update_progress(5, 100, f"并行获取 {total_stocks} 只股票数据...", "data_collection")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_single_stock, symbol): symbol for symbol in train_symbols}

            for future in as_completed(futures):
                if ctx.is_cancelled():
                    ctx.log("任务被取消")
                    executor.shutdown(wait=False, cancel_futures=True)
                    return {"status": "cancelled"}

                symbol = futures[future]
                with progress_lock:
                    completed_count += 1
                    progress = 5 + int((completed_count / total_stocks) * 55)
                    ctx.update_progress(progress, 100, f"获取数据: {completed_count}/{total_stocks}", "data_collection")

                try:
                    result = future.result()
                    if result:
                        stock_results[symbol] = result
                except Exception as e:
                    ctx.log(f"数据获取失败 {symbol}: {e}")

        ctx.log(f"并行获取完成: {len(stock_results)}/{total_stocks} 只股票成功")

        # 处理获取的数据
        for symbol, result in stock_results.items():
            df = result['df']
            available_features = result['features']

            # 统一特征
            if first_symbol_features is None:
                first_symbol_features = available_features
            else:
                available_features = [f for f in first_symbol_features if f in df.columns]
                if len(available_features) != len(first_symbol_features):
                    continue

            # 划分数据（使用二分类标签）
            for idx, row in df.iterrows():
                date_val = row['_date']
                if pd.isna(date_val):
                    continue

                feature_vals = [row.get(f) for f in available_features]
                label_val = row.get('label')  # 使用二分类标签

                if any(pd.isna(v) for v in feature_vals) or pd.isna(label_val):
                    continue

                row_data = {
                    'features': feature_vals,
                    'label': label_val,
                    'symbol': symbol,
                    'date': date_val,
                    'return_5d': row.get('return_5d', 0)  # 保留实际收益率用于评估
                }

                if train_start_dt <= date_val <= train_end_dt:
                    train_data.append(row_data)
                elif valid_start_dt <= date_val <= valid_end_dt:
                    valid_data.append(row_data)
                elif test_start_dt <= date_val <= test_end_dt:
                    test_data.append(row_data)

            success_count += 1

        ctx.log(f"数据收集完成: {success_count} 只股票, 训练 {len(train_data)}, 验证 {len(valid_data)}, 测试 {len(test_data)}")

        if not train_data:
            return {"error": "无法获取足够的训练数据"}

        # 准备训练数据
        ctx.update_progress(65, 100, "准备训练数据...", "preparation")
        feature_cols = first_symbol_features
        X_train = np.array([d['features'] for d in train_data])
        y_train = np.array([d['label'] for d in train_data])

        ctx.log(f"特征数量: {len(feature_cols)}")
        ctx.log(f"训练样本: {len(train_data)}")

        # 创建模型
        ctx.update_progress(70, 100, f"创建 {model_type.upper()} 模型...", "model_creation")
        config_kwargs = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
        }

        model = create_model(model_type, **config_kwargs)

        # 训练
        ctx.update_progress(75, 100, "训练模型中...", "training")
        X_train_df = pd.DataFrame(X_train, columns=feature_cols)
        y_train_series = pd.Series(y_train)
        model.fit(X_train_df, y_train_series)
        model.feature_names_ = feature_cols

        ctx.log("模型训练完成")

        # 保存模型
        ctx.update_progress(85, 100, "保存模型...", "saving")
        model_dir = "models/qlib"
        os.makedirs(model_dir, exist_ok=True)
        model_id = str(uuid.uuid4())[:8]
        model_path = f"{model_dir}/{model_type}_{model_id}.pkl"
        model.save(model_path)

        ctx.log(f"模型已保存: {model_path}")

        # 评估
        ctx.update_progress(90, 100, "评估模型...", "evaluation")
        train_pred = model.predict(X_train_df)
        train_mse = np.mean((train_pred - y_train) ** 2)
        train_mae = np.mean(np.abs(train_pred - y_train))

        # 正确的 IC 计算：按日期分组计算横截面 IC，然后取平均
        def calculate_ic(predictions: np.ndarray, data_list: List[Dict]) -> float:
            """
            计算信息系数 (Information Coefficient)

            正确方法：
            1. 按日期分组
            2. 计算每个日期的横截面 IC (Spearman 相关系数)
            3. 使用预测概率与实际收益率的相关性
            4. 取所有日期 IC 的平均值
            """
            from scipy.stats import spearmanr

            # 构建日期 -> (pred, return) 映射
            date_data = {}
            for i, d in enumerate(data_list):
                date_val = d['date']
                if date_val not in date_data:
                    date_data[date_val] = {'pred': [], 'return': []}
                date_data[date_val]['pred'].append(predictions[i])
                # 使用实际收益率而不是二分类标签
                date_data[date_val]['return'].append(d.get('return_5d', 0))

            # 计算每个日期的横截面 IC
            ics = []
            for date_val, data in date_data.items():
                preds = np.array(data['pred'])
                returns = np.array(data['return'])

                # 需要至少5只股票才能计算相关性
                if len(preds) >= 5:
                    # 检查是否有足够的方差
                    if np.std(preds) > 1e-10 and np.std(returns) > 1e-10:
                        try:
                            ic, _ = spearmanr(preds, returns)
                            if not np.isnan(ic):
                                ics.append(ic)
                        except:
                            pass

            return np.mean(ics) if ics else 0.0

        train_ic = calculate_ic(train_pred, train_data)

        # 计算准确率（二分类）
        train_acc = np.mean((train_pred > 0.5) == y_train)

        metrics = {
            "train": {
                "samples": len(train_data),
                "accuracy": round(float(train_acc), 4),
                "ic": round(float(train_ic), 4) if not np.isnan(train_ic) else 0,
            }
        }

        # 验证集评估
        if valid_data:
            X_valid = np.array([d['features'] for d in valid_data])
            y_valid = np.array([d['label'] for d in valid_data])
            X_valid_df = pd.DataFrame(X_valid, columns=feature_cols)
            valid_pred = model.predict(X_valid_df)
            valid_acc = np.mean((valid_pred > 0.5) == y_valid)
            valid_ic = calculate_ic(valid_pred, valid_data)

            metrics["valid"] = {
                "samples": len(valid_data),
                "accuracy": round(float(valid_acc), 4),
                "ic": round(float(valid_ic), 4) if not np.isnan(valid_ic) else 0,
            }

        # 测试集评估
        if test_data:
            X_test = np.array([d['features'] for d in test_data])
            y_test = np.array([d['label'] for d in test_data])
            X_test_df = pd.DataFrame(X_test, columns=feature_cols)
            test_pred = model.predict(X_test_df)
            test_acc = np.mean((test_pred > 0.5) == y_test)
            test_ic = calculate_ic(test_pred, test_data)

            metrics["test"] = {
                "samples": len(test_data),
                "accuracy": round(float(test_acc), 4),
                "ic": round(float(test_ic), 4) if not np.isnan(test_ic) else 0,
            }

        ctx.update_progress(100, 100, "训练完成", "completed")

        return {
            "model_id": model_id,
            "model_type": model_type,
            "model_path": model_path,
            "feature_count": len(feature_cols),
            "feature_set": feature_set,
            "train_stocks": success_count,
            "train_samples": len(train_data),
            "train_symbols_count": success_count,
            "data_split": {
                "train": {
                    "period": f"{train_start} ~ {train_end}",
                    "samples": len(train_data),
                },
                "valid": {
                    "period": f"{valid_start} ~ {valid_end}",
                    "samples": len(valid_data),
                },
                "test": {
                    "period": f"{test_start} ~ {test_end}",
                    "samples": len(test_data),
                },
            },
            "metrics": metrics,
        }

    except Exception as e:
        ctx.log(f"训练失败: {e}")
        ctx.log(traceback.format_exc())
        raise


def qlib_predict_handler(ctx: TaskContext, **params) -> Dict[str, Any]:
    """
    Qlib 模型预测任务（含回测）

    Args:
        ctx: 任务上下文
        params: 预测参数

    Returns:
        预测结果（含回测数据）
    """
    import joblib
    from quanttool.factors.stock_analyzer import StockAnalyzer
    from quanttool.strategies.adaptive_threshold import IndexMarketDetector
    from pathlib import Path

    ctx.update_progress(0, 100, "初始化预测环境...", "init")
    ctx.log("开始 Qlib 模型预测")

    try:
        # 解析参数
        model_type = params.get('model_type', 'lgb')
        model_path = params.get('model_path', '')
        symbols = params.get('symbols', [])
        use_rich_features = params.get('use_rich_features', True)
        feature_set = params.get('feature_set', 'Alpha158')
        predict_start_date = params.get('predict_start_date', '2025-01-01')
        predict_end_date = params.get('predict_end_date', '2026-03-17')

        # 回测参数
        initial_cash = params.get('initial_cash', 100000)
        commission_rate = params.get('commission_rate', 0.0003)
        slippage_rate = params.get('slippage_rate', 0.0001)

        # 查找模型
        if not model_path:
            model_dir = Path("models/qlib")
            if model_dir.exists():
                pattern = f"{model_type}_*.pkl"
                model_files = list(model_dir.glob(pattern))

                if not model_files:
                    all_models = list(model_dir.glob("*.pkl"))
                    if all_models:
                        model_files = sorted(all_models, key=lambda x: x.stat().st_mtime, reverse=True)[:1]
                else:
                    model_files = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[:1]

                if model_files:
                    model_path = str(model_files[0])

        if not model_path:
            return {"error": "未找到已保存的模型"}

        ctx.log(f"使用模型: {model_path}")

        # 加载模型
        ctx.update_progress(10, 100, "加载模型...", "loading")
        saved_data = joblib.load(model_path)
        model = saved_data.get('model')
        feature_names = saved_data.get('feature_names', [])

        if model is None:
            return {"error": "模型文件无效"}

        # 初始化
        analyzer = StockAnalyzer()

        if use_rich_features:
            from quanttool.strategies.qlib_strategy import QlibFeatureEngineer
            feature_engineer = QlibFeatureEngineer(feature_set=feature_set)

        predictions = {}
        total_symbols = len(symbols)

        # 解析预测日期范围
        predict_start = datetime.fromisoformat(predict_start_date)
        predict_end = datetime.fromisoformat(predict_end_date)

        for i, symbol in enumerate(symbols):
            if ctx.is_cancelled():
                ctx.log("任务被取消")
                return {"status": "cancelled"}

            progress = 10 + int((i / total_symbols) * 85)
            ctx.update_progress(progress, 100, f"预测: {symbol} ({i+1}/{total_symbols})", "predicting")

            try:
                df = analyzer.get_stock_data(symbol, 500)
                if df.empty or len(df) < 120:
                    ctx.log(f"数据不足: {symbol}")
                    continue

                # 确定日期列
                date_column = None
                for col in ['trade_date', 'timestamp']:
                    if col in df.columns:
                        date_column = col
                        break

                if not date_column:
                    continue

                df['_date'] = pd.to_datetime(df[date_column])

                # 特征工程
                if use_rich_features:
                    try:
                        feature_df = feature_engineer.generate_features(df)
                        df = pd.concat([df, feature_df], axis=1)
                    except Exception as e:
                        ctx.log(f"特征工程失败 {symbol}: {e}")
                        continue

                # 使用模型期望的特征
                available_features = [f for f in feature_names if f in df.columns]
                if not available_features:
                    ctx.log(f"特征不匹配: {symbol}")
                    continue

                # ====== 回测逻辑 ======
                cash = initial_cash
                position = 0
                trades = []
                total_commission = 0.0
                total_slippage = 0.0
                buy_date = None
                buy_price = 0.0
                win_trades = 0
                total_sell_trades = 0  # 记录总卖出次数（用于计算胜率）

                inner_model = model.model if hasattr(model, 'model') else model

                for j in range(len(df) - 5):
                    row = df.iloc[j]

                    # 获取交易日期
                    trade_dt = row['_date']
                    if pd.isna(trade_dt):
                        continue

                    # 检查是否在回测日期范围内
                    if trade_dt < predict_start or trade_dt > predict_end:
                        continue

                    # 预测
                    X = df[available_features].iloc[j:j+1].values
                    try:
                        pred = inner_model.predict(X)[0]
                    except:
                        try:
                            pred = float(inner_model.predict(X.reshape(1, -1))[0])
                        except:
                            continue

                    pred_value = float(pred)

                    # 生成信号 (回归模型预测收益率，阈值需要适配)
                    signal = "hold"
                    if pred_value > 0.005:  # 预测上涨 > 0.5%
                        signal = "buy"
                    elif pred_value < -0.005:  # 预测下跌 > 0.5%
                        signal = "sell"

                    close_price = float(row['close'])

                    # 执行交易
                    if signal == "buy" and position == 0 and cash > 0:
                        slippage = close_price * slippage_rate
                        buy_price_actual = close_price + slippage
                        shares = int(cash / buy_price_actual / 100) * 100

                        if shares > 0:
                            commission = max(shares * buy_price_actual * commission_rate, 5)
                            total_cost = shares * buy_price_actual + commission

                            if total_cost <= cash:
                                position = shares
                                cash -= total_cost
                                buy_date = trade_dt
                                buy_price = buy_price_actual
                                total_commission += commission
                                total_slippage += shares * slippage
                                trades.append({
                                    "type": "buy",
                                    "date": str(trade_dt)[:10],
                                    "price": round(buy_price_actual, 2),
                                    "shares": shares,
                                    "commission": round(commission, 2),
                                })

                    elif signal == "sell" and position > 0:
                        if buy_date is None or trade_dt <= buy_date:
                            continue

                        slippage = close_price * slippage_rate
                        sell_price_actual = close_price - slippage
                        sell_amount = position * sell_price_actual
                        commission = max(sell_amount * commission_rate, 5)

                        profit = position * (sell_price_actual - buy_price) - commission
                        cash += sell_amount - commission
                        total_commission += commission
                        total_slippage += position * slippage

                        trades.append({
                            "type": "sell",
                            "date": str(trade_dt)[:10],
                            "price": round(sell_price_actual, 2),
                            "shares": position,
                            "commission": round(commission, 2),
                            "profit": round(profit, 2)
                        })

                        total_sell_trades += 1
                        if profit > 0:
                            win_trades += 1

                        position = 0
                        buy_date = None

                # 计算最终市值
                final_price = float(df['close'].iloc[-1])
                final_capital = cash + position * final_price
                total_return = (final_capital - initial_cash) / initial_cash

                # 计算年化收益
                days = (predict_end - predict_start).days
                annual_return = total_return * 252 / max(days, 1) if days > 0 else 0

                # 获取最新预测
                X_latest = df[available_features].iloc[-1:].values
                try:
                    pred_latest = inner_model.predict(X_latest)[0]
                except:
                    pred_latest = 0.5

                # 数据周期
                data_start = str(df['_date'].min())[:10]
                data_end = str(df['_date'].max())[:10]

                predictions[symbol] = {
                    "prediction": round(float(pred_latest), 4),
                    "signal": "buy" if float(pred_latest) > 0.005 else ("sell" if float(pred_latest) < -0.005 else "hold"),
                    "latest_price": round(float(df['close'].iloc[-1]), 2),
                    "data_period": {
                        "start_date": data_start,
                        "end_date": data_end,
                    },
                    "backtest": {
                        "initial_cash": initial_cash,
                        "final_capital": round(final_capital, 2),
                        "total_return": round(total_return * 100, 2),
                        "annual_return": round(annual_return * 100, 2),
                        "max_drawdown": 0.0,  # 简化计算
                        "sharpe_ratio": round((annual_return - 0.02) / 0.20, 2) if annual_return != 0 else 0,
                        "total_trades": len(trades),
                        "win_rate": round(win_trades / total_sell_trades * 100, 1) if total_sell_trades > 0 else 0,
                        "total_commission": round(total_commission, 2),
                        "total_slippage": round(total_slippage, 2),
                        "trades": trades[-10:],
                    }
                }

            except Exception as e:
                ctx.log(f"预测失败 {symbol}: {e}")
                continue

        ctx.update_progress(100, 100, "预测完成", "completed")
        ctx.log(f"预测完成: {len(predictions)}/{total_symbols} 只股票")

        # 计算汇总
        total_final_capital = sum(p["backtest"]["final_capital"] for p in predictions.values())

        # 注意：每只股票的 trades 列表只保留最后10条，所以这里从各股票的 win_rate 反推
        # 更好的方式是在返回时保存完整统计，但这里用加权平均计算
        total_trades_count = sum(p["backtest"]["total_trades"] for p in predictions.values())

        # 计算所有股票的平均胜率（简单平均）
        avg_win_rate = round(
            sum(p["backtest"]["win_rate"] for p in predictions.values()) / len(predictions),
            1
        ) if predictions else 0

        # 计算策略总收益
        strategy_return = round((total_final_capital - initial_cash * len(predictions)) / (initial_cash * len(predictions)) * 100, 2) if predictions else 0

        # 计算沪深300基准收益
        benchmark_return = 0.0
        try:
            detector = IndexMarketDetector()
            index_df = detector.get_index_data('000300.SH', days=500)

            if not index_df.empty:
                # 确定日期列
                date_col = None
                for col in ['trade_date', 'timestamp']:
                    if col in index_df.columns:
                        date_col = col
                        break

                if date_col:
                    index_df['_date'] = pd.to_datetime(index_df[date_col])

                    # 筛选预测日期范围内的指数数据
                    mask = (index_df['_date'] >= predict_start) & (index_df['_date'] <= predict_end)
                    period_data = index_df[mask]

                    if len(period_data) >= 2:
                        # 按日期排序
                        period_data = period_data.sort_values('_date')
                        start_price = float(period_data['close'].iloc[0])
                        end_price = float(period_data['close'].iloc[-1])
                        benchmark_return = round((end_price - start_price) / start_price * 100, 2)
                        ctx.log(f"沪深300基准收益: {benchmark_return}% (起:{start_price:.2f}, 终:{end_price:.2f})")
        except Exception as e:
            ctx.log(f"计算基准收益失败: {e}")

        # 计算相对收益（超额收益）
        relative_return = round(strategy_return - benchmark_return, 2)

        # 计算总交易成本
        total_commission = sum(p["backtest"].get("total_commission", 0) for p in predictions.values())
        total_slippage = sum(p["backtest"].get("total_slippage", 0) for p in predictions.values())
        total_cost = round(total_commission + total_slippage, 2)

        return {
            "model_path": model_path,
            "model_type": model_type,
            "feature_count": len(feature_names),
            "predictions": predictions,
            "total_stocks": total_symbols,
            "predicted_stocks": len(predictions),
            "predict_period": {
                "start_date": predict_start_date,
                "end_date": predict_end_date,
            },
            "backtest_params": {
                "initial_cash": initial_cash,
                "commission_rate": commission_rate,
                "slippage_rate": slippage_rate,
            },
            "benchmark_return": benchmark_return,
            "summary": {
                "total_return_pct": strategy_return,
                "benchmark_return_pct": benchmark_return,
                "relative_return_pct": relative_return,
                "total_trades": total_trades_count,
                "win_rate": avg_win_rate,
                "total_cost": total_cost,
                "total_commission": round(total_commission, 2),
                "total_slippage": round(total_slippage, 2),
            }
        }

    except Exception as e:
        ctx.log(f"预测失败: {e}")
        ctx.log(traceback.format_exc())
        raise


def stock_analyze_handler(ctx: TaskContext, **params) -> Dict[str, Any]:
    """
    股票分析任务

    Args:
        ctx: 任务上下文
        params: 分析参数

    Returns:
        分析结果
    """
    from quanttool.factors.stock_analyzer import StockAnalyzer

    symbol = params.get('symbol', '')
    days = params.get('days', 360)

    ctx.update_progress(0, 100, f"分析 {symbol}...", "init")
    ctx.log(f"开始分析股票: {symbol}")

    try:
        analyzer = StockAnalyzer()

        ctx.update_progress(20, 100, "获取数据...", "data")
        df = analyzer.get_stock_data(symbol, days)

        if df.empty:
            return {"error": f"无法获取 {symbol} 的数据"}

        ctx.log(f"获取到 {len(df)} 条数据")

        ctx.update_progress(40, 100, "计算技术指标...", "indicators")
        df = analyzer.calculate_technical_indicators(df)

        ctx.update_progress(60, 100, "运行策略分析...", "strategies")
        strategies_result = analyzer.run_trading_strategies(df, symbol)

        ctx.update_progress(80, 100, "生成报告...", "report")
        report = analyzer.generate_report(df, symbol)

        ctx.update_progress(100, 100, "分析完成", "completed")

        return {
            "symbol": symbol,
            "days": days,
            "data_rows": len(df),
            "latest_price": float(df['close'].iloc[-1]) if not df.empty else None,
            "strategies": strategies_result,
            "report": report,
        }

    except Exception as e:
        ctx.log(f"分析失败: {e}")
        ctx.log(traceback.format_exc())
        raise


def market_scan_handler(ctx: TaskContext, **params) -> Dict[str, Any]:
    """
    市场扫描任务

    Args:
        ctx: 任务上下文
        params: 扫描参数

    Returns:
        扫描结果
    """
    from quanttool.factors.stock_analyzer import StockAnalyzer
    from quanttool.cli.commands.analysis_commands import get_csi300_constituents

    market = params.get('market', 'csi300')
    days = params.get('days', 360)
    top_n = params.get('top_n', 10)

    ctx.update_progress(0, 100, f"扫描 {market} 市场...", "init")
    ctx.log(f"开始市场扫描: {market}")

    try:
        # 获取股票列表
        if market == 'csi300':
            stocks = get_csi300_constituents()
        else:
            stocks = []  # 可扩展其他市场

        symbols = [s['code'] if isinstance(s, dict) else s for s in stocks]
        ctx.log(f"扫描 {len(symbols)} 只股票")

        analyzer = StockAnalyzer()
        results = []
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            if ctx.is_cancelled():
                return {"status": "cancelled"}

            progress = int((i / total) * 100)
            ctx.update_progress(progress, 100, f"扫描: {symbol} ({i+1}/{total})", "scanning")

            try:
                df = analyzer.get_stock_data(symbol, days)
                if df.empty or len(df) < 60:
                    continue

                df = analyzer.calculate_technical_indicators(df)

                # 简单评分
                score = 0
                if 'rsi_12' in df.columns:
                    rsi = df['rsi_12'].iloc[-1]
                    if rsi < 30:
                        score += 20
                    elif rsi < 50:
                        score += 10

                if 'macd_dif' in df.columns and 'macd_dea' in df.columns:
                    if df['macd_dif'].iloc[-1] > df['macd_dea'].iloc[-1]:
                        score += 15

                results.append({
                    "symbol": symbol,
                    "score": score,
                    "latest_price": float(df['close'].iloc[-1]),
                })

            except Exception:
                continue

        # 排序取前 N
        results.sort(key=lambda x: x['score'], reverse=True)
        top_results = results[:top_n]

        ctx.update_progress(100, 100, "扫描完成", "completed")
        ctx.log(f"扫描完成，找到 {len(top_results)} 只候选股票")

        return {
            "market": market,
            "total_scanned": total,
            "valid_stocks": len(results),
            "top_stocks": top_results,
        }

    except Exception as e:
        ctx.log(f"扫描失败: {e}")
        ctx.log(traceback.format_exc())
        raise


# 任务名称到处理函数的映射
TASK_HANDLERS = {
    "qlib_train": qlib_train_handler,
    "qlib_predict": qlib_predict_handler,
    "stock_analyze": stock_analyze_handler,
    "market_scan": market_scan_handler,
}


def create_task(name: str, params: Dict[str, Any] = None) -> str:
    """
    创建任务

    Args:
        name: 任务名称
        params: 任务参数

    Returns:
        任务ID
    """
    handler = TASK_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"未知的任务类型: {name}")

    manager = get_task_manager()
    return manager.create_task(name, handler, params or {})
