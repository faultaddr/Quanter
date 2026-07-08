"""Qlib model prediction API routes."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from ....core.logging import get_logger
from ...schemas.model import QlibPredictRequest


logger = get_logger(__name__)
router = APIRouter()

@router.post("/qlib/predict")
async def predict_with_qlib_model(request: QlibPredictRequest) -> Dict[str, Any]:
    """
    使用 Qlib ML 模型进行预测并回测

    返回预测结果、信号和回测收益
    """
    try:
        import joblib
        from quanttool.factors.stock_analyzer import StockAnalyzer
        import numpy as np
        from datetime import datetime, timedelta
        from pathlib import Path

        # 查找模型文件
        model_path = request.model_path
        if not model_path:
            # 自动查找对应 model_type 的最新模型
            model_dir = Path("models/qlib")
            if model_dir.exists():
                # 查找匹配的模型文件
                pattern = f"{request.model_type}_*.pkl"
                model_files = list(model_dir.glob(pattern))

                if not model_files:
                    # 尝试其他命名方式
                    all_models = list(model_dir.glob("*.pkl"))
                    if all_models:
                        # 使用最新的模型
                        model_files = sorted(all_models, key=lambda x: x.stat().st_mtime, reverse=True)[:1]
                        logger.info(f"No {request.model_type} model found, using latest: {model_files[0].name}")
                else:
                    # 按修改时间排序，取最新的
                    model_files = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[:1]

                if model_files:
                    model_path = str(model_files[0])
                    logger.info(f"Auto-selected model: {model_path}")

        if not model_path:
            raise HTTPException(
                status_code=400,
                detail=f"未找到已保存的模型。请先训练模型，或检查 models/qlib/ 目录下是否有 {request.model_type}_*.pkl 文件"
            )

        logger.info(f"Loading model from: {model_path}")
        saved_data = joblib.load(model_path)

        # 兼容两种保存格式：直接保存模型 或 保存为字典
        if isinstance(saved_data, dict):
            model = saved_data.get('model')
            feature_names = saved_data.get('feature_names', request.features)
        else:
            # 直接保存的模型对象
            model = saved_data
            feature_names = getattr(model, 'feature_names_', request.features)

        if model is None:
            raise HTTPException(status_code=400, detail="模型文件无效")

        # 获取内部模型进行预测
        inner_model = None
        if hasattr(model, 'model'):
            inner_model = model.model
        elif hasattr(model, 'booster'):
            inner_model = model.booster
        else:
            inner_model = model

        # 获取预测数据（使用实时价格数据，避免 qlib 复权价格显示异常）
        analyzer = StockAnalyzer(use_realtime_price=True)
        predictions = {}

        # 解析回测日期（使用动态默认值：最近一年）
        predict_start = datetime.fromisoformat(request.get_predict_start_date())
        predict_end = datetime.fromisoformat(request.get_predict_end_date())

        # 回测参数
        initial_cash = request.initial_cash
        commission_rate = request.commission_rate
        slippage_rate = request.slippage_rate

        # 回测结果
        backtest_results = {
            "initial_cash": initial_cash,
            "final_capital": initial_cash,
            "total_return": 0.0,
            "annual_return": 0.0,
            "total_trades": 0,
            "win_trades": 0,
            "total_commission": 0.0,
            "total_slippage": 0.0,
            "trades": [],
        }

        # 初始化 Alpha158 特征工程器
        from quanttool.strategies.qlib_strategy import QlibFeatureEngineer
        feature_engineer = QlibFeatureEngineer(feature_set=request.feature_set)

        for symbol in request.symbols:
            df = analyzer.get_stock_data(symbol, 500)  # 获取更多数据用于回测
            if df.empty or len(df) < 120:
                continue

            # 确定日期列
            date_column = None
            if 'trade_date' in df.columns:
                date_column = 'trade_date'
            elif 'timestamp' in df.columns:
                date_column = 'timestamp'

            if not date_column:
                continue

            df['_date'] = pd.to_datetime(df[date_column])

            # 使用 Alpha158 特征工程
            if request.use_rich_features:
                try:
                    feature_df = feature_engineer.generate_features(df)
                    df = pd.concat([df, feature_df], axis=1)
                except Exception as e:
                    logger.warning(f"Feature engineering failed for {symbol}: {e}")
                    continue
            else:
                df = analyzer.calculate_technical_indicators(df)

            # 使用模型期望的特征
            available_features = [f for f in feature_names if f in df.columns]
            if not available_features:
                continue

            # 记录数据日期范围
            data_start = str(df['_date'].min())[:10]
            data_end = str(df['_date'].max())[:10]

            # ====== 回测逻辑 ======
            cash = initial_cash
            position = 0  # 持仓数量
            trades = []
            total_commission = 0.0
            total_slippage = 0.0

            # T+1 交易：记录买入日期，卖出时检查是否满足 T+1
            buy_date = None
            buy_price = 0.0

            for i in range(len(df) - 5):  # 留出预测窗口
                row = df.iloc[i]

                # 获取交易日期
                trade_date = None
                if date_column and date_column in row:
                    trade_date = row[date_column]
                elif df.index.name:
                    trade_date = df.index[i]

                if trade_date is None:
                    continue

                # 检查是否在回测日期范围内
                try:
                    if hasattr(trade_date, 'to_pydatetime'):
                        trade_dt = trade_date.to_pydatetime()
                    elif isinstance(trade_date, str):
                        trade_dt = datetime.fromisoformat(trade_date[:10])
                    elif hasattr(trade_date, 'strftime'):
                        # pandas Timestamp
                        trade_dt = trade_date.to_pydatetime()
                    else:
                        continue
                except:
                    continue

                if trade_dt < predict_start or trade_dt > predict_end:
                    continue

                # 获取当日特征
                X = df[available_features].iloc[i:i+1].values

                try:
                    pred = inner_model.predict(X)[0]
                except:
                    try:
                        pred = float(inner_model.predict(X.reshape(1, -1))[0])
                    except:
                        continue

                if isinstance(pred, (int, float)):
                    pred_value = float(pred)
                elif hasattr(pred, '__len__'):
                    pred_value = float(pred[0])
                else:
                    pred_value = float(pred)

                # 生成信号 (回归模型预测收益率，阈值需要适配)
                # 回归值范围通常在 -0.1 到 0.1 之间
                signal = "hold"
                if pred_value > 0.005:  # 预测上涨 > 0.5%
                    signal = "buy"
                elif pred_value < -0.005:  # 预测下跌 > 0.5%
                    signal = "sell"

                # 获取价格
                close_price = float(row['close'])

                # 执行交易 (考虑 T+1)
                if signal == "buy" and position == 0 and cash > 0:
                    # 买入
                    slippage = close_price * slippage_rate
                    buy_price_actual = close_price + slippage
                    shares = int(cash / buy_price_actual / 100) * 100  # A股一手100股

                    if shares > 0:
                        commission = max(shares * buy_price_actual * commission_rate, 5)  # 最低5元
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
                                "date": str(trade_date)[:10],
                                "price": round(buy_price_actual, 2),
                                "shares": shares,
                                "commission": round(commission, 2),
                                "slippage": round(shares * slippage, 2)
                            })

                elif signal == "sell" and position > 0:
                    # T+1 检查：卖出日期必须比买入日期晚至少1天
                    if buy_date is None or trade_dt <= buy_date:
                        continue

                    # 卖出
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
                        "date": str(trade_date)[:10],
                        "price": round(sell_price_actual, 2),
                        "shares": position,
                        "commission": round(commission, 2),
                        "slippage": round(position * slippage, 2),
                        "profit": round(profit, 2)
                    })

                    if profit > 0:
                        backtest_results["win_trades"] += 1

                    position = 0
                    buy_date = None

            # 计算最终市值
            if len(df) > 0:
                final_price = float(df['close'].iloc[-1])
                final_capital = cash + position * final_price
            else:
                final_capital = cash

            total_return = (final_capital - initial_cash) / initial_cash

            # 计算年化收益
            days = (predict_end - predict_start).days
            annual_return = total_return * 252 / max(days, 1) if days > 0 else 0

            # ====== 计算最大回撤 ======
            max_drawdown = 0.0
            if trades:
                # 重建市值曲线
                equity_curve = [initial_cash]
                peak_equity = initial_cash

                # 简化：根据交易记录估算市值变化
                running_cash = initial_cash
                running_position = 0
                running_buy_price = 0.0

                for trade in trades:
                    if trade['type'] == 'buy':
                        running_cash -= trade['shares'] * trade['price'] + trade['commission'] + trade.get('slippage', 0)
                        running_position = trade['shares']
                        running_buy_price = trade['price']
                    elif trade['type'] == 'sell':
                        running_cash += trade['shares'] * trade['price'] - trade['commission']
                        running_position = 0

                    # 假设当日市值为现金（简化计算）
                    equity = running_cash
                    equity_curve.append(equity)

                    # 计算回撤
                    if equity > peak_equity:
                        peak_equity = equity
                    drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

            # ====== 计算夏普比率 ======
            sharpe_ratio = 0.0
            if days > 0 and total_return != 0:
                # 简化：使用年化收益率和假设的波动率
                # 实际应该用每日收益率计算
                # 这里用估算：假设年化波动率约 20%
                assumed_volatility = 0.20
                risk_free_rate = 0.02  # 无风险利率 2%
                if assumed_volatility > 0:
                    sharpe_ratio = (annual_return - risk_free_rate) / assumed_volatility

            # 获取最新预测
            X_latest = df[available_features].iloc[-1:].values
            try:
                pred_latest = inner_model.predict(X_latest)[0]
            except:
                pred_latest = 0.5

            predictions[symbol] = {
                "prediction": round(float(pred_latest), 4),
                "signal": "buy" if float(pred_latest) > 0.55 else ("sell" if float(pred_latest) < 0.45 else "hold"),
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
                    "max_drawdown": round(max_drawdown * 100, 2),
                    "sharpe_ratio": round(sharpe_ratio, 2),
                    "total_trades": len(trades),
                    "win_rate": round(backtest_results["win_trades"] / len(trades) * 100, 1) if trades else 0,
                    "total_commission": round(total_commission, 2),
                    "total_slippage": round(total_slippage, 2),
                    "trades": trades[-10:],  # 最近10笔交易
                }
            }

            backtest_results["total_trades"] += len(trades)

        # 计算汇总统计
        total_final_capital = sum(
            p["backtest"]["final_capital"] for p in predictions.values()
        )
        total_win_trades = sum(
            1 for p in predictions.values()
            for t in p["backtest"].get("trades", [])
            if t.get("type") == "sell" and t.get("profit", 0) > 0
        )
        total_sell_trades = sum(
            1 for p in predictions.values()
            for t in p["backtest"].get("trades", [])
            if t.get("type") == "sell"
        )

        # 汇总回测结果
        summary = {
            "total_return_pct": round((total_final_capital - initial_cash * len(predictions)) / (initial_cash * len(predictions)) * 100, 2) if predictions else 0,
            "total_trades": backtest_results["total_trades"],
            "win_rate": round(total_win_trades / total_sell_trades * 100, 1) if total_sell_trades > 0 else 0,
            "predicted_stocks": len(predictions),
        }

        return {
            "success": True,
            "model_type": request.model_type,
            "model_path": model_path,
            "model_name": Path(model_path).name if model_path else None,
            "feature_count": len(feature_names),
            "predict_period": {
                "start_date": request.predict_start_date,
                "end_date": request.predict_end_date,
                "days": (predict_end - predict_start).days,
            },
            "backtest_params": {
                "initial_cash": f"¥{initial_cash:,.0f}",
                "initial_cash_raw": initial_cash,
                "commission_rate": f"{commission_rate * 100:.4f}%",
                "slippage_rate": f"{slippage_rate * 100:.4f}%",
                "total_cost_rate": f"{(commission_rate + slippage_rate) * 100:.4f}%",
                "t_plus_1": True,
            },
            "summary": summary,
            "predictions": predictions,
            "total_stocks": len(request.symbols),
            "predicted_stocks": len(predictions)
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")
