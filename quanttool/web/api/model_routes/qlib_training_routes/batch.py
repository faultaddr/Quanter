"""Qlib batch training API route."""

from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from .....core.logging import get_logger
from ....schemas.model import QlibTrainRequest


logger = get_logger(__name__)
router = APIRouter()

@router.post("/qlib/train")
async def train_qlib_model(request: QlibTrainRequest) -> Dict[str, Any]:
    """
    训练 Qlib ML 模型

    使用沪深300成分股作为训练数据，按年份划分训练/验证/测试集：
    - 训练集: 2020-2023年
    - 验证集: 2024-2025年
    - 测试集: 2026年

    用户输入的股票代码仅用于预测，不参与训练
    """
    try:
        from quanttool.strategies.qlib import create_model
        from quanttool.factors.stock_analyzer import StockAnalyzer
        from quanttool.cli.commands.analysis_commands import get_csi300_constituents
        import numpy as np
        import os
        import uuid

        # 获取沪深300成分股作为训练数据
        csi300_stocks = get_csi300_constituents()
        train_symbols = [s['code'] if isinstance(s, dict) else s for s in csi300_stocks]

        # 限制训练股票数量
        if request.max_train_stocks > 0:
            train_symbols = train_symbols[:request.max_train_stocks]

        logger.info(f"Training with {len(train_symbols)} CSI300 stocks")

        # 获取训练数据
        analyzer = StockAnalyzer()
        train_data = []
        valid_data = []
        test_data = []

        # 解析日期
        train_start_dt = datetime.fromisoformat(request.train_start)
        train_end_dt = datetime.fromisoformat(request.train_end)
        valid_start_dt = datetime.fromisoformat(request.valid_start)
        valid_end_dt = datetime.fromisoformat(request.valid_end)
        test_start_dt = datetime.fromisoformat(request.test_start)
        test_end_dt = datetime.fromisoformat(request.test_end)

        logger.info(f"Date ranges - Train: {request.train_start} to {request.train_end}, "
                   f"Valid: {request.valid_start} to {request.valid_end}, "
                   f"Test: {request.test_start} to {request.test_end}")

        success_count = 0
        first_symbol_features = None  # 记录第一个成功股票的特征列名，确保所有股票使用相同特征

        # 初始化 Alpha158 特征工程器
        from quanttool.strategies.qlib_strategy import QlibFeatureEngineer
        feature_engineer = QlibFeatureEngineer(feature_set=request.feature_set)

        for symbol in train_symbols:
            try:
                # 获取足够的历史数据（约7年，覆盖2019-2026训练+验证+预测期间）
                df = analyzer.get_stock_data(symbol, 2500)
                if df.empty or len(df) < 120:  # Alpha158 需要至少 120 条数据
                    logger.warning(f"Insufficient data for {symbol}: {len(df) if not df.empty else 0} rows")
                    continue

                # 确定日期列
                date_column = None
                if 'trade_date' in df.columns:
                    date_column = 'trade_date'
                elif 'timestamp' in df.columns:
                    date_column = 'timestamp'

                if not date_column:
                    logger.warning(f"No date column found for {symbol}")
                    continue

                df['_date'] = pd.to_datetime(df[date_column])

                if request.use_rich_features:
                    # 使用 Alpha158 特征工程 (150+ 特征)
                    try:
                        feature_df = feature_engineer.generate_features(df)
                        available_features = list(feature_df.columns)
                        df = pd.concat([df, feature_df], axis=1)
                    except Exception as e:
                        logger.warning(f"Feature engineering failed for {symbol}: {e}")
                        continue
                else:
                    # 计算技术指标
                    df = analyzer.calculate_technical_indicators(df)

                    if request.features:
                        # 使用用户指定的特征
                        available_features = [f for f in request.features if f in df.columns]
                    else:
                        # 使用基本特征
                        available_features = ['close', 'volume', 'ma_5', 'ma_10', 'ma_20']

                if not available_features:
                    logger.warning(f"No available features for {symbol}")
                    continue

                # 确保所有股票使用相同的特征列
                if first_symbol_features is None:
                    first_symbol_features = available_features
                else:
                    # 使用第一个股票的特征列，确保一致性
                    available_features = [f for f in first_symbol_features if f in df.columns]
                    if len(available_features) != len(first_symbol_features):
                        logger.warning(f"Feature mismatch for {symbol}, expected {len(first_symbol_features)}, got {len(available_features)}")
                        continue

                logger.info(f"Using {len(available_features)} features for {symbol}")

                # 计算标签（未来5日收益率）
                df['return_5d'] = df['close'].pct_change(5).shift(-5)

                # 调试：输出数据的日期范围
                data_min_date = df['_date'].min()
                data_max_date = df['_date'].max()
                logger.info(f"{symbol}: data range {data_min_date} to {data_max_date}, {len(df)} rows")

                # 按日期划分数据
                row_count = 0
                for idx, row in df.iterrows():
                    date_val = row['_date']
                    if pd.isna(date_val):
                        continue

                    feature_vals = [row.get(f) for f in available_features]
                    label_val = row.get('return_5d')

                    # 过滤无效值
                    if any(pd.isna(v) for v in feature_vals) or pd.isna(label_val):
                        continue

                    row_data = {
                        'features': feature_vals,
                        'label': label_val,
                        'symbol': symbol,
                        'date': date_val
                    }

                    # 划分数据集
                    if train_start_dt <= date_val <= train_end_dt:
                        train_data.append(row_data)
                        row_count += 1
                    elif valid_start_dt <= date_val <= valid_end_dt:
                        valid_data.append(row_data)
                        row_count += 1
                    elif test_start_dt <= date_val <= test_end_dt:
                        test_data.append(row_data)
                        row_count += 1

                if row_count > 0:
                    success_count += 1

            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {e}")
                continue

        logger.info(f"Data collection complete: {success_count} stocks succeeded, "
                   f"train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}")

        if not train_data:
            raise HTTPException(
                status_code=400,
                detail=f"无法获取足够的训练数据。收集了 {success_count} 只股票，训练集 {len(train_data)} 条，"
                       f"验证集 {len(valid_data)} 条，测试集 {len(test_data)} 条。请检查日期范围是否在数据覆盖范围内。"
            )

        # 准备训练数据
        feature_cols = available_features
        X_train = np.array([d['features'] for d in train_data])
        y_train = np.array([d['label'] for d in train_data])

        # 创建模型
        config_kwargs = {
            'n_estimators': request.n_estimators,
            'max_depth': request.max_depth,
            'learning_rate': request.learning_rate,
            'hidden_size': request.hidden_size,
            'num_layers': request.num_layers,
            'dropout': request.dropout,
            'epochs': request.epochs,
            'batch_size': request.batch_size,
        }

        model = create_model(request.model_type, **config_kwargs)

        # 训练
        X_train_df = pd.DataFrame(X_train, columns=feature_cols)
        y_train_series = pd.Series(y_train)
        model.fit(X_train_df, y_train_series)
        model.feature_names_ = feature_cols

        # 保存模型
        model_dir = "models/qlib"
        os.makedirs(model_dir, exist_ok=True)
        model_id = str(uuid.uuid4())[:8]
        model_path = f"{model_dir}/{request.model_type}_{model_id}.pkl"
        model.save(model_path)

        # 评估训练集
        train_pred = model.predict(X_train_df)
        train_mse = np.mean((train_pred - y_train) ** 2)
        train_mae = np.mean(np.abs(train_pred - y_train))
        train_ic = np.corrcoef(train_pred, y_train)[0, 1] if len(train_pred) > 1 else 0

        # 评估验证集
        valid_metrics = {}
        if valid_data:
            X_valid = np.array([d['features'] for d in valid_data])
            y_valid = np.array([d['label'] for d in valid_data])
            X_valid_df = pd.DataFrame(X_valid, columns=feature_cols)
            valid_pred = model.predict(X_valid_df)
            valid_metrics = {
                "samples": len(valid_data),
                "mse": round(float(np.mean((valid_pred - y_valid) ** 2)), 6),
                "mae": round(float(np.mean(np.abs(valid_pred - y_valid))), 6),
                "ic": round(float(np.corrcoef(valid_pred, y_valid)[0, 1]), 4) if len(valid_pred) > 1 else 0,
            }

        # 评估测试集
        test_metrics = {}
        if test_data:
            X_test = np.array([d['features'] for d in test_data])
            y_test = np.array([d['label'] for d in test_data])
            X_test_df = pd.DataFrame(X_test, columns=feature_cols)
            test_pred = model.predict(X_test_df)
            test_metrics = {
                "samples": len(test_data),
                "mse": round(float(np.mean((test_pred - y_test) ** 2)), 6),
                "mae": round(float(np.mean(np.abs(test_pred - y_test))), 6),
                "ic": round(float(np.corrcoef(test_pred, y_test)[0, 1]), 4) if len(test_pred) > 1 else 0,
            }

        return {
            "model_id": model_id,
            "model_type": request.model_type,
            "model_path": model_path,
            "train_symbols_count": len(train_symbols),
            "predict_symbols": request.symbols,  # 用户输入的股票代码（仅用于预测）
            "train_samples": len(train_data),
            "features": feature_cols,
            "feature_count": len(feature_cols),
            "use_rich_features": request.use_rich_features,
            "data_split": {
                "train": {
                    "period": f"{request.train_start} ~ {request.train_end}",
                    "samples": len(train_data),
                },
                "valid": {
                    "period": f"{request.valid_start} ~ {request.valid_end}",
                    "samples": len(valid_data),
                },
                "test": {
                    "period": f"{request.test_start} ~ {request.test_end}",
                    "samples": len(test_data),
                },
            },
            "metrics": {
                "train": {
                    "mse": round(float(train_mse), 6),
                    "mae": round(float(train_mae), 6),
                    "ic": round(float(train_ic), 4) if not np.isnan(train_ic) else 0,
                },
                "valid": valid_metrics,
                "test": test_metrics,
            },
            "backtest_params": {
                "initial_cash": request.initial_cash,
                "commission_rate": request.commission_rate,
                "slippage_rate": request.slippage_rate,
                "t_plus_1": True,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"模型训练失败: {str(e)}")
