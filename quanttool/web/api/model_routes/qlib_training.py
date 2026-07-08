"""Qlib model training API routes."""

from datetime import datetime, timedelta
import json
import queue
import threading
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ....core.logging import get_logger
from ...schemas.model import QlibTrainRequest


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


@router.post("/qlib/train/stream")
async def train_qlib_model_stream(request: QlibTrainRequest):
    """
    使用 SSE 流式推送训练进度

    事件类型:
    - progress: 进度更新
    - log: 日志消息
    - complete: 训练完成
    - error: 错误
    """
    import asyncio

    # 使用同步队列（线程安全）
    message_queue = queue.Queue()

    def send_event(event_type: str, data: Dict[str, Any]):
        """发送SSE事件到队列"""
        message_queue.put({"event": event_type, "data": data})

    def training_worker():
        """后台训练线程"""
        try:
            from quanttool.strategies.qlib import create_model
            from quanttool.factors.stock_analyzer import StockAnalyzer
            from quanttool.cli.commands.analysis_commands import get_csi300_constituents
            import numpy as np
            import os
            import uuid

            # 阶段1: 初始化
            send_event("progress", {
                "stage": "init",
                "progress": 0,
                "message": "初始化训练环境..."
            })

            # 获取沪深300成分股
            csi300_stocks = get_csi300_constituents()
            train_symbols = [s['code'] if isinstance(s, dict) else s for s in csi300_stocks]

            if request.max_train_stocks > 0:
                train_symbols = train_symbols[:request.max_train_stocks]

            total_stocks = len(train_symbols)
            send_event("progress", {
                "stage": "init",
                "progress": 5,
                "message": f"准备获取 {total_stocks} 只沪深300成分股数据"
            })

            # 阶段2: 数据获取
            analyzer = StockAnalyzer()
            train_data = []
            valid_data = []
            test_data = []

            train_start_dt = datetime.fromisoformat(request.train_start)
            train_end_dt = datetime.fromisoformat(request.train_end)
            valid_start_dt = datetime.fromisoformat(request.valid_start)
            valid_end_dt = datetime.fromisoformat(request.valid_end)
            test_start_dt = datetime.fromisoformat(request.test_start)
            test_end_dt = datetime.fromisoformat(request.test_end)

            success_count = 0
            cache_hits = 0
            first_symbol_features = None  # 记录第一个成功股票的特征列名

            # 初始化 Alpha158 特征工程器
            from quanttool.strategies.qlib_strategy import QlibFeatureEngineer
            feature_engineer = QlibFeatureEngineer(feature_set=request.feature_set)

            # 计算实际需要的日期范围
            train_end_date = datetime.fromisoformat(request.train_end)
            start_date = train_end_date - timedelta(days=2500)  # 约 7 年

            # 先并发预加载所有股票数据（显著提升性能）
            send_event("progress", {
                "stage": "data_preload",
                "progress": 5,
                "message": f"并发预加载 {total_stocks} 只股票数据..."
            })

            loaded_count = analyzer.preload_data_for_scan(train_symbols, days=2500)
            send_event("progress", {
                "stage": "data_preload",
                "progress": 10,
                "message": f"预加载完成，成功获取 {loaded_count} 只股票数据"
            })

            # 使用 qlib 原生训练流程
            send_event("progress", {
                "stage": "qlib_setup",
                "progress": 15,
                "message": "初始化 Qlib 训练环境..."
            })

            try:
                from quanttool.infrastructure.data_providers.qlib_data_converter import (
                    QlibDataConverter,
                    QlibTrainingPipeline,
                    QlibDataConfig
                )

                # 配置 Qlib 数据转换器
                qlib_config = QlibDataConfig(
                    cache_dir=".cache/incremental_data",
                    output_dir="qlib_data/cn_data",
                    feature_type="alpha158" if request.use_rich_features else "alpha360",
                    start_date=request.train_start,
                    end_date=request.train_end,
                )

                converter = QlibDataConverter(qlib_config)
                pipeline = QlibTrainingPipeline(converter)

                send_event("log", {"message": f"使用 Qlib 原生训练流程 (特征: {qlib_config.feature_type})"})

                # 阶段3: 转换数据为 qlib 格式
                send_event("progress", {
                    "stage": "data_conversion",
                    "progress": 20,
                    "message": "转换数据为 Qlib 原生格式..."
                })

                # 转换股票代码格式 (000001.SZ -> 000001_SZ)
                qlib_symbols = [s.replace('.', '_') for s in train_symbols]

                # 创建 qlib DatasetH
                dataset = converter.create_qlib_dataset(
                    symbols=qlib_symbols,
                    start_date=request.train_start,
                    end_date=request.test_end,
                    feature_type=qlib_config.feature_type,
                    label_type="return_10"
                )

                send_event("log", {"message": f"Qlib DatasetH 创建成功"})

            except Exception as e:
                send_event("log", {"message": f"Qlib 原生流程失败，回退到 sklearn: {e}"})
                import traceback
                traceback.print_exc()

                # 回退到传统流程
                for i, symbol in enumerate(train_symbols):
                    try:
                        send_event("progress", {
                            "stage": "data_collection",
                            "progress": 10 + int((i / total_stocks) * 50),
                            "current": symbol,
                            "processed": i + 1,
                            "total": total_stocks,
                            "cache_hits": cache_hits,
                            "message": f"正在处理数据: {symbol} ({i + 1}/{total_stocks})"
                        })

                        df = analyzer.get_stock_data(
                            symbol,
                            start_date=start_date,
                            end_date=datetime.now(),
                            force_refresh=False
                        )

                        if len(df) >= 500:
                            cache_hits += 1

                        if df.empty or len(df) < 120:
                            continue

                        date_column = None
                        if 'trade_date' in df.columns:
                            date_column = 'trade_date'
                        elif 'timestamp' in df.columns:
                            date_column = 'timestamp'

                        if not date_column:
                            continue

                        df['_date'] = pd.to_datetime(df[date_column])

                        if request.use_rich_features:
                            try:
                                feature_df = feature_engineer.generate_features(df)
                                available_features = list(feature_df.columns)
                                df = pd.concat([df, feature_df], axis=1)
                            except Exception as e:
                                continue
                        else:
                            df = analyzer.calculate_technical_indicators(df)
                            if request.features:
                                available_features = [f for f in request.features if f in df.columns]
                            else:
                                available_features = ['close', 'volume', 'ma_5', 'ma_10', 'ma_20']

                        if not available_features:
                            continue

                        if first_symbol_features is None:
                            first_symbol_features = available_features
                        else:
                            available_features = [f for f in first_symbol_features if f in df.columns]
                            if len(available_features) != len(first_symbol_features):
                                continue

                        df['return_5d'] = df['close'].pct_change(5).shift(-5)
                        df['label'] = df['return_5d']

                        for idx, row in df.iterrows():
                            date_val = row['_date']
                            if pd.isna(date_val):
                                continue

                            feature_vals = [row.get(f) for f in available_features]
                            label_val = row.get('label')

                            if any(pd.isna(v) for v in feature_vals) or pd.isna(label_val):
                                continue

                            row_data = {
                                'features': feature_vals,
                                'label': label_val,
                                'symbol': symbol,
                                'date': date_val,
                            }

                            if train_start_dt <= date_val <= train_end_dt:
                                train_data.append(row_data)
                            elif valid_start_dt <= date_val <= valid_end_dt:
                                valid_data.append(row_data)
                            elif test_start_dt <= date_val <= test_end_dt:
                                test_data.append(row_data)

                        success_count += 1

                    except Exception as e:
                        continue

                # 传统流程：准备数据
                if train_data:
                    feature_cols = available_features
                    X_train = np.array([d['features'] for d in train_data])
                    y_train = np.array([d['label'] for d in train_data])
                    dataset = None  # 标记使用传统流程

            # 阶段4: 模型训练
            n_estimators = request.n_estimators or 100

            if dataset is not None:
                # 使用 Qlib 原生训练流程
                send_event("progress", {
                    "stage": "training",
                    "progress": 75,
                    "message": f"使用 Qlib 原生 {request.model_type.upper()} 模型训练..."
                })

                try:
                    # 初始化 Qlib
                    import qlib
                    if not hasattr(qlib, '_initialized') or not qlib._initialized:
                        qlib.init(provider_uri="qlib_data/cn_data")
                        qlib._initialized = True

                    # Qlib 内置模型映射表: (模块名, 类名, 模型类型)
                    QLIB_MODELS = {
                        # GBDT 系列
                        'lgb': ('gbdt', 'LGBModel', 'gbdt'),
                        'lightgbm': ('gbdt', 'LGBModel', 'gbdt'),
                        'xgboost': ('xgboost', 'XGBModel', 'gbdt'),
                        'xgb': ('xgboost', 'XGBModel', 'gbdt'),
                        'catboost': ('catboost_model', 'CatBoostModel', 'gbdt'),
                        'double_ensemble': ('double_ensemble', 'DEnsembleModel', 'gbdt'),
                    }

                    model_type_lower = request.model_type.lower()

                    if model_type_lower not in QLIB_MODELS:
                        supported = ', '.join(sorted(QLIB_MODELS.keys()))
                        raise ValueError(f"不支持的模型类型: {request.model_type}。支持: {supported}")

                    module_name, class_name, model_category = QLIB_MODELS[model_type_lower]
                    ModelClass = getattr(__import__(f'qlib.contrib.model.{module_name}', fromlist=[class_name]), class_name)

                    # 创建 GBDT 模型
                    model = ModelClass(
                        loss='mse',
                        n_estimators=n_estimators,
                        max_depth=request.max_depth or 6,
                        learning_rate=request.learning_rate or 0.01,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1,
                    )

                    send_event("log", {"message": f"创建 Qlib 原生 {request.model_type.upper()} 模型成功"})

                    import time
                    train_start_time = time.time()

                    train_msg = f"Qlib {request.model_type.upper()} 训练中 ({n_estimators} 棵树)..."

                    send_event("progress", {
                        "stage": "training",
                        "progress": 80,
                        "message": train_msg
                    })

                    model.fit(dataset)

                    train_elapsed = time.time() - train_start_time
                    send_event("log", {"message": f"Qlib 原生训练完成 (耗时 {train_elapsed:.1f}s)"})

                    # 获取特征数
                    feature_cols = []
                    try:
                        df_sample = dataset.prepare("train", col_set=["feature"])
                        if isinstance(df_sample, dict):
                            feature_cols = list(df_sample["feature"].columns)
                        else:
                            feature_cols = list(df_sample.xs('feature', axis=1, level=0).columns)
                    except Exception:
                        feature_cols = ["alpha158_features"]

                    # 保存模型到指定目录
                    model_dir = "models/qlib"
                    os.makedirs(model_dir, exist_ok=True)
                    model_id = str(uuid.uuid4())[:8]
                    model_path = f"{model_dir}/{request.model_type}_{model_id}.pkl"

                    # 使用 Qlib 的 to_pickle 方法保存模型
                    model.to_pickle(model_path)
                    send_event("log", {"message": f"模型已保存: {model_path}"})

                except Exception as e:
                    send_event("error", {"message": f"Qlib 原生训练失败: {str(e)}"})
                    import traceback
                    traceback.print_exc()
                    return

            else:
                # 传统 sklearn 流程
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

                X_train_df = pd.DataFrame(X_train, columns=feature_cols)
                y_train_series = pd.Series(y_train)

                # 检查数据有效性
                send_event("log", {"message": f"数据形状: X={X_train_df.shape}, y={y_train_series.shape}"})
                if X_train_df.empty or len(X_train_df) == 0:
                    send_event("error", {
                        "message": f"训练数据为空，请检查数据获取和特征工程"
                    })
                    return

                # 处理 NaN 和 Inf 值
                X_train_df = X_train_df.fillna(0).replace([np.inf, -np.inf], 0)
                y_train_series = y_train_series.fillna(0).replace([np.inf, -np.inf], 0)

                send_event("progress", {
                    "stage": "training",
                    "progress": 80,
                    "message": f"开始训练 {request.model_type.upper()} 模型 ({n_estimators} 棵树)..."
                })

                try:
                    import time
                    train_start_time = time.time()
                    model.fit(X_train_df, y_train_series)
                    train_elapsed = time.time() - train_start_time

                    model.feature_names_ = feature_cols
                    send_event("log", {"message": f"模型训练完成 (耗时 {train_elapsed:.1f}s)"})
                except Exception as e:
                    send_event("error", {"message": f"模型训练失败: {str(e)}"})
                    return

            # 阶段5: 模型评估
            send_event("progress", {
                "stage": "evaluation",
                "progress": 90,
                "message": "评估模型性能..."
            })

            # 保存模型（如果还没有保存）
            if dataset is None:
                # 传统 sklearn 流程需要在这里保存
                import joblib
                model_dir = "models/qlib"
                os.makedirs(model_dir, exist_ok=True)
                model_id = str(uuid.uuid4())[:8]
                model_path = f"{model_dir}/{request.model_type}_{model_id}.pkl"
                joblib.dump(model, model_path)
                send_event("log", {"message": f"模型已保存: {model_path}"})

            # 评估
            if dataset is not None:
                # Qlib 原生模型评估
                try:
                    send_event("log", {"message": f"开始评估，数据集类型: {type(dataset).__name__}"})

                    # 从数据集获取训练数据进行评估
                    send_event("log", {"message": "准备获取训练集数据..."})
                    train_df = dataset.prepare("train", col_set=["feature", "label"])
                    send_event("log", {"message": f"训练集数据获取完成，类型: {type(train_df).__name__}"})
                    if isinstance(train_df, dict):
                        send_event("log", {"message": f"训练集是 dict，keys: {list(train_df.keys())}"})
                        X_train_eval = train_df["feature"]
                        y_train_eval = train_df["label"].values.ravel()
                    else:
                        send_event("log", {"message": f"训练集是 DataFrame，shape: {train_df.shape}"})
                        X_train_eval = train_df.xs('feature', axis=1, level=0)
                        y_train_eval = train_df.xs('label', axis=1, level=0).values.ravel()

                    send_event("log", {"message": f"训练集特征形状: {X_train_eval.shape}, 标签形状: {y_train_eval.shape}"})

                    # 使用 Qlib 模型预测
                    send_event("log", {"message": "开始训练集预测..."})
                    if hasattr(model, 'model') and model.model is not None:
                        train_pred = model.model.predict(X_train_eval.values)
                    else:
                        train_pred = model.predict(X_train_eval)
                    send_event("log", {"message": f"训练集预测完成，预测形状: {train_pred.shape}"} )

                    train_pred = train_pred.ravel() if len(train_pred.shape) > 1 else train_pred

                    train_mse = np.mean((train_pred - y_train_eval) ** 2)
                    train_mae = np.mean(np.abs(train_pred - y_train_eval))
                    train_ic = np.corrcoef(train_pred, y_train_eval)[0, 1] if len(train_pred) > 1 else 0

                    send_event("log", {"message": f"训练集评估: MSE={train_mse:.6f}, MAE={train_mae:.6f}, IC={train_ic:.4f}"})

                    valid_metrics = {}
                    test_metrics = {}

                    # 验证集评估
                    send_event("log", {"message": "开始验证集评估..."})
                    try:
                        valid_df = dataset.prepare("valid", col_set=["feature", "label"])
                        send_event("log", {"message": f"验证集数据获取完成，类型: {type(valid_df).__name__}"})
                        if isinstance(valid_df, dict):
                            X_valid_eval = valid_df["feature"]
                            y_valid_eval = valid_df["label"].values.ravel()
                        else:
                            X_valid_eval = valid_df.xs('feature', axis=1, level=0)
                            y_valid_eval = valid_df.xs('label', axis=1, level=0).values.ravel()

                        if hasattr(model, 'model') and model.model is not None:
                            valid_pred = model.model.predict(X_valid_eval.values)
                        else:
                            valid_pred = model.predict(X_valid_eval)

                        valid_pred = valid_pred.ravel() if len(valid_pred.shape) > 1 else valid_pred
                        valid_mse = np.mean((valid_pred - y_valid_eval) ** 2)
                        valid_mae = np.mean(np.abs(valid_pred - y_valid_eval))
                        valid_ic = np.corrcoef(valid_pred, y_valid_eval)[0, 1] if len(valid_pred) > 1 else 0

                        valid_metrics = {
                            "samples": len(y_valid_eval),
                            "mse": round(float(valid_mse), 6),
                            "mae": round(float(valid_mae), 6),
                            "ic": round(float(valid_ic), 4),
                        }
                    except Exception as ve:
                        send_event("log", {"message": f"验证集评估失败: {ve}"})
                        pass

                    # 测试集评估
                    send_event("log", {"message": "开始测试集评估..."})
                    try:
                        test_df = dataset.prepare("test", col_set=["feature", "label"])
                        send_event("log", {"message": f"测试集数据获取完成，类型: {type(test_df).__name__}"})
                        if isinstance(test_df, dict):
                            X_test_eval = test_df["feature"]
                            y_test_eval = test_df["label"].values.ravel()
                        else:
                            X_test_eval = test_df.xs('feature', axis=1, level=0)
                            y_test_eval = test_df.xs('label', axis=1, level=0).values.ravel()

                        if hasattr(model, 'model') and model.model is not None:
                            test_pred = model.model.predict(X_test_eval.values)
                        else:
                            test_pred = model.predict(X_test_eval)

                        test_pred = test_pred.ravel() if len(test_pred.shape) > 1 else test_pred
                        test_mse = np.mean((test_pred - y_test_eval) ** 2)
                        test_mae = np.mean(np.abs(test_pred - y_test_eval))
                        test_ic = np.corrcoef(test_pred, y_test_eval)[0, 1] if len(test_pred) > 1 else 0

                        test_metrics = {
                            "samples": len(y_test_eval),
                            "mse": round(float(test_mse), 6),
                            "mae": round(float(test_mae), 6),
                            "ic": round(float(test_ic), 4),
                        }
                    except Exception as te:
                        send_event("log", {"message": f"测试集评估失败: {te}"})
                        pass

                    send_event("log", {"message": "模型评估完成"})

                except Exception as e:
                    send_event("log", {"message": f"评估警告: {e}"})
                    train_mse = 0
                    train_mae = 0
                    train_ic = 0
                    valid_metrics = {}
                    test_metrics = {}

            else:
                # 传统 sklearn 流程评估
                train_pred = model.predict(X_train_df)
                train_mse = np.mean((train_pred - y_train) ** 2)
                train_mae = np.mean(np.abs(train_pred - y_train))

                def calculate_ic(predictions, data_list):
                    from scipy.stats import spearmanr
                    date_data = {}
                    for i, d in enumerate(data_list):
                        date_val = d['date']
                        if date_val not in date_data:
                            date_data[date_val] = {'pred': [], 'return': []}
                        date_data[date_val]['pred'].append(predictions[i])
                        date_data[date_val]['return'].append(d.get('return_5d', 0))

                    ics = []
                    for date_val, data in date_data.items():
                        preds = np.array(data['pred'])
                        returns = np.array(data['return'])
                        if len(preds) >= 5:
                            if np.std(preds) > 1e-10 and np.std(returns) > 1e-10:
                                try:
                                    ic, _ = spearmanr(preds, returns)
                                    if not np.isnan(ic):
                                        ics.append(ic)
                                except:
                                    pass
                    return np.mean(ics) if ics else 0.0

                train_ic = calculate_ic(train_pred, train_data)

                valid_metrics = {}
                if valid_data:
                    X_valid = np.array([d['features'] for d in valid_data])
                    y_valid = np.array([d['label'] for d in valid_data])
                    X_valid_df = pd.DataFrame(X_valid, columns=feature_cols)
                    valid_pred = model.predict(X_valid_df)
                    valid_mse = np.mean((valid_pred - y_valid) ** 2)
                    valid_mae = np.mean(np.abs(valid_pred - y_valid))
                    valid_metrics = {
                        "samples": len(valid_data),
                        "mse": round(float(valid_mse), 6),
                        "mae": round(float(valid_mae), 6),
                        "ic": round(float(calculate_ic(valid_pred, valid_data)), 4),
                    }

                test_metrics = {}
                if test_data:
                    X_test = np.array([d['features'] for d in test_data])
                    y_test = np.array([d['label'] for d in test_data])
                    X_test_df = pd.DataFrame(X_test, columns=feature_cols)
                    test_pred = model.predict(X_test_df)
                    test_mse = np.mean((test_pred - y_test) ** 2)
                    test_mae = np.mean(np.abs(test_pred - y_test))
                    test_metrics = {
                        "samples": len(test_data),
                        "mse": round(float(test_mse), 6),
                        "mae": round(float(test_mae), 6),
                        "ic": round(float(calculate_ic(test_pred, test_data)), 4),
                    }

            # 阶段6: 完成
            send_event("progress", {
                "stage": "complete",
                "progress": 100,
                "message": "训练完成！"
            })

            # 统计样本数
            train_samples = len(train_data) if train_data else 0
            valid_samples = len(valid_data) if valid_data else 0
            test_samples = len(test_data) if test_data else 0

            # 如果使用 Qlib 原生流程，从 dataset 获取样本数
            if dataset is not None:
                try:
                    train_df = dataset.prepare("train", col_set=["feature"])
                    train_samples = len(train_df) if hasattr(train_df, '__len__') else 0
                    valid_df = dataset.prepare("valid", col_set=["feature"])
                    valid_samples = len(valid_df) if hasattr(valid_df, '__len__') else 0
                    test_df = dataset.prepare("test", col_set=["feature"])
                    test_samples = len(test_df) if hasattr(test_df, '__len__') else 0
                except Exception:
                    pass

            result = {
                "model_id": model_id,
                "model_type": request.model_type,
                "model_path": model_path,
                "train_symbols_count": len(train_symbols),
                "predict_symbols": request.symbols,
                "train_samples": train_samples,
                "features": list(feature_cols) if feature_cols else [],
                "data_split": {
                    "train": {"period": f"{request.train_start} ~ {request.train_end}", "samples": train_samples},
                    "valid": {"period": f"{request.valid_start} ~ {request.valid_end}", "samples": valid_samples},
                    "test": {"period": f"{request.test_start} ~ {request.test_end}", "samples": test_samples},
                },
                "metrics": {
                    "train": {
                        "samples": train_samples,
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

            send_event("complete", {"result": result})

        except Exception as e:
            import traceback
            traceback.print_exc()
            send_event("error", {"message": f"训练失败: {str(e)}"})

    async def event_stream():
        """生成SSE事件流"""
        loop = asyncio.get_event_loop()

        # 在线程池中运行训练工作器
        thread = threading.Thread(target=training_worker)
        thread.start()

        # 从同步队列读取事件并发送
        while True:
            try:
                # 使用 run_in_executor 非阻塞地获取消息
                msg = await loop.run_in_executor(None, lambda: message_queue.get(timeout=0.1))

                event_type = msg.get("event", "message")
                data = msg.get("data", {})

                # 格式化SSE
                yield f"event: {event_type}\n"
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                # 完成或错误时结束
                if event_type in ("complete", "error"):
                    break
            except queue.Empty:
                # 检查线程是否结束
                if not thread.is_alive():
                    break
                # 发送心跳保持连接
                yield "event: heartbeat\ndata: {}\n\n"

        # 等待线程结束
        thread.join()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

