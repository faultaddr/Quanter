# Qlib 官方训练流程

本文档说明如何将 Quanter 系统的数据转换为 qlib 官方能接受的格式，并使用 qlib 官方训练流程进行模型训练。

## 概述

Qlib 是微软开源的量化投资平台，提供了丰富的模型和数据格式支持。本系统支持将数据转换为 qlib 官方格式，完全采用官方训练流程。

### 支持的特征类型

| 特征类型 | 维度 | 说明 |
|---------|------|------|
| Alpha158 | ~158维 | 官方标准特征，包含技术指标、动量、波动率等 |
| Alpha360 | ~360维 | 扩展特征，基于历史价格和收益率 |

### 支持的模型

**GBDT 系列 (推荐入门)**
- `lgb` - LightGBM
- `xgboost` - XGBoost
- `catboost` - CatBoost
- `double_ensemble` - Double Ensemble

**PyTorch 序列模型**
- `lstm` - LSTM
- `gru` - GRU
- `transformer` - Transformer
- `tcn` - 时间卷积网络

**PyTorch 高级模型**
- `gats` - 图注意力网络
- `hist` - HIST 历史感知模型
- 更多模型请运行 `qlib list` 查看

## 快速开始

### 1. 准备数据

首先确保缓存中有股票数据：

```bash
# 获取单只股票数据
python -m quanttool.cli.main data fetch-stock 000001 365

# 批量获取沪深300成分股
python -m quanttool.cli.main data fetch-stock-batch hs300 365
```

### 2. 运行完整流程

最简单的方式是使用 `full-pipeline` 命令：

```bash
# 使用 LightGBM 和 Alpha158 特征
python -m quanttool.cli.main qlib full-pipeline --model lgb --feature alpha158

# 使用 Transformer 和 Alpha360 特征
python -m quanttool.cli.main qlib full-pipeline --model transformer --feature alpha360 --epochs 100
```

### 3. 分步执行

#### 步骤1：转换数据为 qlib 格式

```bash
# 转换缓存数据为 qlib 二进制格式
python -m quanttool.cli.main qlib dump-data --feature alpha158

# 指定输出目录和日期范围
python -m quanttool.cli.main qlib dump-data \
    --output my_qlib_data \
    --feature alpha158 \
    --start 2022-01-01 \
    --end 2024-12-31
```

输出结构：
```
qlib_data/cn_data/
├── calendars/
│   └── day.txt          # 交易日历
├── instruments/
│   └── all.txt          # 股票列表
├── features/
│   ├── 000001.SZ/
│   │   ├── close.bin
│   │   ├── open.bin
│   │   └── ...
│   └── ...
└── meta.json            # 元数据
```

#### 步骤2：训练模型

```bash
# 训练 LightGBM 模型
python -m quanttool.cli.main qlib train --model lgb --feature alpha158

# 训练 Transformer 模型
python -m quanttool.cli.main qlib train \
    --model transformer \
    --feature alpha158 \
    --epochs 100 \
    --hidden 128 \
    --layers 3
```

#### 步骤3：回测评估

```bash
# 使用训练好的模型进行回测
python -m quanttool.cli.main qlib backtest \
    --symbol 000001 \
    --model lgb \
    --feature alpha158
```

## Python API 使用

### 方式一：转换为 qlib 二进制格式

```python
from quanttool.infrastructure.data_providers import QlibDataConverter, QlibDataConfig

# 创建配置
config = QlibDataConfig(
    cache_dir=".cache/incremental_data",
    output_dir="qlib_data/cn_data",
    feature_type="alpha158",
)

# 创建转换器
converter = QlibDataConverter(config)

# 转换数据
result = converter.dump_data(
    symbols=["000001_SZ", "600519_SH"],
    start_date="2020-01-01",
    end_date="2024-12-31",
)

print(f"转换完成: {result['symbol_count']} 只股票")

# 使用 qlib 初始化
import qlib
qlib.init(provider_uri="qlib_data/cn_data")
```

### 方式二：直接创建 DatasetH

```python
from quanttool.infrastructure.data_providers import QlibDataConverter, QlibDataConfig

converter = QlibDataConverter(QlibDataConfig())

# 创建 qlib DatasetH
dataset = converter.create_qlib_dataset(
    symbols=["000001_SZ", "600519_SH"],
    start_date="2020-01-01",
    end_date="2024-12-31",
    feature_type="alpha158",
    label_type="return_10",
)

# 使用 qlib 官方模型训练
from qlib.contrib.model.gbdt import LGBModel

model = LGBModel()
model.fit(dataset)
```

### 方式三：使用训练流水线

```python
from quanttool.infrastructure.data_providers import (
    QlibDataConverter, QlibDataConfig, QlibTrainingPipeline
)

config = QlibDataConfig(feature_type="alpha158")
converter = QlibDataConverter(config)
pipeline = QlibTrainingPipeline(converter)

# 训练 GBDT 模型
result = pipeline.train_gbdt_model(
    symbols=["000001_SZ", "600519_SH"],
    model_type="lgb",
    n_estimators=200,
    max_depth=6,
)

# 训练 PyTorch 模型
result = pipeline.train_pytorch_model(
    symbols=["000001_SZ", "600519_SH"],
    model_type="lstm",
    epochs=100,
    hidden_size=64,
)

# 保存模型
result['model'].save("model_lgb.pkl")
```

## 特征工程

### Alpha158 特征

Alpha158 是 qlib 官方标准特征集，包含约 158 个特征：

- **K线特征** (30个): 动量、波动、相对位置
- **均线特征** (20个): MA5/10/20/30/60
- **EMA特征** (10个): 指数移动平均
- **MACD特征** (3个): DIF, DEA, HIST
- **RSI特征** (6个): RSI6/12/24
- **KDJ特征** (18个): K/D/J 值
- **布林带特征** (6个): 上轨、下轨、带宽
- **其他技术指标**: PSY, BIAS, ROC, ATR, VR 等

### Alpha360 特征

Alpha360 是扩展特征集，基于 60 天历史数据：

- 收益率历史 (60个)
- 相对高低点 (60个)
- 成交量比率 (60个)
- 波动率历史 (60个)
- 动量历史 (60个)

### 自定义特征

```python
from quanttool.infrastructure.data_providers import Alpha158Features
import pandas as pd

# 加载股票数据
df = pd.read_parquet(".cache/incremental_data/000001_SZ_stock_bar.parquet")

# 生成特征
features = Alpha158Features.generate(df)
print(f"特征数量: {len(features.columns)}")
```

## 数据格式说明

### Quanter 缓存格式

```
.cache/incremental_data/
├── 000001_SZ_stock_bar.parquet
├── 600519_SH_stock_bar.parquet
└── data_meta.db
```

每个 parquet 文件包含:
- `timestamp`: 日期时间
- `open`, `high`, `low`, `close`: OHLC
- `volume`, `amount`: 成交量和成交额

### qlib 官方格式

qlib 使用二进制格式存储数据，每个特征一个文件：

```
features/
├── 000001.SZ/
│   ├── close.bin    # 收盘价
│   ├── open.bin     # 开盘价
│   ├── $high.bin    # 最高价
│   ├── $low.bin     # 最低价
│   ├── $volume.bin  # 成交量
│   ├── KMID_5.bin   # Alpha158 特征
│   └── ...
```

## 常见问题

### Q: 为什么需要转换数据格式？

A: qlib 官方训练流程需要特定的数据格式。转换后可以：
1. 使用 qlib 的所有原生模型
2. 利用 qlib 的特征工程（Alpha158/Alpha360）
3. 与 qlib 社区的工作流兼容

### Q: Alpha158 和 Alpha360 选哪个？

A:
- **Alpha158**: 适合大多数场景，计算快，泛化性好
- **Alpha360**: 适合需要更多历史信息的场景，但可能过拟合

### Q: 如何选择模型？

A:
- **入门推荐**: LightGBM (lgb)，训练快，效果好
- **深度学习**: LSTM/GRU，适合时间序列模式
- **最佳性能**: Transformer，但需要更多数据和调参

## 参考链接

- [Qlib 官方文档](https://qlib.readthedocs.io/)
- [Qlib GitHub](https://github.com/microsoft/qlib)
- [Alpha158 论文](https://arxiv.org/abs/2011.09318)
