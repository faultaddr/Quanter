# Qlib Qrun 配置文件使用说明

本目录包含 qlib `qrun` 命令可用的配置文件，用于训练 GBM 模型并进行回测。

## 配置文件说明

| 文件 | 股票池 | 说明 |
|------|--------|------|
| `workflow_lgb.yaml` | 全市场 | 使用全市场股票训练 LightGBM 模型 |
| `workflow_csi300.yaml` | 沪深300 | 仅使用沪深300成分股训练 |

## 使用方法

### 基本用法

```bash
# 进入配置目录
cd /Users/missy/PROJ/NEW_Quanter/Quanter/qlib_config

# 运行全市场配置
qrun workflow_lgb.yaml

# 运行沪深300配置
qrun workflow_csi300.yaml
```

### 指定实验名称

```bash
qrun workflow_lgb.yaml --experiment_name my_experiment
```

### 指定输出目录

```bash
qrun workflow_lgb.yaml --output_dir ./results
```

## 配置参数说明

### 时间配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| train_start | 2018-01-01 | 训练开始日期 |
| train_end | 2022-12-31 | 训练结束日期 |
| valid_start | 2023-01-01 | 验证开始日期 |
| valid_end | 2023-12-31 | 验证结束日期 |
| test_start | 2024-01-01 | 测试开始日期 |
| test_end | 2024-12-31 | 测试结束日期 |

### 模型参数

主要 LightGBM 参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| learning_rate | 0.05 | 学习率 |
| n_estimators | 500 | 树的数量 |
| max_depth | 6-8 | 树的最大深度 |
| num_leaves | 63-127 | 叶子节点数量 |
| colsample_bytree | 0.85 | 特征采样比例 |
| subsample | 0.85 | 样本采样比例 |

### 策略参数

| 参数 | workflow_lgb | workflow_csi300 | 说明 |
|------|--------------|-----------------|------|
| topk | 30 | 10 | 持仓股票数量 |
| n_drop | 5 | 2 | 每次换仓数量 |

## 数据要求

配置文件使用 `~/.qlib/qlib_data/cn_data` 路径下的 qlib 标准数据格式。

如果数据不存在，需要先下载：

```bash
# 下载 qlib 数据
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

## 输出结果

qrun 运行后会输出：

1. **模型文件**: 保存训练好的模型
2. **预测结果**: 测试集的预测值
3. **回测指标**:
   - IC (Information Coefficient)
   - Rank IC
   - 年化收益
   - 夏普比率
   - 最大回撤

## 与现有脚本的对比

| 特性 | qrun 配置 | Python 脚本 (gbm_strategy.py) |
|------|----------|-------------------------------|
| 特征工程 | Alpha158 (qlib 原生) | 自定义特征 |
| 训练管道 | qlib 原生 | 自定义实现 |
| 回测框架 | qlib 原生 | 自定义回测引擎 |
| 灵活性 | 配置驱动 | 代码驱动 |
| 适用场景 | 标准化实验 | 高度定制化 |

## 故障排除

### 1. 数据路径错误

确保 qlib 数据已正确下载到 `~/.qlib/qlib_data/cn_data`。

### 2. 内存不足

减少 `topk` 或缩短训练时间范围。

### 3. 模型不收敛

调整学习率 `learning_rate` 或增加 `n_estimators`。

## 参考

- [Qlib 官方文档](https://qlib.readthedocs.io/)
- [Qlib GitHub](https://github.com/microsoft/qlib)
- [Alpha158 特征说明](https://qlib.readthedocs.io/en/latest/component/data.html#alpha158)
