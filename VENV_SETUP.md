# 虚拟环境设置说明

## 环境信息
- 虚拟环境名称：`quant_trade_env`
- 位置：项目根目录下
- 已安装包：pandas, numpy, matplotlib, seaborn, tushare, baostock, yfinance, plotly, dash, scipy, statsmodels, requests 等

## 如何激活虚拟环境

### 激活环境
```bash
source quant_trade_env/bin/activate
```

### 在环境中运行Python脚本
```bash
source quant_trade_env/bin/activate && python your_script.py
```

### 退出虚拟环境
```bash
deactivate
```

## 关于 Qlib 包

在尝试安装 `qlib>=0.9.0` 时遇到问题，原因可能是：
1. macOS ARM64 架构缺少预编译的二进制文件
2. 版本兼容性问题

如果需要安装 Qlib，可以尝试以下方法：

### 方法1：使用conda安装（推荐）
```bash
# 首先确保已安装conda/miniconda/anaconda
conda create -n qlib_env python=3.8
conda activate qlib_env
pip install --upgrade setuptools wheel
pip install --upgrade pyqlib
```

### 方法2：从源码编译安装
```bash
source quant_trade_env/bin/activate
pip install --upgrade setuptools wheel
git clone https://github.com/microsoft/qlib.git
cd qlib
pip install .
```

注意：从源码编译可能需要较长时间，并且需要系统安装有编译工具链。

## 环境验证

你可以运行以下命令来验证环境是否正常工作：
```bash
source quant_trade_env/bin/activate && python -c "import pandas as pd; import numpy as np; print('Environment is working correctly!')"
```

## 项目使用建议

对于当前项目，大部分所需依赖已经成功安装，可以正常使用以下功能：
- 数据处理：pandas, numpy
- 可视化：matplotlib, seaborn, plotly
- 金融数据获取：tushare, baostock, yfinance
- 统计分析：scipy, statsmodels
- Web应用：dash