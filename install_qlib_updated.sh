#!/bin/bash

# 安装 Qlib 及相关依赖的脚本 - 更新版
echo "开始安装 Qlib 和项目依赖..."

# 首先升级 pip
python3 -m pip install --upgrade pip

# 安装 wheel（有助于构建包）
python3 -m pip install wheel

# 安装基础依赖（除了 Qlib）
echo "安装基础项目依赖..."
pip3 install pandas>=1.3.0 numpy>=1.20.0 matplotlib>=3.4.0 seaborn>=0.11.0 tushare>=1.2.0 baostock>=0.8.0 yfinance>=0.1.0 plotly>=4.0.0 dash>=2.0.0 scipy>=1.7.0 statsmodels>=0.12.0 requests>=2.25.0

# 尝试从 GitHub 安装 Qlib
echo "正在从 GitHub 安装 Qlib..."

# 安装 Git（如果尚未安装）
if ! command -v git &> /dev/null; then
    echo "需要先安装 Git"
    exit 1
fi

# 检查系统架构
ARCH=$(uname -m)
if [[ $ARCH == "arm64" ]]; then
    echo "检测到 ARM64 架构 (Apple Silicon)"
elif [[ $ARCH == "x86_64" ]]; then
    echo "检测到 x86_64 架构"
else
    echo "检测到架构: $ARCH"
fi

# 尝试安装 Qlib 通过官方推荐的方式
if python3 -c "import qlib" &> /dev/null; then
    echo "Qlib 已经安装"
else
    # 首先尝试安装 Qlib 的预构建包
    echo "尝试安装 pyqlib..."
    python3 -m pip install pyqlib || echo "pyqlib 安装失败"

    # 如果上面失败，尝试从官方 GitHub 安装
    if ! python3 -c "import qlib" &> /dev/null; then
        echo "尝试从 GitHub 安装 Qlib..."
        # 克隆 Qlib 仓库并安装
        if [ ! -d "qlib_repo" ]; then
            git clone https://github.com/microsoft/qlib.git qlib_repo
        fi

        cd qlib_repo
        python3 -m pip install -e .
        cd ..
    fi
fi

# 验证安装
if python3 -c "import qlib; print('✅ Qlib version:', qlib.__version__ if hasattr(qlib, '__version__') else 'unknown')" 2>/dev/null; then
    echo "✅ Qlib 安装成功！"
else
    echo "⚠️ Qlib 安装未完全成功，但我们将继续尝试配置环境"

    # 尝试仅安装 Qlib 的必要依赖
    echo "安装 Qlib 的依赖库..."
    python3 -m pip install pyqlib || echo "尝试替代安装方法..."
    python3 -m pip install --upgrade --force-reinstall pyqlib
fi

# 安装更多常用金融分析库
echo "安装其他有用的金融分析库..."
python3 -m pip install alpha-vantage yfinance ta-lib || echo "部分可选库安装失败（不影响主要功能）"

# 再次验证 Qlib
if python3 -c "import qlib" &> /dev/null; then
    echo "🎉 Qlib 已成功安装！"
    python3 -c "import qlib; print('Qlib 版本:', getattr(qlib, '__version__', 'unknown'))"
else
    echo "⚠️ Qlib 未能成功安装，但是我们已安装了大部分依赖项"
    echo "稍后您可能需要参考 Qlib 官方文档进行手动安装："
    echo "https://github.com/microsoft/qlib"
fi

echo "安装过程完成！"