#!/bin/bash

# 安装 Qlib 及相关依赖的脚本
echo "开始安装 Qlib 和项目依赖..."

# 首先升级 pip
python3 -m pip install --upgrade pip

# 检测操作系统并安装 OpenMP 库（如果需要）
OS_TYPE=$(uname -s)
if [[ "$OS_TYPE" == "Darwin" ]]; then
    echo "检测到 macOS 系统，检查 OpenMP 依赖..."

    # 检查是否已安装 libomp
    if command -v brew &> /dev/null; then
        if ! brew list libomp &> /dev/null; then
            echo "正在安装 libomp (OpenMP 库)..."
            brew install libomp
        else
            echo "✅ libomp 已安装"
        fi
    else
        # 检查 conda 环境
        if command -v conda &> /dev/null; then
            if ! conda list libopenmp &> /dev/null; then
                echo "正在通过 conda 安装 libopenmp..."
                conda install -c conda-forge libopenmp -y
            else
                echo "✅ libopenmp 已安装"
            fi
        else
            echo "⚠️ 未找到 Homebrew 或 Conda，无法自动安装 OpenMP 库"
            echo "💡 请手动安装 OpenMP，例如: brew install libomp"
        fi
    fi
elif [[ "$OS_TYPE" == "Linux" ]]; then
    echo "检测到 Linux 系统，检查 OpenMP 依赖..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y libomp-dev
    elif command -v yum &> /dev/null; then
        sudo yum install -y libgomp
    fi
fi

# 安装 wheel（有助于构建包）
python3 -m pip install wheel

# 安装 requirements.txt 中的所有依赖
echo "安装项目依赖..."
python3 -m pip install -r requirements.txt

# 如果安装失败，单独安装 Qlib
if ! python3 -c "import qlib"; then
    echo "Qlib 安装失败，尝试从源码安装依赖..."

    # 安装 Qlib 额外依赖
    python3 -m pip install pyqlib
    python3 -m pip install --upgrade setuptools

    # 尝试安装特定版本的 Qlib
    python3 -m pip install --no-cache-dir "qlib>=0.9.0"
fi

# 验证 Qlib 和 LightGBM 的安装
echo "验证 Qlib 和 LightGBM 安装..."
if python3 -c "import qlib; print('Qlib version:', qlib.__version__ if hasattr(qlib, '__version__') else 'unknown')" && \
   python3 -c "import lightgbm; print('LightGBM version:', lightgbm.__version__ if hasattr(lightgbm, '__version__') else 'unknown')"; then
    echo "✅ Qlib 和 LightGBM 安装成功！"
else
    echo "⚠️ Qlib 或 LightGBM 安装可能存在问题，但继续执行..."
fi

echo "安装完成！"