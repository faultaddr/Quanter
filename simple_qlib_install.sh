#!/bin/bash

echo "开始安装 Qlib..."

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

# 更新 pip
python3 -m pip install --upgrade pip

# 安装 Qlib 的依赖（避免版本冲突）
echo "安装基础依赖..."
python3 -m pip install numpy pandas scipy matplotlib scikit-learn

# 安装 pyqlib（Microsoft Qlib 的简化版本）
echo "安装 pyqlib..."
python3 -m pip install pyqlib

# 检查是否安装成功
if python3 -c "import pyqlib as qlib; print('✅ pyqlib 安装成功!')" 2>/dev/null; then
    echo "✅ pyqlib 成功安装！"

    # 测试基本功能
    python3 -c "
try:
    import pyqlib as qlib
    print('Qlib 版本:', getattr(qlib, '__version__', 'unknown'))
    print('✅ Qlib 导入测试成功')
except ImportError as e:
    print('❌ 导入测试失败:', str(e))
    "
elif python3 -c "import qlib; print('✅ Qlib 安装成功!')" 2>/dev/null; then
    echo "✅ Qlib 成功安装！"

    # 测试基本功能
    python3 -c "
try:
    import qlib
    print('Qlib 版本:', getattr(qlib, '__version__', 'unknown'))
    print('✅ Qlib 导入测试成功')
except ImportError as e:
    print('❌ 导入测试失败:', str(e))
    "
else
    echo "⚠️ pyqlib 安装未完全成功"
    echo "尝试从 GitHub 安装完整版 Qlib..."

    # 克隆 Qlib 仓库
    if [ ! -d "qlib_repo" ]; then
        echo "克隆 Qlib 仓库..."
        git clone https://github.com/microsoft/qlib.git qlib_repo
    fi

    cd qlib_repo
    python3 -m pip install -e .
    cd ..

    # 再次测试
    if python3 -c "import qlib; print('✅ Qlib 安装成功!')" 2>/dev/null; then
        echo "✅ 从 GitHub 安装的 Qlib 成功！"
    else
        echo "❌ Qlib 安装失败"
        echo "您可以稍后按照 Qlib 官方文档手动安装："
        echo "https://github.com/microsoft/qlib/blob/main/README.md"
    fi
fi

# 额外测试 LightGBM 是否可以导入
if python3 -c "import lightgbm; print('✅ LightGBM 可用')" 2>/dev/null; then
    echo "✅ LightGBM 和相关 ML 模型可用"
else
    echo "⚠️ LightGBM 可能存在问题，请确保已安装 OpenMP 库"
fi

echo "Qlib 安装流程完成！"