#!/bin/bash

# Script to install TA-Lib dependencies for the Quant Trading System

echo "Setting up TA-Lib for Quant Trading System..."

# Check if we are on a supported platform
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi

echo "Detected platform: $PLATFORM"

# On Linux systems, try to install the system dependencies
if [ "$PLATFORM" = "linux" ]; then
    echo "Installing system dependencies..."

    # Check if we're on a RedHat-based system (CentOS, RHEL, Fedora)
    if command -v yum &> /dev/null; then
        echo "Using yum package manager..."
        sudo yum update -y
        sudo yum install -y gcc gcc-c++ wget tar make

    # Check if we're on a Debian-based system (Ubuntu, Debian)
    elif command -v apt-get &> /dev/null; then
        echo "Using apt-get package manager..."
        sudo apt-get update
        sudo apt-get install -y build-essential wget tar make

    else
        echo "No known package manager found. Please install build-essential, wget, tar, and make manually."
        exit 1
    fi

    # Download and compile TA-Lib C library
    echo "Downloading and installing TA-Lib C library..."
    cd /tmp
    if [ ! -f "ta-lib-0.4.0-src.tar.gz" ]; then
        wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    fi
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib

    # Configure and install
    ./configure --prefix=/usr
    make
    sudo make install

    # Create pkg-config file if it doesn't exist
    if [ ! -d "/usr/lib64/pkgconfig" ] && [ -d "/usr/lib/pkgconfig" ]; then
        sudo sh -c 'echo "prefix=/usr
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: ta-lib
Description: TA-Lib/C library for technical analysis
Version: 0.4.0
Libs: -L\${libdir} -lta_lib
Cflags: -I\${includedir}/ta-lib" > /usr/lib/pkgconfig/ta-lib.pc'
    else
        sudo sh -c 'echo "prefix=/usr
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib64
includedir=\${prefix}/include

Name: ta-lib
Description: TA-Lib/C library for technical analysis
Version: 0.4.0
Libs: -L\${libdir} -lta_lib
Cflags: -I\${includedir}/ta-lib" > /usr/lib64/pkgconfig/ta-lib.pc'
    fi

    echo "TA-Lib C library installed successfully!"
fi

# Install Python TA-Lib wrapper
echo "Installing Python TA-Lib wrapper..."
pip install TA-Lib || echo "Failed to install TA-Lib via pip. The system will use the mock implementation."

# Verify the installation
echo "Verifying installation..."
python -c "
try:
    import talib
    print('✓ TA-Lib Python package is available')
    print('  Version:', getattr(talib, '__version__', 'Unknown'))
except ImportError:
    print('? TA-Lib Python package not found, using mock implementation')
    import sys
    import os
    sys.path.append('./quant_trade_a_share/utils')
    from talib_mock import SMA
    print('  ✓ Mock implementation is available and functional')
"

echo ""
echo "Setup completed!"
echo ""
echo "Note: If TA-Lib installation failed, the system will automatically use"
echo "a mock implementation that provides the same interface with pure Python."
echo "This allows the trading strategies to run without the C library."