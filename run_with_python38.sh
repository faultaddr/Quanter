#!/bin/bash
# Script to run Quanter with Python 3.8 to avoid PyQLib compatibility issues

# Check if Python 3.8 is available
if ! command -v python3.8 &> /dev/null; then
    echo "❌ Python 3.8 not found!"
    echo "Please install Python 3.8 or higher to use PyQLib features"
    exit 1
fi

echo "✅ Using Python 3.8 to run Quanter"
echo "🔧 Activating PyQLib virtual environment..."

# Activate the PyQLib virtual environment
source pyqlib_env/bin/activate

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found in current directory"
    echo "Please run this script from the Quanter project root"
    deactivate
    exit 1
fi

# Run the application with Python 3.8
echo "🚀 Starting Quanter application..."
python3.8 main.py "$@"

# Deactivate virtual environment
deactivate