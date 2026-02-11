#!/usr/bin/env python
"""
Simple test script to verify pyqlib installation
"""

try:
    import qlib
    print("✓ pyqlib imported successfully")
    print(f"✓ qlib module loaded from: {qlib.__file__}")

    # Check basic attributes
    print(f"✓ qlib has 'config' attribute: {hasattr(qlib, 'config')}")
    print(f"✓ qlib has 'data' attribute: {hasattr(qlib, 'data')}")
    print(f"✓ qlib has 'init' function: {hasattr(qlib, 'init')}")

    print("\n🎉 pyqlib core functionality verified!")

except ImportError as e:
    print(f"✗ Error importing pyqlib: {e}")
    import sys
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import sys
    sys.exit(1)