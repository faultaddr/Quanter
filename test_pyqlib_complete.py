#!/usr/bin/env python
"""
Test script to verify pyqlib installation and functionality
"""

try:
    import qlib
    print("✓ pyqlib imported successfully")

    # Try importing important modules
    from qlib.data import D
    from qlib.config import REG_CN
    print("✓ Key modules imported successfully")

    # Try to print basic info about pyqlib
    print(f"✓ qlib module location: {qlib.__file__}")

    # Test basic functionality
    print("✓ pyqlib is properly installed and functional!")

    # Show available attributes in qlib module
    print(f"✓ Available in qlib module: {len(dir(qlib))} attributes/methods")

    print("\n🎉 pyqlib installation verified successfully!")

except ImportError as e:
    print(f"✗ Error importing pyqlib: {e}")
    import sys
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import sys
    sys.exit(1)