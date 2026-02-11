#!/usr/bin/env python
"""
Test script to verify pyqlib installation and functionality
"""

try:
    import qlib
    print("✓ pyqlib imported successfully")
    print(f"✓ pyqlib version: {qlib.__version__}")

    # Try importing important modules
    from qlib.data import D
    from qlib.config import REG_CN
    print("✓ Key modules imported successfully")

    # Show basic info about pyqlib
    print(f"✓ Available datasets: {dir(D) if hasattr(qlib.data, 'D') else 'Not available'}")

    print("\n🎉 pyqlib installation verified successfully!")

except ImportError as e:
    print(f"✗ Error importing pyqlib: {e}")
    import sys
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import sys
    sys.exit(1)