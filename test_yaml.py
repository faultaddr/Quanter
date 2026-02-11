#!/usr/bin/env python
"""
Test script to verify PyYAML installation
"""

try:
    import yaml
    print(f"✓ PyYAML imported successfully")
    print(f"✓ PyYAML version: {yaml.__version__}")
    print(f"✓ PyYAML module loaded from: {yaml.__file__}")

    # Test basic functionality
    test_data = {'test': 'data', 'number': 42}
    yaml_str = yaml.dump(test_data)
    parsed_data = yaml.safe_load(yaml_str)

    print(f"✓ YAML dump/load test passed: {parsed_data == test_data}")

    print("\n🎉 PyYAML installation verified successfully!")

except ImportError as e:
    print(f"✗ Error importing PyYAML: {e}")
    import sys
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import sys
    sys.exit(1)