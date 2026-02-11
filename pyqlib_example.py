#!/usr/bin/env python
"""
Example usage of pyqlib in the Quanter project
"""

def demonstrate_pyqlib_usage():
    """
    Demonstrates how to use pyqlib for quantitative finance tasks
    """
    print("Example of how to use pyqlib in your project:")
    print()

    example_code = '''
import qlib
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.config import REG_US, REG_CN

# Initialize Qlib
qlib.init(provider_uri="~/.qlib/stock_data/cn_data", region=REG_CN)

# Get stock data
instruments = D.instruments()
fields = ["$close", "$volume"]
start_time = "2020-01-01"
end_time = "2021-01-01"

data = D.features(instruments, fields, start_time, end_time)
print(data.head())
'''

    print(example_code)
    print()
    print("For more information on using pyqlib:")
    print("- Visit: https://github.com/microsoft/qlib")
    print("- Documentation: https://qlib.readthedocs.io/")
    print()
    print("🎉 pyqlib is now ready to use in your project!")

if __name__ == "__main__":
    demonstrate_pyqlib_usage()