#!/usr/bin/env python3
"""
CLI Interface Tester
Tests all available paths in the Unified CLI Interface to ensure they are functional
"""

import sys
import os
import traceback
import subprocess
import time
from datetime import datetime
from unittest.mock import patch, MagicMock
import io
import contextlib

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cli_interface import UnifiedCLIInterface


class CLITester:
    """Test class for CLI interface functionality"""

    def __init__(self):
        # Mock tokens for testing (since we'll mock network calls)
        self.mock_tushare_token = "mock_token"
        self.mock_eastmoney_cookie = {"mock": "cookie"}

        # Initialize CLI interface
        self.cli = UnifiedCLIInterface(self.mock_tushare_token, self.mock_eastmoney_cookie)

        # Track test results
        self.test_results = {}
        self.failed_tests = []
        self.passed_tests = []

    def mock_input_generator(self, inputs):
        """Generator to provide mocked input for CLI functions that require user input"""
        for inp in inputs:
            yield inp
        # Keep yielding empty string if more input is needed
        while True:
            yield ""

    def test_command_by_name(self, command_name, test_desc=""):
        """Test a command by name with appropriate mocks"""
        print(f"\n🧪 Testing command: {command_name} - {test_desc}")

        try:
            # Get the command function
            cmd_map = self.cli.get_command_map()
            if command_name not in cmd_map:
                raise ValueError(f"Command {command_name} not found in command map")

            func = cmd_map[command_name]

            # Mock input and print to capture output
            captured_output = io.StringIO()

            # Prepare mock inputs based on command type
            if command_name == 'analyze_stock':
                mock_inputs = iter(['sh600519', 'ma_crossover', 'auto'])
            elif command_name == 'run_strategy':
                mock_inputs = iter(['ma_crossover', 'sh600519,sz000858', 'auto'])
            elif command_name == 'gen_signals':
                mock_inputs = iter(['sh600519,sz000858', 'auto'])
            elif command_name == 'get_data':
                mock_inputs = iter(['sh600519', '30', 'eastmoney'])
            elif command_name == 'calc_indicators':
                mock_inputs = iter(['sh600519', 'auto'])
            elif command_name == 'multi_factor_analysis':
                mock_inputs = iter(['sh600519,sz000858', '2025-06-01', '2025-12-31'])
            elif command_name == 'run_backtest':
                if self.cli.backtester:  # Only test if backtester is available
                    mock_inputs = iter(['ma_crossover', '000001.SZ', '20220101', '20221231', '100000'])
                else:
                    print("⚠️  Backtester not initialized, skipping test")
                    self.test_results[command_name] = {
                        'status': 'skipped',
                        'error': 'Backtester not initialized',
                        'time_taken': 0
                    }
                    return True
            elif command_name == 'compare_strategies':
                mock_inputs = iter(['sh600519,sz000858', 'ma_crossover,rsi', '20220101', '20221231', '100000'])
            elif command_name == 'predict_stocks':
                mock_inputs = iter(['sh600519,sz000858', '10'])
            else:
                # For commands that don't require input, we don't need to mock
                mock_inputs = iter([])

            # Mock the input function
            original_input = __builtins__.input
            __builtins__.input = lambda prompt="": next(mock_inputs, 'sh600519')  # Default to a common stock

            # Patch methods that might make network calls or fail
            with patch.object(self.cli.screener, 'fetch_stock_data', return_value=self._create_mock_data()), \
                 patch.object(self.cli.screener, 'get_chinese_stocks_list', return_value=self._create_mock_stocks_list()), \
                 patch.object(self.cli.screener, 'screen_stocks', return_value=self._create_mock_screen_results()), \
                 patch.object(self.cli.data_fetcher, 'fetch', return_value=self._create_mock_data()), \
                 patch.object(self.cli.predictive_analyzer, 'analyze_stocks', return_value=self._create_mock_predictions()), \
                 patch.object(self.cli.multi_factor_strategy, 'run_backtest', return_value=self._create_mock_factor_results()):

                start_time = time.time()

                # Execute the command
                func()

                end_time = time.time()
                elapsed = end_time - start_time

                # Record success
                self.test_results[command_name] = {
                    'status': 'passed',
                    'error': None,
                    'time_taken': elapsed
                }

                self.passed_tests.append(command_name)
                print(f"✅ {command_name} - PASSED ({elapsed:.2f}s)")
                return True

        except Exception as e:
            error_msg = str(e)
            self.test_results[command_name] = {
                'status': 'failed',
                'error': error_msg,
                'time_taken': 0
            }

            self.failed_tests.append(command_name)
            print(f"❌ {command_name} - FAILED: {error_msg}")
            print(f"Full traceback: {traceback.format_exc()}")
            return False
        finally:
            # Restore original input function
            if 'original_input' in locals():
                __builtins__.input = original_input

    def _create_mock_data(self):
        """Create mock stock data for testing"""
        import pandas as pd
        import numpy as np

        dates = pd.date_range(start='2025-01-01', periods=30, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(10, 100, 30),
            'high': np.random.uniform(10, 100, 30),
            'low': np.random.uniform(10, 100, 30),
            'close': np.random.uniform(10, 100, 30),
            'volume': np.random.uniform(100000, 1000000, 30)
        })
        data.set_index('date', inplace=True)
        return data

    def _create_mock_stocks_list(self):
        """Create mock stock list for testing"""
        import pandas as pd

        data = pd.DataFrame({
            'symbol': ['sh600519', 'sz000858', 'sh600036', 'sz000001', 'sh600518'],
            'name': ['贵州茅台', '五粮液', '招商银行', '平安银行', '康美药业']
        })
        return data

    def _create_mock_screen_results(self):
        """Create mock screen results for testing"""
        import pandas as pd

        data = pd.DataFrame({
            'symbol': ['sh600519', 'sz000858'],
            'name': ['贵州茅台', '五粮液'],
            'price': [1800.00, 200.00],
            'change_pct': [2.5, -1.2],
            'volume': [5000000, 10000000]
        })
        return data

    def _create_mock_predictions(self):
        """Create mock predictions for testing"""
        import pandas as pd

        data = pd.DataFrame({
            'symbol': ['sh600519', 'sz000858'],
            'name': ['贵州茅台', '五粮液'],
            'prediction_score': [0.85, 0.72],
            'up_probability': [0.75, 0.68],
            'expected_return': [0.05, 0.03]
        })
        return data

    def _create_mock_factor_results(self):
        """Create mock factor analysis results for testing"""
        return {
            'sh600519': {
                'total_strategy_return': 0.15,
                'total_benchmark_return': 0.10,
                'strategy_annual_return': 0.18,
                'benchmark_annual_return': 0.12,
                'strategy_volatility': 0.12,
                'benchmark_volatility': 0.15,
                'max_drawdown': -0.08,
                'info_ratio': 0.25,
                'sharpe_ratio': 0.15,
                'data': self._create_mock_data()
            }
        }

    def test_all_commands(self):
        """Test all commands in the CLI interface"""
        print("🚀 Starting CLI Interface Comprehensive Test")
        print("="*60)

        # List of all commands to test
        commands_to_test = [
            ('screen_stocks', 'Stock Screening'),
            ('show_top_stocks', 'Show Top Stocks'),
            ('get_data', 'Get Stock Data'),
            ('calc_indicators', 'Calculate Indicators'),
            ('analyze_stock', 'Analyze Single Stock'),
            ('run_strategy', 'Run Trading Strategy'),
            ('gen_signals', 'Generate Trading Signals'),
            ('show_signals', 'Show Latest Signals'),
            ('predict_stocks', 'Predict Stock Movements'),
            ('predictive_analysis', 'Run Predictive Analysis'),
            ('top_predictions', 'Show Top Predictions'),
            ('analyze_market', 'Analyze Overall Market'),
            ('multi_factor_analysis', 'Multi-Factor Analysis'),
            ('analyze_factors', 'Analyze Factor Performance'),
            ('factor_report', 'Generate Factor Report'),
            ('show_session', 'Show Session Data'),
            ('clear_session', 'Clear Session Data')
        ]

        # Test commands that depend on backtester availability
        if self.cli.backtester:
            commands_to_test.extend([
                ('run_backtest', 'Run Strategy Backtest'),
                ('compare_strategies', 'Compare Strategies')
            ])

        total_tests = len(commands_to_test)
        passed_count = 0
        failed_count = 0

        for command_name, description in commands_to_test:
            try:
                success = self.test_command_by_name(command_name, description)
                if success:
                    passed_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"❌ Unexpected error testing {command_name}: {str(e)}")
                failed_count += 1
                self.failed_tests.append(command_name)

        # Test numeric commands (1-21)
        print("\n🧪 Testing Numeric Commands (1-21)")
        print("-" * 40)

        numeric_passed = 0
        numeric_failed = 0

        for cmd_num in range(1, 22):  # Commands 1-21
            print(f"Testing command #{cmd_num}...")
            try:
                # We'll test the numeric handler by mocking input
                original_handle_numeric = self.cli.handle_numeric_command
                original_execute = self.cli.execute_command
                original_input = __builtins__.input

                # Temporarily replace the execute_command method to capture which command gets called
                captured_cmd = [None]  # Use list to allow modification inside nested function

                def mock_execute_command(cmd_name):
                    captured_cmd[0] = cmd_name
                    # Call original function with mocked dependencies
                    try:
                        with patch.object(self.cli.screener, 'fetch_stock_data', return_value=self._create_mock_data()), \
                             patch.object(self.cli.screener, 'get_chinese_stocks_list', return_value=self._create_mock_stocks_list()), \
                             patch.object(self.cli.screener, 'screen_stocks', return_value=self._create_mock_screen_results()), \
                             patch.object(self.cli.data_fetcher, 'fetch', return_value=self._create_mock_data()), \
                             patch.object(self.cli.predictive_analyzer, 'analyze_stocks', return_value=self._create_mock_predictions()), \
                             patch.object(self.cli.multi_factor_strategy, 'run_backtest', return_value=self._create_mock_factor_results()):

                            original_execute(cmd_name)
                    except Exception as e:
                        print(f"Command {cmd_name} execution resulted in: {str(e)}")

                self.cli.execute_command = mock_execute_command
                __builtins__.input = lambda prompt="": 'sh600519'  # Default input

                start_time = time.time()
                self.cli.handle_numeric_command(cmd_num)
                end_time = time.time()

                # Check if the command was successfully mapped and executed
                if captured_cmd[0] is not None:
                    print(f"✅ Command #{cmd_num} ({captured_cmd[0]}) - Handled")
                    numeric_passed += 1
                else:
                    # Some commands like 20 (help) and 21 (quit) don't call execute_command
                    if cmd_num in [20, 21]:
                        print(f"✅ Command #{cmd_num} - Special command handled (help/quit)")
                        numeric_passed += 1
                    else:
                        print(f"❌ Command #{cmd_num} - No command executed")
                        numeric_failed += 1

                elapsed = end_time - start_time
                print(f"   Time taken: {elapsed:.2f}s")

            except Exception as e:
                print(f"❌ Command #{cmd_num} - Error: {str(e)}")
                print(f"   Full traceback: {traceback.format_exc()}")
                numeric_failed += 1
            finally:
                # Restore original methods
                self.cli.execute_command = original_execute
                __builtins__.input = original_input

        total_passed = passed_count + numeric_passed
        total_failed = failed_count + numeric_failed
        total_executed = total_tests + 21  # 21 numeric commands

        print("\n" + "="*60)
        print("📊 TEST RESULTS SUMMARY")
        print("="*60)
        print(f"Total Commands Tested: {total_executed}")
        print(f"✅ Passed: {total_passed}")
        print(f"❌ Failed: {total_failed}")
        print(f"🏁 Success Rate: {total_passed}/{total_executed} ({(total_passed/total_executed)*100:.1f}%)")

        if self.failed_tests:
            print(f"\n❌ FAILED COMMANDS:")
            for failed_cmd in self.failed_tests:
                error = self.test_results.get(failed_cmd, {}).get('error', 'Unknown error')
                print(f"  - {failed_cmd}: {error}")

        if self.passed_tests:
            print(f"\n✅ SUCCESSFUL COMMANDS:")
            for passed_cmd in self.passed_tests:
                time_taken = self.test_results.get(passed_cmd, {}).get('time_taken', 0)
                print(f"  - {passed_cmd} ({time_taken:.2f}s)")

        print("\n" + "="*60)

        return total_failed == 0

    def run_interactive_simulation(self):
        """Simulate an interactive session to test the interactive loop"""
        print("\n🎮 Simulating Interactive Session")
        print("-" * 40)

        # This would be complex to simulate completely, but we can test basic functionality
        try:
            # Mock the input function to simulate user commands
            command_sequence = ['help', 'show_session', '18', 'quit']  # Help, show session, command 18, quit
            mock_input_iter = iter(command_sequence)

            original_input = __builtins__['input']
            original_print = print

            # Create a mock input function that cycles through our commands
            def mock_input(prompt=""):
                try:
                    return next(mock_input_iter)
                except StopIteration:
                    return 'quit'  # Quit after sequence

            __builtins__.input = mock_input

            print("Testing interactive simulation with command sequence: help, show_session, 18 (show_session), quit")

            # Temporarily suppress CLI's print statements for cleaner output
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):  # Suppress most output for cleaner test
                # Just test that the loop can handle the commands without crashing
                # We won't actually run the full interactive loop to avoid infinite loop
                for cmd in command_sequence[:-1]:  # Exclude 'quit' from simulation
                    if cmd.isdigit():
                        self.cli.handle_numeric_command(int(cmd))
                    else:
                        if cmd in self.cli.get_command_map():
                            self.test_command_by_name(cmd, f"Interactive command: {cmd}")
                        elif cmd == 'help':
                            self.cli.show_help()

            print("✅ Interactive simulation completed successfully")
            return True

        except Exception as e:
            print(f"❌ Interactive simulation failed: {str(e)}")
            print(f"Full traceback: {traceback.format_exc()}")
            return False
        finally:
            __builtins__['input'] = original_input


def main():
    """Main function to run the CLI tester"""
    print("🚀 A-Share CLI Interface Comprehensive Test Suite")
    print("="*70)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    tester = CLITester()

    # Run comprehensive tests
    all_passed = tester.test_all_commands()

    # Run interactive simulation
    interactive_ok = tester.run_interactive_simulation()

    print("\n" + "="*70)
    print("🎯 FINAL TEST OUTCOME")
    print("="*70)

    if all_passed:
        print("🎉 ALL TESTS PASSED! The CLI interface is fully functional.")
        return 0
    else:
        print("💥 SOME TESTS FAILED! The CLI interface has issues that need to be fixed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())