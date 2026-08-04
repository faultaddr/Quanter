"""Runtime safety policy tests for market-data providers."""

import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from quanttool.core.errors import ConfigurationError


class RuntimePolicyTests(unittest.TestCase):
    """Verify production-sensitive providers fail closed."""

    def test_runtime_defaults_to_development(self):
        from quanttool.core.runtime import RuntimeMode, get_runtime_mode

        self.assertEqual(get_runtime_mode({}), RuntimeMode.DEVELOPMENT)

    def test_runtime_rejects_unknown_value(self):
        from quanttool.core.runtime import get_runtime_mode

        with self.assertRaises(ConfigurationError):
            get_runtime_mode({"QUANTTOOL_ENV": "staging"})

    def test_csv_provider_requires_test_mode(self):
        from quanttool.infrastructure.data_providers.historical.csv_provider import (
            CSVProvider,
        )

        with patch.dict(os.environ, {"QUANTTOOL_ENV": "production"}, clear=False):
            with self.assertRaises(ConfigurationError):
                CSVProvider()

    def test_enhanced_fetcher_import_preserves_proxy_environment(self):
        code = """
import json, os
before = {
    key: os.environ.get(key)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY')
}
import quanttool.infrastructure.data_providers.historical.enhanced_fetcher
after = {key: os.environ.get(key) for key in before}
print(json.dumps({'before': before, 'after': after}, sort_keys=True))
"""
        env = os.environ.copy()
        env.update(
            {
                "HTTP_PROXY": "http://127.0.0.1:18080",
                "HTTPS_PROXY": "http://127.0.0.1:18443",
                "ALL_PROXY": "socks5://127.0.0.1:1080",
                "NO_PROXY": "localhost",
            }
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["before"], payload["after"])

    def test_fetcher_factory_reads_optional_credentials_from_environment(self):
        from quanttool.infrastructure.data_providers.historical import enhanced_fetcher

        with patch.dict(
            os.environ,
            {
                "TUSHARE_TOKEN": "token-from-runtime",
                "EASTMONEY_COOKIE": "cookie-from-runtime",
            },
            clear=False,
        ), patch.object(enhanced_fetcher, "EnhancedDataFetcher") as fetcher_type:
            enhanced_fetcher.create_data_fetcher_with_credentials()

        fetcher_type.assert_called_once_with(
            tushare_token="token-from-runtime",
            eastmoney_cookie="cookie-from-runtime",
        )


if __name__ == "__main__":
    unittest.main()
