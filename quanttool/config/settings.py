import os
from typing import Dict, Any
import yaml
from pathlib import Path


class Settings:
    """Settings manager for QuantTool"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._find_default_config()
        self._config = self._load_config()

    def _find_default_config(self) -> str:
        """Find default config file in standard locations"""
        possible_paths = [
            "config/default.yaml",
            "quanttool/config/default.yaml",
            str(Path.home() / ".quanttool" / "config.yaml"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # If no config exists, create from default
        default_config_content = """
# Default configuration for QuantTool
data:
  tushare_token: ${TUSHARE_TOKEN}
  ashare_config:
    endpoint: ""
    api_key: ""
  providers:
    - tushare
    - ashare
    - csv_mock

calendar:
  timezone: Asia/Shanghai
  trading_hours:
    morning_start: "09:30"
    morning_end: "11:30"
    afternoon_start: "13:00"
    afternoon_end: "15:00"
  holidays: []
  early_closes: []

backtest:
  initial_cash: 100000
  commission_rate: 0.0003
  min_commission: 5.0
  slippage_rate: 0.0001
  execution_model: next_close

trading:
  max_position_size: 0.1
  max_positions: 10

prediction:
  default_horizon: 6
  probability_threshold: 0.55

signal:
  cooldown_bars: 3

storage:
  data_dir: "./data"
  runs_dir: "./runs"
  reports_dir: "./reports"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "quanttool.log"
        """

        os.makedirs("quanttool/config", exist_ok=True)
        with open("quanttool/config/default.yaml", "w") as f:
            f.write(default_config_content.strip())

        return "quanttool/config/default.yaml"

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Replace environment variables in config
        self._resolve_env_vars(config)
        return config

    def _resolve_env_vars(self, obj):
        """Recursively resolve environment variables in config"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                obj[key] = self._resolve_env_vars(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                obj[i] = self._resolve_env_vars(item)
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]  # Remove ${ and }
            obj = os.getenv(env_var, "")
        return obj

    def get(self, key: str, default=None):
        """Get configuration value by dot notation key (e.g., 'data.tushare_token')"""
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def update(self, key: str, value: Any):
        """Update configuration value by dot notation key"""
        keys = key.split(".")
        config_ref = self._config

        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]

        config_ref[keys[-1]] = value

    @property
    def data_dir(self) -> str:
        return self.get("storage.data_dir", "./data")

    @property
    def runs_dir(self) -> str:
        return self.get("storage.runs_dir", "./runs")

    @property
    def reports_dir(self) -> str:
        return self.get("storage.reports_dir", "./reports")


# Global settings instance
settings = Settings()
