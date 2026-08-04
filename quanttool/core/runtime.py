"""Runtime-mode policy for production-sensitive QuantTool components."""

from enum import Enum
import os
from typing import Mapping, Optional

from .errors import ConfigurationError


class RuntimeMode(str, Enum):
    """Supported QuantTool runtime modes."""

    TEST = "test"
    DEVELOPMENT = "development"
    PRODUCTION = "production"


def get_runtime_mode(
    env: Optional[Mapping[str, str]] = None,
) -> RuntimeMode:
    """Resolve the configured runtime mode, rejecting unknown values."""
    values = os.environ if env is None else env
    raw = values.get(
        "QUANTTOOL_ENV",
        RuntimeMode.DEVELOPMENT.value,
    ).strip().lower()
    try:
        return RuntimeMode(raw)
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in RuntimeMode)
        raise ConfigurationError(
            f"Invalid QUANTTOOL_ENV={raw!r}; expected one of: {allowed}"
        ) from exc


def require_test_mode(
    feature: str,
    env: Optional[Mapping[str, str]] = None,
) -> None:
    """Reject use of a test-only feature outside explicit test mode."""
    mode = get_runtime_mode(env)
    if mode is not RuntimeMode.TEST:
        raise ConfigurationError(
            f"{feature} is test-only and cannot run in {mode.value} mode"
        )
