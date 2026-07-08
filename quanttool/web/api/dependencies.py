"""Lazy factories shared by API routers."""

from typing import Dict
import time

from ...core.logging import get_logger


logger = get_logger(__name__)

_circuit_breaker: Dict[str, float] = {}
_CIRCUIT_BREAKER_TIMEOUT = 300


def _is_circuit_open(service_name: str) -> bool:
    """Return whether a service should be skipped by the circuit breaker."""
    if service_name in _circuit_breaker:
        failure_time = _circuit_breaker[service_name]
        if time.time() - failure_time < _CIRCUIT_BREAKER_TIMEOUT:
            return True
        del _circuit_breaker[service_name]
    return False


def _record_failure(service_name: str) -> None:
    """Record a service failure in the circuit breaker."""
    _circuit_breaker[service_name] = time.time()
    logger.warning("Circuit breaker opened for %s", service_name)


def get_minute_provider():
    """Get the legacy AkShare minute data provider with lazy initialization."""
    from ...infrastructure.data_providers.akshare_minute_provider import (
        AkShareMinuteProvider,
    )

    if _is_circuit_open("akshare_minute"):
        raise RuntimeError("AkShare minute provider is circuit-broken")

    if not hasattr(get_minute_provider, "_instance"):
        try:
            get_minute_provider._instance = AkShareMinuteProvider()
            get_minute_provider._instance.initialize()
        except Exception:
            _record_failure("akshare_minute")
            raise
    return get_minute_provider._instance


def get_realtime_provider():
    """Get the unified realtime data provider with lazy initialization."""
    from ...infrastructure.data_providers.realtime.realtime_provider import (
        get_realtime_provider as _get_provider,
    )

    return _get_provider()


def get_incremental_minute_provider():
    """Get the incremental minute data provider with lazy initialization."""
    from ...infrastructure.data_providers.incremental.minute_provider import (
        get_incremental_minute_provider as _get_provider,
    )

    return _get_provider()
