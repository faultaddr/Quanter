"""Shared helpers for QuantTool API routers."""

from typing import Any, Dict, Optional
import time

import numpy as np


_analysis_cache: Dict[str, tuple] = {}
_analysis_cache_ttl = 60


def get_cached_analysis(cache_key: str) -> Optional[Dict]:
    """Return a cached analysis payload if it is still fresh."""
    if cache_key in _analysis_cache:
        data, timestamp = _analysis_cache[cache_key]
        if time.time() - timestamp < _analysis_cache_ttl:
            return data
    return None


def set_cached_analysis(cache_key: str, data: Dict) -> None:
    """Cache an analysis payload and evict stale entries."""
    _analysis_cache[cache_key] = (data, time.time())
    current_time = time.time()
    expired_keys = [
        key
        for key, (_, timestamp) in _analysis_cache.items()
        if current_time - timestamp > _analysis_cache_ttl * 2
    ]
    for key in expired_keys:
        del _analysis_cache[key]


def to_python_types(obj: Any) -> Any:
    """Convert numpy values into JSON-friendly Python values."""
    if obj is None:
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: to_python_types(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python_types(item) for item in obj]
    return obj
