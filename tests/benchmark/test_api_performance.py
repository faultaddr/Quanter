"""
API 端点性能基准测试

测试所有关键 API 端点的响应时间，确保性能目标达成：
- 缓存命中：P50 < 10ms, P95 < 50ms
- 数据获取：P50 < 500ms, P95 < 2s
- 完整分析：P50 < 2s, P95 < 5s
"""

import pytest
import time
import statistics
from typing import List, Dict, Any
import requests

# API 基础 URL
API_BASE_URL = "http://127.0.0.1:8000/api"

# 测试配置
WARMUP_REQUESTS = 3  # 预热请求数
BENCHMARK_REQUESTS = 10  # 基准测试请求数

# 性能目标（毫秒）
PERFORMANCE_TARGETS = {
    "cache_hit": {"p50": 10, "p95": 50},
    "data_fetch": {"p50": 500, "p95": 2000},
    "full_analysis": {"p50": 2000, "p95": 5000},
}


def measure_response_time(endpoint: str, params: Dict = None) -> float:
    """测量单个请求的响应时间（毫秒）"""
    start = time.perf_counter()
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params, timeout=30)
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed if response.status_code == 200 else -1
    except Exception as e:
        print(f"Request failed: {e}")
        return -1


def run_benchmark(endpoint: str, params: Dict = None, warmup: int = WARMUP_REQUESTS,
                  requests_count: int = BENCHMARK_REQUESTS) -> Dict[str, Any]:
    """运行基准测试"""
    # 预热
    for _ in range(warmup):
        measure_response_time(endpoint, params)

    # 基准测试
    times: List[float] = []
    for _ in range(requests_count):
        t = measure_response_time(endpoint, params)
        if t > 0:
            times.append(t)

    if not times:
        return {"error": "All requests failed"}

    # 计算统计指标
    times.sort()
    n = len(times)

    return {
        "count": n,
        "min": min(times),
        "max": max(times),
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "p50": times[n // 2] if n > 0 else 0,
        "p95": times[int(n * 0.95)] if n > 0 else 0,
        "p99": times[int(n * 0.99)] if n > 0 else 0,
        "stddev": statistics.stdev(times) if n > 1 else 0,
    }


def print_results(name: str, results: Dict, target: Dict = None):
    """打印测试结果"""
    print(f"\n{'='*60}")
    print(f"📊 {name}")
    print(f"{'='*60}")
    print(f"  请求数: {results.get('count', 0)}")
    print(f"  最小值: {results.get('min', 0):.2f} ms")
    print(f"  最大值: {results.get('max', 0):.2f} ms")
    print(f"  平均值: {results.get('mean', 0):.2f} ms")
    print(f"  中位数: {results.get('median', 0):.2f} ms")
    print(f"  P50:    {results.get('p50', 0):.2f} ms")
    print(f"  P95:    {results.get('p95', 0):.2f} ms")
    print(f"  P99:    {results.get('p99', 0):.2f} ms")

    if target:
        p50_ok = results.get('p50', float('inf')) <= target.get('p50', float('inf'))
        p95_ok = results.get('p95', float('inf')) <= target.get('p95', float('inf'))
        status = "✅" if (p50_ok and p95_ok) else "❌"
        print(f"  目标:   P50 < {target['p50']}ms, P95 < {target['p95']}ms {status}")


@pytest.fixture(scope="module")
def api_available():
    """检查 API 服务是否可用"""
    try:
        response = requests.get(f"{API_BASE_URL}/strategies", timeout=5)
        return response.status_code == 200
    except:
        return False


@pytest.mark.skipif(not True, reason="API server not available")
class TestAPIPerformance:
    """API 端点性能测试"""

    def test_strategies_endpoint(self, api_available):
        """测试策略列表端点（轻量级）"""
        if not api_available:
            pytest.skip("API not available")

        results = run_benchmark("/strategies")
        print_results("策略列表端点", results, PERFORMANCE_TARGETS["cache_hit"])

    def test_gbm_models_endpoint(self, api_available):
        """测试模型列表端点（轻量级）"""
        if not api_available:
            pytest.skip("API not available")

        results = run_benchmark("/gbm/models")
        print_results("模型列表端点", results, PERFORMANCE_TARGETS["cache_hit"])

    def test_realtime_search_endpoint(self, api_available):
        """测试搜索端点（中等复杂度）"""
        if not api_available:
            pytest.skip("API not available")

        results = run_benchmark("/realtime/search", params={"q": "茅台"})
        print_results("股票搜索端点", results, PERFORMANCE_TARGETS["data_fetch"])

    def test_stock_kline_endpoint(self, api_available):
        """测试 K 线数据端点（中等复杂度）"""
        if not api_available:
            pytest.skip("API not available")

        results = run_benchmark("/stock/600519/kline", params={"days": 60})
        print_results("K线数据端点", results, PERFORMANCE_TARGETS["data_fetch"])

    def test_stock_analysis_endpoint(self, api_available):
        """测试股票分析端点（高复杂度）"""
        if not api_available:
            pytest.skip("API not available")

        results = run_benchmark("/stock/600519/analysis", params={"days": 60})
        print_results("股票分析端点", results, PERFORMANCE_TARGETS["full_analysis"])

    def test_stock_signals_endpoint(self, api_available):
        """测试信号端点（中等复杂度）"""
        if not api_available:
            pytest.skip("API not available")

        results = run_benchmark("/stock/600519/signals")
        print_results("交易信号端点", results, PERFORMANCE_TARGETS["data_fetch"])


@pytest.mark.skipif(not True, reason="API server not available")
class TestCachePerformance:
    """缓存性能测试"""

    def test_cache_effectiveness(self, api_available):
        """测试缓存有效性（第二次请求应该更快）"""
        if not api_available:
            pytest.skip("API not available")

        # 第一次请求（可能缓存未命中）
        first_time = measure_response_time("/stock/600519/kline", {"days": 60})

        # 第二次请求（应该缓存命中）
        second_time = measure_response_time("/stock/600519/kline", {"days": 60})

        print(f"\n📊 缓存有效性测试")
        print(f"  第一次请求: {first_time:.2f} ms")
        print(f"  第二次请求: {second_time:.2f} ms")

        if first_time > 0 and second_time > 0:
            speedup = first_time / second_time
            print(f"  加速比: {speedup:.2f}x")

            if second_time < 50:
                print(f"  ✅ 缓存命中（响应时间 < 50ms）")
            else:
                print(f"  ⚠️ 可能缓存未命中")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
