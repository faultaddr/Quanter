#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API Benchmark 测试框架

用于测试关键 API 端点的响应时间，识别性能瓶颈。

使用方法:
    python tests/benchmark_api.py
    python tests/benchmark_api.py --iterations 10 --base-url http://localhost:8000
"""

import time
import statistics
import argparse
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import json

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    exit(1)


@dataclass
class BenchmarkResult:
    """单个请求的结果"""
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    success: bool
    error: str = ""


@dataclass
class BenchmarkStats:
    """统计结果"""
    endpoint: str
    method: str
    iterations: int
    success_count: int
    failure_count: int
    avg_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


class APIBenchmark:
    """API Benchmark 测试器"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.results: List[BenchmarkResult] = []

    def request(
        self,
        method: str,
        endpoint: str,
        json_data: Any = None,
        params: Dict = None
    ) -> BenchmarkResult:
        """执行单个请求"""
        url = f"{self.base_url}{endpoint}"
        start = time.perf_counter()

        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, json=json_data, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")

            elapsed_ms = (time.perf_counter() - start) * 1000

            return BenchmarkResult(
                endpoint=endpoint,
                method=method,
                status_code=response.status_code,
                response_time_ms=elapsed_ms,
                success=response.status_code == 200
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return BenchmarkResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time_ms=elapsed_ms,
                success=False,
                error=str(e)
            )

    def benchmark(
        self,
        method: str,
        endpoint: str,
        iterations: int = 5,
        json_data: Any = None,
        params: Dict = None
    ) -> BenchmarkStats:
        """对单个端点进行多次测试"""
        times: List[float] = []
        success_count = 0
        failure_count = 0

        print(f"\n测试 {method} {endpoint} ({iterations} 次)...")

        for i in range(iterations):
            result = self.request(method, endpoint, json_data, params)
            self.results.append(result)

            if result.success:
                success_count += 1
                times.append(result.response_time_ms)
                print(f"  [{i+1}/{iterations}] {result.response_time_ms:.2f}ms - 成功")
            else:
                failure_count += 1
                print(f"  [{i+1}/{iterations}] 失败 - {result.error or f'HTTP {result.status_code}'}")

        if times:
            times.sort()
            stats = BenchmarkStats(
                endpoint=endpoint,
                method=method,
                iterations=iterations,
                success_count=success_count,
                failure_count=failure_count,
                avg_ms=statistics.mean(times),
                min_ms=times[0],
                max_ms=times[-1],
                p50_ms=times[len(times) // 2],
                p95_ms=times[int(len(times) * 0.95)] if len(times) >= 20 else times[-1],
                p99_ms=times[int(len(times) * 0.99)] if len(times) >= 100 else times[-1]
            )
        else:
            stats = BenchmarkStats(
                endpoint=endpoint,
                method=method,
                iterations=iterations,
                success_count=0,
                failure_count=iterations,
                avg_ms=0,
                min_ms=0,
                max_ms=0,
                p50_ms=0,
                p95_ms=0,
                p99_ms=0
            )

        return stats

    def run_all_benchmarks(self, iterations: int = 5) -> List[BenchmarkStats]:
        """运行所有预定义的 benchmark"""
        all_stats: List[BenchmarkStats] = []

        # ==================== 快速 API (目标 < 100ms) ====================
        print("\n" + "=" * 60)
        print("快速 API 测试 (目标 < 100ms)")
        print("=" * 60)

        # 策略列表
        stats = self.benchmark("GET", "/api/strategies", iterations)
        all_stats.append(stats)

        # 实时搜索（使用股票代码避免中文编码问题）
        stats = self.benchmark("GET", "/api/realtime/search", iterations, params={"query": "600519"})
        all_stats.append(stats)

        # ==================== 中等 API (目标 < 500ms) ====================
        print("\n" + "=" * 60)
        print("中等 API 测试 (目标 < 500ms)")
        print("=" * 60)

        # 实时行情
        stats = self.benchmark("GET", "/api/realtime/quote/600519", iterations)
        all_stats.append(stats)

        # 批量行情
        stats = self.benchmark(
            "POST", "/api/realtime/batch", iterations,
            json_data={"symbols": ["600519", "000001", "000858"]}
        )
        all_stats.append(stats)

        # ==================== 慢速 API (目标 < 3000ms) ====================
        print("\n" + "=" * 60)
        print("慢速 API 测试 (目标 < 3000ms)")
        print("=" * 60)

        # 股票分析
        stats = self.benchmark("GET", "/api/stock/600519/analysis", iterations, params={"days": 60})
        all_stats.append(stats)

        # K线数据
        stats = self.benchmark("GET", "/api/stock/600519/kline", iterations, params={"days": 60})
        all_stats.append(stats)

        # 交易信号
        stats = self.benchmark("GET", "/api/stock/600519/signals", iterations)
        all_stats.append(stats)

        # ==================== 模型 API ====================
        print("\n" + "=" * 60)
        print("模型 API 测试")
        print("=" * 60)

        # 模型列表
        stats = self.benchmark("GET", "/api/gbm/models", iterations)
        all_stats.append(stats)

        return all_stats

    def print_summary(self, stats_list: List[BenchmarkStats]):
        """打印汇总报告"""
        print("\n" + "=" * 60)
        print("性能汇总报告")
        print("=" * 60)
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"基准 URL: {self.base_url}")
        print()

        # 表头
        print(f"{'端点':<40} {'方法':<6} {'平均':<10} {'P50':<10} {'P95':<10} {'状态'}")
        print("-" * 90)

        # 排序: 按平均响应时间
        sorted_stats = sorted(stats_list, key=lambda x: x.avg_ms)

        for stats in sorted_stats:
            success_rate = stats.success_count / stats.iterations * 100
            status = "✓" if success_rate == 100 else f"{success_rate:.0f}%"

            # 根据响应时间标记
            if stats.avg_ms < 100:
                marker = "🟢"
            elif stats.avg_ms < 500:
                marker = "🟡"
            elif stats.avg_ms < 1000:
                marker = "🟠"
            else:
                marker = "🔴"

            endpoint_short = stats.endpoint[:38] + ".." if len(stats.endpoint) > 40 else stats.endpoint
            print(f"{endpoint_short:<40} {stats.method:<6} {stats.avg_ms:>6.1f}ms  {stats.p50_ms:>6.1f}ms  {stats.p95_ms:>6.1f}ms  {marker}{status}")

        print()
        print("图例: 🟢 <100ms  🟡 <500ms  🟠 <1000ms  🔴 >=1000ms")

    def export_json(self, stats_list: List[BenchmarkStats], filename: str = "benchmark_results.json"):
        """导出结果为 JSON"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
            "results": [
                {
                    "endpoint": s.endpoint,
                    "method": s.method,
                    "iterations": s.iterations,
                    "success_count": s.success_count,
                    "failure_count": s.failure_count,
                    "avg_ms": s.avg_ms,
                    "min_ms": s.min_ms,
                    "max_ms": s.max_ms,
                    "p50_ms": s.p50_ms,
                    "p95_ms": s.p95_ms,
                    "p99_ms": s.p99_ms
                }
                for s in stats_list
            ]
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n结果已导出到: {filename}")


def main():
    parser = argparse.ArgumentParser(description="API Benchmark 测试")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API 基础 URL")
    parser.add_argument("--iterations", type=int, default=5, help="每个端点测试次数")
    parser.add_argument("--export", default="", help="导出结果到 JSON 文件")

    args = parser.parse_args()

    benchmark = APIBenchmark(args.base_url)
    stats = benchmark.run_all_benchmarks(args.iterations)
    benchmark.print_summary(stats)

    if args.export:
        benchmark.export_json(stats, args.export)


if __name__ == "__main__":
    main()
