"""
因子注册与版本管理模块

提供因子注册、版本追踪、元数据管理功能：
- 因子元数据存储
- 版本管理
- 因子有效性记录
- 因子依赖管理
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
import pandas as pd
import json
from pathlib import Path


class FactorCategory(str, Enum):
    """因子类别"""
    TECHNICAL = "technical"           # 技术因子
    FUNDAMENTAL = "fundamental"      # 基本面因子
    QUALITY = "quality"               # 质量因子
    VALUE = "value"                  # 价值因子
    MOMENTUM = "momentum"            # 动量因子
    VOLATILITY = "volatility"        # 波动率因子
    LIQUIDITY = "liquidity"          # 流动性因子
    SENTIMENT = "sentiment"          # 情绪因子
    CUSTOM = "custom"                # 自定义因子


class FactorStatus(str, Enum):
    """因子状态"""
    ACTIVE = "active"                # 活跃
    INACTIVE = "inactive"            # 非活跃
    DEPRECATED = "deprecated"         # 已弃用
    TESTING = "testing"              # 测试中


@dataclass
class FactorMetadata:
    """因子元数据"""
    name: str
    category: FactorCategory
    description: str
    created_at: datetime
    updated_at: datetime
    version: str
    author: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class FactorPerformance:
    """因子表现数据"""
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ir: float = 0.0
    long_short_return: float = 0.0
    turnover: float = 0.0
    last_evaluated: Optional[datetime] = None


@dataclass
class RegisteredFactor:
    """注册的因子"""
    metadata: FactorMetadata
    performance: FactorPerformance
    status: FactorStatus
    compute_func: Optional[Callable] = None  # 因子计算函数


class FactorRegistry:
    """
    因子注册表

    管理因子的注册、版本和元数据
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        初始化因子注册表

        Args:
            storage_path: 存储路径（可选）
        """
        self.storage_path = storage_path
        self._factors: Dict[str, RegisteredFactor] = {}
        self._version_history: Dict[str, List[FactorMetadata]] = {}

    def register(
        self,
        name: str,
        category: FactorCategory,
        description: str,
        compute_func: Optional[Callable] = None,
        parameters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        author: str = "",
        version: str = "1.0.0",
    ) -> RegisteredFactor:
        """
        注册因子

        Args:
            name: 因子名称
            category: 因子类别
            description: 因子描述
            compute_func: 因子计算函数
            parameters: 参数
            dependencies: 依赖的其他因子
            tags: 标签
            author: 作者
            version: 版本号

        Returns:
            RegisteredFactor: 注册的因子
        """
        now = datetime.now()

        metadata = FactorMetadata(
            name=name,
            category=category,
            description=description,
            created_at=now,
            updated_at=now,
            version=version,
            author=author,
            parameters=parameters or {},
            dependencies=dependencies or [],
            tags=tags or [],
        )

        registered_factor = RegisteredFactor(
            metadata=metadata,
            performance=FactorPerformance(),
            status=FactorStatus.ACTIVE,
            compute_func=compute_func,
        )

        # 存储因子
        self._factors[name] = registered_factor

        # 记录版本历史
        if name not in self._version_history:
            self._version_history[name] = []
        self._version_history[name].append(metadata)

        return registered_factor

    def unregister(self, name: str) -> bool:
        """
        注销因子

        Args:
            name: 因子名称

        Returns:
            是否成功
        """
        if name in self._factors:
            self._factors[name].status = FactorStatus.DEPRECATED
            return True
        return False

    def update(
        self,
        name: str,
        compute_func: Optional[Callable] = None,
        parameters: Optional[Dict[str, Any]] = None,
        status: Optional[FactorStatus] = None,
    ) -> bool:
        """
        更新因子

        Args:
            name: 因子名称
            compute_func: 新的计算函数
            parameters: 新的参数
            status: 新的状态

        Returns:
            是否成功
        """
        if name not in self._factors:
            return False

        factor = self._factors[name]

        # 更新元数据
        factor.metadata.updated_at = datetime.now()

        # 更新版本号
        old_version = factor.metadata.version
        major, minor, patch = map(int, old_version.split("."))
        factor.metadata.version = f"{major}.{minor}.{patch + 1}"

        # 更新参数
        if parameters:
            factor.metadata.parameters.update(parameters)

        # 更新计算函数
        if compute_func is not None:
            factor.compute_func = compute_func

        # 更新状态
        if status is not None:
            factor.status = status

        # 记录版本历史
        self._version_history[name].append(factor.metadata)

        return True

    def get(self, name: str) -> Optional[RegisteredFactor]:
        """
        获取因子

        Args:
            name: 因子名称

        Returns:
            RegisteredFactor: 注册的因子（如果存在）
        """
        return self._factors.get(name)

    def list_factors(
        self,
        category: Optional[FactorCategory] = None,
        status: Optional[FactorStatus] = None,
        tags: Optional[List[str]] = None,
    ) -> List[RegisteredFactor]:
        """
        列出因子

        Args:
            category: 因子类别过滤
            status: 状态过滤
            tags: 标签过滤

        Returns:
            符合条件的因子列表
        """
        result = list(self._factors.values())

        if category is not None:
            result = [f for f in result if f.metadata.category == category]

        if status is not None:
            result = [f for f in result if f.status == status]

        if tags:
            result = [f for f in result if any(t in f.metadata.tags for t in tags)]

        return result

    def update_performance(
        self,
        name: str,
        ic_mean: float,
        ic_std: float,
        ir: float,
        long_short_return: float,
        turnover: float = 0.0,
    ) -> bool:
        """
        更新因子表现

        Args:
            name: 因子名称
            ic_mean: 平均IC
            ic_std: IC标准差
            ir: 信息比率
            long_short_return: 多空收益
            turnover: 换手率

        Returns:
            是否成功
        """
        if name not in self._factors:
            return False

        factor = self._factors[name]
        factor.performance = FactorPerformance(
            ic_mean=ic_mean,
            ic_std=ic_std,
            ir=ir,
            long_short_return=long_short_return,
            turnover=turnover,
            last_evaluated=datetime.now(),
        )

        return True

    def get_effective_factors(self, min_ir: float = 0.3) -> List[str]:
        """
        获取有效因子

        Args:
            min_ir: 最小IR阈值

        Returns:
            有效因子名称列表
        """
        effective = []
        for name, factor in self._factors.items():
            if factor.status == FactorStatus.ACTIVE and factor.performance.ir >= min_ir:
                effective.append(name)
        return effective

    def get_factor_dependencies(self, name: str) -> List[str]:
        """
        获取因子依赖

        Args:
            name: 因子名称

        Returns:
            依赖列表
        """
        if name not in self._factors:
            return []
        return self._factors[name].metadata.dependencies

    def get_top_factors(self, n: int = 10, by: str = "ir") -> List[Tuple[str, float]]:
        """
        获取最佳因子

        Args:
            n: 返回数量
            by: 排序依据 (ir, ic, return)

        Returns:
            [(因子名, 分数), ...]
        """
        factors_with_score = []

        for name, factor in self._factors.items():
            if factor.status != FactorStatus.ACTIVE:
                continue

            if by == "ir":
                score = factor.performance.ir
            elif by == "ic":
                score = factor.performance.ic_mean
            elif by == "return":
                score = factor.performance.long_short_return
            else:
                score = 0.0

            factors_with_score.append((name, score))

        # 排序并返回
        return sorted(factors_with_score, key=lambda x: x[1], reverse=True)[:n]

    def save(self, path: Optional[str] = None) -> str:
        """
        保存注册表到文件

        Args:
            path: 文件路径

        Returns:
            保存的路径
        """
        save_path = path or self.storage_path or "factor_registry.json"

        data = {}
        for name, factor in self._factors.items():
            data[name] = {
                "metadata": {
                    "name": factor.metadata.name,
                    "category": factor.metadata.category.value,
                    "description": factor.metadata.description,
                    "created_at": factor.metadata.created_at.isoformat(),
                    "updated_at": factor.metadata.updated_at.isoformat(),
                    "version": factor.metadata.version,
                    "author": factor.metadata.author,
                    "parameters": factor.metadata.parameters,
                    "dependencies": factor.metadata.dependencies,
                    "tags": factor.metadata.tags,
                },
                "performance": {
                    "ic_mean": factor.performance.ic_mean,
                    "ic_std": factor.performance.ic_std,
                    "ir": factor.performance.ir,
                    "long_short_return": factor.performance.long_short_return,
                    "turnover": factor.performance.turnover,
                    "last_evaluated": factor.performance.last_evaluated.isoformat()
                    if factor.performance.last_evaluated
                    else None,
                },
                "status": factor.status.value,
            }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return save_path

    def load(self, path: str) -> int:
        """
        从文件加载注册表

        Args:
            path: 文件路径

        Returns:
            加载的因子数量
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for name, factor_data in data.items():
            try:
                metadata = FactorMetadata(
                    name=factor_data["metadata"]["name"],
                    category=FactorCategory(factor_data["metadata"]["category"]),
                    description=factor_data["metadata"]["description"],
                    created_at=datetime.fromisoformat(factor_data["metadata"]["created_at"]),
                    updated_at=datetime.fromisoformat(factor_data["metadata"]["updated_at"]),
                    version=factor_data["metadata"]["version"],
                    author=factor_data["metadata"].get("author", ""),
                    parameters=factor_data["metadata"].get("parameters", {}),
                    dependencies=factor_data["metadata"].get("dependencies", []),
                    tags=factor_data["metadata"].get("tags", []),
                )

                perf_data = factor_data.get("performance", {})
                performance = FactorPerformance(
                    ic_mean=perf_data.get("ic_mean", 0.0),
                    ic_std=perf_data.get("ic_std", 0.0),
                    ir=perf_data.get("ir", 0.0),
                    long_short_return=perf_data.get("long_short_return", 0.0),
                    turnover=perf_data.get("turnover", 0.0),
                    last_evaluated=datetime.fromisoformat(perf_data["last_evaluated"])
                    if perf_data.get("last_evaluated")
                    else None,
                )

                registered_factor = RegisteredFactor(
                    metadata=metadata,
                    performance=performance,
                    status=FactorStatus(factor_data.get("status", "active")),
                )

                self._factors[name] = registered_factor
                count += 1

            except Exception:
                continue

        return count


# 全局注册表实例
_global_registry: Optional[FactorRegistry] = None


def get_registry() -> FactorRegistry:
    """获取全局因子注册表"""
    global _global_registry
    if _global_registry is None:
        _global_registry = FactorRegistry()
    return _global_registry


def register_factor(
    name: str,
    category: FactorCategory,
    description: str,
    **kwargs
) -> RegisteredFactor:
    """
    便捷函数：注册因子到全局注册表

    Args:
        name: 因子名称
        category: 因子类别
        description: 因子描述
        **kwargs: 其他参数

    Returns:
        注册的因子
    """
    return get_registry().register(name, category, description, **kwargs)


def get_factor(name: str) -> Optional[RegisteredFactor]:
    """
    便捷函数：从全局注册表获取因子

    Args:
        name: 因子名称

    Returns:
        注册的因子
    """
    return get_registry().get(name)
