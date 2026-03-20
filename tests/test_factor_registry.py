"""Tests for factor registry module."""

import pytest
import os
import json
from datetime import datetime
from pathlib import Path

from quanttool.factors.factor_registry import (
    FactorRegistry,
    get_registry,
    register_factor,
    get_factor,
    FactorCategory,
    FactorStatus,
)


class TestFactorRegistry:
    """Test cases for FactorRegistry."""

    @pytest.fixture
    def registry(self):
        """Create registry instance."""
        return FactorRegistry()

    @pytest.fixture
    def temp_storage_path(self, tmp_path):
        """Create temporary storage path."""
        return str(tmp_path / "test_registry.json")

    def test_register_factor(self, registry):
        """Test factor registration."""
        factor = registry.register(
            "pe_ratio",
            FactorCategory.VALUE,
            "Price to Earnings Ratio",
        )
        assert factor.metadata.name == "pe_ratio"
        assert factor.status == FactorStatus.ACTIVE

    def test_get_factor(self, registry):
        """Test getting factor."""
        registry.register("test_factor", FactorCategory.TECHNICAL, "Test")
        factor = registry.get("test_factor")
        assert factor is not None

    def test_get_nonexistent(self, registry):
        """Test getting non-existent factor."""
        factor = registry.get("nonexistent")
        assert factor is None

    def test_list_factors(self, registry):
        """Test listing all factors."""
        registry.register("factor1", FactorCategory.TECHNICAL, "Factor 1")
        registry.register("factor2", FactorCategory.VALUE, "Factor 2")
        factors = registry.list_factors()
        assert len(factors) >= 2

    def test_update_performance(self, registry):
        """Test updating factor performance."""
        registry.register("test_factor", FactorCategory.TECHNICAL, "Test")
        success = registry.update_performance("test_factor", 0.05, 0.02, 2.5, 0.15)
        assert success is True

    def test_get_effective_factors(self, registry):
        """Test getting effective factors."""
        registry.register("good_factor", FactorCategory.TECHNICAL, "Good")
        registry.update_performance("good_factor", 0.05, 0.02, 2.5, 0.15)
        registry.register("bad_factor", FactorCategory.TECHNICAL, "Bad")
        registry.update_performance("bad_factor", 0.01, 0.02, 0.5, 0.05)
        effective = registry.get_effective_factors(min_ir=1.0)
        assert "good_factor" in effective
        assert "bad_factor" not in effective

    def test_save_and_load(self, registry, temp_storage_path):
        """Test saving and loading registry."""
        registry.register("test_factor", FactorCategory.TECHNICAL, "Test Factor")
        saved_path = registry.save(temp_storage_path)
        assert os.path.exists(saved_path)
        new_registry = FactorRegistry()
        count = new_registry.load(temp_storage_path)
        assert count >= 1


class TestGlobalRegistry:
    """Test global registry functions."""

    def test_get_registry(self):
        """Test getting global registry."""
        registry = get_registry()
        assert isinstance(registry, FactorRegistry)

    def test_register_factor_helper(self):
        """Test register_factor helper."""
        factor = register_factor("helper_test", FactorCategory.TECHNICAL, "Helper test")
        assert factor.metadata.name == "helper_test"
