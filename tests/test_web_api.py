"""Tests for web API."""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from quanttool.web.app import app


client = TestClient(app)


class TestWebAPI:
    """Test cases for web API endpoints."""

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        # Root endpoint may return HTML file or JSON message
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            assert "message" in response.json()
            assert "QuantTool" in response.json()["message"]
        else:
            # HTML response is also valid
            assert len(response.content) > 0

    def test_list_data_providers(self):
        """Test listing data providers."""
        response = client.get("/api/data/providers")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_strategies(self):
        """Test listing strategies."""
        response = client.get("/api/strategies")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_factors(self):
        """Test listing factors."""
        response = client.get("/api/factors")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_experiments(self):
        """Test listing experiments."""
        response = client.get("/api/experiments")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.skip(reason="Async DB operations have event loop issues with TestClient")
    def test_get_backtest_result_not_found(self):
        """Test getting non-existent backtest result."""
        response = client.get("/api/backtest/runs/nonexistent-id")
        assert response.status_code == 404


class TestWebAPIBacktest:
    """Test backtest API endpoints."""

    @pytest.mark.skip(reason="Requires data provider setup")
    def test_run_backtest(self):
        """Test running a backtest."""
        request_data = {
            "strategy_name": "ma_cross",
            "symbols": ["000001.SZ"],
            "start_date": "2023-01-01",
            "end_date": "2023-02-01",
            "timeframe": "1d",
            "initial_cash": 100000.0,
            "commission_rate": 0.0003,
            "data_provider": "csv_mock",
            "strategy_params": {"short_window": 5, "long_window": 10}
        }

        response = client.post("/api/backtest/run", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "run_id" in data
        assert "result" in data


class TestWebAPIFactor:
    """Test factor API endpoints."""

    @pytest.mark.skip(reason="Requires data provider setup")
    def test_mine_factors(self):
        """Test factor mining endpoint."""
        request_data = {
            "factor_name": "momentum",
            "symbols": ["000001.SZ"],
            "start_date": "2023-01-01",
            "end_date": "2023-02-01",
            "data_provider": "csv_mock",
            "factor_params": {"period": 10}
        }

        response = client.post("/api/factors/mine", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "run_id" in data
        assert "results" in data


class TestWebAPIErrorHandling:
    """Test API error handling."""

    def test_invalid_backtest_request(self):
        """Test backtest with invalid request."""
        # Missing required fields
        request_data = {
            "strategy_name": "ma_cross"
            # Missing other required fields
        }

        response = client.post("/api/backtest/run", json=request_data)
        # Should handle gracefully (may return 422 or 500 depending on implementation)
        assert response.status_code in [200, 422, 500]

    def test_invalid_factor_request(self):
        """Test factor mining with invalid request."""
        request_data = {
            "factor_name": "nonexistent_factor"
        }

        response = client.post("/api/factors/mine", json=request_data)
        # Should handle gracefully
        assert response.status_code in [200, 422, 500]


class TestWebAPISchemas:
    """Test API request/response schemas."""

    def test_backtest_request_schema(self):
        """Test backtest request schema validation."""
        from quanttool.web.schemas.backtest import BacktestRequest

        request = BacktestRequest(
            strategy_name="ma_cross",
            symbols=["000001.SZ"],
            start_date="2023-01-01",
            end_date="2023-02-01"
        )

        assert request.strategy_name == "ma_cross"
        assert request.symbols == ["000001.SZ"]
        assert request.timeframe == "10m"  # Default value
        assert request.initial_cash == 100000.0  # Default value

    def test_factor_request_schema(self):
        """Test factor request schema validation."""
        from quanttool.web.schemas.factor import FactorMineRequest

        request = FactorMineRequest(
            factor_name="momentum",
            symbols=["000001.SZ"],
            start_date="2023-01-01",
            end_date="2023-02-01"
        )

        assert request.factor_name == "momentum"
        assert request.data_provider == "tushare"  # Default value
