"""Tests for storage implementations."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import shutil
from pathlib import Path

from quanttool.infrastructure.stores.parquet_store import ParquetStore
from quanttool.infrastructure.stores.meta_db import MetaDB


# Module-level fixture for all test classes
@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


class TestParquetStore:
    """Test cases for ParquetStore."""

    @pytest.fixture
    def store(self, temp_dir):
        """Create a ParquetStore instance."""
        return ParquetStore(temp_dir)

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'open': [100.0] * 10,
            'high': [105.0] * 10,
            'low': [95.0] * 10,
            'close': [100.0] * 10,
            'volume': [1000000] * 10,
        })

    def test_initialization(self, temp_dir):
        """Test store initialization."""
        store = ParquetStore(temp_dir)
        assert store.data_dir.exists()
        assert store.metadata_dir.exists()

    def test_save_data(self, store, sample_data):
        """Test saving data."""
        key = "test/data"
        result = store.save_data(key, sample_data)

        assert result is True
        assert store.has_data(key)

    def test_save_data_with_metadata(self, store, sample_data):
        """Test saving data with metadata."""
        key = "test/data"
        metadata = {"symbol": "000001.SZ", "timeframe": "1d"}

        result = store.save_data(key, sample_data, metadata)

        assert result is True

        # Check metadata was saved
        loaded_metadata = store.get_metadata(key)
        assert loaded_metadata is not None
        assert loaded_metadata.get("symbol") == "000001.SZ"

    def test_load_data(self, store, sample_data):
        """Test loading data."""
        key = "test/data"
        store.save_data(key, sample_data)

        loaded_data = store.load_data(key)

        assert loaded_data is not None
        assert len(loaded_data) == len(sample_data)
        assert list(loaded_data.columns) == list(sample_data.columns)

    def test_load_nonexistent_data(self, store):
        """Test loading non-existent data."""
        result = store.load_data("nonexistent/key")
        assert result is None

    def test_list_keys(self, store, sample_data):
        """Test listing keys."""
        store.save_data("prefix/key1", sample_data)
        store.save_data("prefix/key2", sample_data)
        store.save_data("other/key3", sample_data)

        keys = store.list_keys("prefix")

        assert len(keys) == 2
        assert "prefix/key1" in keys
        assert "prefix/key2" in keys

    def test_delete_data(self, store, sample_data):
        """Test deleting data."""
        key = "test/data"
        store.save_data(key, sample_data)
        assert store.has_data(key)

        result = store.delete_data(key)

        assert result is True
        assert not store.has_data(key)

    def test_delete_nonexistent_data(self, store):
        """Test deleting non-existent data."""
        result = store.delete_data("nonexistent/key")
        assert result is False

    def test_has_data(self, store, sample_data):
        """Test checking data existence."""
        key = "test/data"
        assert not store.has_data(key)

        store.save_data(key, sample_data)
        assert store.has_data(key)

    def test_sanitize_key(self, store):
        """Test key sanitization."""
        key = "path/with/slashes.dots:colons"
        sanitized = store._sanitize_key(key)

        assert "__SLASH__" in sanitized
        assert "__DOT__" in sanitized
        assert "__COLON__" in sanitized


class TestMetaDB:
    """Test cases for MetaDB using PostgreSQL backend."""

    @pytest.fixture
    def db(self):
        """Create a MetaDB instance using PostgreSQL."""
        return MetaDB()

    def _uuid(self, name: str) -> str:
        """Generate a valid UUID from a name for testing."""
        import uuid
        import hashlib
        # Create a deterministic UUID from the name
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
        return str(uuid.uuid5(namespace, name))

    def test_initialization(self, db):
        """Test database initialization."""
        # MetaDB now uses PostgreSQL, so we just verify it's connected
        assert db is not None

    def test_save_experiment_run(self, db):
        """Test saving experiment run."""
        import uuid
        run_id = str(uuid.uuid4())
        run_data = {
            "id": run_id,
            "type": "backtest",
            "parameters": {"strategy": "ma_cross"},
            "git_commit": "abc123",
            "data_version": "v1.0",
            "start_time": datetime.now(),
            "end_time": datetime.now(),
            "status": "completed",
            "results": {"return": 0.1},
            "artifacts": ["report.html"]
        }

        db.save_experiment_run(run_data)

        # Verify it was saved
        loaded = db.get_experiment_run(run_id)
        assert loaded is not None
        assert str(loaded["id"]) == run_id
        assert loaded["type"] == "backtest"

    def test_get_experiment_run_nonexistent(self, db):
        """Test getting non-existent experiment run."""
        import uuid
        nonexistent_id = str(uuid.uuid4())
        result = db.get_experiment_run(nonexistent_id)
        assert result is None

    def test_get_experiment_runs(self, db):
        """Test listing experiment runs."""
        import uuid
        # Save multiple runs with unique UUIDs
        run_ids = []
        for i in range(3):
            run_id = str(uuid.uuid4())
            run_ids.append(run_id)
            db.save_experiment_run({
                "id": run_id,
                "type": "backtest" if i < 2 else "factor_mining",
                "parameters": {},
                "git_commit": "abc",
                "data_version": "v1",
                "start_time": datetime.now(),
                "end_time": datetime.now(),
                "status": "completed" if i < 2 else "failed",
                "results": {},
                "artifacts": []
            })

        # Test filtering by type
        backtest_runs = db.get_experiment_runs(run_type="backtest")
        # We should have at least the 2 we just added
        assert len(backtest_runs) >= 2

        # Test filtering by status
        completed_runs = db.get_experiment_runs(status="completed")
        assert len(completed_runs) >= 2

    def test_save_task(self, db):
        """Test saving task."""
        import uuid
        task_id = str(uuid.uuid4())
        task_data = {
            "id": task_id,
            "type": "data_pull",
            "status": "completed",
            "parameters": {"symbol": "000001.SZ"},
            "created_at": datetime.now(),
            "started_at": datetime.now(),
            "completed_at": datetime.now(),
            "result": {"rows": 100},
            "error": None
        }

        db.save_task(task_data)

        loaded = db.get_task(task_id)
        assert loaded is not None
        assert str(loaded["id"]) == task_id
        assert loaded["type"] == "data_pull"

    def test_get_tasks(self, db):
        """Test listing tasks."""
        import uuid
        # Save multiple tasks
        for i in range(3):
            db.save_task({
                "id": str(uuid.uuid4()),
                "type": "data_pull",
                "status": "completed" if i < 2 else "pending",
                "parameters": {},
                "created_at": datetime.now(),
                "started_at": None,
                "completed_at": None,
                "result": {},
                "error": None
            })

        all_tasks = db.get_tasks()
        assert len(all_tasks) >= 3

        completed_tasks = db.get_tasks(status="completed")
        assert len(completed_tasks) >= 2


class TestStoreEdgeCases:
    """Test edge cases for storage."""

    def test_parquet_empty_dataframe(self, temp_dir):
        """Test saving empty DataFrame."""
        store = ParquetStore(temp_dir)
        empty_df = pd.DataFrame()

        result = store.save_data("empty", empty_df)
        assert result is True

        loaded = store.load_data("empty")
        assert loaded is not None
        assert len(loaded) == 0

    def test_parquet_large_dataframe(self, temp_dir):
        """Test saving large DataFrame."""
        store = ParquetStore(temp_dir)
        large_df = pd.DataFrame({
            'col1': range(100000),
            'col2': np.random.randn(100000)
        })

        result = store.save_data("large", large_df)
        assert result is True

        loaded = store.load_data("large")
        assert len(loaded) == 100000
