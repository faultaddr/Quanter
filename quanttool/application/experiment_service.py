"""Experiment service for QuantTool."""

from datetime import datetime
from typing import Dict, Any
from ..infrastructure.stores.meta_db import MetaDB
from ..core.logging import get_logger


logger = get_logger(__name__)


class ExperimentService:
    """Service class for managing experiment runs."""

    def __init__(self):
        """Initialize experiment service."""
        self.db = MetaDB()

    def create_experiment_run(self, run_data: Dict[str, Any]) -> str:
        """
        Create a new experiment run record.

        Args:
            run_data: Dictionary with experiment run information

        Returns:
            ID of the created experiment run
        """
        from uuid import uuid4

        run_id = str(uuid4())

        run_data["id"] = run_id
        run_data["start_time"] = datetime.now()
        run_data["status"] = "running"

        self.db.save_experiment_run(run_data)

        logger.info(f"Created experiment run: {run_id}")

        return run_id

    def update_experiment_run(self, run_id: str, update_data: Dict[str, Any]):
        """
        Update an existing experiment run record.

        Args:
            run_id: ID of the experiment run to update
            update_data: Dictionary with fields to update
        """
        existing_run = self.db.get_experiment_run(run_id)
        if not existing_run:
            raise ValueError(f"Experiment run {run_id} not found")

        # Merge update data with existing data
        for key, value in update_data.items():
            existing_run[key] = value

        self.db.save_experiment_run(existing_run)

        logger.info(f"Updated experiment run: {run_id}")

    def complete_experiment_run(
        self, run_id: str, results: Dict[str, Any] = None, artifacts: list = None
    ):
        """
        Mark an experiment run as completed.

        Args:
            run_id: ID of the experiment run to complete
            results: Results of the experiment
            artifacts: List of artifact file paths
        """
        existing_run = self.db.get_experiment_run(run_id)
        if not existing_run:
            raise ValueError(f"Experiment run {run_id} not found")

        existing_run["status"] = "completed"
        existing_run["end_time"] = datetime.now()
        if results:
            existing_run["results"] = results
        if artifacts:
            existing_run["artifacts"] = artifacts

        self.db.save_experiment_run(existing_run)

        logger.info(f"Completed experiment run: {run_id}")

    def fail_experiment_run(self, run_id: str, error_message: str = None):
        """
        Mark an experiment run as failed.

        Args:
            run_id: ID of the experiment run to mark as failed
            error_message: Error message to record
        """
        existing_run = self.db.get_experiment_run(run_id)
        if not existing_run:
            raise ValueError(f"Experiment run {run_id} not found")

        existing_run["status"] = "failed"
        existing_run["end_time"] = datetime.now()
        if error_message:
            existing_run["error"] = error_message

        self.db.save_experiment_run(existing_run)

        logger.info(f"Marked experiment run as failed: {run_id}")

    def get_experiment_run(self, run_id: str) -> Dict[str, Any]:
        """
        Get information about an experiment run.

        Args:
            run_id: ID of the experiment run

        Returns:
            Dictionary with experiment run information
        """
        run = self.db.get_experiment_run(run_id)
        if not run:
            raise ValueError(f"Experiment run {run_id} not found")

        return run

    def list_experiment_runs(self, run_type: str = None, status: str = None) -> list:
        """
        List experiment runs with optional filtering.

        Args:
            run_type: Filter by experiment type
            status: Filter by experiment status

        Returns:
            List of experiment run dictionaries
        """
        return self.db.get_experiment_runs(run_type=run_type, status=status)
