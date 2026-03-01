"""Task service for managing background jobs in QuantTool."""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Callable, Awaitable
from ..infrastructure.stores.meta_db import MetaDB
from ..core.logging import get_logger


logger = get_logger(__name__)


class TaskService:
    """Service for managing background tasks and job queues."""

    def __init__(self):
        """Initialize task service."""
        self.db = MetaDB()
        self.active_tasks = {}

    def submit_task(
        self, task_type: str, func: Callable[..., Any], *args, **kwargs
    ) -> str:
        """
        Submit a task for background execution.

        Args:
            task_type: Type of task (e.g., 'backtest', 'factor_mining', 'data_pull')
            func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Task ID for tracking the task
        """
        task_id = str(uuid.uuid4())

        # Save task info to DB
        task_data = {
            "id": task_id,
            "type": task_type,
            "status": "pending",
            "parameters": {
                "func_name": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs),
            },
            "created_at": datetime.now(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
        }

        self.db.save_task(task_data)

        # Execute task asynchronously
        asyncio.create_task(self._execute_task(task_id, func, *args, **kwargs))

        logger.info(f"Submitted task {task_id} of type {task_type}")

        return task_id

    async def _execute_task(
        self, task_id: str, func: Callable[..., Any], *args, **kwargs
    ):
        """Execute a task and update its status."""
        try:
            # Update status to running
            task_data = self.db.get_task(task_id)
            if task_data:
                task_data["status"] = "running"
                task_data["started_at"] = datetime.now()
                self.db.save_task(task_data)

            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Update status to completed
            task_data = self.db.get_task(task_id)
            if task_data:
                task_data["status"] = "completed"
                task_data["completed_at"] = datetime.now()
                task_data["result"] = result
                self.db.save_task(task_data)

            logger.info(f"Task {task_id} completed successfully")

        except Exception as e:
            # Update status to failed
            task_data = self.db.get_task(task_id)
            if task_data:
                task_data["status"] = "failed"
                task_data["completed_at"] = datetime.now()
                task_data["error"] = str(e)
                self.db.save_task(task_data)

            logger.error(f"Task {task_id} failed with error: {str(e)}")

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.

        Args:
            task_id: ID of the task to check

        Returns:
            Dictionary with task status information
        """
        task_data = self.db.get_task(task_id)

        if not task_data:
            return {"error": f"Task {task_id} not found"}

        return {
            "id": task_data["id"],
            "type": task_data["type"],
            "status": task_data["status"],
            "created_at": task_data["created_at"],
            "started_at": task_data["started_at"],
            "completed_at": task_data["completed_at"],
            "error": task_data["error"],
        }

    def get_tasks(self, task_type: str = None, status: str = None) -> list:
        """
        Get a list of tasks with optional filtering.

        Args:
            task_type: Filter by task type
            status: Filter by task status

        Returns:
            List of task dictionaries
        """
        return self.db.get_tasks(task_type=task_type, status=status)

    async def run_task_with_callback(
        self,
        task_type: str,
        func: Callable[..., Any],
        callback_func: Callable[[str, Any, Exception], Awaitable[None]],
        *args,
        **kwargs,
    ) -> str:
        """
        Run a task and call a callback function when it completes.

        Args:
            task_type: Type of task
            func: Function to execute
            callback_func: Callback function to call on completion
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())

        # Save task info to DB
        task_data = {
            "id": task_id,
            "type": task_type,
            "status": "pending",
            "parameters": {
                "func_name": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs),
            },
            "created_at": datetime.now(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
        }

        self.db.save_task(task_data)

        # Execute task with callback
        asyncio.create_task(
            self._execute_task_with_callback(
                task_id, func, callback_func, *args, **kwargs
            )
        )

        return task_id

    async def _execute_task_with_callback(
        self,
        task_id: str,
        func: Callable[..., Any],
        callback_func: Callable[[str, Any, Exception], Awaitable[None]],
        *args,
        **kwargs,
    ):
        """Execute a task with a callback on completion."""
        result = None
        error = None

        try:
            # Update status to running
            task_data = self.db.get_task(task_id)
            if task_data:
                task_data["status"] = "running"
                task_data["started_at"] = datetime.now()
                self.db.save_task(task_data)

            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Update status to completed
            task_data = self.db.get_task(task_id)
            if task_data:
                task_data["status"] = "completed"
                task_data["completed_at"] = datetime.now()
                task_data["result"] = result
                self.db.save_task(task_data)

        except Exception as e:
            error = e
            # Update status to failed
            task_data = self.db.get_task(task_id)
            if task_data:
                task_data["status"] = "failed"
                task_data["completed_at"] = datetime.now()
                task_data["error"] = str(e)
                self.db.save_task(task_data)

        # Call the callback function
        try:
            if asyncio.iscoroutinefunction(callback_func):
                await callback_func(task_id, result, error)
            else:
                callback_func(task_id, result, error)
        except Exception as e:
            logger.error(f"Callback function for task {task_id} failed: {str(e)}")

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            task_id: ID of the task to cancel

        Returns:
            True if cancellation was initiated, False otherwise
        """
        # Note: True task cancellation is complex and may not be possible for all tasks
        # In this implementation, we'll just mark the task as cancelled in the DB
        task_data = self.db.get_task(task_id)
        if task_data and task_data["status"] in ["pending", "running"]:
            task_data["status"] = "cancelled"
            task_data["completed_at"] = datetime.now()
            self.db.save_task(task_data)

            logger.info(f"Task {task_id} marked as cancelled")
            return True

        return False
