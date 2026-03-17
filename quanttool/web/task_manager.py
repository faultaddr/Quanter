"""
任务管理器

核心设计：
1. 所有耗时操作作为独立任务执行
2. 后台线程池处理任务
3. Web 端只同步状态，不阻塞主线程
4. 支持任务进度、日志、结果查询

使用示例：
    # 创建任务
    task_id = task_manager.create_task(
        name="qlib_train",
        handler=train_qlib_model,
        params={...}
    )

    # 查询状态
    status = task_manager.get_task_status(task_id)

    # 获取结果
    result = task_manager.get_task_result(task_id)
"""

import threading
import queue
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
import traceback

from quanttool.core.logging import get_logger

logger = get_logger(__name__)


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"      # 等待执行
    RUNNING = "running"      # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失败
    CANCELLED = "cancelled"  # 已取消


class TaskPriority(int, Enum):
    """任务优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class TaskProgress:
    """任务进度"""
    current: int = 0
    total: int = 100
    message: str = ""
    stage: str = ""

    @property
    def percent(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100)


@dataclass
class Task:
    """任务对象"""
    id: str
    name: str
    handler: Callable
    params: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    progress: TaskProgress = field(default_factory=TaskProgress)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    future: Optional[Future] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority.value,
            "status": self.status.value,
            "progress": {
                "current": self.progress.current,
                "total": self.progress.total,
                "percent": round(self.progress.percent, 1),
                "message": self.progress.message,
                "stage": self.progress.stage,
            },
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration": self._get_duration(),
        }

    def _get_duration(self) -> Optional[float]:
        """获取执行时长（秒）"""
        if self.started_at is None:
            return None
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()


class TaskManager:
    """
    任务管理器

    功能：
    1. 任务创建和调度
    2. 后台线程池执行
    3. 进度追踪和日志收集
    4. 结果缓存
    """

    def __init__(
        self,
        max_workers: int = 4,
        max_tasks: int = 100,
        result_ttl: int = 3600,  # 结果保留时间（秒）
    ):
        """
        初始化任务管理器

        Args:
            max_workers: 最大工作线程数
            max_tasks: 最大任务数量
            result_ttl: 结果保留时间（秒）
        """
        self.max_workers = max_workers
        self.max_tasks = max_tasks
        self.result_ttl = result_ttl

        self._tasks: Dict[str, Task] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.RLock()

        # 启动清理线程
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

        logger.info(f"TaskManager initialized with {max_workers} workers")

    def create_task(
        self,
        name: str,
        handler: Callable,
        params: Dict[str, Any] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> str:
        """
        创建任务

        Args:
            name: 任务名称
            handler: 任务处理函数
            params: 任务参数
            priority: 任务优先级

        Returns:
            任务ID
        """
        task_id = str(uuid.uuid4())[:8]

        task = Task(
            id=task_id,
            name=name,
            handler=handler,
            params=params or {},
            priority=priority,
        )

        with self._lock:
            self._tasks[task_id] = task

        # 提交到线程池
        task.future = self._executor.submit(self._run_task, task_id)

        logger.info(f"Task created: {task_id} ({name})")
        return task_id

    def _run_task(self, task_id: str) -> None:
        """执行任务"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return

            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()

        try:
            # 创建任务上下文
            context = TaskContext(task_id, self)

            # 执行任务处理函数
            result = task.handler(context, **task.params)

            # 任务成功完成
            with self._lock:
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = datetime.now()
                task.progress.current = task.progress.total

            logger.info(f"Task completed: {task_id} ({task.name})")

        except Exception as e:
            # 任务失败
            error_msg = f"{type(e).__name__}: {str(e)}"
            tb = traceback.format_exc()

            with self._lock:
                task.status = TaskStatus.FAILED
                task.error = error_msg
                task.completed_at = datetime.now()
                task.logs.append(f"ERROR: {error_msg}")
                task.logs.append(tb)

            logger.error(f"Task failed: {task_id} ({task.name}): {error_msg}")

    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        with self._lock:
            return self._tasks.get(task_id)

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        task = self.get_task(task_id)
        if task is None:
            return None
        return task.to_dict()

    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务结果"""
        task = self.get_task(task_id)
        if task is None or task.status != TaskStatus.COMPLETED:
            return None
        return task.result

    def get_task_logs(self, task_id: str) -> List[str]:
        """获取任务日志"""
        task = self.get_task(task_id)
        if task is None:
            return []
        return task.logs.copy()

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        列出任务

        Args:
            status: 过滤状态
            limit: 返回数量限制

        Returns:
            任务列表
        """
        with self._lock:
            tasks = list(self._tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]

        # 按创建时间降序
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return [t.to_dict() for t in tasks[:limit]]

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False

            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                return False

            # 尝试取消 Future
            if task.future and task.future.cancel():
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                logger.info(f"Task cancelled: {task_id}")
                return True

            return False

    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False

            # 先尝试取消
            if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                if task.future:
                    task.future.cancel()

            del self._tasks[task_id]
            logger.info(f"Task deleted: {task_id}")
            return True

    def update_progress(
        self,
        task_id: str,
        current: int,
        total: int = 100,
        message: str = "",
        stage: str = ""
    ) -> None:
        """更新任务进度"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return

            task.progress.current = current
            task.progress.total = total
            task.progress.message = message
            task.progress.stage = stage

    def add_log(self, task_id: str, message: str) -> None:
        """添加任务日志"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return

            task.logs.append(f"[{datetime.now().isoformat()}] {message}")

    def _cleanup_loop(self) -> None:
        """定期清理过期任务"""
        while True:
            time.sleep(60)  # 每分钟检查一次

            now = datetime.now()
            expired_ids = []

            with self._lock:
                for task_id, task in self._tasks.items():
                    # 清理已完成/失败超过 TTL 的任务
                    if task.completed_at:
                        age = (now - task.completed_at).total_seconds()
                        if age > self.result_ttl:
                            expired_ids.append(task_id)

                # 也清理超过最大数量的旧任务
                if len(self._tasks) > self.max_tasks:
                    # 按创建时间排序
                    sorted_tasks = sorted(
                        self._tasks.items(),
                        key=lambda x: x[1].created_at
                    )
                    # 删除最旧的已完成任务
                    for task_id, task in sorted_tasks:
                        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                            expired_ids.append(task_id)
                            if len(self._tasks) - len(expired_ids) <= self.max_tasks:
                                break

                for task_id in expired_ids:
                    del self._tasks[task_id]

            if expired_ids:
                logger.info(f"Cleaned up {len(expired_ids)} expired tasks")

    def shutdown(self) -> None:
        """关闭任务管理器"""
        self._executor.shutdown(wait=False)
        logger.info("TaskManager shutdown")


class TaskContext:
    """
    任务上下文

    提供给任务处理函数的上下文对象，用于：
    1. 更新进度
    2. 记录日志
    3. 检查是否被取消
    """

    def __init__(self, task_id: str, manager: TaskManager):
        self.task_id = task_id
        self._manager = manager

    def update_progress(
        self,
        current: int,
        total: int = 100,
        message: str = "",
        stage: str = ""
    ) -> None:
        """更新进度"""
        self._manager.update_progress(self.task_id, current, total, message, stage)

    def log(self, message: str) -> None:
        """记录日志"""
        self._manager.add_log(self.task_id, message)

    def is_cancelled(self) -> bool:
        """检查任务是否被取消"""
        task = self._manager.get_task(self.task_id)
        if task is None:
            return True
        return task.status == TaskStatus.CANCELLED

    def get_task(self) -> Optional[Task]:
        """获取当前任务对象"""
        return self._manager.get_task(self.task_id)


# 全局任务管理器实例
_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """获取任务管理器单例"""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
