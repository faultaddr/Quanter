"""Shared event loop management for synchronous wrappers.

This module provides a dedicated event loop thread for running async operations.
All async operations run in the same event loop to ensure connection pool stability.
"""

import asyncio
import concurrent.futures
import threading
import queue
from typing import Any, Callable


class AsyncLoopThread:
    """
    A dedicated thread running an event loop for async operations.

    This ensures all asyncpg connections are managed in a single event loop,
    preventing "connection was closed in the middle of operation" errors.
    """

    def __init__(self):
        self._loop = None
        self._thread = None
        self._ready = threading.Event()
        self._started = False
        self._lock = threading.Lock()

    def start(self):
        """Start the event loop thread."""
        with self._lock:
            if self._started:
                return
            self._started = True

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait()  # Wait for loop to be ready

    def _run_loop(self):
        """Run the event loop in this thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()  # Signal that loop is ready

        # Run forever
        self._loop.run_forever()

    def run_coro(self, coro) -> Any:
        """
        Run a coroutine in the event loop thread.

        Args:
            coro: Coroutine to run

        Returns:
            Result of the coroutine
        """
        if not self._started:
            self.start()

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()  # This blocks until done

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop."""
        if not self._started:
            self.start()
        return self._loop


# Global async loop thread (singleton)
_async_thread: AsyncLoopThread = None
_thread_lock = threading.Lock()


def get_async_thread() -> AsyncLoopThread:
    """Get the global async loop thread."""
    global _async_thread
    with _thread_lock:
        if _async_thread is None:
            _async_thread = AsyncLoopThread()
        return _async_thread


def run_async(coro) -> Any:
    """
    Run an async coroutine synchronously.

    Uses a dedicated event loop thread to ensure connection pool stability.

    Args:
        coro: An async coroutine to run

    Returns:
        The result of the coroutine
    """
    try:
        # Check if we're already in an async context
        asyncio.get_running_loop()
        # We're in an async context - this is unusual but handle it
        # Run in the dedicated thread to avoid loop conflicts
        return get_async_thread().run_coro(coro)
    except RuntimeError:
        # No running loop, use the dedicated thread
        return get_async_thread().run_coro(coro)
