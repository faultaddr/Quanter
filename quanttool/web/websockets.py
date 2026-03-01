"""WebSockets for real-time updates."""

from fastapi import WebSocket
from typing import Dict, Any


async def handle_signal_update(websocket: WebSocket, data: Dict[str, Any]):
    """Handle signal update via WebSocket."""
    await websocket.send_json({"type": "signal", "data": data})


async def handle_task_update(websocket: WebSocket, data: Dict[str, Any]):
    """Handle task update via WebSocket."""
    await websocket.send_json({"type": "task", "data": data})
