"""WebSocket module for real-time signal updates."""

import asyncio
import json
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from ..core.logging import get_logger


logger = get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """Connect a new WebSocket client."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(
            f"Client {client_id} connected. Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, client_id: str):
        """Disconnect a WebSocket client."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(
                f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}"
            )

    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except WebSocketDisconnect:
                self.disconnect(client_id)
                raise

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        disconnected_clients = []

        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(json.dumps(message))
            except WebSocketDisconnect:
                disconnected_clients.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)


manager = ConnectionManager()


async def signal_websocket_endpoint(websocket: WebSocket, client_id: str = "default"):
    """WebSocket endpoint for real-time signal updates."""
    await manager.connect(websocket, client_id)

    try:
        while True:
            # This is a placeholder - in a real implementation, you'd receive messages from clients
            data = await websocket.receive_text()
            parsed_data = json.loads(data)

            # Echo the message back to the sender
            response = {"message": "Received", "received_data": parsed_data}
            await manager.send_personal_message(response, client_id)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
