"""Console notifier implementation for QuantTool."""

from typing import Dict, Any
from ...domain.interfaces.notifier import INotifier
from ...core.registry import registry, ComponentType
from ...core.logging import get_logger


logger = get_logger(__name__)


@registry.register(ComponentType.NOTIFIER, "console")
class ConsoleNotifier(INotifier):
    """Console notifier implementation."""

    def __init__(self):
        """Initialize console notifier."""
        self.initialized = False
        self.config = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the notifier with configuration.

        Args:
            config: Notifier-specific configuration
        """
        self.config = config or {}
        self.initialized = True
        logger.info("Console notifier initialized")

    def send_notification(self, message: str, subject: str = None, **kwargs) -> bool:
        """
        Send a notification to console.

        Args:
            message: Notification message
            subject: Optional subject/title
            **kwargs: Additional parameters

        Returns:
            True if notification sent successfully, False otherwise
        """
        if not self.initialized:
            self.initialize({})

        # Format the message
        if subject:
            formatted_message = f"[{subject}] {message}"
        else:
            formatted_message = message

        # Print to console
        print(formatted_message)

        # Log the notification
        logger.info(f"Notification sent: {formatted_message}")

        return True

    def get_name(self) -> str:
        """
        Get the name of the notifier.

        Returns:
            Notifier name
        """
        return "console"

    def get_description(self) -> str:
        """
        Get a description of the notifier.

        Returns:
            Notifier description
        """
        return "Simple console notifier that prints messages to stdout"
