"""Abstract interface for notifiers in QuantTool."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class INotifier(ABC):
    """Abstract interface for notification systems."""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the notifier with configuration.

        Args:
            config: Notifier-specific configuration
        """
        pass

    @abstractmethod
    def send_notification(self, message: str, subject: str = None, **kwargs) -> bool:
        """
        Send a notification.

        Args:
            message: Notification message
            subject: Optional subject/title
            **kwargs: Additional parameters

        Returns:
            True if notification sent successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the notifier.

        Returns:
            Notifier name
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Get a description of the notifier.

        Returns:
            Notifier description
        """
        pass
