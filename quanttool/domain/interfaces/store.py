"""Abstract interface for data stores in QuantTool."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd


class IStore(ABC):
    """Abstract interface for data storage systems."""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the store with configuration.

        Args:
            config: Store-specific configuration
        """
        pass

    @abstractmethod
    def save_data(
        self, key: str, data: pd.DataFrame, metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Save data to the store.

        Args:
            key: Key to identify the data
            data: Data to save
            metadata: Optional metadata

        Returns:
            True if saved successfully, False otherwise
        """
        pass

    @abstractmethod
    def load_data(self, key: str) -> Optional[pd.DataFrame]:
        """
        Load data from the store.

        Args:
            key: Key to identify the data

        Returns:
            Loaded data or None if not found
        """
        pass

    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """
        List keys in the store with optional prefix filter.

        Args:
            prefix: Optional prefix to filter keys

        Returns:
            List of matching keys
        """
        pass

    @abstractmethod
    def delete_data(self, key: str) -> bool:
        """
        Delete data from the store.

        Args:
            key: Key to identify the data

        Returns:
            True if deleted successfully, False otherwise
        """
        pass

    @abstractmethod
    def has_data(self, key: str) -> bool:
        """
        Check if data exists in the store.

        Args:
            key: Key to identify the data

        Returns:
            True if data exists, False otherwise
        """
        pass

    @abstractmethod
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a key.

        Args:
            key: Key to identify the data

        Returns:
            Metadata dictionary or None if not found
        """
        pass
