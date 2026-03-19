"""Parquet store implementation for QuantTool."""

import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from ...domain.interfaces.store import IStore
from ...core.errors import DataProviderError
from ...core.registry import registry, ComponentType
from ...core.logging import get_logger


logger = get_logger(__name__)


@registry.register(ComponentType.STORE, "parquet")
class ParquetStore(IStore):
    """Parquet store implementation for efficient data storage."""

    def __init__(self, data_dir: str = "./data"):
        """
        Initialize Parquet store.

        Args:
            data_dir: Directory for storing parquet files
        """
        self.data_dir = Path(data_dir)
        self.metadata_dir = self.data_dir / "metadata"

        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the store with configuration.

        Args:
            config: Store-specific configuration
        """
        if "data_dir" in config:
            self.data_dir = Path(config["data_dir"])
            self.metadata_dir = self.data_dir / "metadata"

            # Create directories if they don't exist
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Parquet store initialized with data directory: {self.data_dir}")

    def save_data(
        self, key: str, data: pd.DataFrame, metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Save data to parquet file.

        Args:
            key: Key to identify the data (will be converted to file path)
            data: Data to save
            metadata: Optional metadata

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Convert key to safe filename
            safe_key = self._sanitize_key(key)
            file_path = self.data_dir / f"{safe_key}.parquet"

            # Add metadata to the table as key-value pairs
            table = pa.Table.from_pandas(data)

            # If we have metadata, attach it to the schema
            if metadata:
                # Convert all metadata values to strings for storage
                metadata_strings = {}
                for k, v in metadata.items():
                    if v is None:
                        metadata_strings[k] = "None"
                    elif isinstance(v, (int, float, str, bool)):
                        metadata_strings[k] = str(v)
                    else:
                        metadata_strings[k] = str(
                            v
                        )  # Convert everything else to string

                # Update the table's metadata
                new_metadata = (
                    {**table.schema.metadata, **metadata_strings}
                    if table.schema.metadata
                    else metadata_strings
                )
                table = table.replace_schema_metadata(new_metadata)

            # Write to parquet file
            pq.write_table(table, file_path)

            # Also save metadata separately as JSON if needed
            if metadata:
                import json

                metadata_path = self.metadata_dir / f"{safe_key}_metadata.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, default=str, ensure_ascii=False, indent=2)

            logger.info(f"Data saved successfully to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save data with key {key}: {str(e)}")
            return False

    def load_data(self, key: str) -> Optional[pd.DataFrame]:
        """
        Load data from parquet file.

        Args:
            key: Key to identify the data

        Returns:
            Loaded data or None if not found
        """
        try:
            # Convert key to safe filename
            safe_key = self._sanitize_key(key)
            file_path = self.data_dir / f"{safe_key}.parquet"

            if not file_path.exists():
                logger.info(f"No data found for key: {key}")
                return None

            # Read from parquet file
            table = pq.read_table(file_path)
            data = table.to_pandas()

            logger.info(f"Data loaded successfully from {file_path}")
            return data

        except Exception as e:
            logger.error(f"Failed to load data with key {key}: {str(e)}")
            return None

    def list_keys(self, prefix: str = "") -> List[str]:
        """
        List keys in the store with optional prefix filter.

        Args:
            prefix: Optional prefix to filter keys

        Returns:
            List of matching keys
        """
        try:
            # Convert prefix to safe format
            safe_prefix = self._sanitize_key(prefix)
            keys = []

            for file_path in self.data_dir.glob("*.parquet"):
                filename = file_path.stem
                if filename.endswith("_metadata"):
                    continue  # Skip metadata files

                if filename.startswith(safe_prefix):
                    # Convert filename back to key format
                    key = filename
                    # Undo sanitization transformations
                    key = (
                        key.replace("__SLASH__", "/")
                        .replace("__DOT__", ".")
                        .replace("__COLON__", ":")
                    )
                    keys.append(key)

            logger.info(f"Found {len(keys)} keys with prefix '{prefix}'")
            return sorted(keys)

        except Exception as e:
            logger.error(f"Failed to list keys with prefix '{prefix}': {str(e)}")
            return []

    def delete_data(self, key: str) -> bool:
        """
        Delete data from the store.

        Args:
            key: Key to identify the data

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Convert key to safe filename
            safe_key = self._sanitize_key(key)
            file_path = self.data_dir / f"{safe_key}.parquet"
            metadata_path = self.metadata_dir / f"{safe_key}_metadata.json"

            deleted = False
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted data file: {file_path}")
                deleted = True

            if metadata_path.exists():
                metadata_path.unlink()
                logger.info(f"Deleted metadata file: {metadata_path}")

            if not deleted:
                logger.warning(f"No data found to delete for key: {key}")

            return deleted

        except Exception as e:
            logger.error(f"Failed to delete data with key {key}: {str(e)}")
            return False

    def has_data(self, key: str) -> bool:
        """
        Check if data exists in the store.

        Args:
            key: Key to identify the data

        Returns:
            True if data exists, False otherwise
        """
        # Convert key to safe filename
        safe_key = self._sanitize_key(key)
        file_path = self.data_dir / f"{safe_key}.parquet"
        return file_path.exists()

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a key.

        Args:
            key: Key to identify the data

        Returns:
            Metadata dictionary or None if not found
        """
        try:
            # Try to load from separate metadata file first
            safe_key = self._sanitize_key(key)
            metadata_path = self.metadata_dir / f"{safe_key}_metadata.json"

            if metadata_path.exists():
                import json

                with open(metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)

            # If no separate metadata file, try to load from parquet schema
            file_path = self.data_dir / f"{safe_key}.parquet"
            if file_path.exists():
                table = pq.read_table(file_path)

                if table.schema.metadata:
                    metadata = {}
                    for k, v in table.schema.metadata.items():
                        # Attempt to convert string values back to appropriate types
                        try:
                            # Try to evaluate as Python literal if it looks like one
                            if v == "None":
                                metadata[k] = None
                            elif v.lower() in ("true", "false"):
                                metadata[k] = v.lower() == "true"
                            elif "." in v:
                                # Could be a float
                                try:
                                    metadata[k] = float(v)
                                except ValueError:
                                    metadata[k] = v
                            else:
                                # Could be an int
                                try:
                                    metadata[k] = int(v)
                                except ValueError:
                                    metadata[k] = v
                        except:
                            # If conversion fails, keep as string
                            metadata[k] = v
                    return metadata

            logger.info(f"No metadata found for key: {key}")
            return None

        except Exception as e:
            logger.error(f"Failed to get metadata for key {key}: {str(e)}")
            return None

    def _sanitize_key(self, key: str) -> str:
        """
        Sanitize key to be a valid filename.

        Args:
            key: Original key

        Returns:
            Sanitized key safe for use as filename
        """
        # Replace problematic characters
        safe_key = (
            key.replace("/", "__SLASH__")
            .replace(".", "__DOT__")
            .replace(":", "__COLON__")
        )
        # Further sanitize to ensure it's a valid filename
        safe_chars = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        )
        sanitized = "".join(
            c if c in safe_chars else f"_{ord(c):02x}" for c in safe_key
        )
        return sanitized
