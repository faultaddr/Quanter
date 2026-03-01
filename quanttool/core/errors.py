"""Custom exceptions for QuantTool."""


class QuantToolError(Exception):
    """Base exception for QuantTool."""

    pass


class DataProviderError(QuantToolError):
    """Raised when data provider operations fail."""

    pass


class BacktestError(QuantToolError):
    """Raised when backtesting operations fail."""

    pass


class FactorError(QuantToolError):
    """Raised when factor computation fails."""

    pass


class ModelError(QuantToolError):
    """Raised when model operations fail."""

    pass


class ValidationError(QuantToolError):
    """Raised when validation fails."""

    pass


class ConfigurationError(QuantToolError):
    """Raised when configuration is invalid."""

    pass


class NetworkError(QuantToolError):
    """Raised when network operations fail."""

    pass


class DataNotAvailableError(DataProviderError):
    """Raised when requested data is not available."""

    pass


class UnsupportedOperationError(QuantToolError):
    """Raised when an unsupported operation is attempted."""

    pass
