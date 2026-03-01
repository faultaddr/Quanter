"""Core registry module for QuantTool plugins and services."""

import inspect
from typing import Type, Dict, Any, List, Optional
from abc import ABC
from enum import Enum


class ComponentType(Enum):
    DATA_PROVIDER = "data_provider"
    STRATEGY = "strategy"
    FACTOR = "factor"
    MODEL = "model"
    REPORT = "report"
    NOTIFIER = "notifier"
    STORE = "store"
    CALENDAR = "calendar"


class Registry:
    """Global registry for all plugin types in QuantTool."""

    def __init__(self):
        self._registry: Dict[ComponentType, Dict[str, Type]] = {
            ct: {} for ct in ComponentType
        }

    def register(self, component_type: ComponentType, name: str = None):
        """Decorator to register a class in the registry."""

        def decorator(cls):
            # If no name provided, use the class name lowercased
            registry_name = name or cls.__name__.lower().replace(
                "provider", ""
            ).replace("strategy", "").replace("factor", "").replace(
                "model", ""
            ).replace(
                "report", ""
            ).replace(
                "notifier", ""
            ).replace(
                "store", ""
            ).replace(
                "calendar", ""
            )

            if registry_name in self._registry[component_type]:
                raise ValueError(
                    f"{component_type.value} '{registry_name}' already registered"
                )

            self._registry[component_type][registry_name] = cls
            return cls

        return decorator

    def get(self, component_type: ComponentType, name: str) -> Type:
        """Retrieve a registered class by type and name."""
        if name not in self._registry[component_type]:
            available = list(self._registry[component_type].keys())
            raise ValueError(
                f"{component_type.value} '{name}' not found. Available: {available}"
            )
        return self._registry[component_type][name]

    def list_available(self, component_type: ComponentType) -> List[str]:
        """List all available components of a given type."""
        return list(self._registry[component_type].keys())

    def create_instance(
        self, component_type: ComponentType, name: str, **kwargs
    ) -> Any:
        """Create an instance of a registered class."""
        cls = self.get(component_type, name)
        signature = inspect.signature(cls.__init__)

        # Filter kwargs to only include parameters that the constructor accepts
        filtered_kwargs = {}
        for param_name, param in signature.parameters.items():
            if param_name != "self" and param_name in kwargs:
                filtered_kwargs[param_name] = kwargs[param_name]
            elif param_name != "self" and param.default is not inspect.Parameter.empty:
                # Use default value if available
                continue
            elif param_name != "self" and param.kind == inspect.Parameter.VAR_KEYWORD:
                # Accept any additional parameters
                filtered_kwargs.update(kwargs)
                break

        return cls(**filtered_kwargs)


# Global registry instance
registry = Registry()


def get_registry() -> Registry:
    """Get the global registry instance."""
    return registry
