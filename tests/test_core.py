import pytest
from quanttool.core.registry import Registry, ComponentType, registry

def test_registry_creation():
    """Test that the registry can be created and accessed."""
    assert registry is not None
    assert isinstance(registry, Registry)

def test_component_type_enum():
    """Test that ComponentType enum has expected values."""
    assert hasattr(ComponentType, 'DATA_PROVIDER')
    assert hasattr(ComponentType, 'STRATEGY')
    assert hasattr(ComponentType, 'FACTOR')
    assert hasattr(ComponentType, 'MODEL')

def test_registry_register_and_get():
    """Test basic register/get functionality of the registry."""
    # Create a mock class for testing
    class MockProvider:
        pass

    # Register the mock class
    registry.register(ComponentType.DATA_PROVIDER, "mock")(MockProvider)

    # Retrieve the class
    retrieved_class = registry.get(ComponentType.DATA_PROVIDER, "mock")
    assert retrieved_class == MockProvider

if __name__ == "__main__":
    pytest.main([__file__])