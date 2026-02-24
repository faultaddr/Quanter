import pytest
from quanttool.cli.main import app
import typer
from typer.testing import CliRunner

runner = CliRunner()

def test_hello_command():
    """Test the hello command"""
    result = runner.invoke(app, ["hello", "World"])
    assert result.exit_code == 0
    assert "Hello World! Welcome to QuantTool" in result.stdout

def test_plugins_command():
    """Test the plugins command"""
    result = runner.invoke(app, ["plugins"])
    assert result.exit_code == 0
    # Just ensure it doesn't crash
    assert "Available" in result.stdout or len(result.stdout.strip()) > 0

if __name__ == "__main__":
    pytest.main([__file__])