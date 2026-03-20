"""QuantTool - A comprehensive quantitative trading platform for A-share stocks."""

# 延迟导入，避免循环依赖
# 子模块在使用时通过 __getattr__ 按需加载
__all__ = [
    'strategies',
    'factors',
    'infrastructure',
    'application',
    'domain',
    'core',
    'web',
    'cli',
    'config',
    'reports',
    'ui',
    'ml',
]

__version__ = "0.1.0"
__author__ = "QuantTool Team"


def __getattr__(name: str):
    """延迟导入子模块，避免循环依赖。"""
    if name in __all__:
        import importlib
        module = importlib.import_module(f'.{name}', __package__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")