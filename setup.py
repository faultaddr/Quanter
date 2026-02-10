from setuptools import setup, find_packages

setup(
    name="quant_trade_a_share",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.1.0",
        "numpy",
        "tushare>=1.2.0",
        "baostock>=0.8.0",
        "scipy>=1.5.0",
        "requests>=2.25.0",
    ],
    extras_require={
        'visualization': [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "plotly>=4.0.0",
            "dash>=2.0.0",
        ],
        'advanced': [
            "yfinance>=0.1.0",
            "statsmodels>=0.12.0",
            "qlib>=0.9.0",
        ]
    },
    author="A-Share Analysis Team",
    author_email="a-share-analysis@example.com",
    description="A unified quantitative trading tool for A-Share market",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/quant-trade-a-share",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            'quant-trade-a-share=cli_interface:main',
        ],
    },
)