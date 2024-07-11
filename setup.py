from setuptools import setup, find_packages

setup(
    name="portfolio_optimizer",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "yfinance",
        "arch",
        "mpld3",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "portfolio_optimizer=main:main",
        ],
    },
)
