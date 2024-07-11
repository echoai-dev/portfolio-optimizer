# Getting Started

## Prerequisites
Ensure you have Python 3.8+ installed on your machine. The tool requires several libraries, which can be installed using the provided `requirements.txt` file.

## Installation

1. Clone the repository:

    ```bash
    git clone <repository_url>
    cd portfolio_optimizer
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Install the package:

    ```bash
    python setup.py install
    ```

## Usage

To run the portfolio optimization tool, use the following command:

```bash
portfolio_optimizer --etfs DGRO QQQ SPY VGT VIG VOO VOOG VUG --investment_window 3mo --use_sharpe --forecast_horizon 90 --total_funds 10000
```

This will fetch historical data for the specified ETFs, optimize the portfolio based on the Sharpe ratio, perform Monte Carlo simulations, backtest the optimized portfolio, and generate an HTML report.

