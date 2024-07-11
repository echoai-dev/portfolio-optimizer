# Portfolio Optimization Tool

## Overview

The Portfolio Optimization Tool is designed to optimize a portfolio of ETFs using mean-variance optimization and Monte Carlo simulations. It leverages historical data, GARCH models for volatility forecasting, and various performance metrics such as the Sharpe and Sortino ratios to generate an optimal portfolio. The tool also provides backtesting capabilities to evaluate the performance of the optimized portfolio against historical data.

## Directory Structure

```
./src/
├── __init__.py
├── __pycache__
│   ├── backtesting.cpython-310.pyc
│   ├── data_handler.cpython-310.pyc
│   ├── portfolio_composition.cpython-310.pyc
│   ├── portfolio_optimization.cpython-310.pyc
│   └── visualization.cpython-310.pyc
├── backtesting.py
├── data_handler.py
├── main.py
├── portfolio_composition.py
├── portfolio_optimization.py
├── portfolio_optimizer.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   ├── entry_points.txt
│   ├── requires.txt
│   └── top_level.txt
└── visualization.py
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/yourusername/portfolio-optimization-tool.git
cd portfolio-optimization-tool/src
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Command-Line Arguments

- `--etfs`: List of ETFs to include in the portfolio (default: `['DGRO', 'QQQ', 'SPY', 'VGT', 'VIG', 'VOO', 'VOOG', 'VUG']`)
- `--investment_window`: Investment window for historical data (default: `'3mo'`)
- `--use_sharpe`: Use Sharpe ratio instead of Sortino ratio (flag)
- `--forecast_horizon`: Number of days for forecast (default: `90`)
- `--total_funds`: Total funds available for investment (default: `10000`)

### Running the Tool

```bash
python main.py --etfs DGRO QQQ SPY --investment_window 6mo --use_sharpe --forecast_horizon 90 --total_funds 50000
```

## Project Components

### Data Handling

- **`data_handler.py`**: Fetches historical data for the given ETFs and computes daily returns.

### Portfolio Optimization

- **`portfolio_optimization.py`**: Optimizes the portfolio using mean-variance optimization. Fits GARCH models for volatility forecasting and performs Monte Carlo simulations.

### Backtesting

- **`backtesting.py`**: Backtests the optimized portfolio against historical data and calculates continuous actual probabilities.

### Visualization

- **`visualization.py`**: Generates various plots for Monte Carlo simulations, backtesting results, and portfolio performance metrics.

### Portfolio Composition

- **`portfolio_composition.py`**: Calculates and prints the portfolio composition based on optimized weights.

### Main Script

- **`main.py`**: Orchestrates the data fetching, portfolio optimization, Monte Carlo simulation, backtesting, and visualization. Generates an HTML report with all the plots and portfolio details.

## Example Workflow

1. **Fetch Historical Data**:
   - The tool fetches historical data for the specified ETFs using `yfinance`.
   
2. **Compute Daily Returns**:
   - Computes daily returns for the fetched data.

3. **Optimize Portfolio**:
   - Optimizes the portfolio using mean-variance optimization. Fits GARCH models for volatility forecasting and performs Monte Carlo simulations to evaluate the portfolio.

4. **Monte Carlo Forecast**:
   - Runs Monte Carlo simulations to forecast future portfolio returns.

5. **Backtesting**:
   - Backtests the optimized portfolio against historical data to evaluate its performance.

6. **Generate Plots**:
   - Generates plots for Monte Carlo simulations, backtesting results, and portfolio performance metrics.

7. **Generate HTML Report**:
   - Creates an HTML report with all the plots and portfolio details.

## Output

- **Plots**:
  - Monte Carlo Simulation Results
  - Monte Carlo Forecast
  - Probability of Achieving Return Levels
  - Monte Carlo Forecast vs Backtest
  - Expected vs Actual Probability of Achieving Return Levels

- **HTML Report**:
  - A comprehensive report with all the plots and portfolio details.

## License

This project is licensed under the MIT License.

## Acknowledgements

- `yfinance`: Used for fetching historical data.
- `arch`: Used for GARCH modeling.
- `numpy`, `pandas`, `matplotlib`, `mpld3`: Used for data manipulation and visualization.

## Contact

- For any questions or issues, please contact [Your Name](mailto:your.email@example.com).

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script with:

```
python src/main.py
```

For more options, use:

```
python src/main.py --help
```

## Testing

Run the tests with:

```
python -m unittest discover tests
```

## License

This project is licensed under the MIT License.
