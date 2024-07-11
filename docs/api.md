# API Documentation

## `main.py`

- **parse_arguments()**
    - Parses command-line arguments for the portfolio optimization tool.
    - **Returns**: Parsed arguments.

- **main()**
    - Main function to execute the portfolio optimization process.
    - **Arguments**: None.

## `data_handler.py`

- **fetch_data(etfs: List[str], period: str) -> pd.DataFrame**
    - Fetches historical data for the given ETFs and period.
    - **Arguments**:
        - `etfs`: List of ETF symbols.
        - `period`: Time period for fetching historical data.
    - **Returns**: DataFrame with historical closing prices.

- **compute_daily_returns(data: pd.DataFrame) -> pd.DataFrame**
    - Computes daily returns for the given data.
    - **Arguments**:
        - `data`: DataFrame with historical closing prices.
    - **Returns**: DataFrame with daily returns.

## `portfolio_optimization.py`

- **optimize_portfolio(returns: pd.DataFrame, risk_free_rate: float = 0.02, use_sharpe: bool = True) -> Tuple[np.ndarray, np.ndarray]**
    - Optimizes the portfolio using mean-variance optimization.
    - **Arguments**:
        - `returns`: DataFrame with daily returns.
        - `risk_free_rate`: Risk-free rate for calculating Sharpe/Sortino ratio.
        - `use_sharpe`: Boolean flag to use Sharpe ratio.
    - **Returns**: Tuple containing optimal weights, optimal metrics, and Monte Carlo simulation results.

- **monte_carlo_forecast(returns: pd.DataFrame, weights: np.ndarray, forecast_horizon: int = 90, num_simulations: int = 10000) -> np.ndarray**
    - Performs Monte Carlo forecast for portfolio returns.
    - **Arguments**:
        - `returns`: DataFrame with daily returns.
        - `weights`: Array of portfolio weights.
        - `forecast_horizon`: Number of days for the forecast.
        - `num_simulations`: Number of simulations to run.
    - **Returns**: 2D array of simulated portfolio returns.

## `backtesting.py`

- **backtest(data: pd.DataFrame, optimal_weights: np.ndarray, forecast_horizon: int = 90) -> np.ndarray**
    - Performs backtesting of the optimized portfolio.
    - **Arguments**:
        - `data`: DataFrame with historical data.
        - `optimal_weights`: Array of optimal portfolio weights.
        - `forecast_horizon`: Number of days for backtesting.
    - **Returns**: Array of cumulative returns from backtesting.

- **calculate_continuous_actual_probabilities(monte_carlo_results: np.ndarray, cumulative_backtest_returns: np.ndarray) -> np.ndarray**
    - Calculates continuous actual probabilities based on Monte Carlo results and backtest returns.
    - **Arguments**:
        - `monte_carlo_results`: 2D array of Monte Carlo simulation results.
        - `cumulative_backtest_returns`: Array of cumulative returns from backtesting.
    - **Returns**: Array of continuous actual probabilities.

## `portfolio_composition.py`

- **portfolio_composition(etfs: List[str], weights: np.ndarray, last_close_prices: pd.Series, total_funds: float) -> Tuple[np.ndarray, float]**
    - Calculates and prints the portfolio composition based on optimized weights.
    - **Arguments**:
        - `etfs`: List of ETF symbols.
        - `weights`: Array of optimal portfolio weights.
        - `last_close_prices`: Series with the last close prices of the ETFs.
        - `total_funds`: Total funds available for investment.
    - **Returns**: Tuple containing whole units of ETFs and remaining funds.

## `visualization.py`

- **plot_monte_carlo(results: np.ndarray, etfs: List[str]) -> plt.Figure**
    - Plots the Monte Carlo simulations and prints details of the optimal portfolio.
    - **Arguments**:
        - `results`: 2D array of Monte Carlo simulation results.
        - `etfs`: List of ETF symbols.
    - **Returns**: Matplotlib figure.

- **plot_forecast(monte_carlo_results: np.ndarray) -> plt.Figure**
    - Plots the Monte Carlo simulations of portfolio returns over the forecast horizon.
    - **Arguments**:
        - `monte_carlo_results`: 2D array of Monte Carlo simulation results.
    - **Returns**: Matplotlib figure.

- **plot_probabilities(monte_carlo_results: np.ndarray, return_levels: List[float]) -> plt.Figure**
    - Plots the probability of achieving different return levels over time.
    - **Arguments**:
        - `monte_carlo_results`: 2D array of Monte Carlo simulation results.
        - `return_levels`: List of return levels to plot probabilities for.
    - **Returns**: Matplotlib figure.

- **plot_backtest_vs_forecast(monte_carlo_results: np.ndarray, cumulative_backtest_returns: np.ndarray) -> plt.Figure**
    - Plots Monte Carlo simulations vs. backtested portfolio returns.
    - **Arguments**:
        - `monte_carlo_results`: 2D array of Monte Carlo simulation results.
        - `cumulative_backtest_returns`: Array of cumulative returns from backtesting.
    - **Returns**: Matplotlib figure.

- **plot_backtest_returns_probabilities(monte_carlo_results: np.ndarray, return_levels: List[float], continuous_actual_probabilities: np.ndarray) -> plt.Figure**
    - Plots expected vs. continuous actual probability of achieving different return levels over time.
    - **Arguments**:
        - `monte_carlo_results`: 2D array of Monte Carlo simulation results.
        - `return_levels`: List of return levels to plot probabilities for.
        - `continuous_actual_probabilities`: Array of continuous actual probabilities.
    - **Returns**: Matplotlib figure.

- **generate_html_report(figures: List[plt.Figure], etfs: List[str], optimal_weights: np.ndarray) -> str**
    - Generates an HTML report with all the plots.
    - **Arguments**:
        - `figures`: List of Matplotlib figures.
        - `etfs`: List of ETF symbols.
        - `optimal_weights`: Array of optimal portfolio weights.
    - **Returns**: HTML content as a string.

- **save_html_report(html_content: str, filename: str = "portfolio_optimization_report.html") -> None**
    - Saves the HTML report to a file.
    - **Arguments**:
        - `html_content`: HTML content as a string.
        - `filename`: Name of the file to save the report.
    - **Returns**: None.
