import numpy as np
import pandas as pd

def backtest(data: pd.DataFrame, optimal_weights: np.ndarray, forecast_horizon: int = 90) -> np.ndarray:
    """Perform backtesting of the optimized portfolio."""
    backtest_data = data.iloc[len(data)-forecast_horizon:]
    backtest_returns = backtest_data.pct_change().dropna()
    backtest_portfolio_returns = np.dot(optimal_weights, backtest_returns.T)
    cumulative_returns = np.cumprod(1 + backtest_portfolio_returns) - 1
    return cumulative_returns

def calculate_continuous_actual_probabilities(monte_carlo_results: np.ndarray, cumulative_backtest_returns: np.ndarray) -> np.ndarray:
    """Calculate continuous actual probabilities based on Monte Carlo results and backtest returns."""
    min_length = min(monte_carlo_results.shape[1], len(cumulative_backtest_returns))
    monte_carlo_results = monte_carlo_results[:, :min_length]
    cumulative_backtest_returns = cumulative_backtest_returns[:min_length]
    return np.mean(monte_carlo_results < cumulative_backtest_returns, axis=0)