import argparse
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
from data_handler import fetch_data, compute_daily_returns
from portfolio_optimization import optimize_portfolio, monte_carlo_forecast
from visualization import (plot_monte_carlo, plot_forecast, plot_probabilities,
                           plot_backtest_vs_forecast, plot_backtest_returns_probabilities,
                           generate_html_report, save_html_report)
from backtesting import backtest, calculate_continuous_actual_probabilities
from portfolio_composition import portfolio_composition_with_cov_matrix as portfolio_composition

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from joblib import Parallel, delayed, cpu_count
from typing import Tuple

def bootstrap_sample(returns: pd.DataFrame, block_size: int = 5) -> np.ndarray:
    """Generate a bootstrap sample using block bootstrap to maintain correlation structure."""
    num_blocks = int(np.ceil(len(returns) / block_size))
    indices = np.concatenate([np.random.randint(0, len(returns) - block_size + 1, size=num_blocks) for _ in range(block_size)])
    indices = indices[:len(returns)]
    sample_returns = returns.iloc[indices]
    optimal_weights, _, results = optimize_portfolio(sample_returns)
    return optimal_weights, results

def bootstrap_portfolio_weights(returns: pd.DataFrame, num_samples: int = 100, block_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Generate bootstrap samples of portfolio weights and calculate the mean and covariance matrix."""
    num_cores = min(num_samples, cpu_count() - 1)
    bootstrap_samples = Parallel(n_jobs=num_cores)(
        delayed(bootstrap_sample)(returns, block_size) for _ in tqdm(range(num_samples), desc="Bootstrap")
        )
    bootstrap_weights = np.array([sample[0] for sample in bootstrap_samples])
    #HACK: use the mean of the results
    results = np.array([sample[1] for sample in bootstrap_samples]).mean(axis=0)
    
    logger.info(f"Bootstrap - optimal_weights shape: {bootstrap_weights.shape}")
    logger.info(f"Bootstrap - results shape: {results.shape}")
    
    mean_weights = np.mean(bootstrap_weights, axis=0)
    cov_matrix = np.cov(bootstrap_weights, rowvar=False)
    
    return mean_weights, cov_matrix, results

def parse_arguments():
    parser = argparse.ArgumentParser(description='Portfolio Optimization Tool')
    parser.add_argument('--etfs', nargs='+', default=['DGRO', 'QQQ', 'SPY', 'VGT', 'VIG', 'VOO', 'VOOG', 'VUG'],
                        help='List of ETFs to include in the portfolio')
    parser.add_argument('--investment_window', default='3mo', help='Investment window for historical data')
    parser.add_argument('--use_sharpe', action='store_true', help='Use Sharpe ratio instead of Sortino ratio')
    parser.add_argument('--forecast_horizon', type=int, default=90, help='Number of days for forecast')
    parser.add_argument('--total_funds', type=float, default=10000, help='Total funds available for investment')
    return parser.parse_args()

def main():
    args = parse_arguments()

    try:
        # Fetch data and compute returns
        data = fetch_data(args.etfs, args.investment_window)
        returns = compute_daily_returns(data)

        logger.info(f"Number of assets: {returns.shape[1]}")
        logger.info(f"ETFs: {args.etfs}")

        # Calculate mean and covariance matrix for portfolio weights
        mean_weights, cov_matrix, bootstrap_results = bootstrap_portfolio_weights(returns, num_samples=100, block_size=5)
        
        # Display the results
        print("Optimal Portfolio Weights:")
        for etf, mean_weight in zip(args.etfs, mean_weights):
            print(f"{etf}: {mean_weight:.4f}")
        
        print("\nCovariance Matrix of Portfolio Weights:")
        print(cov_matrix)

        # Perform Monte Carlo forecast with uncertainty propagation
        monte_carlo_results = monte_carlo_forecast(returns, mean_weights, cov_matrix, forecast_horizon=args.forecast_horizon)

        # Generate plots
        figures = []
        figures.append(plot_monte_carlo(bootstrap_results, args.etfs))

        return_levels = [0.01, 0.06, 0.08, 0.10, 0.15]

        # Backtest
        cumulative_backtest_returns = backtest(data, mean_weights, forecast_horizon=args.forecast_horizon)
        figures.append(plot_backtest_vs_forecast(monte_carlo_results, cumulative_backtest_returns))

        # Calculate and plot continuous actual probabilities
        continuous_actual_probabilities = calculate_continuous_actual_probabilities(monte_carlo_results, cumulative_backtest_returns)
        figures.append(plot_backtest_returns_probabilities(monte_carlo_results, return_levels, continuous_actual_probabilities))

        logger.info(f"Log of the Continuous Actual Probabilities: {np.log(continuous_actual_probabilities.prod())}")

        # Calculate portfolio composition
        last_close_prices = data.iloc[-1]
        whole_units, remaining_funds, lower_whole_units, upper_whole_units, allocation_funds, lower_allocation_funds, upper_allocation_funds = portfolio_composition(args.etfs, mean_weights, cov_matrix, last_close_prices, args.total_funds)

        # Generate and save HTML report
        html_content = generate_html_report(figures, args.etfs, mean_weights, whole_units, remaining_funds, lower_whole_units, upper_whole_units, allocation_funds, lower_allocation_funds, upper_allocation_funds)
        save_html_report(html_content)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()