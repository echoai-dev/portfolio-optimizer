import argparse
import logging
import numpy as np
import pandas as pd
from data_handler import fetch_data, compute_daily_returns
from portfolio_optimization import optimize_portfolio, monte_carlo_forecast
from visualization import (plot_monte_carlo, plot_forecast, plot_probabilities,
                           plot_backtest_vs_forecast, plot_backtest_returns_probabilities,
                           generate_html_report, save_html_report)
from backtesting import backtest, calculate_continuous_actual_probabilities
from portfolio_composition import portfolio_composition_with_confidence_intervals as portfolio_composition

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from joblib import Parallel, delayed, cpu_count
from typing import Tuple

def bootstrap_sample(returns: pd.DataFrame) -> np.ndarray:
    """
    Generate bootstrap samples of portfolio weights and calculate the mean, lower bound, and upper bound.

    Parameters:
        returns (pd.DataFrame): The DataFrame containing the returns of the portfolio.

    Returns:
        np.ndarray: The optimal weights for the portfolio.
    """
    sample_indices = np.random.choice(returns.index, size=len(returns), replace=True)
    sample_returns = returns.loc[sample_indices]
    optimal_weights, _, _ = optimize_portfolio(sample_returns)
    return optimal_weights

def bootstrap_portfolio_weights(returns: pd.DataFrame, num_samples: int = 100, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate bootstrap samples of portfolio weights and calculate the mean, lower bound, and upper bound.

    Parameters:
        returns (pd.DataFrame): The DataFrame containing the returns of the portfolio.
        num_samples (int, optional): The number of bootstrap samples to generate. Default is 100.
        confidence_level (float, optional): The confidence level for the confidence interval. Default is 0.95.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the mean weights, lower bound, and upper bound.
    """
    num_cores = min(num_samples, cpu_count()-1)
    bootstrap_weights = Parallel(n_jobs=num_cores)(delayed(bootstrap_sample)(returns) for _ in range(num_samples))
    bootstrap_weights = np.array(bootstrap_weights)
    
    lower_bound = np.percentile(bootstrap_weights, (1 - confidence_level) / 2 * 100, axis=0)
    upper_bound = np.percentile(bootstrap_weights, (1 - (1 - confidence_level) / 2) * 100, axis=0)
    mean_weights = np.mean(bootstrap_weights, axis=0)
    
    return mean_weights, lower_bound, upper_bound


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

        # Calculate confidence bounds for portfolio weights
        mean_weights, lower_bound, upper_bound = bootstrap_portfolio_weights(returns, num_samples=100, confidence_level=0.95)
        
        # Display the results
        print("Optimal Portfolio Weights with Confidence Bounds:")
        for etf, mean_weight, lb, ub in zip(args.etfs, mean_weights, lower_bound, upper_bound):
            print(f"{etf}: {mean_weight:.4f} (95% CI: {lb:.4f} - {ub:.4f})")

        # Optimize portfolio
        optimal_weights, optimal_metrics, monte_carlo = optimize_portfolio(returns, use_sharpe=args.use_sharpe)

        logger.info(f"Optimal weights shape: {optimal_weights.shape}")
        logger.info(f"Optimal weights: {optimal_weights}")

        # Perform Monte Carlo forecast
        monte_carlo_results = monte_carlo_forecast(returns, optimal_weights, forecast_horizon=args.forecast_horizon)

        # Generate plots
        figures = []
        figures.append(plot_monte_carlo(monte_carlo, args.etfs))
        # figures.append(plot_forecast(monte_carlo_results))

        return_levels = [0.01, 0.06, 0.08, 0.10, 0.15]
        # figures.append(plot_probabilities(monte_carlo_results, return_levels))

        # Backtest
        cumulative_backtest_returns = backtest(data, optimal_weights, forecast_horizon=args.forecast_horizon)
        figures.append(plot_backtest_vs_forecast(monte_carlo_results, cumulative_backtest_returns))

        # Calculate and plot continuous actual probabilities
        continuous_actual_probabilities = calculate_continuous_actual_probabilities(monte_carlo_results, cumulative_backtest_returns)
        figures.append(plot_backtest_returns_probabilities(monte_carlo_results, return_levels, continuous_actual_probabilities))

        logger.info(f"Log of the Continuous Actual Probabilities: {np.log(continuous_actual_probabilities.prod())}")

        # Calculate portfolio composition
        last_close_prices = data.iloc[-1]
        whole_units, remaining_funds, lower_whole_units, upper_whole_units, allocation_funds, lower_allocation_funds, upper_allocation_funds = portfolio_composition(args.etfs, mean_weights, lower_bound, upper_bound, last_close_prices, args.total_funds)
        # Generate and save HTML report
        html_content = generate_html_report(figures, args.etfs, optimal_weights,whole_units, remaining_funds, lower_whole_units, upper_whole_units, allocation_funds, lower_allocation_funds, upper_allocation_funds)
        save_html_report(html_content)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()