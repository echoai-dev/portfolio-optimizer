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
from portfolio_composition import portfolio_composition

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        portfolio_composition(args.etfs, optimal_weights, last_close_prices, args.total_funds)

        # Generate and save HTML report
        html_content = generate_html_report(figures, args.etfs, optimal_weights)
        save_html_report(html_content)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()