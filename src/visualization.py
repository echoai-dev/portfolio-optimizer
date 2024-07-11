import logging
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins

logger = logging.getLogger(__name__)

def plot_monte_carlo(results: np.ndarray, etfs: List[str]) -> plt.Figure:
    """Plot the Monte Carlo simulations and print details of the optimal portfolio."""
    return_sharpe_max = results[0,results[2].argmax()]
    risk_sharpe_max = results[1,results[2].argmax()]
    weights_sharpe_max = results[3:,results[2].argmax()]

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(results[1,:], results[0,:], c=results[2], cmap='viridis')
    scatter = ax.scatter(risk_sharpe_max, return_sharpe_max, c='red', marker='*', s=100)
    plt.colorbar(scatter, label='Sharpe Ratio')
    ax.set_xlabel('Portfolio Volatility')
    ax.set_ylabel('Portfolio Return')
    ax.set_title('Monte Carlo Simulation Results')
    plt.tight_layout()

    # Add a tooltip to show ETF weights
    # labels = [f"{etf}: {weight:.4f}" for etf, weight in zip(etfs, results[3:])]
    # tooltip = plugins.PointLabelTooltip(scatter, labels=labels)
    # plugins.connect(fig, tooltip)

    return fig

def plot_forecast(monte_carlo_results: np.ndarray) -> plt.Figure:
    """Plot the Monte Carlo simulations of portfolio returns over the forecast horizon."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(min(1000, monte_carlo_results.shape[0])):  # Plot up to 1000 simulations
        ax.plot(monte_carlo_results[i], color='blue', alpha=0.1)
    ax.set_xlabel('Days')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('Monte Carlo Forecast')
    plt.tight_layout()
    return fig

def plot_probabilities(monte_carlo_results: np.ndarray, return_levels: List[float]) -> plt.Figure:
    """Plot the probability of achieving different return levels over time."""
    probabilities = np.mean(monte_carlo_results[:, None, :] > np.array(return_levels)[:, None], axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, level in enumerate(return_levels):
        ax.plot(probabilities[i], label=f'{level:.0%}')
    ax.set_xlabel('Days')
    ax.set_ylabel('Probability')
    ax.set_title('Probability of Achieving Return Levels')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_backtest_vs_forecast(monte_carlo_results: np.ndarray, cumulative_backtest_returns: np.ndarray) -> plt.Figure:
    """Plot Monte Carlo simulations vs. backtested portfolio returns."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(min(1000, monte_carlo_results.shape[0])):  # Plot up to 1000 simulations
        ax.plot(monte_carlo_results[i], color='blue', alpha=0.1)
    ax.plot(cumulative_backtest_returns, color='red', linewidth=2, label='Backtest')
    ax.set_xlabel('Days')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('Monte Carlo Forecast vs Backtest')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_backtest_returns_probabilities(monte_carlo_results: np.ndarray, return_levels: List[float], 
                                        continuous_actual_probabilities: np.ndarray) -> plt.Figure:
    """Plot expected vs. continuous actual probability of achieving different return levels over time."""
    probabilities = np.mean(monte_carlo_results[:, None, :] > np.array(return_levels)[:, None], axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, level in enumerate(return_levels):
        ax.plot(probabilities[i], label=f'Expected {level:.0%}')
    ax.plot(continuous_actual_probabilities, color='black', linewidth=2, label='Actual')
    ax.set_xlabel('Days')
    ax.set_ylabel('Probability')
    ax.set_title('Expected vs Actual Probability of Achieving Return Levels')
    ax.legend()
    plt.tight_layout()
    return fig

def generate_html_report(figures: List[plt.Figure], etfs: List[str], optimal_weights: np.ndarray, whole_units: np.ndarray, remaining_funds: np.ndarray, lower_whole_units: np.ndarray, upper_whole_units: np.ndarray, allocation_funds: np.ndarray, lower_allocation_funds: np.ndarray, upper_allocation_funds: np.ndarray) -> str:
    """Generate an HTML report with all the plots."""
    html_content = "<html><head><title>Portfolio Optimization Report</title></head><body>"
    
    # Add optimal portfolio weights
    html_content += "<h2>Optimal Portfolio Weights</h2>"
    html_content += "<table border='1'><tr><th>ETF</th><th>Weight</th><th>Units</th><th>Lower Units</th><th>Upper Units</th><th>Allocation Funds</th><th>Lower Allocation Funds</th><th>Upper Allocation Funds</th></tr>"
    for etf, weight, units, lower_units, upper_units, allocation, lower_allocation, upper_allocation in zip(etfs, optimal_weights, whole_units,lower_whole_units, upper_whole_units, allocation_funds, lower_allocation_funds, upper_allocation_funds):
        html_content += f"<tr><td>{etf}</td><td>{weight:.4f}</td><td>{units}</td><td>{lower_units}</td><td>{upper_units}</td><td>${allocation:.2f}</td><td>${lower_allocation:.2f}</td><td>${upper_allocation:.4f}</td></tr>"
    html_content += "</table>"

    # Add all figures
    for i, fig in enumerate(figures):
        html_content += mpld3.fig_to_html(fig)
        plt.close(fig)  # Close the figure to free up memory

    html_content += "</body></html>"
    return html_content

def save_html_report(html_content: str, filename: str = "portfolio_optimization_report.html") -> None:
    """Save the HTML report to a file."""
    with open(filename, "w") as f:
        f.write(html_content)
    logger.info(f"HTML report saved as {filename}")