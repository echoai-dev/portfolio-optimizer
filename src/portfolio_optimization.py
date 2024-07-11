import numpy as np
import pandas as pd
from numba import jit
from typing import Tuple, Dict
from scipy.optimize import minimize
import logging
from arch.__future__ import reindexing
from arch import arch_model

logger = logging.getLogger(__name__)

def fit_garch_and_forecast_volatility(returns: pd.DataFrame) -> Dict[str, float]:
    """Fit GARCH models and forecast volatility for each ETF."""
    volatility_forecasts = {}
    for etf, etf_returns in returns.items():
        mean = etf_returns.mean()
        std = etf_returns.std()
        normalized_returns = (etf_returns - mean) / std
        model = arch_model(normalized_returns, vol='Garch', p=1, o=0, q=1)
        res = model.fit(disp='off')
        forecasts = res.forecast(start=0)
        volatility_forecasts[etf] = np.sqrt(forecasts.variance.iloc[-1])*std
    return volatility_forecasts

@jit(nopython=True)
def calculate_mean(array: np.ndarray) -> np.ndarray:
    """Calculate mean of 2D array along axis 0."""
    return np.sum(array, axis=0) / array.shape[0]

@jit(nopython=True)
def calculate_cov(x: np.ndarray) -> np.ndarray:
    """Calculate covariance matrix."""
    mean = calculate_mean(x)
    n = x.shape[0]
    return np.dot((x - mean).T, (x - mean)) / (n - 1)

@jit(nopython=True)
def calculate_portfolio_metrics(weights: np.ndarray, returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float, use_sharpe: bool) -> Tuple[float, float, float]:
    """Calculate portfolio return, volatility, and Sharpe/Sortino ratio."""
    mean_returns = calculate_mean(returns)
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    if use_sharpe:
        ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    else:
        downside_returns = np.minimum(returns - risk_free_rate / 252, 0)
        downside_deviation = np.sqrt(np.sum(np.square(downside_returns)) / (len(returns) - 1)) * np.sqrt(252)
        ratio = (portfolio_return - risk_free_rate) / downside_deviation
    
    return portfolio_return, portfolio_volatility, ratio

def monte_carlo_simulation(returns: pd.DataFrame, volatility_forecasts: Dict[str, float], num_portfolios: int = 10000, risk_free_rate: float = 0.01, use_sharpe: bool = True) -> np.ndarray:
    """Perform Monte Carlo simulation to optimize portfolio."""
    num_assets = len(returns.columns)
    results = np.zeros((3 + num_assets, num_portfolios))
    mean_returns = returns.mean()

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        portfolio_return = np.sum(weights * mean_returns)
        portfolio_stddev = sum(weights[j] * volatility_forecasts[etf] for j, etf in enumerate(returns.columns))
        
        if use_sharpe:
            ratio = (portfolio_return - risk_free_rate) / portfolio_stddev
        else:
            portfolio_downside_dev = sum(weights[j] * downside_deviation(returns[etf]) for j, etf in enumerate(returns.columns))
            ratio = (portfolio_return - risk_free_rate) / portfolio_downside_dev

        results[0,i] = float(portfolio_return) 
        results[1,i] = float(portfolio_stddev.iloc[0])
        results[2,i] = float(ratio.iloc[0])
        results[3:,i] = weights

    return results

def downside_deviation(returns, target_return=0):
    """Calculate the downside deviation of a series of returns."""
    negative_returns = returns[returns < target_return]
    return np.std(negative_returns, ddof=1)

def optimize_portfolio(returns: pd.DataFrame, risk_free_rate: float = 0.02, use_sharpe: bool = True, max_weight: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Optimize portfolio using mean-variance optimization."""
    volatility_forecasts = fit_garch_and_forecast_volatility(returns)
    results = monte_carlo_simulation(returns, volatility_forecasts, use_sharpe=use_sharpe)
    
    # Find the portfolio with the highest Sharpe/Sortino ratio
    best_portfolio_index = np.argmax(results[2])
    optimal_weights = results[3:, best_portfolio_index]
    optimal_metrics = results[:3, best_portfolio_index]

    logger.info(f"Optimize portfolio - n_assets: {len(returns.columns)}")
    logger.info(f"Optimize portfolio - optimal_weights shape: {optimal_weights.shape}")
    logger.info(f"Optimize portfolio - optimal_weights: {optimal_weights}")
    
    return optimal_weights, optimal_metrics, results

def monte_carlo_forecast(returns: pd.DataFrame, mean_weights: np.ndarray, cov_matrix: np.ndarray, forecast_horizon: int = 90, num_simulations: int = 10000) -> np.ndarray:
    """Perform Monte Carlo forecast for portfolio returns using multivariate normal distribution for weights."""
    mean_returns = returns.mean().values
    cov_returns = returns.cov().values
    
    # Generate random weights from the multivariate normal distribution
    random_weights = np.random.multivariate_normal(mean_weights, cov_matrix, size=num_simulations)
    
    # Ensure that random_weights sums to 1
    random_weights /= random_weights.sum(axis=1)[:,np.newaxis]
    
    simulations = np.random.multivariate_normal(mean_returns, cov_returns, size=(num_simulations, forecast_horizon))
    
    # Reshape random_weights for broadcasting
    # random_weights = random_weights[:, np.newaxis, :]
    # import pdb; pdb.set_trace()
    # Adjust einsum to correctly handle the dimensions
    portfolio_simulations = np.einsum('ijk,ik->ij', simulations, random_weights)
    
    return np.cumprod(1 + portfolio_simulations.squeeze(), axis=1) - 1

