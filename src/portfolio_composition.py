import logging
from typing import List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def portfolio_composition(etfs: List[str], weights: np.ndarray, 
                          last_close_prices: pd.Series, total_funds: float) -> Tuple[np.ndarray, float]:
    """Calculate and print the portfolio composition based on optimized weights."""
    allocation_funds = total_funds * weights
    whole_units = np.floor(allocation_funds / last_close_prices).astype(int)
    remaining_funds = total_funds - np.sum(whole_units * last_close_prices)
    
    logger.info("\nPortfolio Composition:")
    for etf, units, allocation in zip(etfs, whole_units, allocation_funds):
        logger.info(f"{etf}: Buy {units} units (${allocation:.2f})")
    logger.info(f"Remaining Funds: ${remaining_funds:.2f}")
    
    return whole_units, remaining_funds

def portfolio_composition_with_confidence_intervals(etfs: List[str], mean_weights: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray, 
                          last_close_prices: pd.Series, total_funds: float) -> Tuple[np.ndarray, float]:
    """Calculate and print the portfolio composition based on optimized weights and their confidence intervals."""
    allocation_funds = total_funds * mean_weights
    lower_allocation_funds = total_funds * lower_bound
    upper_allocation_funds = total_funds * upper_bound
    
    whole_units = np.floor(allocation_funds / last_close_prices).astype(int)
    lower_whole_units = np.floor(lower_allocation_funds / last_close_prices).astype(int)
    upper_whole_units = np.floor(upper_allocation_funds / last_close_prices).astype(int)
    
    remaining_funds = total_funds - np.sum(whole_units * last_close_prices)
    lower_remaining_funds = total_funds - np.sum(lower_whole_units * last_close_prices)
    upper_remaining_funds = total_funds - np.sum(upper_whole_units * last_close_prices)
    
    logger.info("\nPortfolio Composition:")
    for etf, units, lower_units, upper_units, allocation, lb, ub in zip(etfs, whole_units, lower_whole_units, upper_whole_units, allocation_funds, lower_bound, upper_bound):
        logger.info(f"{etf}: Buy {units} units (95% CI: {lower_units} - {upper_units} units) [${allocation:.2f} (95% CI: ${lb*total_funds:.2f} - ${ub*total_funds:.2f})]")
    
    logger.info(f"Remaining Funds: ${remaining_funds:.2f} (95% CI: ${lower_remaining_funds:.2f} - ${upper_remaining_funds:.2f})")
    
    return whole_units, remaining_funds, lower_whole_units, upper_whole_units, allocation_funds, lower_bound*allocation_funds, upper_bound*allocation_funds