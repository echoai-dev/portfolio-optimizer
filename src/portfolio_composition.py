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