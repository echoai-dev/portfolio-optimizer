import logging
import yfinance as yf
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)

def fetch_data(etfs: List[str], period: str) -> pd.DataFrame:
    """Fetch historical data for the given ETFs and period."""
    try:
        data = {etf: yf.Ticker(etf).history(period=period)['Close'] for etf in etfs}
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Failed to fetch data: {str(e)}")
        raise

def compute_daily_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns for the given data."""
    return data.pct_change().dropna()