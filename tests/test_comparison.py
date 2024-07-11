import unittest
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from src.data_handler import fetch_data as new_fetch_data, compute_daily_returns as new_compute_daily_returns
from src.portfolio_optimization import fit_garch_and_forecast_volatility as new_fit_garch_and_forecast_volatility, monte_carlo_simulation as new_monte_carlo_simulation
from src.backtesting import backtest as new_backtest
from prototype import fetch_data as proto_fetch_data, compute_daily_returns as proto_compute_daily_returns
from prototype import fit_garch_and_forecast_volatility as proto_fit_garch_and_forecast_volatility, monte_carlo_simulation as proto_monte_carlo_simulation
from prototype import backtest as proto_backtest

class TestComparison(unittest.TestCase):
    
    def setUp(self):
        self.etfs = ['DGRO', 'QQQ', 'SPY', 'VGT', 'VIG', 'VOO', 'VOOG', 'VUG']
        self.period = '3mo'
        self.forecast_horizon = 90

        # Fetch data
        self.new_data = new_fetch_data(self.etfs, self.period)
        self.proto_data = proto_fetch_data(self.etfs, self.period)

        # Compute returns
        self.new_returns = new_compute_daily_returns(self.new_data)
        self.proto_returns = proto_compute_daily_returns(self.proto_data)

        # Fit GARCH models and forecast volatility
        self.new_volatility_forecasts = new_fit_garch_and_forecast_volatility(self.new_returns)
        self.proto_volatility_forecasts = proto_fit_garch_and_forecast_volatility(self.proto_returns)

    def test_fetch_data(self):
        pd.testing.assert_frame_equal(self.new_data, self.proto_data, check_dtype=False)

    def test_compute_daily_returns(self):
        pd.testing.assert_frame_equal(self.new_returns, self.proto_returns, check_dtype=False)

    def test_fit_garch_and_forecast_volatility(self):
        for etf in self.etfs:
            new_forecast = self.new_volatility_forecasts[etf].item() if isinstance(self.new_volatility_forecasts[etf], pd.Series) else self.new_volatility_forecasts[etf]
            proto_forecast = self.proto_volatility_forecasts[etf].item() if isinstance(self.proto_volatility_forecasts[etf], pd.Series) else self.proto_volatility_forecasts[etf]
            self.assertAlmostEqual(new_forecast, proto_forecast, places=6)

    def test_monte_carlo_simulation(self):
        new_results = new_monte_carlo_simulation(self.new_returns, self.new_volatility_forecasts)
        proto_results = proto_monte_carlo_simulation(self.proto_returns, self.proto_volatility_forecasts)

        differences = new_results - proto_results

        # Flatten the arrays for comparison
        differences = differences.flatten()

        # # Flatten the arrays for comparison
        # new_flattened = new_results.flatten()
        # proto_flattened = proto_results.flatten()

        # # Calculate the differences
        # differences = new_flattened - proto_flattened

        print(differences.mean(), differences.std(), differences.max(), differences.min())

        # Perform a t-test on the mean of the differences
        # t_statistic, p_value = ttest_rel(new_flattened, proto_flattened)
        
        # self.assertGreater(p_value, 0.05, f"T-test failed: t-statistic = {t_statistic}, p-value = {p_value}")
        self.assertAlmostEqual(differences.mean(), 0, places=6, msg="Mean of differences is not close to zero")

    def test_backtest(self):
        optimal_weights = np.array([0.1] * len(self.etfs))  # Placeholder for optimal weights
        
        new_cumulative_returns = new_backtest(self.new_data, optimal_weights, self.forecast_horizon)
        proto_cumulative_returns = proto_backtest(self.proto_data, optimal_weights, self.forecast_horizon)
        # Calculate the differences
        differences = new_cumulative_returns - proto_cumulative_returns

        # Perform a t-test on the mean of the differences
        # t_statistic, p_value = ttest_rel(new_cumulative_returns, proto_cumulative_returns)
        
        # self.assertGreater(p_value, 0.05, f"T-test failed: t-statistic = {t_statistic}, p-value = {p_value}")
        self.assertAlmostEqual(differences.mean(), 0, places=6, msg="Mean of differences is not close to zero")

if __name__ == '__main__':
    unittest.main()
