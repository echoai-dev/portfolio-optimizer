#%%
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from multiprocessing import Pool, cpu_count

def fetch_data(etfs, period):
    """Fetch historical data for the given ETFs and period."""
    data = {etf: yf.Ticker(etf).history(period=period)['Close'] for etf in etfs}
    return pd.DataFrame(data)

def compute_daily_returns(data):
    """Compute daily returns for the given data."""
    return data.pct_change().dropna()

def fit_garch_and_forecast_volatility(returns):
    """Fit GARCH models and forecast volatility for each ETF."""
    volatility_forecasts = {}
    for etf, etf_returns in returns.items():
        model = arch_model(etf_returns, vol='Garch', p=1, o=0, q=1)
        res = model.fit(disp='off')
        forecasts = res.forecast(start=0)
        volatility_forecasts[etf] = np.sqrt(forecasts.variance.iloc[-1])
    return volatility_forecasts

def downside_deviation(returns, target_return=0):
    """Calculate the downside deviation of a series of returns."""
    negative_returns = returns[returns < target_return]
    return np.std(negative_returns, ddof=1)

def monte_carlo_simulation(returns, volatility_forecasts, num_portfolios=10000, risk_free_rate=0.01, sharpe=True):
    """Perform Monte Carlo simulation to optimize portfolio based on Sharpe or Sortino ratio."""
    num_assets = len(returns.columns)
    results = np.zeros((3 + num_assets, num_portfolios))
    mean_returns = returns.mean()

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        portfolio_return = np.sum(weights * mean_returns)
        portfolio_stddev = sum(weights[j] * volatility_forecasts[etf] for j, etf in enumerate(returns.columns))
        
        if sharpe:
            ratio = (portfolio_return - risk_free_rate) / portfolio_stddev
        else:
            portfolio_downside_dev = sum(weights[j] * downside_deviation(returns[etf]) for j, etf in enumerate(returns.columns))
            ratio = (portfolio_return - risk_free_rate) / portfolio_downside_dev

        results[0,i] = portfolio_return
        results[1,i] = portfolio_stddev
        results[2,i] = ratio
        for j in range(num_assets):
            results[3+j,i] = weights[j]

    return results

def plot_monte_carlo(results, etfs):
    """Plot the Monte Carlo simulations and print details of the optimal portfolio."""
    return_sharpe_max = results[0,results[2].argmax()]
    risk_sharpe_max = results[1,results[2].argmax()]
    weights_sharpe_max = results[3:,results[2].argmax()]

    plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='YlGnBu', marker='o')
    plt.title('Monte Carlo Simulation of Portfolio')
    plt.xlabel('Portfolio Risk (Standard Deviation)')
    plt.ylabel('Portfolio Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(risk_sharpe_max, return_sharpe_max, c='red', marker='*', s=100)
    plt.show()

    print(f"Optimal Portfolio: Return = {return_sharpe_max}, Risk = {risk_sharpe_max}")
    for etf, weight in zip(etfs, weights_sharpe_max):
        print(f"Weight for {etf}: {weight * 100:.2f}%")

    return weights_sharpe_max


def fit_garch_and_forecast(returns):
    """
    Fit the GARCH model and forecast volatility for each asset.

    Parameters:
    - returns: DataFrame of asset returns.

    Returns:
    - Tuple of forecasted volatilities and mean returns for each asset.
    """
    num_assets = len(returns.columns)
    forecast_volatilities = np.zeros(num_assets)
    mean_returns = np.zeros(num_assets)

    for i, etf in enumerate(returns.columns):
        model = arch_model(returns[etf], vol='Garch', p=1, o=0, q=1)
        res = model.fit(disp='off')
        forecast_volatilities[i] = np.sqrt(
            res.forecast(start=0).variance.iloc[-1]
        ).values[0]
        mean_returns[i] = returns[etf].mean()

    return forecast_volatilities, mean_returns


def simulate_portfolio_returns(args):
    """
    Simulate portfolio returns for one simulation.

    Parameters:
    - args: Tuple containing weights, forecasted volatilities, mean returns,
            and forecast horizon.

    Returns:
    - Array of simulated portfolio returns.
    """
    weights_sharpe_max, forecast_volatilities, mean_returns, forecast_horizon = args
    num_assets = len(mean_returns)
    simulated_returns = np.zeros((num_assets, forecast_horizon))

    for i in range(num_assets):
        simulated_returns[i] = np.random.normal(
            mean_returns[i], forecast_volatilities[i], forecast_horizon
        )

    portfolio_simulated_returns = np.dot(weights_sharpe_max, simulated_returns)
    return np.cumprod(1 + portfolio_simulated_returns) - 1


def monte_carlo_forecast(returns, weights_sharpe_max, forecast_horizon=90, num_simulations=10000):
    """
    Perform Monte Carlo simulations for portfolio returns over a forecast horizon.

    Parameters:
    - returns: DataFrame of asset returns.
    - weights_sharpe_max: Array of portfolio weights.
    - forecast_horizon: Number of periods for the forecast.
    - num_simulations: Number of simulations to run.

    Returns:
    - 2D array of simulated portfolio returns for each simulation.
    """
    forecast_volatilities, mean_returns = fit_garch_and_forecast(returns)

    # Prepare arguments for parallel processing
    args = [
        (weights_sharpe_max, forecast_volatilities, mean_returns, forecast_horizon)
        for _ in range(num_simulations)
    ]

    # Use multiprocessing to parallelize the simulations
    with Pool(cpu_count()) as pool:
        monte_carlo_results = pool.map(simulate_portfolio_returns, args)

    return np.array(monte_carlo_results)


def plot_forecast(monte_carlo_results):
    """Plot the Monte Carlo simulations of portfolio returns over the forecast horizon."""
    plt.figure(figsize=(12, 6))
    plt.plot(monte_carlo_results.T, color='lightblue')
    plt.title('Monte Carlo Simulations of Portfolio Returns over the Forecast Horizon')
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Returns')
    plt.show()

def plot_probabilities(monte_carlo_results, return_levels):
    """Plot the probability of achieving different return levels over time."""
    forecast_horizon = monte_carlo_results.shape[1]
    probabilities = np.zeros((len(return_levels), forecast_horizon))

    for i, level in enumerate(return_levels):
        probabilities[i] = np.mean(monte_carlo_results > level, axis=0)

    plt.figure(figsize=(12, 6))
    for i, level in enumerate(return_levels):
        plt.plot(probabilities[i] * 100, label=f'Above {level * 100}% Return')

    plt.title('Probability of Achieving Different Return Levels Over Time')
    plt.xlabel('Trading Days')
    plt.ylabel('Probability (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

def backtest(data, optimal_weights, forecast_horizon=90):
    # Split data into optimization and backtest datasets
    optimization_data = data.iloc[:len(data) - forecast_horizon]
    backtest_data = data.iloc[len(data) - forecast_horizon:]

    # Compute daily returns for the backtest data
    backtest_returns = backtest_data.pct_change().dropna()

    # Apply the optimal weights to the backtest returns
    backtest_portfolio_returns = np.dot(optimal_weights, backtest_returns.T)

    # Calculate cumulative returns for the backtest period
    cumulative_backtest_returns = np.cumprod(1 + backtest_portfolio_returns) - 1

    return cumulative_backtest_returns

def plot_backtest_vs_forecast(monte_carlo_results, cumulative_backtest_returns):
    plt.figure(figsize=(12, 6))
    plt.plot(monte_carlo_results.T*100, color='lightblue', label='_nolegend_')  # Monte Carlo simulations
    plt.plot(cumulative_backtest_returns * 100, color='red', label='Backtested Portfolio')  # Backtested returns
    plt.title('Monte Carlo Simulations vs. Backtested Portfolio Returns over 3 Months')
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Returns (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_continuous_actual_probabilities(monte_carlo_results, cumulative_backtest_returns):
    forecast_horizon = len(cumulative_backtest_returns)
    continuous_actual_probabilities = np.zeros(forecast_horizon)

    for t in range(forecast_horizon):
        continuous_actual_probabilities[t] = np.mean(monte_carlo_results[:, t] < cumulative_backtest_returns[t])

    return continuous_actual_probabilities

def plot_backtest_returns_probabilities(monte_carlo_results, return_levels, continuous_actual_probabilities):
    plt.figure(figsize=(12, 6))
    for i, level in enumerate(return_levels):
        plt.plot((np.mean(monte_carlo_results > level, axis=0) * 100), label=f'Expected Above {level * 100}% Return')

    # Add the continuous actual probability line
    plt.plot(continuous_actual_probabilities * 100, color='black', linewidth=2, label='Continuous Actual Probability')

    plt.title('Expected vs. Continuous Actual Probability of Achieving Different Return Levels Over Time')
    plt.xlabel('Trading Days')
    plt.ylabel('Probability (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

def portfolio_composition(etfs, weights, last_close_prices, total_funds):
    """Calculate and print the portfolio composition based on optimized weights."""
    # Allocate funds based on optimized weights
    allocation_funds = total_funds * weights
    
    # Calculate whole units for each ETF
    whole_units = np.floor(allocation_funds / last_close_prices)
    
    # Calculate the remaining funds after purchasing whole units
    remaining_funds = total_funds - np.sum(whole_units * last_close_prices)
    
    # Print the results
    print("\nPortfolio Composition:")
    for etf, units in zip(etfs, whole_units):
        print(f"{etf}: Buy {int(units)} units")
    print(f"Remaining Funds: ${remaining_funds:.2f}")
    
    return whole_units, remaining_funds


def main(*args):
    # etfs = ['DGRO', 'QQQ', 'VGT', 'VIG', 'VOO', 'VOOG']
    etfs = ['DGRO', 'QQQ', 'SPY', 'VGT', 'VIG', 'VOO', 'VOOG','VUG'] # Current etrade on 2024-07-05
    # etfs = ['VZ','AAPL', 'TSLA', 'NFLX', 'MSFT', 'DIS', 'SBUX', 'BABA', 'SPOT', 'NVDA', 'AMD', 'META', 'AMZN', 'GOOG', 'GOOGL'] 

    investment_window = '3mo'
    sharpe = False
    forecast_horizon = 90

    data = fetch_data(etfs, investment_window)
    returns = compute_daily_returns(data)
    volatility_forecasts = fit_garch_and_forecast_volatility(returns)

    results = monte_carlo_simulation(returns, volatility_forecasts, sharpe=sharpe)
    optimal_weights = plot_monte_carlo(results, etfs)

    monte_carlo_results = monte_carlo_forecast(returns, optimal_weights, forecast_horizon=forecast_horizon)
    plot_forecast(monte_carlo_results)

    return_levels = [0.01, 0.06, 0.08, 0.10,  0.15]
    plot_probabilities(monte_carlo_results, return_levels)

    cumulative_backtest_returns = backtest(data, optimal_weights, forecast_horizon=forecast_horizon)
    plot_backtest_vs_forecast(monte_carlo_results, cumulative_backtest_returns)

    continuous_actual_probabilities = calculate_continuous_actual_probabilities(monte_carlo_results, cumulative_backtest_returns)
    plot_backtest_returns_probabilities(monte_carlo_results, return_levels, continuous_actual_probabilities)

    print(f"Log of the Continuous Actual Probabilities: \n{np.log(continuous_actual_probabilities.prod())}")

    # Get the last close prices
    last_close_prices = data.iloc[-1]
    
    # Define total funds available
    total_funds = 3937.30 #  51131.77 # 5445.81  # or 13660.12 or any other value
    
    # Calculate portfolio composition
    whole_units, remaining_funds = portfolio_composition(etfs, optimal_weights, last_close_prices, total_funds)
    portfolio = {}
    fractional = True
    if fractional:
        fractional_price = optimal_weights*total_funds

    for etf, units in zip(etfs, whole_units):
        portfolio[etf]={"units": units}        

    
    if fractional:
        print("For fractional purchase")
        for etf, price in zip(etfs, fractional_price):
            portfolio[etf]["dollars"] = price
            print(f"{etf}: Buy ${int(price)} ")

    return portfolio



if __name__ == "__main__":
    portfolios = []

    for count in range(5):
        portfolios.append(main())

    for item in portfolios:
        for k,v in item.items():
            item[k] = v['dollars']

    df = pd.DataFrame(portfolios)

    

    
    # # Use multiprocessing to parallelize the simulations
    # with Pool(2) as pool:
    #     monte_carlo_results = pool.map(main, [None, None])

    # portfolio = np.array(monte_carlo_results)


# %%
