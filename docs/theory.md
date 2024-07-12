# Theory of Main Concepts

## Mean-Variance Optimization
Mean-variance optimization is a process that aims to create a portfolio with the highest expected return for a given level of risk. It involves calculating the expected returns, variances, and covariances of the portfolio's assets and then finding the optimal weights that maximize the portfolio's Sharpe or Sortino ratio.

## GARCH Volatility Forecasting
The Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model is used to forecast the volatility of asset returns. The GARCH model captures the time-varying nature of volatility, which helps in better risk management and portfolio optimization.

## Monte Carlo Simulation
Monte Carlo simulation is a technique used to understand the impact of risk and uncertainty in financial, project management, and other forecasting models. By simulating a large number of possible outcomes, we can estimate the probability distribution of the portfolio's future returns.

## Backtesting
Backtesting involves testing a trading strategy or portfolio optimization technique using historical data. It helps in assessing the effectiveness of the strategy and understanding how it would have performed in the past.

## Maths

### Mean-Variance Optimization

1. **Objective**:
   Minimize the portfolio variance for a given level of expected return.

2. **Mathematical Formulation**:
   \[
   \text{Minimize} \quad \sigma_p^2 = \mathbf{w}^\top \mathbf{\Sigma} \mathbf{w}
   \]
   subject to:
   \[
   \mathbf{w}^\top \mathbf{\mu} = \mu_p
   \]
   \[
   \mathbf{1}^\top \mathbf{w} = 1
   \]
   \[
   \mathbf{w} \geq 0
   \]
   
   where:
   - \(\mathbf{w}\) is the vector of portfolio weights.
   - \(\mathbf{\Sigma}\) is the covariance matrix of the asset returns.
   - \(\mathbf{\mu}\) is the vector of expected returns for each asset.
   - \(\mu_p\) is the target portfolio return.
   - \(\sigma_p^2\) is the portfolio variance.
   - \(\mathbf{1}\) is a vector of ones.

### Sharpe Ratio Optimization

1. **Objective**:
   Maximize the Sharpe ratio of the portfolio.

2. **Mathematical Formulation**:
   \[
   \text{Maximize} \quad S = \frac{\mathbf{w}^\top (\mathbf{\mu} - r_f)}{\sqrt{\mathbf{w}^\top \mathbf{\Sigma} \mathbf{w}}}
   \]
   subject to:
   \[
   \mathbf{1}^\top \mathbf{w} = 1
   \]
   \[
   \mathbf{w} \geq 0
   \]
   
   where:
   - \(S\) is the Sharpe ratio.
   - \(r_f\) is the risk-free rate.

### Monte Carlo Simulation

1. **Objective**:
   Simulate possible future returns of the portfolio to understand the distribution of outcomes.

2. **Mathematical Formulation**:
   For each simulation \(i\):
   \[
   R_{p,i}(t) = \sum_{j=1}^n w_j R_{j,i}(t)
   \]
   where:
   - \(R_{p,i}(t)\) is the simulated portfolio return at time \(t\) in simulation \(i\).
   - \(w_j\) is the weight of asset \(j\) in the portfolio.
   - \(R_{j,i}(t)\) is the simulated return of asset \(j\) at time \(t\) in simulation \(i\).

3. **Cumulative Returns**:
   \[
   C_{p,i}(T) = \prod_{t=1}^T (1 + R_{p,i}(t)) - 1
   \]
   where:
   - \(C_{p,i}(T)\) is the cumulative return of the portfolio at the end of the forecast horizon \(T\) in simulation \(i\).

### Backtesting

1. **Objective**:
   Evaluate the performance of the optimized portfolio using historical data.

2. **Mathematical Formulation**:
   \[
   R_{p}(t) = \sum_{j=1}^n w_j R_{j}(t)
   \]
   where:
   - \(R_{p}(t)\) is the portfolio return at time \(t\).
   - \(R_{j}(t)\) is the historical return of asset \(j\) at time \(t\).

3. **Cumulative Returns**:
   \[
   C_{p}(T) = \prod_{t=1}^T (1 + R_{p}(t)) - 1
   \]
   where:
   - \(C_{p}(T)\) is the cumulative return of the portfolio at the end of the backtesting period \(T\).

### Portfolio Composition

1. **Objective**:
   Allocate the available funds to the assets based on the optimized weights.

2. **Mathematical Formulation**:
   \[
   \text{Number of Units of Asset } j = \left\lfloor \frac{w_j \cdot \text{Total Funds}}{\text{Price of Asset } j} \right\rfloor
   \]
   \[
   \text{Remaining Funds} = \text{Total Funds} - \sum_{j=1}^n \left( \text{Number of Units of Asset } j \cdot \text{Price of Asset } j \right)
   \]
   
This formal mathematical notation outlines the core methodologies employed in your portfolio optimization tool. If you need further details or have additional questions, feel free to ask!
