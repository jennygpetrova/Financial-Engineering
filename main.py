import numpy as np
import pandas as pd

# Get weekly closing price data
prices = pd.read_excel("portfolioP.xlsx", index_col=0)
print("\nPortfolio Closing Prices:\n",prices)

# Compute weekly returns
# Drop first row (t = 0)
returns = prices.pct_change().dropna()

# Compute excess returns
excess_returns = returns - 0.00087
print("\nPortfolio Excess Returns\n",excess_returns)

# Transpose excess returns matrix
excess_returns = excess_returns.T # (p x n) matrix of p=9 stocks and n=26 weeks

# Compute de-meaned excess returns matrix Y
expected_returns = excess_returns.mean(axis=1)
Y = excess_returns.sub(expected_returns, axis=0)
print("\nY:\n", Y)

# Compute sample covariance matrix V
V = Y @ Y.T / 26
print("\nV:\n", V)

# Compute holdings vector h_C for minimum variance, fully invested portfolio C
ones = np.ones(9)
V_inv = np.linalg.inv(V)
h_C = V_inv @ ones / (ones.T @ V_inv @ ones)
print("\nh_C:\n", h_C)

# Compute portfolio expected excess return, variance, and standard deviation
portfolio_expected_returns = h_C.T @ expected_returns
print("\nPortfolio Expected Returns\n", portfolio_expected_returns)
portfolio_var = h_C.T @ V @ h_C
print("\nPortfolio Variance:\n", portfolio_var)
portfolio_std_dev = np.sqrt(portfolio_var)
print("\nPortfolio Standard Deviation:\n", portfolio_std_dev)
stock_var = np.diag(V)
print("\nStock Variance:\n", stock_var)

# Scale by 52 for annualized results
annualized_return = portfolio_expected_returns * 52
print("\nAnnualized Return:\n", annualized_return)
annualized_var = portfolio_var * 52
print("\nAnnualized Variance:\n", annualized_var)
annualized_std_dev = portfolio_std_dev * np.sqrt(52)
print("\nAnnualized Standard Deviation:\n", annualized_std_dev)
annualized_stock_var = stock_var * 52
print("\nAnnualized Stock Variance:\n", annualized_stock_var)




