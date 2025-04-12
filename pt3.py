import numpy as np
import pandas as pd

p = 400
n = 26

ones = np.ones(p) # to be used in several computations

# Weekly closing price data
prices = pd.read_excel("snp500.xlsx", index_col=0)

# Weekly returns
# Drop first row (t = 0)
returns = prices.pct_change(fill_method=None).dropna()

# Excess returns
excess_returns = returns - 0.00087
excess_returns = excess_returns.T # (p x n) matrix

# De-meaned excess returns matrix Y
expected_returns = excess_returns.mean(axis=1)
expected_returns_table = expected_returns.to_latex(index=True)
Y = excess_returns.sub(expected_returns, axis=0)

# Sample covariance matrix S
S = Y @ Y.T / n

# Trace, leading eigenvalue, and corresponding eigenvector of S
trace_S = np.trace(S)
eigenvals, eigenvecs = np.linalg.eigh(S)
lambdaS = eigenvals[-1]
h = eigenvecs[:, -1] # estimator
norm = h.T @ h
l = (trace_S - lambdaS) / (n-1)

# Single factor model covariance matrix Σ
term1 = lambdaS - l
term2 = (n / p) * l
sigma = (term1 * np.outer(h, h)) + (term2 * np.eye(p))

# Verify leading eigenvector is the same as S (note: floating point error is possible)
eigenval1, eigenvec1 = np.linalg.eigh(sigma)
lambda1 = eigenval1[-1]

# Holdings vector 'holdings' for minimum variance, fully invested portfolio C
sigma_inv = np.linalg.inv(sigma)
holdings = sigma_inv @ ones / (ones.T @ sigma_inv @ ones)

# Portfolio expected excess return, variance, and standard deviation
portfolio_expected_returns = holdings.T @ expected_returns
portfolio_var = holdings.T @ sigma @ holdings
portfolio_std_dev = np.sqrt(portfolio_var)
stock_var = np.diag(sigma)
stock_std_dev = np.sqrt(stock_var)

# Annualized results (scaled by 52)
annualized_return = portfolio_expected_returns * 52
print("annualized returns", annualized_return)
annualized_var = portfolio_var * 52
print("annualized variance", annualized_var)
annualized_std_dev = portfolio_std_dev * np.sqrt(52)
print("annualized standard deviation", annualized_std_dev)
annualized_stock_var = stock_var * 52
annualized_stock_std_dev = stock_std_dev * np.sqrt(52)


"--------------------------- JSE FOR EIGENVECTORS --------------------------- "
v = l / p
s = np.var(eigenvals)
c_JSE = 1 - (v / s)
m = np.mean(h)
h_JSE = (m * ones) + (c_JSE * (h - (m * ones)))
h_JSE = h_JSE / np.linalg.norm(h_JSE)

sigma_JSE = (term1 * np.outer(h_JSE, h_JSE)) + (term2 * np.eye(p))

# Holdings vector 'holdings_JSE' for minimum variance, fully invested portfolio C using JSE estimator
sigma_inv_JSE = np.linalg.inv(sigma_JSE)
holdings_JSE = sigma_inv_JSE @ ones / (ones.T @ sigma_inv_JSE @ ones)

# Portfolio expected excess return, variance, and standard deviation
portfolio_expected_returns_JSE = holdings_JSE.T @ expected_returns
portfolio_var_JSE = holdings_JSE.T @ sigma_JSE @ holdings_JSE
portfolio_std_dev_JSE = np.sqrt(portfolio_var_JSE)
stock_var_JSE = np.diag(sigma_JSE)
stock_std_dev_JSE = np.sqrt(stock_var_JSE)


# Annualized results (scaled by 52)
annualized_return_JSE = portfolio_expected_returns_JSE * 52
print("annualized returns (JSE)", annualized_return_JSE)
annualized_var_JSE = portfolio_var_JSE * 52
print("annualized variance (JSE)", annualized_var_JSE)
annualized_std_dev_JSE = portfolio_std_dev_JSE * np.sqrt(52)
print("annualized standard deviation (JSE)", annualized_std_dev_JSE)
annualized_stock_var_JSE = stock_var_JSE * 52
annualized_stock_std_dev_JSE = stock_std_dev_JSE * np.sqrt(52)

