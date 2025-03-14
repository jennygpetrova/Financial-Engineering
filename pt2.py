import numpy as np
import pandas as pd

p = 400
n = 26

# Get weekly closing price data
prices = pd.read_excel("snp500.xlsx", index_col=0)
#print("\nPortfolio Closing Prices:\n",prices)

# Compute weekly returns
# Drop first row (t = 0)
returns = prices.pct_change(fill_method=None).dropna()
#print("\nPortfolio Weekly Returns:\n",returns)

# Compute excess returns
excess_returns = returns - 0.00087
#print("\nPortfolio Excess Returns\n",excess_returns)

# Transpose excess returns matrix
excess_returns = excess_returns.T # (p x n) matrix of p=400 stocks and n=26 weeks

# Compute de-meaned excess returns matrix Y
expected_returns = excess_returns.mean(axis=1)
expected_returns_table = expected_returns.to_latex(index=True)
#print(expected_returns_table)
#print("\nExpected Returns:\n",expected_returns)


Y = excess_returns.sub(expected_returns, axis=0)
#print("\nY:\n", Y)

# Compute sample covariance matrix S
S = Y @ Y.T / n
#print("\nS:\n", S)
#print("S shape:", S.shape)

# Find trace, leading eigenvalue, and corresponding eigenvector of S
trace_S = np.trace(S)
#print("\ntr(S):\n", trace_S)
eigvals, eigvecs = np.linalg.eigh(S)
lambdaS = eigvals[-1]
#print("\nleading eigenvalue:\n", lambdaS)
v = eigvecs[:, -1]
#print("\ncorresponding eigenvector:\n", v)
l = (trace_S - lambdaS) / (n-1)
#print("\nl:\n", l)


# Compute single factor model covariance matrix Σ
term1 = lambdaS - l
term2 = (n / p) * l
sigma = (term1 * np.outer(v, v)) + (term2 * np.eye(p))
# print("\nSigma:\n", sigma)
# print(sigma.shape)

# Verify leading eigenvector is the same as S (note: floating point error is possible)
eigvals1, eigvecs1 = np.linalg.eigh(sigma)
lambda1 = eigvals1[-1]
#print("\nlambda1:\n", lambda1)

# Compute holdings vector h_C for minimum variance, fully invested portfolio C
ones = np.ones(p)
sigma_inv = np.linalg.inv(sigma)
h_C = sigma_inv @ ones / (ones.T @ sigma_inv @ ones)
#print("\nh_C:\n", h_C)


# Compute portfolio expected excess return, variance, and standard deviation
portfolio_expected_returns = h_C.T @ expected_returns
print("\nPortfolio Expected Returns\n", portfolio_expected_returns)
portfolio_var = h_C.T @ sigma @ h_C
print("\nPortfolio Variance:\n", portfolio_var)
portfolio_std_dev = np.sqrt(portfolio_var)
print("\nPortfolio Standard Deviation:\n", portfolio_std_dev)
stock_var = np.diag(sigma)
print("\nStock Variance:\n", stock_var)
stock_std_dev = np.sqrt(stock_var)
print("\nStock Standard Deviation:\n", stock_std_dev)


# Scale by 52 for annualized results
annualized_return = portfolio_expected_returns * 52
print("\nAnnualized Return:\n", annualized_return)
annualized_var = portfolio_var * 52
print("\nAnnualized Variance:\n", annualized_var)
annualized_std_dev = portfolio_std_dev * np.sqrt(52)
print("\nAnnualized Standard Deviation:\n", annualized_std_dev)
annualized_stock_var = stock_var * 52
print("\nAnnualized Stock Variance:\n", annualized_stock_var)
annualized_stock_std_dev = stock_std_dev * np.sqrt(52)
print("\nAnnualized Stock Standard Deviation:\n", annualized_stock_std_dev)

