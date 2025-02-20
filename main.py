import numpy as np
import pandas as pd

# Weekly closing prices from Excel file
prices = pd.read_excel("portfolioP.xlsx", index_col=0)

prices_table = prices.to_latex(float_format="%.6f")
print(prices_table)


# Compute weekly returns
returns = prices.pct_change().dropna()
# Compute excess returns
excess_returns = returns - 0.00087
print(excess_returns.head())

# Transpose excess returns matrix
excess_returns = excess_returns.T # (p x n) matrix of p=9 stocks and n=26 weeks
print(excess_returns.head())

# Compute de-meaned excess returns matrix Y
expected_returns = excess_returns.mean(axis=1)
Y = excess_returns.sub(expected_returns, axis=0)
print(Y.head())

# Compute sample covariance matrix V
V = Y @ Y.T / 26
print(V)

# Compute minimum variance, fully invested portfolio C
ones = np.ones(9)
V_inv = np.linalg.inv(V)

# Compute holdings vector h_C
h_C = V_inv @ ones / (ones.T @ V_inv @ ones)

# Compute portfolio expected excess return, variance, and standard deviation
expected_excess_return = h_C.T @ expected_returns
portfolio_variance = h_C.T @ V @ h_C
portfolio_std_dev = np.sqrt(portfolio_variance)

# Step 4: Annualize Results (Assuming 52 weeks in a year)
annualized_return = expected_excess_return * 52
annualized_variance = portfolio_variance * 52
annualized_std_dev = portfolio_std_dev * np.sqrt(52)

# Compute individual stock variances for comparison
individual_variances = np.diag(V)
annualized_individual_variances = individual_variances * 52  # Convert to annual

# Print results
print("Minimum Variance Portfolio Holdings (h_C):")
print(h_C)

print("\nWeekly Expected Excess Return:", expected_excess_return)
print("Weekly Portfolio Variance:", portfolio_variance)
print("Weekly Portfolio Standard Deviation:", portfolio_std_dev)

print("\nAnnualized Expected Excess Return:", annualized_return)
print("Annualized Portfolio Variance:", annualized_variance)
print("Annualized Portfolio Standard Deviation:", annualized_std_dev)

print("\nWeekly Individual Stock Variances:")
print(individual_variances)

print("\nAnnualized Individual Stock Variances:")
print(annualized_individual_variances)

