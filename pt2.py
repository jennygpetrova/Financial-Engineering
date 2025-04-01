import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
#print("\nLeading eigenvalue:\n", lambdaS)
v = eigvecs[:, -1]
#print("\nCorresponding eigenvector:\n", v)
norm = v.T @ v
print(norm)
l = (trace_S - lambdaS) / (n-1)
#print("\nl:\n", l)


# Compute single factor model covariance matrix Σ
term1 = lambdaS - l
term2 = (n / p) * l
sigma = (term1 * np.outer(v, v)) + (term2 * np.eye(p))
#print("\nSigma:\n", sigma)
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
#print("\nPortfolio Expected Returns\n", portfolio_expected_returns)
portfolio_var = h_C.T @ sigma @ h_C
#print("\nPortfolio Variance:\n", portfolio_var)
portfolio_std_dev = np.sqrt(portfolio_var)
#print("\nPortfolio Standard Deviation:\n", portfolio_std_dev)
stock_var = np.diag(sigma)
#print("\nStock Variance:\n", stock_var)
stock_std_dev = np.sqrt(stock_var)
#print("\nStock Standard Deviation:\n", stock_std_dev)


# Scale by 52 for annualized results
annualized_return = portfolio_expected_returns * 52
#print("\nAnnualized Return:\n", annualized_return)
annualized_var = portfolio_var * 52
#print("\nAnnualized Variance:\n", annualized_var)
annualized_std_dev = portfolio_std_dev * np.sqrt(52)
#print("\nAnnualized Standard Deviation:\n", annualized_std_dev)
annualized_stock_var = stock_var * 52
#print("\nAnnualized Stock Variance:\n", annualized_stock_var)
annualized_stock_std_dev = stock_std_dev * np.sqrt(52)
#print("\nAnnualized Stock Standard Deviation:\n", annualized_stock_std_dev)


# PLOT 1
row = prices.iloc[0]
new_df = pd.DataFrame([row])
print(new_df)
for i in range(p):
    new_df.iloc[0, i] = i+1
print(new_df)

x = stock_std_dev
y = expected_returns
z = new_df.iloc[0, :]
fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(x, y, c=z, cmap='viridis')
point = ax.plot(portfolio_std_dev, portfolio_expected_returns, 'ro')
ax.text(portfolio_std_dev, portfolio_expected_returns,
        f'Portfolio Return:\n {portfolio_expected_returns:.4f}\nRisk: {portfolio_std_dev:.4f}',
        fontsize=12, color='red', ha='left', va='bottom')
legend = ax.legend(*scatter.legend_elements(), loc="lower right", title="Company Market Cap Range")
ax.add_artist(legend)
ax.set_xlabel('Risk ($\sigma$)')
ax.set_ylabel('Expected Return ($f$)')
ax.set_title('Expected Excess Returns vs. Risk by Market Cap', size=14)
plt.savefig('returns_vs_risk.png')
plt.show()

# PLOT 2
x = stock_std_dev
y = h_C
fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(x, y, c=y, cmap='viridis')
ax.text(portfolio_std_dev, portfolio_expected_returns,
        f'Portfolio Return:\n {portfolio_expected_returns:.4f}\nRisk: {portfolio_std_dev:.4f}',
        fontsize=14, color='red', ha='left', va='bottom')
legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Portolio Holdings (%)")
ax.add_artist(legend)
ax.set_ylabel('Holdings ($h_C$)')
ax.set_xlabel('Risk ($\sigma$)')
ax.set_title('Portfolio Holdings by Level of Risk', size=14)
plt.savefig('holdings_vs_risk.png')
plt.show()

