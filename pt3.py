import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D


p = 400
n = 104
ones = np.ones(p) # to be used in several computations

prices = pd.read_excel("snp500.xlsx", index_col=0) # weekly closing price data
returns = prices.pct_change(fill_method=None).dropna() # weekly returns
excess_returns = returns - 0.000993 # excess returns
excess_returns = excess_returns.T # (p x n) matrix
expected_returns = excess_returns.mean(axis=1) # de-meaned excess returns matrix Y
expected_returns_table = expected_returns.to_latex(index=True)
#print(expected_returns_table)
Y = excess_returns.sub(expected_returns, axis=0)
S = Y @ Y.T / n # sample covariance matrix S

trace_S = np.trace(S)
eigenvals, eigenvecs = np.linalg.eigh(S)
lambdaS = eigenvals[-1] # leading eigenvalue
h = eigenvecs[:, -1] # estimator (leading eigenvector)
l = (trace_S - lambdaS) / (n-1)
term1 = lambdaS - l
term2 = (n / p) * l
sigma = (term1 * np.outer(h, h)) + (term2 * np.eye(p)) # single factor model covariance matrix sigma_PCA

eigenval1, eigenvec1 = np.linalg.eigh(sigma) # verify identical leading eigenvector
lambda1 = eigenval1[-1]


"---------------------------JSE FOR EIGENVECTORS---------------------------"
v = l / p
lambda_sqrt = np.sqrt(lambdaS)
m = np.average(h)
m_h = lambda_sqrt * m
sum = 0
for i in range(len(h)):
    sum += ((lambda_sqrt * h[i]) - m_h) ** 2
s = sum / p
c_JSE = 1 - (v / s)
h_JSE = (m * ones) + (c_JSE * (h - (m * ones)))
h_JSE = h_JSE / np.linalg.norm(h_JSE)
sigma_JSE = (term1 * np.outer(h_JSE, h_JSE)) + (term2 * np.eye(p)) # single factor model covariance matrix sigma_JSE


"---------------------------PORTFOLIO PERFORMANCE METRICS---------------------------"
# Metrics for sigma_PCA
sigma_inv = np.linalg.inv(sigma)
holdings = sigma_inv @ ones / (ones.T @ sigma_inv @ ones) # holdings for min var, fully invested portfolio C
#print(np.sum(np.array(holdings) > 0, axis=0)) # number of positive holdings
maxh = np.max(holdings)
max_id = np.argmax(holdings)
#print(maxh, max_id) # max holdings and index
minh = np.min(holdings)
min_id = np.argmin(holdings) # min holdings and index
#print(minh, min_id)

portfolio_expected_returns = holdings.T @ expected_returns
print("portfolio expected excess returns:", portfolio_expected_returns)
portfolio_var = holdings.T @ sigma @ holdings
#print("portfolio variance:", portfolio_var)
portfolio_std_dev = np.sqrt(portfolio_var)
#print("portfolio standard deviation:", portfolio_std_dev)
stock_var = np.diag(sigma)
stock_std_dev = np.sqrt(stock_var)

# annualized results (scaled by 52)
annualized_return = portfolio_expected_returns * 52
#print("annualized returns", annualized_return)
annualized_var = portfolio_var * 52
#print("annualized variance", annualized_var)
annualized_std_dev = portfolio_std_dev * np.sqrt(52)
#print("annualized standard deviation", annualized_std_dev)
annualized_stock_var = stock_var * 52
annualized_stock_std_dev = stock_std_dev * np.sqrt(52)


# Metrics for sigma_JSE
sigma_inv_JSE = np.linalg.inv(sigma_JSE)
holdings_JSE = sigma_inv_JSE @ ones / (ones.T @ sigma_inv_JSE @ ones)
#print(np.sum(np.array(holdings_JSE) > 0, axis=0))
maxh_JSE = np.max(holdings_JSE)
max_id_JSE = np.argmax(holdings_JSE)
#print(maxh_JSE, max_id_JSE) # max holdings and index
minh_JSE = np.min(holdings_JSE)
min_id_JSE = np.argmin(holdings_JSE) # min holdings and index
#print(minh_JSE, min_id_JSE)

portfolio_expected_returns_JSE = holdings_JSE.T @ expected_returns
#print("portfolio expected excess returns (JSE)", portfolio_expected_returns_JSE)
portfolio_var_JSE = holdings_JSE.T @ sigma_JSE @ holdings_JSE
#print("portfolio variance (JSE)", portfolio_var_JSE)
portfolio_std_dev_JSE = np.sqrt(portfolio_var_JSE)
#print("portfolio standard deviation (JSE)", portfolio_std_dev_JSE)
stock_var_JSE = np.diag(sigma_JSE)
stock_std_dev_JSE = np.sqrt(stock_var_JSE)

# (JSE) annualized results (scaled by 52)
annualized_return_JSE = portfolio_expected_returns_JSE * 52
#print("annualized returns (JSE)", annualized_return_JSE)
annualized_var_JSE = portfolio_var_JSE * 52
#print("annualized variance (JSE)", annualized_var_JSE)
annualized_std_dev_JSE = portfolio_std_dev_JSE * np.sqrt(52)
#print("annualized standard deviation (JSE)", annualized_std_dev_JSE)
annualized_stock_var_JSE = stock_var_JSE * 52
annualized_stock_std_dev_JSE = stock_std_dev_JSE * np.sqrt(52)


# Sorting variance and standard deviation
# No JSE
sorted_indices1 = np.argsort(stock_var)
bottom10_indices1 = sorted_indices1[:10]
top10_indices1 = sorted_indices1[-10:][::-1]

# JSE
sorted_indices = np.argsort(stock_var_JSE)
bottom10_indices = sorted_indices[:10]
top10_indices = sorted_indices[-10:][::-1]
# print("\nTop 10 values and their indices:")
# for idx in top10_indices1:
#     print(f"{idx} & {round(stock_var[idx], 6)} & {round(stock_std_dev[idx], 6)}")
# print("JSE:")
# for idx in top10_indices:
#     print(f"{idx} & {round(stock_var_JSE[idx],6)} & {round(stock_std_dev_JSE[idx],6)}")
#
# print("\nBottom 10 values and their indices:")
# for idx in bottom10_indices1:
#     print(f"{idx} & {round(stock_var[idx], 6)} & {round(stock_std_dev[idx], 6)}")
# print("JSE:")
# for idx in bottom10_indices:
#     print(f"{idx} & {round(stock_var_JSE[idx],6)} & {round(stock_std_dev_JSE[idx],6)}")


"---------------------------PLOTS!!---------------------------"
# PLOT 1 -
row = prices.iloc[0]
new_df = pd.DataFrame([row])
for i in range(p):
    new_df.iloc[0, i] = i+1

x1 = stock_std_dev
y1 = holdings
x2 = stock_std_dev_JSE
y2 = holdings_JSE
z1 = new_df.iloc[0, :]
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(x1, y1, label='PCA', c=z1, cmap='Blues')
ax.scatter(x2, y2, label='JSE', c=z1, cmap='Reds')
norm = mcolors.Normalize(vmin=z1.min(), vmax=z1.max())
sm = cm.ScalarMappable(norm=norm, cmap='Blues')
sm2 = cm.ScalarMappable(norm=norm, cmap='Reds')
sm.set_array([])
sm2.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar2 = plt.colorbar(sm2, ax=ax)
cbar.set_label("Market Cap Ranking (1 = Highest, 400 = Lowest)".format(z1.min()))
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='PCA',
           markerfacecolor='darkblue', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='JSE',
           markerfacecolor='darkred', markersize=8)
]
ax.legend(handles=legend_elements)
plt.xlabel('Risk ($\sigma$)')
plt.ylabel('Holdings ($h_C$)')
plt.title('Holdings vs. Risk by Market Cap')
plt.savefig('holdings_risk.png')
plt.show()

weight_PCA = holdings * expected_returns * 52
weight_JSE = holdings_JSE * expected_returns * 52
stock_indices = np.arange(p)
plt.figure(figsize=(12, 6))
colors_PCA = ['green' if x >= 0 else 'orange' for x in weight_PCA]
plt.bar(stock_indices, weight_PCA, color=colors_PCA)
plt.xlabel('Stock (Ordered by Market Cap)')
plt.ylabel('Annualized Return Percentage (PCA)')
plt.title('Histogram of Annualized Return (PCA) Per Stock')
plt.axhline(0, color='black', linewidth=1)
plt.tight_layout()
plt.savefig('hist_pca.png')
plt.show()

plt.figure(figsize=(12, 6))
colors_JSE = ['green' if x >= 0 else 'orange' for x in weight_JSE]
plt.bar(stock_indices, weight_JSE, color=colors_JSE)
plt.xlabel('Stock (Ordered by Market Cap)')
plt.ylabel('Annualized Return Percentage (JSE)')
plt.title('Histogram of Annualized Return (JSE) Per Stock')
plt.axhline(0, color='black', linewidth=1)
plt.tight_layout()
plt.savefig('hist_jse.png')
plt.show()


annualized_stock_returns = expected_returns * 52
plt.figure(figsize=(10, 7))
plt.scatter(
    annualized_stock_std_dev,
    annualized_stock_returns,
    color='magenta', alpha=0.6, edgecolors='none',
    label='Individual Stocks (PCA)'
)
plt.scatter(
    annualized_stock_std_dev_JSE,
    annualized_stock_returns,
    color='teal', alpha=0.6, edgecolors='none',
    label='Individual Stocks (JSE)'
)
plt.scatter(
    annualized_std_dev,
    annualized_return,
    color='red', marker='*', s=250, edgecolors='black',
    label='PCA Min-Var Portfolio'
)
plt.scatter(
    annualized_std_dev_JSE,
    annualized_return_JSE,
    color='blue', marker='*', s=250, edgecolors='black',
    label='JSE Min-Var Portfolio'
)
text_PCA = (
    f"PCA:\n"
    f"Return = {annualized_return:.4f}\n" 
    f"Variance = {annualized_var:.4f}"
)
plt.annotate(
    text_PCA,
    (annualized_std_dev, annualized_return),
    textcoords="offset points",
    xytext=(10, 40),
    ha='left',
    color='red',
    fontsize=10,
    arrowprops=dict(arrowstyle="-", color='red')
)
text_JSE = (
    f"JSE:\n"
    f"Return = {annualized_return_JSE:.4f}\n"
    f"Variance = {annualized_var_JSE:.4f}"
)
plt.annotate(
    text_JSE,
    (annualized_std_dev_JSE, annualized_return_JSE),
    textcoords="offset points",
    xytext=(80, -40),
    ha='left',
    color='blue',
    fontsize=10,
    arrowprops=dict(arrowstyle="-", color='blue')
)
plt.title('Risk–Return Profile with Excess Returns')
plt.xlabel('Annualized Standard Deviation (Risk)')
plt.ylabel('Annualized Excess Return')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('scatter.png')
plt.show()
