# plot_example.py
import matplotlib.pyplot as plt
from data_pipeline import download_stock_data, compute_daily_returns  # use your real function names

# Step 1: define which tickers to use
tickers = ["AAPL", "MSFT", "GOOGL"]  # example, can add more

# Step 2: download data using the pipeline
data = download_stock_data(tickers)  # gets historical prices

# Step 3: Compute daily returns
returns = compute_daily_returns(data)  # returns DataFrame with daily returns

# Step 4: pick one or more stocks to plot
plt.figure(figsize=(12, 6))
for stock in tickers:
    plt.plot(returns.index, returns[stock], label=stock)

plt.title('Daily Returns of Selected S&P 500 Stocks (10 Years)')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
plt.show()