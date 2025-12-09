"""
================================================================================
Script: data_pipeline.py
Created by: Anthony M. Schauer

--------------------------------------------------------------------------------
Overview:
This script handles the full pipeline from raw S&P 500 price data to normalized 
daily returns. It prepares data for similarity computation (similarity.py) and 
clustering analysis (clustering.py). Each function is modular and can be tested 
independently or used in the notebook (main_pipeline.ipynb).

--------------------------------------------------------------------------------
Pipeline Steps:
0. Imports 
   - yfinance as yf
   - pandas as pd
   - gc
   - sp500_tickers from tickers_list.py

1. Download stock data
   - Pull adjusted close prices for all S&P 500 tickers using yfinance
   - Download in batches to avoid memory issues

2. Compute daily returns
   - Convert adjusted close prices to percent daily changes
   - Handle missing data carefully (NaNs)

3. Normalize returns
   - Apply z-score normalization (default)
   - Optional: Min-Max normalization for comparison

4. Manage memory
   - Keep RAM usage minimal by deleting temporary dataframes
   - Return only what is needed for downstream analysis

================================================================================
"""

##### Step Zero: Imports
import yfinance as yf
import pandas as pd
import gc
from tickers_list import sp500_tickers

################################################################################

##### Step One: Download Stock Data
def download_stock_data(ticker_list, start="2015-01-01", end="2025-01-01", batch_size=50):
   """
   Downloads adjusted close prices for a list of stock tickers using yfinance.
   
   Parameters:
   - ticker_list: list of stock tickers
   - start: start date for historical data
   - end: end date for historical data
   - batch_size: number of stocks to download at once
   
   Returns:
   - all_data -> a DataFrame containing Adjusted Close prices for each ticker
   """
   
   print("\nStep One: Starting stock download...")
    
   all_data = pd.DataFrame()
    
   # Loop through tickers in batches
   for i in range(0, len(ticker_list), batch_size):
       batch = ticker_list[i:i + batch_size]
       print(f"Downloading batch {i//batch_size + 1}: {batch}")
       
       try:
           temp = yf.download(batch, start=start, end=end, progress=False, auto_adjust=False)["Adj Close"]
       except Exception as e:
           print(f"Error downloading batch {batch}: {e}")
           continue
        
       # Merge into main dataframe
       all_data = pd.concat([all_data, temp], axis=1)
        
       # Free memory
       del temp
       gc.collect()
    
   print("Download complete.\n")
   return all_data

################################################################################

##### Step Two: Compute Daily Returns
def compute_daily_returns(price_df):
   """
   Converts a DataFrame of adjusted close prices into daily percent returns.
   
   Parameters:
   - price_df: DataFrame with dates as index and tickers as columns (adjusted close prices)
   
   Returns:
   - DataFrame with daily percent changes
   """
   print("Step Two: Computing daily returns...")
   
   # Percent change
   returns = price_df.pct_change()
   
   # Handle missing data by dropping rows where all returns are NaN
   returns = returns.dropna(how="all")
   
   print("Daily returns complete.\n")
   return returns

################################################################################

##### Step Three: Normalize Returns 
def normalize_returns(returns_df, method="zscore"):
   """
   Normalizes the daily returns DataFrame.
   
   Parameters:
   - returns_df: DataFrame with daily percent returns
   - method: 'zscore' (default) or 'minmax'
   
   Returns:
   - Normalized DataFrame
   """
   print(f"Step Three: Normalizing returns using {method} method...")
   
   normalized = returns_df.copy()
   
   if method == "zscore":
       normalized = (returns_df - returns_df.mean()) / returns_df.std()
   elif method == "minmax":
       normalized = (returns_df - returns_df.min()) / (returns_df.max() - returns_df.min())
   else:
       raise ValueError("Invalid normalization method. Choose 'zscore' or 'minmax'.")
   
   print("Normalization complete.\n")
   return normalized

################################################################################

##### Step Four: Memory Management
def clean_memory(*dfs):
   """
   Deletes temporary DataFrames to free up memory and runs garbage collection.
   
   Parameters:
   - dfs: one or more DataFrames to delete
   
   Returns:
   - None
   """
   print("Cleaning up memory...")
   
   for df in dfs:
       del df
   
   gc.collect()
   print("Memory cleanup complete.\n")

################################################################################

##### Setp Execute: Run Full S&P 500 Pipeline
if __name__ == "__main__":
   print("\n-----------------------------------------------------------")
   print("Full S&P 500 Pipeline Starting")
   
   # Step One: Download prices
   prices = download_stock_data(sp500_tickers, "2015-01-01", "2025-01-01")
   print(f"Downloaded prices for {len(prices.columns)} tickers.")
   
   # Step Two: Compute daily returns
   returns = compute_daily_returns(prices)
   print(f"Daily returns computed: {returns.shape[0]} days, {returns.shape[1]} tickers.")
   
   # Step Three: Normalize returns
   norm_returns = normalize_returns(returns, method="zscore")
   print("Returns normalized. Ready for similarity/clustering.")
   
   # Step Four: Clean temporary data
   clean_memory(prices, returns)
   
   print("Full S&P 500 Pipeline Complete")
   print("-----------------------------------------------------------\n")
