"""
================================================================================
Script: data_pipeline.py
Created by: Anthony M. Schauer with Malhar Gujar 
--------------------------------------------------------------------------------
Overview:
This script handles the full pipeline from raw S&P 500 price data to normalized 
daily returns. It prepares data for similarity computation (similarity.py) and 
clustering analysis (clustering.py). Each function is modular and can be tested 
independently or used in the notebook (main_pipeline.ipynb).
--------------------------------------------------------------------------------
Pipeline Steps:
0. Imports 
   - Required libraries 
   - Other needed scripts 

1. Download stock data
   - Downloads adjusted close prices for a list of stock tickers using yfinance.
   - Download in batches to avoid memory issues

2. Compute daily returns
   - Converts a DataFrame of adjusted close prices into daily percent returns.
   - Handle missing data carefully (NaNs)

3. Normalize returns
   - Normalizes the daily returns DataFrame
   - Apply z-score normalization

4. Manage memory
   - Deletes temporary DataFrames to free up memory and runs garbage collection.
   - Keep RAM usage minimal by deleting temporary dataframes
   - Return only what is needed for downstream analysis

Execute
   - Runs the pipeline
   - Calls analysis funcitons 
   - Produces outputs
   - Finalizes the process
================================================================================
"""

##### Step Zero: Imports
import yfinance as yf
import pandas as pd
import numpy as np
import gc
import time 
from tickers_list import sp500_tickers
from similarity import compute_pearson, summarize_pearson
from clustering import run_kmeans, run_hierarchical, summarize_clusters, summarize_hierarchical
from sliding_window import run_sliding_window, summarize_sliding_windows
from visuals import plot_pca_clusters, plot_correlation_heatmap, plot_dendrogram_from_distance, plot_cluster_change_over_time


##### Step One: Download Stock Data
def download_stock_data(ticker_list, start="2015-01-01", end="2025-01-01", batch_size=50):
   """
   Parameters:
   - ticker_list: list of stock tickers
   - start: start date for historical data
   - end: end date for historical data
   - batch_size: number of stocks to download at once
   Returns:
   - all_data -> a DataFrame containing Adjusted Close prices for each ticker
   """
   
   print("\nPart One: Starting stock download...")
    
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


##### Step Two: Compute Daily Returns
def compute_daily_returns(price_df):
   """
   Parameters:
   - price_df: DataFrame with dates as index and tickers as columns (adjusted close prices)
   Returns:
   - DataFrame with daily percent changes
   """
   print("Part Two: Computing daily returns...")
   
   # Percent change
   returns = price_df.pct_change(fill_method=None)
   
   # Handle missing data by dropping rows where all returns are NaN
   returns = returns.dropna(how="all")
   
   print("Daily returns complete.")
   return returns


##### Step Three: Normalize Returns 
def normalize_returns(returns_df, method="zscore"):
   """
   Parameters:
   - returns_df: DataFrame with daily percent returns
   - method: 'zscore'
   Returns:
   - Normalized DataFrame
   """
   print(f"Part Three: Normalizing returns using {method} method...")
   
   normalized = returns_df.copy()
   normalized = (returns_df - returns_df.mean()) / returns_df.std()
   
   print("Normalization complete.\n")
   return normalized


##### Step Four: Memory Management
def clean_memory(*dfs):
   """ 
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
   
   ### Part One: Download prices
   prices = download_stock_data(sp500_tickers, "2015-01-01", "2025-01-01")
   print(f"Downloaded prices for {len(prices.columns)} tickers.\n")

   ### Part Two: Compute daily returns
   returns = compute_daily_returns(prices)
   print(f"Daily returns computed: {returns.shape[0]} days, {returns.shape[1]} tickers.\n")
   
   ### Part Three: Normalize returns
   norm_returns = normalize_returns(returns, method="zscore")

   ### Part Four: Compute Similarity
   print("Part Four: Computing Pearson correlation matrix...")
   pearson_mat = compute_pearson(norm_returns)
   pearson_mat_clean = pearson_mat.replace([np.inf, -np.inf], 0).fillna(0)
   print("Pearson similarity matrix computed and cleaned. Shape:", pearson_mat_clean.shape)
   if not np.isfinite(pearson_mat_clean.values).all():
      print("Warning: Pearson matrix still contains non-finite values!")
   norm_returns_clean = norm_returns.replace([np.inf, -np.inf], 0).fillna(0)
   labels_kmeans, _ = run_kmeans(norm_returns_clean.T, k=6)
   summarize_pearson(pearson_mat_clean, labels=labels_kmeans)

   ### Part Five: Run Clustering
   print("\nPart Five: Clustering")
   print(f"Running K-Means clustering with k=6...")
   print("K-Means complete.")
   summarize_clusters(norm_returns_clean, labels_kmeans, method_name="K-Means", returns=returns)
   print(f"Running Hierarchical clustering with k=6...")
   distance = 1 - pearson_mat_clean
   labels_hier, _ = run_hierarchical(distance, k=6, metric="precomputed")
   summarize_hierarchical(distance, labels_hier, method_name="Hierarchical")
   print("Hierarchical clustering complete.")

   ### Part Six: Run Sliding Window
   print("\nPart Six: Sliding Window")
   sliding_results, window_size, step_size = run_sliding_window(
      returns,
      window_size=252,
      step_size=21,
      k=6 )
   summarize_sliding_windows(sliding_results, returns, window_size, step_size)

   ### Part Seven: Visuals
   print("\nPart Seven: Visuals")
   # PCA visualization (K-Means clusters)
   plot_pca_clusters(
      norm_returns_clean,
      labels_kmeans,
      method_name="K-Means"
   )
   # Correlation heatmap (ordered by K-Means clusters)
   plot_correlation_heatmap(
      pearson_mat_clean,
      labels_kmeans
   )
   # Hierarchical dendrogram
   plot_dendrogram_from_distance(distance)
   # Sliding Window Visual
   plot_cluster_change_over_time(sliding_results, returns)

   ### Part Eight: Clean temporary data
   clean_memory(prices, returns)
   
   print("Full S&P 500 Analysis Complete")
   print("-----------------------------------------------------------\n")
