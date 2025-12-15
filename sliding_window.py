"""
================================================================================
Script: sliding_window.py
Created by: Anthony M. Schauer
--------------------------------------------------------------------------------
Overview:
This script implements a sliding window analysis over historical S&P 500 data.
For each time window, it recomputes similarity and clustering using only the
most recent N trading days. This simulates a streaming or evolving market
environment while remaining computationally feasible. It is built off the other 
scripts written under this folder, the similarity and clustering scripts. The 
sliding window is fixed at one trading year, with one trade month as the steps. 
--------------------------------------------------------------------------------
Sliding Window Steps:
0. Imports
   - Required libraries 
   - Other needed scripts

1. Define sliding window parameters
   - Configure parameters for rolling window analysis
   - Set window size, step size
   - Specify number of clusters for clustering methods

2. Extract rolling window data
   - Select the most recent N trading days of return data
   - Advance the window forward by a fixed step size
   - Normalize returns with z-score for each sliding window

3. Compute similarity and clustering
   - Compute similarity measures within each window
   - Convert similarity to distance where required
   - Apply K-Means and Hierarchical clustering

4. Store window results
   - Store clustering labels for each window
   - Track window start and end dates for analysis
================================================================================
"""

##### Step Zero: Imports
import numpy as np
from similarity import compute_pearson
from clustering import run_kmeans, run_hierarchical

################################################################################

##### Step One: Sliding Window Analysis
def run_sliding_window(
    returns_df,
    window_size=252,
    step_size=21,
    k=6
):
    """
    Parameters:
    - returns_df : DataFrame
        Daily returns (rows = dates, columns = tickers)
    - window_size : int
        Number of most recent trading days per window (default: 252)
    - step_size : int
        Number of days to slide the window forward (default: 21)
    - k : int
        Number of clusters for K-Means and Hierarchical clustering
    Returns:
    - results : list of dicts
        Each entry contains clustering results for one window
    """

    print("\nStarting Sliding Window Analysis...")
    print(f"Window size: {window_size} days")
    print(f"Step size: {step_size} days\n")

    results = []
    dates = returns_df.index
    window_end_indices = range(window_size, len(dates), step_size)

    for idx in window_end_indices:
        window_end_date = dates[idx]
        window_start_date = dates[idx - window_size]

        print(f"Processing window ending on {window_end_date.date()}")

        ##### Step Two: Extract Window and Normalize Returns 
        window_returns = returns_df.iloc[idx - window_size:idx]
        
        window_norm = (window_returns - window_returns.mean()) / window_returns.std()
        window_norm = window_norm.replace([np.inf, -np.inf], 0).fillna(0)

        ##### Step Three: Compute Similarities and Clustering 
        pearson_mat = compute_pearson(window_norm)
        pearson_mat_clean = pearson_mat.replace([np.inf, -np.inf], 0).fillna(0)
        distance_mat = 1 - pearson_mat_clean

        labels_kmeans, _ = run_kmeans(window_norm.T, k=k)

        labels_hier, _ = run_hierarchical(distance_mat, k=k, metric="precomputed")

        ##### Step Four: Store Results
        results.append({
            "window_start": window_start_date,
            "window_end": window_end_date,
            "kmeans_labels": labels_kmeans,
            "hierarchical_labels": labels_hier
        })

        print("Window complete.\n")

    print("Sliding Window Analysis Complete.\n")
    return results