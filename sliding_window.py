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

import time
import sys

##### Step One: Sliding Window Analysis
def run_sliding_window(returns_df, window_size=252, step_size=21, k=6 ):
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

    print("Starting Sliding Window Analysis...")
    print(f"Window size: {window_size} days")
    print(f"Step size: {step_size} days\n")

    results = []
    dates = returns_df.index
    window_end_indices = range(window_size, len(dates), step_size)

    kmeans_history = []
    hier_history = []

    for idx in window_end_indices:
        window_end_date = dates[idx]
        window_start_date = dates[idx - window_size]

        window_returns = returns_df.iloc[idx - window_size:idx]
        
        window_norm = (window_returns - window_returns.mean()) / window_returns.std()
        window_norm = window_norm.replace([np.inf, -np.inf], 0).fillna(0)

        pearson_mat = compute_pearson(window_norm)
        pearson_mat_clean = pearson_mat.replace([np.inf, -np.inf], 0).fillna(0)
        distance_mat = 1 - pearson_mat_clean

        labels_kmeans, _ = run_kmeans(window_norm.T, k=k)
        kmeans_history.append(labels_kmeans)

        labels_hier, _ = run_hierarchical(distance_mat, k=k, metric="precomputed")
        hier_history.append(labels_hier)

        results.append({
            "window_start": window_start_date,
            "window_end": window_end_date,
            "kmeans_labels": labels_kmeans,
            "hierarchical_labels": labels_hier
        })

    return results, window_size, step_size


def slow_print(text):
    print(text)
    sys.stdout.flush()
    time.sleep(0.10)

##### Step Two: Summarize Sliding Window
def summarize_sliding_windows(window_results, returns_df, window_size, step_size):
    """
    Parameters:
        window_results : list of dicts
            Output of run_sliding_window()
        returns_df : DataFrame
            Original returns data (rows = dates, cols = tickers)
    """
    print("\n--- SLIDING WINDOW SUMMARY ---")
    total_windows = len(window_results)
    print(f"Window size: {window_size} trading days")
    print(f"Step size: {step_size} trading days")
    num_assets = returns_df.shape[1]
    k = len(np.unique(window_results[0]["kmeans_labels"]))

    print(f"Total windows processed: {total_windows}")
    print(f"Window size: {window_size} trading days")
    print(f"Step size: {step_size} trading days")
    print(f"Number of assets: {num_assets}")
    print(f"Number of clusters (k): {k}")

    kmeans_history = [w["kmeans_labels"] for w in window_results]
    hier_history = [w["hierarchical_labels"] for w in window_results]

    avg_kmeans_change = np.mean([
        np.sum(kmeans_history[i] != kmeans_history[i - 1]) / num_assets
        for i in range(1, total_windows)
    ]) if total_windows > 1 else 0

    avg_hier_change = np.mean([
        np.sum(hier_history[i] != hier_history[i - 1]) / num_assets
        for i in range(1, total_windows)
    ]) if total_windows > 1 else 0

    print(f"Avg K-Means cluster change rate: {avg_kmeans_change:.3f}")
    print(f"Avg Hierarchical cluster change rate: {avg_hier_change:.3f}")

    print("\nPer-window cluster sizes:")
    for i, w in enumerate(window_results):
        start = w["window_start"].strftime("%Y-%m-%d")
        end = w["window_end"].strftime("%Y-%m-%d")
        kmeans_labels = w["kmeans_labels"]
        hier_labels = w["hierarchical_labels"]

        slow_print(f"\nWindow {i+1}: {start} → {end}")
        slow_print(f"  K-Means clusters:")
        for label in np.unique(kmeans_labels):
            count = np.sum(kmeans_labels == label)
            slow_print(f"    Cluster {label} size: {count}")
        slow_print(f"  Hierarchical clusters:")
        for label in np.unique(hier_labels):
            count = np.sum(hier_labels == label)
            slow_print(f"    Cluster {label} size: {count}")