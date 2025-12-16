'''
================================================================================
Script: similarity.py
Created by: Malhar Gujar wih Anthony Schauer
--------------------------------------------------------------------------------
Overview:
This script computes two types of similarity matrices: Pearson correlation and 
Cosine similarity. These matrices help identify the relationships between assets 
in the stock universe. The Pearson correlation matrix measures linear 
relationships, while the Cosine similarity matrix evaluates the angle between 
return vectors to quantify similarity. Both matrices are critical for the 
similarity-based clustering and analysis workflow.
--------------------------------------------------------------------------------
Similarity Computation Steps:
1. Compute Pearson Correlation
   - Computes the Pearson correlation matrix from normalized returns
   - Rows represent dates, and columns represent asset tickers
   - Outputs a correlation matrix with tickers as both the row and column indices

2. Compute Cosine Similarity
   - Computes the Cosine similarity matrix for normalized returns
   - The DataFrame is transposed to use tickers as features
   - Outputs a similarity matrix with tickers as both the row and column indices
================================================================================
'''

import pandas as pd
import numpy as np


def compute_pearson(df):
    """
    Parameters:
        df : DataFrame of normalized returns (rows = dates, columns = tickers)
    Returns:
        DataFrame : Pearson correlation matrix (tickers × tickers)
    """
    corr_matrix = df.corr()
    return corr_matrix

def summarize_pearson(corr_df, labels=None):
    print("\n--- PEARSON CORRELATION SUMMARY ---")
    n_assets = corr_df.shape[0]
    print(f"Number of assets: {n_assets}")
    
    # Overall stats
    tril_idx = np.tril_indices(n_assets, k=-1)
    all_corrs = corr_df.values[tril_idx]
    print(f"Overall correlation: mean={all_corrs.mean():.3f}, std={all_corrs.std():.3f}, min={all_corrs.min():.3f}, max={all_corrs.max():.3f}")

    # Optional: summarize intra-cluster correlations
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = np.where(labels == label)[0]
            if len(idx) > 1:
                sub_corr = corr_df.values[np.ix_(idx, idx)]
                intra_corr = sub_corr[np.triu_indices_from(sub_corr, k=1)]
                print(f"Cluster {label} intra-cluster correlation: mean={intra_corr.mean():.3f}, std={intra_corr.std():.3f}, min={intra_corr.min():.3f}, max={intra_corr.max():.3f}")