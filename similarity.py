'''
================================================================================
Script: similarity.py
Created by: Malhar Gujar
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

def compute_pearson(df):
    """
    Compute Pearson correlation matrix.

    Parameters:
        df : DataFrame of normalized returns (rows = dates, columns = tickers)

    Returns:
        DataFrame : Pearson correlation matrix (tickers × tickers)
    """
    print("Computing Pearson correlation matrix...")
    corr_matrix = df.corr()
    print("Pearson matrix shape:", corr_matrix.shape)
    return corr_matrix


def compute_cosine(df):
    """
    Compute cosine similarity matrix.

    Parameters:
        df : DataFrame of normalized returns (rows = dates, columns = tickers)

    Returns:
        DataFrame : Cosine similarity matrix (tickers × tickers)
    """
    print("Computing Cosine similarity matrix...")
    cos_matrix = cosine_similarity(df.T)
    cos_df = pd.DataFrame(cos_matrix, index=df.columns, columns=df.columns)
    print("Cosine similarity shape:", cos_df.shape)
    return cos_df
 