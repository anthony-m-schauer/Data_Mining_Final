'''
================================================================================
Script: clustering.py
Created by: Malhar Gujar with Anthony Schauer
--------------------------------------------------------------------------------
Overview:
This script provides implementations for two popular clustering algorithms: 
K-Means and Hierarchical Clustering. Both functions take in a DataFrame of 
normalized stock returns and compute cluster labels for the given number of 
clusters (k). These algorithms are essential for grouping similar assets based 
on their return patterns.
--------------------------------------------------------------------------------
Clustering Steps:
1. K-Means Clustering
   - Performs K-Means clustering on the DataFrame of normalized returns
   - Takes in a user-defined number of clusters k
   - Outputs the cluster labels and the fitted model

2. Hierarchical Clustering
   - Performs Agglomerative (Hierarchical) clustering on the DataFrame
   - Allows for different distance metrics (default is Euclidean)
   - Outputs the cluster labels and the fitted model
================================================================================
'''

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA


def run_kmeans(df, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(df)
    return labels, model

def run_hierarchical(df, k, metric="euclidean"):
    model = AgglomerativeClustering(n_clusters=k, metric=metric, linkage="average")
    labels = model.fit_predict(df)
    return labels, model


def summarize_clusters(returns_df, labels, method_name="K-Means", returns=None):
    """
    Parameters:
        returns_df : DataFrame of normalized returns (rows=dates, cols=tickers)
        labels     : array-like cluster labels
        method_name: str, name of clustering method
        returns    : optional, original returns DataFrame to compute avg return/volatility
    """
    tickers = returns_df.columns
    unique_labels = np.unique(labels)

    print(f"\n--- CLUSTER SUMMARY ({method_name}) ---")
    print(f"Number of clusters: {len(unique_labels)}")

    for label in unique_labels:
        members_idx = np.where(labels == label)[0]
        members = tickers[members_idx]
        print(f"\nCluster {label}: {len(members)} assets")
        print(f"\nExample tickers: {members[:5].tolist()}")  # first 5 tickers

    pca = PCA(n_components=2)
    coords = pca.fit_transform(returns_df.T)
    for label in unique_labels:
        members_idx = np.where(labels == label)[0]
        cluster_coords = coords[members_idx]
        mean_coords = cluster_coords.mean(axis=0)
        print(f"Cluster {label} mean PCA coords: {mean_coords}")

    if returns is not None:
        for label in unique_labels:
            members_idx = np.where(labels == label)[0]
            cluster_returns = returns.iloc[:, members_idx]
            avg_return = cluster_returns.mean().mean()
            vol = cluster_returns.std().mean()
            print(f"Cluster {label}: avg return {avg_return:.4f}, avg volatility {vol:.4f}")


def summarize_hierarchical(df, labels, method_name="Hierarchical"):
    """
    Parameters:
        df          : DataFrame used for clustering (tickers × tickers distance matrix)
        labels      : array of cluster labels from run_hierarchical
        method_name : string label for printing
    """
    print(f"\n--- {method_name.upper()} CLUSTER SUMMARY ---")
    
    n_assets = df.shape[0]
    print(f"Number of assets: {n_assets}")
    unique_labels = np.unique(labels)
    print(f"Number of clusters: {len(unique_labels)}")
    
    for label in unique_labels:
        count = np.sum(labels == label)
        print(f"Cluster {label} size: {count}")
    
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if len(idx) > 1:
            sub_dist = df.values[np.ix_(idx, idx)]
            tril_idx = np.tril_indices_from(sub_dist, k=-1)
            mean_dist = sub_dist[tril_idx].mean()
            print(f"Cluster {label} mean pairwise distance: {mean_dist:.3f}")
