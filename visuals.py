"""
================================================================================
Script: visuals.py
Created by: Anthony M. Schauer
--------------------------------------------------------------------------------
Overview:
This module contains visualization and interpretation utilities for the
S&P 500 clustering project. These functions DO NOT perform analysis.
They consume outputs from the pipeline (returns, similarity, clustering)
and produce interpretable visual and textual summaries.
--------------------------------------------------------------------------------
Visuals Included:
1. PCA scatter plot of clusters
2. Pearson correlation heatmap (ordered by cluster)
3. Hierarchical clustering dendrogram
4. Sliding window cluster evolution summary
================================================================================
"""

#### Step 0: Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


#### Step 1: PCA Cluster Plot
def plot_pca_clusters(returns_df, labels, method_name="K-Means"):
    """
    PCA projection of normalized returns with cluster coloring.
    Parameters:
        returns_df : DataFrame (rows = dates, cols = tickers)
        labels     : array-like cluster labels (length = num tickers)
        method_name: str, clustering method name for labeling
    """

    print("\n--- PCA CLUSTER SUMMARY ---")
    print(f"Clustering method: {method_name}")
    print(f"Number of clusters: {len(np.unique(labels))}")

    pca = PCA(n_components=2)
    coords = pca.fit_transform(returns_df.T)

    print("Explained variance ratio:")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.3f}")
    print(f"  Total (PC1+PC2): {pca.explained_variance_ratio_.sum():.3f}")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA Cluster Visualization ({method_name})")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.show()


##### Step 2: Correlation Heatmap
def plot_correlation_heatmap(corr_df, labels):
    """
    Heatmap of Pearson correlation matrix reordered by cluster labels.
    Parameters:
        corr_df : DataFrame (tickers x tickers)
        labels  : array-like cluster labels
    """
    print("\n--- CORRELATION HEATMAP SUMMARY ---")
    
    order = np.argsort(labels)
    reordered = corr_df.values[order][:, order]

    unique_clusters = np.unique(labels)
    intra_corrs = []

    for c in unique_clusters:
        idx = np.where(labels == c)[0]
        if len(idx) > 1:
            sub = corr_df.values[np.ix_(idx, idx)]
            intra_corrs.append(sub[np.triu_indices_from(sub, 1)].mean())

    print(f"Average intra-cluster correlation: {np.mean(intra_corrs):.3f}")
    print(f"Cluster count: {len(unique_clusters)}")

    plt.figure(figsize=(8, 6))
    plt.imshow(reordered, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson Correlation")
    plt.title("Pearson Correlation Heatmap (Cluster-Ordered)")
    plt.tight_layout()
    plt.show()


##### Step 3: Hierarchical Dendrogram
def plot_dendrogram_from_distance(distance_df):
    """
    Dendrogram visualization from a precomputed distance matrix.
    Parameters:
        distance_df : DataFrame (tickers × tickers), distance matrix
    """

    print("\n--- HIERARCHICAL SUMMARY ---")

    condensed = squareform(distance_df.values, checks=False)

    Z = linkage(condensed, method="average")

    print(f"Linkage matrix shape: {Z.shape}")
    print("First 5 linkage merges:")
    print(Z[:5])

    plt.figure(figsize=(10, 5))
    dendrogram(Z, no_labels=True)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Stocks")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()


##### Step 4: Sliding Window Summary
def plot_cluster_change_over_time(window_results, returns_df):
    """
    Plot cluster change rates across sliding windows.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    num_assets = returns_df.shape[1]

    kmeans_history = [w["kmeans_labels"] for w in window_results]
    hier_history = [w["hierarchical_labels"] for w in window_results]
    dates = [w["window_end"] for w in window_results]

    kmeans_changes = [
        np.sum(kmeans_history[i] != kmeans_history[i - 1]) / num_assets
        for i in range(1, len(kmeans_history))
    ]
    hier_changes = [
        np.sum(hier_history[i] != hier_history[i - 1]) / num_assets
        for i in range(1, len(hier_history))
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(dates[1:], kmeans_changes, label="K-Means")
    plt.plot(dates[1:], hier_changes, label="Hierarchical")
    plt.xlabel("Window End Date")
    plt.ylabel("Cluster Change Rate")
    plt.title("Cluster Stability Over Time (Sliding Windows)")
    plt.legend()
    plt.tight_layout()
    plt.show()
