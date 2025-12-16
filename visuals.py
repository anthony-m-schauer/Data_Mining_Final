"""
visuals.py

Overview:
---------
This module contains visualization and interpretation utilities for the
S&P 500 clustering project. These functions DO NOT perform analysis.
They consume outputs from the pipeline (returns, similarity, clustering)
and produce interpretable visual and textual summaries.

Visuals Included:
-----------------
1. PCA scatter plot of clusters
2. Pearson correlation heatmap (ordered by cluster)
3. Hierarchical clustering dendrogram
4. Sliding window cluster evolution summary

Each function also prints structured terminal output to support
interpretation without relying solely on visuals.
"""

# =========================
# Step 0: Imports
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


# =========================
# Step 1: PCA Cluster Plot
# =========================
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

    # PCA on tickers (transpose)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(returns_df.T)

    print("Explained variance ratio:")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.3f}")
    print(f"  Total (PC1+PC2): {pca.explained_variance_ratio_.sum():.3f}")

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA Cluster Visualization ({method_name})")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.show()


# =========================
# Step 2: Correlation Heatmap
# =========================
def plot_correlation_heatmap(corr_df, labels):
    """
    Heatmap of Pearson correlation matrix reordered by cluster labels.

    Parameters:
        corr_df : DataFrame (tickers × tickers)
        labels  : array-like cluster labels
    """

    print("\n--- CORRELATION HEATMAP SUMMARY ---")

    # Reorder by cluster
    order = np.argsort(labels)
    reordered = corr_df.values[order][:, order]

    # Cluster coherence metric
    unique_clusters = np.unique(labels)
    intra_corrs = []

    for c in unique_clusters:
        idx = np.where(labels == c)[0]
        if len(idx) > 1:
            sub = corr_df.values[np.ix_(idx, idx)]
            intra_corrs.append(sub[np.triu_indices_from(sub, 1)].mean())

    print(f"Average intra-cluster correlation: {np.mean(intra_corrs):.3f}")
    print(f"Cluster count: {len(unique_clusters)}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(reordered, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson Correlation")
    plt.title("Pearson Correlation Heatmap (Cluster-Ordered)")
    plt.tight_layout()
    plt.show()


# =========================
# Step 3: Hierarchical Dendrogram
# =========================
def plot_dendrogram_from_distance(distance_df):
    """
    Dendrogram visualization from a precomputed distance matrix.

    Parameters:
        distance_df : DataFrame (tickers × tickers), distance matrix
    """

    print("\n--- DENDROGRAM SUMMARY ---")

    # Convert square distance matrix → condensed
    condensed = squareform(distance_df.values, checks=False)

    # Linkage
    Z = linkage(condensed, method="average")

    print(f"Linkage matrix shape: {Z.shape}")
    print("First 5 linkage merges:")
    print(Z[:5])

    # Plot
    plt.figure(figsize=(10, 5))
    dendrogram(Z, no_labels=True)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Stocks")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()


# =========================
# Step 4: Sliding Window Summary
# =========================
def summarize_sliding_windows(window_results):
    """
    Textual and numeric summary of sliding window clustering results.

    Parameters:
        window_results : list of dicts, output from sliding_window.py
    """

    print("\n--- SLIDING WINDOW SUMMARY ---")
    print(f"Total windows processed: {len(window_results)}")

    cluster_counts = []

    for i, w in enumerate(window_results):
        labels = w["kmeans_labels"]
        unique = len(np.unique(labels))
        cluster_counts.append(unique)

        print(f"Window {i+1}: {w['window_start']} → {w['window_end']}")
        print(f"  Clusters detected: {unique}")

    print("\nCluster count stability:")
    print(f"  Min clusters: {min(cluster_counts)}")
    print(f"  Max clusters: {max(cluster_counts)}")
    print(f"  Mean clusters: {np.mean(cluster_counts):.2f}")
