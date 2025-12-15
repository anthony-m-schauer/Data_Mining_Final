'''
================================================================================
Script: clustering.py
Created by: Malhar Gujar
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

from sklearn.cluster import KMeans, AgglomerativeClustering

def run_kmeans(df, k):
    print(f"Running K-Means clustering with k={k}...")
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(df)
    print("K-Means complete. Labels shape:", labels.shape)
    return labels, model

def run_hierarchical(df, k, metric="euclidean"):
    print(f"Running Hierarchical clustering with k={k} using metric={metric}...")
    model = AgglomerativeClustering(n_clusters=k, metric=metric, linkage="average")
    labels = model.fit_predict(df)
    print("Hierarchical clustering complete. Labels shape:", labels.shape)
    return labels, model

'''
def run_dbscan(df, eps=0.5, min_samples=5):
    print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}...")
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(df)
    print("DBSCAN complete. Labels shape:", labels.shape)
    return labels, model 
'''
