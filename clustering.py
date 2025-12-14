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
def run_hierarchical(df):
    print(f"Running Hierarchical clustering with k={k}...")
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(df)
    print("Hierarchical clustering complete. Labels shape:", labels.shape)
    return labels, model

def run_dbscan(df, eps=0.5, min_samples=5):
    print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}...")
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(df)
    print("DBSCAN complete. Labels shape:", labels.shape)
    return labels, model 
'''
