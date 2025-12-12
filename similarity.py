def compute_pearson(df):
    """
    Compute Pearson correlation matrix.

    Parameters:
        df : DataFrame of normalized returns (rows = dates, columns = tickers)

    Returns:
        DataFrame : Pearson correlation matrix (tickers × tickers)
    """
    print("Computing Pearson correlation matrix...")
    corr_matrix = df.corr()  # pandas automatically computes Pearson
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
    cos_matrix = cosine_similarity(df.T)  # transpose because features = tickers
    cos_df = pd.DataFrame(cos_matrix, index=df.columns, columns=df.columns)
    print("Cosine similarity shape:", cos_df.shape)
    return cos_df
 