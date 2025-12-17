# Decoding Market Behavior with Clustering

**Authors:** Malhar Gujar, Anthony Schauer

---

## Overview

This project implements a complete data mining pipeline for analyzing historical S&P 500 stock behavior using unsupervised learning techniques. The pipeline downloads historical price data, computes and normalizes daily returns, measures similarity between stocks, applies clustering methods, and optionally performs a sliding window analysis to study how market structure evolves over time.

The goal of the project is to identify meaningful stock groupings, compare them to known sector classifications, uncover non-obvious co-movements between companies, and analyze the stability of these clusters over time.

---

## Project Structure

| File | Description |
|-----|-------------|
| `data_pipeline.py` | Main driver script that orchestrates data download, preprocessing, similarity computation, clustering, and optional sliding window analysis. |
| `sliding_window.py` | Implements sliding window analysis over return data, recomputing similarity matrices and clustering results for each window. |
| `clustering.py` | Contains K-Means and Hierarchical clustering functions used throughout the project. |
| `similarity.py` | Computes similarity matrices, including Pearson correlation and cosine similarity. |
| `tickers_list.py` | Defines the list of S&P 500 stock tickers used in the analysis. |
| `visuals.py` | Optional utilities for generating plots and visual summaries of results. |
| `requirements.txt` | Python dependencies required to run the pipeline. |

---

## Pipeline Workflow

### 1. Data Collection
- Historical adjusted close prices are downloaded using `yfinance`.
- Data is retrieved for all S&P 500 tickers defined in `tickers_list.py`.
- Downloads are performed in batches to reduce memory usage.

---

### 2. Return Computation
- Daily returns are computed from adjusted close prices.
- Days containing missing values across all stocks are dropped.
- The resulting dataset is a clean returns matrix with:
  - Rows representing trading days  
  - Columns representing individual stocks  

---

### 3. Normalization
- Daily returns are standardized using z-score normalization.
- This ensures clustering is driven by co-movement patterns rather than absolute price scale or volatility.

---

### 4. Similarity Computation
- A Pearson correlation matrix is computed from the normalized returns.
- Optionally, a cosine similarity matrix can also be calculated.
- These matrices represent pairwise similarity between stocks based on historical behavior.

---

### 5. Clustering

Two clustering methods are applied:

#### K-Means Clustering
- Applied directly to normalized return vectors.
- Produces a fixed number of clusters.

#### Hierarchical Clustering
- Uses a precomputed distance matrix defined as  
  `distance = 1 − Pearson correlation`.
- Captures higher-level structure and relationships between stocks.

---

### 6. Sliding Window Analysis (Optional)
- A sliding time window is applied to the return data to analyze temporal dynamics.
- Default parameters:
  - Window size: 252 trading days (approximately one year)
  - Step size: 21 trading days (approximately one month)
- For each window:
  - Returns are subset to the window
  - Similarity matrices are recomputed
  - Clustering is rerun
- This allows analysis of cluster stability and evolving market structure over time.

Observed changes in hierarchical clustering across windows are expected and reflect real shifts in market relationships rather than implementation errors.

---

### 7. Output and Interpretation
- Summary information is printed to the console during execution.
- Cluster assignments and window-level results are stored in memory for interpretation.
- Optional visualizations can be generated using `visuals.py`.

---

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
