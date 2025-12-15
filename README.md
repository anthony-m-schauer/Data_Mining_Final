# S&P 500 Data Mining Pipeline

## Overview
This repository contains a full data mining pipeline for analyzing historical S&P 500 stock data.  
It downloads stock prices, computes returns, normalizes data, calculates similarity matrices, performs clustering, and applies a sliding window analysis to track evolving market patterns.

---

## Project Structure

| File | Description |
|------|-------------|
| `data_pipeline.py` | Main pipeline script that downloads, processes, and normalizes stock data, computes similarity matrices, runs clustering, and executes sliding window analysis. |
| `sliding_window.py` | Implements sliding window analysis over the return data. Computes similarity and clustering within each window. |
| `clustering.py` | Contains K-Means and Hierarchical clustering functions. |
| `similarity.py` | Computes Pearson and Cosine similarity matrices. |
| `tickers_list.py` | Contains a Python list of current S&P 500 stock tickers. |
| `visuals.py` | (Optional) Generates plots or visualizations of the analysis results. |
| `requirements.txt` | Python dependencies for running the pipeline. |

---

## Pipeline Steps

1. **Download Stock Data**
   - Adjusted close prices for S&P 500 tickers using `yfinance`.
   - Downloads in batches to minimize memory usage.

2. **Compute Daily Returns**
   - Converts price data to daily percent changes.
   - Handles missing values by dropping days with all NaNs.

3. **Normalize Returns**
   - Applies z-score normalization to the daily returns.

4. **Compute Similarity**
   - Pearson correlation matrix of normalized returns.
   - Cosine similarity (optional).

5. **Clustering**
   - K-Means clustering on normalized returns.
   - Hierarchical clustering using precomputed distance (1 - Pearson).

6. **Sliding Window Analysis**
   - Window size: 1 year of trading days (default: 252).
   - Step size: 1 month of trading days (default: 21).
   - Recomputes similarity and clustering for each window.
   - Simulates evolving market behavior and captures temporal trends.

7. **Memory Management**
   - Deletes temporary DataFrames to minimize RAM usage.

---

## Usage

### Run the Full Pipeline
```bash
python data_pipeline.py
