AI-Powered Customer Segmentation & Adaptive Streaming Dashboard
================================================================

Overview
--------
An end‑to‑end system for ecommerce customer intelligence: it ingests the UCI *Online Retail II* dataset, performs robust preprocessing & feature engineering (RFM + behavioral enrichments), explores static clustering methods (KMeans, GMM, DBSCAN) as well as incremental / streaming segmentation (MiniBatchKMeans), tracks customer movements between segments over simulated time, and presents an interactive Streamlit dashboard with dynamic visualizations, churn risk estimation, and batch/stream mode.

<img width="1883" height="845" alt="image" src="https://github.com/user-attachments/assets/052aa569-fbd0-4562-a443-8ae0a0002208" />
<img width="1539" height="827" alt="image" src="https://github.com/user-attachments/assets/82fbffd6-2f1f-44e8-8c22-b6e3c948094b" />



Motivation
----------
Traditional segmentation runs offline on historical snapshots and becomes stale as customer behavior shifts. Modern personalization, retention, and lifecycle marketing need:
1. Near real‑time adaptation to evolving purchase patterns.
2. Explainable, lightweight features (RFM) enriched with behavior (diversity, AOV).
3. Transparent tracking of segment drift + customer transitions.
4. Fast experimentation across clustering paradigms (static vs incremental vs density based) without rebuilding pipelines.

Problem Statement
-----------------
Provide a system that:
* Cleans noisy transactional retail data (missing IDs, cancellations, anomalies).
* Generates resilient customer-level feature vectors for clustering / churn modeling.
* Supports multiple clustering algorithms (partitioning, probabilistic, density, incremental) under a unified interface.
* Simulates streaming arrival of transactions (day / week / row batches) enabling “online” segment refresh.
* Tracks segment size evolution & individual movement events (e.g., high-value → at-risk).
* Exposes an interactive dashboard for analysts / marketers to explore clusters, metrics distributions, and streaming dynamics.

Dataset Selection
-----------------
Dataset: UCI Machine Learning Repository / Kaggle mirror – *Online Retail II* (UK/European online store transactions 2009–2011). Chosen because:
* Realistic B2C multi-order purchase behavior.
* Contains timestamps (enabling temporal splitting & streaming simulation).
* Monetary signals (Quantity, UnitPrice) support RFM + revenue based KPIs.
* Common enough to benchmark approach vs public examples.

Data Challenges Addressed
* Missing / null `CustomerID` rows (removed for customer modeling while optionally retained for transactional aggregate stats).
* Cancellations / returns (negative `Quantity`).
* Zero / negative pricing anomalies.
* Inconsistent column naming variants; normalized automatically (`InvoiceNo`, `InvoiceDate`, `CustomerID`, `Quantity`, `UnitPrice`, `TotalPrice`, `StockCode`).
* Outlier handling (unit price trimming via robust fence in earlier revision; easily extensible).

System Architecture Phases
--------------------------
1. Ingestion & Normalization
	- Flexible column mapping handles variant names (e.g. `invoice_no`, `unit_price`, `customer_id`).
	- Supports CSV / Excel input (here fixed to `data/raw/ecommerce-dataset/online_retail_II.csv`).

2. Cleaning
	- Drop rows without `CustomerID` (core modeling key).
	- Parse `InvoiceDate` to timezone-naïve UTC‑style timestamp.
	- Remove negative quantity, non‑positive prices.
	- Derive `TotalPrice` if missing.

3. Feature Engineering
	- RFM (Recency days, Frequency distinct invoices, Monetary total spend).
	- Product diversity (# unique `StockCode`).
	- Average Unit Price & Average Order Value (AOV).
	- Scaled variants (`*_scaled`) via Standard or MinMax scaling.

4. Static Clustering
	- Algorithms: KMeans, Gaussian Mixture (GMM), DBSCAN, MiniBatchKMeans (used here in static variant too).
	- Optional dimensionality reduction: PCA / UMAP for visualization.
	- Silhouette score (auto‑excludes DBSCAN noise cluster `-1`).

5. Streaming / Incremental Segmentation
	- Transaction stream generator slices by day / week / fixed row batch.
	- Incremental updates via `MiniBatchKMeans.partial_fit`.
	- Periodic full re‑labeling and trend visualization (segment size line chart, transition log).

6. Segment Tracking
	- Tracks each customer’s last cluster and records transition events with timestamps.
	- Maintains time series of cluster population for drift analysis.

7. Churn Risk (Optional)
	- Label: Recency > configurable threshold ⇒ churn=1.
	- Baseline Logistic Regression; balanced class weighting.
	- Produces AUC metric + probability ranking.

8. Dashboard & Visualization
	- Streamlit app with dark theme + custom CSS, KPI cards, tabbed interface.
	- Dynamic cluster filtering and metric distribution (Histogram / Box / ECDF).
	- Embedding scatter (PCA/UMAP) colored by cluster.
	- Streaming progress bar, live segment size line chart, transitions table.

9. Extensibility Hooks
	- Add marketing propensity models (plug into features table).
	- Replace clustering engine with River streaming models.
	- Persist model states / transitions to a database for historical auditing.

File / Module Structure
-----------------------
```
data/
  raw/
	 ecommerce-dataset/online_retail_II.csv          # Fixed input dataset
  processed/                                        # Generated artifacts (RFM, features)
  streaming/                                       # (Optional) prepared streaming slices
notebooks/
  Exploration.ipynb                                # Initial EDA / feature validation
src/
  __init__.py                                      # Public API exports
  data_preprocessing.py                            # Load, clean, feature engineering, scaling
  clustering_engine.py                             # Static & incremental clustering logic
  stream_simulator.py                              # Batch generator (day / week / rows)
  segment_tracking.py                              # Transition & size tracking
  churn_predictor.py                               # Baseline logistic churn model
  visual_dashboard.py                              # Plotly helper utilities
app.py                                             # Streamlit dashboard (fixed dataset path)
run.sh                                             # Convenience launcher (bash)
requirements.txt                                   # Python dependencies
README.md                                          # (This file)
```

Key Design Choices
------------------
* Flexible column normalization removes fragile manual renaming steps.
* Separation of concerns: preprocessing is stateless & testable; clustering module focuses on algorithm orchestration; dashboard purely presentation.
* Partial tolerance mode: pipeline continues even if non‑critical columns missing, enabling partial / incremental ingestion scenarios.
* Noise handling for DBSCAN integrated into metrics (silhouette adaptation).

How to Run
----------
Prerequisites:
* Python 3.10+ (tested on 3.12) and pip.
* (Optional) Bash if using `run.sh`; on Windows you can run commands directly in PowerShell.

1. Create & activate a virtual environment (recommended).
2. Install requirements.
3. Ensure the dataset file exists at `data/raw/ecommerce-dataset/online_retail_II.csv`.
4. Launch Streamlit app.

Quick Commands (PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Or via bash script (if available):
```
bash run.sh
```

Dashboard Usage
* Configuration sidebar (DBSCAN params when selected, scaling method, embedding, sample size).
* Tabs: Static Segmentation / Streaming Simulation / Data Preview.
* Select clusters to filter and inspect metric distributions.
* Run streaming batches repeatedly until complete; observe cluster drift and transitions.
* (Optional) enable churn model to view AUC and high-risk customers.

Tech Stack
----------
* Python: core language.
* pandas / numpy: data manipulation.
* scikit-learn: clustering (KMeans, MiniBatchKMeans, GMM via GaussianMixture, DBSCAN), scaling, logistic regression.
* UMAP (optional): non-linear dimensionality reduction for visualization.
* Plotly + Streamlit: interactive visual analytics UI.
* seaborn / matplotlib (EDA in notebooks).
* river (listed) – prepared for future online ML (not yet integrated in engine).

Performance Notes
-----------------
* MiniBatchKMeans chosen for streaming due to partial_fit support and speed; batch size adaptable.
* Column selection for clustering restricts to scaled numeric features for stable distance metrics.
* DBSCAN is included for density-based insight but not used for streaming (non-incremental in scikit-learn).

Extending / Next Steps
----------------------
* Add River’s streaming clustering / anomaly detection.
* Persist model & tracker state (e.g. SQLite or Parquet snapshots) across sessions.
* Add hyperparameter search (silhouette / Davies–Bouldin optimization) pane.
* Implement outlier flagging service for suspicious spikes.
* Introduce marketing action triggers (webhook / email simulation) on segment change events.

Testing Suggestions
-------------------
* Unit tests for normalization mapping, RFM correctness, and streaming batch generator.
* Synthetic data generation to validate edge cases (all customers single cluster, extreme recency, etc.).

License
-------
This project is released under the MIT License. See below.

MIT License
-----------
Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Acknowledgements
----------------
* UCI / Kaggle dataset authors of Online Retail II.
* scikit-learn & Streamlit communities for tooling.

