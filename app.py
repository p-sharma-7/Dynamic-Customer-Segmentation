import os
from pathlib import Path
import time
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import silhouette_score

from src import (
    load_raw,
    clean_transactions,
    build_customer_feature_table,
    scale_features,
    fit_static,
    IncrementalClusterer,
    SegmentTracker,
    derive_churn_label,
    train_churn_model,
    predict_churn,
)
from src.visual_dashboard import plot_clusters, kpi_summary, segment_transition_table
from src.stream_simulator import stream_batches

st.set_page_config(page_title="AI Customer Segmentation", layout="wide", page_icon="🧬")

# -----------------------------
# Global Style / Theming
# -----------------------------
st.markdown(
    """
    <style>
    :root {
        --bg-card: #1f2937;
        --border-color: #374151;
        --accent: #6366f1;
        --accent-grad: linear-gradient(135deg,#6366f1,#8b5cf6 60%,#ec4899);
    }
    [data-testid=stSidebar] {background: #0f1115 !important;}
    .kpi-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:0.75rem;margin:.75rem 0 1rem;}
    .kpi-card {background:var(--bg-card);border:1px solid var(--border-color);border-radius:12px;padding:.65rem .85rem;position:relative;overflow:hidden;}
    .kpi-card:before {content:"";position:absolute;inset:0;opacity:.12;background:var(--accent-grad);}
    .kpi-value {font-size:1.35rem;font-weight:600;line-height:1.1;margin:0;color:#fbbf24;}
    .kpi-label {font-size:.70rem;letter-spacing:.08em;text-transform:uppercase;color:#9ca3af;}
    .section-box {background:#11161d;border:1px solid #1f2833;padding:1rem 1.1rem;border-radius:14px;margin-bottom:1rem;}
    .stDataFrame {background:#11161d !important;}
    .metric-small {font-size:.75rem;color:#9ca3af;margin-top:-4px}
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
# Fixed dataset path (permanent input)
RAW_FILE = DATA_RAW_DIR / "ecommerce-dataset" / "online_retail_II.csv"

@st.cache_data(show_spinner=False)
def load_and_prepare(scaler: str = "standard"):
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Fixed dataset not found at {RAW_FILE}. Ensure the file exists.")
    raw = load_raw(RAW_FILE)
    cleaned = clean_transactions(raw, allow_partial=True)
    features = build_customer_feature_table(cleaned)
    scaled = scale_features(features, method=scaler)
    return cleaned, scaled.data

# Sidebar controls
st.sidebar.header("Configuration (Fixed Dataset)")
st.sidebar.markdown(f"Using dataset: `{'/'.join(RAW_FILE.parts[-3:])}`")

scaler_method = st.sidebar.selectbox("Scaling", ["standard", "minmax"], index=0)
clust_algo = st.sidebar.selectbox("Algorithm", ["kmeans", "gmm", "minibatch_kmeans", "dbscan"], index=0)
if clust_algo == "dbscan":
    dbscan_eps = st.sidebar.slider("DBSCAN eps", 0.05, 5.0, 0.7, 0.05)
    dbscan_min_samples = st.sidebar.slider("DBSCAN min_samples", 3, 100, 10, 1)
else:
    dbscan_eps = 0.7
    dbscan_min_samples = 5
num_clusters = st.sidebar.slider("Clusters", 2, 15, 6)
reduce_method = st.sidebar.selectbox("Dimensionality Reduction", ["none", "pca", "umap"], index=1)
stream_slice = st.sidebar.selectbox("Stream Slice", ["day", "week", "rows"], index=0)
rows_per_batch = st.sidebar.number_input("Rows/batch (rows mode)", 500, 100000, 5000, step=500)
train_churn = st.sidebar.toggle("Train churn model", value=False)
recency_thresh = st.sidebar.number_input("Churn Recency Threshold (days)", 30, 365, 90, step=15)
sample_n = st.sidebar.number_input("Sample rows to preview", 5, 200, 20, step=5)
st.sidebar.caption("Adjust parameters then switch tabs for Static vs Streaming.")

if not RAW_FILE.exists():
    st.error(f"Fixed dataset missing: {RAW_FILE}")
    st.stop()

with st.spinner("Loading & preprocessing data..."):
    try:
        cleaned_df, feature_df = load_and_prepare(scaler=scaler_method)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.info("Expected columns (case/format agnostic): InvoiceNo, InvoiceDate, CustomerID, Quantity, UnitPrice, StockCode, TotalPrice (optional). Provide at least InvoiceDate, CustomerID.")
        st.stop()

cust_count = cleaned_df['CustomerID'].nunique()
invoice_count = cleaned_df['InvoiceNo'].nunique() if 'InvoiceNo' in cleaned_df.columns else cleaned_df.shape[0]
date_min = cleaned_df['InvoiceDate'].min()
date_max = cleaned_df['InvoiceDate'].max()

# KPI Cards
st.markdown('<div class="kpi-grid">' + ''.join([
    f"""<div class='kpi-card'><div class='kpi-value'>{cust_count:,}</div><div class='kpi-label'>Customers</div></div>""",
    f"""<div class='kpi-card'><div class='kpi-value'>{invoice_count:,}</div><div class='kpi-label'>Invoices</div></div>""",
    f"""<div class='kpi-card'><div class='kpi-value'>{len(cleaned_df):,}</div><div class='kpi-label'>Transactions</div></div>""",
    f"""<div class='kpi-card'><div class='kpi-value'>{(date_max-date_min).days}</div><div class='kpi-label'>Span Days</div></div>""",
]) + '</div>', unsafe_allow_html=True)

tabs = st.tabs(["Static Segmentation", "Streaming Simulation", "Data Preview"])

# -----------------------------
# Static Tab
# -----------------------------
with tabs[0]:
    reduce = None if reduce_method == "none" else reduce_method
    feature_cols_all = [c for c in feature_df.columns if c.endswith('_scaled')]
    selected_cols = st.multiselect("Feature columns", feature_cols_all, default=feature_cols_all)
    working_features = feature_df[selected_cols] if selected_cols else feature_df[feature_cols_all]

    with st.spinner("Clustering customers..."):
        result = fit_static(
            working_features,
            algo=clust_algo if clust_algo != "minibatch_kmeans" else "minibatch_kmeans",
            n_clusters=num_clusters,
            reduce=reduce,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )
        labels = result.labels
        # Silhouette (only if >1 cluster and not gmm maybe but we can compute anyway)
        sil = None
        try:
            # Exclude DBSCAN noise label (-1) for silhouette; need at least 2 clusters >1 sample
            label_for_sil = labels if clust_algo != 'dbscan' else labels[labels != -1]
            if label_for_sil.nunique() > 1 and len(label_for_sil) > label_for_sil.nunique():
                sil = silhouette_score(working_features.loc[label_for_sil.index], label_for_sil)
        except Exception:
            pass

    kpi_df = kpi_summary(labels)
    col_kpi, col_score = st.columns([3,1])
    with col_kpi:
        st.dataframe(kpi_df, use_container_width=True)
    with col_score:
        st.metric("Silhouette", f"{sil:.3f}" if sil is not None else "-")

    # Cluster selection filter
    feature_with_labels = feature_df.join(labels)
    cluster_filter = st.multiselect(
        "Filter clusters", sorted(labels.unique()), default=sorted(labels.unique())
    )
    filtered = feature_with_labels[feature_with_labels['cluster'].isin(cluster_filter)] if cluster_filter else feature_with_labels

    # Embedding plot (respect filter)
    if result.embeddings is not None:
        emb_filtered = result.embeddings.loc[filtered.index]
        st.plotly_chart(plot_clusters(emb_filtered, filtered['cluster']), use_container_width=True)

    # Dynamic metric distribution
    base_metrics = [c for c in ["Recency","Frequency","Monetary","AvgOrderValue"] if c in filtered.columns]
    if base_metrics:
        col_m1, col_m2 = st.columns([2,1])
        with col_m1:
            metric_choice = st.selectbox("Metric", base_metrics, index=0)
        with col_m2:
            chart_type = st.radio("Chart", ["Histogram","Box","ECDF"], horizontal=True)

        plot_df = filtered[[metric_choice,'cluster']].copy()
        if chart_type == "Histogram":
            fig_metric = px.histogram(plot_df, x=metric_choice, color='cluster', marginal='rug', nbins=40, opacity=0.75)
        elif chart_type == "Box":
            fig_metric = px.box(plot_df, x='cluster', y=metric_choice, points='suspectedoutliers')
        else:  # ECDF
            fig_metric = px.ecdf(plot_df, x=metric_choice, color='cluster')
        fig_metric.update_layout(title=f"{chart_type} of {metric_choice} (filtered)")
        st.plotly_chart(fig_metric, use_container_width=True)

    st.write("Sample Customers (after filter)")
    st.dataframe(filtered.head(sample_n), use_container_width=True)

    if train_churn:
        label_series = derive_churn_label(feature_with_labels.rename(columns=str.capitalize), recency_threshold=recency_thresh)
        model = train_churn_model(feature_with_labels, label_series)
        st.info(f"Churn model AUC: {model.auc:.3f}")
        churn_scores = predict_churn(model, feature_with_labels)
        st.dataframe(churn_scores.sort_values(ascending=False).head(sample_n), use_container_width=True)

# -----------------------------
# Streaming Tab
# -----------------------------
with tabs[1]:
    st.write("Simulate incremental learning on chronological batches.")
    reduce = None if reduce_method == "none" else reduce_method
    if 'stream_state' not in st.session_state:
        st.session_state.stream_state = {
            'ic': IncrementalClusterer(n_clusters=num_clusters),
            'tracker': SegmentTracker(),
            'completed': False,
            'batches_processed': 0,
        }
    ss = st.session_state.stream_state
    # Reset if user changed cluster count or algo drastically (simple heuristic)
    if ss['ic'].n_clusters != num_clusters:
        st.session_state.stream_state = {
            'ic': IncrementalClusterer(n_clusters=num_clusters),
            'tracker': SegmentTracker(),
            'completed': False,
            'batches_processed': 0,
        }
        ss = st.session_state.stream_state

    time_col = 'InvoiceDate'
    all_batches = list(stream_batches(cleaned_df, mode=stream_slice, rows_per_batch=rows_per_batch, timestamp_col=time_col))
    total_batches = len(all_batches)
    run_stream = st.button("Run / Continue Streaming", type="primary")
    progress = st.progress(0.0, text="Idle")
    kpi_placeholder = st.empty()
    plot_placeholder = st.empty()
    size_line_placeholder = st.empty()
    sizes_table_placeholder = st.empty()
    transitions_placeholder = st.empty()

    if run_stream and not ss['completed']:
        start_index = ss['batches_processed']
        for i in range(start_index, total_batches):
            batch = all_batches[i]
            cust_features = build_customer_feature_table(batch)
            scaled = scale_features(cust_features).data
            ss['ic'].partial_fit(scaled)
            labels_full = ss['ic'].predict(feature_df)  # full prediction
            ts = batch[time_col].max()
            ss['tracker'].update(pd.to_datetime(ts), labels_full)
            ss['batches_processed'] = i + 1
            progress.progress(ss['batches_processed']/total_batches, text=f"Processed {ss['batches_processed']}/{total_batches} batches")
            if reduce == 'pca':
                from sklearn.decomposition import PCA
                p = PCA(n_components=2, random_state=42)
                emb = p.fit_transform(feature_df[[c for c in feature_df.columns if c.endswith('_scaled')]])
                plot_placeholder.plotly_chart(plot_clusters(pd.DataFrame(emb, columns=['PC1','PC2'], index=feature_df.index), labels_full), use_container_width=True)
            elif reduce == 'umap':
                try:
                    import umap
                    reducer = umap.UMAP(n_components=2, random_state=42)
                    emb = reducer.fit_transform(feature_df[[c for c in feature_df.columns if c.endswith('_scaled')]])
                    plot_placeholder.plotly_chart(plot_clusters(pd.DataFrame(emb, columns=['UMAP1','UMAP2'], index=feature_df.index), labels_full), use_container_width=True)
                except Exception:
                    st.error('UMAP not available.')
            kpi_placeholder.dataframe(kpi_summary(labels_full), use_container_width=True)
            seg_sizes_df = ss['tracker'].segment_sizes()
            if not seg_sizes_df.empty:
                # Line chart for segment sizes over time (melt to long form)
                melt_df = seg_sizes_df.reset_index().melt(id_vars='timestamp', var_name='cluster', value_name='count')
                fig_sizes = px.line(melt_df, x='timestamp', y='count', color='cluster', title='Segment Size Over Time')
                size_line_placeholder.plotly_chart(fig_sizes, use_container_width=True)
                sizes_table_placeholder.dataframe(seg_sizes_df.tail(10), use_container_width=True)
            transitions_placeholder.dataframe(segment_transition_table(ss['tracker'].transition_log()).head(25), use_container_width=True)
            # yield control for responsiveness
            time.sleep(0.05)
        if ss['batches_processed'] >= total_batches:
            ss['completed'] = True
            progress.progress(1.0, text="Streaming complete")
    elif ss['completed']:
        st.success("Streaming already completed. Adjust parameters or reload to rerun.")

# -----------------------------
# Data Preview Tab
# -----------------------------
with tabs[2]:
    st.write("Raw & Feature Data Samples")
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Cleaned Transactions")
        st.dataframe(cleaned_df.head(sample_n), use_container_width=True)
    with col_b:
        st.caption("Customer Features (scaled columns included)")
        st.dataframe(feature_df.head(sample_n), use_container_width=True)

# Footer
st.caption("Adaptive Customer Segmentation Dashboard © 2025 • Responsive UI Mode")
