"""Visualization helpers for Streamlit dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.express as px


def plot_clusters(embeddings: pd.DataFrame, labels: pd.Series):
	df = embeddings.copy()
	df['cluster'] = labels.values
	color_col = 'cluster'
	x_col, y_col = embeddings.columns[:2]
	fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title='Customer Segments', opacity=0.7)
	return fig


def kpi_summary(labels: pd.Series) -> pd.DataFrame:
	counts = labels.value_counts().sort_index()
	pct = (counts / counts.sum() * 100).round(2)
	return pd.DataFrame({'count': counts, 'pct': pct})


def segment_transition_table(transitions: pd.DataFrame) -> pd.DataFrame:
	if transitions.empty:
		return transitions
	return transitions.sort_values('timestamp', ascending=False)


__all__ = [
	'plot_clusters',
	'kpi_summary',
	'segment_transition_table'
]

