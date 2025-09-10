"""Stream simulator: emits transactional batches in chronological order.

Supports slicing by day or week or a fixed number of rows.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Iterator, Literal, Optional

import pandas as pd


SliceMode = Literal["day", "week", "rows"]


def stream_batches(
	df: pd.DataFrame,
	mode: SliceMode = "day",
	rows_per_batch: int = 5000,
	timestamp_col: str = "InvoiceDate",
) -> Iterator[pd.DataFrame]:
	"""Yield batches of the dataframe to simulate streaming.

	mode:
	  - day: one batch per calendar day
	  - week: one batch per ISO week
	  - rows: fixed-size row chunks
	"""
	if timestamp_col not in df.columns:
		raise ValueError(f"timestamp_col '{timestamp_col}' missing from df")

	data = df.sort_values(timestamp_col).copy()
	data[timestamp_col] = pd.to_datetime(data[timestamp_col])

	if mode == "rows":
		for start in range(0, len(data), rows_per_batch):
			yield data.iloc[start:start + rows_per_batch]
		return

	if mode == "day":
		grouper = data[timestamp_col].dt.to_period('D')
	elif mode == "week":
		grouper = data[timestamp_col].dt.to_period('W')
	else:  # pragma: no cover
		raise ValueError("Unsupported mode")

	for _, batch in data.groupby(grouper):
		yield batch


__all__ = ["stream_batches"]

