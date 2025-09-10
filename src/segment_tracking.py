"""Segment tracking utilities.

Tracks customer cluster transitions over time snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd


@dataclass
class TransitionEvent:
	customer_id: str
	from_cluster: int
	to_cluster: int
	timestamp: pd.Timestamp


class SegmentTracker:
	def __init__(self):
		self.history: Dict[str, int] = {}
		self.events: list[TransitionEvent] = []
		self.snapshots: list[Tuple[pd.Timestamp, pd.Series]] = []

	def update(self, timestamp: pd.Timestamp, labels: pd.Series) -> None:
		"""Record a new labeling snapshot and detect movements."""
		labels = labels.astype(int)
		self.snapshots.append((timestamp, labels))
		for cid, new_c in labels.items():
			old_c = self.history.get(cid)
			if old_c is not None and old_c != new_c:
				self.events.append(TransitionEvent(str(cid), old_c, new_c, timestamp))
			self.history[cid] = new_c

	def transition_log(self) -> pd.DataFrame:
		if not self.events:
			return pd.DataFrame(columns=["customer_id","from_cluster","to_cluster","timestamp"])
		return pd.DataFrame([e.__dict__ for e in self.events])

	def segment_sizes(self) -> pd.DataFrame:
		rows = []
		for ts, labels in self.snapshots:
			counts = labels.value_counts().to_dict()
			counts["timestamp"] = ts
			rows.append(counts)
		if not rows:
			return pd.DataFrame()
		df = pd.DataFrame(rows).fillna(0).set_index("timestamp").sort_index()
		return df.astype(int)

__all__ = ["SegmentTracker", "TransitionEvent"]

