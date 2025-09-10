"""Churn predictor module (lightweight baseline).

Defines:
 - derive_churn_label: label customers as churned if recency > threshold.
 - train_churn_model: logistic regression.
 - predict_churn: probability outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def derive_churn_label(features: pd.DataFrame, recency_threshold: int = 90) -> pd.Series:
	if 'Recency' not in features.columns:
		raise ValueError("Features must include 'Recency'")
	return (features['Recency'] > recency_threshold).astype(int).rename('churn')


@dataclass
class ChurnModel:
	model: LogisticRegression
	auc: float
	feature_cols: list[str]


def train_churn_model(features: pd.DataFrame, label: pd.Series, feature_cols: Optional[list[str]] = None) -> ChurnModel:
	if feature_cols is None:
		feature_cols = [c for c in features.columns if c not in label.index.names and features[c].dtype != object]
	X = features[feature_cols]
	y = label.loc[features.index]
	X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
	clf = LogisticRegression(max_iter=500, class_weight='balanced')
	clf.fit(X_train, y_train)
	proba = clf.predict_proba(X_test)[:,1]
	auc = roc_auc_score(y_test, proba)
	return ChurnModel(model=clf, auc=auc, feature_cols=feature_cols)


def predict_churn(model: ChurnModel, features: pd.DataFrame) -> pd.Series:
	X = features[model.feature_cols]
	proba = model.model.predict_proba(X)[:,1]
	return pd.Series(proba, index=features.index, name='churn_probability')


__all__ = [
	'derive_churn_label',
	'train_churn_model',
	'predict_churn',
	'ChurnModel'
]

