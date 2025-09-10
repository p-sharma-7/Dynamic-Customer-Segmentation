"""Data preprocessing with flexible column normalization.

Handles common variants of required columns from the Online Retail dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Canonical column names we will enforce after normalization
CANON = {
    "invoiceno": "InvoiceNo",
    "invoice": "InvoiceNo",
    "invoice_number": "InvoiceNo",
    "invoice_no": "InvoiceNo",
    "invoicedate": "InvoiceDate",
    "invoice_date": "InvoiceDate",
    "customerid": "CustomerID",
    "customer_id": "CustomerID",
    "customer": "CustomerID",
    "quantity": "Quantity",
    "qty": "Quantity",
    "unitprice": "UnitPrice",
    "unit_price": "UnitPrice",
    "price": "UnitPrice",
    "stockcode": "StockCode",
    "stock_code": "StockCode",
    "productcode": "StockCode",
    "totalprice": "TotalPrice",
    "total_price": "TotalPrice",
}

REQUIRED = {"InvoiceNo", "InvoiceDate", "CustomerID", "Quantity", "UnitPrice"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping: Dict[str, str] = {}
    for col in df.columns:
        key = ''.join(ch for ch in col.lower() if ch.isalnum() or ch == '_')
        key = key.replace('__', '_')
        if key in CANON:
            mapping[col] = CANON[key]
    if mapping:
        df = df.rename(columns=mapping)
    return df


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path, dtype={"CustomerID": str}, engine=None)
    else:
        df = pd.read_csv(path, dtype={"CustomerID": str})
    return _normalize_columns(df)


def clean_transactions(df: pd.DataFrame, allow_partial: bool = True) -> pd.DataFrame:
    df = _normalize_columns(df.copy())
    missing = REQUIRED - set(df.columns)
    if missing and not allow_partial:
        raise ValueError(f"Missing required columns: {missing}")
    # If partially missing, try to continue with available subset and later raise if critical ones absent
    critical = {"InvoiceDate", "CustomerID"}
    crit_missing = critical - set(df.columns)
    if crit_missing:
        raise ValueError(f"Critical columns missing after normalization: {crit_missing}")

    # Drop rows without customer
    df = df.dropna(subset=["CustomerID"])  # keep only rows with id

    # Parse date
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    # Fill in defaults if some numeric columns missing
    if "Quantity" not in df.columns:
        df["Quantity"] = 1
    if "UnitPrice" not in df.columns:
        # Attempt to derive from TotalPrice if present
        if "TotalPrice" in df.columns and "Quantity" in df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                df["UnitPrice"] = df["TotalPrice"] / df["Quantity"].replace({0: np.nan})
        else:
            df["UnitPrice"] = 0.0

    # Remove invalid rows
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    # TotalPrice
    if "TotalPrice" not in df.columns:
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df.reset_index(drop=True)


def compute_rfm(transactions: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    if snapshot_date is None:
        snapshot_date = transactions["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        transactions.groupby("CustomerID").agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique") if "InvoiceNo" in transactions.columns else ("InvoiceDate", "count"),
            Monetary=("TotalPrice", "sum"),
        )
    )
    return rfm


def customer_aggregates(transactions: pd.DataFrame) -> pd.DataFrame:
    diversity = transactions.groupby("CustomerID")["StockCode"].nunique().rename("ProductDiversity") if "StockCode" in transactions.columns else None
    avg_price = transactions.groupby("CustomerID")["UnitPrice"].mean().rename("AvgUnitPrice")
    frames = [s for s in [diversity, avg_price] if s is not None]
    return pd.concat(frames, axis=1)


def build_customer_feature_table(transactions: pd.DataFrame, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    rfm = compute_rfm(transactions, snapshot_date)
    extra = customer_aggregates(transactions)
    feat = rfm.join(extra, how="left")
    feat["AvgOrderValue"] = feat["Monetary"] / feat["Frequency"].replace({0: np.nan})
    feat["AvgOrderValue"].fillna(0, inplace=True)
    return feat


@dataclass
class ScaledFeatures:
    data: pd.DataFrame
    scaler_name: str
    scaler: object


def scale_features(features: pd.DataFrame, cols: Optional[Iterable[str]] = None, method: str = "standard") -> ScaledFeatures:
    if cols is None:
        cols = [c for c in features.columns if features[c].dtype != object]
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")
    arr = scaler.fit_transform(features[list(cols)])
    scaled_df = pd.DataFrame(arr, columns=[f"{c}_scaled" for c in cols], index=features.index)
    out = features.join(scaled_df)
    return ScaledFeatures(out, method, scaler)


def save_features(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)
    return path


def run_full_pipeline(raw_path: str | Path, out_path: str | Path, scaler_method: str = "standard"):
    raw = load_raw(raw_path)
    cleaned = clean_transactions(raw)
    cust = build_customer_feature_table(cleaned)
    scaled = scale_features(cust, method=scaler_method)
    save_features(scaled.data, out_path)
    return scaled.data


__all__ = [
    "load_raw",
    "clean_transactions",
    "compute_rfm",
    "customer_aggregates",
    "build_customer_feature_table",
    "scale_features",
    "save_features",
    "run_full_pipeline",
    "ScaledFeatures",
]
