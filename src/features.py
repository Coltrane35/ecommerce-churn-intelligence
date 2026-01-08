from __future__ import annotations

import pandas as pd


def _within_window(df: pd.DataFrame, reference_date: pd.Timestamp, window_days: int) -> pd.DataFrame:
    start = reference_date - pd.Timedelta(days=window_days)
    return df[(df["InvoiceDate"] > start) & (df["InvoiceDate"] <= reference_date)]


def build_rfm_features(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    agg = df.groupby("CustomerID").agg(
        last_purchase=("InvoiceDate", "max"),
        frequency_orders=("InvoiceNo", "nunique"),
        monetary_total=("TotalPrice", "sum"),
        monetary_mean=("TotalPrice", "mean"),
        monetary_median=("TotalPrice", "median"),
    ).reset_index()

    agg["recency_days"] = (reference_date - agg["last_purchase"]).dt.days
    return agg.drop(columns=["last_purchase"])


def build_window_features(df: pd.DataFrame, reference_date: pd.Timestamp, window_days: int) -> pd.DataFrame:
    wdf = _within_window(df, reference_date, window_days)
    out = wdf.groupby("CustomerID").agg(
        **{
            f"orders_last_{window_days}d": ("InvoiceNo", "nunique"),
            f"spend_last_{window_days}d": ("TotalPrice", "sum"),
        }
    ).reset_index()
    return out


def merge_customer_features(rfm: pd.DataFrame, w30: pd.DataFrame, w60: pd.DataFrame, w90: pd.DataFrame) -> pd.DataFrame:
    out = rfm.merge(w30, on="CustomerID", how="left") \
             .merge(w60, on="CustomerID", how="left") \
             .merge(w90, on="CustomerID", how="left")

    for col in out.columns:
        if col.startswith("orders_last_") or col.startswith("spend_last_"):
            out[col] = out[col].fillna(0)

    out["trend_orders_30_vs_90"] = (out["orders_last_30d"] + 1) / (out["orders_last_90d"] + 1)
    out["trend_spend_30_vs_90"] = (out["spend_last_30d"] + 1) / (out["spend_last_90d"] + 1)

    return out
