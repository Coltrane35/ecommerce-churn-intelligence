from __future__ import annotations

import pandas as pd


def customer_last_purchase(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("CustomerID", as_index=False)["InvoiceDate"]
        .max()
        .rename(columns={"InvoiceDate": "last_purchase_date"})
    )


def add_churn_label(
    cust_last: pd.DataFrame,
    reference_date: pd.Timestamp,
    window_days: int,
) -> pd.DataFrame:
    out = cust_last.copy()
    out["days_since_last_purchase"] = (reference_date - out["last_purchase_date"]).dt.days
    out["churn"] = (out["days_since_last_purchase"] > window_days).astype(int)
    return out


def make_snapshot_churn_dataset(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    prediction_window_days: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split transactions into:
    - history: data available up to snapshot_date
    - target: churn label based on purchases after snapshot_date

    A customer is marked as churned if they do not purchase again
    within the prediction window after the snapshot date.
    """
    history = df[df["InvoiceDate"] <= snapshot_date].copy()

    prediction_end_date = snapshot_date + pd.Timedelta(days=prediction_window_days)
    future = df[
        (df["InvoiceDate"] > snapshot_date)
        & (df["InvoiceDate"] <= prediction_end_date)
    ].copy()

    historical_customers = history["CustomerID"].dropna().unique()
    active_future_customers = future["CustomerID"].dropna().unique()

    target = pd.DataFrame({"CustomerID": historical_customers})
    target["churn"] = (~target["CustomerID"].isin(active_future_customers)).astype(int)

    return history, target