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
