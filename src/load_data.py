from __future__ import annotations

import pandas as pd


def load_transactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="ISO-8859-1")

    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        format="%m/%d/%y %H:%M",
        errors="coerce",
    )

    df = df.dropna(subset=["InvoiceDate"])

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    df = df.sort_values("InvoiceDate")

    return df