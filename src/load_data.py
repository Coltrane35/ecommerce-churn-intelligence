from __future__ import annotations

import pandas as pd

# Expected columns for Kaggle "Online Retail" (tunguz)
REQUIRED_COLS = ["InvoiceNo", "InvoiceDate", "CustomerID", "Quantity", "UnitPrice"]


def load_transactions(path: str) -> pd.DataFrame:
    """
    Load Online Retail transactions and perform minimal cleaning for churn/RFM pipeline.

    Key behaviors:
    - Handles non-UTF8 encodings (latin-1/ISO-8859-1) commonly used by this dataset
    - Parses InvoiceDate
    - Drops rows with missing CustomerID/InvoiceDate
    - Converts Quantity/UnitPrice to numeric and removes invalid/negative rows (returns/cancellations)
    - Adds TotalPrice = Quantity * UnitPrice
    - Normalizes CustomerID to string (stable id)
    """
    # 1) Read CSV with robust encoding fallback
    # Most often this dataset works with latin-1.
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        # fallback for this dataset
        df = pd.read_csv(path, encoding="latin-1")

    # 2) Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # 3) Validate required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available columns: {df.columns.tolist()}")

    # 4) Parse dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # 5) Drop rows without essential fields
    df = df.dropna(subset=["InvoiceDate", "CustomerID"])

    # 6) Numeric conversions
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df.dropna(subset=["Quantity", "UnitPrice"])

    # 7) Remove returns/cancellations and invalid values
    # Quantity <= 0 or UnitPrice <= 0 are typically returns/corrections
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()

    # 8) Add total line value
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # 9) Normalize CustomerID
    # Often CustomerID is float in CSV (e.g., 17850.0). Convert to int->str for stable id.
    # Using nullable Int64 first to avoid crashes if any weird values slipped through.
    df["CustomerID"] = df["CustomerID"].astype("int64").astype(str)

    # Optional: ensure InvoiceNo is string (sometimes mixed types)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)

    return df
