from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def build_clv_dataset(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    horizon_days: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = df[df["InvoiceDate"] <= snapshot_date].copy()

    future_end = snapshot_date + pd.Timedelta(days=horizon_days)
    future = df[
        (df["InvoiceDate"] > snapshot_date)
        & (df["InvoiceDate"] <= future_end)
    ].copy()

    future_clv = (
        future.groupby("CustomerID")["TotalPrice"]
        .sum()
        .reset_index()
        .rename(columns={"TotalPrice": "future_clv"})
    )

    return history, future_clv


def train_clv_model(
    features: pd.DataFrame,
    future_clv: pd.DataFrame,
) -> LinearRegression:
    df = features.merge(future_clv, on="CustomerID", how="left")
    df["future_clv"] = df["future_clv"].fillna(0)

    X = df.drop(columns=["CustomerID", "future_clv"])
    y = df["future_clv"]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def predict_clv(
    model: LinearRegression,
    features: pd.DataFrame,
) -> pd.DataFrame:
    X = features.drop(columns=["CustomerID"])
    predictions = model.predict(X)

    return pd.DataFrame(
        {
            "CustomerID": features["CustomerID"],
            "predicted_clv": predictions.clip(min=0),
        }
    )