from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(
    path_in: str,
    path_out: str,
    top_n: int = 10,
) -> None:
    df = pd.read_csv(path_in)

    if df.empty:
        print("Feature importance file is empty.")
        return

    df = df.head(top_n)

    plt.figure(figsize=(8, 5))
    plt.barh(df["feature"], df["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path_out)
    plt.close()

    print(f"Saved feature importance plot: {path_out}")


def plot_value_risk_matrix(
    path_in: str,
    path_out: str,
) -> None:
    df = pd.read_csv(path_in)

    required_columns = {"value_score", "churn_score"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns in priority table: {missing_columns}")

    if df.empty:
        print("Priority table is empty.")
        return

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df["value_score"],
        df["churn_score"],
        alpha=0.4,
    )

    plt.axvline(df["value_score"].median(), linestyle="--")
    plt.axhline(df["churn_score"].median(), linestyle="--")

    plt.xlabel("Customer Value (Predicted CLV)")
    plt.ylabel("Churn Risk")
    plt.title("Customer Value vs Churn Risk")

    plt.tight_layout()
    plt.savefig(path_out)
    plt.close()

    print(f"Saved value-risk matrix: {path_out}")