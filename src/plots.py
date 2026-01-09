from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Input produced by your pipeline
INPUT_PATH = Path("outputs/churn_priority_table.csv")

# Where to save figures
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def value_risk_matrix(df: pd.DataFrame, out_path: Path) -> None:
    """
    Create a Value × Risk matrix (heatmap) to support retention decisioning.
    Rows: value_segment (Low/Mid/High)
    Cols: risk_segment (Low/Mid/High)
    Cell value: number of customers (count)
    """
    needed = {"CustomerID", "value_segment", "risk_segment"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in input: {missing}. Got columns: {df.columns.tolist()}")

    order = ["Low", "Mid", "High"]
    dfx = df.copy()
    dfx["value_segment"] = pd.Categorical(dfx["value_segment"], categories=order, ordered=True)
    dfx["risk_segment"] = pd.Categorical(dfx["risk_segment"], categories=order, ordered=True)

    pivot = (
        dfx.pivot_table(
            index="value_segment",
            columns="risk_segment",
            values="CustomerID",
            aggfunc="count",
            fill_value=0,
            observed=False,  # keeps current behavior and silences future warning intent
        )
        .reindex(index=order, columns=order)
    )

    plt.figure(figsize=(7, 5))
    ax = sns.heatmap(pivot, annot=True, fmt="d")
    ax.set_title("Value × Risk Matrix (Customer Count)")
    ax.set_xlabel("Churn Risk Segment")
    ax.set_ylabel("Customer Value Segment")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def top_retention_targets(df: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    """
    Plot Top N customers by priority_score (Value × Risk) — ready for campaign activation.
    """
    needed = {"CustomerID", "priority_score", "value_segment", "risk_segment", "churn_score"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in input: {missing}. Got columns: {df.columns.tolist()}")

    dfx = df.copy()

    # Ensure numeric
    dfx["priority_score"] = pd.to_numeric(dfx["priority_score"], errors="coerce")
    dfx["churn_score"] = pd.to_numeric(dfx["churn_score"], errors="coerce")
    dfx = dfx.dropna(subset=["priority_score", "churn_score"])

    top = dfx.sort_values("priority_score", ascending=False).head(top_n).copy()

    # Make readable labels: CustomerID | Value/Risk
    top["label"] = top.apply(
        lambda r: f'{r["CustomerID"]}  ({r["value_segment"]}/{r["risk_segment"]})',
        axis=1,
    )

    # Plot (horizontal bar chart)
    plt.figure(figsize=(10, 7))
    plt.barh(top["label"][::-1], top["priority_score"][::-1])
    plt.title(f"Top {top_n} Retention Targets (by Priority Score)")
    plt.xlabel("Priority Score (Value × Risk)")
    plt.ylabel("Customer (Value/Risk)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Run the pipeline first: python -m src.run_pipeline"
        )

    df = pd.read_csv(INPUT_PATH)

    out_matrix = FIG_DIR / "value_risk_matrix.png"
    value_risk_matrix(df, out_matrix)
    print(f"Saved: {out_matrix}")

    out_top = FIG_DIR / "top_retention_targets.png"
    top_retention_targets(df, out_top, top_n=20)
    print(f"Saved: {out_top}")


if __name__ == "__main__":
    main()
