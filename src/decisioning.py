from __future__ import annotations

import pandas as pd


def _safe_qcut(
    series: pd.Series,
    labels: list[str],
) -> pd.Series:
    try:
        return pd.qcut(series, q=len(labels), labels=labels, duplicates="drop")
    except ValueError:
        return pd.Series(["MEDIUM"] * len(series), index=series.index)


def build_priority_table(
    features: pd.DataFrame,
    scores: pd.DataFrame,
    id_col: str,
) -> pd.DataFrame:
    out = features.merge(scores, on=id_col, how="left")

    out["churn_score"] = out["churn_score"].fillna(0)

    if "predicted_clv" in out.columns:
        out["value_score"] = out["predicted_clv"].fillna(0)
    else:
        out["value_score"] = out["monetary_total"].fillna(0)

    out["churn_risk"] = out["churn_score"]
    out["priority_score"] = out["churn_risk"] * out["value_score"]

    out["risk_segment"] = _safe_qcut(
        out["churn_risk"],
        labels=["LOW", "MEDIUM", "HIGH"],
    )

    out["value_segment"] = _safe_qcut(
        out["value_score"],
        labels=["LOW", "MEDIUM", "HIGH"],
    )

    out["segment"] = (
        out["value_segment"].astype(str)
        + "_VALUE_"
        + out["risk_segment"].astype(str)
        + "_RISK"
    )

    out = out.sort_values("priority_score", ascending=False)

    return out