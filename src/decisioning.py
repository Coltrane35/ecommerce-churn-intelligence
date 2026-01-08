from __future__ import annotations

import pandas as pd


def add_value_proxy_segments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["value_proxy"] = out["monetary_total"]

    q1 = out["value_proxy"].quantile(0.33)
    q2 = out["value_proxy"].quantile(0.66)

    def seg(v: float) -> str:
        if v <= q1:
            return "Low"
        if v <= q2:
            return "Mid"
        return "High"

    out["value_segment"] = out["value_proxy"].apply(seg)
    return out


def add_risk_segments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["risk_segment"] = pd.cut(out["churn_score"], bins=[0.0, 0.33, 0.66, 1.0], labels=["Low", "Mid", "High"], include_lowest=True)
    return out


def recommend_action(value_segment: str, risk_segment: str) -> str:
    if value_segment == "High" and risk_segment == "High":
        return "Priority retention: personal offer / call"
    if value_segment in {"High", "Mid"} and risk_segment == "High":
        return "Retention campaign: targeted discount"
    if value_segment == "High" and risk_segment == "Mid":
        return "Monitor + soft engagement"
    if risk_segment == "Low":
        return "No action / regular comms"
    return "Monitor"


def build_priority_table(features: pd.DataFrame, scores: pd.DataFrame, id_col: str = "CustomerID") -> pd.DataFrame:
    out = features.merge(scores[[id_col, "churn_score"]], on=id_col, how="left")
    out = add_value_proxy_segments(out)
    out = add_risk_segments(out)

    out["recommended_action"] = [
        recommend_action(v, r) for v, r in zip(out["value_segment"], out["risk_segment"])
    ]

    v = out["value_proxy"].astype(float)
    v_norm = (v - v.min()) / (v.max() - v.min() + 1e-9)
    out["priority_score"] = (v_norm * out["churn_score"]).astype(float)

    out = out.sort_values("priority_score", ascending=False)

    cols = [
        id_col, "churn_score", "value_proxy", "value_segment", "risk_segment",
        "priority_score", "recommended_action",
        "recency_days", "frequency_orders", "monetary_total"
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols]
