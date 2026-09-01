from __future__ import annotations

import pandas as pd


def build_executive_summary(df: pd.DataFrame) -> dict[str, object]:
    """
    Build portfolio-level business summary metrics.

    Expected columns:
    - CustomerID
    - segment
    - churn_score
    - predicted_clv
    - expected_profit
    - estimated_roi
    """

    if df.empty:
        return {
            "customers": 0,
            "high_risk_customers": 0,
            "high_value_high_risk_customers": 0,
            "total_predicted_clv": 0.0,
            "total_expected_profit": 0.0,
            "average_roi": 0.0,
            "top_segment": "N/A",
            "summary_text": "No customer data is available.",
        }

    customers = int(df["CustomerID"].nunique())

    high_risk_customers = int(
        (df["churn_score"] >= 0.70).sum()
    )

    high_value_high_risk_customers = int(
        (df["segment"] == "HIGH_VALUE_HIGH_RISK").sum()
    )

    total_predicted_clv = float(
        df["predicted_clv"].sum()
    )

    total_expected_profit = float(
        df["expected_profit"].sum()
    )

    average_roi = float(
        df["estimated_roi"].mean()
    )

    segment_profit = (
        df.groupby("segment", dropna=False)["expected_profit"]
        .sum()
        .sort_values(ascending=False)
    )

    top_segment = (
        str(segment_profit.index[0])
        if not segment_profit.empty
        else "N/A"
    )

    high_risk_share = (
        high_risk_customers / customers
        if customers > 0
        else 0.0
    )

    summary_text = (
        f"The portfolio contains {customers:,} customers. "
        f"{high_risk_customers:,} customers "
        f"({high_risk_share:.1%}) have a churn score of at least 0.70. "
        f"{high_value_high_risk_customers:,} customers belong to the "
        f"HIGH_VALUE_HIGH_RISK segment. "
        f"Total predicted CLV is {total_predicted_clv:,.0f}, "
        f"while estimated total retention profit is "
        f"{total_expected_profit:,.0f}. "
        f"The segment generating the highest expected profit is "
        f"{top_segment}. "
        f"Average estimated ROI across the portfolio is "
        f"{average_roi:.2f}."
    )

    return {
        "customers": customers,
        "high_risk_customers": high_risk_customers,
        "high_value_high_risk_customers": (
            high_value_high_risk_customers
        ),
        "total_predicted_clv": total_predicted_clv,
        "total_expected_profit": total_expected_profit,
        "average_roi": average_roi,
        "top_segment": top_segment,
        "summary_text": summary_text,
    }