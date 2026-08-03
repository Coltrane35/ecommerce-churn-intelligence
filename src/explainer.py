from __future__ import annotations

import pandas as pd


def generate_business_explanations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate human-readable business explanations for each customer.

    This is a template-based explanation layer.
    It is designed to be LLM-ready in future iterations.
    """

    out = df.copy()

    def build_explanation(row: pd.Series) -> str:
        customer_id = row.get("CustomerID", "Unknown")
        segment = row.get("segment", "unknown segment")
        action = row.get("recommended_action", "no_action")
        channel = row.get("action_channel", "none")
        timing = row.get("action_timing", "none")
        reason = row.get("action_reason", "No reason available")

        churn_score = row.get("churn_score", 0)
        predicted_clv = row.get("predicted_clv", 0)
        expected_profit = row.get("expected_profit", 0)
        estimated_roi = row.get("estimated_roi", 0)

        return (
            f"Customer {int(customer_id)} is classified as {segment}. "
            f"Reason: {reason}. "
            f"Recommended action: {action} via {channel} within {timing}. "
            f"Churn risk score is {churn_score:.2f}, predicted CLV is {predicted_clv:.0f}. "
            f"Expected profit is {expected_profit:.0f} with estimated ROI of {estimated_roi:.2f}."
        )

    out["business_explanation"] = out.apply(build_explanation, axis=1)

    return out