from __future__ import annotations

import pandas as pd


def estimate_campaign_roi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate campaign ROI for each customer based on:
    - recommended action
    - predicted CLV
    - churn score

    This is a simplified business simulation layer.
    """

    out = df.copy()

    def map_campaign_cost(action: str) -> float:
        if action == "offer_discount":
            return 200.0
        if action == "personal_offer":
            return 120.0
        if action == "loyalty_program":
            return 60.0
        if action == "email_campaign":
            return 20.0
        return 0.0

    out["campaign_cost"] = out["recommended_action"].apply(map_campaign_cost)

    out["expected_retention_value"] = (
        out["predicted_clv"] * out["churn_score"]
    )

    out["expected_profit"] = (
        out["expected_retention_value"] - out["campaign_cost"]
    )

    out["estimated_roi"] = 0.0

    mask = out["campaign_cost"] > 0

    out.loc[mask, "estimated_roi"] = (
        out.loc[mask, "expected_profit"] / out.loc[mask, "campaign_cost"]
    )

    return out