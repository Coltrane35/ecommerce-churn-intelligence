from __future__ import annotations

import pandas as pd


def assign_retention_action(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def map_action(segment: str) -> str:
        if segment == "HIGH_VALUE_HIGH_RISK":
            return "offer_discount"
        if segment == "HIGH_VALUE_MEDIUM_RISK":
            return "personal_offer"
        if segment == "HIGH_VALUE_LOW_RISK":
            return "loyalty_program"
        if segment == "MEDIUM_VALUE_HIGH_RISK":
            return "email_campaign"
        if segment == "LOW_VALUE_HIGH_RISK":
            return "low_priority"
        return "no_action"

    def map_reason(segment: str) -> str:
        if segment == "HIGH_VALUE_HIGH_RISK":
            return "High value and high churn risk"
        if segment == "HIGH_VALUE_MEDIUM_RISK":
            return "High value with moderate churn risk"
        if segment == "HIGH_VALUE_LOW_RISK":
            return "High value and low churn risk"
        if segment == "MEDIUM_VALUE_HIGH_RISK":
            return "Medium value but high churn risk"
        if segment == "LOW_VALUE_HIGH_RISK":
            return "Low value and high churn risk"
        return "Low priority segment"

    out["recommended_action"] = out["segment"].apply(map_action)
    out["action_reason"] = out["segment"].apply(map_reason)

    return out