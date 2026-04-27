from __future__ import annotations

import pandas as pd


def assign_retention_action(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign recommended retention actions based on customer segment.
    """

    out = df.copy()

    def map_action(segment: str) -> str:
        if segment == "HIGH_VALUE_HIGH_RISK":
            return "offer_discount"

        elif segment == "HIGH_VALUE_MEDIUM_RISK":
            return "personal_offer"

        elif segment == "HIGH_VALUE_LOW_RISK":
            return "loyalty_program"

        elif segment == "MEDIUM_VALUE_HIGH_RISK":
            return "email_campaign"

        elif segment == "MEDIUM_VALUE_MEDIUM_RISK":
            return "engagement_campaign"

        elif segment == "LOW_VALUE_HIGH_RISK":
            return "low_priority"

        else:
            return "no_action"

    out["recommended_action"] = out["segment"].apply(map_action)

    return out