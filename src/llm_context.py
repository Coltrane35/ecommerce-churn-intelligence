from __future__ import annotations

import json

import pandas as pd


def build_customer_llm_context(
    row: pd.Series,
) -> dict[str, object]:
    """
    Build structured customer context that can later
    be passed to an LLM.

    The function itself does not call any external API.
    """

    return {
        "customer": {
            "customer_id": int(
                row.get(
                    "CustomerID",
                    0,
                )
            ),
            "segment": row.get(
                "segment",
                "unknown",
            ),
        },
        "churn_prediction": {
            "churn_score": round(
                float(
                    row.get(
                        "churn_score",
                        0.0,
                    )
                ),
                4,
            ),
            "risk_drivers": row.get(
                "top_risk_drivers",
                "",
            ),
            "protective_drivers": row.get(
                "top_protective_drivers",
                "",
            ),
            "model_explanation": row.get(
                "model_explanation",
                "",
            ),
        },
        "customer_value": {
            "predicted_clv": round(
                float(
                    row.get(
                        "predicted_clv",
                        0.0,
                    )
                ),
                2,
            ),
            "priority_score": round(
                float(
                    row.get(
                        "priority_score",
                        0.0,
                    )
                ),
                2,
            ),
        },
        "recommended_action": {
            "action": row.get(
                "recommended_action",
                "no_action",
            ),
            "channel": row.get(
                "action_channel",
                "none",
            ),
            "timing": row.get(
                "action_timing",
                "none",
            ),
            "reason": row.get(
                "action_reason",
                "",
            ),
        },
        "business_case": {
            "campaign_cost": round(
                float(
                    row.get(
                        "campaign_cost",
                        0.0,
                    )
                ),
                2,
            ),
            "expected_profit": round(
                float(
                    row.get(
                        "expected_profit",
                        0.0,
                    )
                ),
                2,
            ),
            "estimated_roi": round(
                float(
                    row.get(
                        "estimated_roi",
                        0.0,
                    )
                ),
                2,
            ),
        },
    }


def build_llm_prompt(
    row: pd.Series,
) -> str:
    """
    Build a prompt ready to be passed to an LLM.
    """

    context = build_customer_llm_context(
        row
    )

    context_json = json.dumps(
        context,
        indent=2,
        ensure_ascii=False,
    )

    return (
        "You are a customer retention analyst.\n\n"
        "Analyze the following customer information:\n\n"
        f"{context_json}\n\n"
        "Provide a short business explanation containing:\n"
        "1. Why the customer may churn.\n"
        "2. Why the recommended action is appropriate.\n"
        "3. What business value may be protected.\n"
        "4. The most important next step for the retention team.\n\n"
        "Use concise, professional business language. "
        "Do not invent information that is not present "
        "in the customer data."
    )


def add_llm_context(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add LLM-ready context and prompt columns
    to the customer decision table.
    """

    out = df.copy()

    out["llm_context"] = out.apply(
        lambda row: json.dumps(
            build_customer_llm_context(
                row
            ),
            ensure_ascii=False,
        ),
        axis=1,
    )

    out["llm_prompt"] = out.apply(
        build_llm_prompt,
        axis=1,
    )

    return out