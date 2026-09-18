from __future__ import annotations

import json

import pandas as pd


def build_customer_llm_context(
    row: pd.Series,
) -> dict[str, object]:
    """
    Build structured, LLM-ready customer context.

    The context contains only information already produced
    by the analytical pipeline. It does not add assumptions
    about feature direction, causality or currency.
    """

    return {
        "customer": {
            "customer_id": int(
                row.get("CustomerID", 0)
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
    Build a grounded prompt for the retention LLM.

    The prompt explicitly prevents unsupported interpretation
    of model contributions and business values.
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
        "1. Why the model considers the customer at risk of churn.\n"
        "2. Why the recommended action is appropriate.\n"
        "3. What business value may be protected.\n"
        "4. The most important next step for the retention team.\n\n"
        "Important interpretation rules:\n"
        "- Risk drivers are features that contributed positively "
        "to the model's churn prediction.\n"
        "- Protective drivers are features that contributed "
        "negatively to the model's churn prediction.\n"
        "- Do not assume that a feature is high, low, increasing, "
        "decreasing, strong or weak unless its actual value or trend "
        "is explicitly provided.\n"
        "- Do not infer causal relationships from model contributions.\n"
        "- Do not invent customer behavior that is not present "
        "in the supplied data.\n"
        "- Do not add a currency symbol or currency name unless "
        "a currency is explicitly provided in the context.\n"
        "- Treat churn score, CLV, expected profit and ROI as "
        "model-derived estimates, not guaranteed outcomes.\n"
        "- Use only the information supplied in the context.\n\n"
        "Use concise, professional business language."
    )


def add_llm_context(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add structured LLM context and prompt columns
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