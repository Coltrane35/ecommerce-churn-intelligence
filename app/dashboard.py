from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.summary import build_executive_summary


DATA_PATH = Path("outputs/churn_priority_table.csv")
FEATURE_IMPORTANCE_PATH = Path("outputs/feature_importance.csv")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}. Run the pipeline first."
        )

    return pd.read_csv(path)


def load_feature_importance(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def assign_risk_level(churn_score: float) -> str:
    """
    Convert churn probability into a simple business risk level.
    """

    if churn_score >= 0.70:
        return "High Risk"

    if churn_score >= 0.40:
        return "Medium Risk"

    return "Low Risk"


def build_risk_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count customers by churn-risk level.
    """

    out = df.copy()

    out["risk_level"] = out["churn_score"].apply(
        assign_risk_level
    )

    risk_order = [
        "Low Risk",
        "Medium Risk",
        "High Risk",
    ]

    counts = (
        out["risk_level"]
        .value_counts()
        .reindex(
            risk_order,
            fill_value=0,
        )
        .rename_axis("risk_level")
        .reset_index(name="customers")
    )

    return counts


def build_churn_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create churn-score buckets for portfolio analysis.
    """

    bins = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.01,
    ]

    labels = [
        "0.0–0.1",
        "0.1–0.2",
        "0.2–0.3",
        "0.3–0.4",
        "0.4–0.5",
        "0.5–0.6",
        "0.6–0.7",
        "0.7–0.8",
        "0.8–0.9",
        "0.9–1.0",
    ]

    out = df.copy()

    out["churn_bucket"] = pd.cut(
        out["churn_score"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    distribution = (
        out["churn_bucket"]
        .value_counts(sort=False)
        .rename_axis("churn_score")
        .reset_index(name="customers")
    )

    return distribution


def build_driver_frequency(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Count how often each local model driver appears.
    """

    if column not in df.columns:
        return pd.DataFrame(
            columns=[
                "feature",
                "customers",
            ]
        )

    drivers = (
        df[column]
        .dropna()
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
    )

    drivers = drivers[
        drivers.ne("")
    ]

    result = (
        drivers
        .value_counts()
        .rename_axis("feature")
        .reset_index(name="customers")
    )

    return result


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Retention Decision Engine",
    layout="wide",
)

st.title("🚀 Customer Retention Decision Engine")

st.caption(
    "Churn prediction, CLV, next-best-action recommendations, "
    "ROI analysis and business explainability."
)


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

try:
    df = load_data(DATA_PATH)
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()


feature_importance_df = load_feature_importance(
    FEATURE_IMPORTANCE_PATH
)


# -------------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------------

st.sidebar.header("Filters")

segment_options = sorted(
    df["segment"]
    .dropna()
    .unique()
    .tolist()
)

action_options = sorted(
    df["recommended_action"]
    .dropna()
    .unique()
    .tolist()
)

segment_filter = st.sidebar.multiselect(
    "Segment",
    options=segment_options,
    default=segment_options,
)

action_filter = st.sidebar.multiselect(
    "Action",
    options=action_options,
    default=action_options,
)

minimum_roi = st.sidebar.number_input(
    "Minimum estimated ROI",
    min_value=0.0,
    value=0.0,
    step=1.0,
)

filtered_df = df[
    df["segment"].isin(segment_filter)
    & df["recommended_action"].isin(action_filter)
    & (df["estimated_roi"] >= minimum_roi)
].copy()


if filtered_df.empty:
    st.warning(
        "No customers match the selected filters."
    )
    st.stop()


# -------------------------------------------------------------------
# Executive Summary
# -------------------------------------------------------------------

summary = build_executive_summary(
    filtered_df
)

st.subheader("🧠 Executive Summary")

(
    summary_col1,
    summary_col2,
    summary_col3,
    summary_col4,
) = st.columns(4)

summary_col1.metric(
    "High Risk Customers",
    f"{summary['high_risk_customers']:,}",
)

summary_col2.metric(
    "High Value / High Risk",
    f"{summary['high_value_high_risk_customers']:,}",
)

summary_col3.metric(
    "Average ROI",
    f"{summary['average_roi']:.2f}",
)

summary_col4.metric(
    "Top Profit Segment",
    str(summary["top_segment"]),
)

st.info(
    str(summary["summary_text"])
)


# -------------------------------------------------------------------
# Portfolio Overview
# -------------------------------------------------------------------

st.subheader("📊 Portfolio Overview")

col1, col2, col3, col4 = st.columns(4)

customer_count = len(
    filtered_df
)

average_churn = (
    filtered_df["churn_score"].mean()
)

total_clv = (
    filtered_df["predicted_clv"].sum()
)

total_profit = (
    filtered_df["expected_profit"].sum()
)

col1.metric(
    "Customers",
    f"{customer_count:,}",
)

col2.metric(
    "Avg Churn Score",
    f"{average_churn:.3f}",
)

col3.metric(
    "Total Predicted CLV",
    f"{total_clv:,.0f}",
)

col4.metric(
    "Total Expected Profit",
    f"{total_profit:,.0f}",
)


# -------------------------------------------------------------------
# Portfolio Health
# -------------------------------------------------------------------

st.divider()

st.subheader("❤️ Portfolio Health")

st.caption(
    "Portfolio-level view of churn risk and the model factors "
    "most frequently associated with customer risk."
)


# -------------------------------------------------------------------
# Risk KPIs
# -------------------------------------------------------------------

high_risk_count = int(
    (
        filtered_df["churn_score"]
        >= 0.70
    ).sum()
)

medium_risk_count = int(
    (
        (
            filtered_df["churn_score"]
            >= 0.40
        )
        & (
            filtered_df["churn_score"]
            < 0.70
        )
    ).sum()
)

low_risk_count = int(
    (
        filtered_df["churn_score"]
        < 0.40
    ).sum()
)

high_risk_clv = float(
    filtered_df.loc[
        filtered_df["churn_score"] >= 0.70,
        "predicted_clv",
    ].sum()
)

(
    health_col1,
    health_col2,
    health_col3,
    health_col4,
) = st.columns(4)

health_col1.metric(
    "High Risk",
    f"{high_risk_count:,}",
)

health_col2.metric(
    "Medium Risk",
    f"{medium_risk_count:,}",
)

health_col3.metric(
    "Low Risk",
    f"{low_risk_count:,}",
)

health_col4.metric(
    "CLV at High Risk",
    f"{high_risk_clv:,.0f}",
)


# -------------------------------------------------------------------
# Risk Distribution
# -------------------------------------------------------------------

risk_chart_col1, risk_chart_col2 = st.columns(2)

with risk_chart_col1:

    st.markdown(
        "#### Customer Risk Levels"
    )

    risk_distribution = (
        build_risk_distribution(
            filtered_df
        )
    )

    st.bar_chart(
        risk_distribution.set_index(
            "risk_level"
        )
    )


with risk_chart_col2:

    st.markdown(
        "#### Churn Score Distribution"
    )

    churn_distribution = (
        build_churn_distribution(
            filtered_df
        )
    )

    st.bar_chart(
        churn_distribution.set_index(
            "churn_score"
        )
    )


# -------------------------------------------------------------------
# Model Insights
# -------------------------------------------------------------------

st.subheader("🧬 Model Insights")

model_col1, model_col2 = st.columns(2)


# -------------------------------------------------------------------
# Most frequent customer-level risk drivers
# -------------------------------------------------------------------

with model_col1:

    st.markdown(
        "#### Most Frequent Risk Drivers"
    )

    risk_driver_frequency = (
        build_driver_frequency(
            filtered_df,
            "top_risk_drivers",
        )
    )

    if risk_driver_frequency.empty:

        st.info(
            "Risk-driver data is not available."
        )

    else:

        top_risk_driver_frequency = (
            risk_driver_frequency
            .head(10)
        )

        st.bar_chart(
            top_risk_driver_frequency
            .set_index("feature")
        )

        st.dataframe(
            top_risk_driver_frequency,
            width="stretch",
            hide_index=True,
        )


# -------------------------------------------------------------------
# Global feature importance
# -------------------------------------------------------------------

with model_col2:

    st.markdown(
        "#### Global Model Feature Importance"
    )

    if feature_importance_df.empty:

        st.info(
            "Feature importance file is not available."
        )

    else:

        importance = (
            feature_importance_df.copy()
        )

        importance[
            "absolute_importance"
        ] = (
            importance["importance"]
            .abs()
        )

        importance = (
            importance
            .sort_values(
                "absolute_importance",
                ascending=False,
            )
            .head(10)
        )

        st.bar_chart(
            importance[
                [
                    "feature",
                    "absolute_importance",
                ]
            ]
            .set_index("feature")
        )

        st.dataframe(
            importance[
                [
                    "feature",
                    "importance",
                    "absolute_importance",
                ]
            ],
            width="stretch",
            hide_index=True,
        )


# -------------------------------------------------------------------
# Protective drivers
# -------------------------------------------------------------------

with st.expander(
    "🛡️ Most Frequent Protective Drivers"
):

    protective_driver_frequency = (
        build_driver_frequency(
            filtered_df,
            "top_protective_drivers",
        )
    )

    if protective_driver_frequency.empty:

        st.info(
            "Protective-driver data is not available."
        )

    else:

        top_protective_drivers = (
            protective_driver_frequency
            .head(15)
        )

        st.dataframe(
            top_protective_drivers,
            width="stretch",
            hide_index=True,
        )


# -------------------------------------------------------------------
# Top retention opportunities
# -------------------------------------------------------------------

st.divider()

st.subheader(
    "🎯 Top Retention Opportunities"
)

top_customers = (
    filtered_df
    .sort_values(
        "priority_score",
        ascending=False,
    )
    .head(20)
)

st.dataframe(
    top_customers[
        [
            "CustomerID",
            "segment",
            "recommended_action",
            "action_channel",
            "action_timing",
            "action_reason",
            "churn_score",
            "predicted_clv",
            "priority_score",
            "expected_profit",
            "estimated_roi",
        ]
    ],
    width="stretch",
    hide_index=True,
)


# -------------------------------------------------------------------
# Segment Distribution
# -------------------------------------------------------------------

st.subheader(
    "📈 Segment Distribution"
)

segment_counts = (
    filtered_df["segment"]
    .value_counts()
    .rename_axis("segment")
    .reset_index(name="count")
)

st.bar_chart(
    segment_counts.set_index(
        "segment"
    )
)


# -------------------------------------------------------------------
# Highest ROI Customers
# -------------------------------------------------------------------

st.subheader(
    "💰 Highest ROI Customers"
)

roi_df = (
    filtered_df
    .sort_values(
        "estimated_roi",
        ascending=False,
    )
    .head(20)
)

st.dataframe(
    roi_df[
        [
            "CustomerID",
            "segment",
            "recommended_action",
            "action_channel",
            "action_timing",
            "campaign_cost",
            "expected_profit",
            "estimated_roi",
        ]
    ],
    width="stretch",
    hide_index=True,
)


# -------------------------------------------------------------------
# Customer Explorer
# -------------------------------------------------------------------

st.divider()

st.subheader(
    "🔍 Customer Explorer"
)

customer_options = sorted(
    filtered_df["CustomerID"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

selected_customer_id = st.selectbox(
    "Select CustomerID",
    options=customer_options,
)

customer_row = filtered_df[
    filtered_df["CustomerID"].astype(int)
    == selected_customer_id
].iloc[0]


st.markdown(
    f"### Customer {selected_customer_id}"
)


# -------------------------------------------------------------------
# Customer Metrics
# -------------------------------------------------------------------

metric1, metric2, metric3, metric4 = (
    st.columns(4)
)

metric1.metric(
    "Churn Score",
    f"{customer_row['churn_score']:.3f}",
)

metric2.metric(
    "Predicted CLV",
    f"{customer_row['predicted_clv']:,.0f}",
)

metric3.metric(
    "Expected Profit",
    f"{customer_row['expected_profit']:,.0f}",
)

metric4.metric(
    "Estimated ROI",
    f"{customer_row['estimated_roi']:.2f}",
)


# -------------------------------------------------------------------
# Customer Details
# -------------------------------------------------------------------

details_left, details_right = (
    st.columns(2)
)

with details_left:

    st.markdown(
        "#### Customer Classification"
    )

    st.write(
        f"**Segment:** "
        f"{customer_row['segment']}"
    )

    st.write(
        f"**Priority score:** "
        f"{customer_row['priority_score']:,.2f}"
    )

    st.write(
        f"**Action reason:** "
        f"{customer_row['action_reason']}"
    )


with details_right:

    st.markdown(
        "#### Recommended Action"
    )

    st.write(
        f"**Action:** "
        f"{customer_row['recommended_action']}"
    )

    st.write(
        f"**Channel:** "
        f"{customer_row['action_channel']}"
    )

    st.write(
        f"**Timing:** "
        f"{customer_row['action_timing']}"
    )

    st.write(
        f"**Campaign cost:** "
        f"{customer_row['campaign_cost']:,.0f}"
    )


# -------------------------------------------------------------------
# Model Explanation
# -------------------------------------------------------------------

st.markdown(
    "#### 🔬 Model Explanation"
)

model_explanation = customer_row.get(
    "model_explanation",
    "No model explanation available.",
)

st.warning(
    model_explanation
)

driver_col1, driver_col2 = (
    st.columns(2)
)

with driver_col1:

    st.markdown(
        "##### Risk Drivers"
    )

    st.write(
        customer_row.get(
            "top_risk_drivers",
            "No risk drivers available.",
        )
    )


with driver_col2:

    st.markdown(
        "##### Protective Drivers"
    )

    st.write(
        customer_row.get(
            "top_protective_drivers",
            "No protective drivers available.",
        )
    )


# -------------------------------------------------------------------
# Business Explanation
# -------------------------------------------------------------------

st.markdown(
    "#### 💡 Business Explanation"
)

business_explanation = (
    customer_row.get(
        "business_explanation",
        "No business explanation available.",
    )
)

st.info(
    business_explanation
)