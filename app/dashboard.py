from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


DATA_PATH = Path("outputs/churn_priority_table.csv")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}. Run the pipeline first."
        )

    return pd.read_csv(path)


st.set_page_config(
    page_title="Customer Retention Decision Engine",
    layout="wide",
)

st.title("🚀 Customer Retention Decision Engine")
st.caption(
    "Churn prediction, CLV, next-best-action recommendations and ROI analysis."
)

try:
    df = load_data(DATA_PATH)
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()


# -------------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------------

st.sidebar.header("Filters")

segment_options = sorted(df["segment"].dropna().unique().tolist())
action_options = sorted(
    df["recommended_action"].dropna().unique().tolist()
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


# -------------------------------------------------------------------
# Portfolio overview
# -------------------------------------------------------------------

st.subheader("📊 Portfolio Overview")

col1, col2, col3, col4 = st.columns(4)

customer_count = len(filtered_df)

average_churn = (
    filtered_df["churn_score"].mean()
    if not filtered_df.empty
    else 0.0
)

total_clv = (
    filtered_df["predicted_clv"].sum()
    if not filtered_df.empty
    else 0.0
)

total_profit = (
    filtered_df["expected_profit"].sum()
    if not filtered_df.empty
    else 0.0
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


if filtered_df.empty:
    st.warning("No customers match the selected filters.")
    st.stop()


# -------------------------------------------------------------------
# Top retention opportunities
# -------------------------------------------------------------------

st.subheader("🎯 Top Retention Opportunities")

top_customers = filtered_df.sort_values(
    "priority_score",
    ascending=False,
).head(20)

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
# Segment distribution
# -------------------------------------------------------------------

st.subheader("📈 Segment Distribution")

segment_counts = (
    filtered_df["segment"]
    .value_counts()
    .rename_axis("segment")
    .reset_index(name="count")
)

st.bar_chart(
    segment_counts.set_index("segment")
)


# -------------------------------------------------------------------
# Highest ROI customers
# -------------------------------------------------------------------

st.subheader("💰 Highest ROI Customers")

roi_df = filtered_df.sort_values(
    "estimated_roi",
    ascending=False,
).head(20)

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
st.subheader("🔍 Customer Explorer")

customer_options = sorted(
    filtered_df["CustomerID"].dropna().astype(int).unique().tolist()
)

selected_customer_id = st.selectbox(
    "Select CustomerID",
    options=customer_options,
)

customer_row = filtered_df[
    filtered_df["CustomerID"].astype(int) == selected_customer_id
].iloc[0]


st.markdown(f"### Customer {selected_customer_id}")

metric1, metric2, metric3, metric4 = st.columns(4)

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


details_left, details_right = st.columns(2)

with details_left:
    st.markdown("#### Customer Classification")
    st.write(f"**Segment:** {customer_row['segment']}")
    st.write(
        f"**Priority score:** "
        f"{customer_row['priority_score']:,.2f}"
    )
    st.write(
        f"**Action reason:** "
        f"{customer_row['action_reason']}"
    )

with details_right:
    st.markdown("#### Recommended Action")
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


st.markdown("#### Business Explanation")

business_explanation = customer_row.get(
    "business_explanation",
    "No business explanation available.",
)

st.info(business_explanation)