from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Retention Decision Engine",
    layout="wide",
)

st.title("🚀 Customer Retention Decision Engine")

df = pd.read_csv("outputs/churn_priority_table.csv")

st.sidebar.header("Filters")

segment_filter = st.sidebar.multiselect(
    "Segment",
    options=sorted(df["segment"].unique()),
    default=sorted(df["segment"].unique()),
)

action_filter = st.sidebar.multiselect(
    "Action",
    options=sorted(df["recommended_action"].unique()),
    default=sorted(df["recommended_action"].unique()),
)

filtered_df = df[
    df["segment"].isin(segment_filter)
    & df["recommended_action"].isin(action_filter)
]

st.subheader("📊 Portfolio Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Customers",
    len(filtered_df),
)

col2.metric(
    "Avg Churn Score",
    round(filtered_df["churn_score"].mean(), 3),
)

col3.metric(
    "Total Predicted CLV",
    f"{filtered_df['predicted_clv'].sum():,.0f}",
)

col4.metric(
    "Total Expected Profit",
    f"{filtered_df['expected_profit'].sum():,.0f}",
)

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
            "predicted_clv",
            "priority_score",
            "expected_profit",
            "estimated_roi",
        ]
    ],
    width="stretch",
)

st.subheader("📈 Segment Distribution")

segment_counts = (
    filtered_df["segment"]
    .value_counts()
    .reset_index()
)

segment_counts.columns = ["segment", "count"]

st.bar_chart(
    segment_counts.set_index("segment")
)

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
            "estimated_roi",
            "expected_profit",
        ]
    ],
    width="stretch",
)