"""Advanced analytics page."""

from __future__ import annotations

import streamlit as st

from analytics.figures import (
    cluster_3d,
    correlation_heatmap,
    country_distribution,
    elbow_curve,
    revenue_trend,
    rfm_distribution,
    top_products,
)
from app import get_data, get_models
from segmentation.model import elbow_scores
from utils.ui import dataframe_download, hero, page_setup


page_setup("Analytics")
bundle = get_data()
segmentation, _ = get_models(hash(bundle.rfm.to_csv(index=False)), hash(bundle.transactions.head(2000).to_csv(index=False)), bundle)

hero("Interactive Analytics", "Explore product demand, country performance, customer value, and model diagnostics.")

with st.sidebar:
    st.title("Analytics Filters")
    countries = ["All"] + sorted(bundle.transactions["Country"].unique().tolist())
    country = st.selectbox("Country", countries)
    min_date, max_date = bundle.transactions["InvoiceDate"].min().date(), bundle.transactions["InvoiceDate"].max().date()
    dates = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

data = bundle.transactions.copy()
if country != "All":
    data = data[data["Country"] == country]
if isinstance(dates, tuple) and len(dates) == 2:
    start, end = dates
    data = data[(data["InvoiceDate"].dt.date >= start) & (data["InvoiceDate"].dt.date <= end)]

tab1, tab2, tab3 = st.tabs(["Revenue", "Customers", "Model Diagnostics"])
with tab1:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(revenue_trend(data), use_container_width=True)
    with right:
        st.plotly_chart(country_distribution(data), use_container_width=True)
    st.plotly_chart(top_products(data), use_container_width=True)

with tab2:
    st.plotly_chart(rfm_distribution(bundle.rfm), use_container_width=True)
    st.plotly_chart(cluster_3d(bundle.rfm, segmentation.labels), use_container_width=True)

with tab3:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(correlation_heatmap(bundle.rfm), use_container_width=True)
    with right:
        st.plotly_chart(elbow_curve(elbow_scores(bundle.rfm)), use_container_width=True)
    st.metric("Selected model", segmentation.model_name)
    st.metric("Silhouette score", f"{segmentation.score:.3f}")

st.markdown("<div class='glass-card panel'><h3>Exports</h3>", unsafe_allow_html=True)
dataframe_download(data, "filtered_transactions.csv", "Download Filtered Transactions")
dataframe_download(bundle.rfm, "rfm_features.csv", "Download RFM Features")
st.markdown("</div>", unsafe_allow_html=True)
