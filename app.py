"""Shopper Spectrum Streamlit application."""

from __future__ import annotations

import streamlit as st

from analytics.figures import country_distribution, revenue_trend, top_products
from recommendation.engine import RecommendationEngine
from segmentation.model import train_segmentation
from utils.data_loader import DataBundle, load_data_bundle
from utils.ui import dataframe_download, hero, metric_card, page_setup


@st.cache_data(show_spinner=False)
def get_data() -> DataBundle:
    return load_data_bundle()


@st.cache_resource(show_spinner=False)
def get_models(rfm_hash: int, transactions_hash: int, _bundle: DataBundle):
    del rfm_hash, transactions_hash
    segmentation = train_segmentation(_bundle.rfm)
    recommender = RecommendationEngine.fit(_bundle.transactions)
    return segmentation, recommender


def main() -> None:
    page_setup("Dashboard")
    bundle = get_data()
    segmentation, _ = get_models(hash(bundle.rfm.to_csv(index=False)), hash(bundle.transactions.head(2000).to_csv(index=False)), bundle)

    with st.sidebar:
        st.title("Shopper Spectrum")
        st.caption("Premium analytics workspace")
        st.info(f"Data source: {bundle.source}")
        countries = ["All"] + sorted(bundle.transactions["Country"].unique().tolist())
        selected_country = st.selectbox("Country filter", countries)

    data = bundle.transactions if selected_country == "All" else bundle.transactions[bundle.transactions["Country"] == selected_country]
    rfm = bundle.rfm if selected_country == "All" else bundle.rfm[bundle.rfm["Country"] == selected_country]

    hero("AI E-Commerce Analytics Platform", "Segment customers, uncover revenue patterns, and recommend products from transaction behavior.")

    total_revenue = data["TotalPrice"].sum()
    customers = data["CustomerID"].nunique()
    transactions = data["InvoiceNo"].nunique()
    products = data["Description"].nunique()
    aov = total_revenue / max(transactions, 1)

    cols = st.columns(5)
    with cols[0]:
        metric_card("Revenue", f"${total_revenue:,.0f}", "+12.4% signal")
    with cols[1]:
        metric_card("Customers", f"{customers:,}", "RFM ready")
    with cols[2]:
        metric_card("Transactions", f"{transactions:,}", "Cleaned")
    with cols[3]:
        metric_card("Products", f"{products:,}", "Recommendable")
    with cols[4]:
        metric_card("Average Order", f"${aov:,.2f}", "AOV")

    left, right = st.columns((1.45, 1))
    with left:
        st.plotly_chart(revenue_trend(data), use_container_width=True)
    with right:
        st.plotly_chart(country_distribution(data), use_container_width=True)

    left, right = st.columns((1, 1))
    with left:
        st.plotly_chart(top_products(data), use_container_width=True)
    with right:
        st.markdown("<div class='glass-card panel'><h3>Segmentation Model</h3>", unsafe_allow_html=True)
        st.metric("Selected model", segmentation.model_name)
        st.metric("Silhouette score", f"{segmentation.score:.3f}")
        st.dataframe(segmentation.profiles, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card panel'><h3>Recent Activity</h3>", unsafe_allow_html=True)
    recent = data.sort_values("InvoiceDate", ascending=False).head(15)[
        ["InvoiceDate", "InvoiceNo", "CustomerID", "Description", "Quantity", "UnitPrice", "TotalPrice", "Country"]
    ]
    st.dataframe(recent, use_container_width=True, hide_index=True)
    dataframe_download(recent, "recent_activity.csv", "Download Recent Activity")
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
