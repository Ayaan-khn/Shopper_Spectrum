"""Customer segmentation page."""

from __future__ import annotations

import streamlit as st

from analytics.figures import cluster_3d, rfm_distribution
from app import get_data, get_models
from segmentation.model import predict_segment
from utils.ui import hero, metric_card, page_setup


page_setup("Customer Segmentation")
bundle = get_data()
segmentation, _ = get_models(hash(bundle.rfm.to_csv(index=False)), hash(bundle.transactions.head(2000).to_csv(index=False)), bundle)

hero("Customer Segmentation", "Predict customer value groups using Recency, Frequency, and Monetary behavior.")

with st.sidebar:
    st.title("Segment Inputs")
    recency = st.number_input("Recency in days", min_value=0.0, value=45.0, step=1.0)
    frequency = st.number_input("Frequency", min_value=1.0, value=4.0, step=1.0)
    monetary = st.number_input("Monetary spend", min_value=1.0, value=350.0, step=25.0)
    run_prediction = st.button("Predict Cluster", use_container_width=True)

if run_prediction:
    result = predict_segment(segmentation, recency, frequency, monetary)
    st.toast(f"Predicted segment: {result['name']}")
else:
    result = predict_segment(segmentation, recency, frequency, monetary)

cols = st.columns(4)
with cols[0]:
    metric_card("Cluster", str(result["name"]), f"Cluster {result['cluster']}")
with cols[1]:
    metric_card("Persona", str(result["persona"]))
with cols[2]:
    metric_card("Model", segmentation.model_name)
with cols[3]:
    metric_card("Silhouette", f"{segmentation.score:.3f}")

st.markdown("<div class='glass-card panel'><h3>Marketing Strategy</h3>", unsafe_allow_html=True)
st.write(result["strategy"])
st.markdown("</div>", unsafe_allow_html=True)

left, right = st.columns(2)
with left:
    st.plotly_chart(cluster_3d(bundle.rfm, segmentation.labels), use_container_width=True)
with right:
    st.plotly_chart(rfm_distribution(bundle.rfm), use_container_width=True)

st.markdown("<div class='glass-card panel'><h3>Cluster Statistics</h3>", unsafe_allow_html=True)
st.dataframe(segmentation.profiles, use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)
