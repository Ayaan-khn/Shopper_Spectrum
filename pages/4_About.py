"""About page."""

from __future__ import annotations

import streamlit as st

from utils.ui import hero, page_setup


page_setup("About")
hero("About Shopper Spectrum", "A modular portfolio-grade e-commerce analytics platform built for business insight and ML demonstration.")

st.markdown(
    """
    <div class="glass-card panel">
      <h3>Workflow</h3>
      <p>Raw transaction data is cleaned, validated, enriched with total price, transformed into RFM features, clustered with multiple algorithms, and paired with an item-based collaborative filtering recommender.</p>
    </div>
    <div class="glass-card panel">
      <h3>Architecture</h3>
      <p><strong>Data</strong> -> preprocessing and validation -> <strong>Segmentation</strong> -> model comparison -> <strong>Recommendation</strong> -> sparse cosine similarity -> <strong>Streamlit UI</strong> -> dashboard, predictions, analytics, exports.</p>
    </div>
    <div class="glass-card panel">
      <h3>Algorithms Used</h3>
      <p>KMeans, DBSCAN, Agglomerative Clustering, StandardScaler, Silhouette Score, customer-product pivot tables, sparse matrices, and cosine similarity.</p>
    </div>
    <div class="glass-card panel">
      <h3>Business Impact</h3>
      <p>The platform helps teams target high-value customers, detect churn risk, personalize product discovery, and monitor revenue health across countries and product categories.</p>
    </div>
    <div class="glass-card panel">
      <h3>Future Scope</h3>
      <p>Production data connectors, scheduled model retraining, user authentication, campaign activation, feature store integration, and A/B testing dashboards.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
