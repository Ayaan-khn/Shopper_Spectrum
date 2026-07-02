"""Product recommendation page."""

from __future__ import annotations

import streamlit as st

from app import get_data, get_models
from utils.ui import hero, page_setup


page_setup("Product Recommendation")
bundle = get_data()
_, recommender = get_models(hash(bundle.rfm.to_csv(index=False)), hash(bundle.transactions.head(2000).to_csv(index=False)), bundle)

hero("Product Recommendation", "Find similar products using item-based collaborative filtering and cosine similarity.")

popular = (
    bundle.transactions.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(20)
    .index.tolist()
)

with st.sidebar:
    st.title("Recommendation Search")
    selected = st.selectbox("Popular products", popular)
    query = st.text_input("Product name", value=selected, placeholder="Try a product name or misspelling")
    search = st.button("Get Recommendations", use_container_width=True)

if search or query:
    matched, recommendations = recommender.recommend(query)
    if matched is None:
        st.error("No close product match found. Try a different product name.")
    else:
        st.success(f"Matched product: {matched}")
        cols = st.columns(5)
        for col, item in zip(cols, recommendations):
            with col:
                st.markdown(
                    f"""
                    <div class="glass-card metric-card">
                      <span>{item.badge}</span>
                      <strong style="font-size:1.05rem">{item.product}</strong>
                      <span class="metric-delta">Similarity {item.score:.2%}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

st.markdown("<div class='glass-card panel'><h3>Autocomplete Catalog</h3>", unsafe_allow_html=True)
st.dataframe({"Product": recommender.products[:300]}, use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)
