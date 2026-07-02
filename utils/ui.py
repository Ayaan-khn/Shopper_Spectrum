"""Shared Streamlit UI helpers."""

from __future__ import annotations

import base64

import pandas as pd
import streamlit as st

from config.settings import APP_NAME, APP_TAGLINE, CSS_DIR


def load_css() -> None:
    """Load the Aurora Glassmorphism stylesheet."""
    css_path = CSS_DIR / "theme.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_setup(title: str = APP_NAME) -> None:
    st.set_page_config(page_title=f"{title} | {APP_NAME}", page_icon="SS", layout="wide", initial_sidebar_state="expanded")
    load_css()


def hero(title: str, subtitle: str = APP_TAGLINE) -> None:
    st.markdown(
        f"""
        <section class="hero glass-card fade-up">
          <div>
            <p class="eyebrow">Aurora Commerce Intelligence</p>
            <h1>{title}</h1>
            <p>{subtitle}</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, delta: str | None = None) -> None:
    delta_html = f"<span class='metric-delta'>{delta}</span>" if delta else ""
    st.markdown(
        f"""
        <div class="metric-card glass-card">
          <span>{label}</span>
          <strong>{value}</strong>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def glass_panel_start(title: str | None = None) -> None:
    if title:
        st.markdown(f"<div class='glass-card panel'><h3>{title}</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='glass-card panel'>", unsafe_allow_html=True)


def glass_panel_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def dataframe_download(data: pd.DataFrame, filename: str, label: str) -> None:
    csv = data.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv).decode()
    st.markdown(
        f"<a class='download-button' href='data:file/csv;base64,{b64}' download='{filename}'>{label}</a>",
        unsafe_allow_html=True,
    )
