"""Plotly chart builders."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PLOTLY_TEMPLATE = "plotly_dark"
COLORWAY = ["#5B8CFF", "#7C5CFF", "#00D4FF", "#38D39F", "#F4B942", "#FF5C7C"]


def _style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY,
        font={"family": "Poppins", "color": "#FFFFFF"},
        margin={"l": 24, "r": 24, "t": 54, "b": 24},
        legend={"orientation": "h", "y": -0.18},
    )
    return fig


def revenue_trend(transactions: pd.DataFrame) -> go.Figure:
    data = transactions.set_index("InvoiceDate").resample("ME")["TotalPrice"].sum().reset_index()
    fig = px.area(data, x="InvoiceDate", y="TotalPrice", title="Monthly Revenue Trend")
    return _style(fig)


def country_distribution(transactions: pd.DataFrame) -> go.Figure:
    data = transactions.groupby("Country", as_index=False)["TotalPrice"].sum().sort_values("TotalPrice", ascending=False).head(10)
    fig = px.bar(data, x="Country", y="TotalPrice", title="Revenue by Country", color="TotalPrice")
    return _style(fig)


def top_products(transactions: pd.DataFrame) -> go.Figure:
    data = transactions.groupby("Description", as_index=False)["Quantity"].sum().sort_values("Quantity", ascending=False).head(10)
    fig = px.bar(data, x="Quantity", y="Description", orientation="h", title="Top Products by Units Sold")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return _style(fig)


def rfm_distribution(rfm: pd.DataFrame) -> go.Figure:
    fig = px.scatter_matrix(rfm, dimensions=["Recency", "Frequency", "Monetary"], title="RFM Distribution")
    return _style(fig)


def cluster_3d(rfm: pd.DataFrame, labels) -> go.Figure:
    frame = rfm.copy()
    frame["Cluster"] = labels.astype(str)
    fig = px.scatter_3d(frame, x="Recency", y="Frequency", z="Monetary", color="Cluster", title="3D Customer Cluster Map")
    return _style(fig)


def correlation_heatmap(rfm: pd.DataFrame) -> go.Figure:
    corr = rfm[["Recency", "Frequency", "Monetary"]].corr()
    fig = px.imshow(corr, text_auto=True, title="RFM Correlation Heatmap", color_continuous_scale="Blues")
    return _style(fig)


def elbow_curve(scores: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scores["k"], y=scores["inertia"], mode="lines+markers", name="Inertia"))
    fig.add_trace(go.Scatter(x=scores["k"], y=scores["silhouette"], mode="lines+markers", name="Silhouette", yaxis="y2"))
    fig.update_layout(
        title="Elbow Curve and Silhouette Score",
        yaxis={"title": "Inertia"},
        yaxis2={"title": "Silhouette", "overlaying": "y", "side": "right"},
    )
    return _style(fig)
