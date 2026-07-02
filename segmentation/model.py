"""Customer segmentation models."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config.settings import CLUSTER_LABELS, SEGMENTATION_MODEL_PATH


FEATURES = ["Recency", "Frequency", "Monetary"]


@dataclass
class SegmentationResult:
    """Training result for segmentation."""

    model_name: str
    score: float
    labels: np.ndarray
    profiles: pd.DataFrame
    scaler: StandardScaler
    model: object
    label_map: dict[int, str]


def _safe_silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    if len(unique_labels) < 2 or len(unique_labels) >= len(features):
        return -1.0
    return float(silhouette_score(features, labels))


def _business_label_profiles(profiles: pd.DataFrame) -> dict[int, str]:
    monetary_rank = profiles["Monetary"].rank(ascending=False, method="first")
    frequency_rank = profiles["Frequency"].rank(ascending=False, method="first")
    recency_rank = profiles["Recency"].rank(ascending=True, method="first")
    composite = monetary_rank + frequency_rank + recency_rank
    ordered_clusters = profiles.assign(score=composite).sort_values("score")["Cluster"].tolist()
    names = ["high_value", "regular", "occasional", "at_risk"]
    return {int(cluster): names[min(i, len(names) - 1)] for i, cluster in enumerate(ordered_clusters)}


def train_segmentation(rfm: pd.DataFrame) -> SegmentationResult:
    """Train several clustering algorithms and choose the best silhouette score."""
    scaler = StandardScaler()
    x = scaler.fit_transform(rfm[FEATURES])
    candidates: list[tuple[str, object, np.ndarray, float]] = []

    for k in range(3, min(7, len(rfm) - 1)):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(x)
        candidates.append((f"KMeans k={k}", model, labels, _safe_silhouette(x, labels)))

    for eps in [0.55, 0.75, 0.95, 1.15]:
        model = DBSCAN(eps=eps, min_samples=6)
        labels = model.fit_predict(x)
        candidates.append((f"DBSCAN eps={eps}", model, labels, _safe_silhouette(x, labels)))

    for k in range(3, min(7, len(rfm) - 1)):
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(x)
        candidates.append((f"Agglomerative k={k}", model, labels, _safe_silhouette(x, labels)))

    model_name, model, labels, score = max(candidates, key=lambda item: item[3])
    if score < 0:
        model = KMeans(n_clusters=4, random_state=42, n_init=20)
        labels = model.fit_predict(x)
        score = _safe_silhouette(x, labels)
        model_name = "KMeans k=4"

    training_frame = rfm.copy()
    training_frame["Cluster"] = labels
    training_frame = training_frame[training_frame["Cluster"] >= 0]
    profiles = (
        training_frame.groupby("Cluster")[FEATURES]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    profiles.columns = ["_".join(col).strip("_") for col in profiles.columns]
    profiles = profiles.reset_index()
    compact_profiles = training_frame.groupby("Cluster")[FEATURES].mean().reset_index()
    label_map = _business_label_profiles(compact_profiles)

    result = SegmentationResult(
        model_name=model_name,
        score=float(score),
        labels=labels,
        profiles=profiles,
        scaler=scaler,
        model=model,
        label_map=label_map,
    )
    SEGMENTATION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, SEGMENTATION_MODEL_PATH)
    return result


def predict_segment(result: SegmentationResult, recency: float, frequency: float, monetary: float) -> dict[str, str | int | float]:
    """Predict a customer segment for supplied RFM values."""
    x = result.scaler.transform(pd.DataFrame([[recency, frequency, monetary]], columns=FEATURES))
    if hasattr(result.model, "predict"):
        cluster = int(result.model.predict(x)[0])
    else:
        centers = result.scaler.transform(result.profiles.rename(columns={
            "Recency_mean": "Recency",
            "Frequency_mean": "Frequency",
            "Monetary_mean": "Monetary",
        })[FEATURES])
        cluster = int(result.profiles.iloc[np.argmin(np.linalg.norm(centers - x, axis=1))]["Cluster"])
    key = result.label_map.get(cluster, "occasional")
    meta = CLUSTER_LABELS[key]
    return {"cluster": cluster, "key": key, **meta}


def elbow_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """Return KMeans inertia and silhouette scores over candidate k values."""
    scaler = StandardScaler()
    x = scaler.fit_transform(rfm[FEATURES])
    rows = []
    for k in range(2, min(10, len(rfm) - 1)):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(x)
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": _safe_silhouette(x, labels)})
    return pd.DataFrame(rows)
