from __future__ import annotations

from recommendation.engine import RecommendationEngine
from segmentation.model import predict_segment, train_segmentation
from utils.data_loader import build_rfm, preprocess_transactions
from utils.demo_data import build_demo_transactions


def test_preprocessing_and_rfm_pipeline() -> None:
    raw = build_demo_transactions(n_customers=40)
    transactions = preprocess_transactions(raw)
    rfm = build_rfm(transactions)

    assert not transactions.empty
    assert {"Recency", "Frequency", "Monetary"}.issubset(rfm.columns)
    assert (transactions["TotalPrice"] > 0).all()


def test_segmentation_prediction() -> None:
    raw = build_demo_transactions(n_customers=80)
    rfm = build_rfm(preprocess_transactions(raw))
    result = train_segmentation(rfm)
    prediction = predict_segment(result, recency=30, frequency=5, monetary=500)

    assert result.score > -1
    assert prediction["name"]


def test_recommendation_engine() -> None:
    raw = build_demo_transactions(n_customers=60)
    transactions = preprocess_transactions(raw)
    engine = RecommendationEngine.fit(transactions)
    matched, recommendations = engine.recommend(engine.products[0])

    assert matched is not None
    assert len(recommendations) == 5
