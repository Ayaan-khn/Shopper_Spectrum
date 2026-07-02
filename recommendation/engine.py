"""Item-based product recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import RECOMMENDER_MODEL_PATH


@dataclass
class ProductRecommendation:
    """Single recommendation result."""

    product: str
    score: float
    badge: str


@dataclass
class RecommendationEngine:
    """Sparse item-based collaborative filtering recommender."""

    products: list[str]
    similarity: np.ndarray

    @classmethod
    def fit(cls, transactions: pd.DataFrame) -> "RecommendationEngine":
        matrix = transactions.pivot_table(
            index="CustomerID",
            columns="Description",
            values="Quantity",
            aggfunc="sum",
            fill_value=0,
        )
        sparse = csr_matrix(matrix.T.values)
        similarity = cosine_similarity(sparse)
        engine = cls(products=list(matrix.columns), similarity=similarity)
        RECOMMENDER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(engine, RECOMMENDER_MODEL_PATH)
        return engine

    def match_product(self, query: str) -> str | None:
        """Find the nearest known product name for a possibly misspelled query."""
        if not query:
            return None
        normalized = query.strip().upper()
        if normalized in self.products:
            return normalized
        matches = get_close_matches(normalized, self.products, n=1, cutoff=0.25)
        return matches[0] if matches else None

    def recommend(self, query: str, top_n: int = 5) -> tuple[str | None, list[ProductRecommendation]]:
        """Return top similar products for a query."""
        matched = self.match_product(query)
        if matched is None:
            return None, []
        idx = self.products.index(matched)
        scores = list(enumerate(self.similarity[idx]))
        scores = sorted(scores, key=lambda item: item[1], reverse=True)
        recommendations: list[ProductRecommendation] = []
        for product_idx, score in scores:
            product = self.products[product_idx]
            if product == matched:
                continue
            badge = "Strong Match" if score >= 0.6 else "Relevant" if score >= 0.35 else "Discovery"
            recommendations.append(ProductRecommendation(product=product, score=float(score), badge=badge))
            if len(recommendations) == top_n:
                break
        return matched, recommendations
