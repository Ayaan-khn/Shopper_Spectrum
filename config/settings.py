"""Application configuration for Shopper Spectrum."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"
MODELS_DIR = ROOT_DIR / "models"
EXPORTS_DIR = ROOT_DIR / "exports"
LOGS_DIR = ROOT_DIR / "logs"
CSS_DIR = ROOT_DIR / "css"

RAW_DATA_CANDIDATES = [
    DATASET_DIR / "online_retail.xlsx",
    DATASET_DIR / "online_retail.csv",
    DATASET_DIR / "Online Retail.xlsx",
    DATASET_DIR / "Online Retail.csv",
]

PROCESSED_DATA_PATH = DATASET_DIR / "processed_transactions.csv"
RFM_DATA_PATH = DATASET_DIR / "rfm_features.csv"
SEGMENTATION_MODEL_PATH = MODELS_DIR / "segmentation_model.joblib"
RECOMMENDER_MODEL_PATH = MODELS_DIR / "recommendation_engine.joblib"

APP_NAME = "Shopper Spectrum"
APP_TAGLINE = "AI-powered e-commerce analytics for segmentation and recommendations."

CLUSTER_LABELS = {
    "high_value": {
        "name": "High Value",
        "icon": "Diamond",
        "persona": "Loyal customers with strong purchase frequency and spend.",
        "strategy": "Offer VIP access, loyalty perks, bundles, and early product drops.",
    },
    "regular": {
        "name": "Regular",
        "icon": "Repeat",
        "persona": "Reliable buyers with steady engagement and healthy order value.",
        "strategy": "Use personalized cross-sell campaigns and subscription incentives.",
    },
    "occasional": {
        "name": "Occasional",
        "icon": "Sparkles",
        "persona": "Lower-frequency shoppers who need timely nudges to return.",
        "strategy": "Send curated offers, seasonal reminders, and discovery recommendations.",
    },
    "at_risk": {
        "name": "At Risk",
        "icon": "Alert",
        "persona": "Customers with older purchase recency and declining engagement.",
        "strategy": "Trigger win-back campaigns, service recovery, and limited-time incentives.",
    },
}
