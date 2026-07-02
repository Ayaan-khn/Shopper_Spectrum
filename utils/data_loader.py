"""Data ingestion and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config.settings import PROCESSED_DATA_PATH, RAW_DATA_CANDIDATES, RFM_DATA_PATH
from utils.demo_data import build_demo_transactions
from utils.logger import get_logger

logger = get_logger(__name__)


REQUIRED_COLUMNS = {
    "InvoiceNo",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}


@dataclass(frozen=True)
class DataBundle:
    """Container for prepared retail data."""

    transactions: pd.DataFrame
    rfm: pd.DataFrame
    source: str


def find_dataset() -> Path | None:
    """Return the first supported dataset found in the dataset directory."""
    for path in RAW_DATA_CANDIDATES:
        if path.exists():
            return path
    return None


def load_raw_data(path: Path | None = None) -> tuple[pd.DataFrame, str]:
    """Load a retail dataset or create demo data when none exists."""
    dataset_path = path or find_dataset()
    if dataset_path is None:
        logger.info("No dataset found. Using generated demo dataset.")
        return build_demo_transactions(), "Demo dataset"

    if dataset_path.suffix.lower() in {".xlsx", ".xls"}:
        data = pd.read_excel(dataset_path)
    elif dataset_path.suffix.lower() == ".csv":
        data = pd.read_csv(dataset_path, encoding_errors="ignore")
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
    return data, dataset_path.name


def preprocess_transactions(data: pd.DataFrame) -> pd.DataFrame:
    """Clean online retail transactions and create total price."""
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    clean = data.copy()
    clean = clean.dropna(subset=["CustomerID", "Description", "InvoiceDate"])
    clean = clean.drop_duplicates()
    clean["InvoiceNo"] = clean["InvoiceNo"].astype(str)
    clean = clean[~clean["InvoiceNo"].str.startswith("C", na=False)]
    clean["Quantity"] = pd.to_numeric(clean["Quantity"], errors="coerce")
    clean["UnitPrice"] = pd.to_numeric(clean["UnitPrice"], errors="coerce")
    clean = clean[(clean["Quantity"] > 0) & (clean["UnitPrice"] > 0)]
    clean["InvoiceDate"] = pd.to_datetime(clean["InvoiceDate"], errors="coerce")
    clean = clean.dropna(subset=["InvoiceDate"])
    clean["CustomerID"] = clean["CustomerID"].astype(float).astype(int).astype(str)
    clean["Description"] = clean["Description"].astype(str).str.strip().str.upper()
    clean["Country"] = clean["Country"].fillna("Unknown").astype(str)
    clean["TotalPrice"] = clean["Quantity"] * clean["UnitPrice"]
    return clean.sort_values("InvoiceDate").reset_index(drop=True)


def build_rfm(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build RFM features at customer level."""
    snapshot_date = transactions["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        transactions.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: int((snapshot_date - x.max()).days)),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
            Country=("Country", lambda x: x.mode().iat[0] if not x.mode().empty else "Unknown"),
        )
        .reset_index()
    )
    rfm["Monetary"] = rfm["Monetary"].round(2)
    return rfm


def load_data_bundle(path: Path | None = None, persist: bool = True) -> DataBundle:
    """Load, clean, feature engineer, and optionally save data artifacts."""
    raw, source = load_raw_data(path)
    transactions = preprocess_transactions(raw)
    rfm = build_rfm(transactions)
    if persist:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        transactions.to_csv(PROCESSED_DATA_PATH, index=False)
        rfm.to_csv(RFM_DATA_PATH, index=False)
    return DataBundle(transactions=transactions, rfm=rfm, source=source)
