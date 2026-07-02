# Shopper Spectrum

Shopper Spectrum is an AI-powered e-commerce analytics platform for customer segmentation, product recommendations, and interactive retail intelligence. It is built as a polished Streamlit SaaS-style application with modular machine learning, reusable UI components, and deployment-ready structure.

## Features

- RFM customer segmentation with KMeans, DBSCAN, and Agglomerative Clustering.
- Automatic best-model selection using silhouette score.
- Business labels such as High Value, Regular, Occasional, and At Risk.
- Item-based collaborative filtering with sparse customer-product matrices and cosine similarity.
- Fuzzy product matching for autocomplete and misspelling recovery.
- Executive dashboard with revenue, customers, transactions, products, AOV, country analysis, top products, and recent activity.
- Advanced analytics for RFM distributions, 3D cluster visualization, correlation heatmap, elbow curve, and exports.
- Aurora Glassmorphism UI with animated CSS background, Plotly dark charts, floating cards, and premium interaction styling.
- Demo dataset fallback when no Online Retail file is available.

## Screenshots

Add screenshots in `images/` after running the app locally.

## Architecture

```text
Raw retail data
  -> preprocessing and validation
  -> transaction features and RFM table
  -> segmentation model comparison
  -> product similarity engine
  -> Streamlit dashboard and pages
  -> exports, logs, and saved models
```

## Folder Structure

```text
project/
  app.py
  requirements.txt
  README.md
  dataset/
  models/
  notebooks/
  pages/
  utils/
  recommendation/
  segmentation/
  analytics/
  css/
  assets/
  images/
  exports/
  config/
  logs/
  tests/
```

## Dataset

Place an Online Retail dataset in `dataset/` using one of these names:

- `online_retail.xlsx`
- `online_retail.csv`
- `Online Retail.xlsx`
- `Online Retail.csv`

Required columns are `InvoiceNo`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, and `Country`.

If no dataset is present, the app generates a realistic demo dataset so the full platform still runs.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Testing

```bash
pytest
```

## Deployment Guide

1. Commit the repository to GitHub.
2. Add the dataset to `dataset/` or connect secure storage in production.
3. Deploy on Streamlit Community Cloud, Docker, Azure App Service, or another Python host.
4. Set resource limits high enough for the customer-product similarity matrix if using a large catalog.
5. Schedule periodic retraining for fresh segmentation and recommendations.

## Future Improvements

- Authentication and role-based access.
- Warehouse connectors for BigQuery, Snowflake, or Postgres.
- Campaign activation exports.
- Real-time event ingestion.
- A/B testing and uplift reporting.
- Model monitoring and automated retraining.

## License

MIT License.

## Contributing

Open issues or pull requests with focused improvements. Keep modules typed, tested, documented, and aligned with the existing architecture.
