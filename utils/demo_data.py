"""Synthetic demo data used when no retail dataset is present."""

from __future__ import annotations

import numpy as np
import pandas as pd


PRODUCTS = [
    "WHITE HANGING HEART T-LIGHT HOLDER",
    "REGENCY CAKESTAND 3 TIER",
    "JUMBO BAG RED RETROSPOT",
    "PARTY BUNTING",
    "LUNCH BAG RED RETROSPOT",
    "SET OF 3 CAKE TINS PANTRY DESIGN",
    "PACK OF 72 RETROSPOT CAKE CASES",
    "ASSORTED COLOUR BIRD ORNAMENT",
    "NATURAL SLATE HEART CHALKBOARD",
    "HEART OF WICKER SMALL",
    "WOODEN PICTURE FRAME WHITE FINISH",
    "VICTORIAN GLASS HANGING T-LIGHT",
    "RABBIT NIGHT LIGHT",
    "VINTAGE SNAP CARDS",
    "ALARM CLOCK BAKELIKE RED",
]

COUNTRIES = ["United Kingdom", "Germany", "France", "Netherlands", "Spain", "Australia", "Norway"]


def build_demo_transactions(seed: int = 42, n_customers: int = 420) -> pd.DataFrame:
    """Create a realistic online retail sample for demos and tests."""
    rng = np.random.default_rng(seed)
    rows = []
    invoice_id = 536000
    start_date = pd.Timestamp("2011-01-01")

    for customer_id in range(12000, 12000 + n_customers):
        segment = rng.choice(["vip", "regular", "occasional", "at_risk"], p=[0.14, 0.34, 0.36, 0.16])
        if segment == "vip":
            orders, recency_floor, spend_scale = rng.integers(8, 18), 0, 1.8
        elif segment == "regular":
            orders, recency_floor, spend_scale = rng.integers(4, 10), 20, 1.2
        elif segment == "occasional":
            orders, recency_floor, spend_scale = rng.integers(1, 5), 60, 0.9
        else:
            orders, recency_floor, spend_scale = rng.integers(1, 4), 180, 0.7

        country = rng.choice(COUNTRIES, p=[0.72, 0.07, 0.07, 0.04, 0.04, 0.03, 0.03])
        for _ in range(int(orders)):
            invoice_id += 1
            order_date = start_date + pd.Timedelta(days=int(rng.integers(recency_floor, 365)))
            basket_size = int(rng.integers(1, 6))
            for product in rng.choice(PRODUCTS, size=basket_size, replace=False):
                quantity = int(max(1, rng.poisson(5 * spend_scale)))
                unit_price = round(float(rng.gamma(2.2, 2.8) + rng.uniform(0.5, 8.0)), 2)
                rows.append(
                    {
                        "InvoiceNo": str(invoice_id),
                        "StockCode": f"SKU{abs(hash(product)) % 99999:05d}",
                        "Description": product,
                        "Quantity": quantity,
                        "InvoiceDate": order_date + pd.Timedelta(hours=int(rng.integers(8, 22))),
                        "UnitPrice": unit_price,
                        "CustomerID": customer_id,
                        "Country": country,
                    }
                )

    return pd.DataFrame(rows)
