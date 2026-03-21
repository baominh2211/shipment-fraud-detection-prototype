from __future__ import annotations

import numpy as np
import pandas as pd

from .data import normalize_columns
from .utils import compose_key


# Common "round" prices often seen in suspicious declarations
ROUND_PRICE_VALUES = [10000.0, 25000.0, 50000.0, 100000.0, 250000.0, 500000.0]


def _attach_shipment_fields(frame: pd.DataFrame) -> pd.DataFrame:
    # Add basic derived fields for shipment analysis
    enriched = frame.copy()

    # Ensure quantity is valid (avoid zero / missing)
    quantity = enriched["net_mass"].clip(lower=1.0).fillna(1.0)
    enriched["quantity"] = quantity.round(3)

    # Unit price = value per unit
    enriched["unit_price"] = (enriched["item_price"] / enriched["quantity"]).replace([np.inf, -np.inf], np.nan)

    # Total value (normalized)
    enriched["total_value"] = enriched["item_price"].astype(float)

    # Estimated tax exposure (risk signal)
    enriched["revenue_at_risk"] = (enriched["item_price"] * enriched["tax_rate"].fillna(0.0) / 100.0).clip(lower=0.0)

    return enriched



def build_synthetic_evaluation_set(
    df: pd.DataFrame,
    anomaly_count: int = 240,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    # Normalize schema + add derived shipment features
    base = _attach_shipment_fields(normalize_columns(df))

    # Shuffle data to avoid ordering bias
    clean = base.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Types of synthetic anomalies to inject
    anomaly_types = [
        "valuation_peer_shift",     # abnormal price (too high / too low)
        "arithmetic_mismatch",      # inconsistent quantity * price
        "rare_pair_swap",           # unusual seller-origin combination
        "repeated_rounded_value",   # suspicious round values
        "tax_indicator_mismatch",   # inconsistent tax info
    ]

    per_type = max(1, anomaly_count // len(anomaly_types))
    available = clean.index.to_numpy()
    injected_rows: list[pd.Series] = []

    # Identify rare patterns for anomaly injection
    origin_counts = clean["country_of_origin"].value_counts(normalize=True)
    rare_origins = list(origin_counts.nsmallest(min(20, len(origin_counts))).index)

    seller_origin_keys = compose_key(clean, ["seller_id", "country_of_origin"])
    rare_seller_origin_pairs = list(seller_origin_keys.value_counts(normalize=True).nsmallest(200).index)

    tax_indicator_keys = compose_key(clean, ["tax_type", "country_of_origin_indicator"])
    rare_tax_indicator_pairs = list(tax_indicator_keys.value_counts(normalize=True).nsmallest(100).index)

    for anomaly_type in anomaly_types:
        # Randomly select rows to transform into anomalies
        chosen = rng.choice(available, size=min(per_type, len(available)), replace=False)

        for idx in chosen:
            row = clean.loc[idx].copy()

            # Keep trace of original row
            row["source_declaration_id"] = row["declaration_id"]

            # Mark as synthetic anomaly
            row["synthetic_is_anomaly"] = 1
            row["synthetic_anomaly_type"] = anomaly_type

            # Assign new ID (avoid collision with real data)
            row["declaration_id"] = int(f"9{len(injected_rows) + 1:08d}")

            if anomaly_type == "valuation_peer_shift":
                # Inflate or deflate price significantly
                multiplier = rng.choice([rng.uniform(0.05, 0.12), rng.uniform(6.0, 12.0)])
                row["unit_price"] = max(0.1, row["unit_price"] * multiplier)
                row["total_value"] = row["quantity"] * row["unit_price"]
                row["item_price"] = row["total_value"]
                row["revenue_at_risk"] = abs(row["total_value"] - clean["total_value"].median())

            elif anomaly_type == "arithmetic_mismatch":
                # Break consistency: total ≠ quantity * unit_price
                expected_total = row["quantity"] * row["unit_price"]
                row["total_value"] = expected_total * rng.choice([rng.uniform(0.45, 0.70), rng.uniform(1.35, 1.80)])
                row["item_price"] = row["total_value"]
                row["revenue_at_risk"] = abs(row["total_value"] - expected_total)

            elif anomaly_type == "rare_pair_swap":
                # Replace with rare seller-origin combination
                if rare_seller_origin_pairs:
                    seller_id, origin = rng.choice(rare_seller_origin_pairs).split("|", 1)
                    row["seller_id"] = seller_id
                    row["country_of_origin"] = origin
                    row["country_of_departure"] = origin
                elif rare_origins:
                    chosen_origin = rng.choice(rare_origins)
                    row["country_of_origin"] = chosen_origin
                    row["country_of_departure"] = chosen_origin

            elif anomaly_type == "repeated_rounded_value":
                # Force suspicious round value
                row["item_price"] = float(rng.choice(ROUND_PRICE_VALUES))
                row["total_value"] = row["item_price"]
                row["unit_price"] = row["total_value"] / max(row["quantity"], 1.0)

            elif anomaly_type == "tax_indicator_mismatch" and rare_tax_indicator_pairs:
                # Assign rare / inconsistent tax indicator combo
                tax_type, indicator = rng.choice(rare_tax_indicator_pairs).split("|", 1)
                row["tax_type"] = tax_type
                row["country_of_origin_indicator"] = indicator

            injected_rows.append(row)

    # Build synthetic anomaly dataset
    synthetic = pd.DataFrame(injected_rows)

    # Mark original data as normal
    normals = clean.copy()
    normals["source_declaration_id"] = normals["declaration_id"]
    normals["synthetic_is_anomaly"] = 0
    normals["synthetic_anomaly_type"] = "baseline"

    # Combine normal + anomaly data
    return pd.concat([normals, synthetic], ignore_index=True)