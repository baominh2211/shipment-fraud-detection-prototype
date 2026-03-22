from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import is_round_value, safe_ratio


ROUND_PRICE_VALUES = [10000, 25000, 50000, 100000, 250000, 500000]

# Semantic contradiction table: (tax_type, origin_indicator) pairs that are
# logically impossible or highly suspicious regardless of frequency.
# - FEU1 (free-zone exemption) should never appear with indicator Y (preferential origin)
# - C (customs union) should never appear with indicator G (general/non-preferential)
# Add domain-specific rules as they are discovered.
CONTRADICTORY_TAX_ORIGIN: set[tuple[str, str]] = {
    ("FEU1", "Y"),
    ("C", "G"),
    ("FEU1", "B"),
}


def apply_rule_features(
    frame: pd.DataFrame,
    global_stats: dict[str, float],
    mismatch_threshold: float = 0.05,
    rare_combo_threshold: float = 5e-4,
) -> pd.DataFrame:
    scored = frame.copy()

    # --- Arithmetic mismatch (continuous severity) ---
    expected_total = scored["quantity"] * scored["unit_price"]
    mismatch_ratio = safe_ratio(
        (expected_total - scored["total_value"]).abs(),
        expected_total.abs().clip(lower=1.0),
    )
    scored["arithmetic_mismatch_ratio"] = mismatch_ratio.fillna(0.0)

    has_arithmetic_fields = (
        scored["quantity"].gt(0)
        & scored["unit_price"].notna()
        & scored["total_value"].notna()
    )
    scored["rule_arithmetic_mismatch"] = (
        has_arithmetic_fields
        & scored["arithmetic_mismatch_ratio"].gt(mismatch_threshold)
    ).astype(int)

    # Continuous severity: 0 when ratio <= threshold, ramps up to 1.0
    scored["rule_arithmetic_severity"] = np.where(
        has_arithmetic_fields,
        (scored["arithmetic_mismatch_ratio"].clip(upper=1.0) - mismatch_threshold).clip(lower=0.0)
        / (1.0 - mismatch_threshold),
        0.0,
    )

    # --- Invalid values (expanded) ---
    scored["rule_zero_net_mass_positive_price"] = (
        scored["net_mass"].le(0) & scored["item_price"].gt(0)
    ).astype(int)
    scored["rule_zero_item_price_positive_mass"] = (
        scored["item_price"].le(0) & scored["net_mass"].gt(0)
    ).astype(int)
    scored["rule_invalid_quantity_or_total"] = (
        (scored["quantity"].lt(0) | scored["total_value"].lt(0))
        & (scored["quantity"].notna() | scored["total_value"].notna())
    ).astype(int)
    scored["rule_negative_net_mass"] = scored["net_mass"].lt(0).astype(int)
    scored["rule_negative_item_price"] = scored["item_price"].lt(0).astype(int)

    # --- Price density ---
    scored["rule_missing_price_per_kg"] = scored["price_per_kg"].isna().astype(int)
    scored["rule_extreme_price_density"] = (
        scored["price_per_kg"].fillna(0).gt(global_stats["price_per_kg_p99"] * 20)
        | scored["price_per_kg"]
        .fillna(global_stats["price_per_kg_p01"])
        .lt(global_stats["price_per_kg_p01"] / 20)
    ).astype(int)

    # Continuous severity for extreme price density
    log_ppk = np.log1p(scored["price_per_kg"].fillna(0).clip(lower=0))
    log_p99 = np.log1p(max(global_stats["price_per_kg_p99"], 1e-6))
    log_p01 = np.log1p(max(global_stats["price_per_kg_p01"], 1e-6))
    log_range = max(log_p99 - log_p01, 0.1)
    scored["rule_price_density_deviation"] = (
        ((log_ppk - (log_p01 + log_p99) / 2).abs() / log_range).clip(0.0, 3.0) / 3.0
    )

    # --- Round price ---
    is_round = is_round_value(scored["item_price"]) | scored["item_price"].round(0).isin(
        ROUND_PRICE_VALUES
    )
    scored["rule_round_item_price"] = is_round.astype(int)
    scored["rule_repeated_rounded_value"] = (
        is_round & scored["round_price_freq"].ge(global_stats["round_price_high_freq"])
    ).astype(int)

    # --- Tax/origin: semantic contradiction + statistical rarity ---
    tax_origin_key = list(
        zip(
            scored["tax_type"].astype(str).values,
            scored["country_of_origin_indicator"].astype(str).values,
        )
    )
    scored["rule_tax_origin_contradiction"] = pd.Series(
        [1 if pair in CONTRADICTORY_TAX_ORIGIN else 0 for pair in tax_origin_key],
        index=scored.index,
        dtype=int,
    )
    scored["rule_tax_indicator_rare"] = scored["pair_freq_tax_indicator"].lt(rare_combo_threshold).astype(int)
    # Combined: either semantic contradiction OR statistically rare
    scored["rule_tax_indicator_mismatch"] = (
        scored["rule_tax_origin_contradiction"] | scored["rule_tax_indicator_rare"]
    ).astype(int)

    # --- Severity-proportional composite rule score ---
    # Uses continuous severity where available; binary flags otherwise
    scored["rule_score_raw"] = (
        0.30 * scored["rule_arithmetic_severity"]
        + 0.15 * scored["rule_zero_net_mass_positive_price"]
        + 0.08 * scored["rule_zero_item_price_positive_mass"]
        + 0.05 * scored["rule_invalid_quantity_or_total"]
        + 0.03 * scored["rule_negative_net_mass"]
        + 0.03 * scored["rule_negative_item_price"]
        + 0.06 * scored["rule_missing_price_per_kg"]
        + 0.10 * scored["rule_price_density_deviation"]
        + 0.04 * scored["rule_round_item_price"]
        + 0.04 * scored["rule_repeated_rounded_value"]
        + 0.07 * scored["rule_tax_origin_contradiction"]
        + 0.05 * scored["rule_tax_indicator_rare"]
    ).clip(0.0, 1.0)
    return scored



def build_explanation(row: pd.Series) -> str:
    """Build a human-readable explanation tied to peer context or rule violation.

    Each reason is a (priority, text) pair.  The final explanation
    concatenates the top 3 reasons.  Reasons reference the specific
    peer group or rule that triggered.
    """
    reasons: list[tuple[float, str]] = []

    # --- Rule violations (Tier 1) ---
    if row.get("rule_arithmetic_mismatch", 0):
        reasons.append(
            (
                1.00,
                f"declared total deviates by {100 * float(row['arithmetic_mismatch_ratio']):.1f}% from quantity × unit_price",
            )
        )
    if row.get("rule_zero_net_mass_positive_price", 0):
        reasons.append((0.82, "net mass is zero or negative while item price stays positive"))
    if row.get("rule_zero_item_price_positive_mass", 0):
        reasons.append((0.80, "item price is zero or negative while net mass stays positive"))
    if row.get("rule_tax_origin_contradiction", 0):
        reasons.append((0.69, "tax-type and origin-indicator combination is logically contradictory"))
    elif row.get("rule_tax_indicator_rare", 0):
        reasons.append((0.65, "tax-type and origin-indicator combination is almost never seen in training history"))
    if row.get("rule_repeated_rounded_value", 0):
        reasons.append((0.58, "declared item price is a repeated rounded value across many unrelated records"))
    elif row.get("rule_round_item_price", 0):
        reasons.append((0.44, "declared item price is a suspiciously round value"))
    if row.get("rule_negative_net_mass", 0):
        reasons.append((0.48, "net mass is negative — invalid declaration"))
    if row.get("rule_negative_item_price", 0):
        reasons.append((0.46, "item price is negative — invalid declaration"))

    # --- Peer-context deviations (Tier 2) ---
    if row.get("valuation_peer_tail", 0.0) > 0.95 and row.get("valuation_peer_count", 0) >= 1:
        pct = max(0.1, min(float(row["valuation_peer_percentile"]), 1 - float(row["valuation_peer_percentile"])) * 100)
        reasons.append(
            (
                0.95,
                f"valuation sits in the most extreme {pct:.1f}% of peer group '{row['peer_level']}' (n={int(row['peer_count'])})",
            )
        )
    if row.get("valuation_metric_peer_z", 0.0) > 2.5 and pd.notna(row.get("valuation_metric_peer_ratio")):
        ratio = float(row["valuation_metric_peer_ratio"])
        direction = "below" if ratio < 1 else "above"
        multiple = (1.0 / max(ratio, 1e-6)) if ratio < 1 else ratio
        reasons.append((0.88, f"valuation is {multiple:.1f}× {direction} the peer median (group '{row['peer_level']}')"))
    if row.get("importer_valuation_z", 0.0) > 2.5:
        reasons.append((0.74, "value deviates sharply from the importer's historical baseline"))
    if row.get("seller_origin_valuation_z", 0.0) > 2.5:
        reasons.append((0.72, "value deviates sharply from the seller-origin historical baseline"))
    if row.get("rarity_score", 0.0) > 0.75:
        reasons.append(
            (
                0.70,
                f"seller-country or product-country combination is historically rare ({100 * float(row.get('max_pair_frequency', 0)):.3f}% frequency)",
            )
        )

    # --- Unsupervised anomaly (Tier 3) ---
    if row.get("isolation_score", 0.0) > 0.75:
        reasons.append((0.52, "overall feature profile is isolated from historical shipment patterns"))

    # --- Business exposure context ---
    business_exp = row.get("business_exposure", 0.0)
    if business_exp > 0.85:
        reasons.append((0.60, "high-value shipment with elevated tax rate — prioritised for investigation"))
    elif business_exp > 0.70:
        reasons.append((0.40, "moderate business exposure amplifies investigation priority"))

    if not reasons:
        reasons.append((0.10, "moderately unusual combined anomaly profile"))

    reasons.sort(key=lambda item: item[0], reverse=True)
    prefix = f"{row['risk_tier']}: " if bool(row.get("flagged", False)) else ""
    return prefix + "; ".join(reason for _, reason in reasons[:3])
