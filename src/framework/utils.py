from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# Small constant to avoid numerical instability (e.g., division by zero)
EPS = 1e-12


def robust_mad(series: pd.Series) -> float:
    """
    Compute Median Absolute Deviation (MAD) in a robust way.

    - Removes NaN and infinite values before computation
    - More robust to outliers than standard deviation

    Returns:
        float: MAD value, or NaN if input is empty
    """
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    median = clean.median()
    return float(np.median(np.abs(clean - median)))


def quantile_rank(reference: np.ndarray, values: Iterable[float]) -> np.ndarray:
    """
    Convert values into percentile ranks based on a reference distribution.

    - Handles NaN by replacing with median of reference
    - Ignores non-finite values in reference
    - Output range: [0, 1]

    Args:
        reference: Array used as the reference distribution
        values: Values to transform into percentile ranks

    Returns:
        np.ndarray: Percentile ranks of input values
    """
    values_arr = np.asarray(list(values), dtype=float)
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.zeros_like(values_arr)
    ref = np.sort(ref)
    safe_values = np.nan_to_num(values_arr, nan=np.nanmedian(ref))
    return np.searchsorted(ref, safe_values, side="right") / ref.size


def compose_key(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """
    Create a composite key by concatenating multiple columns.

    - Converts all values to string
    - Missing values are replaced with a placeholder

    Useful for:
        - Grouping (e.g., country + product)
        - Feature crosses in ML models

    Returns:
        pd.Series: Concatenated key
    """
    values = frame[columns].astype("string").fillna("__MISSING__")
    key = values.iloc[:, 0].astype(str)
    for column in columns[1:]:
        key = key + "|" + values[column].astype(str)
    return key


def safe_ratio(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray
) -> pd.Series:
    """
    Compute ratio safely.

    - Replaces zero denominator with NaN
    - Removes infinite results

    Useful for:
        - price / weight
        - tax / value

    Returns:
        pd.Series: Safe ratio values
    """
    num = pd.Series(numerator, copy=False, dtype=float)
    den = pd.Series(denominator, copy=False, dtype=float).replace(0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan)


def tail_percentile_score(percentile: pd.Series | np.ndarray) -> pd.Series:
    """
    Convert percentile into an anomaly score.

    - Values near 0 or 1 → high anomaly score
    - Values near 0.5 → low anomaly score

    Formula:
        score = 1 - 2 * min(p, 1 - p)

    Returns:
        pd.Series: Score in range [0, 1]
    """
    pct = pd.Series(percentile, copy=False, dtype=float).clip(0.0, 1.0)
    return 1.0 - (2.0 * np.minimum(pct, 1.0 - pct))


def is_round_value(series: pd.Series, base: float = 1000.0) -> pd.Series:
    """
    Detect whether values are 'round numbers' based on a given base.

    Example (base=1000):
        1000, 2000 → True
        1050 → False

    Useful for fraud detection:
        - Fraudulent values are often artificially rounded

    Returns:
        pd.Series: Boolean mask indicating round values
    """
    scaled = pd.Series(series, copy=False, dtype=float) / base
    return pd.Series(
        np.isclose(scaled, np.round(scaled), atol=1e-9),
        index=scaled.index
    )