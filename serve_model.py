"""Minimal Flask web server for SAFIRI anomaly triage.

Usage:
    pip install flask pandas joblib scikit-learn
    python serve_model.py

Endpoints:
    POST /score   — score one or more shipment records (JSON array)
    GET  /health  — health check
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ── Load trained model ──────────────────────────────────────────────
MODEL_PATH = ROOT / "outputs" / "hybrid_detector.joblib"
detector = joblib.load(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")
print(f"  Flag threshold: {detector.flag_threshold_:.6f}")
print(f"  Business boost: {detector.business_boost}")

app = Flask(__name__)

# Column names matching the training CSV (snake_case after normalize_columns)
EXPECTED_COLUMNS = [
    "declaration_id", "date", "office_id", "process_type", "import_type",
    "import_use", "payment_type", "mode_of_transport", "declarant_id",
    "importer_id", "seller_id", "courier_id", "hs6_code",
    "country_of_departure", "country_of_origin", "tax_rate", "tax_type",
    "country_of_origin_indicator", "net_mass", "item_price",
]

# Output columns returned to the caller (the 4 triage components + supporting)
OUTPUT_COLUMNS = [
    "declaration_id",
    "risk_score",
    "anomaly_score",
    "confidence_score",
    "business_exposure",
    "anomaly_evidence",
    "risk_tier",
    "flagged",
    "anomaly_rank",
    "explanation",
]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": str(MODEL_PATH)})


@app.route("/score", methods=["POST"])
def score():
    """Score shipment records.

    Accepts JSON body:
        { "records": [ { "declaration_id": "...", ... }, ... ] }

    Returns:
        { "results": [ { "declaration_id": "...", "risk_score": 0.82, ... }, ... ] }
    """
    payload = request.get_json(force=True)
    if "records" not in payload:
        return jsonify({"error": "Missing 'records' key in JSON body"}), 400

    records = payload["records"]
    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "'records' must be a non-empty list"}), 400

    df = pd.DataFrame(records)

    # Allow both "Title Case" and "snake_case" column names
    col_map = {col.replace("_", " ").title().replace(" ", " "): col for col in EXPECTED_COLUMNS}
    # Also build a map from space-separated title case (as in CSV headers)
    for col in EXPECTED_COLUMNS:
        title_version = col.replace("_", " ").title()
        col_map[title_version] = col
    df = df.rename(columns=col_map)

    # Fill optional columns with NaN if missing
    for col in ["fraud", "critical_fraud"]:
        if col not in df.columns:
            df[col] = 0

    try:
        scored = detector.score(df, threshold_mode="frozen")
    except Exception as exc:
        return jsonify({"error": f"Scoring failed: {exc}"}), 500

    available_cols = [c for c in OUTPUT_COLUMNS if c in scored.columns]
    result_df = scored[available_cols].copy()

    # Convert to JSON-safe types
    result_records = result_df.to_dict(orient="records")
    for rec in result_records:
        for key, val in rec.items():
            if isinstance(val, (float,)) and pd.isna(val):
                rec[key] = None

    return jsonify({"count": len(result_records), "results": result_records})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
