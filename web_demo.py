"""SAFIRI Web Demo — Anomaly Triage Dashboard

Samples 100 real records from test data (80 normal + 20 fraud),
loads the trained model, scores them with calibrated threshold,
and displays results in a web UI.
Flagged anomalies can be explained via trollLLM API.

Usage:
    pip install flask pandas joblib scikit-learn xgboost openai
    python web_demo.py
    # Open http://localhost:5001 in your browser
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ── Load trained model ──────────────────────────────────────────────
MODEL_PATH = ROOT / "outputs" / "hybrid_detector.joblib"
detector = joblib.load(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")
print(f"  Flag threshold (production): {detector.flag_threshold_:.6f}")

# ── Learn peer group stats from training data ────────────────────────
# So generated normal records match the distribution the model expects.
from safiri_hybrid.data import normalize_columns as _normalize_columns

_train_df = pd.read_csv(ROOT / "data" / "df_syn_train_eng.csv")
_train_norm = _normalize_columns(_train_df.copy())
_train_norm["_ppkg"] = _train_norm["item_price"] / _train_norm["net_mass"].replace(0, np.nan)

PEER_STATS = (
    _train_norm.groupby(["hs6_code", "country_of_origin"])
    .agg(count=("item_price", "count"),
         price_median=("item_price", "median"),
         mass_median=("net_mass", "median"))
    .reset_index()
)
TOP_PEERS = PEER_STATS[PEER_STATS["count"] >= 50].sort_values("count", ascending=False).head(15)
_DECLARANTS = _train_norm["declarant_id"].value_counts().head(15).index.tolist()
_IMPORTERS = _train_norm["importer_id"].value_counts().head(15).index.tolist()
_SELLERS = [s for s in _train_norm["seller_id"].dropna().value_counts().head(15).index.tolist() if s]
del _train_df, _train_norm  # free memory

app = Flask(__name__)

# ── Sample data generation ──────────────────────────────────────────
# Normal records are generated to match REAL peer group distributions,
# so the model's peer-score and isolation-score work properly.
# Anomalies are injected with clear, explainable deviations.

# Each anomaly type with its human-readable reason
ANOMALY_CATALOG = {
    "extreme_undervaluation": "Declared price is 99% below the typical price for this product+origin group — a classic indicator of under-invoicing to evade import duties.",
    "extreme_overvaluation": "Declared price is 100–500× above the typical price for this product — may indicate money laundering through inflated invoices.",
    "zero_mass_positive_price": "Net mass is declared as zero while item price is positive — a physically impossible declaration that suggests data manipulation.",
    "negative_values": "Net mass is negative — a physically impossible value that indicates either a data entry error or deliberate manipulation.",
    "tax_origin_contradiction": "Tax type FEU1 (Free Zone exemption) combined with origin indicator Y (Preferential) — these two statuses are mutually exclusive and cannot coexist.",
    "rare_route": "Shipment route Zimbabwe→Papua New Guinea is extremely rare in historical data, with an unknown seller — this unusual combination warrants investigation.",
    "round_price_high_value": "Exactly $1,000,000 declared value with 15% tax rate — suspiciously round amounts on high-value shipments often signal fraudulent declarations.",
    "high_tax_high_value": "Tax rate above 50% on a shipment worth over $10M — extreme financial exposure that creates strong incentive for duty evasion.",
}

ANOMALY_TYPES_20 = [
    "extreme_undervaluation", "extreme_overvaluation", "zero_mass_positive_price",
    "negative_values", "tax_origin_contradiction", "rare_route", "round_price_high_value",
    "high_tax_high_value",
    "extreme_undervaluation", "extreme_overvaluation", "zero_mass_positive_price",
    "negative_values", "tax_origin_contradiction", "rare_route", "round_price_high_value",
    "high_tax_high_value",
    "extreme_undervaluation", "extreme_overvaluation", "zero_mass_positive_price",
    "negative_values",
]


def _generate_normal_record(idx: int, rng: random.Random) -> dict:
    """Generate a normal record matching real peer group distributions."""
    peer = TOP_PEERS.iloc[rng.randint(0, len(TOP_PEERS) - 1)]
    hs6 = int(peer["hs6_code"])
    origin = peer["country_of_origin"]
    price = float(peer["price_median"]) * rng.uniform(0.5, 1.5)
    mass = float(peer["mass_median"]) * rng.uniform(0.5, 1.5)
    return {
        "Declaration ID": f"DEMO{idx:04d}",
        "Date": f"2025-{rng.randint(1,12):02d}-{rng.randint(1,25):02d}",
        "Office ID": rng.choice([30, 40, 20, 10]),
        "Process Type": "B",
        "Import Type": rng.choice([11, 80]),
        "Import Use": rng.choice([21, 26]),
        "Payment Type": rng.choice([11, 43]),
        "Mode of Transport": rng.choice([10, 40]),
        "Declarant ID": rng.choice(_DECLARANTS),
        "Importer ID": rng.choice(_IMPORTERS),
        "Seller ID": rng.choice(_SELLERS),
        "Courier ID": "",
        "HS6 Code": hs6,
        "Country of Departure": origin,
        "Country of Origin": origin,
        "Tax Rate": round(rng.choice([0.0, 4.9, 6.5, 8.0]), 1),
        "Tax Type": rng.choice(["A", "C", "FCN1"]),
        "Country of Origin Indicator": rng.choice(["G", "E", "Y"]),
        "Net Mass": round(mass, 1),
        "Item Price": round(price, 2),
        "Fraud": 0,
        "Critical Fraud": 0,
        "_injected_anomaly": None,
    }


def _inject_anomaly(record: dict, anomaly_type: str, rng: random.Random) -> dict:
    """Mutate a normal record to inject a specific anomaly pattern."""
    record = record.copy()
    record["_injected_anomaly"] = anomaly_type
    peer = TOP_PEERS[TOP_PEERS["hs6_code"] == record["HS6 Code"]].iloc[0]

    if anomaly_type == "extreme_undervaluation":
        record["Item Price"] = round(float(peer["price_median"]) * 0.01, 2)
        record["Net Mass"] = round(float(peer["mass_median"]) * rng.uniform(10, 50), 1)
    elif anomaly_type == "extreme_overvaluation":
        record["Item Price"] = round(float(peer["price_median"]) * rng.uniform(100, 500), 2)
        record["Net Mass"] = round(rng.uniform(0.1, 2.0), 1)
    elif anomaly_type == "zero_mass_positive_price":
        record["Net Mass"] = 0.0
        record["Item Price"] = round(float(peer["price_median"]) * rng.uniform(5, 20), 2)
    elif anomaly_type == "negative_values":
        record["Net Mass"] = round(-rng.uniform(10, 500), 1)
    elif anomaly_type == "tax_origin_contradiction":
        record["Tax Type"] = "FEU1"
        record["Country of Origin Indicator"] = "Y"
    elif anomaly_type == "rare_route":
        record["Country of Departure"] = "ZW"
        record["Country of Origin"] = "PG"
        record["Seller ID"] = "SEL_RARE_001"
    elif anomaly_type == "round_price_high_value":
        record["Item Price"] = 1000000.0
        record["Net Mass"] = round(rng.uniform(500, 5000), 1)
        record["Tax Rate"] = 15.0
    elif anomaly_type == "high_tax_high_value":
        record["Tax Rate"] = round(rng.uniform(50, 200), 1)
        record["Item Price"] = round(rng.uniform(10_000_000, 100_000_000), 2)
        record["Net Mass"] = round(rng.uniform(1000, 10000), 1)
    return record


def generate_demo_dataset(seed: int = 2025) -> pd.DataFrame:
    """Generate 100 records: 80 normal + 20 injected anomalies."""
    rng = random.Random(seed)
    records = []
    for i in range(80):
        records.append(_generate_normal_record(i, rng))
    for j, anomaly_type in enumerate(ANOMALY_TYPES_20[:20]):
        rec = _generate_normal_record(80 + j, rng)
        rec = _inject_anomaly(rec, anomaly_type, rng)
        records.append(rec)
    rng.shuffle(records)
    return pd.DataFrame(records)


# ── Pre-generate and score ──────────────────────────────────────────
print("Generating 100 demo records (80 normal + 20 anomalous)...")
DEMO_DF = generate_demo_dataset()
DEMO_INJECTED = DEMO_DF["_injected_anomaly"].copy()

score_input = DEMO_DF.drop(columns=["_injected_anomaly"])
print("Scoring with trained model (calibrate threshold for demo)...")
original_threshold = detector.flag_threshold_
detector.threshold_config.flag_top_percent = 25.0
SCORED_DF = detector.score(score_input, threshold_mode="calibrate")
DEMO_THRESHOLD = detector.flag_threshold_
SCORED_DF["injected_anomaly"] = DEMO_INJECTED.values

# Print detection summary
n_anomalies = int(DEMO_INJECTED.notna().sum())
n_detected = int(SCORED_DF[SCORED_DF["injected_anomaly"].notna()]["flagged"].sum())
n_fp = int(SCORED_DF[SCORED_DF["injected_anomaly"].isna()]["flagged"].sum())
print(f"  Anomalies: {n_anomalies}/100")
print(f"  Detected (TP): {n_detected}/{n_anomalies}")
print(f"  False positives: {n_fp}")
print(f"  Demo threshold: {DEMO_THRESHOLD:.6f}")


# ── AI explanation (trollLLM) ─────────────────────────────────────────

SYSTEM_PROMPT = """You are a customs risk analyst writing for non-technical stakeholders (managers, auditors, compliance officers). The SAFIRI system flagged a shipment as suspicious.

Write a short, plain-English explanation (2-3 sentences max) that anyone can understand. Follow these rules:

1. Lead with WHAT is suspicious (e.g., "This shipment's declared value is unusually low for its weight")
2. Explain WHY it matters (e.g., "which could indicate under-invoicing to avoid import duties")
3. Suggest a concrete next step (e.g., "Recommend verifying the invoice against market prices for HS code 392690")

Avoid jargon like "z-score", "isolation forest", "peer deviation". Instead use everyday language:
- "unusually low/high compared to similar shipments" instead of "peer deviation"
- "the AI model gives X% fraud probability" instead of "supervised_score"
- "this pattern is rare in our historical data" instead of "isolation score"

Keep it under 45 words. Be direct and actionable."""


def _get_trollllm_client():
    """Create and cache a trollLLM OpenAI-compatible client."""
    from openai import OpenAI as _OpenAI

    CUSTOM_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    return _OpenAI(
        base_url="https://chat.trollllm.xyz/v1",
        api_key="sk-trollllm-f57d0b58b71589a7fcd887577ef42a90a9dfeca061f8677afb58577311afee2c",
        default_headers=CUSTOM_HEADERS,
    )


# ── Discover available models on startup ─────────────────────────────
TROLLLLM_MODELS: list[str] = []
TROLLLLM_DEFAULT_MODEL = "claude-sonnet-4-20250514"  # sensible default

def _discover_models():
    """Fetch available model list from trollLLM /v1/models endpoint."""
    global TROLLLLM_MODELS, TROLLLLM_DEFAULT_MODEL
    try:
        client = _get_trollllm_client()
        models_response = client.models.list()
        TROLLLLM_MODELS = sorted([m.id for m in models_response.data])
        print(f"  trollLLM available models ({len(TROLLLLM_MODELS)}): {TROLLLLM_MODELS}")

        # Pick best available model (prefer claude-sonnet, then any claude, then first)
        for preferred in ["claude-sonnet-4-20250514", "claude-3-5-sonnet", "claude-3-sonnet",
                          "claude-3-haiku", "claude-sonnet", "claude-haiku"]:
            matches = [m for m in TROLLLLM_MODELS if preferred in m]
            if matches:
                TROLLLLM_DEFAULT_MODEL = matches[0]
                break
        else:
            if TROLLLLM_MODELS:
                TROLLLLM_DEFAULT_MODEL = TROLLLLM_MODELS[0]

        print(f"  Using model: {TROLLLLM_DEFAULT_MODEL}")
    except Exception as exc:
        print(f"  ⚠ Could not list trollLLM models: {exc}")
        print(f"  Will try default: {TROLLLLM_DEFAULT_MODEL}")

print("Connecting to trollLLM API (chat.trollllm.xyz)...")
_discover_models()


def explain_with_gpt(record_data: dict) -> str:
    """Call trollLLM API to explain why a record is flagged."""
    try:
        client = _get_trollllm_client()

        user_msg = f"""Analyze this flagged customs record:

Record fields:
- Declaration ID: {record_data.get('declaration_id', 'N/A')}
- HS6 Code: {record_data.get('hs6_code', 'N/A')}
- Country of Origin: {record_data.get('country_of_origin', 'N/A')}
- Country of Departure: {record_data.get('country_of_departure', 'N/A')}
- Net Mass: {record_data.get('net_mass', 'N/A')}
- Item Price: {record_data.get('item_price', 'N/A')}
- Tax Rate: {record_data.get('tax_rate', 'N/A')}
- Tax Type: {record_data.get('tax_type', 'N/A')}
- Origin Indicator: {record_data.get('country_of_origin_indicator', 'N/A')}

Model scores:
- risk_score: {record_data.get('risk_score', 'N/A')}
- anomaly_score: {record_data.get('anomaly_score', 'N/A')}
- confidence_score: {record_data.get('confidence_score', 'N/A')}
- business_exposure: {record_data.get('business_exposure', 'N/A')}
- rule_score: {record_data.get('rule_score', 'N/A')}
- peer_score: {record_data.get('peer_score', 'N/A')}
- isolation_score: {record_data.get('isolation_score', 'N/A')}
- supervised_score: {record_data.get('supervised_score', 'N/A')}
- risk_tier: {record_data.get('risk_tier', 'N/A')}

Model explanation: {record_data.get('explanation', 'N/A')}"""

        response = client.chat.completions.create(
            model=TROLLLLM_DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception as exc:
        error_msg = str(exc)
        # Provide actionable error messages
        if "model" in error_msg.lower() and ("not found" in error_msg.lower() or "invalid" in error_msg.lower()):
            return (f"❌ Model '{TROLLLLM_DEFAULT_MODEL}' không khả dụng trên trollLLM. "
                    f"Các model có sẵn: {', '.join(TROLLLLM_MODELS) if TROLLLLM_MODELS else 'không xác định'}. "
                    f"Chi tiết: {error_msg}")
        elif "401" in error_msg or "unauthorized" in error_msg.lower() or "authentication" in error_msg.lower():
            return f"❌ API key trollLLM không hợp lệ hoặc đã hết hạn. Chi tiết: {error_msg}"
        elif "timeout" in error_msg.lower() or "connect" in error_msg.lower():
            return f"❌ Không thể kết nối đến trollLLM API (chat.trollllm.xyz). Chi tiết: {error_msg}"
        else:
            return f"❌ Lỗi khi gọi trollLLM API: {error_msg}"


# ── HTML Template ───────────────────────────────────────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAFIRI — Anomaly Triage Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0f1923;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a2332 0%, #0d1b2a 100%);
            padding: 20px 30px;
            border-bottom: 2px solid #1e3a5f;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header h1 {
            font-size: 24px;
            color: #4fc3f7;
            font-weight: 600;
        }
        .header .subtitle {
            color: #78909c;
            font-size: 13px;
            margin-top: 4px;
        }
        .stats-bar {
            display: flex;
            gap: 30px;
            padding: 15px 30px;
            background: #1a2332;
            border-bottom: 1px solid #1e3a5f;
        }
        .stat-card {
            background: #0d1b2a;
            border: 1px solid #1e3a5f;
            border-radius: 8px;
            padding: 12px 20px;
            min-width: 150px;
        }
        .stat-card .label { color: #78909c; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
        .stat-card .value { color: #fff; font-size: 24px; font-weight: 700; margin-top: 4px; }
        .stat-card .value.danger { color: #ef5350; }
        .stat-card .value.warning { color: #ffa726; }
        .stat-card .value.success { color: #66bb6a; }

        .controls {
            padding: 15px 30px;
            display: flex;
            gap: 15px;
            align-items: center;
        }
        .controls label { color: #90a4ae; font-size: 13px; }
        .controls select, .controls input {
            background: #1a2332;
            border: 1px solid #1e3a5f;
            color: #e0e0e0;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 13px;
        }
        .btn {
            background: #1e88e5;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            transition: background 0.2s;
        }
        .btn:hover { background: #1565c0; }
        .btn.btn-explain {
            background: #7b1fa2;
            font-size: 11px;
            padding: 4px 10px;
        }
        .btn.btn-explain:hover { background: #6a1b9a; }
        .btn.btn-explain:disabled {
            background: #424242;
            cursor: not-allowed;
        }

        .table-container {
            padding: 0 30px 30px;
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        thead th {
            background: #1a2332;
            color: #4fc3f7;
            padding: 10px 8px;
            text-align: left;
            position: sticky;
            top: 0;
            border-bottom: 2px solid #1e3a5f;
            white-space: nowrap;
            cursor: pointer;
            user-select: none;
        }
        thead th:hover { background: #1e3a5f; }
        tbody tr {
            border-bottom: 1px solid #1e3a5f;
            transition: background 0.15s;
        }
        tbody tr:hover { background: #1a2332; }
        tbody tr.flagged { background: rgba(239, 83, 80, 0.08); }
        tbody tr.flagged:hover { background: rgba(239, 83, 80, 0.15); }
        tbody td { padding: 8px; white-space: nowrap; }

        .score-bar {
            display: inline-block;
            height: 8px;
            border-radius: 4px;
            margin-right: 6px;
            vertical-align: middle;
        }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .badge.high { background: #ef5350; color: white; }
        .badge.medium { background: #ffa726; color: #1a1a1a; }
        .badge.review { background: #42a5f5; color: white; }
        .badge.normal { background: #2e7d32; color: white; }
        .badge.injected { background: #ab47bc; color: white; font-size: 9px; }

        .explanation-cell {
            max-width: 300px;
            white-space: normal;
            word-break: break-word;
            font-size: 11px;
            color: #b0bec5;
        }

        /* GPT Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.7);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .modal-overlay.active { display: flex; }
        .modal {
            background: #1a2332;
            border: 1px solid #1e3a5f;
            border-radius: 12px;
            padding: 24px;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }
        .modal h3 { color: #4fc3f7; margin-bottom: 12px; }
        .modal .gpt-text {
            color: #e0e0e0;
            line-height: 1.6;
            font-size: 14px;
            white-space: pre-wrap;
        }
        .modal .close-btn {
            float: right;
            background: none;
            border: none;
            color: #78909c;
            font-size: 20px;
            cursor: pointer;
        }
        .modal .close-btn:hover { color: #ef5350; }
        .loading { color: #ffa726; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🔍 SAFIRI — Anomaly Triage Dashboard</h1>
            <div class="subtitle">Interpretable Anomaly Triage for Customs / Shipment Data</div>
        </div>
        <div style="text-align: right; font-size: 12px; color: #546e7a;">
            Model threshold: {{ threshold }}<br>
            Business boost: {{ business_boost }}<br>
            <span id="modelBadge" style="color: #ab47bc; font-weight: 600;">AI: loading...</span>
        </div>
    </div>

    <div class="stats-bar">
        <div class="stat-card">
            <div class="label">Total Records</div>
            <div class="value">{{ total }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Flagged</div>
            <div class="value danger">{{ flagged }}</div>
        </div>
        <div class="stat-card">
            <div class="label">High Risk</div>
            <div class="value danger">{{ high_risk }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Medium Risk</div>
            <div class="value warning">{{ medium_risk }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Normal</div>
            <div class="value success">{{ normal }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Injected Anomalies</div>
            <div class="value" style="color: #ab47bc;">{{ fraud_count }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Anomalies Detected</div>
            <div class="value" style="color: #ab47bc;">{{ detected_fraud }}</div>
        </div>
    </div>

    <div class="controls">
        <label>Filter:</label>
        <select id="filterSelect" onchange="applyFilter()">
            <option value="all">All Records</option>
            <option value="flagged">Flagged Only</option>
            <option value="normal">Normal Only</option>
            <option value="injected">Injected Anomalies</option>
            <option value="high">High Risk</option>
        </select>
        <label style="margin-left: 15px;">Search:</label>
        <input type="text" id="searchInput" placeholder="Declaration ID, HS6, Country..." oninput="applyFilter()">
    </div>

    <div class="table-container">
        <table id="mainTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0)">#</th>
                    <th onclick="sortTable(1)">Declaration ID</th>
                    <th onclick="sortTable(2)">Risk Score ↓</th>
                    <th onclick="sortTable(3)">Supervised</th>
                    <th onclick="sortTable(4)">Anomaly Score</th>
                    <th onclick="sortTable(5)">Confidence</th>
                    <th onclick="sortTable(6)">Business Exp.</th>
                    <th onclick="sortTable(7)">Risk Tier</th>
                    <th>HS6</th>
                    <th>Origin</th>
                    <th>Net Mass</th>
                    <th>Item Price</th>
                    <th>Tax Rate</th>
                    <th>Anomaly Type</th>
                    <th>Explanation</th>
                    <th>AI Explain</th>
                </tr>
            </thead>
            <tbody>
                {% for r in records %}
                <tr class="{{ 'flagged' if r.flagged else '' }}"
                    data-flagged="{{ r.flagged|int }}"
                    data-tier="{{ r.risk_tier }}"
                    data-injected="{{ r.injected_anomaly or '' }}"
                    data-idx="{{ loop.index0 }}">
                    <td>{{ r.anomaly_rank }}</td>
                    <td style="font-weight:600;">{{ r.declaration_id }}</td>
                    <td>
                        <span class="score-bar" style="width:{{ (r.risk_score * 80)|int }}px; background: {{ '#ef5350' if r.risk_score > 0.7 else '#ffa726' if r.risk_score > 0.4 else '#66bb6a' }};"></span>
                        {{ r.risk_score | float | round(4) }}
                    </td>
                    <td>
                        <span class="score-bar" style="width:{{ (r.get('supervised_score', 0) * 80)|int }}px; background: {{ '#e040fb' if r.get('supervised_score', 0) > 0.5 else '#78909c' }};"></span>
                        {{ r.get('supervised_score', 0) | float | round(4) }}
                    </td>
                    <td>
                        <span class="score-bar" style="width:{{ (r.anomaly_score * 80)|int }}px; background: {{ '#42a5f5' if r.anomaly_score > 0.7 else '#78909c' }};"></span>
                        {{ r.anomaly_score | float | round(4) }}
                    </td>
                    <td>{{ r.confidence_score | float | round(3) }}</td>
                    <td>{{ "%.3f"|format(r.business_exposure) }}</td>
                    <td>
                        {% if r.risk_tier == 'High Risk' %}
                            <span class="badge high">HIGH</span>
                        {% elif r.risk_tier == 'Medium Risk' %}
                            <span class="badge medium">MEDIUM</span>
                        {% elif r.risk_tier == 'Review' %}
                            <span class="badge review">REVIEW</span>
                        {% else %}
                            <span class="badge normal">NORMAL</span>
                        {% endif %}
                    </td>
                    <td>{{ r.hs6_code }}</td>
                    <td>{{ r.country_of_origin }}</td>
                    <td>{{ r.net_mass | float | round(1) }}</td>
                    <td>{{ "{:,.2f}".format(r.item_price | float) }}</td>
                    <td>{{ r.tax_rate | float | round(1) }}%</td>
                    <td>
                        {% if r.injected_anomaly %}
                            <span class="badge injected">{{ r.injected_anomaly }}</span>
                        {% else %}
                            —
                        {% endif %}
                    </td>
                    <td class="explanation-cell">{{ r.explanation }}</td>
                    <td>
                        {% if r.flagged %}
                        <button class="btn btn-explain" onclick="explainRecord({{ loop.index0 }})">🤖 AI</button>
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <!-- GPT Explanation Modal -->
    <div class="modal-overlay" id="modalOverlay">
        <div class="modal">
            <button class="close-btn" onclick="closeModal()">✕</button>
            <h3>🤖 AI Explanation — <span id="modalRecordId"></span></h3>
            <div id="modalModel" style="font-size: 11px; color: #78909c; margin-bottom: 10px;">Model: loading...</div>
            <div id="modalContent" class="gpt-text loading">Loading...</div>
        </div>
    </div>

    <script>
        const records = {{ records_json|safe }};

        // Fetch and display current model info on load
        fetch('/api/models').then(r => r.json()).then(data => {
            const badge = document.getElementById('modelBadge');
            if (badge) {
                badge.textContent = 'AI: ' + (data.current_model || 'N/A');
                badge.title = 'Available: ' + (data.available_models || []).join(', ');
            }
        }).catch(() => {});

        function applyFilter() {
            const filter = document.getElementById('filterSelect').value;
            const search = document.getElementById('searchInput').value.toLowerCase();
            const rows = document.querySelectorAll('#mainTable tbody tr');
            rows.forEach(row => {
                const flagged = row.dataset.flagged === '1';
                const tier = row.dataset.tier;
                const injected = row.dataset.injected;
                const text = row.textContent.toLowerCase();

                let show = true;
                if (filter === 'flagged') show = flagged;
                else if (filter === 'normal') show = !flagged;
                else if (filter === 'injected') show = !!injected;
                else if (filter === 'high') show = tier === 'High Risk';

                if (show && search) {
                    show = text.includes(search);
                }
                row.style.display = show ? '' : 'none';
            });
        }

        let sortDir = {};
        function sortTable(colIdx) {
            const table = document.getElementById('mainTable');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const dir = sortDir[colIdx] = !(sortDir[colIdx] || false);

            rows.sort((a, b) => {
                let aVal = a.cells[colIdx].textContent.trim().replace(/[,%]/g, '');
                let bVal = b.cells[colIdx].textContent.trim().replace(/[,%]/g, '');
                const aNum = parseFloat(aVal);
                const bNum = parseFloat(bVal);
                if (!isNaN(aNum) && !isNaN(bNum)) {
                    return dir ? aNum - bNum : bNum - aNum;
                }
                return dir ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            });
            rows.forEach(r => tbody.appendChild(r));
        }

        async function explainRecord(idx) {
            const record = records[idx];
            document.getElementById('modalRecordId').textContent = record.declaration_id;
            document.getElementById('modalContent').textContent = 'Đang gọi trollLLM API...';
            document.getElementById('modalContent').className = 'gpt-text loading';
            document.getElementById('modalModel').textContent = 'Model: đang kết nối...';
            document.getElementById('modalOverlay').classList.add('active');

            try {
                const resp = await fetch('/api/explain', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({index: idx})
                });
                const data = await resp.json();
                if (data.error) {
                    document.getElementById('modalContent').textContent = '❌ ' + data.error;
                    document.getElementById('modalContent').className = 'gpt-text';
                } else {
                    document.getElementById('modalContent').textContent = data.explanation;
                    document.getElementById('modalContent').className = 'gpt-text';
                    document.getElementById('modalModel').textContent = 'Model: ' + (data.model_used || 'unknown');
                }
            } catch(e) {
                document.getElementById('modalContent').textContent = '❌ Lỗi kết nối: ' + e.message;
                document.getElementById('modalContent').className = 'gpt-text';
            }
        }

        function closeModal() {
            document.getElementById('modalOverlay').classList.remove('active');
        }

        // Close modal on overlay click
        document.getElementById('modalOverlay').addEventListener('click', function(e) {
            if (e.target === this) closeModal();
        });
    </script>
</body>
</html>
"""


# ── Routes ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    # Prepare records for template
    output_cols = [
        "declaration_id", "risk_score", "anomaly_score", "confidence_score",
        "business_exposure", "anomaly_evidence", "risk_tier", "flagged",
        "anomaly_rank", "explanation", "hs6_code", "country_of_origin",
        "country_of_departure", "net_mass", "item_price", "tax_rate",
        "tax_type", "country_of_origin_indicator", "rule_score",
        "peer_score", "isolation_score", "supervised_score",
        "injected_anomaly",
    ]
    available = [c for c in output_cols if c in SCORED_DF.columns]
    records = SCORED_DF[available].to_dict(orient="records")

    # Fix NaN for JSON
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                rec[k] = 0.0
            elif v is None:
                rec[k] = ""

    flagged_count = int(SCORED_DF["flagged"].sum())
    high_risk = int((SCORED_DF["risk_tier"] == "High Risk").sum())
    medium_risk = int((SCORED_DF["risk_tier"] == "Medium Risk").sum())
    normal_count = int((SCORED_DF["risk_tier"] == "Normal").sum())
    fraud_count = int(SCORED_DF["injected_anomaly"].notna().sum())
    detected_fraud = int(
        SCORED_DF[SCORED_DF["injected_anomaly"].notna()]["flagged"].sum()
    )

    return render_template_string(
        HTML_TEMPLATE,
        records=records,
        records_json=json.dumps(records, default=str),
        total=len(records),
        flagged=flagged_count,
        high_risk=high_risk,
        medium_risk=medium_risk,
        normal=normal_count,
        fraud_count=fraud_count,
        detected_fraud=detected_fraud,
        threshold=f"{DEMO_THRESHOLD:.4f}",
        business_boost=f"{detector.business_boost:.2f}",
    )


@app.route("/api/explain", methods=["POST"])
def api_explain():
    """Call trollLLM to explain a specific record."""
    payload = request.get_json(force=True)
    idx = payload.get("index", 0)

    if idx < 0 or idx >= len(SCORED_DF):
        return jsonify({"error": "Invalid index"}), 400

    row = SCORED_DF.iloc[idx]
    record_data = {}
    for col in SCORED_DF.columns:
        val = row[col]
        if isinstance(val, (np.integer,)):
            val = int(val)
        elif isinstance(val, (np.floating,)):
            val = float(val)
        elif isinstance(val, (np.bool_,)):
            val = bool(val)
        record_data[col] = val

    explanation = explain_with_gpt(record_data)
    return jsonify({
        "explanation": explanation,
        "declaration_id": str(row.get("declaration_id", "")),
        "model_used": TROLLLLM_DEFAULT_MODEL,
    })


@app.route("/api/models", methods=["GET"])
def api_models():
    """Return available trollLLM models and the currently selected one."""
    return jsonify({
        "available_models": TROLLLLM_MODELS,
        "current_model": TROLLLLM_DEFAULT_MODEL,
    })


@app.route("/api/set_model", methods=["POST"])
def api_set_model():
    """Switch the trollLLM model used for explanations."""
    global TROLLLLM_DEFAULT_MODEL
    payload = request.get_json(force=True)
    model = payload.get("model", "")
    if model and (model in TROLLLLM_MODELS or model):  # allow custom model names too
        TROLLLLM_DEFAULT_MODEL = model
        return jsonify({"status": "ok", "model": TROLLLLM_DEFAULT_MODEL})
    return jsonify({"error": "Invalid model name"}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "records": len(SCORED_DF)})


if __name__ == "__main__":
    print(f"\nOpen http://localhost:5001 in your browser")
    print(f"AI Explanation: trollLLM (chat.trollllm.xyz) — model: {TROLLLLM_DEFAULT_MODEL}")
    app.run(host="0.0.0.0", port=5001, debug=False)