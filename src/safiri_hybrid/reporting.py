from __future__ import annotations

from pathlib import Path

import pandas as pd


def _top_table(scored: pd.DataFrame, limit: int = 20) -> str:
    cols = [
        "declaration_id",
        "risk_score",
        "anomaly_score",
        "confidence_score",
        "business_exposure",
        "supervised_score",
        "risk_tier",
        "flagged",
        "explanation",
    ]
    available = [c for c in cols if c in scored.columns]
    top = scored.head(limit)[available].copy()
    for col in ["risk_score", "anomaly_score", "confidence_score", "business_exposure", "supervised_score"]:
        if col in top.columns:
            top[col] = top[col].map(lambda value: f"{value:.4f}")
    return top.to_markdown(index=False)


def write_markdown_report(
    output_dir: Path,
    detector,
    metrics: dict,
    valid_scored: pd.DataFrame,
    test_scored: pd.DataFrame,
    synthetic_scored: pd.DataFrame,
) -> None:
    business_boost = getattr(detector, "business_boost", 0.0)
    use_supervised = getattr(detector, "use_supervised", False)
    lambda_supervised = getattr(detector, "lambda_supervised", 0.0)
    use_deep_learning = getattr(detector, "use_deep_learning", False)

    if use_supervised:
        evidence_formula = (
            f"`evidence = {detector.lambda_rule:.2f}*rule + {detector.lambda_peer:.2f}*peer "
            f"+ {detector.lambda_if:.2f}*IF + {lambda_supervised:.2f}*supervised`"
        )
    else:
        evidence_formula = (
            f"`evidence = {detector.lambda_rule:.2f}*rule + {detector.lambda_peer:.2f}*peer "
            f"+ {detector.lambda_if:.2f}*IF`"
        )

    lines = [
        "# SAFIRI — Interpretable Anomaly Triage Report",
        "",
        "This system ranks customs/shipment records for investigation with limited review resources.",
        "Training-only fitting, validation-based threshold calibration, and frozen-threshold evaluation on test and synthetic stress tests.",
        "",
        "## Output Components (per record)",
        "",
        "| Component | Meaning |",
        "|-----------|---------|",
        "| **risk_score** | Investigation priority (anomaly evidence × business exposure) |",
        "| **anomaly_score** | Pure deviation from normal behaviour (peer + unsupervised) |",
        "| **confidence_score** | Evidence reliability (peer group size, tier agreement, data completeness) |",
        "| **supervised_score** | XGBoost fraud probability (Tier 4, supervised learning) |",
        "| **explanation** | Human-readable reason tied to peer context or rule violation |",
        "",
        "## Pipeline Configuration",
        "",
        f"- Anomaly evidence formula: {evidence_formula}",
        f"- Risk formula: `risk = evidence × (1 + {business_boost:.2f} × business_exposure)`",
        f"- LOF enabled: `{detector.use_lof}`",
        f"- Supervised (XGBoost) enabled: `{use_supervised}`",
        f"- Deep learning (MLP) enabled: `{use_deep_learning}`",
        f"- Validation-calibrated flag threshold on `risk_score`: `{float(detector.flag_threshold_):.6f}`",
        f"- Review capacity target: top `{detector.threshold_config.flag_top_percent:.2f}%` of validation cases",
        "",
    ]

    for section_name, section_value in metrics.items():
        lines.append(f"## {section_name.replace('_', ' ').title()}")
        lines.append("")
        if isinstance(section_value, dict):
            for key, value in section_value.items():
                if isinstance(value, dict):
                    lines.append(f"### {key}")
                    lines.append("")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            lines.append(f"#### {sub_key}")
                            lines.append("")
                            for sub_sub_key, sub_sub_value in sub_value.items():
                                if isinstance(sub_sub_value, float):
                                    lines.append(f"- `{sub_sub_key}`: {sub_sub_value:.4f}")
                                else:
                                    lines.append(f"- `{sub_sub_key}`: {sub_sub_value}")
                            lines.append("")
                        elif isinstance(sub_value, float):
                            lines.append(f"- `{sub_key}`: {sub_value:.4f}")
                        else:
                            lines.append(f"- `{sub_key}`: {sub_value}")
                    lines.append("")
                else:
                    if isinstance(value, float):
                        lines.append(f"- `{key}`: {value:.4f}")
                    else:
                        lines.append(f"- `{key}`: {value}")
            lines.append("")
        else:
            if isinstance(section_value, float):
                lines.append(f"- `{section_name}`: {section_value:.4f}")
            else:
                lines.append(f"- `{section_name}`: {section_value}")
            lines.append("")

    lines.extend(
        [
            "## Top Validation Cases",
            "",
            _top_table(valid_scored),
            "",
            "## Top Test Cases",
            "",
            _top_table(test_scored),
            "",
            "## Top Synthetic Anomalies",
            "",
            _top_table(synthetic_scored[synthetic_scored["synthetic_is_anomaly"] == 1]),
            "",
        ]
    )

    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
