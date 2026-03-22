from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from safiri_hybrid import (
    HybridAnomalyDetector,
    build_synthetic_evaluation_set,
    find_best_f1_threshold,
    load_split,
    summarize_metrics,
    write_markdown_report,
)
from safiri_hybrid.evaluation import tier_diagnostic_metrics, ranking_metrics, threshold_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the merged Safiri hybrid anomaly pipeline.")
    parser.add_argument("--data-dir", default="data", help="Directory containing train/valid/test CSVs.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for artifacts.")
    parser.add_argument(
        "--flag-top-percent",
        type=float,
        default=5.0,
        help="Validation-calibrated review capacity, expressed as the top percentage of records to flag.",
    )
    parser.add_argument(
        "--synthetic-anomalies",
        type=int,
        default=240,
        help="Number of controlled synthetic anomalies to inject for stress testing.",
    )
    parser.add_argument(
        "--calibrate-f1",
        action="store_true",
        default=False,
        help="Use label-aware F1-maximising threshold calibration on validation set.",
    )
    parser.add_argument(
        "--lambda-rule",
        type=float,
        default=0.03,
        help="Weight for rule score in risk = λ₁·rule + λ₂·peer + λ₃·IF + λ₄·supervised.",
    )
    parser.add_argument(
        "--lambda-peer",
        type=float,
        default=0.04,
        help="Weight for peer score.",
    )
    parser.add_argument(
        "--lambda-if",
        type=float,
        default=0.03,
        help="Weight for unsupervised (IF+LOF) score.",
    )
    parser.add_argument(
        "--lambda-supervised",
        type=float,
        default=0.90,
        help="Weight for supervised (XGBoost) score.",
    )
    parser.add_argument("--no-lof", action="store_true", default=False, help="Disable LOF detector.")
    parser.add_argument(
        "--no-supervised",
        action="store_true",
        default=False,
        help="Disable supervised XGBoost model (fall back to unsupervised-only).",
    )
    parser.add_argument(
        "--use-deep-learning",
        action="store_true",
        default=False,
        help="Additionally train an MLP and blend with XGBoost (60/40).",
    )
    parser.add_argument(
        "--business-boost",
        type=float,
        default=0.10,
        help="Alpha: how much business exposure amplifies risk_score (0=no boost).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _print_tier_table(metrics: dict, split_name: str, label_col: str, scored_df) -> None:
    """Print a detailed per-tier metrics table for a given split."""
    tier_diag = metrics.get("tier_diagnostics", {}).get(split_name, {})
    if not tier_diag:
        return

    # Also compute overall risk_score metrics for comparison
    y = scored_df[label_col].to_numpy(dtype=int)
    overall_ranking = ranking_metrics(y, scored_df["risk_score"].to_numpy(dtype=float))
    overall_threshold = threshold_metrics(y, scored_df["flagged"].to_numpy(dtype=int))

    header = f"{'Tier':<25} {'ROC-AUC':>8} {'AP':>8} {'P@5%':>8} {'P@10%':>8} {'R@5%':>8} {'R@10%':>8}"
    sep = "-" * len(header)
    print(f"\n  {sep}")
    print(f"  {split_name.upper()} -- Per-Tier Diagnostics (label: {label_col})")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    for tier_name, tier_data in tier_diag.items():
        roc = tier_data.get("roc_auc")
        ap = tier_data.get("average_precision")
        p5 = tier_data.get("precision_at_5pct")
        p10 = tier_data.get("precision_at_10pct")
        r5 = tier_data.get("recall_at_5pct")
        r10 = tier_data.get("recall_at_10pct")
        print(f"  {tier_name:<25} {roc:>8.4f} {ap:>8.4f} {p5:>8.4f} {p10:>8.4f} {r5:>8.4f} {r10:>8.4f}")
    # Overall combined
    print(f"  {'-' * len(header)}")
    print(f"  {'COMBINED risk_score':<25} {overall_ranking.get('roc_auc', 0):>8.4f} {overall_ranking.get('average_precision', 0):>8.4f} "
          f"{overall_ranking.get('precision_at_5pct', 0):>8.4f} {overall_ranking.get('precision_at_10pct', 0):>8.4f} "
          f"{overall_ranking.get('recall_at_5pct', 0):>8.4f} {overall_ranking.get('recall_at_10pct', 0):>8.4f}")
    print(f"  Threshold F1={overall_threshold.get('f1', 0):.4f}  Prec={overall_threshold.get('precision', 0):.4f}  Rec={overall_threshold.get('recall', 0):.4f}  Flag-rate={overall_threshold.get('flag_rate', 0):.2%}")
    print(f"  {sep}")


def _generate_charts(metrics: dict, output_dir: Path) -> None:
    """Generate tier comparison bar charts and save to outputs/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[CHART] matplotlib not installed -- skipping chart generation.")
        return

    tier_diag = metrics.get("tier_diagnostics", {})
    splits = [s for s in ["validation", "test"] if s in tier_diag]
    if not splits:
        print("[CHART] No tier diagnostics available -- skipping.")
        return

    # Collect data for the chart
    tier_names = []
    for split in splits:
        for tier in tier_diag[split]:
            if tier not in tier_names:
                tier_names.append(tier)

    # -- Chart 1: ROC-AUC comparison across tiers (grouped bar) --
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("SAFIRI -- Tier Performance Comparison", fontsize=14, fontweight="bold")

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    tier_labels = {
        "tier1_rule": "T1: Rules",
        "tier2_peer": "T2: Peer Stats",
        "tier3_if": "T3: IF+LOF",
        "tier4_supervised": "T4: XGBoost",
    }

    for ax_idx, split in enumerate(splits):
        ax = axes[ax_idx] if len(splits) > 1 else axes
        data = tier_diag[split]
        tiers = list(data.keys())
        labels = [tier_labels.get(t, t) for t in tiers]

        # ROC-AUC bars
        roc_values = [data[t].get("roc_auc", 0) for t in tiers]
        ap_values = [data[t].get("average_precision", 0) for t in tiers]

        x = np.arange(len(tiers))
        width = 0.35
        bars1 = ax.bar(x - width / 2, roc_values, width, label="ROC-AUC", color=colors[:len(tiers)], alpha=0.85)
        bars2 = ax.bar(x + width / 2, ap_values, width, label="Avg Precision", color=colors[:len(tiers)], alpha=0.45)

        ax.set_title(f"{split.capitalize()}", fontsize=12)
        ax.set_xlabel("Detection Tier")
        ax.set_ylabel("Score" if ax_idx == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
        ax.legend(fontsize=8, loc="upper left")

        # Add value labels on bars
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    chart_path = output_dir / "tier_comparison.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[CHART] Saved tier comparison chart: {chart_path}")

    # -- Chart 2: Recall@k% stacked comparison --
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig2.suptitle("SAFIRI -- Recall at Different Review Budgets", fontsize=14, fontweight="bold")

    for ax_idx, split in enumerate(splits):
        ax = axes2[ax_idx] if len(splits) > 1 else axes2
        data = tier_diag[split]
        tiers = list(data.keys())
        labels = [tier_labels.get(t, t) for t in tiers]

        recall_5 = [data[t].get("recall_at_5pct", 0) for t in tiers]
        recall_10 = [data[t].get("recall_at_10pct", 0) for t in tiers]
        recall_20 = [data[t].get("recall_at_20pct", 0) for t in tiers]

        x = np.arange(len(tiers))
        width = 0.25
        ax.bar(x - width, recall_5, width, label="Recall@5%", color="#4C72B0")
        ax.bar(x, recall_10, width, label="Recall@10%", color="#55A868")
        ax.bar(x + width, recall_20, width, label="Recall@20%", color="#C44E52")

        ax.set_title(f"{split.capitalize()}", fontsize=12)
        ax.set_xlabel("Detection Tier")
        ax.set_ylabel("Recall" if ax_idx == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 0.7)
        ax.legend(fontsize=8, loc="upper left")

        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)

    plt.tight_layout()
    chart_path2 = output_dir / "tier_recall_comparison.png"
    fig2.savefig(chart_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[CHART] Saved recall comparison chart: {chart_path2}")


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 70)
    print("  SAFIRI Hybrid Anomaly Detection Pipeline")
    print("#" * 70)
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Weights:     rule={args.lambda_rule}, peer={args.lambda_peer}, if={args.lambda_if}, supervised={args.lambda_supervised}")
    print(f"  Business boost (alpha): {args.business_boost}")
    print(f"  LOF: {'ON' if not args.no_lof else 'OFF'}  |  Supervised: {'ON' if not args.no_supervised else 'OFF'}  |  Deep Learning: {'ON' if args.use_deep_learning else 'OFF'}")
    print(f"  Flag top: {args.flag_top_percent}%  |  F1-calibrate: {args.calibrate_f1}  |  Random state: {args.random_state}")
    print()

    # -- Load data --
    print("[1/6] Loading data splits...")
    train = load_split(ROOT / args.data_dir, "train")
    valid = load_split(ROOT / args.data_dir, "valid")
    test = load_split(ROOT / args.data_dir, "test")
    print(f"       Train: {len(train):,}  |  Valid: {len(valid):,}  |  Test: {len(test):,}")

    # -- Fit detector --
    print("\n[2/6] Fitting detector on training data...")
    detector = HybridAnomalyDetector(
        flag_top_percent=args.flag_top_percent,
        random_state=args.random_state,
        lambda_rule=args.lambda_rule,
        lambda_peer=args.lambda_peer,
        lambda_if=args.lambda_if,
        lambda_supervised=args.lambda_supervised,
        use_lof=not args.no_lof,
        use_supervised=not args.no_supervised,
        use_deep_learning=args.use_deep_learning,
        business_boost=args.business_boost,
    )
    detector.fit(train)

    # -- Score validation --
    print("\n[3/6] Scoring validation set...")
    threshold_mode = "calibrate_f1" if args.calibrate_f1 else "calibrate"
    valid_scored = detector.score(valid, threshold_mode=threshold_mode)

    if args.calibrate_f1 and "fraud" in valid_scored.columns:
        best_thr, best_f1 = find_best_f1_threshold(valid_scored["fraud"], valid_scored["risk_score"])
        detector.flag_threshold_ = best_thr
        valid_scored = detector.score(valid, threshold_mode="frozen")
        print(f"  F1-calibrated threshold: {best_thr:.6f} (validation F1={best_f1:.4f})")

    # -- Score test --
    print("\n[4/6] Scoring test set...")
    test_scored = detector.score(test, threshold_mode="frozen")

    # -- Build & score synthetic --
    print(f"\n[5/6] Building synthetic evaluation ({args.synthetic_anomalies} anomalies)...")
    synthetic_eval = build_synthetic_evaluation_set(
        valid,
        anomaly_count=args.synthetic_anomalies,
        random_state=args.random_state,
    )
    synthetic_scored = detector.score(synthetic_eval, threshold_mode="frozen")

    # -- Save artifacts --
    print("\n[6/6] Saving artifacts...")
    valid_scored.to_csv(output_dir / "valid_scored.csv", index=False)
    test_scored.to_csv(output_dir / "test_scored.csv", index=False)
    synthetic_scored.to_csv(output_dir / "synthetic_eval_scored.csv", index=False)
    joblib.dump(detector, output_dir / "hybrid_detector.joblib")

    metrics = summarize_metrics(valid_scored, test_scored, synthetic_scored)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_markdown_report(output_dir, detector, metrics, valid_scored, test_scored, synthetic_scored)

    # -- Print tier diagnostics --
    print("\n" + "#" * 70)
    print("  TIER-BY-TIER PERFORMANCE DIAGNOSTICS")
    print("#" * 70)
    _print_tier_table(metrics, "validation", "fraud", valid_scored)
    _print_tier_table(metrics, "test", "fraud", test_scored)

    # -- Generate charts --
    _generate_charts(metrics, output_dir)

    # -- Final summary --
    print("\n" + "#" * 70)
    print("  PIPELINE COMPLETE -- SUMMARY")
    print("#" * 70)
    print(f"  Artifacts saved to: {output_dir}")
    print(f"  Validation threshold: {float(detector.flag_threshold_):.6f}")
    val_f1 = metrics.get("validation_risk_threshold", {}).get("f1", 0)
    test_f1 = metrics.get("test_risk_threshold", {}).get("f1", 0)
    val_auc = metrics.get("validation_risk_ranking", {}).get("roc_auc", 0)
    test_auc = metrics.get("test_risk_ranking", {}).get("roc_auc", 0)
    val_ap = metrics.get("validation_risk_ranking", {}).get("average_precision", 0)
    test_ap = metrics.get("test_risk_ranking", {}).get("average_precision", 0)
    print(f"\n  {'Metric':<20} {'Validation':>12} {'Test':>12}")
    print(f"  {'-' * 44}")
    print(f"  {'ROC-AUC':<20} {val_auc:>12.4f} {test_auc:>12.4f}")
    print(f"  {'Avg Precision':<20} {val_ap:>12.4f} {test_ap:>12.4f}")
    print(f"  {'F1 (threshold)':<20} {val_f1:>12.4f} {test_f1:>12.4f}")
    print(f"\n  Score weights: rule={args.lambda_rule}, peer={args.lambda_peer}, if={args.lambda_if}, supervised={args.lambda_supervised}")
    print(f"  Business boost (alpha): {args.business_boost}")
    print(f"  LOF: {'ON' if not args.no_lof else 'OFF'}  |  Supervised: {'ON' if not args.no_supervised else 'OFF'}  |  Deep Learning: {'ON' if args.use_deep_learning else 'OFF'}")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
