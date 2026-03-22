from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from .data import normalize_columns
from .rules import apply_rule_features, build_explanation
from .utils import compose_key, quantile_rank, robust_mad, safe_ratio, tail_percentile_score


CATEGORY_COLUMNS = [
    "office_id",
    "process_type",
    "import_type",
    "import_use",
    "payment_type",
    "mode_of_transport",
    "declarant_id",
    "importer_id",
    "seller_id",
    "courier_id",
    "hs6_code",
    "country_of_departure",
    "country_of_origin",
    "tax_type",
    "country_of_origin_indicator",
]

PAIR_COLUMNS = {
    "hs6_origin": ["hs6_code", "country_of_origin"],
    "seller_origin": ["seller_id", "country_of_origin"],
    "importer_seller": ["importer_id", "seller_id"],
    "route": ["country_of_departure", "country_of_origin"],
    "tax_indicator": ["tax_type", "country_of_origin_indicator"],
}

PEER_LEVELS = [
    ("hs6_origin_import", ["hs6_code", "country_of_origin", "import_type"]),
    ("hs6_origin", ["hs6_code", "country_of_origin"]),
    ("hs6", ["hs6_code"]),
]

PEER_METRICS = ["valuation_metric", "item_price", "net_mass"]
LOG_METRICS = ["valuation_metric", "item_price", "net_mass", "price_per_kg", "quantity", "total_value"]

# Columns used by the supervised model (same 18 as original code)
SUPERVISED_FEATURE_COLUMNS = [
    "office_id",
    "process_type",
    "import_type",
    "import_use",
    "payment_type",
    "mode_of_transport",
    "declarant_id",
    "importer_id",
    "seller_id",
    "courier_id",
    "hs6_code",
    "country_of_departure",
    "country_of_origin",
    "tax_rate",
    "tax_type",
    "country_of_origin_indicator",
    "net_mass",
    "item_price",
]


@dataclass
class ThresholdConfig:
    flag_top_percent: float = 5.0
    peer_min_count: int = 25


class HybridAnomalyDetector:
    """Interpretable anomaly triage system for customs/shipment data.

    Paradigm: rank records for investigation with limited review resources.
    Four reasons a record is noteworthy:
        (i)    statistical deviation from peers
        (ii)   logic/rule violations
        (iii)  high business risk (exposure)
        (iv)   supervised model prediction (XGBoost trained on historical labels)

    Four output components per record:
        - risk_score:       investigation priority (anomaly evidence × business exposure)
        - anomaly_score:    pure deviation from normal behaviour (peer + unsupervised)
        - confidence_score: evidence reliability (peer group size, tier agreement, data completeness)
        - explanation:      human-readable, tied to peer context or rule violation

    Score composition (matching 3+1 tier spec):
        anomaly_evidence = λ₁·s_rule + λ₂·s_peer + λ₃·s_IF + λ₄·s_supervised
        business_exposure = f(log_total_value, tax_rate)  [no label leakage]
        risk_score = anomaly_evidence × (1 + α·business_exposure)

    Tier independence:
        - Rule flags are NOT fed into Isolation Forest features
        - LOF operates on peer-relative features for local anomaly sensitivity
        - Supervised model uses label-encoded raw features (like the original code)
    """

    def __init__(
        self,
        flag_top_percent: float = 5.0,
        peer_min_count: int = 25,
        random_state: int = 42,
        lambda_rule: float = 0.03,
        lambda_peer: float = 0.04,
        lambda_if: float = 0.03,
        lambda_supervised: float = 0.90,
        use_lof: bool = True,
        lof_weight: float = 0.35,
        business_boost: float = 0.10,
        use_supervised: bool = True,
        use_deep_learning: bool = False,
    ) -> None:
        self.threshold_config = ThresholdConfig(
            flag_top_percent=flag_top_percent,
            peer_min_count=peer_min_count,
        )
        self.random_state = random_state
        self.lambda_rule = lambda_rule
        self.lambda_peer = lambda_peer
        self.lambda_if = lambda_if
        self.lambda_supervised = lambda_supervised
        self.use_lof = use_lof
        self.lof_weight = lof_weight  # within IF tier: weight for LOF vs IF
        self.business_boost = business_boost  # α: how much business exposure amplifies risk
        self.use_supervised = use_supervised
        self.use_deep_learning = use_deep_learning

        self.iforest = IsolationForest(
            n_estimators=160,
            max_samples="auto",
            contamination=0.05,
            random_state=random_state,
            n_jobs=-1,
        )
        if self.use_lof:
            self.lof = LocalOutlierFactor(
                n_neighbors=20,
                contamination=0.05,
                novelty=True,
                n_jobs=-1,
            )

        # Supervised model components (initialized during fit)
        self.supervised_model_ = None
        self.label_encoders_: dict[str, LabelEncoder] = {}
        self.minmax_scalers_: dict[str, MinMaxScaler] = {}
        self.dl_model_ = None  # Deep learning alternative

    # ------------------------------------------------------------------ #
    #  Supervised preprocessing (matching original code approach)          #
    # ------------------------------------------------------------------ #
    def _fit_supervised_encoders(self, base: pd.DataFrame) -> None:
        """Fit LabelEncoders for categorical columns and MinMaxScalers for numerics."""
        categorical_cols = [c for c in SUPERVISED_FEATURE_COLUMNS if c not in ("net_mass", "item_price", "tax_rate")]
        for col in categorical_cols:
            le = LabelEncoder()
            values = base[col].astype(str).fillna("__MISSING__")
            le.fit(values)
            self.label_encoders_[col] = le

        for col in ["net_mass", "item_price"]:
            scaler = MinMaxScaler()
            vals = base[col].fillna(0).to_numpy(dtype=float).reshape(-1, 1)
            scaler.fit(vals)
            self.minmax_scalers_[col] = scaler

    def _encode_supervised_features(self, base: pd.DataFrame) -> np.ndarray:
        """Encode features for the supervised model, matching the original code's approach."""
        encoded_parts = []
        categorical_cols = [c for c in SUPERVISED_FEATURE_COLUMNS if c not in ("net_mass", "item_price", "tax_rate")]
        for col in categorical_cols:
            le = self.label_encoders_[col]
            values = base[col].astype(str).fillna("__MISSING__")
            # Vectorised encoding: build a lookup dict once, map in bulk
            class_to_int = {cls: idx for idx, cls in enumerate(le.classes_)}
            unseen_code = len(le.classes_)
            encoded = values.map(class_to_int).fillna(unseen_code).astype(int).to_numpy()
            encoded_parts.append(encoded.reshape(-1, 1))

        # Tax rate as integer-encoded (like original)
        tax_rate_vals = base["tax_rate"].fillna(0).to_numpy(dtype=float).reshape(-1, 1)
        encoded_parts.append(tax_rate_vals)

        # Numeric columns with MinMaxScaler
        for col in ["net_mass", "item_price"]:
            vals = base[col].fillna(0).to_numpy(dtype=float).reshape(-1, 1)
            scaled = self.minmax_scalers_[col].transform(vals)
            encoded_parts.append(scaled)

        return np.hstack(encoded_parts)

    def _fit_supervised_model(self, base: pd.DataFrame) -> None:
        """Train XGBoost (and optionally MLP) on the training data with oversampling."""
        from xgboost import XGBClassifier
        from imblearn.over_sampling import RandomOverSampler

        labels = base["fraud"].fillna(0).astype(int).to_numpy()
        if len(np.unique(labels)) < 2:
            print("WARNING: Only one class in training labels, skipping supervised model.")
            self.use_supervised = False
            return

        self._fit_supervised_encoders(base)
        X_encoded = self._encode_supervised_features(base)

        # Random oversampling to balance classes (like original code)
        ros = RandomOverSampler(random_state=self.random_state)
        X_resampled, y_resampled = ros.fit_resample(X_encoded, labels)

        # XGBoost with tuned hyperparameters (from original code)
        self.supervised_model_ = XGBClassifier(
            base_score=0.5,
            booster="gbtree",
            colsample_bylevel=1,
            colsample_bynode=1,
            colsample_bytree=0.6,
            gamma=1,
            importance_type="gain",
            learning_rate=0.5994,
            max_delta_step=0,
            max_depth=4,
            min_child_weight=2,
            n_estimators=424,
            n_jobs=-1,
            num_parallel_tree=1,
            random_state=self.random_state,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            subsample=1.0,
            tree_method="exact",
            verbosity=0,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
        )
        self.supervised_model_.fit(X_resampled, y_resampled)

        # Optionally train deep learning model (MLP)
        if self.use_deep_learning:
            from sklearn.neural_network import MLPClassifier
            self.dl_model_ = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                batch_size=256,
                learning_rate="adaptive",
                learning_rate_init=0.001,
                max_iter=200,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.15,
                verbose=False,
            )
            self.dl_model_.fit(X_resampled, y_resampled)

    def _supervised_predict_proba(self, base: pd.DataFrame) -> np.ndarray:
        """Get fraud probability from the supervised model."""
        X_encoded = self._encode_supervised_features(base)

        if self.use_deep_learning and self.dl_model_ is not None:
            # Blend XGBoost and DL predictions
            xgb_proba = self.supervised_model_.predict_proba(X_encoded)[:, 1]
            dl_proba = self.dl_model_.predict_proba(X_encoded)[:, 1]
            return 0.6 * xgb_proba + 0.4 * dl_proba
        else:
            return self.supervised_model_.predict_proba(X_encoded)[:, 1]

    # ------------------------------------------------------------------ #
    #  Core fit / score                                                    #
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame) -> "HybridAnomalyDetector":
        t_total = time.perf_counter()
        print("\n" + "=" * 70)
        print("  SAFIRI Hybrid Anomaly Detector -- FIT (Training)")
        print("=" * 70)

        # -- Data preparation --
        t0 = time.perf_counter()
        base = self._prepare_base_frame(df)
        self.train_size_ = len(base)
        print(f"\n[DATA]  Prepared {self.train_size_:,} training records  ({time.perf_counter() - t0:.2f}s)")

        # -- Statistics & frequency maps --
        t0 = time.perf_counter()
        self.global_stats_ = self._build_global_stats(base)
        self.category_freq_maps_, self.category_default_freq_ = self._fit_frequency_maps(base, CATEGORY_COLUMNS)
        self.pair_freq_maps_, self.pair_default_freq_ = self._fit_pair_frequency_maps(base)
        self.peer_stats_, self.peer_reference_arrays_ = self._fit_peer_stats(base)
        self.importer_baseline_map_ = self._fit_group_median_map(base, ["importer_id"], "valuation_metric")
        self.seller_origin_baseline_map_ = self._fit_group_median_map(
            base, ["seller_id", "country_of_origin"], "valuation_metric"
        )
        self.round_price_freq_map_, self.round_price_default_freq_ = self._fit_round_price_map(base)
        print(f"[STATS] Built global stats & frequency maps  ({time.perf_counter() - t0:.2f}s)")

        # -- Feature engineering (includes Tier 1 rules + Tier 2 peer features) --
        t0 = time.perf_counter()
        feature_frame, lof_frame, enriched = self._build_feature_frame(base)
        self.feature_columns_ = list(feature_frame.columns)
        self.feature_fill_values_ = feature_frame.median(numeric_only=True)
        train_matrix = feature_frame.fillna(self.feature_fill_values_)
        n_rule_fires = int(enriched["rule_score_raw"].gt(0).sum()) if "rule_score_raw" in enriched.columns else 0
        print(f"\n{'-' * 50}")
        print(f"[TIER 1 -- Rules]  Feature engineering complete  ({time.perf_counter() - t0:.2f}s)")
        print(f"  • {len(self.feature_columns_)} features extracted")
        print(f"  • Rule fires on training data: {n_rule_fires:,} / {self.train_size_:,} records")

        # -- Tier 2: Peer stats already computed in _build_feature_frame --
        peer_levels = enriched["peer_level"].value_counts().to_dict() if "peer_level" in enriched.columns else {}
        print(f"\n[TIER 2 -- Peer Statistics]")
        print(f"  • Peer level distribution: {peer_levels}")

        # -- Tier 3: Isolation Forest + LOF --
        t0 = time.perf_counter()
        self.iforest.fit(train_matrix)
        t_if = time.perf_counter() - t0
        print(f"\n[TIER 3 -- Unsupervised (IF + LOF)]")
        print(f"  • Isolation Forest fitted on {train_matrix.shape[1]} features  ({t_if:.2f}s)")

        if self.use_lof:
            t0 = time.perf_counter()
            self.lof_columns_ = list(lof_frame.columns)
            self.lof_fill_values_ = lof_frame.median(numeric_only=True)
            lof_matrix = lof_frame.fillna(self.lof_fill_values_)
            self.lof.fit(lof_matrix)
            print(f"  • LOF fitted on {lof_matrix.shape[1]} peer features  ({time.perf_counter() - t0:.2f}s)")
        else:
            print(f"  • LOF: disabled")

        # -- Tier 4: Supervised XGBoost --
        if self.use_supervised:
            t0 = time.perf_counter()
            self._fit_supervised_model(base)
            t_sup = time.perf_counter() - t0
            if self.supervised_model_ is not None:
                fraud_count = int(base["fraud"].fillna(0).sum())
                print(f"\n[TIER 4 -- Supervised XGBoost]")
                print(f"  • Training labels: {fraud_count:,} fraud / {self.train_size_ - fraud_count:,} legit")
                print(f"  • XGBoost: {self.supervised_model_.n_estimators} trees, max_depth={self.supervised_model_.max_depth}, lr={self.supervised_model_.learning_rate:.4f}")
                print(f"  • RandomOverSampler applied for class balance")
                if self.use_deep_learning and hasattr(self, 'dl_model_') and self.dl_model_ is not None:
                    print(f"  • MLP blend enabled (60/40 XGB/MLP)")
                print(f"  • Fit time: {t_sup:.2f}s")
            else:
                print(f"\n[TIER 4 -- Supervised] Skipped (single-class labels)")
        else:
            print(f"\n[TIER 4 -- Supervised] Disabled")

        # -- Build reference distributions --
        t0 = time.perf_counter()
        isolation_raw = -self.iforest.score_samples(train_matrix)
        lof_raw = None
        if self.use_lof:
            lof_matrix = lof_frame.fillna(self.lof_fill_values_)
            lof_raw = -self.lof.score_samples(lof_matrix)

        supervised_proba = None
        if self.use_supervised and self.supervised_model_ is not None:
            supervised_proba = self._supervised_predict_proba(base)

        component_frame = self._compose_component_frame(
            enriched, isolation_raw, lof_raw, supervised_proba, fit_mode=True
        )
        self.component_reference_ = {
            name: np.sort(component_frame[name].to_numpy(dtype=float))
            for name in [
                "peer_score_raw",
                "unsupervised_score_raw",
                "business_exposure_raw",
            ]
        }
        self.flag_threshold_ = None
        print(f"\n[COMPOSE] Built reference distributions  ({time.perf_counter() - t0:.2f}s)")

        print(f"\n{'=' * 70}")
        print(f"  FIT COMPLETE -- Total time: {time.perf_counter() - t_total:.2f}s")
        print(f"{'=' * 70}\n")
        return self

    def score(self, df: pd.DataFrame, threshold_mode: str = "frozen") -> pd.DataFrame:
        t_total = time.perf_counter()
        n_records = len(df)
        print(f"\n[SCORE] Scoring {n_records:,} records  (threshold_mode={threshold_mode})")

        # -- Tier 1 + 2: Feature engineering (rules + peer) --
        t0 = time.perf_counter()
        base = self._prepare_base_frame(df)
        feature_frame, lof_frame, enriched = self._build_feature_frame(base)
        matrix = feature_frame.reindex(columns=self.feature_columns_).fillna(self.feature_fill_values_)
        print(f"  [Tier 1+2] Feature engineering  ({time.perf_counter() - t0:.2f}s)")

        # -- Tier 3: IF + LOF scoring --
        t0 = time.perf_counter()
        isolation_raw = -self.iforest.score_samples(matrix)
        lof_raw = None
        if self.use_lof:
            lof_matrix = lof_frame.reindex(columns=self.lof_columns_).fillna(self.lof_fill_values_)
            lof_raw = -self.lof.score_samples(lof_matrix)
        print(f"  [Tier 3]   IF{'+LOF' if self.use_lof else ''} scoring  ({time.perf_counter() - t0:.2f}s)")

        # -- Tier 4: Supervised prediction --
        t0 = time.perf_counter()
        supervised_proba = None
        if self.use_supervised and self.supervised_model_ is not None:
            supervised_proba = self._supervised_predict_proba(base)
            print(f"  [Tier 4]   XGBoost predict_proba  ({time.perf_counter() - t0:.2f}s)")
        else:
            print(f"  [Tier 4]   Skipped (disabled or not fitted)")

        # -- Compose final scores --
        t0 = time.perf_counter()
        scored = self._compose_component_frame(
            enriched, isolation_raw, lof_raw, supervised_proba, fit_mode=False
        )
        print(f"  [Compose]  Score composition  ({time.perf_counter() - t0:.2f}s)")

        score_col = "risk_score"
        if threshold_mode == "calibrate" or self.flag_threshold_ is None:
            self.flag_threshold_ = float(
                np.quantile(scored[score_col], 1 - (self.threshold_config.flag_top_percent / 100.0))
            )
        elif threshold_mode == "calibrate_f1":
            # Label-aware calibration -- requires 'fraud' or 'synthetic_is_anomaly' column
            from .evaluation import find_best_f1_threshold
            label_col = "synthetic_is_anomaly" if "synthetic_is_anomaly" in scored.columns else "fraud"
            if label_col in scored.columns:
                best_thr, _ = find_best_f1_threshold(scored[label_col], scored[score_col])
                self.flag_threshold_ = best_thr
            else:
                self.flag_threshold_ = float(
                    np.quantile(scored[score_col], 1 - (self.threshold_config.flag_top_percent / 100.0))
                )
        elif threshold_mode != "frozen":
            raise ValueError("threshold_mode must be 'calibrate', 'calibrate_f1', or 'frozen'")

        scored["flag_threshold"] = float(self.flag_threshold_)
        scored["flagged"] = scored[score_col] >= float(self.flag_threshold_)
        scored["anomaly_rank"] = scored[score_col].rank(method="first", ascending=False).astype(int)
        scored["risk_tier"] = scored.apply(self._risk_tier, axis=1)
        scored["explanation"] = scored.apply(build_explanation, axis=1)

        n_flagged = int(scored["flagged"].sum())
        print(f"  [Result]   Flagged: {n_flagged:,} / {n_records:,}  ({100 * n_flagged / max(n_records, 1):.1f}%)  threshold={float(self.flag_threshold_):.6f}  ({time.perf_counter() - t_total:.2f}s total)")
        return scored.sort_values(score_col, ascending=False).reset_index(drop=True)

    def _risk_tier(self, row: pd.Series) -> str:
        if not bool(row["flagged"]):
            return "Normal"
        score = float(row["risk_score"])
        if score >= max(float(self.flag_threshold_), 0.88):
            return "High Risk"
        if score >= max(float(self.flag_threshold_) * 0.92, 0.68):
            return "Medium Risk"
        return "Review"

    def _prepare_base_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        base = normalize_columns(df).copy()
        for column in CATEGORY_COLUMNS:
            base[column] = base[column].astype("string").fillna("__MISSING__")
        for column in ["tax_rate", "net_mass", "item_price", "fraud", "critical_fraud"]:
            base[column] = pd.to_numeric(base[column], errors="coerce")
        for column in ["quantity", "unit_price", "total_value", "revenue_at_risk"]:
            if column not in base.columns:
                base[column] = np.nan
            base[column] = pd.to_numeric(base[column], errors="coerce")

        base["date"] = pd.to_datetime(base["date"], errors="coerce")
        base["month"] = base["date"].dt.month.fillna(0).astype(int)
        base["day_of_week"] = base["date"].dt.dayofweek.fillna(0).astype(int)
        base["day_of_month"] = base["date"].dt.day.fillna(0).astype(int)
        base["is_month_end"] = base["date"].dt.is_month_end.fillna(False).astype(int)
        base["has_seller_id"] = base["seller_id"].ne("__MISSING__").astype(int)
        base["has_courier_id"] = base["courier_id"].ne("__MISSING__").astype(int)

        base["price_per_kg"] = np.where(base["net_mass"] > 0, base["item_price"] / base["net_mass"], np.nan)
        base["unit_value"] = np.where(base["quantity"] > 0, base["total_value"] / base["quantity"], np.nan)
        base["valuation_metric"] = base["unit_value"].where(base["unit_value"].notna(), base["price_per_kg"])
        base["round_price_bucket"] = (base["item_price"].fillna(0) / 1000.0).round().astype(int) * 1000

        for metric in LOG_METRICS:
            safe_metric = base[metric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            base[f"log_{metric}"] = np.log1p(np.clip(safe_metric, 0.0, None))
        return base

    def _build_global_stats(self, base: pd.DataFrame) -> dict[str, float]:
        stats: dict[str, float] = {"peer_count": float(len(base))}
        for metric in PEER_METRICS:
            log_metric = f"log_{metric}"
            stats[f"{metric}_median"] = float(base[metric].median(skipna=True))
            stats[f"{log_metric}_median"] = float(base[log_metric].median(skipna=True))
            stats[f"{log_metric}_mad"] = float(robust_mad(base[log_metric]))
        stats["price_per_kg_p01"] = float(base["price_per_kg"].quantile(0.01))
        stats["price_per_kg_p99"] = float(base["price_per_kg"].quantile(0.99))
        round_freq = base["round_price_bucket"].value_counts(normalize=True)
        stats["round_price_high_freq"] = float(round_freq.quantile(0.95)) if not round_freq.empty else 0.0
        stats["tax_rate_p90"] = float(base["tax_rate"].quantile(0.90))
        return stats

    def _fit_frequency_maps(
        self, base: pd.DataFrame, columns: list[str]
    ) -> tuple[dict[str, pd.Series], dict[str, float]]:
        freq_maps: dict[str, pd.Series] = {}
        defaults: dict[str, float] = {}
        for column in columns:
            freq = base[column].value_counts(normalize=True)
            freq_maps[column] = freq
            defaults[column] = 1.0 / (len(base) + len(freq))
        return freq_maps, defaults

    def _fit_pair_frequency_maps(
        self, base: pd.DataFrame
    ) -> tuple[dict[str, pd.Series], dict[str, float]]:
        freq_maps: dict[str, pd.Series] = {}
        defaults: dict[str, float] = {}
        for name, columns in PAIR_COLUMNS.items():
            key = compose_key(base, columns)
            freq = key.value_counts(normalize=True)
            freq_maps[name] = freq
            defaults[name] = 1.0 / (len(base) + len(freq))
        return freq_maps, defaults

    def _fit_peer_stats(
        self, base: pd.DataFrame
    ) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, np.ndarray]]]:
        peer_stats: dict[str, pd.DataFrame] = {}
        peer_refs: dict[str, dict[str, np.ndarray]] = {}
        for level_name, group_columns in PEER_LEVELS:
            aggregations: dict[str, pd.NamedAgg] = {
                "peer_count": pd.NamedAgg(column="declaration_id", aggfunc="size")
            }
            for metric in PEER_METRICS:
                aggregations[f"{metric}_median"] = pd.NamedAgg(column=metric, aggfunc="median")
                aggregations[f"log_{metric}_median"] = pd.NamedAgg(column=f"log_{metric}", aggfunc="median")
                aggregations[f"log_{metric}_mad"] = pd.NamedAgg(
                    column=f"log_{metric}", aggfunc=robust_mad
                )
            peer_stats[level_name] = (
                base.groupby(group_columns, dropna=False).agg(**aggregations).reset_index()
            )

            ref_map: dict[str, np.ndarray] = {}
            grouped = base[group_columns + ["valuation_metric"]].dropna(subset=["valuation_metric"])
            for keys, grp in grouped.groupby(group_columns, dropna=False):
                key_tuple = keys if isinstance(keys, tuple) else (keys,)
                ref_map["|".join(str(value) for value in key_tuple)] = np.sort(
                    grp["valuation_metric"].to_numpy(dtype=float)
                )
            peer_refs[level_name] = ref_map
        return peer_stats, peer_refs

    def _fit_group_median_map(self, base: pd.DataFrame, columns: list[str], value_col: str) -> pd.Series:
        key = compose_key(base, columns)
        return pd.DataFrame({"key": key, value_col: base[value_col]}).groupby("key", dropna=False)[
            value_col
        ].median()

    def _fit_round_price_map(self, base: pd.DataFrame) -> tuple[pd.Series, float]:
        freq = base["round_price_bucket"].value_counts(normalize=True)
        default = 1.0 / (len(base) + len(freq)) if len(base) else 0.0
        return freq, default

    def _build_feature_frame(
        self, base: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Build feature frames for IF, LOF, and the full enriched frame.

        Returns:
            (if_features, lof_features, enriched_frame)

        Key change: rule flags are NOT included in IF/LOF features to preserve
        tier independence.
        """
        enriched = base.copy()
        enriched = self._apply_frequency_features(enriched)
        enriched = self._apply_pair_frequency_features(enriched)
        enriched = self._apply_history_features(enriched)
        enriched = self._apply_peer_features(enriched)
        enriched = apply_rule_features(enriched, self.global_stats_)

        # IF features: statistical + frequency features only (NO rule flags)
        if_feature_columns = [
            "log_item_price",
            "log_net_mass",
            "log_valuation_metric",
            "log_price_per_kg",
            "log_quantity",
            "log_total_value",
            "tax_rate",
            "month",
            "day_of_week",
            "day_of_month",
            "is_month_end",
            "has_seller_id",
            "has_courier_id",
            "valuation_metric_peer_z",
            "item_price_peer_z",
            "net_mass_peer_z",
            "valuation_peer_tail",
            "importer_valuation_z",
            "seller_origin_valuation_z",
            "freq_office_id",
            "freq_process_type",
            "freq_import_type",
            "freq_import_use",
            "freq_payment_type",
            "freq_mode_of_transport",
            "freq_declarant_id",
            "freq_importer_id",
            "freq_seller_id",
            "freq_courier_id",
            "freq_hs6_code",
            "freq_country_of_departure",
            "freq_country_of_origin",
            "freq_tax_type",
            "freq_country_of_origin_indicator",
            "pair_freq_hs6_origin",
            "pair_freq_seller_origin",
            "pair_freq_importer_seller",
            "pair_freq_route",
            "pair_freq_tax_indicator",
            "round_price_freq",
            "peer_count",
        ]

        # LOF features: peer-relative features only (for local anomaly detection)
        lof_feature_columns = [
            "valuation_metric_peer_z",
            "item_price_peer_z",
            "net_mass_peer_z",
            "valuation_peer_tail",
            "importer_valuation_z",
            "seller_origin_valuation_z",
            "log_valuation_metric",
            "log_price_per_kg",
        ]

        return (
            enriched[if_feature_columns].copy(),
            enriched[lof_feature_columns].copy(),
            enriched,
        )

    def _apply_frequency_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy()
        for column in CATEGORY_COLUMNS:
            default = self.category_default_freq_[column]
            enriched[f"freq_{column}"] = enriched[column].map(self.category_freq_maps_[column]).fillna(
                default
            )
        return enriched

    def _apply_pair_frequency_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy()
        for name, columns in PAIR_COLUMNS.items():
            default = self.pair_default_freq_[name]
            key = compose_key(enriched, columns)
            enriched[f"pair_freq_{name}"] = key.map(self.pair_freq_maps_[name]).fillna(default)
        enriched["max_pair_frequency"] = enriched[
            ["pair_freq_hs6_origin", "pair_freq_seller_origin", "pair_freq_importer_seller", "pair_freq_route"]
        ].min(axis=1)
        enriched["round_price_freq"] = enriched["round_price_bucket"].map(self.round_price_freq_map_).fillna(
            self.round_price_default_freq_
        )
        return enriched

    def _apply_history_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy()
        importer_key = compose_key(enriched, ["importer_id"])
        seller_origin_key = compose_key(enriched, ["seller_id", "country_of_origin"])
        enriched["importer_valuation_median"] = importer_key.map(self.importer_baseline_map_)
        enriched["seller_origin_valuation_median"] = seller_origin_key.map(self.seller_origin_baseline_map_)

        importer_ratio = safe_ratio(enriched["valuation_metric"], enriched["importer_valuation_median"])
        seller_ratio = safe_ratio(enriched["valuation_metric"], enriched["seller_origin_valuation_median"])
        enriched["importer_valuation_z"] = np.abs(np.log(importer_ratio.clip(lower=1e-6).fillna(1.0)))
        enriched["seller_origin_valuation_z"] = np.abs(np.log(seller_ratio.clip(lower=1e-6).fillna(1.0)))
        return enriched

    def _apply_peer_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy()
        for level_name, group_columns in PEER_LEVELS:
            stats = self.peer_stats_[level_name].rename(
                columns={
                    column: f"{column}__{level_name}"
                    for column in self.peer_stats_[level_name].columns
                    if column not in group_columns
                }
            )
            enriched = enriched.merge(stats, on=group_columns, how="left")

        enriched["peer_level"] = "global"
        enriched["peer_count"] = self.global_stats_["peer_count"]
        for metric in PEER_METRICS:
            enriched[f"{metric}_peer_median"] = self.global_stats_[f"{metric}_median"]
            enriched[f"log_{metric}_peer_median"] = self.global_stats_[f"log_{metric}_median"]
            enriched[f"log_{metric}_peer_mad"] = self.global_stats_[f"log_{metric}_mad"]

        for level_name, _ in PEER_LEVELS:
            eligible = enriched["peer_level"].eq("global") & enriched[
                f"peer_count__{level_name}"
            ].fillna(0).ge(self.threshold_config.peer_min_count)
            enriched.loc[eligible, "peer_level"] = level_name
            enriched.loc[eligible, "peer_count"] = enriched.loc[eligible, f"peer_count__{level_name}"]
            for metric in PEER_METRICS:
                enriched.loc[eligible, f"{metric}_peer_median"] = enriched.loc[
                    eligible, f"{metric}_median__{level_name}"
                ]
                enriched.loc[eligible, f"log_{metric}_peer_median"] = enriched.loc[
                    eligible, f"log_{metric}_median__{level_name}"
                ]
                enriched.loc[eligible, f"log_{metric}_peer_mad"] = enriched.loc[
                    eligible, f"log_{metric}_mad__{level_name}"
                ]

        for metric in PEER_METRICS:
            scale = (enriched[f"log_{metric}_peer_mad"] * 1.4826).clip(lower=0.1)
            enriched[f"{metric}_peer_z"] = (
                (enriched[f"log_{metric}"] - enriched[f"log_{metric}_peer_median"]).abs() / scale
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            enriched[f"{metric}_peer_ratio"] = safe_ratio(
                enriched[metric], enriched[f"{metric}_peer_median"]
            )

        enriched["valuation_peer_percentile"] = self._lookup_peer_percentiles(enriched)
        enriched["valuation_peer_tail"] = tail_percentile_score(enriched["valuation_peer_percentile"])
        enriched["valuation_peer_count"] = enriched["peer_count"]
        return enriched

    def _lookup_peer_percentiles(self, frame: pd.DataFrame) -> pd.Series:
        global_ref = np.sort(frame["valuation_metric"].dropna().to_numpy(dtype=float))
        percentiles = pd.Series(np.nan, index=frame.index, dtype=float)
        for level_name, columns in PEER_LEVELS:
            mask = frame["peer_level"].eq(level_name)
            if not mask.any():
                continue
            keys = compose_key(frame.loc[mask], columns)
            values = frame.loc[mask, "valuation_metric"].to_numpy(dtype=float)
            refs = self.peer_reference_arrays_[level_name]
            pct_values = []
            for key, value in zip(keys, values):
                reference = refs.get(key, global_ref)
                pct_values.append(float(quantile_rank(reference, [value])[0]))
            percentiles.loc[mask] = pct_values
        global_mask = percentiles.isna()
        if global_mask.any():
            percentiles.loc[global_mask] = quantile_rank(
                global_ref, frame.loc[global_mask, "valuation_metric"]
            )
        return percentiles.fillna(0.5)

    def _compose_component_frame(
        self,
        enriched: pd.DataFrame,
        isolation_raw: np.ndarray,
        lof_raw: np.ndarray | None,
        supervised_proba: np.ndarray | None,
        fit_mode: bool,
    ) -> pd.DataFrame:
        scored = enriched.copy()

        # --- Tier 2: Peer score (contextual + rarity merged) ---
        scored["peer_score_raw"] = (
            0.35 * scored["valuation_metric_peer_z"]
            + 0.08 * scored["item_price_peer_z"]
            + 0.04 * scored["net_mass_peer_z"]
            + 0.18 * scored["valuation_peer_tail"]
            + 0.08 * scored["importer_valuation_z"]
            + 0.08 * scored["seller_origin_valuation_z"]
            # Rarity merged into peer score
            + 0.19 * np.maximum.reduce(
                [
                    -np.log10(scored["pair_freq_hs6_origin"].clip(lower=1e-12)) / 12.0,
                    -np.log10(scored["pair_freq_seller_origin"].clip(lower=1e-12)) / 12.0,
                    -np.log10(scored["pair_freq_importer_seller"].clip(lower=1e-12)) / 12.0,
                    -np.log10(scored["pair_freq_route"].clip(lower=1e-12)) / 12.0,
                ]
            ).clip(0.0, 1.0)
        )

        # --- Tier 3: Unsupervised score (IF + LOF combined) ---
        scored["isolation_score_raw"] = isolation_raw
        if lof_raw is not None:
            scored["lof_score_raw"] = lof_raw
            scored["unsupervised_score_raw"] = (
                (1.0 - self.lof_weight) * isolation_raw
                + self.lof_weight * lof_raw
            )
        else:
            scored["lof_score_raw"] = 0.0
            scored["unsupervised_score_raw"] = isolation_raw

        # --- Tier 4: Supervised score ---
        if supervised_proba is not None:
            scored["supervised_score_raw"] = supervised_proba
        else:
            scored["supervised_score_raw"] = 0.0

        # --- Business exposure (no label leakage) ---
        log_value = np.log1p(
            (scored["item_price"].fillna(0).clip(lower=0) * scored["net_mass"].fillna(0).clip(lower=0))
        )
        tax_rate_norm = scored["tax_rate"].fillna(0).clip(lower=0)
        scored["business_exposure_raw"] = (
            0.65 * (log_value / max(log_value.max(), 1e-6))
            + 0.35 * (tax_rate_norm / max(tax_rate_norm.max(), 1e-6))
        ).clip(0.0, 1.0)

        if fit_mode:
            return scored

        # Quantile-rank the raw scores against training distribution
        scored["peer_score"] = quantile_rank(
            self.component_reference_["peer_score_raw"], scored["peer_score_raw"]
        )
        scored["isolation_score"] = quantile_rank(
            self.component_reference_["unsupervised_score_raw"], scored["unsupervised_score_raw"]
        )
        scored["rule_score"] = scored["rule_score_raw"].clip(0.0, 1.0)
        scored["business_exposure"] = quantile_rank(
            self.component_reference_["business_exposure_raw"], scored["business_exposure_raw"]
        )

        # Supervised score: already a probability [0, 1], no need to quantile-rank
        scored["supervised_score"] = scored["supervised_score_raw"].clip(0.0, 1.0)

        # Rarity score kept for explanation purposes
        scored["rarity_score"] = np.maximum.reduce(
            [
                -np.log10(scored["pair_freq_hs6_origin"].clip(lower=1e-12)) / 12.0,
                -np.log10(scored["pair_freq_seller_origin"].clip(lower=1e-12)) / 12.0,
                -np.log10(scored["pair_freq_importer_seller"].clip(lower=1e-12)) / 12.0,
                -np.log10(scored["pair_freq_route"].clip(lower=1e-12)) / 12.0,
            ]
        ).clip(0.0, 1.0)

        # ========== OUTPUT COMPONENT 1: anomaly_score ==========
        # Pure deviation from normal behaviour (statistical + unsupervised evidence).
        # Does NOT include business context -- purely "how unusual is this record?"
        scored["anomaly_score"] = (
            0.60 * scored["peer_score"]
            + 0.40 * scored["isolation_score"]
        )

        # ========== OUTPUT COMPONENT 2: risk_score ==========
        # Investigation priority = anomaly evidence × (1 + α·business_exposure)
        # anomaly_evidence now combines all FOUR tiers:
        #   - Tier 1: rule violations
        #   - Tier 2: peer deviation
        #   - Tier 3: unsupervised IF+LOF
        #   - Tier 4: supervised XGBoost prediction
        # business_exposure amplifies risk for high-value / high-tax shipments
        # WITHOUT introducing label leakage.
        if self.use_supervised and supervised_proba is not None:
            anomaly_evidence = (
                self.lambda_rule * scored["rule_score"]
                + self.lambda_peer * scored["peer_score"]
                + self.lambda_if * scored["isolation_score"]
                + self.lambda_supervised * scored["supervised_score"]
            )
        else:
            # Fallback to 3-tier when supervised is disabled
            total_weight = self.lambda_rule + self.lambda_peer + self.lambda_if
            anomaly_evidence = (
                (self.lambda_rule / total_weight) * scored["rule_score"]
                + (self.lambda_peer / total_weight) * scored["peer_score"]
                + (self.lambda_if / total_weight) * scored["isolation_score"]
            )
        scored["anomaly_evidence"] = anomaly_evidence
        scored["risk_score"] = (
            anomaly_evidence * (1.0 + self.business_boost * scored["business_exposure"])
        ).clip(0.0, 1.0)

        # ========== OUTPUT COMPONENT 3: confidence_score ==========
        # Evidence reliability: how trustworthy is the anomaly signal?
        # (a) Peer group size -- larger groups give more reliable z-scores
        peer_confidence = np.clip(
            np.log1p(scored["peer_count"]) / np.log1p(max(self.train_size_, 2)), 0.0, 1.0
        )
        # (b) Peer level specificity -- finer groups are more informative
        peer_level_weight = scored["peer_level"].map(
            {"hs6_origin_import": 1.0, "hs6_origin": 0.75, "hs6": 0.50, "global": 0.20}
        ).fillna(0.20)
        # (c) Tier agreement -- multiple tiers flagging the same record
        if self.use_supervised and supervised_proba is not None:
            agreement = (
                scored[["peer_score", "isolation_score"]].gt(0.70).sum(axis=1)
                + scored["rule_score"].gt(0.0).astype(int)
                + scored["supervised_score"].gt(0.50).astype(int)
            ) / 4.0
        else:
            agreement = (
                scored[["peer_score", "isolation_score"]].gt(0.70).sum(axis=1)
                + scored["rule_score"].gt(0.0).astype(int)
            ) / 3.0
        # (d) Data completeness -- key arithmetic fields present
        arithmetic_presence = scored[["net_mass", "item_price"]].gt(0).all(axis=1).astype(float)

        scored["confidence_score"] = np.clip(
            0.30 * peer_confidence
            + 0.25 * peer_level_weight
            + 0.30 * agreement
            + 0.15 * arithmetic_presence,
            0.0,
            1.0,
        )

        return scored
