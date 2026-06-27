"""Capability 7 -- Feature Warnings & Recommendations.

The "what should I do about these features" layer. It flags columns that add no
signal (constant / quasi-constant), columns that are duplicates of each other
(exact copies or near-perfect correlation), features that look *too* predictive of
the target (likely leakage), and then gives concrete, justified encoding and
scaling recommendations so the output plugs straight into a modelling pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .base import Insight, safe_float, sort_insights
from .correlation import correlation_ratio, cramers_v
from .utils import infer_semantic_types

QUASI_CONSTANT_DOMINANCE = 0.99
DUPLICATE_CORR = 0.98
DUPLICATE_CRAMERS = 0.99
LEAKAGE_ASSOCIATION = 0.95
SCALE_RATIO_TRIGGER = 10.0
MAX_DUP_CATEGORICAL = 30


@dataclass
class FeatureFlag:
    feature: str
    issue: str
    detail: str
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "issue": self.issue, "detail": self.detail, "recommendation": self.recommendation}


@dataclass
class DuplicateGroup:
    features: list[str]
    method: str
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {"features": self.features, "method": self.method, "similarity": round(self.similarity, 4)}


@dataclass
class LeakageWarning:
    feature: str
    association: float
    measure: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "association": round(self.association, 4), "measure": self.measure, "reason": self.reason}


@dataclass
class EncodingRec:
    feature: str
    cardinality: int
    strategy: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "cardinality": self.cardinality, "strategy": self.strategy, "reason": self.reason}


@dataclass
class ScalingRec:
    needed: bool
    method: str
    reason: str
    columns_needing_transform: list[str] = field(default_factory=list)
    sensitive_models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "needed": self.needed,
            "method": self.method,
            "reason": self.reason,
            "columns_needing_transform": self.columns_needing_transform,
            "sensitive_models": self.sensitive_models,
        }


@dataclass
class RecommendationResult:
    constant_features: list[FeatureFlag]
    duplicate_groups: list[DuplicateGroup]
    leakage_warnings: list[LeakageWarning]
    encoding: list[EncodingRec]
    scaling: ScalingRec
    insights: list[Insight]

    def to_dict(self) -> dict[str, Any]:
        return {
            "constant_features": [f.to_dict() for f in self.constant_features],
            "duplicate_groups": [g.to_dict() for g in self.duplicate_groups],
            "leakage_warnings": [w.to_dict() for w in self.leakage_warnings],
            "encoding": [e.to_dict() for e in self.encoding],
            "scaling": self.scaling.to_dict(),
            "insights": [i.to_dict() for i in self.insights],
        }


class FeatureAdvisor:
    """Generate feature-quality warnings and preprocessing recommendations."""

    def analyze(self, df: pd.DataFrame, target_column: str | None = None) -> RecommendationResult:
        types = infer_semantic_types(df)
        feature_cols = [c for c in df.columns if c != target_column]

        constant = self._constant_features(df, feature_cols)
        duplicates = self._duplicate_features(df, types, target_column)
        leakage = self._leakage_warnings(df, types, target_column) if target_column else []
        encoding = self._encoding_recs(df, types, target_column)
        scaling = self._scaling_rec(df, types, target_column)

        insights = self._build_insights(constant, duplicates, leakage, encoding, scaling)
        return RecommendationResult(
            constant_features=constant,
            duplicate_groups=duplicates,
            leakage_warnings=leakage,
            encoding=encoding,
            scaling=scaling,
            insights=sort_insights(insights),
        )

    # --------------------------------------------------------------- constants
    def _constant_features(self, df: pd.DataFrame, feature_cols: list[str]) -> list[FeatureFlag]:
        flags: list[FeatureFlag] = []
        n = len(df)
        for col in feature_cols:
            series = df[col]
            non_null = series.dropna()
            n_unique = int(non_null.nunique())
            if n_unique <= 1:
                flags.append(
                    FeatureFlag(
                        feature=col,
                        issue="constant",
                        detail=f"Only {n_unique} distinct value across {n} rows.",
                        recommendation="Drop -- a constant feature carries zero information.",
                    )
                )
                continue
            if not non_null.empty:
                dominance = non_null.value_counts(normalize=True).iloc[0]
                if dominance >= QUASI_CONSTANT_DOMINANCE:
                    flags.append(
                        FeatureFlag(
                            feature=col,
                            issue="quasi_constant",
                            detail=f"One value covers {dominance * 100:.1f}% of rows (near-zero variance).",
                            recommendation="Consider dropping; keep only if the rare values are known to be important.",
                        )
                    )
        return flags

    # -------------------------------------------------------------- duplicates
    def _duplicate_features(
        self,
        df: pd.DataFrame,
        types: Any,
        target_column: str | None,
    ) -> list[DuplicateGroup]:
        groups: list[DuplicateGroup] = []
        feature_cols = [c for c in df.columns if c != target_column]

        # 1) Exact duplicates via a content hash of each column.
        signatures: dict[int, list[str]] = {}
        for col in feature_cols:
            try:
                sig = int(pd.util.hash_pandas_object(df[col], index=False).sum())
            except TypeError:
                sig = int(pd.util.hash_pandas_object(df[col].astype(str), index=False).sum())
            signatures.setdefault(sig, []).append(col)
        exact_seen: set[str] = set()
        for cols in signatures.values():
            if len(cols) > 1 and self._truly_equal(df, cols):
                groups.append(DuplicateGroup(features=cols, method="exact", similarity=1.0))
                exact_seen.update(cols)

        # 2) Near-duplicate numeric features via high correlation.
        numeric = [c for c in types.numeric if c != target_column and c not in exact_seen]
        if len(numeric) >= 2:
            corr = df[numeric].corr().abs()
            used: set[str] = set()
            cols = list(corr.columns)
            for i, a in enumerate(cols):
                if a in used:
                    continue
                partners = [a]
                for b in cols[i + 1:]:
                    if b not in used and safe_float(corr.loc[a, b], 0.0) >= DUPLICATE_CORR:
                        partners.append(b)
                        used.add(b)
                if len(partners) > 1:
                    used.update(partners)
                    sim = float(corr.loc[partners[0], partners[1]])
                    groups.append(DuplicateGroup(features=partners, method="correlation", similarity=sim))

        # 3) Functionally-dependent categoricals (one relabels the other).
        categorical = [c for c in (types.categorical + types.boolean) if c != target_column and c not in exact_seen][:MAX_DUP_CATEGORICAL]
        for i, a in enumerate(categorical):
            for b in categorical[i + 1:]:
                if df[a].nunique() < 2 or df[b].nunique() < 2:
                    continue
                v = cramers_v(df[a], df[b])
                if v >= DUPLICATE_CRAMERS:
                    groups.append(DuplicateGroup(features=[a, b], method="cramers_v", similarity=v))
        return groups

    @staticmethod
    def _truly_equal(df: pd.DataFrame, cols: list[str]) -> bool:
        first = df[cols[0]]
        return all(first.equals(df[c]) for c in cols[1:])

    # ---------------------------------------------------------------- leakage
    def _leakage_warnings(
        self,
        df: pd.DataFrame,
        types: Any,
        target_column: str,
    ) -> list[LeakageWarning]:
        if target_column not in df.columns:
            return []
        target = df[target_column]
        target_numeric = pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > 20
        warnings: list[LeakageWarning] = []
        feature_cols = [c for c in df.columns if c != target_column]

        for col in feature_cols:
            assoc, measure = self._association_with_target(df[col], target, target_numeric, types, col)
            if assoc is None:
                continue
            reasons = []
            if assoc >= LEAKAGE_ASSOCIATION:
                reasons.append(f"near-deterministic association ({measure}={assoc:.3f})")
            if target_column.lower() in col.lower() and col.lower() != target_column.lower():
                reasons.append("name resembles the target (possible derived feature)")
            if reasons:
                warnings.append(
                    LeakageWarning(
                        feature=col,
                        association=assoc,
                        measure=measure,
                        reason="; ".join(reasons) + " -- verify it is available at prediction time.",
                    )
                )
        return sorted(warnings, key=lambda w: w.association, reverse=True)

    @staticmethod
    def _association_with_target(
        feature: pd.Series,
        target: pd.Series,
        target_numeric: bool,
        types: Any,
        col: str,
    ) -> tuple[float | None, str]:
        feature_numeric = pd.api.types.is_numeric_dtype(feature) and col in types.numeric
        frame = pd.DataFrame({"f": feature, "t": target}).dropna()
        if frame.shape[0] < 5:
            return None, ""
        if target_numeric and feature_numeric:
            if frame["f"].nunique() < 2:
                return None, ""
            return abs(safe_float(stats.pearsonr(frame["f"], frame["t"])[0], 0.0)), "pearson"
        if target_numeric and not feature_numeric:
            return correlation_ratio(frame["f"], frame["t"]), "correlation_ratio"
        if not target_numeric and feature_numeric:
            return correlation_ratio(frame["t"], frame["f"]), "correlation_ratio"
        return cramers_v(frame["f"], frame["t"]), "cramers_v"

    # --------------------------------------------------------------- encoding
    def _encoding_recs(self, df: pd.DataFrame, types: Any, target_column: str | None) -> list[EncodingRec]:
        recs: list[EncodingRec] = []
        has_target = target_column is not None and target_column in df.columns
        candidates = [c for c in (types.categorical + types.identifier) if c != target_column]
        for col in candidates:
            card = int(df[col].nunique(dropna=True))
            if card <= 1:
                continue
            if card <= 10:
                strategy, reason = "one-hot", "Low cardinality -- one-hot keeps it interpretable without exploding width."
            elif card <= 50:
                strategy = "target-encoding" if has_target else "one-hot"
                reason = (
                    "Medium cardinality -- target/mean encoding avoids a wide sparse matrix."
                    if has_target
                    else "Medium cardinality -- one-hot is acceptable; switch to target encoding once a target is set."
                )
            elif card <= 1000:
                strategy = "target-encoding" if has_target else "frequency-encoding"
                reason = "High cardinality -- one-hot would be too wide; use target/frequency encoding."
            else:
                strategy, reason = "embedding", "Very high cardinality -- learn a dense embedding (or hash) instead of explicit columns."
            recs.append(EncodingRec(feature=col, cardinality=card, strategy=strategy, reason=reason))
        return recs

    # ---------------------------------------------------------------- scaling
    def _scaling_rec(self, df: pd.DataFrame, types: Any, target_column: str | None) -> ScalingRec:
        numeric = [c for c in types.numeric if c != target_column]
        sensitive = ["k-NN", "SVM", "linear/logistic regression", "PCA", "neural networks"]
        if len(numeric) < 1:
            return ScalingRec(False, "none", "No numeric features to scale.")

        stds = df[numeric].std(ddof=0).replace(0, np.nan).dropna()
        ranges = (df[numeric].max() - df[numeric].min()).replace(0, np.nan).dropna()
        if stds.empty or ranges.empty:
            return ScalingRec(False, "none", "Numeric features have no spread.")

        ratio = float(ranges.max() / ranges.min()) if ranges.min() > 0 else float("inf")
        skewed = [
            c for c in numeric
            if df[c].dropna().shape[0] >= 3 and abs(safe_float(stats.skew(df[c].dropna()), 0.0)) >= 1.0
        ]
        # Heavy outliers anywhere => RobustScaler is the safer default.
        has_outliers = any(self._has_outliers(df[c]) for c in numeric)

        needed = ratio > SCALE_RATIO_TRIGGER or len(numeric) > 1
        if has_outliers:
            method, reason = "RobustScaler", "Outliers present -- median/IQR scaling resists their influence."
        elif self._mostly_bounded(df, numeric):
            method, reason = "MinMaxScaler", "Features are bounded -- min-max preserves the [0,1] range for NN/distance models."
        else:
            method, reason = "StandardScaler", "Differing scales -- standardise to zero mean/unit variance for distance- and gradient-based models."
        if ratio > SCALE_RATIO_TRIGGER:
            reason += f" Feature ranges differ by ~{ratio:.0f}x."
        return ScalingRec(
            needed=needed,
            method=method,
            reason=reason + " Tree-based models (RF/XGBoost) do not require scaling.",
            columns_needing_transform=skewed,
            sensitive_models=sensitive,
        )

    @staticmethod
    def _has_outliers(series: pd.Series) -> bool:
        arr = series.dropna().to_numpy(dtype="float64")
        if arr.size < 4:
            return False
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            return False
        return bool(((arr < q1 - 3 * iqr) | (arr > q3 + 3 * iqr)).any())

    @staticmethod
    def _mostly_bounded(df: pd.DataFrame, numeric: list[str]) -> bool:
        bounded = 0
        for c in numeric:
            arr = df[c].dropna()
            if not arr.empty and arr.min() >= 0 and arr.max() <= 1:
                bounded += 1
        return bounded == len(numeric)

    # --------------------------------------------------------------- insights
    def _build_insights(
        self,
        constant: list[FeatureFlag],
        duplicates: list[DuplicateGroup],
        leakage: list[LeakageWarning],
        encoding: list[EncodingRec],
        scaling: ScalingRec,
    ) -> list[Insight]:
        insights: list[Insight] = []

        if leakage:
            top = leakage[0]
            insights.append(
                Insight(
                    title="Potential target leakage",
                    insight=(
                        f"{len(leakage)} feature(s) are suspiciously predictive of the target; "
                        f"'{top.feature}' has {top.measure}={top.association:.3f}."
                    ),
                    action="Confirm each leaky feature is known *before* prediction time; remove post-outcome features.",
                    metric={"warnings": [w.to_dict() for w in leakage[:10]]},
                    confidence=0.7,
                    severity="critical",
                    category="recommendations",
                )
            )

        if constant:
            insights.append(
                Insight(
                    title="Drop constant / quasi-constant features",
                    insight=f"{len(constant)} feature(s) carry (almost) no variance: {', '.join(f.feature for f in constant[:8])}.",
                    action="Remove these columns; they cannot help a model and may break some algorithms.",
                    metric={"features": [f.to_dict() for f in constant]},
                    confidence=0.9,
                    severity="warning",
                    category="recommendations",
                )
            )

        if duplicates:
            example = duplicates[0]
            insights.append(
                Insight(
                    title="Duplicate / redundant features",
                    insight=(
                        f"{len(duplicates)} group(s) of duplicate features detected "
                        f"(e.g. {', '.join(example.features)} via {example.method})."
                    ),
                    action="Keep one representative per group to cut redundancy and multicollinearity.",
                    metric={"groups": [g.to_dict() for g in duplicates]},
                    confidence=0.85,
                    severity="warning",
                    category="recommendations",
                )
            )

        if encoding:
            non_trivial = [e for e in encoding if e.strategy != "one-hot"]
            if non_trivial:
                insights.append(
                    Insight(
                        title="Encoding recommendations",
                        insight=(
                            f"{len(non_trivial)} categorical feature(s) need smarter encoding than one-hot "
                            f"(e.g. '{non_trivial[0].feature}' -> {non_trivial[0].strategy})."
                        ),
                        action="Apply the recommended encoder per feature to balance dimensionality and signal.",
                        metric={"encoding": [e.to_dict() for e in encoding]},
                        confidence=0.75,
                        severity="info",
                        category="recommendations",
                    )
                )

        if scaling.needed:
            insights.append(
                Insight(
                    title="Scaling recommendation",
                    insight=f"Numeric features should be scaled with {scaling.method}. {scaling.reason}",
                    action=(
                        "Scale before distance/gradient-based models"
                        + (f"; transform skewed columns first: {', '.join(scaling.columns_needing_transform[:5])}." if scaling.columns_needing_transform else ".")
                    ),
                    metric=scaling.to_dict(),
                    confidence=0.7,
                    severity="info",
                    category="recommendations",
                )
            )

        return insights


def recommend_features(df: pd.DataFrame, target_column: str | None = None) -> RecommendationResult:
    """Convenience wrapper around :class:`FeatureAdvisor`."""

    return FeatureAdvisor().analyze(df, target_column=target_column)
