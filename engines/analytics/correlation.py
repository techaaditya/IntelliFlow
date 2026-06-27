"""Capability 2 -- Correlation & Feature Intelligence.

Answers the questions a modeller asks before building anything: which features
move together (Pearson for linear, Spearman for monotonic, Cramer's V for
categorical pairs), which are so collinear they should not coexist in a linear
model (VIF), which features actually relate to the target (a unified relevance
ranking built on mutual information so it catches non-linear signal), and which
relationships are non-linear and would be wasted on a purely linear model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from .base import ChartSpec, Insight, safe_float, sort_insights
from .utils import infer_semantic_types, maybe_sample

HIGH_CORR_THRESHOLD = 0.8
VIF_THRESHOLD = 5.0
NONLINEAR_GAP = 0.15  # |spearman| - |pearson| above this => monotonic non-linearity
MAX_CATEGORICAL_COLS = 30  # Cramer's V is O(k^2); cap to stay responsive
MI_SAMPLE_ROWS = 50_000


@dataclass
class CorrelationPair:
    feature_a: str
    feature_b: str
    method: str
    value: float

    def to_dict(self) -> dict[str, Any]:
        return {"feature_a": self.feature_a, "feature_b": self.feature_b, "method": self.method, "value": round(self.value, 4)}


@dataclass
class VIFRecord:
    feature: str
    vif: float

    def to_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "vif": None if self.vif is None else round(self.vif, 3)}


@dataclass
class TargetAssociation:
    feature: str
    linear_corr: float | None
    monotonic_corr: float | None
    relevance: float  # mutual information (>=0), the unified ranking key
    direction: str
    nonlinear: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "linear_corr": None if self.linear_corr is None else round(self.linear_corr, 4),
            "monotonic_corr": None if self.monotonic_corr is None else round(self.monotonic_corr, 4),
            "relevance": round(self.relevance, 4),
            "direction": self.direction,
            "nonlinear": self.nonlinear,
        }


@dataclass
class CorrelationResult:
    pearson: dict[str, Any] | None
    spearman: dict[str, Any] | None
    cramers_v: dict[str, Any] | None
    high_correlations: list[CorrelationPair]
    vif: list[VIFRecord]
    target_ranking: list[TargetAssociation]
    nonlinear_relationships: list[dict[str, Any]]
    insights: list[Insight]
    heatmap: ChartSpec | None = None
    dendrogram: ChartSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pearson": self.pearson,
            "spearman": self.spearman,
            "cramers_v": self.cramers_v,
            "high_correlations": [p.to_dict() for p in self.high_correlations],
            "vif": [v.to_dict() for v in self.vif],
            "target_ranking": [t.to_dict() for t in self.target_ranking],
            "nonlinear_relationships": self.nonlinear_relationships,
            "heatmap": self.heatmap.to_dict() if self.heatmap else None,
            "dendrogram": self.dendrogram.to_dict() if self.dendrogram else None,
            "insights": [i.to_dict() for i in self.insights],
        }


class CorrelationAnalyzer:
    """Compute correlation structure and feature relevance for a dataset."""

    def __init__(
        self,
        high_corr_threshold: float = HIGH_CORR_THRESHOLD,
        vif_threshold: float = VIF_THRESHOLD,
    ) -> None:
        self.high_corr_threshold = high_corr_threshold
        self.vif_threshold = vif_threshold

    def analyze(self, df: pd.DataFrame, target_column: str | None = None) -> CorrelationResult:
        types = infer_semantic_types(df)
        feature_numeric = [c for c in types.numeric if c != target_column]
        feature_categorical = [c for c in (types.categorical + types.boolean) if c != target_column][:MAX_CATEGORICAL_COLS]

        pearson_df = self._safe_corr(df[feature_numeric], "pearson") if len(feature_numeric) >= 2 else None
        spearman_df = self._safe_corr(df[feature_numeric], "spearman") if len(feature_numeric) >= 2 else None
        cramers_df = self._cramers_matrix(df, feature_categorical) if len(feature_categorical) >= 2 else None

        high_corr = self._high_correlations(pearson_df, spearman_df, cramers_df)
        vif = self._compute_vif(df, feature_numeric)

        target_ranking: list[TargetAssociation] = []
        nonlinear: list[dict[str, Any]] = []
        if target_column is not None and target_column in df.columns:
            target_ranking, nonlinear = self._target_analysis(
                df, target_column, feature_numeric, feature_categorical
            )

        heatmap = self._heatmap_spec(pearson_df)
        dendrogram = self._dendrogram_spec(pearson_df)
        insights = self._build_insights(high_corr, vif, target_ranking, nonlinear, target_column)

        return CorrelationResult(
            pearson=self._matrix_payload(pearson_df),
            spearman=self._matrix_payload(spearman_df),
            cramers_v=self._matrix_payload(cramers_df),
            high_correlations=high_corr,
            vif=vif,
            target_ranking=target_ranking,
            nonlinear_relationships=nonlinear,
            insights=sort_insights(insights),
            heatmap=heatmap,
            dendrogram=dendrogram,
        )

    # ---------------------------------------------------------------- matrices
    @staticmethod
    def _safe_corr(frame: pd.DataFrame, method: str) -> pd.DataFrame | None:
        numeric = frame.select_dtypes(include=[np.number])
        numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]  # drop constants
        if numeric.shape[1] < 2:
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            matrix = numeric.corr(method=method)
        return matrix

    def _cramers_matrix(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame | None:
        usable = [c for c in columns if df[c].nunique(dropna=True) > 1]
        if len(usable) < 2:
            return None
        matrix = pd.DataFrame(np.eye(len(usable)), index=usable, columns=usable)
        for i, a in enumerate(usable):
            for b in usable[i + 1:]:
                value = cramers_v(df[a], df[b])
                matrix.loc[a, b] = value
                matrix.loc[b, a] = value
        return matrix

    def _high_correlations(
        self,
        pearson: pd.DataFrame | None,
        spearman: pd.DataFrame | None,
        cramers: pd.DataFrame | None,
    ) -> list[CorrelationPair]:
        pairs: list[CorrelationPair] = []
        for matrix, method in ((pearson, "pearson"), (spearman, "spearman"), (cramers, "cramers_v")):
            if matrix is None:
                continue
            cols = list(matrix.columns)
            for i, a in enumerate(cols):
                for b in cols[i + 1:]:
                    value = matrix.loc[a, b]
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        continue
                    if abs(value) >= self.high_corr_threshold:
                        pairs.append(CorrelationPair(a, b, method, float(value)))
        return sorted(pairs, key=lambda p: abs(p.value), reverse=True)

    # -------------------------------------------------------------------- VIF
    def _compute_vif(self, df: pd.DataFrame, numeric_cols: list[str]) -> list[VIFRecord]:
        """Variance Inflation Factor per numeric feature.

        ``VIF_i = 1 / (1 - R^2_i)`` where ``R^2_i`` comes from regressing feature
        ``i`` on every other numeric feature. Computed with ``lstsq`` on
        median-imputed, standardised complete cases so it stays finite under mild
        collinearity and reports ``inf`` only for genuine linear dependence.
        """

        usable = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1]
        if len(usable) < 2:
            return []

        data = df[usable].apply(lambda s: s.fillna(s.median()))
        std = data.std(ddof=0).replace(0, np.nan)
        data = (data - data.mean()) / std
        data = data.dropna(axis=1, how="any")
        if data.shape[1] < 2 or data.shape[0] < 3:
            return []

        matrix = data.to_numpy()
        n = matrix.shape[0]
        records: list[VIFRecord] = []
        for idx, col in enumerate(data.columns):
            y = matrix[:, idx]
            others = np.delete(matrix, idx, axis=1)
            design = np.column_stack([np.ones(n), others])
            coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            resid = y - design @ coef
            ss_res = float(resid @ resid)
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2 = min(max(r2, 0.0), 1.0)
            vif = float("inf") if r2 >= 1.0 - 1e-10 else 1.0 / (1.0 - r2)
            records.append(VIFRecord(col, vif))
        return sorted(records, key=lambda r: (np.isfinite(r.vif), r.vif), reverse=True)

    # -------------------------------------------------------------- target rank
    def _target_analysis(
        self,
        df: pd.DataFrame,
        target_column: str,
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> tuple[list[TargetAssociation], list[dict[str, Any]]]:
        target = df[target_column]
        is_classification = not pd.api.types.is_numeric_dtype(target) or target.nunique(dropna=True) <= 20
        features = numeric_cols + categorical_cols
        if not features:
            return [], []

        work = df[features + [target_column]].dropna(subset=[target_column]).copy()
        if work.empty:
            return [], []
        work, _ = maybe_sample(work, MI_SAMPLE_ROWS)

        # Build an encoded numeric feature matrix for mutual information.
        encoded = pd.DataFrame(index=work.index)
        discrete_mask: list[bool] = []
        for col in features:
            series = work[col]
            if col in categorical_cols or not pd.api.types.is_numeric_dtype(series):
                encoded[col] = pd.factorize(series)[0]
                discrete_mask.append(True)
            else:
                encoded[col] = series.fillna(series.median())
                discrete_mask.append(False)

        y = work[target_column]
        if is_classification:
            y_enc = pd.factorize(y)[0]
            mi = mutual_info_classif(encoded.to_numpy(), y_enc, discrete_features=discrete_mask, random_state=0)
        else:
            y_enc = y.to_numpy(dtype="float64")
            mi = mutual_info_regression(encoded.to_numpy(), y_enc, discrete_features=discrete_mask, random_state=0)

        associations: list[TargetAssociation] = []
        nonlinear: list[dict[str, Any]] = []
        target_numeric = pd.to_numeric(y, errors="coerce") if not is_classification else None

        for col, relevance, discrete in zip(features, mi, discrete_mask):
            linear = monotonic = None
            direction = "n/a"
            is_nl = False
            if not discrete and target_numeric is not None:
                feat = pd.to_numeric(work[col], errors="coerce")
                mask = feat.notna() & target_numeric.notna()
                if mask.sum() >= 3 and feat[mask].nunique() > 1 and target_numeric[mask].nunique() > 1:
                    linear = safe_float(stats.pearsonr(feat[mask], target_numeric[mask])[0])
                    monotonic = safe_float(stats.spearmanr(feat[mask], target_numeric[mask])[0])
                    if linear is not None:
                        direction = "positive" if linear >= 0 else "negative"
                    if linear is not None and monotonic is not None and abs(monotonic) - abs(linear) >= NONLINEAR_GAP:
                        is_nl = True
            associations.append(
                TargetAssociation(
                    feature=col,
                    linear_corr=linear,
                    monotonic_corr=monotonic,
                    relevance=float(max(relevance, 0.0)),
                    direction=direction,
                    nonlinear=is_nl,
                )
            )
            if is_nl:
                nonlinear.append(
                    {"feature": col, "linear_corr": round(linear, 4), "monotonic_corr": round(monotonic, 4)}
                )

        associations.sort(key=lambda a: a.relevance, reverse=True)
        # A strong-relevance / weak-linear feature is also a non-linear signal.
        if associations:
            top_relevance = associations[0].relevance or 1.0
            for a in associations:
                if not a.nonlinear and a.linear_corr is not None and abs(a.linear_corr) < 0.1 and a.relevance >= 0.4 * top_relevance and a.relevance > 0:
                    a.nonlinear = True
                    nonlinear.append({"feature": a.feature, "linear_corr": round(a.linear_corr, 4), "relevance": round(a.relevance, 4)})
        return associations, nonlinear

    # ---------------------------------------------------------------- charts
    @staticmethod
    def _heatmap_spec(matrix: pd.DataFrame | None) -> ChartSpec | None:
        if matrix is None or matrix.shape[0] < 2:
            return None
        labels = list(matrix.columns)
        values = [[safe_float(matrix.iloc[i, j], 0.0) for j in range(len(labels))] for i in range(len(labels))]
        return ChartSpec(
            kind="heatmap",
            title="Pearson correlation heatmap",
            data={"labels": labels, "matrix": values, "zmin": -1, "zmax": 1},
            layout={"colorscale": "RdBu"},
        )

    @staticmethod
    def _dendrogram_spec(matrix: pd.DataFrame | None) -> ChartSpec | None:
        if matrix is None or matrix.shape[0] < 3:
            return None
        labels = list(matrix.columns)
        corr = matrix.to_numpy(dtype="float64")
        corr = np.nan_to_num(corr, nan=0.0)
        distance = 1.0 - np.abs(corr)
        np.fill_diagonal(distance, 0.0)
        distance = (distance + distance.T) / 2.0  # enforce symmetry for squareform
        try:
            condensed = squareform(distance, checks=False)
            link = linkage(condensed, method="average")
            order = leaves_list(link).tolist()
        except Exception:
            return None
        return ChartSpec(
            kind="dendrogram",
            title="Feature correlation clustering",
            data={
                "labels": labels,
                "linkage": link.tolist(),
                "leaf_order": order,
                "ordered_labels": [labels[i] for i in order],
            },
        )

    # --------------------------------------------------------------- insights
    def _build_insights(
        self,
        high_corr: list[CorrelationPair],
        vif: list[VIFRecord],
        target_ranking: list[TargetAssociation],
        nonlinear: list[dict[str, Any]],
        target_column: str | None,
    ) -> list[Insight]:
        insights: list[Insight] = []

        if high_corr:
            top = high_corr[0]
            insights.append(
                Insight(
                    title="Highly correlated feature pairs",
                    insight=(
                        f"{len(high_corr)} feature pair(s) exceed |corr| {self.high_corr_threshold:.0%}; "
                        f"strongest: '{top.feature_a}' ~ '{top.feature_b}' ({top.method} {top.value:.2f})."
                    ),
                    action="Drop or combine one feature from each redundant pair to reduce multicollinearity.",
                    metric={"pairs": [p.to_dict() for p in high_corr[:10]]},
                    confidence=0.85,
                    severity="warning",
                    category="correlation",
                )
            )

        flagged_vif = [v for v in vif if not np.isfinite(v.vif) or v.vif > self.vif_threshold]
        if flagged_vif:
            worst = flagged_vif[0]
            worst_val = "inf" if not np.isfinite(worst.vif) else f"{worst.vif:.1f}"
            insights.append(
                Insight(
                    title="Multicollinearity (high VIF)",
                    insight=(
                        f"{len(flagged_vif)} feature(s) have VIF > {self.vif_threshold:.0f} "
                        f"(worst: '{worst.feature}' = {worst_val}), indicating redundant linear information."
                    ),
                    action="Remove the highest-VIF features iteratively, or use PCA/regularisation for linear models.",
                    metric={"flagged": [v.to_dict() for v in flagged_vif[:10]]},
                    confidence=0.8,
                    severity="warning",
                    category="correlation",
                )
            )

        if target_ranking:
            top = target_ranking[: min(5, len(target_ranking))]
            insights.append(
                Insight(
                    title=f"Top predictors of '{target_column}'",
                    insight=(
                        "Most relevant features (mutual information): "
                        + ", ".join(f"{a.feature} ({a.relevance:.3f})" for a in top)
                        + "."
                    ),
                    action="Prioritise these features; consider dropping zero-relevance features to simplify the model.",
                    metric={"ranking": [a.to_dict() for a in target_ranking[:15]]},
                    confidence=0.75,
                    severity="info",
                    category="correlation",
                    visualization=ChartSpec(
                        kind="bar",
                        title=f"Feature relevance to {target_column}",
                        data={"x": [round(a.relevance, 4) for a in top], "y": [a.feature for a in top]},
                        layout={"orientation": "h", "xaxis_title": "mutual information"},
                    ),
                )
            )

        if nonlinear:
            insights.append(
                Insight(
                    title="Non-linear relationships detected",
                    insight=(
                        f"{len(nonlinear)} feature(s) relate to the target non-linearly (high rank/MI but weak linear "
                        f"correlation), e.g. '{nonlinear[0]['feature']}'."
                    ),
                    action="Use tree-based/gradient-boosting models or add spline/polynomial terms to capture these.",
                    metric={"features": nonlinear[:10]},
                    confidence=0.65,
                    severity="info",
                    category="correlation",
                )
            )

        return insights

    @staticmethod
    def _matrix_payload(matrix: pd.DataFrame | None) -> dict[str, Any] | None:
        if matrix is None:
            return None
        labels = list(matrix.columns)
        values = [[safe_float(matrix.iloc[i, j]) for j in range(len(labels))] for i in range(len(labels))]
        return {"columns": labels, "matrix": values}


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V association between two categorical series.

    Uses the Bergsma-Wicher correction so the statistic is not inflated for small
    samples or sparse contingency tables. Returns a value in ``[0, 1]``.
    """

    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Correlation ratio (eta) measuring how much a categorical explains a numeric.

    Returns ``eta`` in ``[0, 1]``: the square root of between-group variance over
    total variance. Useful for ranking categorical features against a numeric
    target without one-hot encoding.
    """

    frame = pd.DataFrame({"cat": categories, "val": pd.to_numeric(values, errors="coerce")}).dropna()
    if frame.empty or frame["cat"].nunique() < 2:
        return 0.0
    grand_mean = frame["val"].mean()
    ss_total = float(((frame["val"] - grand_mean) ** 2).sum())
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for _, group in frame.groupby("cat", observed=True):
        ss_between += len(group) * (group["val"].mean() - grand_mean) ** 2
    return float(np.sqrt(max(0.0, ss_between) / ss_total))


def analyze_correlations(df: pd.DataFrame, target_column: str | None = None, **kwargs: Any) -> CorrelationResult:
    """Convenience wrapper around :class:`CorrelationAnalyzer`."""

    return CorrelationAnalyzer(**kwargs).analyze(df, target_column=target_column)
