"""Capability 6 -- Anomaly Detection (Cloudflare style).

Three complementary lenses on "what looks wrong":

* **Univariate** -- per-column statistical outliers via IQR fences and a robust
  (median/MAD) modified Z-score, which does not get dragged around by the very
  outliers it is trying to find.
* **Multivariate** -- an Isolation Forest that catches rows which are odd only in
  combination, plus a per-feature contribution breakdown so an alert says *why*
  a row is anomalous.
* **Time-series** -- seasonal-trend decomposition (STL) of a metric, flagging
  residual spikes/dips and sustained level shifts, then suggesting root causes by
  finding other metrics that co-move with the anomalous one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .base import ChartSpec, Insight, safe_float, sort_insights
from .utils import coerce_datetime, infer_semantic_types, maybe_sample

Z_THRESHOLD = 3.0
MAD_THRESHOLD = 3.5
IQR_MULTIPLIER = 1.5
IFOREST_SAMPLE_ROWS = 200_000
_SEASONAL_PERIODS = {"min": 60, "h": 24, "D": 7, "W": 52, "M": 12}
_FREQ_MAP = {"minute": "min", "hour": "h", "day": "D", "week": "W", "month": "M"}


@dataclass
class UnivariateAnomaly:
    column: str
    method: str
    n_outliers: int
    outlier_pct: float
    lower_bound: float | None
    upper_bound: float | None
    example_indices: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "method": self.method,
            "n_outliers": self.n_outliers,
            "outlier_pct": round(self.outlier_pct, 4),
            "lower_bound": safe_float(self.lower_bound),
            "upper_bound": safe_float(self.upper_bound),
            "example_indices": [_safe_index(i) for i in self.example_indices],
        }


@dataclass
class MultivariateResult:
    n_anomalies: int
    anomaly_pct: float
    contamination: str | float
    feature_contributions: list[dict[str, Any]]
    top_anomalies: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_anomalies": self.n_anomalies,
            "anomaly_pct": round(self.anomaly_pct, 4),
            "contamination": self.contamination,
            "feature_contributions": self.feature_contributions,
            "top_anomalies": self.top_anomalies,
        }


@dataclass
class TimeSeriesAnomaly:
    timestamp: str
    value: float
    expected: float
    residual: float
    zscore: float
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "value": round(self.value, 4),
            "expected": round(self.expected, 4),
            "residual": round(self.residual, 4),
            "zscore": round(self.zscore, 3),
            "direction": self.direction,
        }


@dataclass
class AnomalyResult:
    univariate: list[UnivariateAnomaly]
    multivariate: MultivariateResult | None
    timeseries_anomalies: list[TimeSeriesAnomaly]
    level_shifts: list[dict[str, Any]]
    root_causes: list[dict[str, Any]]
    insights: list[Insight]
    timeseries_chart: ChartSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "univariate": [u.to_dict() for u in self.univariate],
            "multivariate": self.multivariate.to_dict() if self.multivariate else None,
            "timeseries_anomalies": [a.to_dict() for a in self.timeseries_anomalies],
            "level_shifts": self.level_shifts,
            "root_causes": self.root_causes,
            "timeseries_chart": self.timeseries_chart.to_dict() if self.timeseries_chart else None,
            "insights": [i.to_dict() for i in self.insights],
        }


class AnomalyDetector:
    """Univariate, multivariate and time-series anomaly detection."""

    def __init__(self, z_threshold: float = Z_THRESHOLD, iqr_multiplier: float = IQR_MULTIPLIER) -> None:
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier

    def analyze(
        self,
        df: pd.DataFrame,
        *,
        timestamp_col: str | None = None,
        value_col: str | None = None,
        columns: list[str] | None = None,
        freq: str = "day",
        agg: str = "sum",
        contamination: str | float = "auto",
    ) -> AnomalyResult:
        types = infer_semantic_types(df)
        numeric_cols = [c for c in (columns or types.numeric) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

        univariate = self._univariate(df, numeric_cols)
        multivariate = self._multivariate(df, numeric_cols, contamination) if len(numeric_cols) >= 2 else None

        ts_anoms: list[TimeSeriesAnomaly] = []
        shifts: list[dict[str, Any]] = []
        root_causes: list[dict[str, Any]] = []
        ts_chart: ChartSpec | None = None
        if timestamp_col and value_col and timestamp_col in df.columns and value_col in df.columns:
            ts_anoms, shifts, root_causes, ts_chart = self._timeseries(
                df, timestamp_col, value_col, freq, agg, numeric_cols
            )

        insights = self._build_insights(univariate, multivariate, ts_anoms, shifts, root_causes)
        return AnomalyResult(
            univariate=univariate,
            multivariate=multivariate,
            timeseries_anomalies=ts_anoms,
            level_shifts=shifts,
            root_causes=root_causes,
            insights=sort_insights(insights),
            timeseries_chart=ts_chart,
        )

    # ------------------------------------------------------------ univariate
    def _univariate(self, df: pd.DataFrame, numeric_cols: list[str]) -> list[UnivariateAnomaly]:
        results: list[UnivariateAnomaly] = []
        for col in numeric_cols:
            series = df[col].dropna()
            arr = series.to_numpy(dtype="float64")
            if arr.size < 4 or np.unique(arr).size < 2:
                continue
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            if iqr <= 0:
                # Fall back to a robust MAD rule when the IQR is degenerate.
                lower, upper, mask = self._mad_bounds(arr)
                method = "modified_zscore"
            else:
                lower = q1 - self.iqr_multiplier * iqr
                upper = q3 + self.iqr_multiplier * iqr
                mask = (arr < lower) | (arr > upper)
                method = "iqr"
            n_out = int(mask.sum())
            if n_out == 0:
                continue
            example_idx = series.index[mask][:10].tolist()
            results.append(
                UnivariateAnomaly(
                    column=col,
                    method=method,
                    n_outliers=n_out,
                    outlier_pct=n_out / arr.size,
                    lower_bound=float(lower),
                    upper_bound=float(upper),
                    example_indices=example_idx,
                )
            )
        return sorted(results, key=lambda r: r.outlier_pct, reverse=True)

    @staticmethod
    def _mad_bounds(arr: np.ndarray) -> tuple[float, float, np.ndarray]:
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        if mad == 0:
            std = float(np.std(arr))
            if std == 0:
                return med, med, np.zeros_like(arr, dtype=bool)
            z = (arr - med) / std
        else:
            z = 0.6745 * (arr - med) / mad
        mask = np.abs(z) > MAD_THRESHOLD
        scale = mad / 0.6745 if mad > 0 else float(np.std(arr))
        return med - MAD_THRESHOLD * scale, med + MAD_THRESHOLD * scale, mask

    # ---------------------------------------------------------- multivariate
    def _multivariate(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        contamination: str | float,
    ) -> MultivariateResult | None:
        data = df[numeric_cols].apply(lambda s: s.fillna(s.median()))
        data = data.loc[:, data.nunique() > 1]
        if data.shape[1] < 2 or data.shape[0] < 20:
            return None

        sampled, _ = maybe_sample(data, IFOREST_SAMPLE_ROWS)
        mean = sampled.mean()
        std = sampled.std(ddof=0).replace(0, 1.0)
        standardized = (data - mean) / std

        forest = IsolationForest(n_estimators=200, contamination=contamination, random_state=0)
        forest.fit((sampled - mean) / std)
        labels = forest.predict(standardized)
        scores = forest.score_samples(standardized)  # lower => more anomalous
        anomaly_mask = labels == -1
        n_anom = int(anomaly_mask.sum())
        if n_anom == 0:
            return MultivariateResult(0, 0.0, contamination, [], [])

        # Why are they anomalous? Average |z| per feature over flagged rows.
        flagged_z = standardized.loc[anomaly_mask].abs()
        contributions = (
            flagged_z.mean().sort_values(ascending=False).head(10).round(3).to_dict()
        )
        feature_contributions = [{"feature": k, "mean_abs_z": float(v)} for k, v in contributions.items()]

        order = np.argsort(scores)
        top_idx = [i for i in order if anomaly_mask[i]][:10]
        top_anomalies = [
            {"index": _safe_index(df.index[i]), "score": round(float(scores[i]), 4)} for i in top_idx
        ]
        return MultivariateResult(
            n_anomalies=n_anom,
            anomaly_pct=n_anom / data.shape[0],
            contamination=contamination,
            feature_contributions=feature_contributions,
            top_anomalies=top_anomalies,
        )

    # ---------------------------------------------------------- time-series
    def _timeseries(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        value_col: str,
        freq: str,
        agg: str,
        numeric_cols: list[str],
    ) -> tuple[list[TimeSeriesAnomaly], list[dict[str, Any]], list[dict[str, Any]], ChartSpec | None]:
        alias = _FREQ_MAP.get(freq, freq)
        ts = coerce_datetime(df[timestamp_col])
        value = pd.to_numeric(df[value_col], errors="coerce")
        frame = pd.DataFrame({"ts": ts, "value": value}).dropna()
        if frame.empty:
            return [], [], [], None

        series = getattr(frame.set_index("ts")["value"].resample(alias), agg)()
        series = series.dropna()
        if series.size < 4:
            return [], [], [], None

        resid, expected, decomposed = self._decompose(series, alias)
        std = float(np.nanstd(resid))
        anomalies: list[TimeSeriesAnomaly] = []
        if std > 0:
            z = (resid - np.nanmean(resid)) / std
            for ts_idx, zval in z.items():
                if abs(zval) > self.z_threshold:
                    anomalies.append(
                        TimeSeriesAnomaly(
                            timestamp=ts_idx.isoformat(),
                            value=float(series.loc[ts_idx]),
                            expected=float(expected.loc[ts_idx]) if ts_idx in expected.index else float(series.mean()),
                            residual=float(resid.loc[ts_idx]),
                            zscore=float(zval),
                            direction="spike" if zval > 0 else "dip",
                        )
                    )
        anomalies.sort(key=lambda a: abs(a.zscore), reverse=True)

        shifts = self._level_shifts(series)
        root_causes = self._root_causes(df, timestamp_col, value_col, alias, agg, numeric_cols, series)
        chart = self._timeseries_chart(series, anomalies, value_col, freq)
        return anomalies, shifts, root_causes, chart

    def _decompose(self, series: pd.Series, alias: str) -> tuple[pd.Series, pd.Series, bool]:
        """Return (residual, expected, decomposed?).

        Prefer STL with the natural seasonal period for the frequency; fall back
        to a centred rolling median when the series is too short to decompose.
        """

        period = _SEASONAL_PERIODS.get(alias, 7)
        if series.size >= 2 * period and period >= 2:
            try:
                from statsmodels.tsa.seasonal import STL

                result = STL(series, period=period, robust=True).fit()
                expected = result.trend + result.seasonal
                return result.resid, expected, True
            except Exception:
                pass
        window = max(3, min(period, series.size // 2 or 3))
        expected = series.rolling(window, center=True, min_periods=1).median()
        return series - expected, expected, False

    @staticmethod
    def _level_shifts(series: pd.Series) -> list[dict[str, Any]]:
        """Detect sustained level shifts via a simple rolling-mean excursion test."""

        if series.size < 8:
            return []
        window = max(3, series.size // 10)
        roll = series.rolling(window, min_periods=1).mean()
        overall_std = float(series.std(ddof=0))
        if overall_std == 0:
            return []
        diff = roll.diff(window)
        shifts = []
        for ts_idx, change in diff.items():
            if pd.notna(change) and abs(change) > 2 * overall_std:
                shifts.append(
                    {
                        "timestamp": ts_idx.isoformat(),
                        "change": round(float(change), 4),
                        "direction": "up" if change > 0 else "down",
                    }
                )
        # Collapse to the most pronounced few.
        shifts.sort(key=lambda s: abs(s["change"]), reverse=True)
        return shifts[:5]

    def _root_causes(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        value_col: str,
        alias: str,
        agg: str,
        numeric_cols: list[str],
        target_series: pd.Series,
    ) -> list[dict[str, Any]]:
        """Other metrics that co-move with the target, as root-cause candidates."""

        others = [c for c in numeric_cols if c != value_col]
        if not others:
            return []
        ts = coerce_datetime(df[timestamp_col])
        causes: list[dict[str, Any]] = []
        for col in others:
            frame = pd.DataFrame({"ts": ts, "v": pd.to_numeric(df[col], errors="coerce")}).dropna()
            if frame.empty:
                continue
            other_series = getattr(frame.set_index("ts")["v"].resample(alias), agg)()
            joined = pd.concat([target_series, other_series], axis=1, join="inner").dropna()
            if joined.shape[0] < 3 or joined.iloc[:, 1].nunique() < 2:
                continue
            corr = safe_float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
            if corr is not None and abs(corr) >= 0.5:
                causes.append({"feature": col, "correlation": round(corr, 3)})
        causes.sort(key=lambda c: abs(c["correlation"]), reverse=True)
        return causes[:10]

    # --------------------------------------------------------------- charts
    @staticmethod
    def _timeseries_chart(
        series: pd.Series,
        anomalies: list[TimeSeriesAnomaly],
        value_col: str,
        freq: str,
    ) -> ChartSpec:
        return ChartSpec(
            kind="line",
            title=f"{value_col} per {freq} with anomalies",
            data={
                "x": [idx.isoformat() for idx in series.index],
                "y": [safe_float(v) for v in series.to_numpy()],
                "anomaly_x": [a.timestamp for a in anomalies],
                "anomaly_y": [a.value for a in anomalies],
            },
            layout={"xaxis_title": "time", "yaxis_title": value_col},
        )

    # --------------------------------------------------------------- insights
    def _build_insights(
        self,
        univariate: list[UnivariateAnomaly],
        multivariate: MultivariateResult | None,
        ts_anoms: list[TimeSeriesAnomaly],
        shifts: list[dict[str, Any]],
        root_causes: list[dict[str, Any]],
    ) -> list[Insight]:
        insights: list[Insight] = []

        if ts_anoms:
            worst = ts_anoms[0]
            rc = f" Co-moving metrics: {', '.join(c['feature'] for c in root_causes[:3])}." if root_causes else ""
            insights.append(
                Insight(
                    title="Time-series anomalies detected",
                    insight=(
                        f"{len(ts_anoms)} anomalous period(s); largest is a {worst.direction} on {worst.timestamp[:10]} "
                        f"(value {worst.value:.1f} vs expected {worst.expected:.1f}, z={worst.zscore:.1f})." + rc
                    ),
                    action="Investigate the flagged dates; correlate with releases, campaigns or incidents.",
                    metric={"anomalies": [a.to_dict() for a in ts_anoms[:10]], "root_causes": root_causes[:5]},
                    confidence=0.75,
                    severity="warning",
                    category="anomaly",
                )
            )

        if shifts:
            top = shifts[0]
            insights.append(
                Insight(
                    title="Level shift in metric",
                    insight=f"A sustained {top['direction']} shift (~{abs(top['change']):.1f}) occurred around {top['timestamp'][:10]}.",
                    action="Treat as a regime change: re-baseline alerts and check for a structural cause.",
                    metric={"shifts": shifts},
                    confidence=0.6,
                    severity="warning",
                    category="anomaly",
                )
            )

        if multivariate and multivariate.n_anomalies > 0:
            drivers = ", ".join(f["feature"] for f in multivariate.feature_contributions[:3])
            insights.append(
                Insight(
                    title="Multivariate outliers (Isolation Forest)",
                    insight=(
                        f"{multivariate.n_anomalies:,} row(s) ({multivariate.anomaly_pct * 100:.1f}%) are joint outliers; "
                        f"most driven by: {drivers}."
                    ),
                    action="Review these rows for data-entry errors, fraud, or genuinely rare events before modeling.",
                    metric=multivariate.to_dict(),
                    confidence=0.7,
                    severity="warning" if multivariate.anomaly_pct > 0.01 else "info",
                    category="anomaly",
                )
            )

        if univariate:
            worst = univariate[0]
            insights.append(
                Insight(
                    title="Univariate outliers",
                    insight=(
                        f"{len(univariate)} column(s) contain statistical outliers; '{worst.column}' has "
                        f"{worst.n_outliers:,} ({worst.outlier_pct * 100:.1f}%) outside its {worst.method} bounds."
                    ),
                    action="Cap/winsorize, transform, or quarantine extreme values depending on whether they are valid.",
                    metric={"columns": [u.to_dict() for u in univariate[:10]]},
                    confidence=0.7,
                    severity="info",
                    category="anomaly",
                )
            )

        return insights


def _safe_index(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def detect_anomalies(df: pd.DataFrame, **kwargs: Any) -> AnomalyResult:
    """Convenience wrapper around :class:`AnomalyDetector`."""

    return AnomalyDetector().analyze(df, **kwargs)
