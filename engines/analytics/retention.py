"""Capability 4 -- Cohort Retention Analysis (Netflix / Amplitude style).

Groups users into cohorts (by first-seen period or an explicit grouping column)
and measures how many come back over time. The key correctness detail, and the
thing naive implementations get wrong, is *maturity*: a cohort that signed up
yesterday cannot have a Day-30 number, so it must be excluded from the Day-30
denominator rather than counted as a zero. Every retention rate here is computed
only over cohorts old enough to be observed at that horizon.

Outputs: N-period retention curve, a cohort x offset retention matrix (heatmap),
a new-cohort trend (are recent cohorts retaining worse?), per-segment retention,
and a churn-risk table of engaged users who have gone quiet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .base import ChartSpec, Insight, safe_float, sort_insights
from .utils import coerce_datetime, wilson_interval

RetentionUnit = Literal["day", "week"]
DEFAULT_HORIZONS = {"day": [0, 1, 7, 14, 30, 60, 90], "week": [0, 1, 2, 4, 8, 12]}
MAX_HEATMAP_OFFSETS = 12
MAX_SEGMENTS = 10


@dataclass
class RetentionPoint:
    offset: int
    retained: int
    eligible: int
    rate: float
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset": self.offset,
            "retained": self.retained,
            "eligible": self.eligible,
            "rate": round(self.rate, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
        }


@dataclass
class ChurnRisk:
    total_users: int
    engaged_users: int
    at_risk: int
    at_risk_rate: float
    threshold_days: float
    feature_columns: list[str]
    top_at_risk: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_users": self.total_users,
            "engaged_users": self.engaged_users,
            "at_risk": self.at_risk,
            "at_risk_rate": round(self.at_risk_rate, 4),
            "threshold_days": round(self.threshold_days, 2),
            "feature_columns": self.feature_columns,
            "top_at_risk": self.top_at_risk,
        }


@dataclass
class SegmentRetention:
    segment: str
    cohort_size: int
    curve: list[RetentionPoint]

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment": self.segment,
            "cohort_size": self.cohort_size,
            "curve": [p.to_dict() for p in self.curve],
        }


@dataclass
class RetentionResult:
    retention_unit: str
    cohort_period: str
    n_users: int
    n_cohorts: int
    curve: list[RetentionPoint]
    cohort_matrix: dict[str, Any]
    new_cohort_trend: dict[str, Any]
    segments: list[SegmentRetention]
    churn: ChurnRisk
    insights: list[Insight]
    heatmap: ChartSpec | None = None
    curve_chart: ChartSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "retention_unit": self.retention_unit,
            "cohort_period": self.cohort_period,
            "n_users": self.n_users,
            "n_cohorts": self.n_cohorts,
            "curve": [p.to_dict() for p in self.curve],
            "cohort_matrix": self.cohort_matrix,
            "new_cohort_trend": self.new_cohort_trend,
            "segments": [s.to_dict() for s in self.segments],
            "churn": self.churn.to_dict(),
            "heatmap": self.heatmap.to_dict() if self.heatmap else None,
            "curve_chart": self.curve_chart.to_dict() if self.curve_chart else None,
            "insights": [i.to_dict() for i in self.insights],
        }


class RetentionAnalyzer:
    """Cohort retention, churn-risk and trend analysis over an event log."""

    def analyze(
        self,
        df: pd.DataFrame,
        *,
        user_col: str,
        timestamp_col: str,
        cohort_col: str | None = None,
        cohort_period: str = "W",
        retention_unit: RetentionUnit = "day",
        horizons: list[int] | None = None,
        segment_col: str | None = None,
    ) -> RetentionResult:
        for col in (user_col, timestamp_col):
            if col not in df.columns:
                raise ValueError(f"Column {col!r} not found in dataset.")
        horizons = sorted(set(horizons or DEFAULT_HORIZONS[retention_unit]))
        unit_days = 7 if retention_unit == "week" else 1

        events, user_tbl, max_observed = self._prepare(
            df, user_col, timestamp_col, cohort_col, cohort_period, unit_days, segment_col
        )
        if user_tbl.empty:
            raise ValueError("No usable (user, timestamp) rows for retention analysis.")

        active = events.drop_duplicates(["user", "offset"])
        curve = self._retention_curve(active, user_tbl, horizons)
        matrix_payload, heatmap = self._cohort_matrix(active, user_tbl)
        trend = self._new_cohort_trend(active, user_tbl, horizons)
        segments = (
            self._segment_curves(active, user_tbl, horizons, segment_col) if segment_col else []
        )
        churn = self._churn_risk(user_tbl, active, max_observed, unit_days, user_col)

        insights = self._build_insights(curve, trend, segments, churn, retention_unit)
        curve_chart = self._curve_chart(curve, retention_unit)

        return RetentionResult(
            retention_unit=retention_unit,
            cohort_period=cohort_period if cohort_col is None else f"column:{cohort_col}",
            n_users=int(user_tbl.shape[0]),
            n_cohorts=int(user_tbl["cohort_label"].nunique()),
            curve=curve,
            cohort_matrix=matrix_payload,
            new_cohort_trend=trend,
            segments=segments,
            churn=churn,
            insights=sort_insights(insights),
            heatmap=heatmap,
            curve_chart=curve_chart,
        )

    # ------------------------------------------------------------- preparation
    def _prepare(
        self,
        df: pd.DataFrame,
        user_col: str,
        timestamp_col: str,
        cohort_col: str | None,
        cohort_period: str,
        unit_days: int,
        segment_col: str | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
        data = pd.DataFrame({"user": df[user_col].to_numpy(), "ts": coerce_datetime(df[timestamp_col])})
        if cohort_col and cohort_col in df.columns:
            data["cohort_src"] = df[cohort_col].to_numpy()
        if segment_col and segment_col in df.columns:
            data["seg"] = df[segment_col].to_numpy()
        data = data.dropna(subset=["user", "ts"])
        if data.empty:
            return data, pd.DataFrame(), pd.Timestamp.now()

        max_observed = data["ts"].max()
        data["cohort_date"] = data.groupby("user")["ts"].transform("min")

        day_diff = (data["ts"].dt.normalize() - data["cohort_date"].dt.normalize()).dt.days
        data["offset"] = (day_diff // unit_days).astype(int)

        if cohort_col and "cohort_src" in data:
            data["cohort_label"] = data.groupby("user")["cohort_src"].transform("first").astype(str)
        else:
            data["cohort_label"] = data["cohort_date"].dt.to_period(cohort_period).astype(str)

        agg = {
            "cohort_date": ("cohort_date", "first"),
            "cohort_label": ("cohort_label", "first"),
            "first_ts": ("ts", "min"),
            "last_ts": ("ts", "max"),
            "n_events": ("ts", "size"),
        }
        if "seg" in data:
            agg["seg"] = ("seg", "first")
        user_tbl = data.groupby("user").agg(**agg)
        user_tbl["max_offset_obs"] = (
            (max_observed.normalize() - user_tbl["cohort_date"].dt.normalize()).dt.days // unit_days
        ).astype(int)
        return data, user_tbl, max_observed

    # --------------------------------------------------------- retention curve
    def _retention_curve(
        self,
        active: pd.DataFrame,
        user_tbl: pd.DataFrame,
        horizons: list[int],
    ) -> list[RetentionPoint]:
        points: list[RetentionPoint] = []
        active_by_offset = active.groupby("offset")["user"].apply(lambda s: set(s))
        max_obs = user_tbl["max_offset_obs"]
        for h in horizons:
            eligible = int((max_obs >= h).sum())
            if eligible == 0:
                continue
            users_at_h = active_by_offset.get(h, set())
            retained = len(users_at_h)
            rate = retained / eligible if eligible else 0.0
            _, lo, hi = wilson_interval(retained, eligible)
            points.append(RetentionPoint(offset=h, retained=retained, eligible=eligible, rate=rate, ci_low=lo, ci_high=hi))
        return points

    # ----------------------------------------------------------- cohort matrix
    def _cohort_matrix(
        self,
        active: pd.DataFrame,
        user_tbl: pd.DataFrame,
    ) -> tuple[dict[str, Any], ChartSpec | None]:
        sizes = user_tbl.groupby("cohort_label").size()
        cohort_max_offset = user_tbl.groupby("cohort_label")["max_offset_obs"].max()
        labels = self._order_cohorts(sizes.index)
        offsets = list(range(0, min(MAX_HEATMAP_OFFSETS, int(user_tbl["max_offset_obs"].max()) + 1)))
        if not offsets:
            offsets = [0]

        # ``active`` already carries cohort_label (inherited from the event frame).
        counts = active.groupby(["cohort_label", "offset"])["user"].nunique()

        matrix: list[list[float | None]] = []
        for label in labels:
            size = int(sizes.get(label, 0)) or 1
            max_off = int(cohort_max_offset.get(label, 0))
            row: list[float | None] = []
            for off in offsets:
                if off > max_off:
                    row.append(None)  # cohort not mature enough -> unknown, not zero
                else:
                    n = int(counts.get((label, off), 0))
                    row.append(round(n / size, 4))
            matrix.append(row)

        payload = {
            "cohorts": labels,
            "offsets": offsets,
            "sizes": [int(sizes.get(l, 0)) for l in labels],
            "matrix": matrix,
        }
        heatmap = ChartSpec(
            kind="heatmap",
            title="Cohort retention heatmap",
            data={"labels": labels, "x": offsets, "matrix": matrix, "zmin": 0, "zmax": 1},
            layout={"xaxis_title": "periods since first seen", "yaxis_title": "cohort", "colorscale": "Blues"},
        )
        return payload, heatmap

    # -------------------------------------------------------------- trend
    def _new_cohort_trend(
        self,
        active: pd.DataFrame,
        user_tbl: pd.DataFrame,
        horizons: list[int],
    ) -> dict[str, Any]:
        """Is early retention drifting as cohorts get newer?

        Regress each cohort's retention at the first meaningful horizon (>=1)
        against cohort recency. A negative slope means newer cohorts retain worse.
        """

        target_offset = next((h for h in horizons if h >= 1), None)
        if target_offset is None:
            return {"available": False}

        sizes = user_tbl.groupby("cohort_label").size()
        cohort_max = user_tbl.groupby("cohort_label")["max_offset_obs"].max()
        mature = [c for c in sizes.index if cohort_max.get(c, 0) >= target_offset and sizes.get(c, 0) >= 5]
        if len(mature) < 3:
            return {"available": False, "reason": "not enough mature cohorts"}

        counts = (
            active[active["offset"] == target_offset]
            .groupby("cohort_label")["user"].nunique()
        )
        ordered = self._order_cohorts(pd.Index(mature))
        rates = [counts.get(c, 0) / sizes.get(c, 1) for c in ordered]
        x = np.arange(len(ordered), dtype="float64")
        slope = float(np.polyfit(x, rates, 1)[0]) if len(set(rates)) > 1 else 0.0
        direction = "declining" if slope < -1e-4 else ("improving" if slope > 1e-4 else "stable")
        return {
            "available": True,
            "horizon": target_offset,
            "slope_per_cohort": round(slope, 5),
            "direction": direction,
            "first_cohort_rate": round(rates[0], 4),
            "last_cohort_rate": round(rates[-1], 4),
            "cohorts": ordered,
            "rates": [round(r, 4) for r in rates],
        }

    # ------------------------------------------------------------- segments
    def _segment_curves(
        self,
        active: pd.DataFrame,
        user_tbl: pd.DataFrame,
        horizons: list[int],
        segment_col: str,
    ) -> list[SegmentRetention]:
        if "seg" not in user_tbl.columns:
            return []
        seg_series = user_tbl["seg"].fillna("(unknown)").astype(str)
        sizes = seg_series.groupby(seg_series).size().sort_values(ascending=False)
        user_seg = seg_series.to_dict()
        active = active.copy()
        active["seg"] = active["user"].map(user_seg)

        results: list[SegmentRetention] = []
        for value in sizes.head(MAX_SEGMENTS).index:
            sub_users = user_tbl[seg_series == value]
            sub_active = active[active["seg"] == value]
            curve = self._retention_curve(sub_active, sub_users, horizons)
            results.append(SegmentRetention(segment=str(value), cohort_size=int(sub_users.shape[0]), curve=curve))
        return results

    # --------------------------------------------------------------- churn
    def _churn_risk(
        self,
        user_tbl: pd.DataFrame,
        active: pd.DataFrame,
        max_observed: pd.Timestamp,
        unit_days: int,
        user_col: str,
    ) -> ChurnRisk:
        tbl = user_tbl.copy()
        distinct = active.groupby("user")["offset"].nunique()
        tbl["active_units"] = distinct.reindex(tbl.index).fillna(1).astype(int)
        tbl["recency_days"] = (max_observed.normalize() - tbl["last_ts"].dt.normalize()).dt.days
        tbl["tenure_days"] = (tbl["last_ts"].dt.normalize() - tbl["first_ts"].dt.normalize()).dt.days
        tbl["avg_gap_days"] = tbl["tenure_days"] / (tbl["active_units"] - 1).clip(lower=1)

        engaged = tbl[tbl["active_units"] >= 2]
        typical_gap = float(engaged["avg_gap_days"].replace(0, np.nan).median()) if not engaged.empty else float(unit_days)
        if not np.isfinite(typical_gap) or typical_gap <= 0:
            typical_gap = float(unit_days)
        threshold = max(2.0 * typical_gap, float(unit_days))

        feature_cols = ["recency_days", "tenure_days", "active_units", "n_events", "avg_gap_days"]
        if engaged.empty:
            return ChurnRisk(
                total_users=int(tbl.shape[0]),
                engaged_users=0,
                at_risk=0,
                at_risk_rate=0.0,
                threshold_days=threshold,
                feature_columns=feature_cols,
            )

        scores = (engaged["recency_days"] / (2.0 * threshold)).clip(0.0, 1.0)
        at_risk_mask = engaged["recency_days"] > threshold
        ranked = scores[at_risk_mask].sort_values(ascending=False)
        top = []
        for user_id in ranked.head(25).index:
            row = engaged.loc[user_id]
            top.append(
                {
                    "user": json_user(user_id),
                    "risk_score": round(float(scores.loc[user_id]), 4),
                    "recency_days": int(row["recency_days"]),
                    "tenure_days": int(row["tenure_days"]),
                    "active_units": int(row["active_units"]),
                    "n_events": int(row["n_events"]),
                }
            )

        return ChurnRisk(
            total_users=int(tbl.shape[0]),
            engaged_users=int(engaged.shape[0]),
            at_risk=int(at_risk_mask.sum()),
            at_risk_rate=float(at_risk_mask.mean()),
            threshold_days=threshold,
            feature_columns=feature_cols,
            top_at_risk=top,
        )

    # --------------------------------------------------------------- insights
    def _build_insights(
        self,
        curve: list[RetentionPoint],
        trend: dict[str, Any],
        segments: list[SegmentRetention],
        churn: ChurnRisk,
        unit: str,
    ) -> list[Insight]:
        insights: list[Insight] = []
        prefix = "D" if unit == "day" else "W"

        if curve:
            headline = self._pick_headline_point(curve)
            if headline is not None:
                sev = "warning" if headline.rate < 0.3 else "info"
                insights.append(
                    Insight(
                        title=f"{prefix}{headline.offset} retention",
                        insight=(
                            f"{prefix}{headline.offset} retention is {headline.rate * 100:.1f}% "
                            f"({headline.retained:,} of {headline.eligible:,} eligible users), 95% CI "
                            f"[{headline.ci_low * 100:.1f}%, {headline.ci_high * 100:.1f}%]."
                        ),
                        action="Benchmark against your category; invest in onboarding/habit loops if early retention is weak.",
                        metric={"offset": headline.offset, "rate": round(headline.rate, 4)},
                        confidence=0.8,
                        severity=sev,
                        category="retention",
                        visualization=self._curve_chart(curve, unit),
                    )
                )

            cliff = self._biggest_cliff(curve)
            if cliff is not None:
                insights.append(
                    Insight(
                        title="Largest retention drop",
                        insight=(
                            f"The steepest fall is {prefix}{cliff[0].offset} -> {prefix}{cliff[1].offset}: "
                            f"retention drops from {cliff[0].rate * 100:.1f}% to {cliff[1].rate * 100:.1f}%."
                        ),
                        action=f"Target re-engagement before {prefix}{cliff[1].offset} (lifecycle email, push, content).",
                        metric={"from": cliff[0].to_dict(), "to": cliff[1].to_dict()},
                        confidence=0.7,
                        severity="info",
                        category="retention",
                    )
                )

        if trend.get("available") and trend.get("direction") == "declining":
            insights.append(
                Insight(
                    title="Newer cohorts retaining worse",
                    insight=(
                        f"{prefix}{trend['horizon']} retention is declining across cohorts "
                        f"({trend['first_cohort_rate'] * 100:.1f}% -> {trend['last_cohort_rate'] * 100:.1f}%)."
                    ),
                    action="Investigate recent product/marketing changes; quality of newly acquired users may be dropping.",
                    metric=trend,
                    confidence=0.65,
                    severity="warning",
                    category="retention",
                )
            )

        if len(segments) >= 2:
            scored = [(s, self._segment_score(s)) for s in segments if self._segment_score(s) is not None]
            if len(scored) >= 2:
                best = max(scored, key=lambda x: x[1])
                worst = min(scored, key=lambda x: x[1])
                if best[1] - worst[1] > 0.05:
                    insights.append(
                        Insight(
                            title="Retention varies by segment",
                            insight=(
                                f"Segment '{best[0].segment}' retains at {best[1] * 100:.1f}% vs "
                                f"'{worst[0].segment}' at {worst[1] * 100:.1f}% (later horizon)."
                            ),
                            action=f"Prioritise '{worst[0].segment}' for retention experiments; learn from '{best[0].segment}'.",
                            metric={"best": best[0].segment, "worst": worst[0].segment},
                            confidence=0.6,
                            severity="info",
                            category="retention",
                        )
                    )

        if churn.engaged_users > 0 and churn.at_risk_rate > 0:
            sev = "warning" if churn.at_risk_rate >= 0.2 else "info"
            insights.append(
                Insight(
                    title="Users at risk of churn",
                    insight=(
                        f"{churn.at_risk:,} of {churn.engaged_users:,} engaged users "
                        f"({churn.at_risk_rate * 100:.1f}%) have been inactive beyond their normal cadence "
                        f"(> {churn.threshold_days:.0f} days)."
                    ),
                    action="Trigger win-back campaigns for the at-risk list; feed these features to a churn model.",
                    metric={"at_risk": churn.at_risk, "at_risk_rate": round(churn.at_risk_rate, 4), "threshold_days": round(churn.threshold_days, 1)},
                    confidence=0.6,
                    severity=sev,
                    category="retention",
                )
            )

        return insights

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _order_cohorts(labels: pd.Index) -> list[str]:
        labels = [str(l) for l in labels]
        try:
            return [str(p) for p in sorted(labels, key=lambda x: pd.Period(x))]
        except Exception:
            return sorted(labels)

    @staticmethod
    def _pick_headline_point(curve: list[RetentionPoint]) -> RetentionPoint | None:
        # Prefer a classic mid-horizon (7) if present, else the largest available.
        by_offset = {p.offset: p for p in curve}
        for preferred in (7, 30, 1):
            if preferred in by_offset:
                return by_offset[preferred]
        non_zero = [p for p in curve if p.offset > 0]
        return non_zero[-1] if non_zero else (curve[-1] if curve else None)

    @staticmethod
    def _biggest_cliff(curve: list[RetentionPoint]) -> tuple[RetentionPoint, RetentionPoint] | None:
        if len(curve) < 2:
            return None
        worst = None
        worst_drop = -1.0
        for a, b in zip(curve, curve[1:]):
            drop = a.rate - b.rate
            if drop > worst_drop:
                worst_drop = drop
                worst = (a, b)
        return worst

    @staticmethod
    def _segment_score(segment: SegmentRetention) -> float | None:
        later = [p for p in segment.curve if p.offset > 0]
        return later[-1].rate if later else None

    @staticmethod
    def _curve_chart(curve: list[RetentionPoint], unit: str) -> ChartSpec | None:
        if not curve:
            return None
        prefix = "D" if unit == "day" else "W"
        return ChartSpec(
            kind="line",
            title="Retention curve",
            data={
                "x": [f"{prefix}{p.offset}" for p in curve],
                "y": [round(p.rate, 4) for p in curve],
                "ci_low": [round(p.ci_low, 4) for p in curve],
                "ci_high": [round(p.ci_high, 4) for p in curve],
            },
            layout={"xaxis_title": "horizon", "yaxis_title": "retention rate"},
        )


def json_user(value: Any) -> Any:
    """Coerce a user identifier into a JSON-native scalar."""

    return safe_float(value) if isinstance(value, (np.floating,)) else (
        int(value) if isinstance(value, (np.integer,)) else value
    )


def analyze_retention(df: pd.DataFrame, *, user_col: str, timestamp_col: str, **kwargs: Any) -> RetentionResult:
    """Convenience wrapper around :class:`RetentionAnalyzer`."""

    return RetentionAnalyzer().analyze(df, user_col=user_col, timestamp_col=timestamp_col, **kwargs)
