"""Capability 3 -- Funnel Analysis (Mixpanel / Firebase style).

Tracks how a population flows through an ordered sequence of steps, where it
leaks, and how confident we can be in each conversion rate. Two input shapes are
supported:

* **Event log (long)** -- one row per event with user / event / (optional)
  timestamp columns. Conversion is counted on *unique users*, and with a
  timestamp the funnel is enforced sequentially (step ``k`` must happen at or
  after step ``k-1``), exactly like product analytics tools.
* **Wide / indicator** -- one row per unit with a boolean (or 0/1) column per
  step.

Every conversion rate carries a Wilson confidence interval so a 60% rate on 20
users is never confused with 60% on 20,000.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .base import ChartSpec, Insight, sort_insights
from .utils import coerce_datetime, wilson_interval

MAX_SEGMENTS = 12


@dataclass
class FunnelStep:
    name: str
    users: int
    conversion_from_prev: float
    conversion_from_top: float
    dropoff: int
    dropoff_rate: float
    ci_low: float
    ci_high: float
    median_seconds_to_next: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "users": self.users,
            "conversion_from_prev": round(self.conversion_from_prev, 4),
            "conversion_from_top": round(self.conversion_from_top, 4),
            "dropoff": self.dropoff,
            "dropoff_rate": round(self.dropoff_rate, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
        }
        if self.median_seconds_to_next is not None:
            data["median_seconds_to_next"] = round(self.median_seconds_to_next, 2)
        return data


@dataclass
class SegmentFunnel:
    segment: str
    top_users: int
    bottom_users: int
    overall_conversion: float
    ci_low: float
    ci_high: float
    step_users: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment": self.segment,
            "top_users": self.top_users,
            "bottom_users": self.bottom_users,
            "overall_conversion": round(self.overall_conversion, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "step_users": self.step_users,
        }


@dataclass
class FunnelResult:
    steps: list[FunnelStep]
    overall_conversion: float
    overall_ci: tuple[float, float]
    biggest_dropoff: dict[str, Any] | None
    segments: list[SegmentFunnel]
    insights: list[Insight]
    visualization: ChartSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "overall_conversion": round(self.overall_conversion, 4),
            "overall_ci": [round(self.overall_ci[0], 4), round(self.overall_ci[1], 4)],
            "biggest_dropoff": self.biggest_dropoff,
            "segments": [s.to_dict() for s in self.segments],
            "visualization": self.visualization.to_dict() if self.visualization else None,
            "insights": [i.to_dict() for i in self.insights],
        }


class FunnelAnalyzer:
    """Compute conversion funnels with dropout, segments and confidence intervals."""

    def analyze(
        self,
        df: pd.DataFrame,
        steps: list[str],
        *,
        user_col: str | None = None,
        event_col: str | None = None,
        timestamp_col: str | None = None,
        segment_col: str | None = None,
        ordered: bool = True,
    ) -> FunnelResult:
        if len(steps) < 2:
            raise ValueError("A funnel needs at least two steps.")

        reached, piv, seg_per_user = self._reached_matrix(
            df, steps, user_col, event_col, timestamp_col, segment_col, ordered
        )
        if reached.empty:
            raise ValueError("No users matched any funnel step; check the step names and columns.")

        counts = [int(reached[step].sum()) for step in steps]
        funnel_steps = self._build_steps(steps, counts, piv, reached, timestamp_col)

        top = counts[0]
        bottom = counts[-1]
        overall = bottom / top if top else 0.0
        _, lo, hi = wilson_interval(bottom, top)

        biggest = self._biggest_dropoff(funnel_steps)
        segments = self._segment_funnels(steps, reached, seg_per_user) if seg_per_user is not None else []
        insights = self._build_insights(funnel_steps, overall, (lo, hi), biggest, segments)

        return FunnelResult(
            steps=funnel_steps,
            overall_conversion=overall,
            overall_ci=(lo, hi),
            biggest_dropoff=biggest,
            segments=segments,
            insights=sort_insights(insights),
            visualization=self._funnel_chart(funnel_steps),
        )

    # --------------------------------------------------------- reached matrix
    def _reached_matrix(
        self,
        df: pd.DataFrame,
        steps: list[str],
        user_col: str | None,
        event_col: str | None,
        timestamp_col: str | None,
        segment_col: str | None,
        ordered: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.Series | None]:
        """Return a boolean ``user x step`` reached matrix (and timestamp pivot).

        ``reached[k]`` is monotone non-increasing across steps by construction, so
        downstream counting and segmentation are trivial sums.
        """

        if event_col is not None:
            if user_col is None:
                raise ValueError("user_col is required for event-log funnels.")
            piv, seg = self._event_pivot(df, steps, user_col, event_col, timestamp_col, segment_col)
            has_time = timestamp_col is not None
            present = piv.notna() if has_time else (piv.fillna(0) > 0)
        else:
            # Wide / indicator format: one row per unit, one column per step.
            missing = [s for s in steps if s not in df.columns]
            if missing:
                raise ValueError(f"Step columns not found in dataset: {missing}")
            present = df[steps].apply(lambda c: c.fillna(0).astype(bool) if not pd.api.types.is_bool_dtype(c) else c.fillna(False))
            piv = None
            seg = df[segment_col] if segment_col and segment_col in df.columns else None

        if timestamp_col is not None and ordered and piv is not None:
            reached = self._sequential_reach(piv, steps)
        else:
            reached = self._cumulative_reach(present, steps)

        return reached, piv, seg

    @staticmethod
    def _event_pivot(
        df: pd.DataFrame,
        steps: list[str],
        user_col: str,
        event_col: str,
        timestamp_col: str | None,
        segment_col: str | None,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        sub = df[df[event_col].isin(steps)].copy()
        seg = None
        if timestamp_col is not None:
            sub[timestamp_col] = coerce_datetime(sub[timestamp_col])
            piv = sub.groupby([user_col, event_col])[timestamp_col].min().unstack(event_col)
        else:
            piv = sub.groupby([user_col, event_col]).size().unstack(event_col)
        piv = piv.reindex(columns=steps)
        if segment_col and segment_col in df.columns:
            seg = df.groupby(user_col)[segment_col].first()
            seg = seg.reindex(piv.index)
        return piv, seg

    @staticmethod
    def _sequential_reach(piv: pd.DataFrame, steps: list[str]) -> pd.DataFrame:
        reached = pd.DataFrame(index=piv.index)
        mask = piv[steps[0]].notna()
        prev_time = piv[steps[0]]
        reached[steps[0]] = mask
        for step in steps[1:]:
            step_time = piv[step]
            ok = mask & step_time.notna() & (step_time >= prev_time)
            reached[step] = ok
            mask = ok
            prev_time = step_time.where(ok)
        return reached

    @staticmethod
    def _cumulative_reach(present: pd.DataFrame, steps: list[str]) -> pd.DataFrame:
        reached = pd.DataFrame(index=present.index)
        cum = present[steps[0]].astype(bool)
        reached[steps[0]] = cum
        for step in steps[1:]:
            cum = cum & present[step].astype(bool)
            reached[step] = cum
        return reached

    # ------------------------------------------------------------ step builder
    def _build_steps(
        self,
        steps: list[str],
        counts: list[int],
        piv: pd.DataFrame | None,
        reached: pd.DataFrame,
        timestamp_col: str | None,
    ) -> list[FunnelStep]:
        top = counts[0] or 1
        result: list[FunnelStep] = []
        for i, (name, count) in enumerate(zip(steps, counts)):
            prev = counts[i - 1] if i > 0 else count
            conv_prev = count / prev if prev else 0.0
            conv_top = count / top
            dropoff = max(prev - count, 0) if i > 0 else 0
            drop_rate = dropoff / prev if (i > 0 and prev) else 0.0
            _, lo, hi = wilson_interval(count, prev if i > 0 else top)
            median_secs = None
            if piv is not None and timestamp_col is not None and i < len(steps) - 1:
                median_secs = self._median_time(piv, reached, steps[i], steps[i + 1])
            result.append(
                FunnelStep(
                    name=name,
                    users=count,
                    conversion_from_prev=conv_prev,
                    conversion_from_top=conv_top,
                    dropoff=dropoff,
                    dropoff_rate=drop_rate,
                    ci_low=lo,
                    ci_high=hi,
                    median_seconds_to_next=median_secs,
                )
            )
        return result

    @staticmethod
    def _median_time(piv: pd.DataFrame, reached: pd.DataFrame, step_a: str, step_b: str) -> float | None:
        mask = reached[step_b]
        if not mask.any():
            return None
        delta = (piv.loc[mask, step_b] - piv.loc[mask, step_a]).dropna()
        if delta.empty:
            return None
        return float(delta.dt.total_seconds().median())

    @staticmethod
    def _biggest_dropoff(steps: list[FunnelStep]) -> dict[str, Any] | None:
        transitions = steps[1:]
        if not transitions:
            return None
        worst = max(transitions, key=lambda s: s.dropoff_rate)
        idx = steps.index(worst)
        return {
            "from_step": steps[idx - 1].name,
            "to_step": worst.name,
            "dropoff": worst.dropoff,
            "dropoff_rate": round(worst.dropoff_rate, 4),
        }

    # ---------------------------------------------------------------- segments
    def _segment_funnels(
        self,
        steps: list[str],
        reached: pd.DataFrame,
        seg_per_user: pd.Series,
    ) -> list[SegmentFunnel]:
        seg = seg_per_user.reindex(reached.index).fillna("(unknown)")
        top_sizes = seg.groupby(seg).size().sort_values(ascending=False)
        results: list[SegmentFunnel] = []
        for value in top_sizes.head(MAX_SEGMENTS).index:
            mask = (seg == value).to_numpy()
            sub = reached.loc[mask]
            counts = [int(sub[step].sum()) for step in steps]
            top = counts[0]
            bottom = counts[-1]
            if top == 0:
                continue
            _, lo, hi = wilson_interval(bottom, top)
            results.append(
                SegmentFunnel(
                    segment=str(value),
                    top_users=top,
                    bottom_users=bottom,
                    overall_conversion=bottom / top,
                    ci_low=lo,
                    ci_high=hi,
                    step_users=counts,
                )
            )
        return results

    # ---------------------------------------------------------------- insights
    def _build_insights(
        self,
        steps: list[FunnelStep],
        overall: float,
        ci: tuple[float, float],
        biggest: dict[str, Any] | None,
        segments: list[SegmentFunnel],
    ) -> list[Insight]:
        insights: list[Insight] = []

        insights.append(
            Insight(
                title="Overall funnel conversion",
                insight=(
                    f"{overall * 100:.1f}% of users complete the full funnel "
                    f"({steps[-1].users:,} of {steps[0].users:,}), 95% CI "
                    f"[{ci[0] * 100:.1f}%, {ci[1] * 100:.1f}%]."
                ),
                action="Compare against your target conversion; focus optimisation on the largest leak below.",
                metric={"overall_conversion": round(overall, 4), "top_users": steps[0].users, "bottom_users": steps[-1].users},
                confidence=self._ci_confidence(steps[0].users),
                severity="info",
                category="funnel",
                visualization=self._funnel_chart(steps),
            )
        )

        if biggest and biggest["dropoff_rate"] > 0:
            sev = "critical" if biggest["dropoff_rate"] >= 0.5 else "warning"
            insights.append(
                Insight(
                    title="Biggest funnel drop-off",
                    insight=(
                        f"The largest leak is '{biggest['from_step']}' -> '{biggest['to_step']}': "
                        f"{biggest['dropoff_rate'] * 100:.1f}% of users ({biggest['dropoff']:,}) drop here."
                    ),
                    action=f"Investigate friction in the '{biggest['to_step']}' step (UX, latency, requirements, messaging).",
                    metric=biggest,
                    confidence=0.8,
                    severity=sev,
                    category="funnel",
                )
            )

        # Time-to-convert, if available.
        timed = [s for s in steps if s.median_seconds_to_next is not None]
        if timed:
            slowest = max(timed, key=lambda s: s.median_seconds_to_next or 0)
            insights.append(
                Insight(
                    title="Slowest funnel transition",
                    insight=(
                        f"Users take the longest moving on from '{slowest.name}' "
                        f"(median {self._humanize(slowest.median_seconds_to_next)})."
                    ),
                    action="Long gaps signal hesitation or a multi-session journey; consider nudges/reminders here.",
                    metric={"step": slowest.name, "median_seconds": round(slowest.median_seconds_to_next or 0, 1)},
                    confidence=0.6,
                    severity="info",
                    category="funnel",
                )
            )

        # Segment spread.
        if len(segments) >= 2:
            best = max(segments, key=lambda s: s.overall_conversion)
            worst = min(segments, key=lambda s: s.overall_conversion)
            if best.overall_conversion - worst.overall_conversion > 0.05:
                insights.append(
                    Insight(
                        title="Conversion varies by segment",
                        insight=(
                            f"'{best.segment}' converts at {best.overall_conversion * 100:.1f}% vs "
                            f"'{worst.segment}' at {worst.overall_conversion * 100:.1f}%."
                        ),
                        action=f"Study what '{best.segment}' does differently; replicate it for '{worst.segment}'.",
                        metric={"best": best.to_dict(), "worst": worst.to_dict()},
                        confidence=0.7,
                        severity="info",
                        category="funnel",
                    )
                )

        return insights

    @staticmethod
    def _funnel_chart(steps: list[FunnelStep]) -> ChartSpec:
        return ChartSpec(
            kind="funnel",
            title="Conversion funnel",
            data={
                "stages": [s.name for s in steps],
                "users": [s.users for s in steps],
                "conversion_from_top": [round(s.conversion_from_top, 4) for s in steps],
            },
        )

    @staticmethod
    def _ci_confidence(n: int) -> float:
        # More users => tighter CI => more confidence in the conversion figure.
        if n <= 0:
            return 0.1
        return float(min(0.95, 0.3 + 0.65 * (np.log10(max(n, 1)) / 5.0)))

    @staticmethod
    def _humanize(seconds: float | None) -> str:
        if seconds is None:
            return "n/a"
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}m"
        if seconds < 86400:
            return f"{seconds / 3600:.1f}h"
        return f"{seconds / 86400:.1f}d"


def analyze_funnel(df: pd.DataFrame, steps: list[str], **kwargs: Any) -> FunnelResult:
    """Convenience wrapper around :class:`FunnelAnalyzer`."""

    return FunnelAnalyzer().analyze(df, steps, **kwargs)
