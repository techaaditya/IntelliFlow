"""Capability 5 -- Event Stream Analytics.

Treats a timestamped event log the way a product-analytics backend would:
aggregates volume over time, measures whether activity is accelerating or
decaying (velocity), reconstructs sessions from inter-event gaps, mines the most
common short journeys (event n-grams / transitions), and estimates time-to-event
with a Kaplan-Meier survival curve that correctly censors users who never
converted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .base import ChartSpec, Insight, safe_float, sort_insights
from .utils import coerce_datetime

FreqAlias = Literal["minute", "hour", "day", "week"]
_FREQ_MAP = {"minute": "min", "hour": "h", "day": "D", "week": "W"}
MAX_SEQUENCES = 10
MAX_TRANSITIONS = 15


@dataclass
class SessionStats:
    n_sessions: int
    n_users: int
    sessions_per_user: float
    mean_events_per_session: float
    median_events_per_session: float
    mean_duration_seconds: float
    median_duration_seconds: float
    timeout_minutes: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_sessions": self.n_sessions,
            "n_users": self.n_users,
            "sessions_per_user": round(self.sessions_per_user, 3),
            "mean_events_per_session": round(self.mean_events_per_session, 3),
            "median_events_per_session": round(self.median_events_per_session, 3),
            "mean_duration_seconds": round(self.mean_duration_seconds, 2),
            "median_duration_seconds": round(self.median_duration_seconds, 2),
            "timeout_minutes": self.timeout_minutes,
        }


@dataclass
class SequencePattern:
    sequence: list[str]
    count: int
    share: float

    def to_dict(self) -> dict[str, Any]:
        return {"sequence": self.sequence, "count": self.count, "share": round(self.share, 4)}


@dataclass
class SurvivalCurve:
    event_label: str
    unit: str
    times: list[float]
    survival: list[float]
    at_risk: list[int]
    median_time: float | None
    n_events: int
    n_censored: int
    estimator: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_label": self.event_label,
            "unit": self.unit,
            "times": [round(t, 4) for t in self.times],
            "survival": [round(s, 4) for s in self.survival],
            "at_risk": self.at_risk,
            "median_time": None if self.median_time is None else round(self.median_time, 4),
            "n_events": self.n_events,
            "n_censored": self.n_censored,
            "estimator": self.estimator,
        }


@dataclass
class EventStreamResult:
    freq: str
    n_events: int
    start: str | None
    end: str | None
    span_seconds: float
    timeseries: list[dict[str, Any]]
    velocity: dict[str, Any]
    sessions: SessionStats | None
    top_sequences: list[SequencePattern]
    top_transitions: list[dict[str, Any]]
    survival: SurvivalCurve | None
    insights: list[Insight]
    timeseries_chart: ChartSpec | None = None
    survival_chart: ChartSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "freq": self.freq,
            "n_events": self.n_events,
            "start": self.start,
            "end": self.end,
            "span_seconds": round(self.span_seconds, 2),
            "timeseries": self.timeseries,
            "velocity": self.velocity,
            "sessions": self.sessions.to_dict() if self.sessions else None,
            "top_sequences": [s.to_dict() for s in self.top_sequences],
            "top_transitions": self.top_transitions,
            "survival": self.survival.to_dict() if self.survival else None,
            "timeseries_chart": self.timeseries_chart.to_dict() if self.timeseries_chart else None,
            "survival_chart": self.survival_chart.to_dict() if self.survival_chart else None,
            "insights": [i.to_dict() for i in self.insights],
        }


class EventStreamAnalyzer:
    """Aggregation, sessions, journeys and survival over an event stream."""

    def analyze(
        self,
        df: pd.DataFrame,
        *,
        timestamp_col: str,
        user_col: str | None = None,
        event_col: str | None = None,
        freq: FreqAlias | str = "day",
        session_timeout_minutes: float = 30.0,
        target_event: str | None = None,
    ) -> EventStreamResult:
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column {timestamp_col!r} not found in dataset.")

        ts = coerce_datetime(df[timestamp_col])
        work = df.copy()
        work["_ts"] = ts
        work = work.dropna(subset=["_ts"])
        if work.empty:
            raise ValueError("No rows with a parseable timestamp were found.")

        work = work.sort_values("_ts")
        start, end = work["_ts"].min(), work["_ts"].max()
        span = float((end - start).total_seconds())

        timeseries, ts_chart, alias = self._aggregate(work["_ts"], freq)
        velocity = self._velocity(timeseries)

        sessions = sessionized = None
        if user_col and user_col in work.columns:
            sessions, sessionized = self._sessions(work, user_col, event_col, session_timeout_minutes)

        sequences: list[SequencePattern] = []
        transitions: list[dict[str, Any]] = []
        if event_col and event_col in work.columns and user_col and user_col in work.columns:
            sequences, transitions = self._sequences(sessionized, work, user_col, event_col)

        survival = surv_chart = None
        if target_event is not None and event_col and user_col:
            survival = self._survival(work, user_col, event_col, target_event, span)
            surv_chart = self._survival_chart(survival)

        insights = self._build_insights(
            timeseries, velocity, sessions, sequences, transitions, survival, alias
        )

        return EventStreamResult(
            freq=alias,
            n_events=int(work.shape[0]),
            start=start.isoformat(),
            end=end.isoformat(),
            span_seconds=span,
            timeseries=timeseries,
            velocity=velocity,
            sessions=sessions,
            top_sequences=sequences,
            top_transitions=transitions,
            survival=survival,
            insights=sort_insights(insights),
            timeseries_chart=ts_chart,
            survival_chart=surv_chart,
        )

    # ----------------------------------------------------------- aggregation
    def _aggregate(self, ts: pd.Series, freq: str) -> tuple[list[dict[str, Any]], ChartSpec, str]:
        alias = _FREQ_MAP.get(freq, freq)
        series = pd.Series(1, index=pd.DatetimeIndex(ts.values)).resample(alias).sum()
        points = [{"period": idx.isoformat(), "count": int(val)} for idx, val in series.items()]
        chart = ChartSpec(
            kind="line",
            title=f"Events per {freq}",
            data={"x": [p["period"] for p in points], "y": [p["count"] for p in points]},
            layout={"xaxis_title": "time", "yaxis_title": "events"},
        )
        return points, chart, freq

    def _velocity(self, timeseries: list[dict[str, Any]]) -> dict[str, Any]:
        counts = np.array([p["count"] for p in timeseries], dtype="float64")
        if counts.size < 2:
            return {"available": False, "n_periods": int(counts.size)}
        x = np.arange(counts.size, dtype="float64")
        slope = float(np.polyfit(x, counts, 1)[0])
        mean = float(counts.mean())
        half = counts.size // 2
        first_half = float(counts[:half].mean()) if half else mean
        second_half = float(counts[half:].mean())
        growth = (second_half - first_half) / first_half if first_half else 0.0
        direction = "accelerating" if growth > 0.05 else ("decelerating" if growth < -0.05 else "steady")
        peak_idx = int(counts.argmax())
        return {
            "available": True,
            "n_periods": int(counts.size),
            "mean_per_period": round(mean, 3),
            "slope_per_period": round(slope, 4),
            "normalized_slope": round(slope / mean, 5) if mean else 0.0,
            "first_half_mean": round(first_half, 3),
            "second_half_mean": round(second_half, 3),
            "growth_rate": round(growth, 4),
            "direction": direction,
            "peak_period": timeseries[peak_idx]["period"],
            "peak_count": int(counts[peak_idx]),
        }

    # -------------------------------------------------------------- sessions
    def _sessions(
        self,
        work: pd.DataFrame,
        user_col: str,
        event_col: str | None,
        timeout_minutes: float,
    ) -> tuple[SessionStats, pd.DataFrame]:
        d = work[[user_col, "_ts"] + ([event_col] if event_col and event_col in work.columns else [])].copy()
        d = d.sort_values([user_col, "_ts"])
        gap = d.groupby(user_col, observed=True)["_ts"].diff()
        timeout = pd.Timedelta(minutes=timeout_minutes)
        new_session = gap.isna() | (gap > timeout)
        # Global cumulative sum over time-sorted-by-user rows gives unique ids.
        d["session_id"] = new_session.cumsum()

        grouped = d.groupby("session_id")
        events_per = grouped.size()
        durations = grouped["_ts"].agg(lambda s: (s.max() - s.min()).total_seconds())
        n_users = int(d[user_col].nunique())

        stats = SessionStats(
            n_sessions=int(events_per.shape[0]),
            n_users=n_users,
            sessions_per_user=events_per.shape[0] / n_users if n_users else 0.0,
            mean_events_per_session=float(events_per.mean()),
            median_events_per_session=float(events_per.median()),
            mean_duration_seconds=float(durations.mean()),
            median_duration_seconds=float(durations.median()),
            timeout_minutes=timeout_minutes,
        )
        return stats, d

    # ------------------------------------------------------------ sequences
    def _sequences(
        self,
        sessionized: pd.DataFrame | None,
        work: pd.DataFrame,
        user_col: str,
        event_col: str,
    ) -> tuple[list[SequencePattern], list[dict[str, Any]]]:
        if sessionized is not None and "session_id" in sessionized.columns and event_col in sessionized.columns:
            seqs = sessionized.groupby("session_id")[event_col].apply(list)
        else:
            seqs = work.sort_values([user_col, "_ts"]).groupby(user_col)[event_col].apply(list)

        bigram_counter: Counter = Counter()
        trigram_counter: Counter = Counter()
        transition_counter: Counter = Counter()
        for seq in seqs:
            seq = [str(s) for s in seq]
            for a, b in zip(seq, seq[1:]):
                transition_counter[(a, b)] += 1
                bigram_counter[(a, b)] += 1
            for a, b, c in zip(seq, seq[1:], seq[2:]):
                trigram_counter[(a, b, c)] += 1

        total_bigrams = sum(bigram_counter.values()) or 1
        total_trigrams = sum(trigram_counter.values()) or 1
        patterns: list[SequencePattern] = []
        for gram, count in trigram_counter.most_common(MAX_SEQUENCES):
            patterns.append(SequencePattern(list(gram), count, count / total_trigrams))
        if not patterns:
            for gram, count in bigram_counter.most_common(MAX_SEQUENCES):
                patterns.append(SequencePattern(list(gram), count, count / total_bigrams))

        transitions = [
            {"from": a, "to": b, "count": count, "share": round(count / total_bigrams, 4)}
            for (a, b), count in transition_counter.most_common(MAX_TRANSITIONS)
        ]
        return patterns, transitions

    # -------------------------------------------------------------- survival
    def _survival(
        self,
        work: pd.DataFrame,
        user_col: str,
        event_col: str,
        target_event: str,
        span_seconds: float,
    ) -> SurvivalCurve | None:
        first_ts = work.groupby(user_col)["_ts"].min()
        last_ts = work.groupby(user_col)["_ts"].max()
        target = work[work[event_col].astype(str) == str(target_event)]
        if target.empty:
            return None
        target_ts = target.groupby(user_col)["_ts"].min()

        unit = "days" if span_seconds >= 2 * 86400 else "hours"
        divisor = 86400.0 if unit == "days" else 3600.0

        durations: list[float] = []
        observed: list[int] = []
        for user in first_ts.index:
            entry = first_ts[user]
            if user in target_ts.index and target_ts[user] >= entry:
                dur = (target_ts[user] - entry).total_seconds()
                ev = 1
            else:
                dur = (last_ts[user] - entry).total_seconds()
                ev = 0
            durations.append(max(dur, 0.0) / divisor)
            observed.append(ev)

        durations_arr = np.asarray(durations, dtype="float64")
        observed_arr = np.asarray(observed, dtype="int64")
        if observed_arr.sum() == 0:
            return None

        times, surv, at_risk, estimator = self._kaplan_meier(durations_arr, observed_arr)
        median = self._median_survival(times, surv)
        return SurvivalCurve(
            event_label=str(target_event),
            unit=unit,
            times=times.tolist(),
            survival=surv.tolist(),
            at_risk=at_risk.tolist(),
            median_time=median,
            n_events=int(observed_arr.sum()),
            n_censored=int((observed_arr == 0).sum()),
            estimator=estimator,
        )

    @staticmethod
    def _kaplan_meier(durations: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Kaplan-Meier survival estimate; statsmodels when available, else manual.

        Manual fallback implements the standard product-limit estimator:
        ``S(t) = prod_{t_i <= t} (1 - d_i / n_i)`` over event times.
        """

        try:
            from statsmodels.duration.survfunc import SurvfuncRight

            sf = SurvfuncRight(durations, observed)
            times = np.asarray(sf.surv_times, dtype="float64")
            surv = np.asarray(sf.surv_prob, dtype="float64")
            at_risk = np.array([int((durations >= t).sum()) for t in times], dtype="int64")
            return times, surv, at_risk, "statsmodels"
        except Exception:
            pass

        order = np.argsort(durations)
        d_sorted = durations[order]
        e_sorted = observed[order]
        event_times = np.unique(d_sorted[e_sorted == 1])
        surv_prob = 1.0
        times_out, surv_out, risk_out = [], [], []
        for t in event_times:
            n_i = int((d_sorted >= t).sum())
            d_i = int(((d_sorted == t) & (e_sorted == 1)).sum())
            if n_i == 0:
                continue
            surv_prob *= 1.0 - d_i / n_i
            times_out.append(float(t))
            surv_out.append(float(surv_prob))
            risk_out.append(n_i)
        return np.asarray(times_out), np.asarray(surv_out), np.asarray(risk_out, dtype="int64"), "manual"

    @staticmethod
    def _median_survival(times: np.ndarray, surv: np.ndarray) -> float | None:
        below = np.where(surv <= 0.5)[0]
        if below.size == 0:
            return None
        return float(times[below[0]])

    # --------------------------------------------------------------- charts
    @staticmethod
    def _survival_chart(survival: SurvivalCurve | None) -> ChartSpec | None:
        if survival is None or not survival.times:
            return None
        return ChartSpec(
            kind="line",
            title=f"Time to '{survival.event_label}' (Kaplan-Meier)",
            data={"x": survival.times, "y": survival.survival},
            layout={"xaxis_title": survival.unit, "yaxis_title": "survival probability"},
        )

    # --------------------------------------------------------------- insights
    def _build_insights(
        self,
        timeseries: list[dict[str, Any]],
        velocity: dict[str, Any],
        sessions: SessionStats | None,
        sequences: list[SequencePattern],
        transitions: list[dict[str, Any]],
        survival: SurvivalCurve | None,
        freq: str,
    ) -> list[Insight]:
        insights: list[Insight] = []

        if velocity.get("available"):
            insights.append(
                Insight(
                    title="Event velocity trend",
                    insight=(
                        f"Event volume is {velocity['direction']} "
                        f"({velocity['first_half_mean']:.0f} -> {velocity['second_half_mean']:.0f} per {freq}, "
                        f"{velocity['growth_rate'] * 100:+.1f}%). Peak was {velocity['peak_count']:,} on {velocity['peak_period']}."
                    ),
                    action=(
                        "Investigate the decline (engagement, seasonality, instrumentation)."
                        if velocity["direction"] == "decelerating"
                        else "Confirm capacity can handle continued growth."
                    ),
                    metric=velocity,
                    confidence=0.7,
                    severity="warning" if velocity["direction"] == "decelerating" else "info",
                    category="events",
                    visualization=self._timeseries_chart(timeseries, freq),
                )
            )

        if sessions is not None and sessions.n_sessions > 0:
            insights.append(
                Insight(
                    title="Session behaviour",
                    insight=(
                        f"{sessions.n_sessions:,} sessions across {sessions.n_users:,} users "
                        f"({sessions.sessions_per_user:.1f}/user); median {sessions.median_events_per_session:.0f} events and "
                        f"{sessions.median_duration_seconds / 60:.1f} min per session."
                    ),
                    action="Short, shallow sessions suggest friction or unclear value; deepen core flows.",
                    metric=sessions.to_dict(),
                    confidence=0.75,
                    severity="info",
                    category="events",
                )
            )

        if sequences:
            top = sequences[0]
            insights.append(
                Insight(
                    title="Most common user journey",
                    insight=(
                        "The most frequent journey is "
                        + " -> ".join(top.sequence)
                        + f" ({top.count:,} times, {top.share * 100:.1f}% of journeys of this length)."
                    ),
                    action="Optimise and instrument this dominant path; ensure it leads toward value/conversion.",
                    metric={"top_sequences": [s.to_dict() for s in sequences[:5]]},
                    confidence=0.7,
                    severity="info",
                    category="events",
                )
            )

        if survival is not None:
            median_txt = (
                f"median {survival.median_time:.1f} {survival.unit}"
                if survival.median_time is not None
                else "median not reached (most users never converted)"
            )
            insights.append(
                Insight(
                    title=f"Time to '{survival.event_label}'",
                    insight=(
                        f"{survival.n_events:,} users reached '{survival.event_label}' "
                        f"({survival.n_censored:,} censored); {median_txt}."
                    ),
                    action="Compress time-to-value: streamline the path from first touch to this event.",
                    metric={
                        "median_time": survival.median_time,
                        "unit": survival.unit,
                        "n_events": survival.n_events,
                        "n_censored": survival.n_censored,
                    },
                    confidence=0.65,
                    severity="info",
                    category="events",
                    visualization=self._survival_chart(survival),
                )
            )

        return insights

    @staticmethod
    def _timeseries_chart(timeseries: list[dict[str, Any]], freq: str) -> ChartSpec:
        return ChartSpec(
            kind="line",
            title=f"Events per {freq}",
            data={"x": [p["period"] for p in timeseries], "y": [p["count"] for p in timeseries]},
            layout={"xaxis_title": "time", "yaxis_title": "events"},
        )


def analyze_events(df: pd.DataFrame, *, timestamp_col: str, **kwargs: Any) -> EventStreamResult:
    """Convenience wrapper around :class:`EventStreamAnalyzer`."""

    return EventStreamAnalyzer().analyze(df, timestamp_col=timestamp_col, **kwargs)
