"""Selective-prediction evaluation for *event* trading.

Why this module exists
----------------------
Conventional directional accuracy is a poor summary of a trading system. A model
that is 50.4% accurate on every bar can still be profitable if the few bars it is
genuinely confident about are the ones that move, and a model that is 60%
accurate can be a disaster if its edge sits entirely inside a cost threshold
that fees and slippage then eat.

The question this module answers is therefore not "how often was the sign
right?" but:

    *If I only act on the most confident 5% of bars, how often does the event I
    care about actually happen, and is that better than acting at random?*

That is a precision-versus-coverage question, and it is the decision-relevant
one for a system that is allowed to stay in cash.

Two design commitments
----------------------
**Thin evidence is reported, never dressed up.** Every precision figure carries
the number of events behind it and a binomial confidence interval. Three correct
out of four is a 95% interval spanning 19%-99%; printing "75% precision" without
that would be the single easiest way to manufacture a fake edge in this project.

**Lift, not just precision, is the headline.** A 90% event rate is only good if
the event is rare. A model that finds every event by predicting "yes" every bar
has 100% precision in a rising market and no edge at all. ``lift`` divides
precision by the unconditional base rate, so it cannot be inflated by a
favourable class balance, and it is the number a decision should rest on.

Scores are expected to be *ranking* scores (higher means "more likely"), not
calibrated probabilities. A conviction z-score of ``predicted_return /
predicted_volatility`` is the canonical case. Probability calibration lives in
:func:`brier_score` and :func:`calibration_table`, which are only meaningful when
a genuine probability is supplied.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


def event_study_record(
    *,
    symbol: str,
    interval: str,
    model_kind: str,
    horizon: int,
    seq_len: int,
    epochs: int,
    event: dict[str, Any],
    conviction: list[dict[str, Any]],
    val_rows: int,
    train_rows: int,
    holdout_rows: int,
    threshold: float | None = None,
    k: float | None = None,
) -> dict[str, Any]:
    """Assemble a JSON-serialisable record of one validation-only event study.

    ``holdout_rows`` is carried purely so the record states how much data was
    *not* touched. A study that quietly graded the holdout would otherwise look
    identical to one that respected the protocol, and the holdout is the only
    sample a strategy can be trusted on.
    """
    return {
        "symbol": symbol,
        "interval": interval,
        "model_kind": model_kind,
        "horizon": horizon,
        "seq_len": seq_len,
        "epochs": epochs,
        "event": {
            "threshold": threshold,
            "k": k,
            "n": event.get("n"),
            "n_events": event.get("n_events"),
            "event_rate": event.get("event_rate"),
            "n_tests": event.get("n_tests"),
            "verdict": event.get("verdict"),
            "significant_k": event.get("significant_k"),
            "by_k": event.get("by_k"),
            "selectivity_curve": event.get("selectivity_curve"),
        },
        "conviction": conviction,
        "split": {
            "train_rows": train_rows,
            "val_rows": val_rows,
            "holdout_rows_untouched": holdout_rows,
        },
    }


def save_event_report(
    record: dict[str, Any],
    report_dir: Path,
    *,
    label: str = "study",
) -> Path:
    """Write an event-study record to ``report_dir`` and return its path."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(record.get("symbol", "unknown"))).strip("-")
    path = report_dir / f"event-{label}-{slug}-{stamp}-{uuid.uuid4().hex[:6]}.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path


def load_event_reports(report_dir: Path) -> list[dict[str, Any]]:
    """Load every readable event-study record, newest first.

    A corrupt or partially-written file is skipped rather than raising: a
    monitoring view that refuses to load because one study crashed is worse
    than one that shows the rest.
    """
    report_dir = Path(report_dir)
    if not report_dir.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(report_dir.glob("event-*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict) or "event" not in payload:
            continue
        payload["_file"] = path.name
        records.append(payload)
    return records


__all__ = [
    "absolute_move_events",
    "brier_score",
    "calibration_table",
    "event_report",
    "event_study_record",
    "load_event_reports",
    "precision_at_k",
    "save_event_report",
    "selectivity_curve",
    "standardized_move_events",
]


def _clean(
    scores: np.ndarray | Sequence[float] | pd.Series,
    events: np.ndarray | Sequence[bool] | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Align and drop rows where either side is missing.

    Dropping a score without dropping its event (or the reverse) would shift the
    base rate and quietly inflate lift, so the two are filtered together.
    """
    score_array = np.asarray(scores, dtype=float).reshape(-1)
    event_array = np.asarray(
        pd.Series(events).astype("boolean") if _is_nullable(events) else events,
        dtype=float,
    ).reshape(-1)
    size = min(score_array.size, event_array.size)
    if size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    score_array, event_array = score_array[:size], event_array[:size]
    keep = np.isfinite(score_array) & np.isfinite(event_array)
    return score_array[keep], (event_array[keep] > 0.5).astype(float)


def _is_nullable(values: Any) -> bool:
    if isinstance(values, pd.Series):
        return bool(values.dtype == "boolean" or str(values.dtype).startswith("Int"))
    return False


def _wilson_interval(successes: int, total: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because event counts here are often
    single digits, exactly the regime where the normal approximation produces
    intervals that extend past 0 and 1.
    """
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    return (max(0.0, lower), min(1.0, upper))


def absolute_move_events(
    forward_return: pd.Series,
    *,
    up_threshold: float,
    down_threshold: float | None = None,
) -> pd.Series:
    """Label bars whose forward return cleared an absolute percentage threshold.

    This is the literal reading of a "+2% a day" objective: the threshold is the
    same for every symbol. Rows that moved *between* the two thresholds are
    labelled ``NaN`` rather than 0, because "rose 0.4%" is not the same claim as
    "failed to rise 2%", and a classifier cannot learn the difference if the two
    are merged.

    Caveat worth stating plainly: an absolute threshold is not comparable across
    assets. +2% is a routine day for a low-volatility altcoin and a rare event for
    BTC, so an event model fitted on one symbol does not transfer to another.
    :func:`standardized_move_events` is the cross-symbol-safe alternative.
    """
    if down_threshold is None:
        down_threshold = -abs(up_threshold)
    values = pd.Series(forward_return, dtype=float)
    return pd.Series(
        np.where(
            values >= abs(up_threshold), 1.0,
            np.where(values <= down_threshold, 0.0, np.nan),
        ),
        index=values.index,
    )


def standardized_move_events(
    forward_return: pd.Series,
    trailing_volatility: pd.Series,
    *,
    horizon: int = 1,
    k: float = 2.0,
    vol_floor: float = 1e-6,
) -> pd.Series:
    """Label bars whose forward move is a ``k``-sigma event *for that symbol*.

    The move is divided by the expected move over ``horizon`` bars, so a "2-sigma
    event" means the same thing for a quiet stablecoin-ish pair and a violent
    memecoin. Trailing volatility is strictly backward-looking, so labelling with
    it cannot leak future information into a training set.

    Rows inside the band are ``NaN`` (no decisive event) exactly as in
    :func:`absolute_move_events`.
    """
    values = pd.Series(forward_return, dtype=float)
    vol = pd.Series(trailing_volatility, dtype=float).reindex(values.index).ffill()
    scale = vol * math.sqrt(max(1, int(horizon)))
    scale = scale.replace(0.0, np.nan).clip(lower=float(vol_floor))
    z = values / scale
    return pd.Series(
        np.where(z >= k, 1.0, np.where(z <= -k, 0.0, np.nan)), index=values.index
    )


def _bonferroni_z(alpha: float, n_tests: int) -> float:
    """Two-sided normal quantile for a Bonferroni-adjusted significance level.

    Widening the interval this way is what keeps a sweep over 20 coverage levels
    from manufacturing a significant result out of noise: the effective
    per-comparison alpha is ``alpha / n_tests``.
    """
    adjusted = float(alpha) / max(1, int(n_tests))
    return NormalDist().inv_cdf(1.0 - adjusted / 2.0)


def _longest_true_run(
    points: Sequence[dict[str, Any]], *, min_run: int = 2
) -> list[dict[str, Any]]:
    """Longest run of adjacent coverage levels that are individually significant.

    A real edge produces a plateau across neighbouring coverage levels. An
    isolated significant point is what noise looks like after a sweep, and
    ``min_run`` rejects it.
    """
    best: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for point in points:
        if point.get("significant"):
            current.append(point)
            if len(current) > len(best):
                best = list(current)
        else:
            current = []
    return best if len(best) >= min_run else []


def precision_at_k(
    scores: np.ndarray | Sequence[float],
    events: np.ndarray | Sequence[bool],
    k: float,
    *,
    z: float = 1.959963985,
) -> dict[str, Any]:
    """Precision, recall and lift when acting on the most confident ``k`` fraction.

    ``k`` is a fraction of bars, not a count, so it means the same thing for a
    symbol with 400 test bars and one with 40,000.
    """
    if not 0.0 < k <= 1.0:
        raise ValueError(f"k must be in (0, 1], got {k!r}")

    score_array, event_array = _clean(scores, events)
    n = score_array.size
    if n == 0:
        return {
            "k": float(k), "n": 0, "n_events": 0, "n_taken": 0,
            "precision": None, "precision_ci_low": None, "precision_ci_high": None,
            "recall": None, "lift": None, "base_rate": None, "significant": False,
        }

    base_rate = float(event_array.mean())
    take = max(1, int(round(k * n)))
    # Highest score = strongest belief the event occurs. Ties are broken by
    # original position so a constant-score model still yields a stable slice
    # rather than an arbitrary one.
    order = np.lexsort((np.arange(n), -score_array))[:take]
    taken = event_array[order]
    hits = int(taken.sum())
    precision = hits / take
    recall = hits / float(event_array.sum()) if event_array.sum() else None
    lift = precision / base_rate if base_rate > 0 else None
    low, high = _wilson_interval(hits, take)

    return {
        "k": float(k),
        "n": int(n),
        "n_events": int(event_array.sum()),
        "n_taken": int(take),
        "n_hits": hits,
        "precision": float(precision),
        "precision_ci_low": float(low),
        "precision_ci_high": float(high),
        "recall": float(recall) if recall is not None else None,
        "lift": float(lift) if lift is not None else None,
        "base_rate": base_rate,
        # The honest headline test: does the interval exclude the base rate? If
        # it does not, acting on this slice is not distinguishable from random.
        "significant": bool(base_rate > 0 and low > base_rate),
    }


def selectivity_curve(
    scores: np.ndarray | Sequence[float],
    events: np.ndarray | Sequence[bool],
    *,
    n_points: int = 20,
    min_taken: int = 10,
    z: float = 1.959963985,
) -> list[dict[str, Any]]:
    """Precision and lift across coverage levels, from very selective to broad.

    The interesting question for a selective strategy is not a single operating
    point but the whole curve: it shows whether any coverage level beats random
    behaviour, and whether the profitable region is so narrow it could never
    carry enough trades to be real.

    Slices holding fewer than ``min_taken`` observations are omitted rather than
    reported, because a curve that ends in a spike of 100% precision on 3 bars
    invites exactly the over-reading this module exists to prevent.
    """
    score_array, event_array = _clean(scores, events)
    n = score_array.size
    if n == 0:
        return []

    points: list[dict[str, Any]] = []
    for i in range(1, n_points + 1):
        k = i / n_points
        if k * n < min_taken:
            break
        points.append(precision_at_k(score_array, event_array, k, z=z))
    return points


def event_report(
    scores: np.ndarray | Sequence[float],
    events: np.ndarray | Sequence[bool],
    *,
    top_fractions: Sequence[float] = (0.01, 0.05, 0.10, 0.20, 0.50),
    n_curve_points: int = 20,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Full selective-performance report for one scored series.

    Returns the per-coverage table, the selectivity curve, and a ``verdict``
    string that states plainly whether *any* coverage level showed an edge that
    survives its own confidence interval.

    Two corrections guard the verdict, because scanning coverage levels is itself
    a multiple-testing problem. Evaluating 20 thresholds and reporting the most
    impressive one finds a "significant" result by chance most of the time even
    when the scores are pure noise.

    * **Bonferroni adjustment.** The interval is widened by the number of
      coverage levels examined, so the family-wise error rate holds.
    * **Contiguity requirement.** Significance must hold across a *neighbouring
      run* of coverage levels. A genuine edge is a plateau -- if the model knows
      something at 10%-15% coverage it also knows it at 8%-12% -- whereas a
      spurious point is an isolated spike.
    """
    score_array, event_array = _clean(scores, events)
    n = score_array.size

    top_list = [k for k in top_fractions if 0.0 < k <= 1.0]
    n_tests = max(1, len(top_list) + n_curve_points)
    z = _bonferroni_z(alpha, n_tests)

    by_k = [precision_at_k(score_array, event_array, k, z=z) for k in top_list]
    curve = selectivity_curve(score_array, event_array, n_points=n_curve_points, z=z)

    all_points = sorted(
        {round(p["k"], 10): p for p in (*by_k, *curve)}.values(),
        key=lambda p: p["k"],
    )
    best_run = _longest_true_run(all_points)

    if n == 0:
        verdict = "no-scored-bars"
    elif event_array.sum() == 0:
        verdict = "no-events-in-window"
    elif not best_run:
        verdict = "no-edge"
    else:
        peak = max(best_run, key=lambda p: (p["lift"] or 0.0, p["n_taken"]))
        verdict = (
            f"candidate-edge@k{peak['k']:.2f} lift={peak['lift']:.2f} "
            f"on {peak['n_taken']} bars "
            f"({len(best_run)} adjacent coverage levels significant)"
        )

    return {
        "n": int(n),
        "n_events": int(event_array.sum()) if n else 0,
        "event_rate": float(event_array.mean()) if n else None,
        "n_tests": n_tests,
        "by_k": by_k,
        "selectivity_curve": curve,
        "significant_k": [p["k"] for p in best_run],
        "verdict": verdict,
    }


def brier_score(probability: np.ndarray | Sequence[float], events: np.ndarray | Sequence[bool]) -> float | None:
    """Mean squared error of probability forecasts, only for real probabilities.

    Returns ``None`` for a ranking score fed in by mistake: a z-score is not a
    probability, and scoring one as if it were produces a number that looks
    meaningful and means nothing.
    """
    prob = np.asarray(probability, dtype=float).reshape(-1)
    event = np.asarray(events, dtype=float).reshape(-1)
    size = min(prob.size, event.size)
    if size == 0:
        return None
    prob, event = prob[:size], event[:size]
    keep = np.isfinite(prob) & np.isfinite(event)
    prob, event = prob[keep], (event[keep] > 0.5).astype(float)
    if prob.size == 0:
        return None
    if prob.min() < 0.0 or prob.max() > 1.0:
        raise ValueError(
            f"brier_score needs probabilities in [0, 1], got range "
            f"[{prob.min():.4f}, {prob.max():.4f}]; a ranking score is not a probability"
        )
    return float(np.mean((prob - event) ** 2))


def calibration_table(
    probability: np.ndarray | Sequence[float],
    events: np.ndarray | Sequence[bool],
    *,
    bins: int = 10,
) -> list[dict[str, Any]]:
    """Observed event rate per predicted-probability bucket.

    A well-calibrated model has ``observed`` near ``predicted`` in every bucket.
    Systematic over-prediction (all observed below predicted) is the usual
    failure for a model trained on a rare, non-stationary event.
    """
    prob = np.asarray(probability, dtype=float).reshape(-1)
    event = np.asarray(events, dtype=float).reshape(-1)
    size = min(prob.size, event.size)
    if size == 0:
        return []
    prob, event = prob[:size], event[:size]
    keep = np.isfinite(prob) & np.isfinite(event)
    prob, event = prob[keep], (event[keep] > 0.5).astype(float)
    if prob.size == 0 or prob.min() < 0.0 or prob.max() > 1.0:
        return []

    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        # First bin is closed on the left, last on the right, so 0.0 and 1.0
        # both land in a bucket instead of falling outside the grid.
        mask = (prob >= lo) & (prob < hi) if i < bins - 1 else (prob >= lo) & (prob <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        observed = float(event[mask].mean())
        rows.append({
            "lower": float(lo),
            "upper": float(hi),
            "n": count,
            "predicted": float(prob[mask].mean()),
            "observed": observed,
        })
    return rows
