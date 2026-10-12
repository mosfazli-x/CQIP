"""Target construction: volatility normalisation and triple-barrier labelling.

The default CQIP regression target is the ``horizon``-bar forward log return. For
a *directional* model that target has two well-known defects:

* It is not scale-stable. A 12-bar forward return on a calm asset is an order of
  magnitude smaller than the same return on a violent one, so a single unweighted
  Gaussian NLL spends most of its capacity on whichever assets happen to be
  volatile, and the learned scale is not comparable across symbols.
* It is path-blind. A steady +3% and a +3% that first round-tripped through -3%
  are the same label, although only the first is a position anyone would hold.

``add_volatility_normalized_target`` and ``make_triple_barrier_labels`` address
those two defects. Both are strictly causal: every quantity used to build a label
at bar ``i`` is observable at or before the close of bar ``i``. Nothing here may
read a price, statistic or article dated after the label's own bar, because these
columns become supervised training targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

EPS = 1e-12

TargetKind = Literal["log_return", "vol_normalized", "triple_barrier", "move_event"]

#: Column produced by :func:`add_volatility_normalized_target`.
VOL_NORMALIZED_COLUMN = "target_scaled"
#: Binary "decisive move happened" label used by the ``move_event`` target.
EVENT_COLUMN = "target_move_event"
#: Columns produced by :func:`make_triple_barrier_labels`.
TRIPLE_BARRIER_COLUMNS = ("tb_label", "tb_return", "tb_exit_bars", "tb_barrier")
#: Column holding the binary up/down label that supervises the auxiliary head.
DIRECTION_COLUMN = "target_direction_binary"


def trailing_volatility(
    close: pd.Series, *, window: int = 64, min_periods: int | None = None
) -> pd.Series:
    """Per-bar realised volatility of log returns, using trailing data only.

    Deliberately **not** annualised and deliberately not shifted: the value at
    index ``i`` is the standard deviation of the ``window`` returns ending at
    ``i``, which is exactly what a live system could compute at that bar's close.
    """
    log_returns = np.log(close.astype(float).replace(0, np.nan)).diff()
    return log_returns.rolling(
        window, min_periods=min_periods if min_periods is not None else max(4, window // 4)
    ).std(ddof=1)


def add_volatility_normalized_target(
    frame: pd.DataFrame,
    *,
    source: str = "target_return",
    window: int = 64,
    horizon: int = 1,
    out: str = VOL_NORMALIZED_COLUMN,
    vol_floor: float = 1e-5,
) -> pd.DataFrame:
    """Add a target expressed in units of the asset's own recent volatility.

    The scaled target is ``forward_return / (sigma * sqrt(horizon))`` where
    ``sigma`` is the trailing per-bar volatility at the label's bar. Dividing by
    ``sqrt(horizon)`` matters as much as dividing by ``sigma``: a raw 12-bar
    return is expected to be about ``sqrt(12)`` times a 1-bar return, so without
    it the "normalised" target still scales with the horizon and the model learns
    the horizon instead of the difficulty of the trade.

    ``sigma`` is floored at ``vol_floor`` so a dead-flat series cannot produce an
    unbounded target that would dominate the loss.
    """
    if source not in frame.columns:
        raise KeyError(f"source target {source!r} is not present in the frame")

    out_frame = frame.copy()
    sigma = trailing_volatility(out_frame["close"], window=window).clip(lower=vol_floor)
    scale = sigma * float(np.sqrt(max(1, horizon)))
    out_frame[out] = (out_frame[source].astype(float) / scale.replace(0, np.nan)).astype(np.float64)
    out_frame["target_vol_scale"] = scale
    return out_frame


@dataclass(frozen=True)
class TripleBarrierConfig:
    """Volatility-scaled take-profit / stop-loss / time barriers.

    ``take_profit_vol`` and ``stop_loss_vol`` are multiples of one bar's expected
    move, not percentage points, so the barriers sit at the same statistical
    distance for a calm asset and a violent one. ``max_hold`` defaults to the
    label's own horizon so triple-barrier labelling is a drop-in refinement of
    the fixed-horizon target rather than a different forecasting problem.
    """

    take_profit_vol: float = 1.0
    stop_loss_vol: float = 1.0
    max_hold: int | None = None
    vol_window: int = 32
    vol_floor: float = 1e-6
    price_col: str = "close"
    high_col: str = "high"
    low_col: str = "low"

    def validated(self) -> TripleBarrierConfig:
        if self.take_profit_vol <= 0:
            raise ValueError("take_profit_vol must be positive")
        if self.stop_loss_vol <= 0:
            raise ValueError("stop_loss_vol must be positive")
        if self.max_hold is not None and self.max_hold < 1:
            raise ValueError("max_hold must be at least 1 bar")
        if self.vol_window < 2:
            raise ValueError("vol_window must be at least 2")
        return self


def make_triple_barrier_labels(
    frame: pd.DataFrame,
    config: TripleBarrierConfig | None = None,
    *,
    horizon: int = 1,
    max_hold: int | None = None,
) -> pd.DataFrame:
    """Label each bar with the first barrier touched, and the return realised there.

    Adds ``tb_label`` (``+1`` take-profit, ``-1`` stop-loss, ``0`` time exit),
    ``tb_return`` (log return at the exit), ``tb_exit_bars`` (bars held) and
    ``tb_barrier`` (which barrier fired). Rows whose future is unavailable, or
    whose trailing volatility is not yet defined, are ``NaN`` and must be dropped
    by the caller before training.

    When a single bar's high *and* low span both barriers the **stop-loss is
    assumed to have fired first**. The intrabar path is unknowable from OHLCV, and
    assuming the favourable ordering would bias every label upward.
    """
    cfg = (config or TripleBarrierConfig()).validated()
    hold = int(max_hold or cfg.max_hold or horizon)
    if hold < 1:
        raise ValueError("max_hold must be at least 1 bar")

    required = {cfg.price_col, cfg.high_col, cfg.low_col}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"triple-barrier labelling needs columns {sorted(missing)}")

    out = frame.copy()
    close = out[cfg.price_col].to_numpy(dtype=float)
    high = out[cfg.high_col].to_numpy(dtype=float)
    low = out[cfg.low_col].to_numpy(dtype=float)
    sigma = trailing_volatility(
        out[cfg.price_col], window=cfg.vol_window
    ).to_numpy(dtype=float)

    # Expected move scales with the square root of holding time.
    reach = sigma * np.sqrt(hold)
    take_profit_k = np.exp(cfg.take_profit_vol * reach)
    stop_loss_k = np.exp(-cfg.stop_loss_vol * reach)

    n = len(out)
    labels = np.full(n, np.nan)
    returns = np.full(n, np.nan)
    exit_bars = np.full(n, np.nan)
    barrier = np.full(n, "", dtype=object)

    for start in range(n - hold):
        entry = close[start]
        sigma_i = sigma[start]
        if not np.isfinite(entry) or entry <= 0 or not np.isfinite(sigma_i):
            continue

        take_price = entry * take_profit_k[start]
        stop_price = entry * stop_loss_k[start]
        if not (np.isfinite(take_price) and np.isfinite(stop_price)):
            continue

        label = 0.0
        exit_index = start + hold
        for offset in range(1, hold + 1):
            bar_low = low[start + offset]
            bar_high = high[start + offset]
            hit_stop = np.isfinite(bar_low) and bar_low <= stop_price
            hit_take = np.isfinite(bar_high) and bar_high >= take_price
            # Pessimistic intra-bar ordering: an ambiguous bar counts as a stop.
            if hit_stop:
                label, exit_index = -1.0, start + offset
                break
            if hit_take:
                label, exit_index = 1.0, start + offset
                break

        exit_close = close[exit_index]
        if not np.isfinite(exit_close) or exit_close <= 0:
            continue

        labels[start] = label
        returns[start] = float(np.log(exit_close / entry))
        exit_bars[start] = float(exit_index - start)
        barrier[start] = "take_profit" if label > 0 else "stop_loss" if label < 0 else "time"

    out["tb_label"] = labels
    out["tb_return"] = returns
    out["tb_exit_bars"] = exit_bars
    out["tb_barrier"] = barrier
    return out


def directional_labels(target: pd.Series) -> pd.Series:
    """Binary up/down target for the auxiliary direction head.

    Exact zeros are labelled ``NaN`` rather than folded into one class, so a
    dead-flat stretch of bars is dropped instead of being scored as a coin flip
    the model could learn to love.
    """
    values = target.astype(float)
    return pd.Series(
        np.where(values > 0, 1.0, np.where(values < 0, 0.0, np.nan)), index=target.index
    )


@dataclass(frozen=True)
class TargetSpec:
    """Which columns a trainer should regress and classify against."""

    kind: TargetKind
    regression: str
    direction: str | None
    horizon: int
    detail: dict[str, Any] = field(default_factory=dict)
    #: True when `regression` is expressed in units of trailing volatility
    #: rather than as a raw log return. Consumers must undo this scaling before
    #: treating the label as a tradeable return.
    normalize_volatility: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "regression": self.regression,
            "direction": self.direction,
            "horizon": self.horizon,
            "normalize_volatility": self.normalize_volatility,
            **self.detail,
        }


def make_move_event_label(
    out: pd.DataFrame,
    *,
    source: str = "target_return",
    threshold: float = 0.02,
    k: float | None = None,
    vol_window: int = 64,
    vol_column: str = "target_volatility",
) -> pd.Series:
    """Binary label for "was there a decisive move", censoring the middle band.

    A ``+2% a day`` objective is an *event* claim, not a direction claim: the
    question is whether a large move happened, not which way a small one went.
    Two parameterisations are supported:

    * ``threshold`` (absolute): ``forward return >= +2%`` is a long event,
      ``<= -2%`` a short event. This is the literal objective, but it is not
      comparable across assets -- +2% is unremarkable for a quiet altcoin and
      exceptional for BTC, so the same threshold selects very different
      frequencies on different symbols.
    * ``k`` (standardised): the move is divided by the symbol's own trailing
      volatility, so ``k=2`` means a 2-sigma move everywhere. This is the form
      that transfers across a 216-symbol universe.

    Rows inside the band are ``NaN``: "rose 0.4%" is not evidence against
    "will rise 2%", and training a classifier to call those a failure teaches it
    to predict the noise between thresholds. Censoring is why event models get
    thin samples, which the evaluator must then report rather than hide.
    """
    values = pd.to_numeric(out[source], errors="coerce")
    if k is not None:
        vol = pd.to_numeric(
            out[vol_column] if vol_column in out.columns else pd.Series(np.nan, index=out.index),
            errors="coerce",
        )
        if vol.isna().all():
            vol = trailing_volatility(out["close"], window=vol_window)
        scale = vol.replace(0.0, np.nan)
        values = values / scale
        bound = float(k)
    else:
        bound = float(threshold)

    return pd.Series(
        np.where(values >= bound, 1.0, np.where(values <= -bound, 0.0, np.nan)),
        index=out.index,
    )


def build_target_frame(
    frame: pd.DataFrame,
    *,
    horizon: int = 1,
    kind: TargetKind = "log_return",
    normalize_volatility: bool = False,
    triple_barrier: TripleBarrierConfig | None = None,
    vol_window: int = 64,
    move_event_threshold: float | None = None,
    move_event_k: float | None = None,
) -> tuple[pd.DataFrame, TargetSpec]:
    """Materialise the target columns a run needs and describe them.

    Adds every derived target the requested configuration will use. All of them
    are built here rather than lazily so that a run's target definition is fully
    determined by the frame that comes out, and so the split/purge geometry can
    be computed from real column presence rather than assumptions.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1 bar")

    out = frame.copy()
    if kind == "move_event":
        # Regression stays the raw forward return: the model still learns *where*
        # price is heading, while the direction column becomes the rare,
        # thresholded event. Using the raw return as the regression target keeps
        # every bar usable, instead of discarding the entire middle band.
        out[EVENT_COLUMN] = make_move_event_label(
            out,
            source="target_return",
            threshold=move_event_threshold if move_event_threshold is not None else 0.02,
            k=move_event_k,
            vol_window=vol_window,
        )
        spec = TargetSpec(
            kind="move_event",
            regression="target_return",
            direction=EVENT_COLUMN,
            horizon=horizon,
            detail={
                "threshold": (
                    None if move_event_k is not None
                    else float(move_event_threshold if move_event_threshold is not None else 0.02)
                ),
                "k": None if move_event_k is None else float(move_event_k),
            },
        )
    elif kind == "triple_barrier" or triple_barrier is not None:
        out = make_triple_barrier_labels(
            out, triple_barrier, horizon=horizon, max_hold=(triple_barrier.max_hold if triple_barrier else None)
        )
        spec = TargetSpec(
            kind="triple_barrier",
            regression="tb_return",
            direction=None,
            horizon=horizon,
            detail={
                "max_hold": int(triple_barrier.max_hold or horizon) if triple_barrier else horizon,
                "take_profit_vol": (triple_barrier.take_profit_vol if triple_barrier else 1.0),
                "stop_loss_vol": (triple_barrier.stop_loss_vol if triple_barrier else 1.0),
            },
        )
    else:
        # `regression` stays pointed at the *raw* target even when scaling is
        # requested; the normalisation block below is what renames it. Setting
        # the scaled name here meant the scaler was later asked to read
        # `target_scaled` as its own source, before that column existed.
        spec = TargetSpec(
            kind="vol_normalized" if normalize_volatility else "log_return",
            regression="target_return",
            direction=None,
            horizon=horizon,
        )

    # Guard on the column being absent, not on the *name* already pointing at
    # it. The non-triple-barrier branch above sets `regression` to
    # VOL_NORMALIZED_COLUMN as soon as `normalize_volatility` is true, so a
    # name-based guard was always False here and the scaled column was never
    # actually created -- `build_target_frame(normalize_volatility=True)` raised
    # "target column 'target_scaled' missing" on every call.
    if normalize_volatility and VOL_NORMALIZED_COLUMN not in out.columns:
        out = add_volatility_normalized_target(
            out, source=spec.regression, window=vol_window, horizon=horizon
        )
        spec = TargetSpec(
            kind="vol_normalized",  # type: ignore[arg-type]
            regression=VOL_NORMALIZED_COLUMN,
            direction=spec.direction,
            horizon=horizon,
            detail=spec.detail,
            normalize_volatility=True,
        )

    if spec.regression not in out.columns:
        raise ValueError(
            f"target column {spec.regression!r} missing after target construction; "
            f"available={sorted(set(out.columns) & set(('target_return', 'tb_return', VOL_NORMALIZED_COLUMN)))}"
        )

    # Always materialise the binary direction label, whatever the regression
    # target is, so the auxiliary head can be switched on for any target kind.
    # It is derived from the *regression* column rather than reusing `tb_label`:
    # `tb_label` is in {-1, 0, +1}, and feeding that to binary cross-entropy
    # turns every stop-loss into a label of -1, which the loss accepts silently
    # and then optimises towards as if it were a probability. Time exits become
    # NaN and are masked out, because a bar that hit neither barrier carries no
    # directional information.
    #
    # A `move_event` target already carries its own thresholded direction label
    # in EVENT_COLUMN. Overwriting it with the plain sign of the return would
    # silently reduce the event problem back to the direction problem it was
    # built to escape, so the sign label is written only when it is not already
    # present, and the spec keeps pointing at the event column.
    if spec.direction is None:
        out[DIRECTION_COLUMN] = directional_labels(out[spec.regression])
        spec = TargetSpec(
            kind=spec.kind,
            regression=spec.regression,
            direction=DIRECTION_COLUMN,
            horizon=spec.horizon,
            detail=spec.detail,
            normalize_volatility=spec.normalize_volatility,
        )
    return out, spec


__all__ = [
    "DIRECTION_COLUMN",
    "TargetSpec",
    "TripleBarrierConfig",
    "VOL_NORMALIZED_COLUMN",
    "TRIPLE_BARRIER_COLUMNS",
    "add_volatility_normalized_target",
    "build_target_frame",
    "directional_labels",
    "make_triple_barrier_labels",
    "trailing_volatility",
]
