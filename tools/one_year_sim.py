"""One-year, $1,000, fee-aware walk-forward simulation of a selective crypto robot.

Design, and why each piece is the way it is:

**What "highest probability of profit" means here.** Every bar, each symbol gets a
score ``predicted_return / predicted_volatility`` -- the model's mean divided by
its own predicted standard deviation. That ratio is the quantity a probabilistic
forecaster is actually claiming something about, and it is the only thing in this
codebase that ranks bars by confidence rather than by direction alone. Raw
``predicted_return`` cannot be compared across assets: a 1% expected move on BTC
and a 1% expected move on a 12%-volatility alt are not the same bet. The ratio
is scale-free, so it ranks across the universe.

**The score is calibrated before it is traded, never after.** The forecaster
heads are Gaussian and the repository's own stored metrics show their bands are
badly miscalibrated (68% bands containing 85-97% of outcomes), so the implied
``Phi(mean/sd)`` would systematically overstate confidence. ``vol_scale`` is
fitted on the *validation* slice only -- data the network did not train on -- so
the standardized residual has unit standard deviation, and the probability is
recomputed from that corrected score. Test bars never touch the fit.

**Ranking power is measured before it is monetised.** An AUC and a bootstrap
interval are reported for the score against realised direction. A bot that
"targets the highest probability of profit" can only work if the score actually
separates winners from losers; if the AUC interval straddles 0.5, the selection
rule is choosing noise and the P&L number below is a cost bill, not a result.
This is printed whether it passes or fails.

**Walk-forward, with the training window strictly in the past.** Each refit
trains on bars ``[0, test_start - gap)`` and predicts bars
``[test_start, test_end)``, where ``gap = purge + embargo`` is sized from the
label horizon and the sequence length so no training label or overlapping window
straddles the boundary. Predictions are made on a frame that begins ``seq_len``
bars *before* the test window and the head is dropped, so the first test bar has
real history behind it without any test bar being consumed by a training label.

**Selection is cross-sectional.** At each bar the K symbols with the highest
calibrated probability go long and the K lowest go short. Ranking within a single
series would let a model that only knows "this asset drifts up" sit long forever;
ranking across the universe forces a decision at every bar.

**Money is split into fixed slots, not shared.** ``$1000 / (2 * K)`` per slot,
each slot run through the audited :class:`BacktestEngine`. Fixed slots mean a
losing position cannot shrink a winning one's budget, no slot can borrow from
another, and total exposure never exceeds equity. Idle slot capital earns 0%,
which is conservative for a venue that pays some yield on balances.

**Real-world frictions.** 10bps taker fee plus 5bps slippage on every fill, both
configurable. Fills happen at ``open[i+1]`` for a signal decided at the close of
bar ``i``. No leverage, no borrow, no margin.

Usage:
    python tools/one_year_sim.py
    python tools/one_year_sim.py --top-k 2 --epochs 25
    python tools/one_year_sim.py --diagnostics-only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cqip.backtest import BacktestEngine, compute_metrics, periods_per_year  # noqa: E402
from cqip.config import Settings  # noqa: E402
from cqip.features import build_feature_frame, make_targets, resolve_purge_embargo  # noqa: E402
from cqip.forecaster import ForecasterTrainer  # noqa: E402
from cqip.logging_config import configure_logging  # noqa: E402
from cqip.storage import ParquetStore  # noqa: E402

# Pegged assets carry no directional signal and would dilute a cross-sectional
# ranking with rows whose "move" is exchange noise around a fixed peg.
STABLECOIN_BASES = {
    "USDT", "USDC", "BUSD", "DAI", "FDUSD", "TUSD", "USDP", "PYUSD", "USD1",
    "USDD", "EURC", "EURS", "FRAX", "LUSD", "GUSD", "USTB", "SUSDE", "USDE",
    "XAUT",  # tokenised gold: not a crypto, tracks bullion
}

SIM_START = pd.Timestamp("2025-10-01T00:00:00Z")


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #

def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (ties averaged). No sklearn dependency at call sites."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    ok = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[ok], labels[ok]
    pos, neg = labels > 0, labels <= 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = pd.Series(scores).rank(method="average").to_numpy()
    # Mann-Whitney U: P(score_pos > score_neg) with ties counted as half.
    return float((order[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _bootstrap_auc(
    scores: np.ndarray, labels: np.ndarray, *, n_boot: int, seed: int
) -> tuple[float, float, float]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    ok = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[ok], labels[ok]
    point = _auc(scores, labels)
    if scores.size < 50:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    n = scores.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        value = _auc(scores[idx], labels[idx])
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _bootstrap_mean(
    values: np.ndarray, *, n_boot: int, seed: int
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 10:
        return (float(values.mean()) if values.size else float("nan")), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    return float(values.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# --------------------------------------------------------------------------- #
# universe
# --------------------------------------------------------------------------- #

@dataclass
class Universe:
    hourly: pd.DataFrame
    symbols: list[str]
    detail: pd.DataFrame


def select_universe(
    store: ParquetStore, *, top_k_assets: int, min_hourly_notional: float, min_days: int
) -> Universe:
    """Liquidest non-stablecoin pairs with real coverage on both timeframes.

    Liquidity is the *median* hourly notional, not the mean: a single 10-sigma
    print should not earn a thin asset a place in a $1,000 book, and a $1,000
    order against $50k/hour is 2% of a single bar's volume, which is why 5bps of
    slippage is a defensible assumption at this tier.
    """
    hourly = store.read(
        "ohlcv",
        filters=[("interval", "==", "1h")],
        columns=["ts", "open", "high", "low", "close", "volume", "symbol"],
    )
    if hourly.empty:
        raise SystemExit("no hourly candles stored; run `cqip ingest candles` first")

    hourly = hourly[hourly["ts"] >= SIM_START].copy()
    rows = []
    for symbol, group in hourly.groupby("symbol"):
        base = str(symbol).split("_")[0]
        if base in STABLECOIN_BASES:
            continue
        group = group.sort_values("ts")
        ts = pd.to_datetime(group["ts"])
        span = (ts.max() - ts.min()).total_seconds() / 86_400.0
        if span < min_days:
            continue
        notional = float((group["close"] * group["volume"]).median())
        if notional < min_hourly_notional:
            continue
        rows.append({
            "symbol": str(symbol),
            "bars_1h": len(group),
            "span_days": span,
            "med_hourly_notional": notional,
            "med_close": float(group["close"].median()),
        })

    detail = pd.DataFrame(rows).sort_values("med_hourly_notional", ascending=False)
    if detail.empty:
        raise SystemExit("no symbol passed the liquidity and coverage filters")
    symbols = detail.head(top_k_assets)["symbol"].tolist()

    daily = store.read(
        "ohlcv", filters=[("interval", "==", "1d")],
        columns=["ts", "close", "symbol"],
    )
    if not daily.empty:
        daily = daily[daily["ts"] >= SIM_START]
        have = set(daily["symbol"].astype(str).unique())
        symbols = [s for s in symbols if s in have]

    detail = detail[detail["symbol"].isin(symbols)].reset_index(drop=True)
    return Universe(hourly=hourly, symbols=symbols, detail=detail)


# --------------------------------------------------------------------------- #
# per-symbol preparation
# --------------------------------------------------------------------------- #

@dataclass
class Prepared:
    symbol: str
    interval: str
    frame: pd.DataFrame
    feature_columns: list[str]
    test_from: int          # first index traded; everything before is warm-up
    fold_bounds: list[tuple[int, int]]


def prepare_symbol(
    store: ParquetStore,
    symbol: str,
    interval: str,
    *,
    warmup_bars: int,
    trade_bars: int,
    refit_bars: int,
    horizon: int,
    min_test_bars: int,
) -> Prepared:
    """Build the causal feature frame and lay out the walk-forward folds.

    ``trade_bars`` is how many bars at the *end* of the history the bot actually
    trades, and it is the parameter that differs between intervals. Hourly
    history only reaches back one year, so there is nothing to warm up on except
    a slice of that same year: it trades everything after ``warmup_bars``. Daily
    history runs back to 2013, so the bot warms up on all of it and trades only
    the final ``trade_bars`` -- which is what makes the daily run the one with a
    real out-of-sample history behind it.

    Anchoring the folds to the *end* rather than to an absolute offset is what
    keeps the daily run to a handful of refits. Laying them out from a fixed
    index instead would schedule a fold for every refit window between that
    index and the present, which on four years of daily bars is over a hundred
    fits for bars nobody asked about.
    """
    candles = store.read(
        "ohlcv", filters=[("interval", "==", interval)], symbols=[symbol],
        columns=["ts", "open", "high", "low", "close", "volume", "symbol"],
    )
    candles = candles.sort_values("ts").reset_index(drop=True)
    if interval == "1h":
        candles = candles[candles["ts"] >= SIM_START].reset_index(drop=True)
    if len(candles) < warmup_bars + min_test_bars + refit_bars:
        raise ValueError(f"{symbol} {interval}: only {len(candles)} bars")

    frame, columns = build_feature_frame(candles)
    frame = make_targets(frame, horizon=horizon)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["target_return"]).reset_index(drop=True)
    frame["close"] = frame["close"].astype(float)
    for column in ("open", "high", "low"):
        frame[column] = frame[column].astype(float)

    total = len(frame)
    # The traded window is the last `trade_bars` bars, and it must leave enough
    # room ahead of it for a warm-up or the first fold has nothing to train on.
    span = min(int(trade_bars), total - min_test_bars)
    test_from = max(warmup_bars, total - span)
    test_from = min(test_from, total - min_test_bars)
    if test_from < warmup_bars:
        raise ValueError(
            f"{symbol} {interval}: {total} bars is not enough for a "
            f"{warmup_bars}-bar warm-up plus {min_test_bars} traded bars"
        )

    # Expanding-window refits anchored to the traded window: each fold trains on
    # everything strictly before it and predicts only its own slice.
    bounds: list[tuple[int, int]] = []
    start = test_from
    while start < total:
        end = min(start + refit_bars, total)
        bounds.append((start, end))
        start = end

    return Prepared(symbol=symbol, interval=interval, frame=frame,
                    feature_columns=columns, test_from=test_from, fold_bounds=bounds)


# --------------------------------------------------------------------------- #
# walk-forward forecasting
# --------------------------------------------------------------------------- #

@dataclass
class WalkForwardResult:
    oos: pd.DataFrame                       # ts, score, target_return, realized_dir
    diagnostics: dict[str, Any] = field(default_factory=dict)


def walk_forward(
    prepared: Prepared,
    trainer: ForecasterTrainer,
    *,
    seq_len: int,
    horizon: int,
    epochs: int,
    kind: str,
    val_ratio: float,
    seed: int,
    n_boot: int,
) -> WalkForwardResult:
    """Refit per fold on strictly-past data; predict only the unseen slice."""
    frame = prepared.frame
    columns = prepared.feature_columns
    purge, embargo = resolve_purge_embargo(horizon=horizon, seq_len=seq_len, embargo_bars=0)
    gap = purge + embargo

    pieces: list[pd.DataFrame] = []
    fold_log: list[dict[str, Any]] = []

    for fold_index, (test_start, test_end) in enumerate(prepared.fold_bounds):
        train_end = test_start - gap
        if train_end - seq_len < 400:
            continue

        val_len = max(120, int(train_end * val_ratio))
        val_start = train_end - val_len
        if val_start - seq_len < 300:
            val_start = seq_len + 300
            val_len = train_end - val_start
        if val_len < 60 or val_start - seq_len < 300:
            continue

        train = frame.iloc[:val_start].reset_index(drop=True)
        val = frame.iloc[val_start:train_end].reset_index(drop=True)
        if len(train) <= seq_len or len(val) <= seq_len:
            continue

        try:
            model, info = trainer.train(
                train, val, columns, target_column="target_return",
                seq_len=seq_len, horizon=horizon, kind=kind, epochs=epochs, seed=seed,
            )
        except Exception as exc:  # noqa: BLE001
            fold_log.append({"fold": fold_index, "error": str(exc)[:180]})
            continue

        # Calibration is fitted on the validation slice: the network did not
        # train on it, and the test slice is still untouched at this point.
        val_preds = trainer.predict(model, val, columns, info, seq_len=seq_len)
        if val_preds.empty:
            continue
        val_aligned = val.iloc[-len(val_preds):].reset_index(drop=True)
        val_vol = val_aligned["target_volatility"].to_numpy(dtype=float)
        val_score = val_preds["predicted_return"].to_numpy(dtype=float) / np.where(
            val_vol > 1e-9, val_vol, np.nan
        )
        val_target = val_aligned["target_return"].to_numpy(dtype=float)
        finite = np.isfinite(val_score) & np.isfinite(val_target) & np.isfinite(val_vol)
        if finite.sum() < 50:
            continue
        # `target_volatility` is the *realized* trailing scale. Dividing by it
        # gives the model score; comparing the residual to it gives the
        # correction that makes Phi(score) an honest probability.
        vol_scale = float(
            np.std(val_target[finite] - val_score[finite] * val_vol[finite], ddof=1)
            / max(np.mean(val_vol[finite]), 1e-12)
        )
        if not np.isfinite(vol_scale) or vol_scale <= 0:
            vol_scale = 1.0
        vol_scale = float(np.clip(vol_scale, 0.2, 10.0))

        # Predict the test slice with `seq_len` bars of history carried in front
        # of it, so bar 0 of the test window has real history and no test bar is
        # consumed by a training label.
        predict_frame = frame.iloc[max(0, test_start - seq_len):test_end].reset_index(drop=True)
        preds = trainer.predict(model, predict_frame, columns, info, seq_len=seq_len)
        if preds.empty:
            continue
        aligned = predict_frame.iloc[-len(preds):].reset_index(drop=True)
        raw_score = aligned["target_volatility"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            score = (
                preds["predicted_return"].to_numpy(dtype=float)
                / (np.where(raw_score > 1e-9, raw_score, np.nan) * vol_scale)
            )
        block = pd.DataFrame({
            "ts": preds["ts"].to_numpy(),
            "score": score,
            "target_return": aligned["target_return"].to_numpy(dtype=float),
            # Kept so a later run can gate on the model's own predicted *magnitude*
            # without refitting: `score` is scale-free, so it ranks well but says
            # nothing about whether a move is big enough to pay for the spread.
            "predicted_return": preds["predicted_return"].to_numpy(dtype=float),
            "target_volatility": raw_score,
            "vol_scale": vol_scale,
        })
        block = block[np.isfinite(block["score"]) & np.isfinite(block["target_return"])]
        if not block.empty:
            pieces.append(block)
            fold_log.append({
                "fold": fold_index, "train_bars": len(train), "test_bars": len(block),
                "vol_scale": round(vol_scale, 3),
            })

    if not pieces:
        return WalkForwardResult(oos=pd.DataFrame(), diagnostics={"folds": fold_log})

    oos = pd.concat(pieces, ignore_index=True).drop_duplicates("ts").sort_values("ts")
    oos = oos.reset_index(drop=True)

    labels = (oos["target_return"] > 0).astype(float)
    auc_point, auc_low, auc_high = _bootstrap_auc(
        oos["score"].to_numpy(), labels.to_numpy(), n_boot=n_boot, seed=seed
    )
    realised = oos["target_return"].to_numpy(dtype=float)
    direction = np.sign(realised) == np.sign(oos["score"].to_numpy())
    accuracy = float(np.nanmean(direction))
    corr = float(np.corrcoef(realised, oos["score"].to_numpy())[0, 1]) if len(oos) > 3 else float("nan")

    # Calibration: claimed probability versus observed win rate, by score decile.
    claimed = _normal_cdf(oos["score"].to_numpy())
    ranks = pd.Series(oos["score"]).rank(method="first", pct=True)
    decile = np.clip((ranks * 10).astype(int), 0, 9)
    calibration = []
    for bucket in range(10):
        mask = decile == bucket
        if not mask.any():
            continue
        calibration.append({
            "decile": bucket + 1,
            "n": int(mask.sum()),
            "mean_score": float(np.mean(oos["score"].to_numpy()[mask])),
            "claimed_p_profit": float(np.mean(claimed[mask])),
            "observed_p_profit": float(np.mean(labels.to_numpy()[mask])),
        })

    return WalkForwardResult(oos=oos, diagnostics={
        "folds": fold_log,
        "oos_bars": int(len(oos)),
        "oos_start": str(oos["ts"].iloc[0]),
        "oos_end": str(oos["ts"].iloc[-1]),
        "directional_accuracy": accuracy,
        "correlation": corr,
        "auc": auc_point,
        "auc_ci_low": auc_low,
        "auc_ci_high": auc_high,
        "auc_passes": bool(np.isfinite(auc_low) and auc_low > 0.5),
        "mean_vol_scale": float(np.mean([f.get("vol_scale", 1.0) for f in fold_log]))
        if fold_log else float("nan"),
        "calibration": calibration,
    })


# --------------------------------------------------------------------------- #
# cross-sectional selection
# --------------------------------------------------------------------------- #

def build_position_matrix(
    oos_by_symbol: dict[str, pd.DataFrame], *, top_k: int, prob_bar: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long the K best probabilities, short the K worst, one column per symbol.

    Returns the signed weight matrix (rows = timestamps) and the per-bar
    calibrated probability that drove it. A symbol absent from a bar gets weight
    0 rather than being forward-filled: it had no quote, so it could not be
    traded, and carrying a stale position would invent a fill that never happened.
    """
    stamps = sorted(set().union(*(set(df["ts"]) for df in oos_by_symbol.values())))
    index = pd.Index(stamps, name="ts")
    weights = pd.DataFrame(0.0, index=index, columns=sorted(oos_by_symbol))
    probs = pd.DataFrame(np.nan, index=index, columns=sorted(oos_by_symbol))

    probability: dict[pd.Timestamp, pd.Series] = {}
    for symbol, frame in oos_by_symbol.items():
        column = pd.Series(
            _normal_cdf(frame["score"].to_numpy(dtype=float)),
            index=pd.DatetimeIndex(frame["ts"]),
        )
        for stamp, value in column.items():
            probability.setdefault(stamp, {})[symbol] = value  # type: ignore[union-attr]
        probs.loc[column.index, symbol] = column.to_numpy()

    n_assets = len(weights.columns)
    k = max(1, min(top_k, n_assets // 2))
    slot = 1.0 / (2 * k)

    for stamp, row in probability.items():
        present = [s for s in weights.columns if s in row and np.isfinite(row[s])]
        if len(present) < 2:
            continue
        ranked = sorted(present, key=lambda s: row[s], reverse=True)
        longs, shorts = ranked[:k], ranked[-k:]
        # A bar where the model has no conviction is a bar to stay flat; paying
        # 15bps to trade a coin flip is how a selective system stops being selective.
        if row[longs[0]] < prob_bar or (1.0 - row[shorts[0]]) < prob_bar:
            continue
        for symbol in longs:
            weights.at[stamp, symbol] = slot
        for symbol in shorts:
            weights.at[stamp, symbol] = -slot

    return weights, probs


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #

def _hours_per_bar(interval: str) -> float:
    """Wall-clock hours one bar of ``interval`` spans, for holding-period reporting."""
    if interval.endswith("m"):
        return float(interval[:-1]) / 60.0
    if interval.endswith("h"):
        return float(interval[:-1])
    if interval.endswith("d"):
        return 24.0 * float(interval[:-1])
    return 1.0


def align_gate_to_decision(grid: pd.DataFrame, stamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Shift a forecast grid so row ``t`` carries the forecast acted on at open(t+1).

    Weight and forecast rows are stamped with the bar whose *close* produced them,
    and the engine trades ``held.shift(1)``. The gate has to read that same shifted
    row. Reading the unshifted row would let a forecast computed after a bar's close
    decide a fill that happened before that close existed -- a one-bar look-ahead
    that flatters exactly the selective strategies this gate exists to measure.
    """
    return grid.reindex(index=stamps).shift(1)


@dataclass
class ExecutionPolicy:
    """Every knob that decides *whether* to trade, kept out of the accounting.

    The defaults reproduce the original behaviour exactly -- rebalance to the
    published weight on every bar, gate nothing -- so the mechanics stay
    comparable while the policy is swept.
    """

    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    rebalance_band_bps: float = 0.0
    min_expected_move_bps: float = 0.0
    min_hold_bars: int = 0
    max_fills_per_day: int | None = None
    gate_mode: str = "flat"  # "flat" = exit when conviction dies, "hold" = keep it

    @property
    def round_trip_bps(self) -> float:
        return self.fee_bps + self.slippage_bps

    def describe(self) -> dict[str, Any]:
        return {
            "fee_bps": self.fee_bps,
            "slippage_bps": self.slippage_bps,
            "round_trip_bps": self.round_trip_bps,
            "rebalance_band_bps": self.rebalance_band_bps,
            "min_expected_move_bps": self.min_expected_move_bps,
            "min_hold_bars": self.min_hold_bars,
            "max_fills_per_day": self.max_fills_per_day,
            "gate_mode": self.gate_mode,
        }


def run_portfolio(
    store: ParquetStore,
    symbols: Sequence[str],
    interval: str,
    weights: pd.DataFrame,
    *,
    initial_capital: float,
    fee_bps: float,
    slippage_bps: float,
    rebalance_band_bps: float = 0.0,
    score_grid: pd.DataFrame | None = None,
    expected_move: pd.DataFrame | None = None,
    min_expected_move_bps: float = 0.0,
    min_hold_bars: int = 0,
    max_fills_per_day: int | None = None,
    gate_mode: str = "flat",
) -> dict[str, Any]:
    """Execute the weight matrix as ONE account, matching the engine's convention.

    The obvious implementation gives every symbol its own BacktestEngine run. That
    funds one sub-account per symbol while only ``max_slots`` of them are ever open
    at the same time, so the idle ones dilute the aggregate return, inflate the fee
    total, and leave the result sitting on a capital base it was never funded with.
    This chains a single account instead: exposure is a signed fraction of the one
    pot, so a slot going idle hands its budget back to the same account rather than
    freezing inside a private one.

    Timing matches ``BacktestEngine.run_signal_strategy``: a signal computed on bar
    ``t`` is filled at ``open(t+1)`` and marked at ``close(t+1)``, so bar ``t+1``
    earns ``close/open - 1`` on the weight decided at bar ``t``.

    On top of that sit three *restraint* controls, all off by default. Each one
    exists because a rank-only signal with no magnitude is most often noise, and
    paying a spread to express noise is how a selective system stops being
    selective:

    ``expected_move`` / ``min_expected_move_bps``
        Take a position only when the model's own predicted move clears a
        multiple of round-trip cost. Below the bar the strategy prefers to be
        flat; the gate never manufactures a position it would not otherwise hold.
    ``min_hold_bars``
        Once a symbol is on, do not resize it until the position has aged. This
        stops the book from trimming a holding every time the ranking wobbles by a
        hair, which is the dominant cost in a per-bar rebalancer.
    ``max_fills_per_day``
        A hard budget, spent in order of conviction. It is a ceiling, never a
        target: a day with no qualifying opportunity produces no fills at all.

    ``round_trips`` counts complete flat-to-flat episodes per symbol, not fills.
    One position trimmed twenty times in a day is one trade that was managed, and
    reporting it as twenty is how a churning book looks busy instead of expensive.
    """
    stamps = weights.index
    active = [s for s in weights.columns if weights[s].abs().sum() > 0]
    if not active or len(stamps) < 2:
        return {"error": "no active positions"}

    fee_rate = fee_bps / 10_000.0
    slip_rate = slippage_bps / 10_000.0
    band = rebalance_band_bps / 10_000.0
    gate_rate = min_expected_move_bps / 10_000.0
    if gate_mode not in {"flat", "hold"}:
        raise ValueError(f"gate_mode must be 'flat' or 'hold', got {gate_mode!r}")

    move_grid: pd.DataFrame | None = None
    if min_expected_move_bps > 0:
        if expected_move is None:
            raise ValueError(
                "min_expected_move_bps was requested but no expected_move grid was supplied"
            )
        # ``expected_move`` is stamped with the bar whose close produced the
        # forecast, and the decision acted on at open(t+1) is the one shifted from
        # bar t, so the gate must read the same shifted row.
        move_grid = align_gate_to_decision(expected_move, stamps)

    # Bar return actually earned by a position filled at this bar's open. An asset
    # we cannot quote on a bar we hold it is not holdable, so its weight is zeroed
    # rather than carried at a price nobody could have traded. Prices are loaded
    # for every column of the weight matrix, not only the currently-active ones: a
    # symbol that is flat for the whole window still needs a validity entry, or
    # the unpriceable-weight pass below has nothing to look it up in.
    asset_return = pd.DataFrame(0.0, index=stamps, columns=list(weights.columns))
    valid_bars: dict[str, np.ndarray] = {}
    open_px_all: dict[str, np.ndarray] = {}
    close_px_all: dict[str, np.ndarray] = {}
    for symbol in weights.columns:
        ohlc = store.read(
            "ohlcv", filters=[("interval", "==", interval)], symbols=[symbol],
            columns=["ts", "open", "close", "symbol"],
        ).sort_values("ts")
        if ohlc.empty:
            valid_bars[symbol] = np.zeros(len(stamps), dtype=bool)
            open_px_all[symbol] = np.full(len(stamps), np.nan)
            close_px_all[symbol] = np.full(len(stamps), np.nan)
            continue
        ohlc["ts"] = pd.DatetimeIndex(ohlc["ts"])
        if interval == "1h":
            ohlc = ohlc[ohlc["ts"] >= SIM_START]
        bars = ohlc.set_index("ts").reindex(stamps)
        open_px = bars["open"].to_numpy(dtype=float)
        close_px = bars["close"].to_numpy(dtype=float)
        valid = np.isfinite(open_px) & np.isfinite(close_px) & (open_px > 0)
        valid_bars[symbol] = valid
        open_px_all[symbol] = np.where(valid, open_px, np.nan)
        close_px_all[symbol] = np.where(valid, close_px, np.nan)
        asset_return[symbol] = np.where(
            valid, close_px / np.where(open_px > 0, open_px, 1.0) - 1.0, 0.0,
        )
    asset_return.attrs["valid"] = valid_bars
    asset_return.attrs["opens"] = open_px_all
    asset_return.attrs["closes"] = close_px_all

    # A weight we have no quote for is a weight we cannot hold, so it is flattened
    # to zero rather than carried at a price nobody could not have traded.
    held = weights.reindex(stamps).fillna(0.0)
    unpriceable_bars = 0
    for symbol in held.columns:
        bars_ok = asset_return.attrs["valid"][symbol]
        unpriceable_bars += int((~bars_ok & held[symbol].abs().gt(1e-12)).sum())
        held.loc[~bars_ok, symbol] = 0.0

    opens = asset_return.attrs["opens"]
    closes = asset_return.attrs["closes"]
    decision = held.shift(1).fillna(0.0)

    cash = float(initial_capital)
    units = {symbol: 0.0 for symbol in active}
    previous_close = {symbol: float("nan") for symbol in active}
    curve = np.empty(len(stamps), dtype=float)
    bar_returns = np.empty(len(stamps), dtype=float)
    contribution = {symbol: 0.0 for symbol in active}
    trade_log: list[dict[str, Any]] = []
    fee_total = 0.0
    slippage_total = 0.0
    turnover_notional = 0.0
    blocked_by_gate = 0
    blocked_by_hold = 0
    blocked_by_budget = 0
    last_trade_bar = {symbol: -(10**9) for symbol in active}
    # A round trip is a flat-to-flat episode on one symbol. While it is open we
    # accumulate its exact cash flows, so the episode's P&L is a sum of real fills
    # rather than an estimate from first entry to last exit.
    episode_open = {symbol: False for symbol in active}
    episode_start = {symbol: 0 for symbol in active}
    episode_flow = {symbol: 0.0 for symbol in active}   # signed cash out, excl. fees
    episode_fees = {symbol: 0.0 for symbol in active}
    episode_notional = {symbol: 0.0 for symbol in active}
    # Direction the episode was opened in. It cannot be recovered from the net
    # flow: a long entry is cash *out* (positive flow) and the exit cancels it, so
    # at close the sign is zero and would label every completed round trip "short".
    episode_side = {symbol: "" for symbol in active}
    round_trips: list[dict[str, Any]] = []
    fills_today = 0
    current_day: Any = None

    curve[0] = cash
    bar_returns[0] = 0.0

    # Same loop as BacktestEngine.run_signal_strategy, generalised to a book of
    # assets: rebalance to the weight published on the previous bar, fill at this
    # bar's open with slippage, mark at this bar's close. Keeping the convention
    # identical is what makes this comparable with the single-asset benchmarks.
    for position in range(1, len(stamps)):
        targets = decision.iloc[position]
        equity_at_open = cash + sum(
            units[s] * opens[s][position] for s in active
        )
        if equity_at_open <= 0:
            curve[position:] = 0.0
            bar_returns[position:] = -1.0
            break

        day = stamps[position].date()
        if day != current_day:
            current_day = day
            fills_today = 0

        # Attribution is done on marked value, not on fills: a symbol's P&L for the
        # bar is the change in what its position is worth, less the cash its trades
        # consumed. Booking only the fills misses the revaluation of units that were
        # already held, which is most of a long/short book's return.
        base_value = {
            symbol: units[symbol] * (
                previous_close[symbol]
                if np.isfinite(previous_close[symbol]) else 0.0
            )
            for symbol in active
        }
        traded_cash: dict[str, float] = {symbol: 0.0 for symbol in active}

        def gated_weight(symbol: str, _t=targets, _p=position) -> float:
            """Target weight for ``symbol`` after the expected-move gate.

            Returns the ungated weight when the gate is off or the symbol's
            forecast clears the bar, and 0.0 when conviction is insufficient.
            """
            if move_grid is None:
                return float(_t[symbol])
            predicted = move_grid.iat[_p, move_grid.columns.get_loc(symbol)]
            predicted = float(predicted) if pd.notna(predicted) else 0.0
            if abs(predicted) < gate_rate:
                return 0.0
            return float(_t[symbol])

        def conviction(symbol: str, _t=targets, _p=position, _e=equity_at_open) -> float:
            """How much this symbol deserves the budget, in comparable units."""
            if move_grid is not None:
                predicted = move_grid.iat[_p, move_grid.columns.get_loc(symbol)]
                return abs(float(predicted)) if pd.notna(predicted) else 0.0
            open_px_s = float(opens[symbol][_p])
            if not np.isfinite(open_px_s) or open_px_s <= 0:
                return 0.0
            delta_s = float(_t[symbol]) * _e / open_px_s - units[symbol]
            return abs(delta_s) * open_px_s

        # Spend the daily budget where it matters most. Two ordering rules, and
        # the second is the important one: a capped day must never let a fresh
        # entry displace the exit of a position that is already on the book.
        #
        # Conviction cannot be |target weight|: a ±1/6 book ties on all six slots,
        # so that key degenerates to column order and the budget gets spent on
        # whichever symbol sorts first rather than the strongest opportunity.
        def rank(symbol: str) -> tuple[int, float, str]:
            target_after_gate = gated_weight(symbol)
            held = abs(units[symbol])
            reducing = held > 1e-12 and abs(target_after_gate) < held - 1e-12
            return (0 if reducing else 1, -conviction(symbol), symbol)

        order = sorted(active, key=rank)

        for symbol in order:
            open_px = float(opens[symbol][position])
            target_weight = float(targets[symbol])

            # Restraint 1: is the model's own predicted move worth the spread?
            # Read through ``gated_weight`` so the ordering pass above and this
            # pass cannot disagree about which symbols the gate admitted.
            gated = gated_weight(symbol)
            if gated == 0.0 and move_grid is not None:
                blocked_by_gate += 1
                if units[symbol] == 0.0:
                    continue          # never open below the bar
                if gate_mode == "hold":
                    continue          # keep it, but do not resize it
                target_weight = 0.0   # "flat": conviction died, close it
            else:
                target_weight = gated

            desired = (
                0.0 if not np.isfinite(open_px) or open_px <= 0
                else target_weight * equity_at_open / open_px
            )
            delta = desired - units[symbol]

            # Restraint 2: do not churn a position that has not aged.
            if (
                min_hold_bars > 0
                and abs(delta) > 1e-12
                and position - last_trade_bar[symbol] < min_hold_bars
            ):
                blocked_by_hold += 1
                continue

            # Targeting the published weight every bar means trimming each holding
            # back on every price move, and at 15bps a side that churn is the
            # dominant cost in the account. A band keeps the position until its
            # weight has drifted far enough to be worth re-establishing, which is
            # how a real venue-experienced book is run.
            if abs(delta * open_px) / equity_at_open < band:
                continue
            if abs(delta) <= 1e-12:
                continue

            # Restraint 3: hard daily ceiling. A day with no qualifying opportunity
            # simply produces no fills; nothing is forced to hit a count.
            if max_fills_per_day is not None and fills_today >= max_fills_per_day:
                blocked_by_budget += 1
                continue

            fill_price = open_px * (1.0 + slip_rate * np.sign(delta))
            fee = abs(delta) * fill_price * fee_rate
            cash -= delta * fill_price + fee
            prev_units = units[symbol]
            units[symbol] = desired
            last_trade_bar[symbol] = position
            fills_today += 1
            fee_total += fee
            # Slippage is not a cash line of its own; it is the difference between
            # the fill and the touch, and it is already inside the mark below.
            slippage_total += abs(delta) * open_px * slip_rate
            traded_cash[symbol] += delta * fill_price + fee
            turnover_notional += abs(delta) * fill_price
            trade_log.append({
                "ts": stamps[position].isoformat(), "symbol": symbol,
                "side": "buy" if delta > 0 else "sell",
                "quantity": abs(delta), "price": float(fill_price),
                "notional": abs(delta) * fill_price, "fee": float(fee),
            })

            def close_episode(at_bar: int, _s=symbol) -> None:
                gross_pnl = -episode_flow[_s]
                net_pnl = gross_pnl - episode_fees[_s]
                deployed = episode_notional[_s]
                round_trips.append({
                    "symbol": _s,
                    "entry_ts": stamps[episode_start[_s]].isoformat(),
                    "exit_ts": stamps[at_bar].isoformat(),
                    "bars_held": at_bar - episode_start[_s],
                    "hours_held": (at_bar - episode_start[_s]) * _hours_per_bar(interval),
                    "deployed_notional": float(deployed),
                    "gross_pnl": float(gross_pnl),
                    "net_pnl": float(net_pnl),
                    "gross_return": float(gross_pnl / deployed) if deployed > 0 else 0.0,
                    "net_return": float(net_pnl / deployed) if deployed > 0 else 0.0,
                    "side": episode_side[_s],
                })
                episode_open[_s] = False

            # Round-trip bookkeeping. An episode opens when a flat symbol is given a
            # position and closes when it returns to flat, so the reported trade
            # count is decisions rather than fills.
            #
            # A fill that crosses straight through zero -- long straight to short --
            # does both at once, so it is split into the leg that retires the old
            # position and the leg that opens the new one. Accounting for the whole
            # ``delta`` against one episode would weld a long and a short into a
            # single round trip that never actually flattened, and would charge the
            # new position for notional it never deployed.
            closing_leg = 0.0
            opening_leg = delta
            if prev_units != 0.0 and (
                desired == 0.0 or np.sign(desired) != np.sign(prev_units)
            ):
                closing_leg = -prev_units
                opening_leg = delta - closing_leg      # exactly ``desired``

            if episode_open[symbol] and closing_leg != 0.0:
                episode_flow[symbol] += closing_leg * fill_price
                episode_fees[symbol] += abs(closing_leg) * fill_price * fee_rate
                episode_notional[symbol] += abs(closing_leg) * fill_price
                close_episode(position)
            if opening_leg != 0.0 and not episode_open[symbol]:
                episode_open[symbol] = True
                episode_start[symbol] = position
                episode_flow[symbol] = 0.0
                episode_fees[symbol] = 0.0
                episode_notional[symbol] = 0.0
                episode_side[symbol] = "long" if opening_leg > 0 else "short"
            if episode_open[symbol]:
                episode_flow[symbol] += opening_leg * fill_price
                episode_fees[symbol] += abs(opening_leg) * fill_price * fee_rate
                episode_notional[symbol] += abs(opening_leg) * fill_price

        opening_equity = equity_at_open
        equity = cash + sum(units[s] * float(closes[s][position]) for s in active)
        for symbol in active:
            close_px = float(closes[symbol][position])
            end_value = units[symbol] * close_px
            contribution[symbol] += (end_value - base_value[symbol]) - traded_cash[symbol]
            previous_close[symbol] = close_px

        curve[position] = equity
        bar_returns[position] = (
            (equity - opening_equity) / opening_equity if opening_equity > 0 else 0.0
        )

    frame = pd.DataFrame(
        {"equity": curve, "returns": bar_returns},
        index=stamps,
    )
    metrics = compute_metrics(
        bar_returns,
        initial_capital=initial_capital,
        periods_per_year=periods_per_year(interval, label_horizon=1),
    )

    max_slots = int((held.abs() > 1e-12).sum(axis=1).max())
    gross = held.abs().sum(axis=1)
    per_symbol = {
        symbol: {
            "total_return": (contribution[symbol] / initial_capital) if initial_capital else 0.0,
            "contribution": contribution[symbol],
            "trades": sum(1 for t in trade_log if t["symbol"] == symbol),
        }
        for symbol in active
    }

    return {
        "equity": frame,
        "metrics": metrics,
        "trades": trade_log,
        "n_trades": len(trade_log),
        "round_trips": round_trips,
        "n_round_trips": len(round_trips),
        "total_fees": fee_total,
        "total_slippage": slippage_total,
        "total_cost": fee_total + slippage_total,
        "turnover_notional": turnover_notional,
        "turnover_x_initial": turnover_notional / initial_capital if initial_capital else 0.0,
        "restraint": {
            "blocked_by_gate": blocked_by_gate,
            "blocked_by_min_hold": blocked_by_hold,
            "blocked_by_budget": blocked_by_budget,
        },
        "policy": {
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "rebalance_band_bps": rebalance_band_bps,
            "min_expected_move_bps": min_expected_move_bps,
            "min_hold_bars": min_hold_bars,
            "max_fills_per_day": max_fills_per_day,
            "gate_mode": gate_mode,
        },
        "exposure": float((gross > 0).mean()),
        "mean_gross_exposure": float(gross.mean()),
        "max_gross_exposure": float(gross.max()),
        "per_symbol": per_symbol,
        "max_concurrent_slots": max_slots,
        "per_slot_capital": initial_capital / max_slots if max_slots else 0.0,
        "unpriceable_bar_symbols": int(unpriceable_bars),
    }




def benchmark_buy_and_hold(
    store: ParquetStore, symbol: str, interval: str, stamps: pd.DatetimeIndex,
    *, initial_capital: float, fee_bps: float, slippage_bps: float,
) -> dict[str, Any]:
    ohlc = store.read(
        "ohlcv", filters=[("interval", "==", interval)], symbols=[symbol],
        columns=["ts", "open", "close"],
    ).sort_values("ts")
    ohlc["ts"] = pd.DatetimeIndex(ohlc["ts"])
    ohlc = ohlc[ohlc["ts"].isin(stamps)].reset_index(drop=True)
    if len(ohlc) < 10:
        return {"error": "insufficient benchmark data"}
    engine = BacktestEngine(Settings(), periods_per_year=periods_per_year(interval, label_horizon=1))
    engine.fee_rate = fee_bps / 10_000.0
    engine.slippage_rate = slippage_bps / 10_000.0
    result = engine.run_buy_and_hold(ohlc, initial_capital=initial_capital)
    # `run_buy_and_hold` emits len(ohlc) - 1 bars, the i-th being the move from
    # close[i] to close[i+1], so it is stamped with the *later* close's time.
    equity = pd.Series(result["equity_curve"], index=ohlc["ts"].iloc[1:].to_numpy())
    return {"metrics": result["metrics"], "equity": equity}


def benchmark_equal_weight(
    store: ParquetStore, symbols: Sequence[str], interval: str, stamps: pd.DatetimeIndex,
    *, initial_capital: float, fee_bps: float, slippage_bps: float,
) -> dict[str, Any]:
    """Buy the whole universe once at the first stamp and hold, cost-charged."""
    curves = []
    for symbol in symbols:
        ohlc = store.read(
            "ohlcv", filters=[("interval", "==", interval)], symbols=[symbol],
            columns=["ts", "close"],
        ).sort_values("ts")
        ohlc["ts"] = pd.DatetimeIndex(ohlc["ts"])
        ohlc = ohlc[ohlc["ts"].isin(stamps)].reset_index(drop=True)
        if len(ohlc) < 10:
            continue
        closes = ohlc["close"].to_numpy(dtype=float)
        equity = closes / closes[0] * (initial_capital / len(symbols))
        # Charged on the way in only; the benchmark never trades again, and the
        # bot pays the same entry cost every time it resizes.
        equity[0] *= (1 - fee_bps / 10_000.0) * (1 + slippage_bps / 10_000.0)
        curves.append(pd.Series(equity, index=ohlc["ts"]))
    if not curves:
        return {"error": "insufficient benchmark data"}
    total = pd.concat(curves, axis=1).sum(axis=1).sort_index()
    returns = total.pct_change().fillna(0.0)
    return {
        "metrics": compute_metrics(
            returns.to_numpy(), initial_capital=initial_capital,
            periods_per_year=periods_per_year(interval, label_horizon=1),
        ),
        "equity": total,
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def _money(value: float | None) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"${value:,.2f}"


def _pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{100 * value:+.{digits}f}%"


def run_interval(
    store: ParquetStore,
    universe: Universe,
    interval: str,
    args: argparse.Namespace,
    settings: Settings,
) -> dict[str, Any]:
    trainer = ForecasterTrainer(settings)
    warmup = args.warmup_1h if interval == "1h" else args.warmup_1d
    refit = args.refit_1h if interval == "1h" else args.refit_1d
    # Hourly trades everything after its warm-up, because one year is all the
    # exchange will serve. Daily trades the final year and warms up on all of
    # the history behind it.
    trade_bars = 10 ** 9 if interval == "1h" else args.trade_1d

    oos_by_symbol: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, Any] = {}
    prepared_cache: dict[str, Prepared] = {}

    for symbol in universe.symbols:
        started = time.time()
        try:
            prepared = prepare_symbol(
                store, symbol, interval,
                warmup_bars=warmup, trade_bars=trade_bars, refit_bars=refit, horizon=1,
                min_test_bars=args.min_test_bars,
            )
        except ValueError as exc:
            print(f"  {symbol:<12} skipped: {exc}")
            continue
        prepared_cache[symbol] = prepared

        result = walk_forward(
            prepared, trainer,
            seq_len=args.seq_len, horizon=1, epochs=args.epochs, kind=args.kind,
            val_ratio=0.25, seed=settings.seed, n_boot=args.n_boot,
        )
        if result.oos.empty:
            print(f"  {symbol:<12} no out-of-sample predictions")
            continue
        oos_by_symbol[symbol] = result.oos
        diagnostics[symbol] = result.diagnostics
        print(
            f"  {symbol:<12} oos={len(result.oos):>5} bars  "
            f"acc={result.diagnostics['directional_accuracy']:.3f}  "
            f"auc={result.diagnostics['auc']:.3f} "
            f"[{result.diagnostics['auc_ci_low']:.3f}, {result.diagnostics['auc_ci_high']:.3f}]  "
            f"volscale={result.diagnostics['mean_vol_scale']:.2f}  "
            f"({time.time() - started:.1f}s)"
        )

    if not oos_by_symbol:
        return {"error": "no symbol produced out-of-sample predictions"}

    weights, probs = build_position_matrix(oos_by_symbol, top_k=args.top_k,
                                           prob_bar=args.prob_bar)

    # Expected-move grid: the model's own predicted return per symbol per bar, in
    # the same units as the return it is trying to earn. This is the input an
    # expected-move gate needs -- `probs` is a probability and `score` has been
    # divided by trailing volatility, so neither can say whether a move is big
    # enough to clear the spread.
    move_grid = pd.DataFrame(
        {
            symbol: pd.Series(
                frame["predicted_return"].to_numpy(dtype=float),
                index=pd.DatetimeIndex(frame["ts"]),
            )
            for symbol, frame in oos_by_symbol.items()
            if "predicted_return" in frame.columns
        }
    ) if args.min_expected_move_bps > 0 else None

    if args.save_weights:
        # Persisting the weight matrix separates the expensive part (training)
        # from the cheap part (execution), so fee, slippage and slot-count
        # sensitivity can be replayed without refitting anything. Without it a
        # cost sweep costs another full training run per scenario. The interval is
        # part of the name so a two-interval run keeps both books rather than the
        # second one silently overwriting the first.
        path = Path(args.save_weights)
        requested = [i.strip() for i in args.intervals.split(",") if i.strip()]
        if len(requested) > 1:
            path = path.with_name(f"{path.stem}_{interval}{path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        weights.to_parquet(path)
        print(f"  weights saved: {path}  shape {weights.shape}")

    execution = run_portfolio(
        store, universe.symbols, interval, weights,
        initial_capital=args.capital, fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        rebalance_band_bps=args.rebalance_band_bps,
        expected_move=move_grid,
        min_expected_move_bps=args.min_expected_move_bps,
        min_hold_bars=args.min_hold_bars,
        max_fills_per_day=args.max_fills_per_day,
        gate_mode=args.gate_mode,
    )
    if "error" in execution:
        return {"error": execution["error"]}

    stamps = execution["equity"].index
    btc = benchmark_buy_and_hold(
        store, "BTC_USDT", interval, stamps,
        initial_capital=args.capital, fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    equal = benchmark_equal_weight(
        store, universe.symbols, interval, stamps,
        initial_capital=args.capital, fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )

    # Pooled ranking power across the whole universe, which is the number the
    # selection rule actually depends on.
    pooled = pd.concat(
        [df.assign(symbol=s) for s, df in oos_by_symbol.items()], ignore_index=True
    )

    if args.save_predictions:
        # Persist the per-bar forecasts so trading-logic experiments can be replayed
        # without refitting. `score` alone cannot drive an expected-move gate: it is
        # divided by trailing volatility, so it ranks across the universe but carries
        # no information about how big the move is.
        path = Path(args.save_predictions)
        requested = [i.strip() for i in args.intervals.split(",") if i.strip()]
        if len(requested) > 1:
            path = path.with_name(f"{path.stem}_{interval}{path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        keep = [
            c for c in ("ts", "symbol", "score", "predicted_return", "target_return",
                        "target_volatility", "vol_scale")
            if c in pooled.columns
        ]
        pooled[keep].sort_values(["ts", "symbol"]).to_parquet(path, index=False)
        print(f"  predictions saved: {path}  rows {len(pooled)}")
    pooled_labels = (pooled["target_return"] > 0).astype(float)
    pooled_auc, pooled_low, pooled_high = _bootstrap_auc(
        pooled["score"].to_numpy(), pooled_labels.to_numpy(),
        n_boot=args.n_boot, seed=settings.seed,
    )

    # Excess return is only meaningful bar-for-bar, so restrict to timestamps
    # where both the strategy and the benchmark actually have a mark. Padding
    # either side with zeros would manufacture a difference out of alignment.
    strategy_returns = execution["equity"]["returns"]
    if "metrics" in btc:
        common = strategy_returns.index.intersection(btc["equity"].index)
        bench_returns = (
            btc["equity"].reindex(common).pct_change().fillna(0.0)
        )
        excess_mean, excess_low, excess_high = _bootstrap_mean(
            (strategy_returns.reindex(common) - bench_returns).to_numpy(),
            n_boot=args.n_boot, seed=settings.seed,
        )
        n_common = len(common)
    else:
        excess_mean, excess_low, excess_high, n_common = (float("nan"),) * 4

    total_fees = execution["total_fees"]

    # Cost sensitivity re-runs the same weights through the same engine at
    # different fee levels. It is the only way to tell a strategy that has no
    # edge apart from one whose edge is real but smaller than the bill for
    # harvesting it.
    #
    # The fee-0 row is the *truly* gross value: slippage is zeroed there too.
    # Leaving slippage on while calling the row "before any costs" understates
    # the gross edge by exactly the spread drag, which at this turnover is the
    # difference between an edge worth having and no edge at all -- so the row
    # has to be honest about both levers.
    sweep: list[dict[str, Any]] = []
    if args.fee_sweep:
        for fee in [float(f) for f in args.fee_sweep.split(",") if f.strip()]:
            trial = run_portfolio(
                store, universe.symbols, interval, weights,
                initial_capital=args.capital, fee_bps=fee,
                slippage_bps=0.0 if fee == 0 else args.slippage_bps,
                rebalance_band_bps=args.rebalance_band_bps,
                expected_move=move_grid,
                min_expected_move_bps=args.min_expected_move_bps,
                min_hold_bars=args.min_hold_bars,
                max_fills_per_day=args.max_fills_per_day,
                gate_mode=args.gate_mode,
            )
            if "error" in trial:
                continue
            sweep.append({
                "fee_bps": fee,
                "slippage_bps": 0.0 if fee == 0 else args.slippage_bps,
                "is_gross": fee == 0,
                "total_return": trial["metrics"].get("total_return"),
                "sharpe": trial["metrics"].get("sharpe"),
                "max_drawdown": trial["metrics"].get("max_drawdown"),
                "fees_paid": sum(abs(t.get("fee", 0.0)) for t in trial["trades"]),
                "slippage_paid": trial["total_slippage"],
                "turnover_x_initial": trial["turnover_x_initial"],
                "n_fills": trial["n_trades"],
                "n_trades": trial["n_round_trips"],
                "final_equity": trial["metrics"].get("final_equity"),
            })

    return {
        "interval": interval,
        "n_symbols": len(oos_by_symbol),
        "weights": weights,
        "probabilities": probs,
        "execution": execution,
        "metrics": execution["metrics"],
        "benchmarks": {"btc_usdt_buy_hold": btc, "equal_weight_buy_hold": equal},
        "diagnostics": diagnostics,
        "pooled": {
            "oos_bars": int(len(pooled)),
            "auc": pooled_auc,
            "auc_ci_low": pooled_low,
            "auc_ci_high": pooled_high,
            "auc_passes": bool(np.isfinite(pooled_low) and pooled_low > 0.5),
            "directional_accuracy": float(np.mean(
                np.sign(pooled["target_return"]) == np.sign(pooled["score"])
            )),
            "correlation": float(np.corrcoef(
                pooled["target_return"], pooled["score"]
            )[0, 1]),
            "mean_bar_win_rate": float(np.mean(
                pooled.groupby("ts")["target_return"].apply(lambda s: (s > 0).mean())
            )),
        },
        "excess_vs_btc": {
            "mean_bar_return": excess_mean,
            "ci_low": excess_low,
            "ci_high": excess_high,
            "significant": bool(np.isfinite(excess_low) and excess_low > 0),
            "n_common_bars": n_common,
        },
            "total_fees": total_fees,
        "total_slippage": execution["total_slippage"],
        "fee_sweep": sweep,
        "trading_days": (stamps[-1] - stamps[0]).total_seconds() / 86_400.0,
    }


def print_report(result: dict[str, Any], args: argparse.Namespace) -> None:
    interval = result["interval"]
    metrics = result["metrics"]
    pooled = result["pooled"]

    print()
    print("=" * 78)
    print(f"{interval.upper()} BOT   capital ${args.capital:,.0f}   "
          f"{result['n_symbols']} symbols   {result['trading_days']:.0f} trading days")
    print("=" * 78)

    print("\n-- Does the probability score actually rank? " + "-" * 34)
    verdict = "PASSES" if pooled["auc_passes"] else "FAILS"
    print(f"   pooled out-of-sample bars : {pooled['oos_bars']:,}")
    print(f"   directional accuracy      : {pooled['directional_accuracy']:.4f}  (0.5000 = coin flip)")
    print(f"   return/score correlation : {pooled['correlation']:+.4f}")
    print(f"   AUC                       : {pooled['auc']:.4f} "
          f"[{pooled['auc_ci_low']:.4f}, {pooled['auc_ci_high']:.4f}]")
    print(f"   ranking power             : {verdict} "
          f"(95% interval {'excludes' if pooled['auc_passes'] else 'includes'} 0.500)")
    if not pooled["auc_passes"]:
        print("   -> the score cannot separate winners from losers. Selecting the")
        print("      'highest probability' trades is selecting noise.")

    print("\n-- Calibrated probability vs realised win rate " + "-" * 24)
    reference = next(iter(result["diagnostics"].values()), {})
    calibration = reference.get("calibration") or []
    if calibration:
        print("   (one representative symbol; decile 10 = its most bullish bars)")
        print(f"   {'decile':>7} {'n':>7} {'mean score':>12} {'claimed':>9} {'observed':>10}")
        for row in calibration:
            print(f"   {row['decile']:>7} {row['n']:>7} {row['mean_score']:>12.3f} "
                  f"{row['claimed_p_profit']:>9.3f} {row['observed_p_profit']:>10.3f}")
        spread = calibration[-1]["observed_p_profit"] - calibration[0]["observed_p_profit"]
        print(f"   observed win-rate spread across deciles: {spread:+.3f}")
    scales = [d.get("mean_vol_scale") for d in result["diagnostics"].values()
              if np.isfinite(d.get("mean_vol_scale", np.nan))]
    if scales:
        print(f"   mean volatility correction applied: {np.mean(scales):.2f}x "
              f"(>1 means the model understates its own error)")

    print("\n-- Result " + "-" * 66)
    final = metrics.get("final_equity", float("nan"))
    print(f"   starting capital          : ${args.capital:,.2f}")
    print(f"   ending capital            : {_money(final)}")
    print(f"   NET PROFIT / LOSS         : {_money(final - args.capital)}   "
          f"({_pct(metrics.get('total_return'))})")
    print(f"   max drawdown              : {_pct(metrics.get('max_drawdown'))}")
    print(f"   Sharpe / Sortino          : {metrics.get('sharpe', 0):.2f} / "
          f"{metrics.get('sortino', 0):.2f}")
    print(f"   trades executed           : {result['execution']['n_trades']:,}")
    print(f"   fees paid                 : {_money(result['total_fees'])}")
    print(f"   slippage paid             : {_money(result['total_slippage'])}")
    print(f"   time in market            : {100 * result['execution']['exposure']:.1f}% of bars")
    print(f"   mean gross exposure       : {result['execution']['mean_gross_exposure']:.2f} "
          f"(max {result['execution']['max_gross_exposure']:.2f}, 1.00 = fully deployed)")

    print("\n-- Against benchmarks, same window, same fees " + "-" * 32)
    print(f"   {'':<28} {'return':>10} {'Sharpe':>8} {'maxDD':>9}")
    rows = [
        ("this bot", metrics),
        ("BTC buy-and-hold", (result["benchmarks"]["btc_usdt_buy_hold"] or {}).get("metrics") or {}),
        ("universe equal-weight hold", (result["benchmarks"]["equal_weight_buy_hold"] or {}).get("metrics") or {}),
    ]
    for label, m in rows:
        print(f"   {label:<28} {_pct(m.get('total_return')):>10} "
              f"{m.get('sharpe', 0):>8.2f} {_pct(m.get('max_drawdown')):>9}")
    excess = result["excess_vs_btc"]
    print(f"   excess return vs BTC per bar: {excess['mean_bar_return']:+.6f} "
          f"[{excess['ci_low']:+.6f}, {excess['ci_high']:+.6f}]")
    print(f"   statistically significant   : {excess['significant']}")

    if args.prob_bar > 0:
        print(f"\n   conviction filter active: bars with P(profit) < {args.prob_bar:.2f} "
              f"or P(loss) < {args.prob_bar:.2f} were skipped.")

    symbols = sorted(result["execution"]["per_symbol"].items(),
                     key=lambda kv: -(kv[1].get("contribution") or -9e9))
    print("\n-- Per-symbol contribution " + "-" * 48)
    print("   dollar P&L is each symbol's share of the one shared account, net of "
          "the fees and")
    print("   slippage it caused, so the column sums to the account's net P&L")
    print(f"   {'symbol':<14} {'P&L':>11} {'return':>9} {'trades':>8}")
    for symbol, m in symbols[:12]:
        pnl = float(m.get("contribution") or 0.0)
        print(f"   {symbol:<14} {_money(pnl):>11} "
              f"{_pct(m.get('total_return')):>9} {int(m.get('trades') or 0):>8}")

    if result.get("fee_sweep"):
        print("\n-- Cost sensitivity: same signals, different venue " + "-" * 21)
        print("   the fee=0 row is genuinely gross: fees AND slippage are both zeroed there")
        print(f"   {'fee':>6} {'net P&L':>11} {'return':>9} {'Sharpe':>8} "
              f"{'fees paid':>11} {'maxDD':>9}")
        for row in result["fee_sweep"]:
            print(f"   {row['fee_bps']:>5.0f}b {_money((row['final_equity'] or 0) - args.capital):>11} "
                  f"{_pct(row['total_return']):>9} {(row['sharpe'] or 0):>8.2f} "
                  f"{_money(row['fees_paid']):>11} {_pct(row['max_drawdown']):>9}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capital", type=float, default=1_000.0)
    parser.add_argument("--top-k", type=int, default=3,
                        help="longs and shorts per bar (2*top_k slots)")
    parser.add_argument("--assets", type=int, default=25)
    parser.add_argument("--prob-bar", type=float, default=0.0,
                        help="skip bars where best P(profit) is below this")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--kind", default="lstm")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--warmup-1h", type=int, default=1_500,
                        help="hourly bars held back for training before trading starts")
    parser.add_argument("--refit-1h", type=int, default=720,
                        help="hourly bars between model refits (720 = monthly)")
    parser.add_argument("--warmup-1d", type=int, default=730,
                        help="minimum daily bars required before trading starts")
    parser.add_argument("--refit-1d", type=int, default=30,
                        help="daily bars between model refits (30 = monthly)")
    parser.add_argument("--trade-1d", type=int, default=365,
                        help="daily bars traded at the end of history (one year)")
    parser.add_argument("--min-test-bars", type=int, default=20)
    parser.add_argument("--min-hourly-notional", type=float, default=25_000.0)
    parser.add_argument("--min-days", type=float, default=355.0)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--intervals", default="1h,1d")
    parser.add_argument("--save-weights", default=None,
                        help="write the cross-sectional weight matrix to parquet")
    parser.add_argument("--rebalance-band-bps", type=float, default=0.0,
                        help="skip a resize until the drifted weight moves this far (0 = rebalance every bar)")
    parser.add_argument("--fee-sweep", default=None,
                        help="comma-separated fee bps to replay against the saved weights")
    parser.add_argument("--save-predictions", default=None,
                        help="write the per-bar OOS forecasts to parquet so trading "
                             "logic can be re-run without refitting")
    parser.add_argument("--min-expected-move-bps", type=float, default=0.0,
                        help="only trade when |predicted move| clears this (0 = off)")
    parser.add_argument("--min-hold-bars", type=int, default=0,
                        help="minimum bars a position must age before it can be resized")
    parser.add_argument("--max-fills-per-day", type=int, default=None,
                        help="hard daily fill ceiling; never a target")
    parser.add_argument("--gate-mode", choices=("flat", "hold"), default="flat",
                        help="what a below-threshold signal does to an open position")
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--save", default=None, help="write the full report as JSON")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    settings = Settings(log_level="ERROR", allow_model_download=False)
    # The module-level loggers in cqip capture INFO on first use via
    # `get_logger`, so constructing a quiet Settings is not enough on its own.
    configure_logging(settings, force=True)
    store = ParquetStore(settings=settings)

    print(f"CQIP one-year simulation | capital ${args.capital:,.0f} | "
          f"top {args.top_k} long + top {args.top_k} short | "
          f"{args.fee_bps:.0f}bps fee + {args.slippage_bps:.0f}bps slippage")
    print(f"window starts {SIM_START.date()}, the earliest hour the exchange will serve")

    universe = select_universe(
        store, top_k_assets=args.assets,
        min_hourly_notional=args.min_hourly_notional, min_days=args.min_days,
    )
    print(f"\nUniverse: {len(universe.symbols)} liquid pairs, "
          f"median hourly notional ${universe.detail['med_hourly_notional'].min():,.0f} "
          f"to ${universe.detail['med_hourly_notional'].max():,.0f}")
    print(universe.detail[["symbol", "bars_1h", "span_days", "med_hourly_notional"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "capital": args.capital,
        "config": {k: v for k, v in vars(args).items()},
        "universe": universe.detail.to_dict(orient="records"),
        "intervals": {},
    }

    for interval in [i.strip() for i in args.intervals.split(",") if i.strip()]:
        print(f"\n{'=' * 78}\n{interval.upper()} -- walk-forward forecast\n{'=' * 78}")
        started = time.time()
        result = run_interval(store, universe, interval, args, settings)
        if "error" in result:
            print(f"  {result['error']}")
            continue
        print(f"  forecast stage finished in {time.time() - started:.1f}s")
        print_report(result, args)
        report["intervals"][interval] = {
            "metrics": result["metrics"],
            "pooled_diagnostics": result["pooled"],
            "excess_vs_btc": result["excess_vs_btc"],
            "exposure": result["execution"]["exposure"],
            "n_trades": result["execution"]["n_trades"],
            "total_fees": result["total_fees"],
            "total_slippage": result["total_slippage"],
            "trading_days": result["trading_days"],
            "benchmarks": {
                name: (value or {}).get("metrics")
                for name, value in result["benchmarks"].items()
            },
            "per_symbol": result["execution"]["per_symbol"],
            "per_symbol_diagnostics": result["diagnostics"],
            "slot_accounting": {
                "max_concurrent_slots": result["execution"]["max_concurrent_slots"],
                "per_slot_capital": result["execution"]["per_slot_capital"],
            },
            "fee_sweep": result.get("fee_sweep") or [],
        }

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'bot':<8} {'start':>10} {'end':>12} {'net P&L':>12} {'return':>9} {'maxDD':>9}")
    for interval, block in report["intervals"].items():
        m = block["metrics"]
        print(f"{interval:<8} ${args.capital:>9,.0f} {_money(m.get('final_equity')):>12} "
              f"{_money((m.get('final_equity') or args.capital) - args.capital):>12} "
              f"{_pct(m.get('total_return')):>9} {_pct(m.get('max_drawdown')):>9}")
    total = sum(
        (block["metrics"].get("final_equity") or args.capital) - args.capital
        for block in report["intervals"].values()
    )
    print(f"{'combined':<8} ${2 * args.capital:>9,.0f} "
          f"{_money(2 * args.capital + total):>12} {_money(total):>12} "
          f"{_pct(total / (2 * args.capital)):>9}")

    print("\nRead this before quoting the number above:")
    for interval, block in report["intervals"].items():
        pooled = block["pooled_diagnostics"]
        if not pooled["auc_passes"]:
            print(f"  - {interval}: the score's 95% AUC interval "
                  f"[{pooled['auc_ci_low']:.3f}, {pooled['auc_ci_high']:.3f}] includes 0.500, "
                  f"so 'highest probability of profit' had no edge to select on.")
    print("  - the comparison that matters is each bot against buy-and-hold on the")
    print("    same bars, not the absolute dollar figure.")

    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nsaved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())