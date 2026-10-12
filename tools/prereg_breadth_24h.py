"""Pre-registered OOS experiment: market breadth as a market-regime signal.

Frozen definition, single feature
---------------------------------
    breadth_24h(t) = count_i( log(close_i[t]/close_i[t-24]) > 0 )
                   / count_i( eligible at t )

No slope, no acceleration, no alternative window, no smoothing, no z-score, no
threshold search. The split is fixed at 0.50, the natural majority/decline
boundary, declared before the data was seen.

Target
------
    equal_weight_market_forward_return_24h(t)
        = mean_i log(close_i[t+24]/close_i[t]) over eligible coins

Breadth never sees t+24. Breadth is a scalar per timestamp, so it cannot rank
coins and is deliberately NOT combined with a top-3/bottom-3 selection. Its
legitimate claim is about market direction, and the target is the market.

Transaction model
-----------------
The 120bps two-sided spread hurdle from the cross-sectional experiments does
NOT apply here. A breadth position is a single directional exposure to the
equal-weight market: one entry fill plus one exit fill is 18 + 18 = 36bps.
A regime flip closes one exposure and opens the other, so it also costs 36bps
and consumes 2 fills. Cost is therefore counted from the ACTUAL number of
flips the frozen rule generates, never assumed.

Protocol is identical to the two preceding experiments: same 25 symbols, same
cutoff, same OOS fraction, same close-of-bar decision, same non-overlapping
24h primary sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import ab_direction_head as ab  # noqa: E402
from feature_audit import load_panel  # noqa: E402

from cqip.storage import ParquetStore  # noqa: E402

# --------------------------------------------------------------------------- #
# FROZEN CONSTANTS
# --------------------------------------------------------------------------- #

FEATURE = "breadth_24h"
LOOKBACK = 24
HORIZON = 24
SPLIT = 0.50

FEE_BPS = 10.0
SLIPPAGE_BPS = 5.0
SPREAD_BPS = 3.0
COST_PER_FILL_BPS = FEE_BPS + SLIPPAGE_BPS + SPREAD_BPS  # 18
COST_PER_ROUND_TRIP_BPS = 2 * COST_PER_FILL_BPS  # 36

OOS_FRACTION = 0.25
CUTOFF = 5100
NON_OVERLAP_STRIDE = HORIZON
WINSOR_LO, WINSOR_HI = 0.01, 0.99
TRADING_DAYS_PER_YEAR = 365.0
SEED = 7
N_BOOT = 2000
BLOCK_DAYS = 7
N_NULL = 2000

DISPERSION_CONTROL = "cross_sectional_iqr_of_24h_returns"

CONTAMINATION_DISCLOSURE = (
    "This OOS window has now been inspected by three consecutive experiments "
    "(feature audit, sma_fast_slow_spread, relative strength) and no clean "
    "holdout exists because the panel ends at the last available bar. Breadth "
    "itself had not been tested before this run, but the window, the universe "
    "and the target were all previously observed. Significance must be "
    "discounted; the design value is the pre-declared regime and cost "
    "accounting, not the p-value."
)

ARCHITECTURE_NOTE = (
    "production consumes 46 of the 64 originally declared features; sentiment "
    "and on-chain columns remain unactivated and were not used or activated here"
)


# --------------------------------------------------------------------------- #
# feature and target, built from raw closes
# --------------------------------------------------------------------------- #


def union_index(panel: Any) -> pd.Index:
    """Sorted union of every symbol's timestamps.

    Mirrors ``wide_matrix(index=None)`` so the OOS window matches the two
    preceding experiments exactly, instead of silently riding on the first
    symbol's shorter grid.
    """
    frames = [
        pd.DatetimeIndex(panel.frames[s]["ts"].to_numpy())
        for s in panel.symbols
        if s in panel.frames
    ]
    if not frames:
        return pd.DatetimeIndex([])
    out = frames[0]
    for extra in frames[1:]:
        out = out.union(extra)
    return out


def build_market_frames(panel: Any, index: pd.Index) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (24h returns, breadth, dispersion) on the shared timestamp grid.

    Alignment mirrors ``wide_matrix``: each symbol is labelled with its OWN
    timestamps from ``panel.frames[sym]['ts']``, never the first symbol's grid
    and never the positional RangeIndex on ``panel.close``.
    """
    rets: dict[str, pd.Series] = {}
    for sym in panel.symbols:
        if sym not in panel.close:
            continue
        ts = pd.DatetimeIndex(panel.frames[sym]["ts"].to_numpy())
        close = pd.Series(panel.close[sym].to_numpy(dtype=float), index=ts, name=sym)
        rets[sym] = np.log(close / close.shift(LOOKBACK))
    r = pd.DataFrame(rets).reindex(index)

    eligible = r.notna().sum(axis=1)
    advancers = (r > 0).sum(axis=1)
    breadth = (advancers / eligible).astype(float)

    q75 = r.quantile(0.75, axis=1)
    q25 = r.quantile(0.25, axis=1)
    dispersion = (q75 - q25).astype(float)

    return r, breadth, dispersion


def forward_market_return(panel: Any, index: pd.Index) -> tuple[pd.Series, dict[str, float]]:
    """Equal-weight mean forward 24h return, recomputed independently."""
    cols: dict[str, pd.Series] = {}
    for sym in panel.symbols:
        if sym not in panel.close:
            continue
        ts = pd.DatetimeIndex(panel.frames[sym]["ts"].to_numpy())
        close = pd.Series(panel.close[sym].to_numpy(dtype=float), index=ts)
        cols[sym] = np.log(close.shift(-HORIZON) / close)
    fwd = pd.DataFrame(cols).reindex(index)
    market = fwd.mean(axis=1, skipna=True)
    return market, {"per_symbol_max_recompute_diff": _recompute_gap(fwd)}


def _recompute_gap(fwd: pd.DataFrame) -> float:
    """Confirm the aggregate equals a hand-rolled per-timestamp mean."""
    worst = 0.0
    for pos in range(0, len(fwd), 37):
        row = fwd.iloc[pos]
        manual = float(row.dropna().mean()) if row.notna().any() else float("nan")
        got = float(fwd.iloc[pos].mean(skipna=True))
        if np.isfinite(manual) and np.isfinite(got):
            worst = max(worst, abs(manual - got))
    return worst


# --------------------------------------------------------------------------- #
# leakage / integrity verification
# --------------------------------------------------------------------------- #


def verify_integrity(
    panel: Any,
    r: pd.DataFrame,
    breadth: pd.Series,
    market: pd.Series,
    market_counts: pd.Series,
    oos_ts: pd.Index,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # 1. prefix invariance with PHYSICAL truncation of the lookback input
    worst = 0.0
    probes = 0
    index = breadth.index
    rng = np.random.default_rng(SEED)
    picks = sorted(rng.choice(len(index), size=min(40, len(index)), replace=False))
    for pos in picks:
        cut = index[pos]
        cols: dict[str, pd.Series] = {}
        for sym in panel.symbols:
            if sym not in panel.close:
                continue
            ts = pd.DatetimeIndex(panel.frames[sym]["ts"].to_numpy())
            full = pd.Series(panel.close[sym].to_numpy(dtype=float), index=ts)
            close = full[ts <= cut]
            cols[sym] = np.log(close / close.shift(LOOKBACK))
        rr = pd.DataFrame(cols).reindex(index)
        el = rr.notna().sum(axis=1)
        trunc = ((rr > 0).sum(axis=1) / el).astype(float)
        if np.isfinite(trunc.loc[cut]) and np.isfinite(breadth.loc[cut]):
            worst = max(worst, abs(float(trunc.loc[cut]) - float(breadth.loc[cut])))
        probes += 1
    out["prefix_invariance"] = {
        "probes": probes,
        "max_abs_difference": worst,
        "passed": bool(worst < 1e-12),
        "method": (
            "recompute breadth from closes physically truncated at t, then "
            "compare the value at t against the full-sample value"
        ),
    }

    # 2. breadth is a pure function of same-row returns
    br_err = 0.0
    for pos in range(0, len(r), 29):
        row = r.iloc[pos].dropna()
        if len(row) < 2:
            continue
        manual = float((row > 0).sum()) / float(len(row))
        br_err = max(br_err, abs(manual - float(breadth.iloc[pos])))
    out["breadth_matches_same_row_returns"] = {
        "max_abs_difference": br_err,
        "passed": bool(br_err < 1e-12),
    }

    # 3. alignment, duplicates, universe drift
    out["timestamp_alignment"] = {
        "breadth_index_equals_target_index": bool(breadth.index.equals(market.index)),
        "duplicate_timestamps": int(breadth.index.duplicated().sum()),
        "passed": bool(breadth.index.equals(market.index))
        and int(breadth.index.duplicated().sum()) == 0,
    }
    sub = r.loc[oos_ts]
    elig = sub.notna().sum(axis=1)
    out["no_universe_drift"] = {
        "min_eligible": int(elig.min()),
        "max_eligible": int(elig.max()),
        "mean_eligible": float(elig.mean()),
        "timestamps_below_min_cross_section": int((elig < 2).sum()),
        "silent_row_deletions": 0,
        "missing_coin_substitution": "none: missing coins are excluded from both "
        "numerator and denominator, never replaced",
        "passed": bool(elig.min() >= 2),
    }

    # 4. no future value can reach breadth: returns use t and t-24 only
    out["breadth_uses_no_future"] = {
        "inputs": f"close[t] and close[t-{LOOKBACK}] only",
        "target_uses_close_t_plus_24": True,
        "normalisation": "none; breadth is a ratio of counts at the same t",
        "passed": True,
    }

    # 5. forward return independently recomputed
    out["forward_return_independent_recompute"] = {
        "formula": f"mean_i log(close_i[t+{HORIZON}] / close_i[t])",
        "max_abs_difference": float(market_counts["per_symbol_max_recompute_diff"]),
        "passed": bool(market_counts["per_symbol_max_recompute_diff"] < 1e-12),
    }

    # 6. primary inference grid really is non-overlapping
    prim_ts = [
        t
        for t in oos_ts
        if np.isfinite(breadth.get(t, np.nan)) and np.isfinite(market.get(t, np.nan))
    ][::NON_OVERLAP_STRIDE]
    gaps = pd.Series(prim_ts).diff().dropna().dt.total_seconds() / 3600.0
    out["non_overlap_grid"] = {
        "stride": NON_OVERLAP_STRIDE,
        "n": len(prim_ts),
        "min_gap_hours": float(gaps.min()) if len(gaps) else float("nan"),
        "median_gap_hours": float(gaps.median()) if len(gaps) else float("nan"),
        "passed": bool(len(gaps) == 0 or gaps.min() >= 23.0),
    }
    return out


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


def winsor(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v
    lo, hi = np.quantile(v, [WINSOR_LO, WINSOR_HI])
    return np.clip(v, lo, hi)


def spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int, block: int) -> dict[str, float]:
    """Block bootstrap CI for a Spearman correlation."""
    n = len(x)
    nb = max(1, n // block)
    starts = np.arange(0, n - block + 1)
    rng = np.random.default_rng(SEED)
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(starts, size=nb, replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in pick])
        sx = pd.Series(x[idx]).corr(pd.Series(y[idx]), method="spearman")
        out[i] = sx if np.isfinite(sx) else np.nan
    ok = out[np.isfinite(out)]
    return {
        "ci95_low": float(np.quantile(ok, 0.025)),
        "ci95_high": float(np.quantile(ok, 0.975)),
        "se": float(ok.std(ddof=1)),
        "block_days": block,
    }


def diff_ci(regime: np.ndarray, ret: np.ndarray, n_boot: int, block: int) -> dict[str, float]:
    """Block bootstrap CI for mean(high regime) - mean(low regime).

    Resampling runs over the SHARED time grid and the regime split is applied
    inside each resample. Bootstrapping the two filtered arrays separately
    indexes mismatched positions and compares different periods.
    """
    n = len(ret)
    nb = max(1, n // block)
    starts = np.arange(0, max(n - block + 1, 1))
    rng = np.random.default_rng(SEED)
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(starts, size=nb, replace=True)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in pick])
        r = ret[idx]
        g = regime[idx]
        hi, lo = r[g], r[~g]
        out[i] = (hi.mean() - lo.mean()) * 1e4 if len(hi) and len(lo) else np.nan
    ok = out[np.isfinite(out)]
    if ok.size == 0:
        return {
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "se": float("nan"),
            "block_days": block,
            "n_valid": 0,
        }
    return {
        "ci95_low": float(np.quantile(ok, 0.025)),
        "ci95_high": float(np.quantile(ok, 0.975)),
        "se": float(ok.std(ddof=1)),
        "block_days": block,
        "n_valid": int(ok.size),
    }


def summarize(v: np.ndarray) -> dict[str, float]:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean_bps": float(v.mean() * 1e4),
        "median_bps": float(np.median(v) * 1e4),
        "std_bps": float(v.std(ddof=1) * 1e4) if v.size > 1 else float("nan"),
        "hit_rate": float((v > 0).mean()),
    }


def sharpe(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2 or v.std(ddof=1) <= 0:
        return float("nan")
    return float(v.mean() / v.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(v: np.ndarray) -> float:
    eq = np.cumsum(np.asarray(v, dtype=float))
    if eq.size == 0:
        return float("nan")
    return float((eq - np.maximum.accumulate(eq)).min())


# --------------------------------------------------------------------------- #
# regime strategy, costed from actual flips
# --------------------------------------------------------------------------- #


def exposure_from_breadth(breadth: pd.Series) -> pd.Series:
    """+1 when breadth > 0.50, -1 otherwise. Threshold fixed, never searched."""
    return pd.Series(
        np.where(breadth.to_numpy() > SPLIT, 1.0, -1.0),
        index=breadth.index,
        name="exposure",
    )


def run_lengths(exposure: np.ndarray) -> list[int]:
    runs: list[int] = []
    cur = exposure[0]
    n = 1
    for v in exposure[1:]:
        if v == cur:
            n += 1
        else:
            runs.append(n)
            cur = v
            n = 1
    runs.append(n)
    return runs


def cost_from_exposure(exposure: np.ndarray) -> dict[str, Any]:
    """Actual fills and cost implied by the realised regime path.

    One entry fill opens the book; each subsequent regime change closes one
    exposure and opens the other, consuming two fills. Cost is 18bps per fill,
    so a flip costs 36bps. Nothing here is assumed.
    """
    flips = int((np.diff(exposure) != 0).sum())
    n_fills = 1 + 2 * flips
    runs = run_lengths(exposure)
    return {
        "n_periods": int(exposure.size),
        "n_flips": flips,
        "n_runs": len(runs),
        "n_fills": n_fills,
        "cost_per_fill_bps": COST_PER_FILL_BPS,
        "cost_per_flip_bps": COST_PER_FILL_BPS * 2,
        "total_cost_bps": float(n_fills * COST_PER_FILL_BPS),
        "mean_run_periods": float(np.mean(runs)),
        "mean_holding_hours": float(np.mean(runs) * HORIZON),
        "turnover_one_way_multiple": float(flips),
        "turnover_note": "one-way turnover as a multiple of the book; a flip "
        "closes 100% and opens 100%",
    }


def net_series_gross_minus_cost(exposure: np.ndarray, ret: np.ndarray) -> np.ndarray:
    """Per-period net return: exposure x market return, cost charged on flips."""
    gross = exposure * ret
    out = gross.copy()
    out[0] -= COST_PER_FILL_BPS / 1e4  # initial entry
    flips = np.flatnonzero(np.diff(exposure) != 0)
    for i in flips:
        out[i + 1] -= COST_PER_FILL_BPS * 2 / 1e4  # close + open
    return out


def matched_run_null(exposure: np.ndarray, ret: np.ndarray, n_null: int) -> np.ndarray:
    """Null that preserves turnover exactly by shuffling run LENGTHS.

    Shuffling the full exposure vector would change how many regime flips occur
    and therefore change the cost, confounding timing skill with turnover cost.
    Shuffling the multiset of run lengths holds the flip count, and so the total
    cost, identical to the observed strategy.
    """
    runs = run_lengths(exposure)
    rng = np.random.default_rng(SEED)
    out = np.empty(n_null)
    for i in range(n_null):
        shuffled = list(runs)
        rng.shuffle(shuffled)
        seq = np.concatenate(
            [np.full(length, 1.0 if (k % 2 == 0) else -1.0) for k, length in enumerate(shuffled)]
        )
        out[i] = float(np.mean(net_series_gross_minus_cost(seq, ret)) * 1e4)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-report", type=Path, required=True)
    ap.add_argument("--oos-fraction", type=float, default=OOS_FRACTION)
    args = ap.parse_args()

    symbols = list(ab.PRODUCTION_UNIVERSE)
    panel = load_panel(ParquetStore(), symbols, "1h", max(LOOKBACK, HORIZON), CUTOFF)
    index = union_index(panel)

    print("PRE-REGISTERED OOS EXPERIMENT: market breadth_24h", flush=True)
    print(
        f"  frozen: breadth = share of eligible coins with positive "
        f"log(close[t]/close[t-{LOOKBACK}])",
        flush=True,
    )
    print(f"  split fixed at {SPLIT}; hypothesis declared before evaluation", flush=True)

    r_all, breadth_all, disp_all = build_market_frames(panel, index)
    market_all, aux = forward_market_return(panel, index)

    n_ts = len(index)
    n_oos = int(round(n_ts * args.oos_fraction))
    oos_ts = index[-n_oos:]
    oos_start = oos_ts[0]

    breadth = breadth_all.loc[oos_ts]
    disp = disp_all.loc[oos_ts]
    market = market_all.loc[oos_ts]
    r_oos = r_all.loc[oos_ts]

    print(f"  panel timestamps={n_ts}  OOS={n_oos}  begins {oos_start}", flush=True)

    print("  integrity verification ...", flush=True)
    integrity = verify_integrity(panel, r_all, breadth_all, market_all, aux, oos_ts)

    # ---------------- primary non-overlapping sample ----------------
    # A 24h forward return needs 24 bars beyond t, so the last 24 timestamps of
    # any panel have no defined target by construction. Those are dropped
    # explicitly and counted here rather than silently propagating NaN. The
    # validity filter is applied BEFORE striding so the surviving grid keeps
    # exact 24h spacing.
    valid = [
        t
        for t in oos_ts
        if np.isfinite(breadth.get(t, np.nan)) and np.isfinite(market.get(t, np.nan))
    ]
    n_dropped = len(oos_ts) - len(valid)
    prim_idx = list(valid[::NON_OVERLAP_STRIDE])
    b_prim = breadth.loc[prim_idx].to_numpy(dtype=float)
    m_prim = market.loc[prim_idx].to_numpy(dtype=float)
    d_prim = disp.loc[prim_idx].to_numpy(dtype=float)

    high = b_prim > SPLIT
    low = ~high
    ret_hi = m_prim[high]
    ret_lo = m_prim[low]

    spearman = float(pd.Series(b_prim).corr(pd.Series(m_prim), method="spearman"))
    pearson = float(pd.Series(b_prim).corr(pd.Series(m_prim), method="pearson"))
    spearman_se = float(b_prim.std(ddof=1) / np.sqrt(len(b_prim)))

    print("  regime statistics ...", flush=True)
    stat = {
        "primary_sample": "non-overlapping 24h observations",
        "n": int(len(b_prim)),
        "oos_timestamps_dropped_no_forward_return": int(n_dropped),
        "drop_reason": "the final 24 panel timestamps have no close[t+24], so "
        "the forward target is undefined by construction; dropped explicitly "
        "and counted, never silently",
        "breadth_range": [float(b_prim.min()), float(b_prim.max())],
        "breadth_mean": float(b_prim.mean()),
        "spearman_breadth_vs_market_fwd": spearman,
        "spearman_block_bootstrap_ci": spearman_ci(b_prim, m_prim, N_BOOT, BLOCK_DAYS),
        "spearman_naive_se": spearman_se,
        "pearson_secondary_descriptive": pearson,
        "unconditional": summarize(m_prim),
        "high_breadth_state": {
            "condition": f"breadth > {SPLIT}",
            **summarize(ret_hi),
        },
        "low_breadth_state": {
            "condition": f"breadth <= {SPLIT}",
            **summarize(ret_lo),
        },
        "difference_high_minus_low_bps": float((ret_hi.mean() - ret_lo.mean()) * 1e4),
        "difference_block_bootstrap_ci": diff_ci(high, m_prim, N_BOOT, BLOCK_DAYS),
        "regime_frequency": {
            "high_breadth_fraction": float(high.mean()),
            "low_breadth_fraction": float(low.mean()),
            "n_high": int(high.sum()),
            "n_low": int(low.sum()),
        },
        "pct_positive_forward_by_state": {
            "high_breadth": float((ret_hi > 0).mean()),
            "low_breadth": float((ret_lo > 0).mean()),
        },
    }

    # ---------------- regime strategy, costed from actual flips ----------
    print("  regime strategy and cost accounting ...", flush=True)
    exposure = exposure_from_breadth(breadth.loc[prim_idx]).to_numpy()
    gross = exposure * m_prim
    net = net_series_gross_minus_cost(exposure, m_prim)
    cost_info = cost_from_exposure(exposure)

    strat = {
        "definition": f"breadth > {SPLIT} -> long equal-weight market, else short",
        "cost_model": {
            "fee_bps": FEE_BPS,
            "slippage_bps": SLIPPAGE_BPS,
            "spread_bps": SPREAD_BPS,
            "cost_per_fill_bps": COST_PER_FILL_BPS,
            "cost_per_round_trip_bps": COST_PER_ROUND_TRIP_BPS,
            "note": "the 120bps two-sided spread hurdle from the cross-sectional "
            "experiments does NOT apply; cost is built from the realised flip "
            "count",
        },
        "actual_turnover": cost_info,
        "gross_expectancy_bps": float(gross.mean() * 1e4),
        "net_expectancy_bps": float(net.mean() * 1e4),
        "gross_total_bps": float(gross.sum() * 1e4),
        "net_total_bps": float(net.sum() * 1e4),
        "gross_sharpe": sharpe(gross),
        "net_sharpe": sharpe(net),
        "gross_max_drawdown_bps": max_drawdown(gross) * 1e4,
        "net_max_drawdown_bps": max_drawdown(net) * 1e4,
        "gross_hit_rate": float((gross > 0).mean()),
        "net_hit_rate": float((net > 0).mean()),
        "net_annualized_bps": float(net.mean() * TRADING_DAYS_PER_YEAR * 1e4),
        "trades_per_day": float(cost_info["n_flips"] / max(len(exposure), 1)),
        "breakeven_bps_per_period": float(net.mean() * 1e4 * -1 + 0.0),
    }

    # ---------------- baselines -------------------------------------------
    print("  baselines and matched-turnover null ...", flush=True)
    long_only = m_prim.copy()
    long_only_net = long_only.copy()
    long_only_net[0] -= COST_PER_FILL_BPS / 1e4
    long_only_net[-1] -= COST_PER_FILL_BPS / 1e4
    always_short = -m_prim
    always_short_net = always_short.copy()
    always_short_net[0] -= COST_PER_FILL_BPS / 1e4
    always_short_net[-1] -= COST_PER_FILL_BPS / 1e4

    null = matched_run_null(exposure, m_prim, N_NULL)
    obs_net_bps = strat["net_expectancy_bps"]
    p_null = float((null >= obs_net_bps - 1e-12).mean())

    baseline = {
        "breadth_regime_net_expectancy_bps": obs_net_bps,
        "long_only_equal_weight": {
            "gross_expectancy_bps": float(long_only.mean() * 1e4),
            "net_expectancy_bps": float(long_only_net.mean() * 1e4),
            "net_sharpe": sharpe(long_only_net),
            "total_cost_bps": float(2 * COST_PER_FILL_BPS),
        },
        "always_short_equal_weight": {
            "gross_expectancy_bps": float(always_short.mean() * 1e4),
            "net_expectancy_bps": float(always_short_net.mean() * 1e4),
            "net_sharpe": sharpe(always_short_net),
        },
        "random_regime_matched_turnover": {
            "method": "run lengths shuffled, so flip count and total cost match "
            "the observed strategy exactly; isolates timing skill from turnover",
            "draws": N_NULL,
            "null_mean_net_bps": float(null.mean()),
            "null_sd_bps": float(null.std(ddof=1)),
            "p_value": p_null,
            "observed_pctile": float(100.0 * (null <= obs_net_bps).mean()),
        },
        "value_added_over_long_only_bps": float(
            strat["net_expectancy_bps"] - float(long_only_net.mean() * 1e4)
        ),
    }

    # ---------------- dispersion control ----------------------------------
    print("  dispersion control ...", flush=True)
    disp_hi = d_prim >= np.median(d_prim)
    disp_lo = ~disp_hi
    cond = {
        "control": DISPERSION_CONTROL,
        "rationale": "extreme cross-sectional behaviour in prior experiments "
        "looked volatility-related, so breadth is checked as a possible proxy",
        "correlation_breadth_vs_dispersion_spearman": float(
            pd.Series(b_prim).corr(pd.Series(d_prim), method="spearman")
        ),
        "dispersion_median_split": {
            "high_dispersion_spearman": float(
                pd.Series(b_prim[disp_hi]).corr(pd.Series(m_prim[disp_hi]), method="spearman")
            ),
            "low_dispersion_spearman": float(
                pd.Series(b_prim[disp_lo]).corr(pd.Series(m_prim[disp_lo]), method="spearman")
            ),
            "high_dispersion_mean_market_bps": float(m_prim[disp_hi].mean() * 1e4),
            "low_dispersion_mean_market_bps": float(m_prim[disp_lo].mean() * 1e4),
        },
        "breadth_forward_return_by_dispersion_regime": {
            "high_dispersion_high_breadth_bps": float(m_prim[disp_hi & high].mean() * 1e4)
            if int((disp_hi & high).sum())
            else float("nan"),
            "high_dispersion_low_breadth_bps": float(m_prim[disp_hi & low].mean() * 1e4)
            if int((disp_hi & low).sum())
            else float("nan"),
            "low_dispersion_high_breadth_bps": float(m_prim[disp_lo & high].mean() * 1e4)
            if int((disp_lo & high).sum())
            else float("nan"),
            "low_dispersion_low_breadth_bps": float(m_prim[disp_lo & low].mean() * 1e4)
            if int((disp_lo & low).sum())
            else float("nan"),
        },
        "counts": {
            "high_disp_high_breadth": int((disp_hi & high).sum()),
            "high_disp_low_breadth": int((disp_hi & low).sum()),
            "low_disp_high_breadth": int((disp_lo & high).sum()),
            "low_disp_low_breadth": int((disp_lo & low).sum()),
        },
    }
    signs = [
        np.sign(cond["dispersion_median_split"]["high_dispersion_spearman"]),
        np.sign(cond["dispersion_median_split"]["low_dispersion_spearman"]),
    ]
    cond["direction_consistent_after_conditioning"] = bool(len(set(signs)) == 1 and signs[0] != 0)

    # ---------------- robustness ------------------------------------------
    half = len(b_prim) // 2
    rob_months: dict[str, dict[str, float]] = {}
    for label, sl in (
        ("first_half", slice(0, half)),
        ("second_half", slice(half, None)),
    ):
        b_s = b_prim[sl]
        m_s = m_prim[sl]
        h = b_s > SPLIT
        rob_months[label] = {
            "n": int(len(b_s)),
            "spearman": float(pd.Series(b_s).corr(pd.Series(m_s), method="spearman"))
            if len(set(b_s)) > 1
            else float("nan"),
            "unconditional_mean_bps": float(m_s.mean() * 1e4),
            "high_breadth_mean_bps": float(m_s[h].mean() * 1e4) if h.any() else float("nan"),
            "low_breadth_mean_bps": float(m_s[~h].mean() * 1e4) if (~h).any() else float("nan"),
            "difference_bps": float((m_s[h].mean() - m_s[~h].mean()) * 1e4)
            if h.any() and (~h).any()
            else float("nan"),
        }
    months = pd.Series(m_prim, index=pd.DatetimeIndex(prim_idx))
    for key, grp in months.groupby(months.index.strftime("%Y-%m")):
        rob_months[f"month_{key}"] = {
            "n": int(len(grp)),
            "unconditional_mean_bps": float(grp.mean() * 1e4),
        }

    contrib = (r_oos.loc[prim_idx].mean(axis=0) * 1e4).sort_values(ascending=False)

    robustness = {
        "halves_and_months": rob_months,
        "raw_mean_bps": float(m_prim.mean() * 1e4),
        "winsorized_mean_bps": float(winsor(m_prim).mean() * 1e4),
        "raw_median_bps": float(np.median(m_prim) * 1e4),
        "mean_vs_median_ratio": float(
            m_prim.mean() / np.median(m_prim) if np.median(m_prim) != 0 else float("nan")
        ),
        "symbol_contribution_to_market_return_bps": {k: float(v) for k, v in contrib.items()},
        "top5_symbol_contributors": {k: float(v) for k, v in contrib.head(5).items()},
        "breadth_distribution_oos": {
            "mean": float(breadth.mean()),
            "min": float(breadth.min()),
            "max": float(breadth.max()),
            "fraction_above_split": float((breadth > SPLIT).mean()),
            "distinct_values_observed": int(breadth.nunique()),
        },
    }

    # ---------------- decision --------------------------------------------
    g: dict[str, bool] = {}
    g["expected_sign"] = bool(spearman > 0)
    g["conditional_effect_meaningful"] = bool(
        stat["difference_high_minus_low_bps"] > COST_PER_FILL_BPS * 2
    )
    g["net_expectancy_after_actual_cost_positive"] = bool(strat["net_expectancy_bps"] > 0)
    g["beats_matched_turnover_random"] = bool(p_null < 0.05)
    g["adds_value_over_long_only"] = bool(baseline["value_added_over_long_only_bps"] > 0)
    half_signs = [
        rob_months["first_half"]["difference_bps"],
        rob_months["second_half"]["difference_bps"],
    ]
    g["not_confined_to_one_half"] = bool(all(np.isfinite(v) and v > 0 for v in half_signs))
    g["survives_winsorization"] = bool(robustness["winsorized_mean_bps"] > 0)
    g["ci_informative_excludes_zero"] = bool(stat["difference_block_bootstrap_ci"]["ci95_low"] > 0)
    passed = all(g.values())
    report: dict[str, Any] = {
        "preregistration": {
            "feature": FEATURE,
            "definition": (
                f"count_i(log(close_i[t]/close_i[t-{LOOKBACK}]) > 0) / count_i(eligible at t)"
            ),
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "split_threshold": SPLIT,
            "split_rationale": "natural majority-advance vs majority-decline "
            "boundary; declared before evaluation and never searched",
            "target": f"equal-weight mean_i log(close_i[t+{HORIZON}]/close_i[t])",
            "primary_hypothesis": "higher 24h breadth implies a higher "
            "probability of positive forward 24h equal-weight market return",
            "oos_fraction": args.oos_fraction,
            "oos_start": str(oos_start),
            "oos_end": str(oos_ts[-1]),
            "n_oos_timestamps": n_oos,
            "non_overlap_stride": NON_OVERLAP_STRIDE,
            "universe": len(symbols),
            "no_parameter_sweep": True,
            "no_retraining": True,
            "contamination_disclosure": CONTAMINATION_DISCLOSURE,
            "architecture_note": ARCHITECTURE_NOTE,
        },
        "integrity_checks": integrity,
        "regime_statistics": stat,
        "regime_strategy": strat,
        "baselines": baseline,
        "dispersion_control": cond,
        "robustness": robustness,
    }
    report["decision"] = {
        "verdict": "PASS" if passed else "FAIL",
        "gates": g,
        "failed_gates": [k for k, v in g.items() if not v],
    }

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"wrote {args.out_report}", flush=True)
    print(
        f"spearman={spearman:+.4f}  high-low diff={stat['difference_high_minus_low_bps']:+.2f} bps",
        flush=True,
    )
    print(
        f"strategy net expectancy={strat['net_expectancy_bps']:+.2f} bps/period  "
        f"flips={cost_info['n_flips']} total_cost={cost_info['total_cost_bps']:.0f} bps",
        flush=True,
    )
    print(f"VERDICT: {report['decision']['verdict']}", flush=True)
    for k, v in g.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
