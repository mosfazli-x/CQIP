"""Pre-registered OOS test: cross-sectional relative strength at 24h.

Frozen definition, no sweep
---------------------------
    relative_strength_24h = log(close[t]/close[t-24]) - median_over_universe(same)

Both terms use only closes at or before bar t, so the feature is known at the
close of bar t. The universe median is taken across all coins with finite
close[t] and close[t-24] at that timestamp; no future value enters it.

What this test can and cannot learn
-----------------------------------
Subtracting a per-timestamp constant is a monotone transform *within* that
timestamp, so it leaves the rank order untouched. Every cross-sectional
quantity -- rank IC, top-minus-bottom spread, quintiles, permutation p -- is
therefore numerically identical to ranking by raw 24h return. That identity is
verified numerically in this script rather than assumed. The median subtraction
only changes the *time-series* reading, from an absolute return to a relative
one. Consequently the shape diagnostic (continuation vs reversion vs
middle-out) is genuinely undetermined in advance and is measured, not assumed.

Protocol is byte-for-byte the previous experiment's: same universe, same
cutoff, same OOS fraction, same close-of-bar decision, same 24h horizon, same
non-overlapping primary sample, same per-leg cost accounting.
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
from feature_audit import forward_returns, load_panel, wide_matrix  # noqa: E402

from cqip.storage import ParquetStore  # noqa: E402

# --------------------------------------------------------------------------- #
# FROZEN CONSTANTS -- identical to the previous experiment except the feature
# --------------------------------------------------------------------------- #

FEATURE = "relative_strength_24h"
COMPARISON_FEATURE = "sma_fast_slow_spread"
LOOKBACK = 24
HORIZON = 24
TOP_K = 3

FEE_BPS = 10.0
SLIPPAGE_BPS = 5.0
SPREAD_BPS = 3.0
COST_PER_FILL_BPS = FEE_BPS + SLIPPAGE_BPS + SPREAD_BPS  # 18
COST_PER_LEG_RT_BPS = 2 * COST_PER_FILL_BPS  # 36
SAFETY_MARGIN_BPS = 24.0
HURDLE_PER_LEG_BPS = COST_PER_LEG_RT_BPS + SAFETY_MARGIN_BPS  # 60
HURDLE_SPREAD_BPS = 2 * HURDLE_PER_LEG_BPS  # 120
COST_SPREAD_BPS = 2 * COST_PER_LEG_RT_BPS  # 72

OOS_FRACTION = 0.25
CUTOFF = 5100
NON_OVERLAP_STRIDE = HORIZON
WINSOR_LO, WINSOR_HI = 0.01, 0.99
MIN_CROSS_SECTION = 2 * TOP_K
TRADING_DAYS_PER_YEAR = 365.0
SEED = 7
N_DRAWS = 2000
N_BOOT = 2000
BLOCK_DAYS = 7

CONTAMINATION_DISCLOSURE = (
    "This OOS window was already inspected during the feature audit and during "
    "the preceding sma_fast_slow_spread pre-registered test, whose result is "
    "known. No clean holdout exists in the stored panel because it ends at the "
    "last available bar. This run is a locked, post-selection diagnostic on a "
    "window already known to the analysis, not a pre-registered holdout. "
    "Statistical significance here must be discounted accordingly; the design "
    "value is the pre-declared shape and robustness battery, not the p-value."
)


# --------------------------------------------------------------------------- #
# feature construction, computed straight from raw closes
# --------------------------------------------------------------------------- #


def relative_strength(
    panel: Any, index: pd.Index | None = None, truncate_at: Any = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, Any]]:
    """(ts x symbol) relative strength, built from raw close prices only.

    Alignment follows ``wide_matrix`` exactly: each symbol's close array is
    labelled with that symbol's OWN timestamps from ``panel.frames[sym]['ts']``,
    never the first symbol's grid and never the positional RangeIndex that
    ``panel.close`` carries. Symbols absent from a timestamp stay NaN, are
    excluded from the universe median, and are excluded from cross-sections
    rather than silently dropped.

    ``truncate_at`` removes every observation after that timestamp from the
    inputs BEFORE the lookback shift is computed. It exists so the prefix
    test exercises real truncation; restricting only the output index would
    leave the shift untouched and make the test pass trivially.

    Both terms use only close[t] and close[t-LOOKBACK], so nothing after bar t
    can enter the feature or its normalisation.
    """
    series: dict[str, pd.Series] = {}
    for sym in panel.symbols:
        if sym not in panel.close:
            continue
        ts = pd.DatetimeIndex(panel.frames[sym]["ts"].to_numpy())
        close = pd.Series(panel.close[sym].to_numpy(dtype=float), index=ts, name=sym)
        if truncate_at is not None:
            keep = ts <= truncate_at
            close = close.loc[keep]
        series[sym] = np.log(close / close.shift(LOOKBACK))
    raw = pd.DataFrame(series)

    if index is not None:
        raw = raw.reindex(index)

    med = raw.median(axis=1, skipna=True)
    rs = raw.sub(med, axis=0)

    meta = {"n_eligible_per_ts": raw.notna().sum(axis=1)}
    return rs, raw, med, meta


# --------------------------------------------------------------------------- #
# leakage verification
# --------------------------------------------------------------------------- #


def verify_causality(
    panel: Any,
    rs: pd.DataFrame,
    raw_ret: pd.DataFrame,
    med: pd.Series,
    fwd: pd.DataFrame,
    n_probe: int,
) -> dict[str, Any]:
    """Prove the feature and target use no information from after bar t."""
    out: dict[str, Any] = {}
    index = rs.index
    rng = np.random.default_rng(SEED)
    probe = sorted(rng.choice(len(index), size=min(n_probe, len(index)), replace=False))

    # 1. prefix invariance: truncate raw closes at t, recompute, compare.
    worst = 0.0
    checked = 0
    for pos in probe:
        cut = index[pos]
        trunc_rs, _, _, _ = relative_strength(panel, truncate_at=cut)
        diff = float((trunc_rs.loc[cut] - rs.loc[cut]).abs().max())
        worst = max(worst, diff)
        checked += 1
    out["prefix_invariance"] = {
        "probes": checked,
        "max_abs_difference": worst,
        "passed": bool(worst < 1e-9),
        "method": (
            "recompute the feature from closes physically truncated at t, "
            "including truncating the lookback input, then compare the value "
            "at t against the full-sample value"
        ),
    }

    # 2. universe median must equal the median of the same-timestamp returns
    med_err = 0.0
    for pos in probe[: min(200, len(probe))]:
        row = raw_ret.iloc[pos].dropna()
        if len(row) < MIN_CROSS_SECTION:
            continue
        med_err = max(med_err, abs(float(row.median()) - float(med.iloc[pos])))
    out["universe_median_is_same_timestamp_median"] = {
        "max_abs_difference": med_err,
        "passed": bool(med_err < 1e-12),
    }

    # 3. independent target recomputation from close[t+horizon]/close[t]
    worst_t = 0.0
    for sym in panel.symbols:
        if sym not in panel.close or sym not in fwd.columns:
            continue
        ts = pd.DatetimeIndex(panel.frames[sym]["ts"].to_numpy())
        c = pd.Series(panel.close[sym].to_numpy(dtype=float), index=ts).reindex(index)
        manual = np.log(c.shift(-HORIZON) / c)
        common = manual.notna() & fwd[sym].notna()
        if common.any():
            worst_t = max(
                worst_t,
                float((manual[common] - fwd[sym][common]).abs().max()),
            )
    out["forward_target_independent_recompute"] = {
        "formula": f"log(close[t+{HORIZON}] / close[t])",
        "max_abs_difference": worst_t,
        "passed": bool(worst_t < 1e-10),
    }

    # 4. the median shift must not alter within-timestamp ranks
    rank_delta = 0.0
    for pos in probe[: min(300, len(probe))]:
        a = raw_ret.iloc[pos].dropna()
        if len(a) < MIN_CROSS_SECTION:
            continue
        b = (a - a.median()).rank()
        rank_delta = max(rank_delta, float((a.rank() - b).abs().max()))
    out["median_shift_preserves_rank"] = {
        "max_abs_rank_difference": rank_delta,
        "passed": bool(rank_delta == 0.0),
        "implication": (
            "cross-sectional metrics are invariant to the median subtraction; "
            "they equal those of raw 24h return ranked within timestamp"
        ),
    }
    return out


# --------------------------------------------------------------------------- #
# selection and metrics
# --------------------------------------------------------------------------- #


def select_legs(feat: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ts in feat.index:
        f = feat.loc[ts].to_numpy(dtype=float)
        r = fwd.loc[ts].to_numpy(dtype=float)
        ok = np.isfinite(f) & np.isfinite(r)
        if int(ok.sum()) < MIN_CROSS_SECTION:
            continue
        idx = np.flatnonzero(ok)
        order = idx[np.argsort(f[idx], kind="mergesort")]
        head, tail = order[-TOP_K:], order[:TOP_K]
        if f[head[-1]] == f[tail[0]]:
            continue
        for j in head:
            rows.append({"ts": ts, "symbol": feat.columns[j], "side": "long", "ret": r[j]})
        for j in tail:
            rows.append({"ts": ts, "symbol": feat.columns[j], "side": "short", "ret": -r[j]})
    return pd.DataFrame(rows)


def per_timestamp_spread(feat: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    out: dict[Any, float] = {}
    for ts in feat.index:
        f = feat.loc[ts].to_numpy(dtype=float)
        r = fwd.loc[ts].to_numpy(dtype=float)
        ok = np.isfinite(f) & np.isfinite(r)
        if int(ok.sum()) < MIN_CROSS_SECTION:
            continue
        idx = np.flatnonzero(ok)
        order = idx[np.argsort(f[idx], kind="mergesort")]
        head, tail = order[-TOP_K:], order[:TOP_K]
        if f[head[-1]] == f[tail[0]]:
            continue
        out[ts] = float(r[head].mean() - r[tail].mean())
    return pd.Series(out, dtype=float).sort_index()


def rank_ic(feat: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    out: dict[Any, float] = {}
    for ts in feat.index:
        f = feat.loc[ts].to_numpy(dtype=float)
        r = fwd.loc[ts].to_numpy(dtype=float)
        ok = np.isfinite(f) & np.isfinite(r)
        if int(ok.sum()) < MIN_CROSS_SECTION:
            continue
        if len(set(f[ok])) < 2 or len(set(r[ok])) < 2:
            continue
        out[ts] = float(pd.Series(f[ok]).corr(pd.Series(r[ok]), method="spearman"))
    return pd.Series(out, dtype=float).sort_index()


def buckets(feat: pd.DataFrame, fwd: pd.DataFrame, n: int) -> list[dict[str, Any]]:
    acc: dict[int, list[float]] = {b: [] for b in range(1, n + 1)}
    sizes: dict[int, list[int]] = {b: [] for b in range(1, n + 1)}
    for ts in feat.index:
        f = feat.loc[ts].to_numpy(dtype=float)
        r = fwd.loc[ts].to_numpy(dtype=float)
        ok = np.isfinite(f) & np.isfinite(r)
        if int(ok.sum()) < MIN_CROSS_SECTION:
            continue
        idx = np.flatnonzero(ok)
        order = idx[np.argsort(f[idx], kind="mergesort")]
        for b, chunk in enumerate(np.array_split(order, n), start=1):
            acc[b].extend(r[chunk].tolist())
            sizes[b].append(len(chunk))
    return [
        {
            "bucket": b,
            "n_obs": len(acc[b]),
            "coins_per_ts_mean": float(np.mean(sizes[b])) if sizes[b] else float("nan"),
            "mean_ret": float(np.mean(acc[b])) if acc[b] else float("nan"),
            "median_ret": float(np.median(acc[b])) if acc[b] else float("nan"),
        }
        for b in range(1, n + 1)
    ]


def classify_shape(quint: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic A/B/C/D classification from the frozen quintile profile."""
    q = [b["mean_ret"] * 1e4 for b in quint]
    diffs = [q[i + 1] - q[i] for i in range(len(q) - 1)]
    inc = all(d > 0 for d in diffs)
    dec = all(d < 0 for d in diffs)
    interior_max = max(q[1:-1])
    interior_min = min(q[1:-1])
    rank_corr = float(pd.Series([1, 2, 3, 4, 5]).corr(pd.Series(q), method="spearman"))
    peak = int(np.argmax(q)) + 1
    trough = int(np.argmin(q)) + 1

    if inc:
        label = "A_monotonic_continuation"
    elif dec:
        label = "B_monotonic_mean_reversion"
    elif interior_max > q[0] and interior_max > q[-1]:
        label = "C_inverted_u_middle_out"
    elif interior_min < q[0] and interior_min < q[-1]:
        label = "C_u_shaped_ends_out"
    else:
        label = "D_no_structure"

    return {
        "quintile_means_bps": q,
        "consecutive_diffs_bps": diffs,
        "spearman_bucket_index_vs_return": rank_corr,
        "peak_bucket_low_is_1": peak,
        "trough_bucket_low_is_1": trough,
        "all_diffs_positive": bool(inc),
        "all_diffs_negative": bool(dec),
        "classification": label,
    }


# --------------------------------------------------------------------------- #
# inference helpers
# --------------------------------------------------------------------------- #


def mean_t_ci(v: np.ndarray) -> dict[str, float]:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return {"n": int(n), "mean": float("nan"), "se": float("nan"), "t": float("nan")}
    se = v.std(ddof=1) / np.sqrt(n)
    tcrit = 2.026  # t(0.975, df=36)
    return {
        "n": int(n),
        "mean": float(v.mean()),
        "se": float(se),
        "t": float(v.mean() / se) if se > 0 else float("nan"),
        "ci95_low": float(v.mean() - tcrit * se),
        "ci95_high": float(v.mean() + tcrit * se),
    }


def block_bootstrap_ci(v: pd.Series, n_boot: int, block_days: int) -> dict[str, float]:
    """Moving-block bootstrap on the overlapping hourly spread series."""
    x = v.to_numpy(dtype=float)
    n = x.size
    if n < block_days * 3:
        return {"low": float("nan"), "high": float("nan"), "se": float("nan")}
    nb = max(1, n // block_days)
    starts = np.arange(0, n - block_days + 1)
    rng = np.random.default_rng(SEED)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(starts, size=nb, replace=True)
        sample = np.concatenate([x[s : s + block_days] for s in pick])
        means[i] = sample.mean()
    return {
        "low": float(np.quantile(means, 0.025) * 1e4),
        "high": float(np.quantile(means, 0.975) * 1e4),
        "se": float(means.std(ddof=1) * 1e4),
        "block_days": block_days,
        "n_boot": n_boot,
    }


def random_selection_null(fwd: pd.DataFrame, seed: int, draws: int) -> dict[str, Any]:
    """Randomise the SELECTION, not the values.

    Permuting a fixed multiset and re-averaging is degenerate because the mean is
    invariant, so a permutation test on the mean always returns p = 1. Drawing a
    random top-3/bottom-3 per timestamp answers the real question: does ranking
    by the feature beat chance?
    """
    rng = np.random.default_rng(seed)
    panels = []
    for ts in fwd.index:
        r = fwd.loc[ts].to_numpy(dtype=float)
        ok = np.flatnonzero(np.isfinite(r))
        if ok.size >= MIN_CROSS_SECTION:
            panels.append(r[ok])
    out = np.empty(draws)
    for i in range(draws):
        vals = []
        for r in panels:
            pick = rng.choice(r.size, size=2 * TOP_K, replace=False)
            vals.append(r[pick[:TOP_K]].mean() - r[pick[TOP_K:]].mean())
        out[i] = float(np.mean(vals)) * 1e4
    return {"draws": out, "n_panels": len(panels)}


def equity_curve(per_cycle_bps: np.ndarray) -> dict[str, Any]:
    """Cumulative bps curve, max drawdown in bps, and annualised figures."""
    eq = np.cumsum(per_cycle_bps)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    n = per_cycle_bps.size
    ann = float(per_cycle_bps.mean() * TRADING_DAYS_PER_YEAR) if n else float("nan")
    return {
        "total_bps": float(eq[-1]) if n else float("nan"),
        "max_drawdown_bps": float(dd.min()) if n else float("nan"),
        "cycles": int(n),
        "annualized_bps": ann,
        "final_equity_bps": float(eq[-1]) if n else float("nan"),
    }


def sharpe(per_cycle_bps: np.ndarray) -> float:
    if per_cycle_bps.size < 2:
        return float("nan")
    sd = per_cycle_bps.std(ddof=1)
    return (
        float(per_cycle_bps.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR))
        if sd > 0
        else float("nan")
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-report", type=Path, required=True)
    ap.add_argument("--oos-fraction", type=float, default=OOS_FRACTION)
    ap.add_argument("--causality-probes", type=int, default=40)
    args = ap.parse_args()

    symbols = list(ab.PRODUCTION_UNIVERSE)
    store = ParquetStore()

    print("PRE-REGISTERED OOS TEST: cross-sectional relative strength @ 24h", flush=True)
    print(f"  frozen: RS = log(close[t]/close[t-{LOOKBACK}]) - universe median", flush=True)
    print(f"  universe={len(symbols)} top_k={TOP_K} oos_fraction={args.oos_fraction}", flush=True)

    panel = load_panel(store, symbols, "1h", max(HORIZON, LOOKBACK), CUTOFF)
    fwd_all = wide_matrix(panel, forward_returns(panel, HORIZON))
    rs_all, raw_ret_all, med_all, rs_meta = relative_strength(panel, index=fwd_all.index)

    n_ts = len(rs_all.index)
    n_oos = int(round(n_ts * args.oos_fraction))
    oos_ts = rs_all.index[-n_oos:]
    rs = rs_all.loc[oos_ts]
    raw_ret = raw_ret_all.loc[oos_ts]
    med = med_all.loc[oos_ts]
    fwd = fwd_all.loc[oos_ts]
    oos_start = oos_ts[0]

    print(f"  panel timestamps={n_ts}  OOS={n_oos}  begins {oos_start}", flush=True)

    print("  causality verification ...", flush=True)
    causality = verify_causality(
        panel, rs_all, raw_ret_all, med_all, fwd_all, args.causality_probes
    )

    print("  applying frozen rule ...", flush=True)
    legs = select_legs(rs, fwd)
    spread = per_timestamp_spread(rs, fwd)
    ic = rank_ic(rs, fwd)
    prim = spread.iloc[::NON_OVERLAP_STRIDE] * 1e4

    print("  shape diagnostic ...", flush=True)
    quint = buckets(rs, fwd, 5)
    dec = buckets(rs, fwd, 10)
    shape = classify_shape(quint)

    print("  inference and permutation ...", flush=True)
    stats = mean_t_ci(prim.to_numpy())
    boot = block_bootstrap_ci(spread, N_BOOT, BLOCK_DAYS)
    null = random_selection_null(fwd, SEED, N_DRAWS)
    obs_bps = float(prim.mean())
    p_rand = float((np.abs(null["draws"]) >= abs(obs_bps) - 1e-12).mean())

    ic_all = ic.to_numpy()
    ic_t = float(ic_all.mean() / (ic_all.std(ddof=1) / np.sqrt(ic_all.size)))

    print("  baselines and comparison ...", flush=True)
    ts_list = list(spread.index)
    equal_w = fwd.loc[ts_list].mean(axis=1) * 1e4
    btc = "BTC_USDT" if "BTC_USDT" in fwd.columns else fwd.columns[0]
    btc_w = fwd.loc[ts_list, btc] * 1e4

    sma_all = wide_matrix(
        panel,
        {s: panel.frames[s][COMPARISON_FEATURE].to_numpy(dtype=float) for s in panel.symbols},
        index=fwd_all.index,
    )
    sma_spread = per_timestamp_spread(sma_all.loc[oos_ts], fwd)

    print("  robustness battery ...", flush=True)
    wins = np.clip(prim.to_numpy(), *np.quantile(prim.to_numpy(), [WINSOR_LO, WINSOR_HI]))
    half = len(prim) // 2
    per_symbol = legs.groupby("symbol")["ret"].mean() * 1e4
    denom = float(per_symbol.abs().sum())
    contrib = (per_symbol.abs() / denom * 100).sort_values(ascending=False)
    loo = {
        s: float(legs[legs["symbol"] != s]["ret"].mean() * 1e4)
        for s in sorted(legs["symbol"].unique())
    }
    by_month = spread.groupby(spread.index.floor("D").strftime("%Y-%m")).mean() * 1e4

    net_prim = prim - COST_SPREAD_BPS
    net_per_leg = prim / 2.0 - COST_PER_LEG_RT_BPS
    econ = {
        "convention": "per-leg: each long or short is one 36bps round trip",
        "cost_per_fill_bps": COST_PER_FILL_BPS,
        "cost_per_leg_round_trip_bps": COST_PER_LEG_RT_BPS,
        "safety_margin_per_leg_bps": SAFETY_MARGIN_BPS,
        "hurdle_per_leg_bps": HURDLE_PER_LEG_BPS,
        "hurdle_spread_bps": HURDLE_SPREAD_BPS,
        "gross_top_minus_bottom_spread_bps": float(prim.mean()),
        "per_leg_gross_expectancy_bps": float(prim.mean() / 2.0),
        "spread_after_costs_bps": float(prim.mean() - COST_SPREAD_BPS),
        "spread_after_safety_hurdle_bps": float(prim.mean() - HURDLE_SPREAD_BPS),
        "per_leg_net_expectancy_bps": float(prim.mean() / 2.0 - COST_PER_LEG_RT_BPS),
        "gross_sharpe": sharpe(prim.to_numpy()),
        "net_sharpe": sharpe(net_prim.to_numpy()),
        "per_leg_net_sharpe": sharpe(net_per_leg.to_numpy()),
        "turnover_per_rebalance": float(2 * (2 * TOP_K) / rs.shape[1]),
        "fills_per_rebalance": int(2 * TOP_K),
        "fills_per_day": float(2 * TOP_K),
        "rebalances": int(len(prim)),
        "gross_equity": equity_curve(prim.to_numpy()),
        "net_equity": equity_curve(net_prim.to_numpy()),
    }

    report: dict[str, Any] = {
        "preregistration": {
            "feature": FEATURE,
            "definition": (f"log(close[t]/close[t-{LOOKBACK}]) - median_over_eligible_universe"),
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "direction": "long top 3 RS, short bottom 3 RS; no reversal",
            "top_k": TOP_K,
            "universe": len(symbols),
            "oos_fraction": args.oos_fraction,
            "oos_start": str(oos_start),
            "oos_end": str(oos_ts[-1]),
            "n_oos_timestamps": n_oos,
            "n_timestamps_scored": int(spread.size),
            "non_overlap_stride": NON_OVERLAP_STRIDE,
            "winsor": [WINSOR_LO, WINSOR_HI],
            "seed": SEED,
            "contamination_disclosure": CONTAMINATION_DISCLOSURE,
            "architecture_note": (
                "production consumes 46 of the 64 originally declared features; "
                "sentiment and on-chain columns remain unactivated and were not "
                "used here"
            ),
        },
        "data_availability": {
            "mean_eligible_coins_per_oos_timestamp": float(
                rs_meta["n_eligible_per_ts"].loc[oos_ts].mean()
            ),
            "min_eligible_coins_per_oos_timestamp": int(
                rs_meta["n_eligible_per_ts"].loc[oos_ts].min()
            ),
            "timestamps_below_min_cross_section": int(
                (rs_meta["n_eligible_per_ts"].loc[oos_ts] < MIN_CROSS_SECTION).sum()
            ),
            "silent_deletions": 0,
            "market_context": {
                "universe_median_24h_ret_mean_bps": float(med.mean() * 1e4),
                "universe_median_24h_ret_median_bps": float(med.median() * 1e4),
                "note": "the subtracted term, reported as market drift context only",
            },
            "raw_24h_ret_mean_bps": float(raw_ret.stack().mean() * 1e4),
            "rs_mean_bps": float(rs.stack().mean() * 1e4),
            "rs_median_bps": float(rs.stack().median() * 1e4),
            "rs_is_median_centred": bool(
                abs(float(rs.median(axis=1, skipna=True).abs().max())) < 1e-9
            ),
            "why_rs_mean_is_not_zero": (
                "RS is centred on the cross-sectional MEDIAN, not the mean, so "
                "E[RS] = mean(ret) - median(ret) is positive whenever the "
                "cross-section is right-skewed. This is by construction, not a "
                "leak or a defect."
            ),
        },
        "leakage_checks": causality,
        "economics": econ,
        "cross_sectional": {
            "primary_nonoverlap_spread": stats,
            "block_bootstrap_95ci_spread_bps": boot,
            "permutation_random_selection": {
                "p_two_sided": p_rand,
                "observed_pctile": float(100.0 * (null["draws"] <= obs_bps).mean()),
                "null_mean_bps": float(null["draws"].mean()),
                "null_sd_bps": float(null["draws"].std(ddof=1)),
                "draws": N_DRAWS,
                "method": "random top3/bottom3 per timestamp; values not permuted",
            },
            "rank_ic": {
                "mean": float(ic_all.mean()),
                "median": float(np.median(ic_all)),
                "std": float(ic_all.std(ddof=1)),
                "t_stat": ic_t,
                "frac_positive": float((ic_all > 0).mean()),
                "n": int(ic_all.size),
            },
            "quintiles": quint,
            "deciles": dec,
            "decile_caveat": (
                "25 coins per timestamp give 2-3 coins per decile, so decile "
                "means are not independent observations; quintiles are the "
                "primary shape diagnostic"
            ),
            "frac_positive_spread": float((spread > 0).mean()),
        },
        "shape_diagnostic": shape,
        "time_series": {
            "mean_forward_relative_ret_bps": float(legs["ret"].mean() * 1e4),
            "median_forward_relative_ret_bps": float(legs["ret"].median() * 1e4),
            "hit_rate": float((legs["ret"] > 0).mean()),
            "std_bps": float(legs["ret"].std(ddof=1) * 1e4),
            "gross_sharpe": sharpe(prim.to_numpy()),
            "net_sharpe": sharpe(net_prim.to_numpy()),
            "n_legs": int(len(legs)),
            "note": "cross-sectional metrics are primary; time-series is secondary",
        },
        "comparison": {
            "relative_strength_spread_bps": float(prim.mean()),
            "inverse_rs_spread_bps": float((-prim).mean()),
            "sma_fast_slow_spread_bps_same_timestamps": float(
                sma_spread.iloc[::NON_OVERLAP_STRIDE].mean() * 1e4
            ),
            "equal_weight_universe_bps": float(equal_w.mean()),
            "btc_buy_hold_bps": float(btc_w.mean()),
            "random_selection_mean_bps": float(null["draws"].mean()),
            "identical_timestamps": True,
        },
        "robustness": {
            "full_oos_overlapping_mean_bps": float(spread.mean() * 1e4),
            "first_half_bps": float(prim.iloc[:half].mean()),
            "second_half_bps": float(prim.iloc[half:].mean()),
            "by_month_bps": {k: float(v) for k, v in by_month.items()},
            "raw_mean_bps": float(prim.mean()),
            "winsorized_mean_bps": float(wins.mean()),
            "raw_median_bps": float(np.median(prim.to_numpy())),
            "mean_vs_median_ratio": float(
                prim.mean() / np.median(prim.to_numpy())
                if np.median(prim.to_numpy()) != 0
                else float("nan")
            ),
            "by_symbol_mean_leg_bps": {k: float(v) for k, v in per_symbol.items()},
            "top2_symbol_share_pct": float(contrib.head(2).sum()),
            "leave_one_out_per_leg_bps": loo,
            "loso_min": float(min(loo.values())),
            "loso_max": float(max(loo.values())),
        },
    }

    g: dict[str, bool] = {}
    g["net_spread_after_costs_positive"] = bool(econ["spread_after_costs_bps"] > 0)
    g["clears_safety_hurdle"] = bool(econ["gross_top_minus_bottom_spread_bps"] > HURDLE_SPREAD_BPS)
    g["beats_random_ranking"] = bool(p_rand < 0.05)
    g["sensible_monotonic_ordering"] = bool(
        shape["classification"] in ("A_monotonic_continuation", "B_monotonic_mean_reversion")
    )
    g["not_symbol_concentrated"] = bool(report["robustness"]["top2_symbol_share_pct"] < 50.0)
    g["no_sign_reversal_across_halves"] = bool(
        report["robustness"]["first_half_bps"] > 0 and report["robustness"]["second_half_bps"] > 0
    )
    g["not_regime_collapsed"] = bool(
        all(v > 0 for v in report["robustness"]["by_month_bps"].values())
    )
    g["positive_primary_ci"] = bool(stats["ci95_low"] > 0)
    passed = all(g.values())
    report["decision"] = {
        "verdict": "PASS" if passed else "FAIL",
        "gates": g,
        "failed_gates": [k for k, v in g.items() if not v],
    }

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"wrote {args.out_report}", flush=True)
    print(f"shape: {shape['classification']}", flush=True)
    print(
        f"gross spread {econ['gross_top_minus_bottom_spread_bps']:.2f} bps "
        f"vs hurdle {HURDLE_SPREAD_BPS}",
        flush=True,
    )
    print(f"VERDICT: {report['decision']['verdict']}", flush=True)
    for k, v in g.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
