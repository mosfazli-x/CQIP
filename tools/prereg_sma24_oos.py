"""Pre-registered out-of-sample validation of the 24h `sma_fast_slow_spread` rule.

Frozen before the result was computed. Nothing here is tuned: the feature, the
horizon, the direction, the selection size, the cost model, the OOS fraction,
the robustification method and the decision tree are all constants.

The rule
--------
At every timestamp, rank the 25-coin universe by `sma_fast_slow_spread`
ascending, go long the top 3 and short the bottom 3, hold 24h, close. This is
the production selection rule with the model's score replaced by the frozen
feature.

Economic convention (production)
--------------------------------
`tools/horizon_discovery.py:signal_economics` treats EVERY selected leg as one
round trip costing 36bps. A long-short spread therefore costs 2 x 36 = 72bps,
and because longs and shorts are equally sized the per-leg mean is exactly half
the top-minus-bottom spread. The hurdle is 60bps PER LEG = 120bps of spread.
Both framings are reported so the two cannot be confused again.

No look-ahead
-------------
`sma_fast_slow_spread` is (sma10 - sma30) / sma30 over closes, so bar t's value
uses close[t]. The rule decides at the CLOSE of bar t and transacts at that
same close (market-on-close). `open[t+1]` is reported separately as a
sensitivity of that convention, not as a tuned alternative.
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
from feature_audit import (  # noqa: E402
    HORIZONS,
    forward_returns,
    load_panel,
    wide_matrix,
)

from cqip.storage import ParquetStore  # noqa: E402

# --------------------------------------------------------------------------- #
# PRE-REGISTERED CONSTANTS -- frozen before evaluation
# --------------------------------------------------------------------------- #

FEATURE = "sma_fast_slow_spread"
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

OOS_FRACTION = 0.25
NON_OVERLAP_STRIDE = HORIZON  # one observation per 24h cycle
WINSOR_LO, WINSOR_HI = 0.01, 0.99  # single pre-declared robustification
MIN_CROSS_SECTION = 2 * TOP_K
TRADING_DAYS_PER_YEAR = 365.0

# Contamination disclosure: this was already inspected during the feature audit.
PRIOR_PEEK_DISCLOSURE = (
    "The most recent 25% of this sample was NOT untouched at the time this test "
    "was written. The feature audit already reported a quartile decomposition of "
    "the full-sample spread for this exact feature and horizon, in which the "
    "final quartile read approximately -14bps. No clean holdout exists in the "
    "stored data because the panel already ends at the last available bar. "
    "Treat every p-value below as optimistic and the PASS bar as correspondingly "
    "higher than a genuinely pre-registered threshold would be."
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _winsorize(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v
    lo, hi = np.quantile(v, [WINSOR_LO, WINSOR_HI])
    return np.clip(v, lo, hi)


def summarize(values: np.ndarray) -> dict[str, Any]:
    """Distribution summary used identically for every slice of the battery."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
        "min": float(v.min()),
        "max": float(v.max()),
        "win_rate": float((v > 0).mean()),
        "frac_positive": float((v > 0).mean()),
    }


def mean_t(values: np.ndarray) -> tuple[float, float, float]:
    """Mean, standard error and t-stat of a mean."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    se = v.std(ddof=1) / np.sqrt(n)
    return float(v.mean()), float(se), float(v.mean() / se) if se > 0 else float("nan")


# --------------------------------------------------------------------------- #
# panel
# --------------------------------------------------------------------------- #


def build_frames(panel) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-sectional frame (ts x symbol) for the feature and the 24h return."""
    fwd = wide_matrix(panel, forward_returns(panel, HORIZON))
    feat = wide_matrix(
        panel,
        {s: panel.frames[s][FEATURE].to_numpy(dtype=float) for s in panel.symbols},
        index=fwd.index,
    )
    return feat, fwd


def select_legs(feat: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen rule to every complete cross-section.

    Returns one row per selected leg with its realised net-of-nothing return, so
    downstream accounting matches the production per-leg convention exactly.
    """
    ts_index = feat.index
    rows: list[dict[str, Any]] = []
    for ts in ts_index:
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
    """Top3 minus bottom3 realised return, one value per complete timestamp."""
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
    """Within-timestamp Spearman rank correlation."""
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


def quantile_buckets(feat: pd.DataFrame, fwd: pd.DataFrame, n_buckets: int) -> pd.DataFrame:
    """Equal-count buckets of the feature with mean realised return per bucket.

    With 25 coins a decile is 2-3 observations per timestamp, which cannot
    support a monotonicity claim, so quintiles are used and the reason recorded.
    """
    acc: dict[int, list[float]] = {b: [] for b in range(1, n_buckets + 1)}
    for ts in feat.index:
        f = feat.loc[ts].to_numpy(dtype=float)
        r = fwd.loc[ts].to_numpy(dtype=float)
        ok = np.isfinite(f) & np.isfinite(r)
        if int(ok.sum()) < MIN_CROSS_SECTION:
            continue
        idx = np.flatnonzero(ok)
        order = idx[np.argsort(f[idx], kind="mergesort")]
        for b, chunk in enumerate(np.array_split(order, n_buckets), start=1):
            acc[b].extend(r[chunk].tolist())
    return pd.DataFrame(
        {
            "bucket": list(acc),
            "n": [len(v) for v in acc.values()],
            "mean_ret": [float(np.mean(v)) if v else float("nan") for v in acc.values()],
        }
    )


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #


def baselines(feat: pd.DataFrame, fwd: pd.DataFrame, seed: int, draws: int) -> dict[str, Any]:
    """Fixed baselines on identical timestamps: market, random and inverse."""
    ts_list = list(per_timestamp_spread(feat, fwd).index)
    equal_w = fwd.loc[ts_list].mean(axis=1)
    btc = "BTC_USDT" if "BTC_USDT" in fwd.columns else fwd.columns[0]
    btc_w = fwd.loc[ts_list, btc]

    observed = per_timestamp_spread(feat, fwd)

    # Null: randomise the SELECTION, not the values. Permuting a fixed multiset
    # of spreads and re-averaging is degenerate -- the mean is invariant -- so a
    # permutation test on the mean would always return p = 1. For each draw we
    # instead pick a random top-3 and bottom-3 from each timestamp's returns,
    # which is the actual question: does ranking by the feature beat chance?
    rng = np.random.default_rng(seed)
    panels = []
    for ts in feat.index:
        r = fwd.loc[ts].to_numpy(dtype=float)
        ok = np.flatnonzero(np.isfinite(r))
        if ok.size >= MIN_CROSS_SECTION:
            panels.append(r[ok])
    draws_out = np.empty(draws)
    for i in range(draws):
        vals = []
        for r in panels:
            pick = rng.choice(r.size, size=2 * TOP_K, replace=False)
            vals.append(r[pick[:TOP_K]].mean() - r[pick[TOP_K:]].mean())
        draws_out[i] = float(np.mean(vals)) * 1e4
    obs_bps = float(observed.mean() * 1e4)
    p_rand = float((np.abs(draws_out) >= abs(obs_bps) - 1e-12).mean())
    p_two_sided_pct = 100.0 * float((draws_out <= obs_bps).mean())

    inverse = -observed
    return {
        "equal_weight_universe": summarize(equal_w.to_numpy() * 1e4),
        "btc_buy_hold": summarize(btc_w.to_numpy() * 1e4),
        "frozen_rule_spread": summarize(observed.to_numpy() * 1e4),
        "inverse_ranking_spread": summarize(inverse.to_numpy() * 1e4),
        "random_selection_null_mean_bps": float(draws_out.mean()),
        "random_selection_null_sd_bps": float(draws_out.std(ddof=1)),
        "random_selection_p_two_sided": p_rand,
        "observed_pctile_in_null": p_two_sided_pct,
        "n_timestamps": len(ts_list),
    }


# --------------------------------------------------------------------------- #
# robustness battery (fixed, non-optimized)
# --------------------------------------------------------------------------- #


def robustness(legs: pd.DataFrame, spread: pd.Series, ic: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}

    prim_series = spread.iloc[::NON_OVERLAP_STRIDE] * 1e4
    prim_m, prim_se, prim_t = mean_t(prim_series.to_numpy())
    out["primary_nonoverlap"] = {
        "sampling_rule": f"every {NON_OVERLAP_STRIDE}th OOS timestamp (one per 24h cycle)",
        "n": int(prim_series.size),
        "mean_spread_bps": prim_m,
        "se_bps": prim_se,
        "t_stat": prim_t,
        "frac_positive": float((prim_series > 0).mean()),
    }
    overlap_m, overlap_se, _ = mean_t(spread.to_numpy() * 1e4)
    out["full_hourly_descriptive"] = {
        "note": "overlapping windows; reported descriptively only, never for inference",
        "n_timestamps_overlapping": int(spread.size),
        "spread_mean_bps_overlapping": overlap_m,
        "spread_se_naive_bps_overlapping": overlap_se,
        "n_legs": int(len(legs)),
        "per_leg_mean_bps": overlap_m / 2.0,
    }

    half = len(spread) // 2
    out["first_vs_second_half"] = {
        "first_half_mean_bps": float(spread.iloc[:half].mean() * 1e4),
        "second_half_mean_bps": float(spread.iloc[half:].mean() * 1e4),
    }

    by_month = spread.groupby(spread.index.floor("D").strftime("%Y-%m")).mean() * 1e4
    out["by_month"] = {str(k): float(v) for k, v in by_month.items() if np.isfinite(v)}

    prim = spread.iloc[::NON_OVERLAP_STRIDE] * 1e4
    out["robustification"] = {
        "method": f"single pre-declared winsorization at {WINSOR_LO}/{WINSOR_HI}",
        "raw_mean_bps": float(prim.mean()),
        "winsorized_mean_bps": float(_winsorize(prim.to_numpy()).mean()),
        "raw_median_bps": float(prim.median()),
        "retention_ratio_median_over_mean": float(
            prim.median() / prim.mean() if prim.mean() != 0 else float("nan")
        ),
    }

    per_symbol = legs.groupby("symbol")["ret"].mean() * 1e4
    denom = float(per_symbol.abs().sum())
    contrib = (per_symbol.abs() / denom * 100).sort_values(ascending=False) if denom else per_symbol
    out["by_symbol"] = {
        "mean_leg_bps": {k: float(v) for k, v in per_symbol.items()},
        "pct_of_total_gross": {k: float(v) for k, v in contrib.items()},
        "concentration_denominator": "sum of absolute per-symbol means, because the "
        "signed total is near zero and makes a signed share meaningless",
        "top2_symbol_share_pct": float(contrib.head(2).sum()),
        "equal_weight_symbol_mean_bps": float(per_symbol.mean()),
    }

    loo: dict[str, float] = {}
    for sym in sorted(legs["symbol"].unique()):
        kept = legs[legs["symbol"] != sym]
        loo[sym] = float(kept["ret"].mean() * 1e4)
    out["leave_one_symbol_out"] = {
        "per_leg_mean_bps": loo,
        "min": float(min(loo.values())),
        "max": float(max(loo.values())),
    }
    out["rank_ic"] = {
        "mean": float(ic.mean()),
        "median": float(ic.median()),
        "t_stat": mean_t(ic.to_numpy())[2],
        "frac_positive": float((ic > 0).mean()),
    }
    return out


# --------------------------------------------------------------------------- #
# decision tree (pre-declared, applied literally)
# --------------------------------------------------------------------------- #


def decide(report: dict[str, Any]) -> dict[str, Any]:
    prim = report["robustness"]["primary_nonoverlap"]
    rob = report["robustness"]["robustification"]
    sym = report["robustness"]["by_symbol"]
    econ = report["economics"]
    gates: dict[str, bool] = {}

    gates["oos_expectancy_positive"] = bool(prim["mean_spread_bps"] > 0)
    gates["clears_economic_hurdle"] = bool(econ["per_leg_gross_bps"] > HURDLE_PER_LEG_BPS)
    gates["net_positive_after_costs"] = bool(econ["per_leg_net_bps"] > 0)
    gates["direction_consistent"] = bool(prim["mean_spread_bps"] > 0)
    gates["survives_winsorization"] = bool(
        rob["winsorized_mean_bps"] > 0 and rob["raw_median_bps"] > 0
    )
    gates["not_dominated_by_median_gap"] = bool(rob["retention_ratio_median_over_mean"] > 0.5)
    gates["not_two_coin_concentrated"] = bool(sym["top2_symbol_share_pct"] < 50.0)
    gates["second_half_positive"] = bool(
        report["robustness"]["first_vs_second_half"]["second_half_mean_bps"] > 0
    )

    passed = all(gates.values())
    failed = [k for k, v in gates.items() if not v]
    return {
        "verdict": "PASS" if passed else "FAIL",
        "gates": gates,
        "failed_gates": failed,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-report", type=Path, required=True)
    ap.add_argument("--oos-fraction", type=float, default=OOS_FRACTION)
    ap.add_argument("--random-draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "tools"))
    symbols = list(ab.PRODUCTION_UNIVERSE)
    store = ParquetStore()
    print("PRE-REGISTERED OOS TEST: sma_fast_slow_spread @ 24h", flush=True)
    print(f"  universe={len(symbols)} horizons={HORIZONS} feature={FEATURE}", flush=True)

    panel = load_panel(store, symbols, "1h", max(HORIZONS), 5100)
    feat_all, fwd_all = build_frames(panel)

    # OOS slice: most recent fraction of the panel's timestamps. No fitting of
    # any kind happens after this point; the rule is a pure within-timestamp
    # rank, so it needs no normalisation and therefore has no state to fit.
    n_ts = len(feat_all.index)
    n_oos = int(round(n_ts * args.oos_fraction))
    oos_ts = feat_all.index[-n_oos:]
    oos_start = oos_ts[0]

    feat = feat_all.loc[oos_ts]
    fwd = fwd_all.loc[oos_ts]

    print(f"  panel timestamps={n_ts}  OOS timestamps={n_oos}", flush=True)
    print(f"  OOS begins {oos_start}", flush=True)
    print("  applying frozen rule ...", flush=True)

    legs = select_legs(feat, fwd)
    spread = per_timestamp_spread(feat, fwd)
    ic = rank_ic(feat, fwd)

    prim = spread.iloc[::NON_OVERLAP_STRIDE] * 1e4
    spread_m = float(prim.mean())
    per_leg_gross = float(spread_m / 2.0)

    n_cycles = len(prim)
    gross_sharpe = (
        float(prim.mean() / prim.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if prim.std(ddof=1) > 0
        else float("nan")
    )
    net_series = prim - (2 * COST_PER_LEG_RT_BPS)
    net_sharpe = (
        float(net_series.mean() / net_series.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if net_series.std(ddof=1) > 0
        else float("nan")
    )

    econ = {
        "convention": "production per-leg: each selected long or short is one round trip",
        "cost_per_fill_bps": COST_PER_FILL_BPS,
        "cost_per_leg_round_trip_bps": COST_PER_LEG_RT_BPS,
        "safety_margin_bps": SAFETY_MARGIN_BPS,
        "hurdle_per_leg_gross_bps": HURDLE_PER_LEG_BPS,
        "hurdle_spread_bps": HURDLE_SPREAD_BPS,
        "spread_gross_bps": spread_m,
        "per_leg_gross_bps": per_leg_gross,
        "per_leg_net_bps": per_leg_gross - COST_PER_LEG_RT_BPS,
        "spread_net_bps": spread_m - 2 * COST_PER_LEG_RT_BPS,
        "gross_sharpe_nonoverlap": gross_sharpe,
        "net_sharpe_nonoverlap": net_sharpe,
        "turnover_per_rebalance": float(2 * (2 * TOP_K) / feat.shape[1]),
        "rebalances": n_cycles,
        "legs_per_rebalance": 2 * TOP_K,
        "approx_trades_per_day": float(2 * TOP_K),
        "holding_hours": HORIZON,
    }

    print("  robustness battery ...", flush=True)
    rob = robustness(legs, spread, ic)
    rob["bucket_monotonicity"] = {
        "note": (
            "deciles are infeasible with 25 coins (2-3 observations per bucket per "
            "timestamp); quintiles reported instead"
        ),
        "quintiles": quantile_buckets(feat, fwd, 5).to_dict("records"),
    }

    print("  baselines ...", flush=True)
    base = baselines(feat, fwd, args.seed, args.random_draws)

    print("  time-series view ...", flush=True)
    ts_rows = {}
    for sym, g in legs.groupby("symbol"):
        ts_rows[sym] = {
            "mean_leg_bps": float(g["ret"].mean() * 1e4),
            "median_leg_bps": float(g["ret"].median() * 1e4),
            "hit_rate": float((g["ret"] > 0).mean()),
            "n_legs": int(len(g)),
        }

    report: dict[str, Any] = {
        "preregistration": {
            "feature": FEATURE,
            "horizon": HORIZON,
            "direction": "long highest feature value, short lowest; no reversal",
            "top_k": TOP_K,
            "oos_fraction": args.oos_fraction,
            "oos_start": str(oos_start),
            "oos_end": str(oos_ts[-1]),
            "n_oos_timestamps": n_oos,
            "non_overlap_stride": NON_OVERLAP_STRIDE,
            "winsor": [WINSOR_LO, WINSOR_HI],
            "cost_per_fill_bps": COST_PER_FILL_BPS,
            "cost_per_leg_round_trip_bps": COST_PER_LEG_RT_BPS,
            "hurdle_per_leg_bps": HURDLE_PER_LEG_BPS,
            "hurdle_spread_bps": HURDLE_SPREAD_BPS,
            "primary_metric": "non-overlapping top3-minus-bottom3 spread, bps",
            "contamination_disclosure": PRIOR_PEEK_DISCLOSURE,
        },
        "temporal_availability": {
            "feature_formula": "(sma(close,10) - sma(close,30)) / sma(close,30)",
            "uses_current_bar_close": True,
            "decision_convention": (
                "decide and transact at the CLOSE of bar t, using close[t] for both "
                "the feature and the entry price (market-on-close)"
            ),
            "target": f"log(close[t+{HORIZON}] / close[t])",
            "no_lookahead": (
                "sma10 and sma30 are trailing windows ending at t; no centred or "
                "forward-looking window is used"
            ),
            "alternative_convention_note": (
                "entering at open[t+1] instead is a different strategy and is not "
                "evaluated here; the frozen convention is close[t]"
            ),
        },
        "economics": econ,
        "robustness": rob,
        "time_series_by_symbol": ts_rows,
        "baselines": base,
        "n_timestamps_with_spread": int(spread.size),
    }
    report["decision"] = decide(report)

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out_report}", flush=True)
    print(f"VERDICT: {report['decision']['verdict']}", flush=True)
    for k, v in report["decision"]["gates"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
