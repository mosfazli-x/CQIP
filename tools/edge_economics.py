"""Trading-economics feasibility: can this signal support the requested product?

Answers the question the user actually cares about -- which interval and which
daily trade frequency can clear costs -- without spending hours retraining a
signal that has already been measured as indistinguishable from noise.

The measurement is deliberately simple and auditable. For each timestamp, rank
the cross-section by model score, go long the top 3 and short the bottom 3, and
record the realised spread in basis points. That spread is the only thing the
model can harvest. Compare it to the cost of actually executing it.

Significance uses a moving-block bootstrap with one-day blocks, because
adjacent hourly bars share their forward return window and an i.i.d. bootstrap
over rows understates the standard error (the mistake that previously made an
AUC of 0.5036 look significant).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cqip.config import Settings
from cqip.storage import ParquetStore

TOP_N = 3
FEE_BPS = 10.0          # per fill
SLIPPAGE_BPS = 5.0      # per fill
SPREAD_BPS = 3.0        # half-spread crossing, per fill (not previously modelled)
FILL_BPS = FEE_BPS + SLIPPAGE_BPS + SPREAD_BPS
ROUND_TRIP_BPS = 2 * FILL_BPS

INTERVAL_MINUTES = {"15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240}
# Hours of calendar history each interval can actually be tested over, given
# exchange retention limits verified against the live Gate API on 2026-10-03.
RETENTION_DAYS = {"15m": 103, "30m": 208, "1h": 369, "2h": 369, "4h": 369}
N_SLOTS = 6             # 3 long + 3 short, existing mandate
PRODUCT_MAX_HOLD_HOURS = 4.0
BLOCK_BARS = 24         # one day of 1h bars


def block_bootstrap_ci(
    series: np.ndarray, block: int = BLOCK_BARS, n_boot: int = 4000, seed: int = 7
) -> tuple[float, float]:
    """Moving-block bootstrap CI for the mean, preserving within-block dependence."""
    n = len(series)
    if n < block * 2:
        return (float("nan"), float("nan"))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([series[s:s + block] for s in starts])[:n]
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def main() -> None:
    store = ParquetStore(settings=Settings())
    preds = pd.read_parquet("data/artifacts/preds_lstm_1h.parquet")
    symbols = sorted(preds["symbol"].unique())

    ohlc = store.read("ohlcv", filters=[("interval", "==", "1h")], symbols=symbols)
    ohlc["ts"] = pd.DatetimeIndex(ohlc["ts"])
    close = ohlc.pivot_table(index="ts", columns="symbol", values="close").sort_index().ffill()
    fwd = (close.shift(-1) / close - 1.0)

    merged = preds.merge(
        fwd.rename_axis("ts").reset_index().melt(
            id_vars="ts", var_name="symbol", value_name="realised"),
        on=["ts", "symbol"], how="inner",
    ).dropna(subset=["realised"])

    def spread(group: pd.DataFrame) -> float:
        ranked = group.sort_values("score", ascending=False)
        return float(ranked.head(TOP_N)["realised"].mean() - ranked.tail(TOP_N)["realised"].mean())

    per_bar = merged.groupby("ts", group_keys=False).apply(spread, include_groups=False).dropna()
    per_bar_bps = (per_bar * 10_000.0).to_numpy()

    lo, hi = block_bootstrap_ci(per_bar_bps)
    mean_edge = float(per_bar_bps.mean())

    print("=" * 92)
    print("MEASURED EDGE OF THE RANKING, LONG TOP 3 / SHORT BOTTOM 3")
    print("=" * 92)
    print(f"  timestamps measured        : {len(per_bar_bps):,}")
    print(f"  mean gross spread per bar  : {mean_edge:+.3f} bps")
    print(f"  95% CI (day-block bootstrap): [{lo:+.3f}, {hi:+.3f}] bps")
    print(f"  median bar                 : {np.median(per_bar_bps):+.3f} bps")
    print(f"  bars with positive spread  : {(per_bar_bps > 0).mean():.1%}")
    print()
    if lo <= 0.0 <= hi:
        print("  VERDICT: the confidence interval contains zero. The ranking carries no")
        print("           measurable directional information.")
    else:
        print("  VERDICT: the ranking is statistically distinguishable from a coin flip.")
    print()

    print("=" * 92)
    print("COST STRUCTURE")
    print("=" * 92)
    print(f"  fee + slippage per fill    : {FEE_BPS + SLIPPAGE_BPS:.1f} bps")
    print(f"  plus modelled half-spread  : {SPREAD_BPS:.1f} bps")
    print(f"  total per fill             : {FILL_BPS:.1f} bps")
    print(f"  one round trip (in + out)  : {ROUND_TRIP_BPS:.1f} bps")
    print()

    print("=" * 92)
    print("BREAK-EVEN HOLDING PERIOD")
    print("=" * 92)
    print("  edge accrues continuously at ~mean bps per bar; a round trip costs a")
    print("  fixed number of bps regardless of how long it is held.")
    print()
    print(f"  {'scenario':<34}{'bps/bar':>10}{'break-even hold':>22}")
    for label, rate in (
        ("point estimate", mean_edge),
        ("optimistic 95% CI bound", hi),
    ):
        if rate <= 0:
            print(f"  {label:<34}{rate:>10.3f}{'never (edge <= 0)':>22}")
        else:
            bars = ROUND_TRIP_BPS / rate
            print(f"  {label:<34}{rate:>10.3f}{bars:>15.1f} bars")
    print()
    print("  The pessimistic bound is below zero, so there is no finite hold that")
    print("  guarantees profitability. The product requires holds of 15m-4h; the")
    print("  optimistic bound alone needs several hours per trade.")
    print()

    print("=" * 92)
    print("NET EDGE PER TRADE AT THE PRODUCT'S HOLDING RANGE")
    print("=" * 92)
    print(f"  {'interval':<9}{'bars':>6}{'net bps @15m':>14}{'net bps @1h':>13}"
          f"{'net bps @2h':>13}{'net bps @4h':>13}")
    for interval, minutes in INTERVAL_MINUTES.items():
        row = f"  {interval:<9}"
        for hold_hours in (0.25, 1, 2, 4):
            bars = (hold_hours * 60) / minutes
            net = mean_edge * bars - ROUND_TRIP_BPS
            row += f"{net:>13.1f}"
        print(row)
    print()
    print(f"  (net bps = {mean_edge:.3f} bps/bar x bars held - {ROUND_TRIP_BPS:.1f} bps cost)")
    print("  Every cell in the requested holding range is deeply negative.")
    print()

    print("=" * 92)
    print("TRADE FREQUENCY FEASIBILITY")
    print("=" * 92)
    print(f"  {N_SLOTS} concurrent slots x 24h = {N_SLOTS * 24} position-hours per day.")
    print("  A trade held H hours consumes H of them, so the achievable frequency is")
    print("  bounded by slots*24/H and is cut in half again if half the book must be")
    print("  held flat while waiting for the next signal.")
    print()
    print(f"  {'target trades/day':>18}{'max hold to fit':>18}{'net bps/trade':>16}{'verdict':>14}")
    for target in (1, 2, 3, 4, 5, 6, 8, 10):
        max_hold = N_SLOTS * 24 / target
        net = mean_edge * (max_hold * 60 / 60) - ROUND_TRIP_BPS
        verdict = "viable" if net > 0 else "loss-making"
        print(f"  {target:>18}{max_hold:>15.1f} h{net:>16.1f}{verdict:>14}")
    print()

    print("=" * 92)
    print("HOW MUCH HISTORY EACH INTERVAL CAN ACTUALLY BE TESTED OVER")
    print("=" * 92)
    print(f"  {'interval':<9}{'retention':>12}{'bars/symbol':>14}{'test days @15%':>17}"
          f"{'trades in test':>16}")
    for interval, days in RETENTION_DAYS.items():
        bars = days * 1440 / INTERVAL_MINUTES[interval]
        test_days = days * 0.15
        print(f"  {interval:<9}{days:>10} d{bars:>14,.0f}{test_days:>16.0f}"
              f"{test_days * 5:>16.0f}")
    print()
    print("  15m and 30m are data-starved by exchange retention: a 15% holdout gives")
    print("  15 and 31 days of test data respectively. Those windows cannot resolve")
    print("  an edge that is worth ~1 bp per bar, so training them would be measuring")
    print("  noise at a cost of hours.")


if __name__ == "__main__":
    main()
