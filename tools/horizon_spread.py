"""Edge per trade at each horizon -- the metric that actually decides the interval.

Trading cost is a fixed number of basis points per round trip, independent of how
long the position is held. So the decision-relevant quantity is not edge per bar
but edge per trade. A longer bar interval earns more bps per trade if and only if
the model's edge persists across bars instead of decaying.

This measures the realised long-top3 / short-bottom3 spread in basis points for
every holding horizon, with a day-block bootstrap interval, and prices each one
against the full round-trip cost.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cqip.config import Settings
from cqip.storage import ParquetStore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_economics import FILL_BPS, ROUND_TRIP_BPS, TOP_N, block_bootstrap_ci  # noqa: E402

BAR_MINUTES = 60
HORIZONS = [1, 2, 4, 6, 8, 12, 16, 24, 48, 96]
# The product's stated holding range, mapped onto hourly bars.
PRODUCT_HOLDS_H = [0.25, 0.5, 1, 2, 3, 4]


def main() -> None:
    store = ParquetStore(settings=Settings())
    preds = pd.read_parquet("data/artifacts/preds_lstm_1h.parquet")
    symbols = sorted(preds["symbol"].unique())

    ohlc = store.read("ohlcv", filters=[("interval", "==", "1h")], symbols=symbols)
    ohlc["ts"] = pd.DatetimeIndex(ohlc["ts"])
    close = ohlc.pivot_table(index="ts", columns="symbol", values="close").sort_index().ffill()

    ranking = preds[["ts", "symbol", "score"]].copy()

    print("=" * 104)
    print("REALISED EDGE PER TRADE BY HOLDING HORIZON (long top 3 / short bottom 3)")
    print("=" * 104)
    print(f"  round-trip cost modelled at {ROUND_TRIP_BPS:.1f} bps "
          f"({FILL_BPS:.1f} bps per fill x 2)")
    print()

    rows = []
    for k in HORIZONS:
        fwd = (close.shift(-k) / close - 1.0)
        merged = ranking.merge(
            fwd.rename_axis("ts").reset_index().melt(
                id_vars="ts", var_name="symbol", value_name="realised"),
            on=["ts", "symbol"], how="inner",
        ).dropna(subset=["realised"])

        def spread(group: pd.DataFrame) -> float:
            r = group.sort_values("score", ascending=False)
            return float(r.head(TOP_N)["realised"].mean() - r.tail(TOP_N)["realised"].mean())

        series = merged.groupby("ts", group_keys=False).apply(
            spread, include_groups=False).dropna().to_numpy() * 10_000.0

        lo, hi = block_bootstrap_ci(series)
        rows.append({
            "bars": k,
            "hours": k * BAR_MINUTES / 60,
            "n": len(series),
            "gross_bps": float(series.mean()),
            "ci_lo": lo,
            "ci_hi": hi,
            "net_bps": float(series.mean()) - ROUND_TRIP_BPS,
            "hit_rate": float((series > 0).mean()),
        })

    header = (f"  {'hold':>7}{'bars':>6}{'gross bps':>11}{'95% CI':>20}"
              f"{'net bps':>10}{'hit rate':>11}  verdict")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        ci = f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
        sig = r["ci_lo"] > 0
        verdict = "net +ve" if r["net_bps"] > 0 else "net -ve"
        if not sig:
            verdict += " (edge not significant)"
        print(f"  {r['hours']:>6.2f}h{r['bars']:>6}{r['gross_bps']:>11.2f}"
              f"{ci:>20}{r['net_bps']:>10.2f}{r['hit_rate']:>11.1%}  {verdict}")

    print()
    best = max(rows, key=lambda r: r["net_bps"])
    first_pos = next((r for r in rows if r["net_bps"] > 0), None)
    print(f"  best net outcome : {best['net_bps']:+.2f} bps at a {best['hours']:.0f}h hold")
    if first_pos is None:
        print("  no tested horizon clears costs net of fees")
    else:
        print(f"  first horizon to clear costs: {first_pos['hours']:.0f}h "
              f"({first_pos['net_bps']:+.2f} bps)")

    print()
    print("=" * 104)
    print("EDGE DECAY -- does the signal persist, or does it vanish after one bar?")
    print("=" * 104)
    one_bar = rows[0]["gross_bps"]
    print(f"  {'bars':>6}{'gross bps':>12}{'x one-bar':>12}{'efficiency':>14}")
    for r in rows:
        ratio = r["gross_bps"] / one_bar if one_bar else np.nan
        # Linear accumulation would give ratio == bars. Efficiency below 1 means
        # the edge decays; above 1 means it compounds.
        eff = ratio / r["bars"] if r["bars"] else np.nan
        print(f"  {r['bars']:>6}{r['gross_bps']:>12.2f}{ratio:>12.2f}{eff:>14.2f}")
    print()
    print("  Efficiency near 1.0 means the edge accrues steadily and longer holds")
    print("  genuinely pay. Efficiency well below 1.0 means the model only knows the")
    print("  next bar and every extra bar of holding is dead weight you pay for.")

    print()
    print("=" * 104)
    print("THE PRODUCT'S REQUESTED RANGE, PRICED")
    print("=" * 104)
    print(f"  {'requested hold':>16}{'intervals that can hold it':>34}{'net bps':>11}")
    for hold_h in PRODUCT_HOLDS_H:
        allowed = [iv for iv, m in (("15m", 15), ("30m", 30), ("1h", 60),
                                    ("2h", 120), ("4h", 240)) if m / 60 <= hold_h]
        nearest = min(rows, key=lambda r: abs(r["hours"] - hold_h))
        print(f"  {hold_h:>14.2f}h{', '.join(allowed):>34}{nearest['net_bps']:>11.2f}")
    print()
    print("  A hold shorter than the bar interval cannot be executed on that grid at")
    print("  all, so the interval and the holding period are not independent choices.")


if __name__ == "__main__":
    main()
