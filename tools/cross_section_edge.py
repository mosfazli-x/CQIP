"""Cross-sectional edge measurement, which is the statistic that actually matters.

The previous diagnostic measured per-bar AUC. For a book that ranks 25 symbols
each timestamp and goes long the top 3 / short the bottom 3, the quantity of
interest is not "does this one symbol go up" but "does the ranking order the
cross-section correctly, within each timestamp". Those are different questions
and can disagree.

This measures the edge three ways, all read-only on the frozen model:
  1. cross-sectional Spearman of score vs realised rank, within each timestamp
  2. the top3-minus-bottom3 realised spread the book actually captures
  3. the same spread after costs, and how it is distributed over time
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
COST_PER_FILL = 0.0015  # 10bps fee + 5bps slippage, per side


def cross_sectional_spearman(frame: pd.DataFrame, score: str) -> pd.Series:
    """Rank correlation between score and realised return, computed within each ts."""
    def one(group: pd.DataFrame) -> float:
        if len(group) < 3:
            return np.nan
        a = group[score].rank()
        b = group["realised"].rank()
        if a.std() == 0 or b.std() == 0:
            return np.nan
        return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))

    return frame.groupby("ts", group_keys=False).apply(one)


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

    print(f"OOS bars (timestamps): {merged['ts'].nunique()}")
    print(f"symbols per timestamp: {merged.groupby('ts').size().min()}")
    print()

    print("=" * 96)
    print("1. CROSS-SECTIONAL RANK CORRELATION, WITHIN EACH TIMESTAMP")
    print("=" * 96)
    for score in ("score", "predicted_return"):
        rho = cross_sectional_spearman(merged, score).dropna()
        # Effective sample: neighbouring timestamps overlap in their forward
        # window, so treat one day (24 bars) as roughly one independent draw.
        n_eff = len(rho) / 24.0
        se = 1.0 / np.sqrt(n_eff - 1.0)
        z = float(rho.mean() / se)
        print(f"  {score:<18} mean rho {rho.mean():+.5f}  SE~{se:.5f}  z={z:+6.2f}  "
              f"{'significant' if abs(z) > 1.96 else 'NOT significant'}")

    print()
    print("=" * 96)
    print("2. THE SPREAD THE BOOK ACTUALLY CAPTURES (long top 3, short bottom 3)")
    print("=" * 96)

    def spread(group: pd.DataFrame, score: str) -> float:
        ranked = group.sort_values(score, ascending=False)
        long = ranked.head(TOP_N)["realised"].mean()
        short = ranked.tail(TOP_N)["realised"].mean()
        return float(long - short)

    for score in ("score", "predicted_return"):
        col = score
        s = merged.groupby("ts", group_keys=False).apply(
            lambda g, c=col: spread(g, c), include_groups=False)
        s = s.dropna()
        gross_bps = s * 10_000.0
        n_eff = len(s) / 24.0
        se_bps = float(gross_bps.std() / np.sqrt(n_eff))
        z = float(gross_bps.mean() / se_bps)
        print(f"  {score}")
        print(f"    mean gross spread/bar : {gross_bps.mean():+7.2f} bps   (SE~{se_bps:.2f}, z={z:+.2f})")
        print(f"    median                : {gross_bps.median():+7.2f} bps")
        print(f"    share of bars positive: {(s > 0).mean():6.1%}")
        # Round trip = two fills, so a position opened and closed pays 2x the
        # per-fill cost. A spread earned per bar is compared to one round trip.
        print(f"    cost of one round trip: {COST_PER_FILL * 2 * 10_000:7.2f} bps")
        print(f"    gross/cost ratio      : {gross_bps.mean() / (COST_PER_FILL * 2 * 10_000):7.3f}")
        print()

    print("=" * 96)
    print("3. WHERE DOES THE GROSS RETURN COME FROM? (monthly, ranked by score)")
    print("=" * 96)

    def spread_frame(score: str) -> pd.Series:
        return merged.groupby("ts", group_keys=False).apply(
            lambda g: spread(g, score), include_groups=False).dropna()

    s = spread_frame("score")
    monthly = s.groupby(s.index.to_period("M")).mean() * 10_000.0
    print(monthly.to_string(float_format=lambda v: f"{v:+8.2f} bps/bar"))
    print()
    best_month = monthly.idxmax()
    without_best = monthly.drop(best_month).mean()
    print(f"  best month          : {best_month}  {monthly.max():+.2f} bps")
    print(f"  mean excl. that month: {without_best:+.2f} bps")
    print(f"  share of total edge from the best month: "
          f"{monthly.max() / monthly.sum() if monthly.sum() != 0 else np.nan:.1%}")
    print(f"  months positive     : {(monthly > 0).sum()} / {len(monthly)}")
    print()

    top_days = s.groupby(s.index.date).mean().sort_values(ascending=False)
    print("  top 10 days:")
    for day, value in top_days.head(10).items():
        print(f"    {day}  {value * 10_000:+8.2f} bps")
    print()
    total = s.sum()
    print(f"  share of total gross edge from the top 10 days: "
          f"{top_days.head(10).sum() / total:.1%}")
    print(f"  days contributing a loss: {(s.groupby(s.index.date).mean() < 0).sum()} / {len(top_days)}")


if __name__ == "__main__":
    main()
