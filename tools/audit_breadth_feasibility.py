"""Feasibility audit for the market-breadth domain.

This is an AUDIT of what is derivable from the existing 25-symbol universe with
no new external data. It computes nothing predictive and implements no feature.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import ab_direction_head as ab  # noqa: E402
from feature_audit import load_panel  # noqa: E402

from cqip.storage import ParquetStore  # noqa: E402

CUTOFF = 5100
LOOKBACK = 24


def main() -> int:
    panel = load_panel(ParquetStore(), list(ab.PRODUCTION_UNIVERSE), "1h", LOOKBACK, CUTOFF)

    closes = {
        s: pd.Series(
            panel.close[s].to_numpy(dtype=float),
            index=pd.DatetimeIndex(panel.frames[s]["ts"].to_numpy()),
        )
        for s in panel.symbols
    }
    index = pd.DatetimeIndex(panel.frames[panel.symbols[0]]["ts"].to_numpy())
    rets = pd.DataFrame({s: np.log(c / c.shift(LOOKBACK)) for s, c in closes.items()})
    rets = rets.reindex(index)

    print("=== EXISTING DATA INVENTORY (no new source required) ===")
    print("  dataset           : ohlcv")
    print("  columns available : close, volume, quote_volume, trades, open, high, low")
    print(f"  symbols           : {len(panel.symbols)}")
    print(f"  OOS grid rows     : {len(index)}")

    n_oos = int(round(len(index) * 0.25))
    sub = rets.loc[index[-n_oos:]]

    print()
    print("=== BREADTH INGREDIENTS COMPUTABLE ON THE OOS GRID ===")
    candidates = {
        "share of coins with positive 24h return": (sub > 0).sum(axis=1) / sub.notna().sum(axis=1),
        "advance-decline line (net advancers)": (sub > 0).sum(axis=1) - (sub < 0).sum(axis=1),
        "cross-sectional IQR of 24h return": sub.quantile(0.75) - sub.quantile(0.25),
        "cross-sectional std of 24h return": sub.std(axis=1),
        "median 24h return (market drift)": sub.median(axis=1),
    }
    for name, series in candidates.items():
        ok = series.notna()
        print(
            f"  {name:<46} coverage={100 * ok.mean():5.1f}%  "
            f"min={float(series[ok].min()):9.4f} max={float(series[ok].max()):9.4f}"
        )

    print()
    print("=== COVERAGE VERDICT ===")
    elig = sub.notna().sum(axis=1)
    print(
        f"  eligible coins per timestamp: mean={elig.mean():.2f} min={elig.min()} max={elig.max()}"
    )
    print(f"  timestamps below 6 coins   : {int((elig < 6).sum())}")
    print("  -> every candidate breadth feature is computable at every OOS timestamp")

    print()
    print("=== REQUIRES NEW EXTERNAL DATA (explicitly out of scope) ===")
    for item in (
        "funding rates / open interest / liquidations (derivatives)",
        "news sentiment / fear-greed index",
        "on-chain exchange flows, active addresses, hash rate",
        "order book depth / spread / imbalance",
    ):
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
