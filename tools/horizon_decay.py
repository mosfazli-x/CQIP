"""Horizon-decay diagnostic: does the existing model carry information at ANY horizon?

Answers the interval question without training anything. The model was fit to a
1-step target on 1h bars, but its raw expected return can be compared against
realized forward returns at every candidate bar interval the product cares
about. If the correlation is flat-zero across all of them, no amount of
resampling will make this model tradeable and retraining is the only path. If it
peaks somewhere, that interval is where to spend the compute.

Nothing here is fitted, thresholded or selected on the test window: it is a
read-only measurement of an already-frozen model's output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from cqip.config import Settings
from cqip.storage import ParquetStore

BAR_MINUTES = {"15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240}
# Candidate horizons expressed in 1h bars. 15m/30m are *shorter* than the bar the
# model was fit on, so they cannot be observed from 1h data at all -- they are
# listed to make that absence explicit rather than silently omitted.
HORIZON_BARS = {"15m": 0.25, "30m": 0.5, "1h": 1, "2h": 2, "4h": 4}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC with tie correction."""
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    ranks_full = np.empty(len(s), dtype=float)
    ranks_full[order] = ranks
    n_pos = float(label.sum())
    n_neg = float(len(label) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks_full[label == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main() -> None:
    store = ParquetStore(settings=Settings())
    preds = pd.read_parquet("data/artifacts/preds_lstm_1h.parquet")
    symbols = sorted(preds["symbol"].unique())

    ohlc = store.read("ohlcv", filters=[("interval", "==", "1h")], symbols=symbols)
    ohlc["ts"] = pd.DatetimeIndex(ohlc["ts"])
    close = (
        ohlc.pivot_table(index="ts", columns="symbol", values="close")
        .sort_index()
        .ffill()
    )

    print("model      : hourly LSTM, 1-step target")
    print(f"symbols    : {len(symbols)}")
    print(f"OOS window : {preds['ts'].min()}  ->  {preds['ts'].max()}")
    print(f"bars       : {preds['ts'].nunique()} x {len(symbols)} = {len(preds)}")
    print()
    print("The model was fit on 1h bars, so 15m and 30m horizons cannot be observed")
    print("from this data. They are shown as NOT OBSERVABLE rather than estimated.")
    print()

    rows = []
    for interval, bars in HORIZON_BARS.items():
        if bars < 1:
            rows.append({
                "interval": interval, "bars": bars, "n": 0,
                "spearman": np.nan, "pearson": np.nan, "auc": np.nan,
                "note": "not observable from 1h bars",
            })
            continue
        k = int(bars)
        # Realized forward return over k bars, measured close-to-close.
        fwd = (close.shift(-k) / close - 1.0)
        merged = preds.merge(
            fwd.rename_axis("ts").reset_index().melt(
                id_vars="ts", var_name="symbol", value_name="realised"),
            on=["ts", "symbol"], how="inner",
        ).dropna(subset=["realised"])

        score = merged["predicted_return"].to_numpy(dtype=float)
        realised = merged["realised"].to_numpy(dtype=float)
        label = (realised > 0).astype(float)

        rows.append({
            "interval": interval, "bars": bars, "n": len(merged),
            "spearman": _spearman(score, realised),
            "pearson": float(np.corrcoef(score, realised)[0, 1]),
            "auc": _auc(score, label),
            "note": "",
        })

    frame = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:+.6f}"))
    print()

    observed = frame.dropna(subset=["auc"])
    best = observed.loc[observed["auc"].idxmax()]
    print(f"strongest horizon by AUC : {best['interval']}  (AUC {best['auc']:.4f})")
    print(f"weakest  horizon by AUC  : {observed.loc[observed['auc'].idxmin()]['interval']}")
    print(f"spread across horizons   : {observed['auc'].max() - observed['auc'].min():.4f}")
    print()

    # Is any of this distinguishable from a coin flip once serial correlation is
    # accounted for? Neighbouring 1h bars overlap heavily, so the effective sample
    # is far smaller than the row count.
    print("Serial-dependence-adjusted significance")
    print("-" * 78)
    eff_factor = 24.0  # one day of 1h bars assumed to carry ~1 independent draw
    for row in observed.itertuples():
        n_eff = row.n / eff_factor
        # Normal approximation to the AUC sampling distribution.
        q = row.auc * (1 - row.auc)
        se = float(np.sqrt(max(q, 1e-12) / max(n_eff, 1.0)))
        z = (row.auc - 0.5) / se
        verdict = "distinguishable from 0.5" if abs(z) > 1.96 else "NOT distinguishable from 0.5"
        print(f"  {row.interval:>4}: AUC {row.auc:.4f}  SE~{se:.4f}  z={z:+6.2f}  {verdict}")
    print()
    print("An AUC whose interval contains 0.5 cannot support a trade filter: the gate")
    print("would be discarding trades on noise.")


if __name__ == "__main__":
    main()
