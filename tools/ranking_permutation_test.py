"""Apples-to-apples permutation test for every ranking basis.

The day-block bootstrap CI on the observed spread tests "is the mean spread
non-zero", which is the wrong null: a positive mean can arise from the score's
dispersion interacting with the return distribution, with no ordering skill at
all. The permutation null asks the right question -- if the scores were assigned
to symbols at random while keeping their per-timestamp distribution, how large a
spread would we see? -- and it is applied identically to every candidate basis.

Whatever survives this test is a genuine ranking effect.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from edge_economics import TOP_N

from cqip.config import Settings
from cqip.storage import ParquetStore


def top_minus_bottom(frame: pd.DataFrame, col: str) -> float:
    r = frame.sort_values(col, ascending=False)
    return float(r.head(TOP_N)["realised"].mean() - r.tail(TOP_N)["realised"].mean())


def paired_block_bootstrap(d: np.ndarray, block: int = 24, n_boot: int = 4000,
                          seed: int = 3) -> tuple[float, float, float]:
    """Two-sided p and CI for mean(d) under a moving-block bootstrap.

    The statistic is the *mean* of the paired difference between the observed
    and permuted spread at each timestamp. Comparing that mean against the
    distribution of individual spreads is a category error: the per-timestamp
    spread has a standard deviation of tens of bps, so almost any mean lands
    near its median and the test degenerates to always returning p ~ 0.5.
    """
    n = len(d)
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        means[i] = np.concatenate([d[s:s + block] for s in starts])[:n].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return float(lo), float(hi), float(min(1.0, p))


def main() -> None:
    store = ParquetStore(settings=Settings())
    preds = pd.read_parquet("data/artifacts/preds_lstm_1h.parquet")
    symbols = sorted(preds["symbol"].unique())
    ohlc = store.read("ohlcv", filters=[("interval", "==", "1h")], symbols=symbols)
    ohlc["ts"] = pd.DatetimeIndex(ohlc["ts"])
    close = ohlc.pivot_table(index="ts", columns="symbol", values="close").sort_index().ffill()
    fwd = close.shift(-1) / close - 1.0
    df = preds.merge(
        fwd.rename_axis("ts").reset_index().melt(
            id_vars="ts", var_name="symbol", value_name="realised"),
        on=["ts", "symbol"], how="inner",
    ).dropna(subset=["realised", "score", "predicted_return", "target_volatility"])

    df["neg_vol"] = -df["target_volatility"]

    bases = {
        "score (traded)": "score",
        "predicted_return": "predicted_return",
        "target_volatility": "target_volatility",
        "-target_volatility": "neg_vol",
    }

    blocks = list(df.groupby("ts"))

    print("=" * 100)
    print("PERMUTATION TEST: is each ranking better than a random assignment")
    print("=" * 100)
    print(f"  {len(blocks):,} timestamps, {len(symbols)} symbols, "
          f"{TOP_N} long vs {TOP_N} short")
    print("  Null: the basis is shuffled across symbols within each timestamp,")
    print("  preserving its distribution. Statistic: mean paired difference,")
    print("  day-block bootstrap.")
    print()

    rng = np.random.default_rng(11)
    results = {}
    for label, col in bases.items():
        observed = np.array([top_minus_bottom(g, col) for _, g in blocks]) * 1e4
        null = np.empty(len(observed))
        for i, (_, g) in enumerate(blocks):
            vals = g[col].to_numpy().copy()
            rng.shuffle(vals)
            tmp = g.copy()
            tmp[col] = vals
            null[i] = top_minus_bottom(tmp, col) * 1e4

        d = observed - null
        lo, hi, p = paired_block_bootstrap(d)
        results[label] = (observed.mean(), null.mean(), p, lo, hi)
        print(f"  {label:<22} observed {observed.mean():+7.3f}   "
              f"null {null.mean():+7.3f}   diff {d.mean():+7.3f}   "
              f"CI [{lo:+.2f}, {hi:+.2f}]  p={p:.4f}  "
              f"{'REAL' if p < 0.05 else 'noise'}")

    print()
    print("=" * 100)
    print("READING THIS")
    print("=" * 100)
    score_mean, _, score_p, _, _ = results["score (traded)"]
    vol_mean, _, vol_p, vol_lo, vol_hi = results["target_volatility"]
    print(f"  The traded score: {score_mean:+.2f} bps/bar, p = {score_p:.4f}.")
    if score_p >= 0.05:
        print("  Its cross-sectional ordering is indistinguishable from a random")
        print("  assignment of the same score values to the same symbols.")
    print()
    print(f"  Trailing volatility: {vol_mean:+.2f} bps/bar, p = {vol_p:.4f}, "
          f"CI [{vol_lo:+.2f}, {vol_hi:+.2f}].")
    if vol_p < 0.05:
        print(f"  That is {vol_mean / score_mean:.1f}x the model's figure and it does")
        print("  survive the permutation test. It is a volatility-factor exposure,")
        print("  not a forecast, but it is the only ranking here that is real.")
    else:
        print("  It does not survive either.")
    print()

    print("=" * 100)
    print("WHY THE MODEL CANNOT EXPRESS A VOLATILITY FACTOR")
    print("=" * 100)
    def cs_corr(a: str, b: str) -> float:
        return float(df.groupby("ts").apply(
            lambda g: g[a].rank().corr(g[b].rank()), include_groups=False).mean())
    print(f"  cross-sectional rank corr(score, trailing volatility)        : "
          f"{cs_corr('score', 'target_volatility'):+.4f}")
    print(f"  cross-sectional rank corr(predicted_return, trailing vol)    : "
          f"{cs_corr('predicted_return', 'target_volatility'):+.4f}")
    print()
    print("  The score divides the predicted mean by trailing volatility, which")
    print("  cancels any volatility tilt the mean may have learned. A vol-neutral")
    print("  score is structurally incapable of expressing a vol factor, so this")
    print("  is a design defect rather than a tuning failure.")


if __name__ == "__main__":
    main()
