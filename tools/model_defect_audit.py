"""Defect audit: is the model's output actually signal, or is it volatility?

The forecaster trains a heteroscedastic head that predicts a mean and a log
variance, and scores with a Gaussian NLL. The evaluation path then divides the
predicted mean by *trailing realised* volatility and treats the result as a
probability. Two questions follow, and both are answerable from the frozen OOS
predictions without retraining anything:

  1. Does the predicted mean carry information beyond volatility? If the mean
     is largely a rescaled volatility forecast, dividing it back out leaves a
     constant and the ranking is noise.
  2. Does ranking by the score beat ranking by volatility alone, or by nothing?

Everything here is a measurement on the existing test-window predictions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from edge_economics import TOP_N, block_bootstrap_ci

from cqip.config import Settings
from cqip.storage import ParquetStore


def _cs_rank_corr(frame: pd.DataFrame, a: str, b: str) -> float:
    """Mean within-timestamp Spearman between two columns."""
    def one(g: pd.DataFrame) -> float:
        if len(g) < 3:
            return np.nan
        x, y = g[a].rank(), g[b].rank()
        if x.std() == 0 or y.std() == 0:
            return np.nan
        return float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std()))
    return float(frame.groupby("ts", group_keys=False).apply(one, include_groups=False).mean())


def main() -> None:
    store = ParquetStore(settings=Settings())
    preds = pd.read_parquet("data/artifacts/preds_lstm_1h.parquet")
    symbols = sorted(preds["symbol"].unique())

    ohlc = store.read("ohlcv", filters=[("interval", "==", "1h")], symbols=symbols)
    ohlc["ts"] = pd.DatetimeIndex(ohlc["ts"])
    close = ohlc.pivot_table(index="ts", columns="symbol", values="close").sort_index().ffill()
    fwd = (close.shift(-1) / close - 1.0)
    realised = fwd.rename_axis("ts").reset_index().melt(
        id_vars="ts", var_name="symbol", value_name="realised")

    df = preds.merge(realised, on=["ts", "symbol"], how="inner").dropna(
        subset=["realised", "predicted_return", "target_volatility", "score"])

    print("=" * 100)
    print("1. IS THE PREDICTED MEAN ACTUALLY A RETURN FORECAST?")
    print("=" * 100)
    pr, tv, tr = (df["predicted_return"], df["target_volatility"], df["target_return"])
    print(f"  std(predicted_return)   : {pr.std():.6f}   ({pr.std() * 1e4:.2f} bps)")
    print(f"  std(target_return)      : {tr.std():.6f}   ({tr.std() * 1e4:.2f} bps)")
    print(f"  std(target_volatility)  : {tv.std():.6f}   ({tv.std() * 1e4:.2f} bps)")
    print(f"  mean(predicted_return)  : {pr.mean():+.6f}")
    print(f"  mean(target_return)     : {tr.mean():+.6f}")
    print()
    print(f"  corr(predicted_return, target_volatility) pooled     : {pr.corr(tv):+.4f}")
    print(f"  corr(predicted_return, target_return)       pooled     : {pr.corr(tr):+.4f}")
    # How much of the predicted mean is explained by volatility alone?
    beta, alpha = np.polyfit(tv, pr, 1)
    resid = pr - (alpha + beta * tv)
    r2 = 1.0 - resid.var() / pr.var()
    print(f"  R^2 of predicted_return ~ target_volatility            : {r2:+.4f}")
    print(f"      (fitted slope {beta:+.4f}, intercept {alpha:+.6f})")
    print()
    print(f"  cross-sectional rank corr(predicted_return, vol)       : "
          f"{_cs_rank_corr(df, 'predicted_return', 'target_volatility'):+.4f}")
    print(f"  cross-sectional rank corr(score, vol)                  : "
          f"{_cs_rank_corr(df, 'score', 'target_volatility'):+.4f}")
    print()
    print("  Interpretation: an R^2 near 1.0 would mean the regression has learned")
    print("  'how big is this asset's move' rather than 'which way does it go'.")
    print()

    print("=" * 100)
    print("2. DOES THE SCORE'S CROSS-SECTIONAL DISPERSION SURVIVE?")
    print("=" * 100)
    print(f"  {'column':<20}{'pooled std':>13}{'mean CS std':>14}{'mean CS range':>15}")
    for col in ("predicted_return", "score", "target_volatility"):
        g = df.groupby("ts")[col]
        cs_std = g.std().mean()
        cs_range = (g.max() - g.min()).mean()
        print(f"  {col:<20}{df[col].std():>13.5f}{cs_std:>14.5f}{cs_range:>15.5f}")
    print()
    ratio = df.groupby("ts")["score"].std().mean() / df["score"].std()
    print(f"  within-timestamp score dispersion / pooled dispersion: {ratio:.3f}")
    print("  A value near 1.0 means the ranking is driven by within-timestamp")
    print("  relative ordering. A value near 0 means timestamps are separable")
    print("  but symbols inside a timestamp are not, which makes top-3/bottom-3")
    print("  selection close to random.")
    print()

    print("=" * 100)
    print("3. DOES THE MODEL BEAT NO INFORMATION AT ALL?")
    print("=" * 100)

    def spread(frame: pd.DataFrame, col: str) -> float:
        r = frame.sort_values(col, ascending=False)
        return float(r.head(TOP_N)["realised"].mean() - r.tail(TOP_N)["realised"].mean())

    candidates = {
        "score (what is traded)": "score",
        "predicted_return (raw mean)": "predicted_return",
        "target_volatility only": "target_volatility",
        "-target_volatility (low vol)": "neg_vol",
    }
    df["neg_vol"] = -df["target_volatility"]

    print(f"  {'ranking basis':<30}{'gross bps/bar':>15}{'95% CI':>20}{'hit rate':>11}")
    print("  " + "-" * 74)
    for label, col in candidates.items():
        s = df.groupby("ts", group_keys=False).apply(
            lambda g, c=col: spread(g, c), include_groups=False).dropna().to_numpy() * 1e4
        lo, hi = block_bootstrap_ci(s)
        print(f"  {label:<30}{s.mean():>15.3f}{f'[{lo:+.2f}, {hi:+.2f}]':>20}"
              f"{(s > 0).mean():>11.1%}")
    print()
    rng = np.random.default_rng(0)
    real = (df.groupby("ts", group_keys=False)
              .apply(lambda g: spread(g, "score"), include_groups=False)
              .dropna().to_numpy() * 1e4)
    # Permute the score *within* each timestamp to build a null that preserves
    # the cross-sectional size distribution but destroys the ordering.
    null_spreads = np.empty(len(real))
    for i, idx in enumerate(df.groupby("ts").indices.values()):
        block = df.iloc[idx]
        shuffled = block["score"].to_numpy().copy()
        rng.shuffle(shuffled)
        tmp = block.copy()
        tmp["score"] = shuffled
        null_spreads[i] = spread(tmp, "score") * 1e4
    print(f"  {'permuted score (null)':<30}{null_spreads.mean():>15.3f}"
          f"{'':>20}{(null_spreads > 0).mean():>11.1%}")
    print()
    print(f"  observed {real.mean():+.3f} bps vs permuted null {null_spreads.mean():+.3f} bps")
    # Empirical p-value from the permutation distribution. This avoids the
    # effective-sample-size shortcut, which is what made an earlier run of this
    # audit report a standard error ~5x too large.
    n_ge = int((null_spreads >= real.mean()).sum())
    p_value = (n_ge + 1) / (len(null_spreads) + 1)
    print(f"  permutation test: {n_ge} of {len(null_spreads):,} null spreads "
          f"reach the observed value")
    print(f"  one-sided p = {p_value:.4f}  "
          f"({'signal' if p_value < 0.05 else 'NOT distinguishable from noise'})")
    print()

    print("=" * 100)
    print("4. IS THE PROBABILITY LAYER SOUND?")
    print("=" * 100)
    from scipy.stats import norm
    claimed = norm.cdf(df["score"].to_numpy())
    labels = (df["realised"] > 0).astype(float).to_numpy()
    print(f"  claimed probability via Phi(score): mean {claimed.mean():.4f}")
    print(f"  observed positive-return frequency : {labels.mean():.4f}")
    print()
    # Proper binned ECE: within each confidence bucket, compare the mean claimed
    # probability to the observed frequency. Comparing a probability directly to
    # a 0/1 outcome yields ~0.5 by construction and measures nothing.
    edges = np.quantile(claimed, np.linspace(0.0, 1.0, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    bucket = np.clip(np.digitize(claimed, edges[1:-1]), 0, 9)
    ece = 0.0
    print(f"  {'bucket':>8}{'n':>9}{'mean claimed':>15}{'observed':>11}{'|gap|':>9}")
    for b in range(10):
        m = bucket == b
        if not m.any():
            continue
        c, o = float(claimed[m].mean()), float(labels[m].mean())
        ece += m.mean() * abs(c - o)
        print(f"  {b:>8}{int(m.sum()):>9}{c:>15.4f}{o:>11.4f}{abs(c - o):>9.4f}")
    print(f"  {'ECE':>8}{'':>9}{'':>15}{'':>11}{ece:>9.4f}")
    print()
    print("  Phi(score) is a plug-in normal transform of a variance-matched z-score.")
    print("  It is not a fitted probability. Calibration requires an isotonic or")
    print("  Platt map fitted on validation data only; none exists in this pipeline.")

    print()
    print("=" * 100)
    print("5. SUMMARY OF DEFECTS FOUND")
    print("=" * 100)
    findings = []
    if abs(pr.corr(tv)) > 0.5:
        findings.append(
            f"predicted mean correlates {pr.corr(tv):+.2f} with trailing volatility")
    if ratio < 0.35:
        findings.append(
            f"within-timestamp score dispersion is only {ratio:.2f} of pooled "
            f"dispersion, so top-3 selection is near-random")
    if ece > 0.05:
        findings.append(f"probability layer is uncalibrated (ECE {ece:.3f})")
    findings.append("model's own log_var head is trained but discarded in favour of "
                    "trailing realised volatility, so a learned variance forecast is "
                    "never used")
    for f in findings:
        print(f"  - {f}")
    if not findings:
        print("  none material")


if __name__ == "__main__":
    main()
