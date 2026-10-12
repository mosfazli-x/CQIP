"""A/B test: does enabling directional supervision fix the signal-learning defect?

The audit found that `one_year_sim.py` calls `trainer.train()` without
`direction_column` or `aux_direction_weight`, so the auxiliary BCE loss never
runs and the mean head is free to collapse to a constant. This script trains the
two variants on identical data with an identical seed and compares them on
identical OOS predictions.

Design choices, and why:

* One training event per variant on a single cutoff. Production refits every
  `refit_bars`, but a 720-bar fold yields only ~30 independent days, which cannot
  resolve an effect worth a couple of bps per bar. Training once and evaluating
  on the whole remaining history gives ~150 independent days. Both variants get
  identical treatment, so the comparison stays fair. This is a single-cutoff A/B,
  not a reproduction of the production refit loop, and is reported as such.
* The universe is pinned to the exact 25 symbols the production hourly run
  traded. Resolving "every USDT pair in the store" silently widens the
  cross-section to 211 names and changes the difficulty of the ranking test.
* Per-symbol models, matching production: `prepare_symbol` builds one symbol's
  frame, so the cross-section is reassembled only at evaluation time.
* `--mode probe` trains one symbol across a weight grid, because the Gaussian NLL
  gradient on the mean head is O(1/sigma) ~ 100 while the BCE gradient is O(1),
  so a small weight can be numerically irrelevant.

Lower loss is explicitly *not* treated as evidence of improvement: adding a BCE
term to a negative NLL makes the total more negative regardless of skill. The
decision statistic is the paired permutation test on OOS cross-sectional ordering,
read together with the replay through the production execution path.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from one_year_sim import build_position_matrix, prepare_symbol, run_portfolio  # noqa: E402

from cqip.config import Settings  # noqa: E402
from cqip.features import resolve_purge_embargo  # noqa: E402
from cqip.forecaster import (  # noqa: E402
    ForecasterTrainer,
    _forward_full,
    _member_model_info,
    model_has_direction_head,
)
from cqip.storage import ParquetStore  # noqa: E402

# The exact 25-symbol cross-section the production hourly run traded, so the A/B
# is "same data" in the strict sense and not a broader universe that would
# silently change the cross-sectional difficulty.
PRODUCTION_UNIVERSE = [
    "AAVE_USDT", "ADA_USDT", "ASTER_USDT", "BNB_USDT", "BTC_USDT", "DOGE_USDT",
    "ENA_USDT", "ETH_USDT", "FARTCOIN_USDT", "HYPE_USDT", "H_USDT", "LINK_USDT",
    "LTC_USDT", "NEAR_USDT", "PEPE_USDT", "PUMP_USDT", "SOL_USDT", "SUI_USDT",
    "TAO_USDT", "TRX_USDT", "UNI_USDT", "WLD_USDT", "XPL_USDT", "XRP_USDT",
    "ZEC_USDT",
]

TOP_N = 3
DAY_BLOCK = 24


@dataclass
class Variant:
    name: str
    direction_column: str | None
    aux_weight: float


VARIANTS = {
    "A_current": Variant("A_current", None, 0.0),
    "B_directional": Variant("B_directional", "target_direction", 0.0),  # weight from CLI
}


# --------------------------------------------------------------------------- #
# inference that keeps the auxiliary head and log_var
# --------------------------------------------------------------------------- #
def forward_all(model, trainer: ForecasterTrainer, frame: pd.DataFrame,
                feature_columns, info: dict, seq_len: int) -> dict[str, np.ndarray | None]:
    """Run the model and keep `log_var` and the direction logit.

    `ForecasterTrainer.predict` collapses the head to (mean, predicted_volatility)
    and discards the logit, so the auxiliary probability and the model's own
    Gaussian NLL cannot be measured through it. This mirrors its scaling and
    windowing exactly and returns all three heads. Instrumentation only: no
    library code is modified.
    """
    scaled = trainer._scale_frame(frame, list(feature_columns), _member_model_info(info))
    windowed = trainer._windows(scaled, seq_len)
    model.eval()
    means, log_vars, logits = [], [], []
    with torch.no_grad():
        for start in range(0, len(windowed), 256):
            chunk = torch.from_numpy(windowed[start:start + 256]).to(trainer.device)
            m, lv, lg = _forward_full(model, chunk)
            means.append(m.cpu().numpy())
            log_vars.append(lv.cpu().numpy())
            if lg is not None:
                logits.append(lg.cpu().numpy())
    if not means:
        return {"mean": np.array([]), "log_var": np.array([]), "logit": None}
    lg = np.concatenate(logits) if len(logits) == len(means) else None
    return {
        "mean": np.concatenate(means).astype(np.float64),
        "log_var": np.concatenate(log_vars).astype(np.float64),
        "logit": None if lg is None else lg.astype(np.float64),
    }


def aligned_frame(frame: pd.DataFrame, n_pred: int) -> pd.DataFrame:
    """Rows of `frame` that produced the predictions, in the same order."""
    return frame.iloc[len(frame) - n_pred:].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def paired_block_bootstrap(d: np.ndarray, block: int = DAY_BLOCK, n_boot: int = 4000,
                           seed: int = 3) -> tuple[float, float, float]:
    n = len(d)
    # Fewer than one full block means there is no block structure to resample, and
    # the loop below would concatenate an empty list. Report "not measurable"
    # rather than raising, so a thin cross-section is visible in the output
    # instead of aborting the whole run.
    if n == 0 or n < block:
        return float("nan"), float("nan"), float("nan")
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        means[i] = np.concatenate([d[s:s + block] for s in starts])[:n].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return float(lo), float(hi), float(min(1.0, p))


def _top_minus_bottom(group: pd.DataFrame, col: str) -> float:
    # Average ranks, so exact ties are resolved symmetrically instead of by
    # whatever order the rows happened to arrive in.
    ranked = group.dropna(subset=[col, "realised"])
    # Fewer names than 2*TOP_N makes head and tail the same rows, so the spread
    # would be identically zero rather than merely small.
    if len(ranked) < 2 * TOP_N:
        return np.nan
    ranked = ranked.sort_values(col, ascending=False, kind="mergesort")
    head, tail = ranked.head(TOP_N), ranked.tail(TOP_N)
    if head[col].iloc[-1] == tail[col].iloc[0]:
        return np.nan  # degenerate overlap; skip rather than bias
    return float(head["realised"].mean() - tail["realised"].mean())


def within_ts_rank_corr(df: pd.DataFrame, col: str) -> tuple[float, float]:
    """Mean and standard error of the within-timestamp Spearman rho."""
    vals = []
    for _, g in df.groupby("ts"):
        if len(g) < 3:
            continue
        a, b = g[col].rank(), g["realised"].rank()
        if a.std() == 0 or b.std() == 0:
            continue
        vals.append(float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std())))
    arr = np.asarray(vals, dtype=float)
    # Day-block bootstrap over timestamps: adjacent bars share forward windows.
    n = len(arr)
    # A cross-section too thin to rank contributes no usable blocks, so there is
    # nothing to correlate and nothing to resample.
    if n < DAY_BLOCK:
        return float("nan"), float("nan")
    n_blocks = int(np.ceil(n / DAY_BLOCK))
    rng = np.random.default_rng(5)
    means = np.empty(2000)
    for i in range(2000):
        starts = rng.integers(0, n - DAY_BLOCK + 1, size=n_blocks)
        means[i] = np.concatenate([arr[s:s + DAY_BLOCK] for s in starts])[:n].mean()
    return float(arr.mean()), float(means.std())


def permutation_p(df: pd.DataFrame, col: str, seed: int = 11) -> dict[str, float]:
    """Paired permutation test on the cross-sectional top3-minus-bottom3 spread."""
    blocks = list(df.groupby("ts"))
    observed = np.array([_top_minus_bottom(g, col) for _, g in blocks], dtype=float)
    rng = np.random.default_rng(seed)
    null = np.empty(len(observed))
    for i, (_, g) in enumerate(blocks):
        vals = g[col].to_numpy().copy()
        rng.shuffle(vals)
        tmp = g.copy()
        tmp[col] = vals
        null[i] = _top_minus_bottom(tmp, col)
    ok = np.isfinite(observed) & np.isfinite(null)
    # No timestamp had enough names for a non-degenerate head/tail split, so there
    # is no spread to test. Report that explicitly rather than averaging an empty
    # selection, which yields a bare NaN with no indication of why.
    if ok.sum() == 0:
        return {"observed_bps": float("nan"), "null_bps": float("nan"),
                "diff_bps": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p": float("nan"), "n_days": 0}
    d = (observed[ok] - null[ok]) * 1e4
    lo, hi, p = paired_block_bootstrap(d)
    return {
        "observed_bps": float(observed[ok].mean() * 1e4),
        "null_bps": float(null[ok].mean() * 1e4),
        "diff_bps": float(d.mean()),
        "ci_lo": lo, "ci_hi": hi, "p": p,
        "n_days": int(ok.sum()),
    }


def balanced_accuracy(pred_up: np.ndarray, actual_up: np.ndarray) -> float:
    pos = actual_up == 1
    neg = ~pos
    tpr = float((pred_up[pos] == 1).mean()) if pos.any() else np.nan
    tnr = float((pred_up[neg] == 0).mean()) if neg.any() else np.nan
    return 0.5 * (tpr + tnr)


def calibration_table(prob: np.ndarray, actual_up: np.ndarray, bins: int = 10) -> dict:
    edges = np.quantile(prob, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bucket = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    rows = []
    for b in range(bins):
        m = bucket == b
        if not m.any():
            continue
        c, o = float(prob[m].mean()), float(actual_up[m].mean())
        ece += m.mean() * abs(c - o)
        rows.append((b, int(m.sum()), c, o, abs(c - o)))
    return {"ece": float(ece), "buckets": rows}


def gaussian_nll(pred: np.ndarray, target: np.ndarray, sigma: np.ndarray) -> float:
    """Mean Gaussian NLL at a fixed sigma. Comparable across variants because the
    sigma entering the formula is identical for both."""
    s = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, np.nan)
    ok = np.isfinite(pred) & np.isfinite(target) & np.isfinite(s)
    if ok.sum() < 10:
        return float("nan")
    z = (target[ok] - pred[ok]) / s[ok]
    return float(np.mean(0.5 * np.log(2 * np.pi * s[ok] ** 2) + 0.5 * z ** 2))


def own_objective_nll(pred: np.ndarray, target: np.ndarray,
                      predicted_vol: np.ndarray) -> float:
    """Gaussian NLL using each model's own predicted sigma, i.e. the quantity it
    actually trained on. Not comparable across models: a model can lower it by
    shrinking its predicted sigma, which is exactly the collapse being tested."""
    v = np.where(np.isfinite(predicted_vol) & (predicted_vol > 1e-12),
                 predicted_vol, np.nan)
    ok = np.isfinite(pred) & np.isfinite(target) & np.isfinite(v)
    if ok.sum() < 10:
        return float("nan")
    z = (target[ok] - pred[ok]) / v[ok]
    return float(np.mean(0.5 * np.log(2 * np.pi * v[ok] ** 2) + 0.5 * z ** 2))


def direction_bce(prob: np.ndarray, actual_up: np.ndarray) -> float:
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    y = actual_up.astype(float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(score: np.ndarray, label: np.ndarray) -> float:
    ok = np.isfinite(score)
    s, y = score[ok], label[ok].astype(bool)
    if len(np.unique(y)) < 2:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def invariants(df: pd.DataFrame, metas: list[dict]) -> dict:
    """Cheap structural checks that must hold before any metric is believed."""
    out: dict[str, object] = {}
    out["rows"] = int(len(df))
    out["duplicate_ts_symbol"] = int(df.duplicated(["ts", "symbol"]).sum())
    out["monotonic_per_symbol"] = bool(
        df.sort_values(["symbol", "ts"]).groupby("symbol")["ts"].is_monotonic_increasing.all())
    train_end = max(pd.Timestamp(m["train_end_ts"]) for m in metas if "train_end_ts" in m)
    out["oos_starts_after_train"] = bool(pd.Timestamp(df["ts"].min()) > train_end)
    out["last_train_ts"] = str(train_end)
    out["oos_start_ts"] = str(pd.Timestamp(df["ts"].min()))
    # target_return and the independently recomputed realised return must agree;
    # if they drift apart, the fill alignment is off by some number of bars.
    out["max_target_vs_realised_gap"] = float((df["target_return"] - df["realised"]).abs().max())
    finite = np.isfinite(df["score"]) & np.isfinite(df["target_volatility"])
    if "vol_scale" in df:
        out["score_consistent"] = bool(np.allclose(
            df.loc[finite, "score"].to_numpy(),
            (df.loc[finite, "predicted_return"]
             / (df.loc[finite, "target_volatility"] * df.loc[finite, "vol_scale"])).to_numpy(),
            rtol=1e-9, atol=1e-12))
    out["rows_per_symbol_min"] = int(df.groupby("symbol").size().min())
    out["rows_per_symbol_max"] = int(df.groupby("symbol").size().max())
    out["ties_in_score"] = int(len(df) - df["score"].nunique())
    out["ties_at_selection_boundary"] = int(
        sum(1 for _, g in df.groupby("ts") if len(g) >= 2 * TOP_N and
            np.isfinite(g["score"]).sum() >= 2 * TOP_N
            and g["score"].sort_values(ascending=False).to_numpy()[TOP_N - 1]
            == g["score"].sort_values(ascending=False).to_numpy()[-TOP_N]))
    return out


def evaluate(df: pd.DataFrame, label: str) -> dict:
    out: dict[str, object] = {"label": label}

    pr = df["predicted_return"].to_numpy(dtype=float)
    tr = df["target_return"].to_numpy(dtype=float)
    out["corr_pred_target"] = float(np.corrcoef(pr, tr)[0, 1])
    out["pred_mean_bps"] = float(pr.mean() * 1e4)
    out["pred_std_bps"] = float(pr.std() * 1e4)
    out["pred_p05_bps"] = float(np.quantile(pr, 0.05) * 1e4)
    out["pred_p95_bps"] = float(np.quantile(pr, 0.95) * 1e4)
    out["target_mean_bps"] = float(tr.mean() * 1e4)
    out["target_std_bps"] = float(tr.std() * 1e4)
    out["pred_vs_realised_drift_bps"] = out["pred_mean_bps"] - out["target_mean_bps"]

    for col in ("score", "predicted_return"):
        rho, se = within_ts_rank_corr(df, col)
        out[f"rank_corr_{col}"] = rho
        out[f"rank_corr_se_{col}"] = se
        out[f"perm_{col}"] = permutation_p(df, col)

    up_pred = (df["predicted_return"] > 0).to_numpy()
    up_act = (df["target_return"] > 0).to_numpy()
    out["directional_accuracy"] = float((up_pred == up_act).mean())
    out["balanced_accuracy"] = balanced_accuracy(up_pred, up_act)

    cal = calibration_table(df["prob"].to_numpy(dtype=float), up_act.astype(float))
    out["ece"] = cal["ece"]
    out["cal_buckets"] = cal["buckets"]

    out["oos_nll_shared_sigma"] = gaussian_nll(pr, tr, df["target_volatility"].to_numpy(float))
    if "predicted_volatility" in df:
        pv = df["predicted_volatility"].to_numpy(dtype=float)
        out["oos_nll_own_predicted_sigma"] = own_objective_nll(pr, tr, pv)
        out["predicted_vol_median"] = float(np.median(pv))
        out["oos_z_abs_mean"] = float(np.mean(np.abs((tr - pr) / np.where(pv > 1e-12, pv, np.nan))))
    if "direction_prob" in df:
        dp = df["direction_prob"].to_numpy(dtype=float)
        src = "aux_head"
    else:
        dp = df["prob"].to_numpy(dtype=float)
        src = "phi_score_fallback"
    out["direction_prob_source"] = src
    out["oos_direction_bce"] = direction_bce(dp, up_act)
    out["oos_direction_auc"] = auc(dp, up_act.astype(float))
    return out


def replay_execution(df: pd.DataFrame, store: ParquetStore, interval: str,
                     args) -> dict:
    """Push the variant's scores through the production selection and execution path.

    Replayed twice on the identical weight matrix: once at the production fee and
    slippage, once with both zeroed. The zero-cost row is the honest gross value,
    and the gap between them is the cost drag at this turnover.
    """
    by_symbol = {
        s: g[["ts", "score", "predicted_return"]].sort_values("ts")
        for s, g in df.groupby("symbol")
    }
    weights, _probs = build_position_matrix(by_symbol, top_k=args.top_k,
                                            prob_bar=args.prob_bar)
    out = run_portfolio(
        store, sorted(by_symbol), interval, weights,
        initial_capital=args.capital, fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        rebalance_band_bps=args.rebalance_band_bps,
        min_hold_bars=args.min_hold_bars,
        max_fills_per_day=args.max_fills_per_day,
        gate_mode=args.gate_mode,
    )
    gross = run_portfolio(
        store, sorted(by_symbol), interval, weights,
        initial_capital=args.capital, fee_bps=0.0, slippage_bps=0.0,
        rebalance_band_bps=args.rebalance_band_bps,
        min_hold_bars=args.min_hold_bars,
        max_fills_per_day=args.max_fills_per_day,
        gate_mode=args.gate_mode,
    )
    if "error" in out:
        return {"error": out["error"]}
    m = out["metrics"]
    gm = {} if "error" in gross else gross["metrics"]
    return {
        "net_return": m.get("total_return"),
        "gross_return_zero_cost": gm.get("total_return"),
        "cost_drag": (gm.get("total_return") or 0.0) - (m.get("total_return") or 0.0),
        "sharpe": m.get("sharpe"),
        "max_drawdown": m.get("max_drawdown"),
        "win_rate": m.get("win_rate"),
        "profit_factor": m.get("profit_factor"),
        "n_round_trips": out["n_round_trips"],
        "round_trips_per_day": out["n_round_trips"] / max(len(weights), 1) * 24,
        "total_cost": out["total_cost"],
        "turnover_x_initial": out["turnover_x_initial"],
        "restraint": out["restraint"],
        "exposure": out.get("exposure"),
        "mean_gross_exposure": out.get("mean_gross_exposure"),
        "max_gross_exposure": out.get("max_gross_exposure"),
        "active_bars": int((weights.abs().sum(axis=1) > 0).sum()),
        "total_bars": int(len(weights)),
    }


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def build_split(prep, gap: int, val_ratio: float, seq_len: int):
    """Reproduce the production train/val geometry for a single cutoff."""
    train_end = prep.test_from - gap
    val_len = max(120, int(train_end * val_ratio))
    val_start = train_end - val_len
    if val_start - seq_len < 300:
        val_start = seq_len + 300
        val_len = train_end - val_start
    if val_len < 60 or val_start - seq_len < 300:
        raise ValueError("insufficient history for the requested split")
    frame = prep.frame
    return (frame.iloc[:val_start].reset_index(drop=True),
            frame.iloc[val_start:train_end].reset_index(drop=True),
            frame.iloc[train_end:].reset_index(drop=True))


def train_variant(symbol: str, interval: str, variant: Variant, *, weight: float,
                  epochs: int, seed: int, val_ratio: float, seq_len: int,
                  cutoff: int | None) -> tuple[pd.DataFrame, dict]:
    prep = prepare_symbol(
        ParquetStore(settings=Settings()), symbol, interval,
        warmup_bars=cutoff if cutoff is not None else 1500,
        trade_bars=10**9, refit_bars=720, horizon=1, min_test_bars=20,
    )
    purge, embargo = resolve_purge_embargo(horizon=1, seq_len=seq_len, embargo_bars=0)
    gap = purge + embargo
    train, val, oos = build_split(prep, gap, val_ratio, seq_len)
    if len(oos) <= seq_len:
        return pd.DataFrame(), {"symbol": symbol, "error": "no OOS"}

    trainer = ForecasterTrainer(Settings())
    kw: dict[str, object] = {}
    if variant.direction_column is not None:
        kw["direction_column"] = variant.direction_column
        kw["aux_direction_weight"] = weight

    t0 = time.time()
    model, info = trainer.train(
        train, val, prep.feature_columns, target_column="target_return",
        seq_len=seq_len, horizon=1, kind="lstm", epochs=epochs, seed=seed, **kw,
    )
    fit_seconds = time.time() - t0

    # vol_scale is fitted on validation only and normalised by the mean
    # validation volatility, mirroring one_year_sim exactly.
    val_out = forward_all(model, trainer, val, prep.feature_columns, info, seq_len)
    val_aligned = aligned_frame(val, len(val_out["mean"]))
    val_vol = val_aligned["target_volatility"].to_numpy(dtype=float)
    val_score = val_out["mean"] / np.where(val_vol > 1e-9, val_vol, np.nan)
    val_target = val_aligned["target_return"].to_numpy(dtype=float)
    finite = np.isfinite(val_score) & np.isfinite(val_target) & np.isfinite(val_vol)
    if finite.sum() < 50:
        return pd.DataFrame(), {"symbol": symbol, "error": "too few finite val rows"}
    vs = float(np.std(val_target[finite] - val_score[finite] * val_vol[finite], ddof=1)
               / max(np.mean(val_vol[finite]), 1e-12))
    if not np.isfinite(vs) or vs <= 0:
        vs = 1.0
    vs = float(np.clip(vs, 0.2, 10.0))

    # Carry seq_len bars of history in front of the test slice so its first bar
    # has real history and no test bar is consumed by a training label.
    frame = prep.frame
    train_end = len(train) + len(val)
    predict_frame = frame.iloc[max(0, train_end - seq_len):].reset_index(drop=True)
    test = forward_all(model, trainer, predict_frame, prep.feature_columns, info, seq_len)
    aligned = aligned_frame(predict_frame, len(test["mean"]))
    raw_vol = aligned["target_volatility"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = test["mean"] / (np.where(raw_vol > 1e-9, raw_vol, np.nan) * vs)

    out = pd.DataFrame({
        "ts": predict_frame["ts"].to_numpy()[seq_len:],
        "symbol": symbol,
        "score": score,
        "predicted_return": test["mean"],
        "predicted_volatility": np.sqrt(np.exp(test["log_var"])),
        "target_return": aligned["target_return"].to_numpy(dtype=float),
        "target_volatility": raw_vol,
        "vol_scale": vs,
    })
    if test["logit"] is not None:
        out["direction_prob"] = 1.0 / (1.0 + np.exp(-np.clip(test["logit"], -30, 30)))

    # Same geometry, validation side, for the paired loss comparison.
    val_pred_vol = np.sqrt(np.exp(val_out["log_var"]))
    val_metrics = {
        "val_nll_shared_sigma": gaussian_nll(val_out["mean"], val_target, val_vol),
        "val_nll_own_predicted_sigma": own_objective_nll(
            val_out["mean"], val_target, val_pred_vol),
    }
    if val_out["logit"] is not None:
        val_metrics["val_direction_bce"] = direction_bce(
            1.0 / (1.0 + np.exp(-np.clip(val_out["logit"], -30, 30))),
            (val_target > 0).astype(float))
        val_metrics["val_direction_accuracy"] = float(
            ((val_out["logit"] > 0).astype(float) == (val_target > 0)).mean())

    meta = {
        "symbol": symbol,
        "train_bars": len(train),
        "val_bars": len(val),
        "oos_bars": len(oos),
        "vol_scale": round(vs, 4),
        "has_direction_head": model_has_direction_head(model),
        "train_end_ts": str(frame["ts"].iloc[train_end - 1]),
        "final_val_loss": float(info["metrics"].get("final_val_loss", float("nan"))),
        "val_direction_accuracy": info["metrics"].get("val_direction_accuracy"),
        "fit_seconds": round(fit_seconds, 1),
        **val_metrics,
    }
    return out, meta


def realised_frame(preds: pd.DataFrame, store: ParquetStore, interval: str) -> pd.DataFrame:
    """Attach the realised next-bar return from raw closes, independent of the model."""
    syms = sorted(preds["symbol"].unique())
    ohlc = store.read("ohlcv", filters=[("interval", "==", interval)], symbols=syms,
                      columns=["ts", "close", "symbol"])
    ohlc["ts"] = pd.DatetimeIndex(ohlc["ts"])
    close = ohlc.pivot_table(index="ts", columns="symbol", values="close").sort_index().ffill()
    # log of the RATIO, not of the simple return. log(ratio - 1) is undefined for
    # every down-bar, which silently deletes ~half the sample and leaves a frame
    # containing only up-moves.
    with np.errstate(divide="ignore", invalid="ignore"):
        realised = (np.log(close.shift(-1) / close).rename_axis("ts").reset_index()
                    .melt(id_vars="ts", var_name="symbol", value_name="realised"))
    out = preds.merge(realised, on=["ts", "symbol"], how="inner")
    cols = ["realised", "predicted_return", "predicted_volatility", "target_return",
            "target_volatility", "score"]
    have = [c for c in cols if c in out.columns]
    finite = np.ones(len(out), dtype=bool)
    for c in have:
        finite &= np.isfinite(out[c].to_numpy(dtype=float))
    out = out.loc[finite]
    if len(out) < len(preds):
        print(f"    dropped {len(preds) - len(out)} of {len(preds)} rows as non-finite",
              flush=True)
    # The traded probability is Phi(score) in production. It is recorded here only
    # to be measured; it is not treated as calibrated anywhere.
    out["prob"] = norm.cdf(out["score"].to_numpy(dtype=float))
    return out.reset_index(drop=True)


def mean_of(metas: list[dict], key: str) -> float:
    vals = [float(m[key]) for m in metas
            if key in m and m[key] is not None and np.isfinite(float(m[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def run_probe(args) -> None:
    print("PROBE: aux_direction_weight on a single symbol")
    print(f"  symbol {args.probe_symbol}, cutoff {args.cutoff}, seed {args.seed}")
    print()
    print(f"  {'weight':>8}{'dirhead':>9}{'corr(pred,target)':>21}{'pred mean bps':>16}"
          f"{'val loss':>12}{'fit s':>8}")
    print("  " + "-" * 74)
    store = ParquetStore(settings=Settings())
    for w in args.probe_weights:
        v = Variant("probe", "target_direction", w)
        preds, meta = train_variant(
            args.probe_symbol, args.interval, v, weight=w, epochs=args.epochs,
            seed=args.seed, val_ratio=args.val_ratio, seq_len=args.seq_len,
            cutoff=args.cutoff)
        if preds.empty:
            print(f"  {w:>8.2f}  {meta.get('error')}")
            continue
        d = realised_frame(preds, store, args.interval)
        c = float(np.corrcoef(d["predicted_return"], d["target_return"])[0, 1])
        print(f"  {w:>8.2f}{str(meta['has_direction_head']):>9}{c:>21.4f}"
              f"{d['predicted_return'].mean() * 1e4:>16.2f}"
              f"{meta['final_val_loss']:>12.4f}{meta['fit_seconds']:>8.0f}")


def run_ab(args) -> None:
    store = ParquetStore(settings=Settings())
    if args.all_symbols:
        syms = sorted(store.read(
            "ohlcv", filters=[("interval", "==", args.interval)],
            columns=["ts", "symbol"])["symbol"].unique())
        syms = [s for s in syms if s.endswith("_USDT") and not s.startswith("USD")]
    else:
        syms = list(PRODUCTION_UNIVERSE)
    print("A/B: directional supervision")
    print(f"  interval {args.interval}, cutoff {args.cutoff}, seed {args.seed}, "
          f"epochs {args.epochs}, aux weight {args.weight}")
    print(f"  universe {len(syms)} symbols"
          f"{' (all)' if args.all_symbols else ' (production 25)'}")
    print(f"  execution: top_k {args.top_k}, fee {args.fee_bps}bps, "
          f"slip {args.slippage_bps}bps, prob_bar {args.prob_bar}, "
          f"min_hold {args.min_hold_bars}, max_fills/day {args.max_fills_per_day}")
    print()

    results = {}
    args.out_predictions.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    for key, variant in (("A_current", VARIANTS["A_current"]),
                         ("B_directional", Variant("B_directional",
                                                   "target_direction", args.weight))):
        pieces, metas = [], []
        for i, s in enumerate(syms, 1):
            try:
                p, m = train_variant(
                    s, args.interval, variant,
                    weight=args.weight if variant.direction_column else 0.0,
                    epochs=args.epochs, seed=args.seed, val_ratio=args.val_ratio,
                    seq_len=args.seq_len, cutoff=args.cutoff)
            except Exception as exc:  # noqa: BLE001
                metas.append({"symbol": s, "error": str(exc)[:120]})
                continue
            if not p.empty:
                pieces.append(p)
            metas.append(m)
            print(f"  [{key}] {i:>2}/{len(syms)} {s:<14} "
                  f"train={m.get('train_bars')} oos={m.get('oos_bars')} "
                  f"val_loss={m.get('final_val_loss')} "
                  f"dirhead={m.get('has_direction_head')} {m.get('fit_seconds')}s",
                  flush=True)
        if not pieces:
            print(f"  [{key}] no usable predictions")
            continue
        preds = pd.concat(pieces, ignore_index=True)
        df = realised_frame(preds, store, args.interval)
        metrics = evaluate(df, key)
        metrics["invariants"] = invariants(df, metas)
        metrics["val_nll_shared_sigma"] = mean_of(metas, "val_nll_shared_sigma")
        metrics["val_nll_own_predicted_sigma"] = mean_of(metas, "val_nll_own_predicted_sigma")
        metrics["val_direction_bce"] = mean_of(metas, "val_direction_bce")
        metrics["reported_final_val_loss"] = mean_of(metas, "final_val_loss")
        metrics["symbols_with_direction_head"] = sum(
            1 for m in metas if m.get("has_direction_head"))
        metrics["execution"] = replay_execution(df, store, args.interval, args)
        results[key] = {"metrics": metrics, "metas": metas}
        df.to_parquet(args.out_predictions / f"ab_{key}.parquet", index=False)
        print(f"  [{key}] pooled rows {len(df):,}, timestamps {df['ts'].nunique():,}, "
              f"execution net {metrics['execution'].get('net_return')}")
        print(flush=True)

    report = {k: v["metrics"] for k, v in results.items()}
    if "A_current" in results and "B_directional" in results:
        a, b = results["A_current"]["metrics"], results["B_directional"]["metrics"]
        report["_delta_B_minus_A"] = {
            "corr_pred_target": b["corr_pred_target"] - a["corr_pred_target"],
            "rank_corr_score": b["rank_corr_score"] - a["rank_corr_score"],
            "perm_p_score": b["perm_score"]["p"],
            "perm_p_predicted_return": b["perm_predicted_return"]["p"],
            "oos_direction_auc": b["oos_direction_auc"] - a["oos_direction_auc"],
            "oos_nll_shared_sigma": b["oos_nll_shared_sigma"] - a["oos_nll_shared_sigma"],
            "pred_mean_bps": b["pred_mean_bps"] - a["pred_mean_bps"],
        }
    report["_metas"] = {k: v["metas"] for k, v in results.items()}
    report["_config"] = vars(args)
    args.out_report.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.out_report}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["probe", "ab"], default="ab")
    p.add_argument("--interval", default="1h")
    p.add_argument("--cutoff", type=int, default=5100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--weight", type=float, default=1.0)
    p.add_argument("--val-ratio", type=float, default=0.25)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--all-symbols", action="store_true",
                   help="use every USDT pair instead of the production 25")
    p.add_argument("--probe-symbol", default="BTC_USDT")
    p.add_argument("--probe-weights", type=float, nargs="+",
                   default=[0.0, 0.25, 1.0, 4.0])
    p.add_argument("--capital", type=float, default=1_000.0)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--prob-bar", type=float, default=0.0)
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--rebalance-band-bps", type=float, default=0.0)
    p.add_argument("--min-hold-bars", type=int, default=0)
    p.add_argument("--max-fills-per-day", type=int, default=None)
    p.add_argument("--gate-mode", choices=("flat", "hold"), default="flat")
    p.add_argument("--out-predictions", type=Path,
                   default=Path("data/artifacts/reports/ab"))
    p.add_argument("--out-report", type=Path,
                   default=Path("data/artifacts/reports/ab_direction_head.json"))
    args = p.parse_args()
    if args.mode == "probe":
        run_probe(args)
    else:
        run_ab(args)


if __name__ == "__main__":
    main()
