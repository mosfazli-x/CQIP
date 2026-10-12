"""Single-fold label ablation: is the *target formulation* hiding the signal?

The corrected direction-head A/B (`tools/ab_direction_head.py`, reported in Â§5 of
`data/artifacts/reports/interval_feasibility.md`) established that the auxiliary
BCE head is mechanically sound: it is present on all 25 models, trains on every
symbol, and reports the base rate almost exactly (BCE 0.6937 ~ ln 2, ECE 0.0140,
direction AUC 0.502). The head is not broken. It is being asked a question its
inputs cannot answer, so this script changes the question and nothing else.

Three formulations, compared on identical data:

1. ``raw``        -- ``log(close.shift(-1) / close)``. The production baseline.
2. ``vol_scaled`` -- the same forward return divided by strictly trailing
   volatility, so the label is measured in units of the asset's own recent noise.
3. ``triple_bar`` -- volatility-scaled take-profit / stop-loss barriers with a
   fixed time barrier, scored as a classification with time exits censored.

What is held fixed, and why each one matters:

* **Architecture, optimiser, loss, features, symbol count, universe, epochs,
  seed, ``seq_len`` and split geometry are identical.** Only the supervised
  columns move. Every metric and invariant is imported from
  ``ab_direction_head`` rather than reimplemented, so the arms are scored by the
  same code that produced the accepted A/B and cannot silently drift from it.
* **One labeled frame per symbol, three arms trained off it.** The arms cannot
  diverge in row count because they are literally fitted from the same
  DataFrame. Recomputing the label per arm would let a longer warm-up quietly
  shorten one arm's history and hand it a different split.
* **Purge is widened to the longest label horizon** (the triple barrier's
  ``max_hold``, default 3 bars = 3h) and applied identically to every arm.
  ``resolve_purge_embargo(horizon=1, ...)`` is what the A/B used; it is
  insufficient here, because a barrier label at bar ``i`` reads prices up to
  ``i + max_hold``. Giving only the barrier arm the wider purge would be the
  exact look-ahead this experiment exists to avoid.
* **No silent row deletion.** ``ForecasterTrainer`` does not drop NaN targets, and
  the three labels have different warm-ups (``target_volatility`` 24 bars,
  the vol scaler 64, the barrier ``vol_window`` plus ``max_hold`` of future). The
  *intersection* of valid rows is taken once and applied to all three arms. NaN
  *direction* labels are deliberately excluded from that mask: the trainer masks
  them inside ``binary_direction_loss`` by design, and censoring triple-barrier
  time exits here as well would bias that arm's class balance by construction.
* **Logits, probabilities and predicted variance are preserved.**
  ``ForecasterTrainer.predict`` discards ``log_var`` and the direction logit, so
  the auxiliary probability and the model's own NLL are only measurable through
  the same instrumentation the A/B used.
* **A null is always measured.** The paired within-timestamp permutation test is
  joined by an explicit random-score control on the identical rows, so a small
  positive number cannot be mistaken for skill.
* **Uncertainty respects overlapping forward windows.** Day-block bootstrap and
  permutation, never a row-level bootstrap; adjacent bars share forward returns,
  so resampling rows would report CIs several times too narrow.

Reported per arm: AUC on both the arm's own label and a common next-bar outcome,
directional and balanced accuracy, within-timestamp rank correlation, permutation
p, class balance, ECE with reliability buckets, Brier, decile monotonicity, mean
and median forward outcome per decile, valid-observation counts and standard
errors. No trading parameter is searched anywhere. A fixed-configuration cost
diagnostic runs only for the arm with the strongest evidence, purely to check
whether any ranking gain pays for itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import ab_direction_head as ab  # noqa: E402  corrected metrics + invariants
from one_year_sim import build_position_matrix, prepare_symbol, run_portfolio  # noqa: E402

from cqip.config import Settings  # noqa: E402
from cqip.features import resolve_purge_embargo  # noqa: E402
from cqip.forecaster import ForecasterTrainer, model_has_direction_head  # noqa: E402
from cqip.labeling import (  # noqa: E402
    TripleBarrierConfig,
    add_volatility_normalized_target,
    directional_labels,
    make_triple_barrier_labels,
)
from cqip.storage import ParquetStore  # noqa: E402

TOP_N = ab.TOP_N
DAY_BLOCK = ab.DAY_BLOCK


@dataclass(frozen=True)
class Arm:
    """One label formulation.

    ``regression``/``direction`` name columns on the shared labeled frame.
    ``unscale`` names the column that converts the regression head's output back
    into a raw log return, or is ``None`` when the head already predicts one.
    """

    key: str
    title: str
    regression: str
    direction: str
    unscale: str | None


ARMS: dict[str, Arm] = {
    "raw": Arm("raw", "raw next-bar log return", "target_return", "lab_dir_raw", None),
    "vol_scaled": Arm(
        "vol_scaled",
        "forward return / trailing volatility",
        "target_scaled",
        "lab_dir_raw",
        "target_vol_scale",
    ),
    "triple_bar": Arm(
        "triple_bar",
        "vol-scaled triple barrier, censored time exits",
        "tb_return",
        "lab_dir_tb",
        None,
    ),
}


# --------------------------------------------------------------------------- #
# label construction
# --------------------------------------------------------------------------- #
def build_label_frame(
    frame: pd.DataFrame,
    *,
    max_hold: int,
    tp_vol: float,
    sl_vol: float,
    vol_window: int,
    tb_vol_window: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Materialise all three labels on one frame and take the valid-row intersection.

    Reuses the library's own builders so the experiment inherits production label
    semantics. ``build_target_frame`` is deliberately not called here: it derives
    the binary direction label as ``sign(tb_return)``, which keeps every bar and so
    quietly reduces the barrier problem back to the plain-return problem. The
    barrier arm instead supervises ``tb_label`` with time exits censored, which is
    what makes it a *classification* problem rather than a relabelled return.
    """
    out = add_volatility_normalized_target(
        frame, source="target_return", window=vol_window, horizon=1
    )
    cfg = TripleBarrierConfig(
        take_profit_vol=tp_vol, stop_loss_vol=sl_vol, max_hold=max_hold, vol_window=tb_vol_window
    )
    out = make_triple_barrier_labels(out, cfg, horizon=1, max_hold=max_hold)

    # Both return-based arms ask the same up/down question, so they share one
    # label column; writing it twice would suggest the vol arm changes the
    # classification target when it does not.
    out["lab_dir_raw"] = directional_labels(out["target_return"])
    tb = pd.to_numeric(out["tb_label"], errors="coerce")
    out["lab_dir_tb"] = pd.Series(
        np.where(tb > 0, 1.0, np.where(tb < 0, 0.0, np.nan)), index=out.index
    )

    reg_cols = [a.regression for a in ARMS.values()]
    keep = np.ones(len(out), dtype=bool)
    detail: dict[str, Any] = {}
    for col in reg_cols:
        keep &= np.isfinite(pd.to_numeric(out[col], errors="coerce").to_numpy(float))
    detail["rows_before_mask"] = int(len(out))
    detail["rows_dropped_by_mask"] = int((~keep).sum())
    detail["mask_reason"] = (
        "union of per-label warm-ups: target_volatility needs 24 "
        "bars, the vol scaler needs its window, and the barrier "
        "needs vol_window + max_hold bars of future"
    )

    out = out.loc[keep].reset_index(drop=True)
    detail["rows_after_mask"] = int(len(out))

    for key, arm in ARMS.items():
        r = pd.to_numeric(out[arm.regression], errors="coerce").to_numpy(float)
        d = pd.to_numeric(out[arm.direction], errors="coerce").to_numpy(float)
        finite_d = d[np.isfinite(d)]
        detail[key] = {
            "regression_column": arm.regression,
            "direction_column": arm.direction,
            "regression_mean_bps": float(np.mean(r) * 1e4),
            "regression_std_bps": float(np.std(r) * 1e4),
            "n_valid_regression": int(np.isfinite(r).sum()),
            "n_valid_direction": int(len(finite_d)),
            "direction_base_rate": float(finite_d.mean()) if len(finite_d) else float("nan"),
            "n_censored_direction": int(np.isnan(d).sum()),
        }
    exits = pd.to_numeric(out["tb_exit_bars"], errors="coerce")
    detail["triple_bar"]["time_exit_rate"] = float((tb == 0).mean())
    detail["triple_bar"]["mean_exit_bars"] = float(exits.mean())
    detail["triple_bar"]["censored_direction_rate"] = float(
        np.isnan(pd.to_numeric(out["lab_dir_tb"], errors="coerce").to_numpy(float)).mean()
    )
    return out, detail


# --------------------------------------------------------------------------- #
# per-symbol, per-arm training
# --------------------------------------------------------------------------- #
@dataclass
class _Frame:
    """Minimal stand-in exposing the fields `ab.build_split` reads."""

    frame: pd.DataFrame
    test_from: int


def fit_arm(
    symbol: str,
    labeled: pd.DataFrame,
    test_from: int,
    arm: Arm,
    *,
    feature_columns: list[str],
    epochs: int,
    seed: int,
    val_ratio: float,
    seq_len: int,
    purge_bars: int,
    aux_weight: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train one arm on the shared labeled frame and predict the OOS slice.

    The split is the production geometry (`ab.build_split`) driven off the
    *shared* frame and the *shared* `test_from`, widened by `purge_bars` for
    every arm alike.
    """
    gap = purge_bars
    train, val, oos = ab.build_split(
        _Frame(frame=labeled, test_from=test_from), gap, val_ratio, seq_len
    )
    if len(oos) <= seq_len:
        return pd.DataFrame(), {"symbol": symbol, "arm": arm.key, "error": "no OOS"}
    if not np.isfinite(train[arm.regression].to_numpy(dtype=float)).all():
        return pd.DataFrame(), {
            "symbol": symbol,
            "arm": arm.key,
            "error": "non-finite regression target reached the trainer",
        }

    trainer = ForecasterTrainer(Settings())
    t0 = time.time()
    model, info = trainer.train(
        train,
        val,
        feature_columns,
        target_column=arm.regression,
        direction_column=arm.direction,
        aux_direction_weight=aux_weight,
        seq_len=seq_len,
        horizon=1,
        kind="lstm",
        epochs=epochs,
        seed=seed,
    )
    fit_seconds = time.time() - t0

    # `vs` is production's validation-only volatility scale. It is fitted on
    # predictions already converted back to raw log-return units, otherwise the
    # vol-scaled arm would have its residuals divided by sigma twice.
    val_out = ab.forward_all(model, trainer, val, feature_columns, info, seq_len)
    val_al = ab.aligned_frame(val, len(val_out["mean"]))
    val_vol = val_al["target_volatility"].to_numpy(dtype=float)
    val_pred_raw = unscale_predictions(arm, val_out["mean"], val_al)
    val_target = val_al["target_return"].to_numpy(dtype=float)
    finite = np.isfinite(val_pred_raw) & np.isfinite(val_target) & np.isfinite(val_vol)
    if finite.sum() < 50:
        return pd.DataFrame(), {
            "symbol": symbol,
            "arm": arm.key,
            "error": "too few finite val rows",
        }
    resid = np.std(val_target[finite] - val_pred_raw[finite], ddof=1) / max(
        float(np.mean(val_vol[finite])), 1e-12
    )
    vs = float(np.clip(resid if np.isfinite(resid) and resid > 0 else 1.0, 0.2, 10.0))

    # Carry seq_len bars of history in front of the OOS slice so its first bar is
    # fully seeded and no OOS bar is consumed by a training label.
    frame = labeled
    train_end = len(train) + len(val)
    predict_frame = frame.iloc[max(0, train_end - seq_len) :].reset_index(drop=True)
    test = ab.forward_all(model, trainer, predict_frame, feature_columns, info, seq_len)
    aligned = ab.aligned_frame(predict_frame, len(test["mean"]))
    raw_vol = aligned["target_volatility"].to_numpy(dtype=float)
    pred_raw = unscale_predictions(arm, test["mean"], aligned)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = pred_raw / (np.where(raw_vol > 1e-9, raw_vol, np.nan) * vs)

    out = pd.DataFrame(
        {
            "ts": predict_frame["ts"].to_numpy()[seq_len:],
            "symbol": symbol,
            "arm": arm.key,
            "score": score,
            "predicted_return": pred_raw,
            "head_output": test["mean"],
            "predicted_volatility": np.sqrt(np.exp(test["log_var"])),
            "target_return": aligned["target_return"].to_numpy(dtype=float),
            "target_volatility": raw_vol,
            "vol_scale": vs,
            arm.regression: aligned[arm.regression].to_numpy(dtype=float),
            arm.direction: aligned[arm.direction].to_numpy(dtype=float),
            "tb_label": aligned["tb_label"].to_numpy(dtype=float),
            "tb_return": aligned["tb_return"].to_numpy(dtype=float),
            "tb_exit_bars": aligned["tb_exit_bars"].to_numpy(dtype=float),
        }
    )
    if test["logit"] is not None:
        out["direction_prob"] = 1.0 / (1.0 + np.exp(-np.clip(test["logit"], -30, 30)))

    val_pred_vol = np.sqrt(np.exp(val_out["log_var"]))
    # Validation-side NLL, on the validation predictions -- not the OOS ones.
    # Both are expressed in raw log-return units so the three arms are comparable.
    vm = {
        "val_nll_shared_sigma": ab.gaussian_nll(val_pred_raw, val_target, val_vol),
        "val_nll_own_predicted_sigma": ab.own_objective_nll(val_pred_raw, val_target, val_pred_vol),
    }
    if val_out["logit"] is not None:
        vp = 1.0 / (1.0 + np.exp(-np.clip(val_out["logit"], -30, 30)))
        y = aligned_label(arm, val_al)
        vm["val_direction_bce"] = ab.direction_bce(vp, y) if y is not None else float("nan")
        vm["val_direction_accuracy"] = (
            float(((val_out["logit"] > 0).astype(float) == y).mean())
            if y is not None
            else float("nan")
        )

    meta = {
        "symbol": symbol,
        "arm": arm.key,
        "train_bars": len(train),
        "val_bars": len(val),
        "oos_bars": len(oos),
        "vol_scale": round(vs, 4),
        "has_direction_head": model_has_direction_head(model),
        "train_end_ts": str(frame["ts"].iloc[train_end - 1]),
        "final_val_loss": float(info["metrics"].get("final_val_loss", float("nan"))),
        "fit_seconds": round(fit_seconds, 1),
        **vm,
    }
    return out, meta


def unscale_predictions(arm: Arm, head: np.ndarray, aligned: pd.DataFrame) -> np.ndarray:
    """Convert the regression head's output into a raw log return.

    Within-timestamp ranking is invariant to a positive scale factor, so this
    does not change any selection. It matters for the NLL figures and for
    reporting the head's output in tradeable units.
    """
    if arm.unscale is None:
        return np.asarray(head, dtype=float)
    scale = pd.to_numeric(aligned[arm.unscale], errors="coerce").to_numpy(dtype=float)
    return np.asarray(head, dtype=float) * scale


def aligned_label(arm: Arm, aligned: pd.DataFrame) -> np.ndarray | None:
    d = pd.to_numeric(aligned[arm.direction], errors="coerce").to_numpy(dtype=float)
    return d if np.isfinite(d).any() else None


# --------------------------------------------------------------------------- #
# extra metrics the A/B does not already provide
# --------------------------------------------------------------------------- #
def brier(prob: np.ndarray, label: np.ndarray) -> float:
    ok = np.isfinite(prob) & np.isfinite(label)
    return float(np.mean((prob[ok] - label[ok].astype(float)) ** 2)) if ok.any() else np.nan


def decile_profile(df: pd.DataFrame, score: str, outcome: str) -> dict[str, Any]:
    """Mean/median outcome and win rate per score decile, plus monotonicity.

    Monotonicity is the Spearman rho of decile index against decile mean outcome.
    A mean-only readout can look non-monotonic through noise alone; an edge that
    orders the deciles is a much stronger claim than one that merely lifts the
    top bucket.
    """
    cols = [c for c in (score, outcome) if c in df.columns]
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 10 * TOP_N:
        return {"buckets": [], "monotonicity": float("nan"), "note": "too few rows for deciles"}
    edges = np.quantile(d[score], np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    d = d.assign(bucket=np.clip(np.digitize(d[score], edges[1:-1]), 0, 9))
    rows = []
    for b, g in d.groupby("bucket"):
        rows.append(
            {
                "bucket": int(b),
                "n": int(len(g)),
                "mean_bps": float(g[outcome].mean() * 1e4),
                "median_bps": float(g[outcome].median() * 1e4),
                "win_rate": float((g[outcome] > 0).mean()),
                "mean_score": float(g[score].mean()),
            }
        )
    idx = np.array([r["bucket"] for r in rows], dtype=float)
    mn = np.array([r["mean_bps"] for r in rows], dtype=float)
    mono = float(spearmanr(idx, mn).statistic) if len(rows) > 2 else float("nan")
    top = d[d.bucket == d.bucket.max()][outcome].to_numpy(dtype=float) * 1e4
    bot = d[d.bucket == d.bucket.min()][outcome].to_numpy(dtype=float) * 1e4
    pad = max(0, len(bot) - len(top))
    padv = np.full(pad, np.nan)
    diff = top - np.concatenate([bot, padv])
    diff = diff[np.isfinite(diff)]
    _lo, _hi, p = (
        ab.paired_block_bootstrap(diff)
        if len(diff) >= DAY_BLOCK
        else (float("nan"), float("nan"), float("nan"))
    )
    return {
        "buckets": rows,
        "monotonicity": mono,
        "top_minus_bottom_bps": float(diff.mean()) if len(diff) else float("nan"),
        "top_minus_bottom_p": p,
    }


def boot_auc_ci(
    df: pd.DataFrame, score: str, label: str, *, draws: int = 2000, seed: int = 101
) -> tuple[float, float, float]:
    """AUC with a day-block bootstrap CI.

    Resampling rows would be wrong here: 25 symbols share each hour's market move
    and neighbouring bars share forward windows, so the row count overstates the
    evidence by roughly the size of a day. Blocks are whole days of the pooled
    cross-section, which keeps the within-timestamp dependence intact.
    """
    cols = [c for c in (score, label, "ts") if c in df.columns]
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(d) < 3 * DAY_BLOCK:
        return (
            ab.auc(d[score].to_numpy(float), d[label].to_numpy(float)),
            float("nan"),
            float("nan"),
        )
    order = np.argsort(d["ts"].to_numpy(), kind="mergesort")
    s = d[score].to_numpy(dtype=float)[order]
    y = d[label].to_numpy(dtype=float)[order]
    _days, inverse = np.unique(d["ts"].to_numpy()[order], return_inverse=True)
    blocks = [np.flatnonzero(inverse == i) for i in range(int(inverse.max()) + 1)]
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(draws):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), size=len(blocks))])
        v = ab.auc(s[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return ab.auc(s, y), float("nan"), float("nan")
    return (
        ab.auc(s, y),
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    )


def random_score_null(
    df: pd.DataFrame, score: str, *, draws: int = 25, seed: int = 41
) -> dict[str, float]:
    """Rank correlation and AUC under a random score on the identical rows.

    The permutation test already builds a within-timestamp null for the spread.
    This is the simpler global control, and reporting both guards against a
    harness bug that would corrupt either one.
    """
    rng = np.random.default_rng(seed)
    rhos, aucs = [], []
    up = (df["realised"] > 0).to_numpy(dtype=float)
    for _ in range(draws):
        noise = rng.normal(size=len(df))
        tmp = df.assign(_noise=noise)
        rho, _se = ab.within_ts_rank_corr(tmp, "_noise")
        rhos.append(rho)
        aucs.append(ab.auc(noise, up))
    return {
        "rank_corr_mean": float(np.mean(rhos)),
        "rank_corr_sd": float(np.std(rhos)),
        "fwd_auc_mean": float(np.nanmean(aucs)),
        "fwd_auc_sd": float(np.nanstd(aucs)),
        "draws": draws,
    }


def evaluate_arm(
    df: pd.DataFrame, arm: Arm, metas: list[dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    """All readouts for one arm, on one common row set.

    Two discrimination readouts are reported on purpose. The arm's *own* label
    asks "did this label become learnable?". The common next-bar outcome asks the
    question that actually matters -- "does this label expose structure that
    predicting the raw next-bar return does not?" -- and is the only readout
    directly comparable across arms.
    """
    base = ab.evaluate(df, f"label_ablation::{arm.key}")
    out: dict[str, Any] = {k: v for k, v in base.items() if k != "cal_buckets"}
    out["arm"] = arm.key
    out["title"] = arm.title
    out["regression_target"] = arm.regression
    out["direction_target"] = arm.direction

    prob = df["direction_prob"].to_numpy(dtype=float) if "direction_prob" in df.columns else None
    own = pd.to_numeric(df[arm.direction], errors="coerce").to_numpy(dtype=float)
    up_fwd = (df["target_return"] > 0).to_numpy(dtype=float)
    native = arm.regression

    out["n_rows"] = int(len(df))
    out["n_timestamps"] = int(df["ts"].nunique())
    out["n_symbols"] = int(df["symbol"].nunique())
    out["n_valid_own_label"] = int(np.isfinite(own).sum())
    out["own_label_base_rate"] = float(np.nanmean(own))
    out["n_censored_own_label"] = int(np.isnan(own).sum())
    out["next_bar_base_rate"] = float(np.nanmean(up_fwd))
    out["head_output_mean"] = float(df["head_output"].mean())
    out["head_output_std"] = float(df["head_output"].std())

    if prob is not None:
        ok = np.isfinite(own)
        out["own_auc"] = ab.auc(prob[ok], own[ok])
        out["own_brier"] = brier(prob, own)
        out["own_ece"] = ab.calibration_table(prob[ok], own[ok])["ece"]
        out["own_balanced_accuracy"] = ab.balanced_accuracy((prob > 0.5).astype(float), own)
        out["own_directional_accuracy"] = float(((prob > 0.5).astype(float) == own)[ok].mean())
        out["own_bce"] = ab.direction_bce(prob[ok], own[ok])
    out["fwd_auc_score"] = ab.auc(df["score"].to_numpy(dtype=float), up_fwd)
    out["fwd_brier_score"] = brier(df["prob"].to_numpy(dtype=float), up_fwd)
    out["fwd_ece_score"] = ab.calibration_table(df["prob"].to_numpy(dtype=float), up_fwd)["ece"]
    out["fwd_bce_score"] = ab.direction_bce(df["prob"].to_numpy(dtype=float), up_fwd)

    out["null_control"] = random_score_null(df, "score", seed=seed)
    out["deciles_next_bar"] = decile_profile(df, "score", "realised")
    out["deciles_native"] = decile_profile(df, "score", native)
    out["deciles_tb_return"] = (
        decile_profile(df, "score", "tb_return") if arm.key == "triple_bar" else None
    )

    out["invariants"] = label_invariants(df, metas, arm)
    return out


def label_invariants(df: pd.DataFrame, metas: list[dict[str, Any]], arm: Arm) -> dict[str, Any]:
    """Structural checks specific to this experiment, on top of `ab.invariants`."""
    inv = ab.invariants(df, metas)
    ok = np.isfinite(df[arm.regression].to_numpy(dtype=float))
    inv["regression_column"] = arm.regression
    inv["no_nan_regression_target"] = bool(ok.all())
    inv["no_silent_deletion"] = bool(ok.all())
    # Censored direction labels must survive into the frame: the trainer masks
    # them internally, and a harness that had already dropped them would leave no
    # trace that the censoring ever happened.
    inv["censored_direction_present"] = int(np.isnan(df[arm.direction].to_numpy(dtype=float)).sum())
    inv["score_is_finite"] = bool(np.isfinite(df["score"].to_numpy(dtype=float)).all())
    inv["pred_vol_positive"] = bool((df["predicted_volatility"].to_numpy(dtype=float) > 0).all())
    inv["direction_prob_in_range"] = bool(
        np.isfinite(df["direction_prob"]).all() if "direction_prob" in df.columns else False
    )
    if "direction_prob" in df.columns:
        p = df["direction_prob"].to_numpy(dtype=float)
        inv["direction_prob_in_range"] = bool(((p > 0) & (p < 1)).all())
    counts = df.groupby("ts")["symbol"].nunique()
    inv["min_cross_section"] = int(counts.min())
    inv["cross_section_at_least_2_topn"] = bool((counts >= 2 * TOP_N).all())
    inv["n_fit_errors"] = sum(1 for m in metas if "error" in m)
    return inv


# --------------------------------------------------------------------------- #
# economic diagnostic (best arm only, no search)
# --------------------------------------------------------------------------- #
def economic_diagnostic(
    df: pd.DataFrame, store: ParquetStore, interval: str, *, args, arm: str
) -> dict[str, Any]:
    """Fixed-configuration replay at realistic all-in costs.

    Nothing is tuned: fixed ``top_k``, no probability bar, no parameter sweep. The
    only question is whether a ranking gain, if one exists, survives costs. Costs
    are charged per fill, so a round trip pays twice.
    """
    by_symbol = {
        s: g[["ts", "score", "predicted_return"]].sort_values("ts") for s, g in df.groupby("symbol")
    }
    weights, _probs = build_position_matrix(by_symbol, top_k=args.top_k, prob_bar=args.prob_bar)
    # Half-spread is charged through slippage, since the engine takes only a fee
    # and a slippage argument. All-in per fill = fee + slippage + half-spread.
    all_in_slip = args.slippage_bps + args.spread_bps

    def run(fee: float, slip: float) -> dict[str, Any]:
        return run_portfolio(
            store,
            sorted(by_symbol),
            interval,
            weights,
            initial_capital=args.capital,
            fee_bps=fee,
            slippage_bps=slip,
            rebalance_band_bps=args.rebalance_band_bps,
            min_hold_bars=args.min_hold_bars,
            max_fills_per_day=args.max_fills_per_day,
            gate_mode=args.gate_mode,
        )

    net, gross, double = (
        run(args.fee_bps, all_in_slip),
        run(0.0, 0.0),
        run(args.fee_bps * 2, all_in_slip * 2),
    )
    if "error" in net:
        return {"arm": arm, "error": net["error"]}

    def summarise(r: dict[str, Any]) -> dict[str, Any]:
        m = r["metrics"]
        return {
            "total_return": m.get("total_return"),
            "sharpe": m.get("sharpe"),
            "max_drawdown": m.get("max_drawdown"),
            "win_rate": m.get("win_rate"),
            "profit_factor": m.get("profit_factor"),
            "n_round_trips": r["n_round_trips"],
            "total_cost": r["total_cost"],
            "turnover_x_initial": r["turnover_x_initial"],
        }

    n_rt = net["n_round_trips"]
    n_bars = max(len(weights), 1)
    net_ret = net["metrics"].get("total_return") or 0.0
    gross_ret = gross["metrics"].get("total_return") or 0.0
    return {
        "arm": arm,
        "net": summarise(net),
        "gross": summarise(gross),
        "double_cost": summarise(double),
        "cost_drag": gross_ret - net_ret,
        "all_in_cost_bps_per_fill": args.fee_bps + args.slippage_bps + args.spread_bps,
        "all_in_cost_bps_round_trip": 2 * (args.fee_bps + args.slippage_bps + args.spread_bps),
        "cost_per_round_trip": (net["total_cost"] / n_rt) if n_rt else float("nan"),
        "net_bps_per_round_trip": (net_ret / n_rt * 1e4) if n_rt else float("nan"),
        "gross_bps_per_round_trip": (gross_ret / n_rt * 1e4) if n_rt else float("nan"),
        "round_trips_per_day": n_rt / n_bars * 24,
        "mean_gross_exposure": net.get("mean_gross_exposure"),
        "note": "diagnostic only; no parameter was searched",
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--interval", default="1h")
    p.add_argument("--cutoff", type=int, default=5100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--aux-weight", type=float, default=1.0)
    p.add_argument("--val-ratio", type=float, default=0.25)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--vol-window", type=int, default=64)
    p.add_argument("--tb-vol-window", type=int, default=32)
    p.add_argument(
        "--max-hold", type=int, default=3, help="barrier time exit in bars; 3 = 3h on a 1h series"
    )
    p.add_argument("--tp-vol", type=float, default=1.0)
    p.add_argument("--sl-vol", type=float, default=1.0)
    p.add_argument("--symbols", default="", help="comma list; default is production 25")
    p.add_argument("--arms", default="raw,vol_scaled,triple_bar")
    p.add_argument("--economic-arm", default="", help="blank = strongest evidence")
    p.add_argument("--capital", type=float, default=1_000.0)
    p.add_argument("--top-k", type=int, default=TOP_N)
    p.add_argument("--prob-bar", type=float, default=0.0)
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument(
        "--spread-bps", type=float, default=3.0, help="half-spread per fill, charged via slippage"
    )
    p.add_argument("--rebalance-band-bps", type=float, default=0.0)
    p.add_argument("--min-hold-bars", type=int, default=0)
    p.add_argument("--max-fills-per-day", type=int, default=None)
    p.add_argument("--gate-mode", choices=("flat", "hold"), default="flat")
    p.add_argument(
        "--out-predictions", type=Path, default=Path("data/artifacts/reports/label_ablation")
    )
    p.add_argument(
        "--out-report", type=Path, default=Path("data/artifacts/reports/label_ablation.json")
    )
    args = p.parse_args()

    arm_keys = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arm_keys if a not in ARMS]
    if unknown:
        p.error(f"unknown arms {unknown}; choose from {sorted(ARMS)}")
    arms = [ARMS[a] for a in arm_keys]
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()] or list(ab.PRODUCTION_UNIVERSE)
    store = ParquetStore(settings=Settings())

    # Purge must cover the longest label horizon present in *any* arm and is
    # applied identically everywhere. resolve_purge_embargo's own default is the
    # label horizon, which is 1 for the return arms and max_hold for the barrier.
    purge, embargo = resolve_purge_embargo(
        horizon=args.max_hold, seq_len=args.seq_len, embargo_bars=0
    )
    gap = purge + embargo

    print("LABEL ABLATION -- single fold, everything but the target held fixed")
    print(
        f"  interval={args.interval} cutoff={args.cutoff} seed={args.seed} "
        f"epochs={args.epochs} seq_len={args.seq_len} aux_weight={args.aux_weight}"
    )
    print(f"  arms={[a.key for a in arms]}")
    print(
        f"  purge={purge} embargo={embargo} (widest label horizon = max_hold "
        f"{args.max_hold} bars, applied to every arm)"
    )
    print(
        f"  barrier: tp={args.tp_vol}vol sl={args.sl_vol}vol "
        f"max_hold={args.max_hold} tb_vol_window={args.tb_vol_window}; "
        f"vol_window={args.vol_window}"
    )
    print(f"  universe={len(syms)} symbols (pinned)")
    print(
        f"  costs/fill: fee {args.fee_bps} + slip {args.slippage_bps} + spread "
        f"{args.spread_bps} = {args.fee_bps + args.slippage_bps + args.spread_bps}bps"
    )
    print()

    args.out_predictions.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "split": {
            "purge": purge,
            "embargo": embargo,
            "gap": gap,
            "seq_len": args.seq_len,
            "val_ratio": args.val_ratio,
        },
        "label_detail": {},
        "errors": [],
    }
    pieces: dict[str, list[pd.DataFrame]] = {a.key: [] for a in arms}
    metas: dict[str, list[dict[str, Any]]] = {a.key: [] for a in arms}

    for i, sym in enumerate(syms, 1):
        try:
            prep = prepare_symbol(
                store,
                sym,
                args.interval,
                warmup_bars=args.cutoff,
                trade_bars=10**9,
                refit_bars=720,
                horizon=1,
                min_test_bars=20,
            )
        except Exception as exc:  # noqa: BLE001
            for a in arms:
                metas[a.key].append({"symbol": sym, "error": str(exc)[:200]})
                report["errors"].append(
                    {"symbol": sym, "arm": a.key, "stage": "prepare", "error": str(exc)[:200]}
                )
            print(f"  {i:>2}/{len(syms)} {sym:<14} PREPARE FAILED: {str(exc)[:80]}")
            continue

        labeled, detail = build_label_frame(
            prep.frame,
            max_hold=args.max_hold,
            tp_vol=args.tp_vol,
            sl_vol=args.sl_vol,
            vol_window=args.vol_window,
            tb_vol_window=args.tb_vol_window,
        )
        report["label_detail"][sym] = detail
        # The masked frame is shorter than prep.frame, so the OOS boundary has to
        # be re-expressed in masked indices before any arm can use it.
        masked_test_from = int(min(prep.test_from, len(labeled)))

        line = [f"  {i:>2}/{len(syms)} {sym:<14} rows {len(labeled):>5}"]
        for arm in arms:
            try:
                df_m, m = fit_arm(
                    sym,
                    labeled,
                    masked_test_from,
                    arm,
                    feature_columns=prep.feature_columns,
                    epochs=args.epochs,
                    seed=args.seed,
                    val_ratio=args.val_ratio,
                    seq_len=args.seq_len,
                    purge_bars=gap,
                    aux_weight=args.aux_weight,
                )
            except Exception as exc:  # noqa: BLE001
                m = {"symbol": sym, "arm": arm.key, "error": str(exc)[:200]}
                report["errors"].append(
                    {"symbol": sym, "arm": arm.key, "stage": "fit", "error": str(exc)[:200]}
                )
            metas[arm.key].append(m)
            if "error" in m or df_m.empty:
                line.append(f"{arm.key}=ERR")
                continue
            pieces[arm.key].append(df_m)
            line.append(f"{arm.key}[oos={m['oos_bars']},loss={m['final_val_loss']:.3f}]")
        print(" ".join(line), flush=True)

    # ------------------------------------------------------------------ #
    # assemble and evaluate
    # ------------------------------------------------------------------ #
    frames: dict[str, pd.DataFrame] = {}
    for arm in arms:
        if not pieces[arm.key]:
            print(f"  [{arm.key}] no usable predictions; skipping")
            continue
        raw_df = pd.concat(pieces[arm.key], ignore_index=True)
        df = ab.realised_frame(raw_df, store, args.interval)
        # `realised_frame` only finite-checks a fixed column list, so the arm's own
        # regression label is checked here. The *direction* label is deliberately
        # excluded from this filter: triple-barrier time exits are NaN on purpose,
        # the trainer masks them inside `binary_direction_loss`, and dropping them
        # here would delete the censoring that makes the barrier arm a
        # classification problem rather than a relabelled return.
        for col in (arm.regression, "tb_label", "tb_return"):
            if col in df.columns:
                df = df[np.isfinite(pd.to_numeric(df[col], errors="coerce"))].copy()
        df = df[np.isfinite(df["score"]) & np.isfinite(df["realised"])].reset_index(drop=True)
        frames[arm.key] = df
        df.to_parquet(args.out_predictions / f"label_{arm.key}.parquet", index=False)
        report[arm.key] = evaluate_arm(df, arm, metas[arm.key], seed=args.seed)
        r = report[arm.key]
        pm = r["perm_score"]
        print(
            f"  [{arm.key:<10}] rows {len(df):>6,} ts {r['n_timestamps']:>5} "
            f"rho {r['rank_corr_score']:+.5f}Â±{r['rank_corr_se_score']:.5f} "
            f"(null {r['null_control']['rank_corr_mean']:+.5f}) "
            f"perm p {pm['p']:.4f} own_auc {r.get('own_auc', float('nan')):.4f} "
            f"fwd_auc {r['oos_direction_auc']:.4f} "
            f"d1-d10 {r['deciles_next_bar']['top_minus_bottom_bps']:+.2f}bps "
            f"mono {r['deciles_next_bar']['monotonicity']:+.3f}",
            flush=True,
        )

    # Cross-arm alignment: the arms must be scored on byte-identical rows, or the
    # comparison is not controlled.
    if len(frames) > 1:
        keys = {k: set(map(tuple, v[["ts", "symbol"]].to_numpy())) for k, v in frames.items()}
        base = next(iter(keys))
        report["alignment"] = {
            "identical_across_arms": all(keys[k] == keys[base] for k in keys),
            "rows_per_arm": {k: int(len(v)) for k, v in frames.items()},
            "pairwise_symmetric_difference": {
                k: int(len(keys[base] ^ keys[k])) for k in keys if k != base
            },
        }
        print(
            f"  alignment identical={report['alignment']['identical_across_arms']} "
            f"{report['alignment']['rows_per_arm']}"
        )

    # Rank by evidence, and say plainly what the evidence is.
    def evidence(m: dict[str, Any]) -> tuple[float, float]:
        return (
            float(m.get("rank_corr_score", float("nan"))),
            -float(m.get("perm_score", {}).get("p", 1.0)),
        )

    ranked = sorted((k for k in frames), key=lambda k: evidence(report[k]), reverse=True)
    report["ranking_by_evidence"] = [
        {
            "arm": k,
            "rank_corr": report[k]["rank_corr_score"],
            "perm_p": report[k]["perm_score"]["p"],
            "own_auc": report[k].get("own_auc"),
            "fwd_auc": report[k]["oos_direction_auc"],
        }
        for k in ranked
    ]
    report["evidence_caveat"] = (
        "arms share one seed and one split, so a rank ordering on a single fold is "
        "a screening result, not a selection: a gap smaller than its own standard "
        "error does not separate the arms"
    )

    econ_arm = args.economic_arm or (ranked[0] if ranked else "")
    if econ_arm and econ_arm in frames:
        ed = economic_diagnostic(frames[econ_arm], store, args.interval, args=args, arm=econ_arm)
        report["economic_diagnostic"] = ed
        if "error" in ed:
            print(f"  economic diagnostic failed: {ed['error']}")
        else:
            print(
                f"  economic diagnostic ({econ_arm}): "
                f"gross {ed['gross']['total_return']:.4f} net "
                f"{ed['net']['total_return']:.4f} "
                f"all-in {ed['all_in_cost_bps_per_fill']}bps/fill "
                f"({ed['all_in_cost_bps_round_trip']}bps/rt) "
                f"{ed['round_trips_per_day']:.2f} rt/day"
            )

    args.out_report.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.out_report}")
    if report["errors"]:
        print(f"  {len(report['errors'])} fit errors recorded in the report")


if __name__ == "__main__":
    main()
