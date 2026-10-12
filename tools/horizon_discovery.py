"""Horizon discovery: is the *holding horizon* the binding constraint?

The label ablation (`tools/label_ablation.py`) ruled out the target formulation:
raw next-bar return, volatility-adjusted return and a vol-scaled triple barrier
all produced next-bar AUCs of 0.499-0.505 and a gross expectancy of +0.09 bps per
round trip against a 36 bps all-in cost. Since three materially different labels
agreed, the label is not what is missing. This script asks the next question in the
same controlled style: **do the existing features carry predictive information at a
longer holding horizon than next-bar?**

Distinction from `tools/horizon_decay.py`, which is easy to confuse with this:
that script trains nothing and reads a frozen 1-step model's output against
longer-horizon realised returns. That answers "is the 1h model's output related to
6h outcomes?", which is not the question here. Here each horizon gets its own fit,
so the only variable that changes is the horizon the model is trained to forecast.

Held fixed across every horizon: feature set, architecture, optimiser, loss, epochs,
seed, symbol universe, data source, val ratio, selection rule and evaluation code.
No hyperparameter search, no feature selection, no feature engineering, no portfolio
optimisation, no expected-move gate.

Two consequences of the leakage rule are worth stating, because they are the only
places the horizons are *not* strictly identical:

* **The purge is horizon-specific.** A row at ``t`` supervised by
  ``log(P[t+h]/P[t])`` reads prices up to ``t+h``, so the purge must be at least
  ``h``. That is the rule being applied correctly, not a confound: it is applied
  identically within each horizon, and at ~8.7k bars the difference between a 1-bar
  and a 24-bar purge is 23 rows.
* **The expected move grows with the horizon.** A 24h return has far more room to
  clear a 36 bps hurdle than a 1h return. That is the economic reason this
  experiment is worth running at all, and it is reported per horizon as
  ``expected_move_bps`` against ``required_gross_edge_bps``.

Row alignment is exact across horizons. Every symbol's frame is trimmed once by the
*largest* horizon, so ``log(P[t+h]/P[t])`` is finite for all ``h`` on every
remaining row and all horizons are scored on identical timestamps.

Interpretation rule enforced by the reporting, not left to the reader: a horizon
only advances on a *combination* of ordering, economic size, decile monotonicity,
separation from the null, and a real improvement on the +0.09 bps/round-trip gross
baseline. A headline AUC of 0.507 is explicitly not a breakthrough.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import ab_direction_head as ab  # noqa: E402  corrected metrics + invariants
import label_ablation as la  # noqa: E402  decile profile, null control, AUC CI
from one_year_sim import build_position_matrix, prepare_symbol, run_portfolio  # noqa: E402

from cqip.config import Settings  # noqa: E402
from cqip.features import make_targets, resolve_purge_embargo  # noqa: E402
from cqip.forecaster import ForecasterTrainer, model_has_direction_head  # noqa: E402
from cqip.labeling import directional_labels  # noqa: E402
from cqip.storage import ParquetStore  # noqa: E402

TOP_N = ab.TOP_N
DAY_BLOCK = ab.DAY_BLOCK
DEFAULT_HORIZONS = [1, 2, 4, 6, 12, 24]
#: Gross expectancy the label ablation established, in bps per round trip. Any
#: horizon that cannot clearly beat this has not moved the binding constraint.
ABLATION_GROSS_BPS_PER_RT = 0.09


@dataclass
class _Frame:
    """Minimal stand-in exposing the fields `ab.build_split` reads."""

    frame: pd.DataFrame
    test_from: int


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #
def horizon_frame(base: pd.DataFrame, bars: int) -> pd.DataFrame:
    """The shared feature frame carrying the h-bar target and its trailing sigma.

    ``make_targets`` is the production builder, so ``target_return`` is
    ``log(P[t+h]/P[t])`` -- the log of the *ratio*, which is defined for down-bars,
    unlike ``log(P[t+h]/P[t] - 1)``, which is NaN whenever price fell and would
    silently delete half of every sample. ``target_volatility`` is the strictly
    trailing 24-bar sigma scaled by ``sqrt(h)``, i.e. the expected size of this
    horizon's move, so the score is always in units of its own noise.
    """
    return make_targets(base, horizon=bars)


def realised_forward(
    preds: pd.DataFrame, store: ParquetStore, interval: str, bars: int
) -> pd.DataFrame:
    """Attach the realised h-bar forward return, recomputed from raw closes.

    Independent of every model column, so it doubles as a check that the label and
    the fill alignment agree.
    """
    syms = sorted(preds["symbol"].unique())
    ohlc = store.read(
        "ohlcv",
        filters=[("interval", "==", interval)],
        symbols=syms,
        columns=["ts", "close", "symbol"],
    )
    ohlc["ts"] = pd.DatetimeIndex(ohlc["ts"])
    close = ohlc.pivot_table(index="ts", columns="symbol", values="close").sort_index().ffill()
    with np.errstate(divide="ignore", invalid="ignore"):
        realised = (
            np.log(close.shift(-bars) / close)
            .rename_axis("ts")
            .reset_index()
            .melt(id_vars="ts", var_name="symbol", value_name="realised")
        )
    return preds.merge(realised, on=["ts", "symbol"], how="inner")


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def fit_horizon(
    symbol: str,
    frame: pd.DataFrame,
    test_from: int,
    bars: int,
    *,
    feature_columns: list[str],
    epochs: int,
    seed: int,
    val_ratio: float,
    seq_len: int,
    aux_weight: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit one horizon for one symbol and predict its OOS slice."""
    purge, embargo = resolve_purge_embargo(horizon=bars, seq_len=seq_len, embargo_bars=0)
    gap = purge + embargo
    train, val, oos = ab.build_split(
        _Frame(frame=frame, test_from=test_from), gap, val_ratio, seq_len
    )
    if len(oos) <= seq_len:
        return pd.DataFrame(), {"symbol": symbol, "horizon_bars": bars, "error": "no OOS"}
    if not np.isfinite(train["target_return"].to_numpy(dtype=float)).all():
        return pd.DataFrame(), {
            "symbol": symbol,
            "horizon_bars": bars,
            "error": "non-finite target reached the trainer",
        }

    trainer = ForecasterTrainer(Settings())
    t0 = time.time()
    model, info = trainer.train(
        train,
        val,
        feature_columns,
        target_column="target_return",
        direction_column="horizon_direction",
        aux_direction_weight=aux_weight,
        seq_len=seq_len,
        horizon=bars,
        kind="lstm",
        epochs=epochs,
        seed=seed,
    )
    fit_seconds = time.time() - t0

    # Production's validation-only volatility scale, fitted on this horizon's own
    # residuals and its own trailing sigma.
    val_out = ab.forward_all(model, trainer, val, feature_columns, info, seq_len)
    val_al = ab.aligned_frame(val, len(val_out["mean"]))
    val_vol = val_al["target_volatility"].to_numpy(dtype=float)
    val_target = val_al["target_return"].to_numpy(dtype=float)
    finite = np.isfinite(val_out["mean"]) & np.isfinite(val_target) & np.isfinite(val_vol)
    if finite.sum() < 50:
        return pd.DataFrame(), {
            "symbol": symbol,
            "horizon_bars": bars,
            "error": "too few finite val rows",
        }
    resid = np.std(val_target[finite] - val_out["mean"][finite], ddof=1) / max(
        float(np.mean(val_vol[finite])), 1e-12
    )
    vs = float(np.clip(resid if np.isfinite(resid) and resid > 0 else 1.0, 0.2, 10.0))

    train_end = len(train) + len(val)
    predict_frame = frame.iloc[max(0, train_end - seq_len) :].reset_index(drop=True)
    test = ab.forward_all(model, trainer, predict_frame, feature_columns, info, seq_len)
    aligned = ab.aligned_frame(predict_frame, len(test["mean"]))
    raw_vol = aligned["target_volatility"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = test["mean"] / (np.where(raw_vol > 1e-9, raw_vol, np.nan) * vs)

    out = pd.DataFrame(
        {
            "ts": predict_frame["ts"].to_numpy()[seq_len:],
            "symbol": symbol,
            "horizon_bars": bars,
            "score": score,
            "predicted_return": test["mean"],
            "predicted_volatility": np.sqrt(np.exp(test["log_var"])),
            "target_return": aligned["target_return"].to_numpy(dtype=float),
            "target_volatility": raw_vol,
            "horizon_direction": aligned["horizon_direction"].to_numpy(dtype=float),
            "vol_scale": vs,
        }
    )
    if test["logit"] is not None:
        out["direction_prob"] = 1.0 / (1.0 + np.exp(-np.clip(test["logit"], -30, 30)))

    vm = {
        "val_nll_shared_sigma": ab.gaussian_nll(val_out["mean"], val_target, val_vol),
        "val_nll_own_predicted_sigma": ab.own_objective_nll(
            val_out["mean"], val_target, np.sqrt(np.exp(val_out["log_var"]))
        ),
        "val_target_std_bps": float(np.nanstd(val_target) * 1e4),
    }
    if val_out["logit"] is not None:
        y = pd.to_numeric(val_al["horizon_direction"], errors="coerce").to_numpy(float)
        if np.isfinite(y).any():
            p = 1.0 / (1.0 + np.exp(-np.clip(val_out["logit"], -30, 30)))
            vm["val_direction_bce"] = ab.direction_bce(p, y)
            vm["val_direction_accuracy"] = float(((val_out["logit"] > 0).astype(float) == y).mean())
    meta = {
        "symbol": symbol,
        "horizon_bars": bars,
        "train_bars": len(train),
        "val_bars": len(val),
        "oos_bars": len(oos),
        "purge": purge,
        "embargo": embargo,
        "gap": gap,
        "vol_scale": round(vs, 4),
        "has_direction_head": model_has_direction_head(model),
        "train_end_ts": str(frame["ts"].iloc[train_end - 1]),
        "final_val_loss": float(info["metrics"].get("final_val_loss", float("nan"))),
        "fit_seconds": round(fit_seconds, 1),
        **vm,
    }
    return out, meta


def se_get(metrics: dict[str, Any], key: str, default: Any = float("nan")) -> Any:
    """Read a signal-economics field, tolerating a refused computation.

    A horizon whose cross-section was too thin to measure has no economics; that
    must surface as "not measurable" rather than as a zero that reads like a pass.
    """
    se = metrics.get("signal_economics", {})
    if "error" in se:
        return default
    value = se.get(key, default)
    return default if value is None else value


# --------------------------------------------------------------------------- #
# signal-level economics (no portfolio tuning)
# --------------------------------------------------------------------------- #
def signal_economics(
    df: pd.DataFrame, *, top_k: int, round_trip_cost_bps: float, safety_margin_bps: float
) -> dict[str, Any]:
    """Pre-cost expectancy of the fixed production selection, per signal.

    Uses the production rule -- long the ``top_k`` best scores, short the ``top_k``
    worst, every bar -- with nothing tuned. Each selected entry is treated as one
    round trip of the horizon's length, which is the only assumption made, and it
    is the assumption the target itself encodes.

    The point of the required-edge comparison is scale: a 24h move is many times a
    1h move, so a larger ``expected_move_bps`` lowers the *relative* size of the cost
    hurdle even when the realised expectancy stays near zero.
    """
    longs: list[np.ndarray] = []
    shorts: list[np.ndarray] = []
    for _stamp, g in df.groupby("ts", sort=True):
        ranked = g.dropna(subset=["score", "realised"])
        # A cross-section thinner than 2*top_k makes head and tail the same rows,
        # so the spread would be identically zero rather than merely small.
        if len(ranked) < 2 * top_k:
            continue
        ranked = ranked.sort_values("score", ascending=False, kind="mergesort")
        head, tail = ranked.head(top_k), ranked.tail(top_k)
        if head["score"].iloc[-1] == tail["score"].iloc[0]:
            continue
        longs.append(head["realised"].to_numpy(dtype=float))
        shorts.append(-tail["realised"].to_numpy(dtype=float))
    if not longs:
        return {"error": "no timestamp with a non-degenerate cross-section"}

    long_r = np.concatenate(longs)
    short_r = np.concatenate(shorts)
    per_signal = np.concatenate([long_r, short_r])
    required = round_trip_cost_bps + safety_margin_bps
    gross_bps = float(per_signal.mean() * 1e4)
    n_bars = df["ts"].nunique()
    return {
        "top_k": top_k,
        "n_signals": int(per_signal.size),
        "n_long": int(long_r.size),
        "n_short": int(short_r.size),
        "signal_frequency": float(len(longs) / max(n_bars, 1)),
        "gross_expectancy_bps_per_signal": gross_bps,
        "gross_expectancy_bps_long": float(long_r.mean() * 1e4),
        "gross_expectancy_bps_short": float(short_r.mean() * 1e4),
        "expected_move_bps": float(np.abs(per_signal).mean() * 1e4),
        "median_move_bps": float(np.median(np.abs(per_signal)) * 1e4),
        "win_rate": float((per_signal > 0).mean()),
        "required_gross_edge_bps": float(required),
        "round_trip_cost_bps": float(round_trip_cost_bps),
        "safety_margin_bps": float(safety_margin_bps),
        "net_expectancy_bps_per_signal": gross_bps - float(round_trip_cost_bps),
        "covers_cost": bool(gross_bps > round_trip_cost_bps),
        "covers_required_edge": bool(gross_bps > required),
        "edge_as_multiple_of_cost": (
            gross_bps / float(round_trip_cost_bps) if round_trip_cost_bps else float("nan")
        ),
        "movement_to_cost_ratio": (
            float(np.abs(per_signal).mean() * 1e4) / float(round_trip_cost_bps)
            if round_trip_cost_bps
            else float("nan")
        ),
        "vs_ablation_baseline_bps": gross_bps - ABLATION_GROSS_BPS_PER_RT,
    }


def portfolio_replay(
    df: pd.DataFrame, store: ParquetStore, interval: str, *, args, bars: int
) -> dict[str, Any]:
    """Fixed-configuration replay: identical selection and costs at every horizon.

    Turnover is deliberately *not* tied to the horizon. Tying it would be portfolio
    engineering, which this experiment excludes, and it would change two things at
    once. The horizon-implied cadence is reported separately as a compatibility
    figure for the ~5 trades/day objective.
    """
    by_symbol = {
        s: g[["ts", "score", "predicted_return"]].sort_values("ts") for s, g in df.groupby("symbol")
    }
    weights, _probs = build_position_matrix(by_symbol, top_k=args.top_k, prob_bar=args.prob_bar)
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

    net, gross = run(args.fee_bps, all_in_slip), run(0.0, 0.0)
    if "error" in net:
        return {"error": net["error"]}
    n_rt = net["n_round_trips"]
    n_bars = max(len(weights), 1)
    net_ret = net["metrics"].get("total_return") or 0.0
    gross_ret = gross["metrics"].get("total_return") or 0.0
    out: dict[str, Any] = {
        "net_total_return": net_ret,
        "gross_total_return": gross_ret,
        "gross_sharpe": gross["metrics"].get("sharpe"),
        "net_sharpe": net["metrics"].get("sharpe"),
        "net_max_drawdown": net["metrics"].get("max_drawdown"),
        "gross_win_rate": gross["metrics"].get("win_rate"),
        "gross_profit_factor": gross["metrics"].get("profit_factor"),
        "n_round_trips": n_rt,
        "total_bars": n_bars,
        "round_trips_per_day": n_rt / n_bars * 24,
        "signal_frequency": float((weights.abs().sum(axis=1) > 0).mean()),
        "cost_drag": gross_ret - net_ret,
        "gross_bps_per_round_trip": (gross_ret / n_rt * 1e4) if n_rt else float("nan"),
        "net_bps_per_round_trip": (net_ret / n_rt * 1e4) if n_rt else float("nan"),
        "all_in_cost_bps_per_fill": args.fee_bps + args.slippage_bps + args.spread_bps,
        "all_in_cost_bps_round_trip": 2 * (args.fee_bps + args.slippage_bps + args.spread_bps),
        "mean_gross_exposure": net.get("mean_gross_exposure"),
        # Horizon-implied cadence if each signal were held its own horizon. This is
        # a compatibility statement about the ~5 trades/day objective, not a setting.
        "horizon_implied_rt_per_day": 24.0 / bars,
        "compatible_with_5_per_day": bool(1.0 <= 24.0 / bars <= 10.0),
        "note": "fixed configuration; no parameter was searched",
    }
    return out


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate_horizon(
    df: pd.DataFrame,
    bars: int,
    metas: list[dict[str, Any]],
    *,
    store: ParquetStore,
    interval: str,
    args,
) -> dict[str, Any]:
    """Every required readout for one horizon, plus the null control."""
    base = ab.evaluate(df, f"horizon::{bars}h")
    out: dict[str, Any] = {k: v for k, v in base.items() if k != "cal_buckets"}
    out["horizon_bars"] = bars
    out["n_rows"] = int(len(df))
    out["n_timestamps"] = int(df["ts"].nunique())
    out["n_symbols"] = int(df["symbol"].nunique())

    df = df.copy()
    df["_dir"] = (df["target_return"] > 0).astype(float)
    prob = df["direction_prob"].to_numpy(dtype=float) if "direction_prob" in df.columns else None
    if prob is not None:
        o, lo, hi = la.boot_auc_ci(df, "direction_prob", "_dir")
        out["oos_auc_direction_head"] = o
        out["oos_auc_direction_head_ci"] = [lo, hi]
        out["oos_auc_excludes_null"] = bool(lo > 0.5 or hi < 0.5)
    o, lo, hi = la.boot_auc_ci(df, "score", "_dir")
    out["oos_auc_score"] = o
    out["oos_auc_score_ci"] = [lo, hi]
    out["oos_auc_score_excludes_null"] = bool(lo > 0.5 or hi < 0.5)
    out["oos_directional_accuracy"] = float(
        ((df["predicted_return"] > 0).astype(float) == df["_dir"]).mean()
    )
    out["oos_balanced_accuracy"] = ab.balanced_accuracy(
        (df["predicted_return"] > 0).astype(float), df["_dir"].to_numpy()
    )

    out["null_control"] = la.random_score_null(df, "score", seed=args.seed)
    out["deciles"] = la.decile_profile(df, "score", "realised")
    out["target_std_bps"] = float(df["target_return"].std() * 1e4)
    out["mean_abs_target_bps"] = float(df["target_return"].abs().mean() * 1e4)
    out["expected_move_bps_universe"] = out["mean_abs_target_bps"]

    rt_cost = 2 * (args.fee_bps + args.slippage_bps + args.spread_bps)
    out["signal_economics"] = signal_economics(
        df, top_k=args.top_k, round_trip_cost_bps=rt_cost, safety_margin_bps=args.safety_margin_bps
    )
    out["portfolio_replay"] = portfolio_replay(df, store, interval, args=args, bars=bars)

    inv = ab.invariants(df, metas)
    up = df["realised"] > 0
    inv["both_directions_present"] = bool(up.any() and (~up).any())
    inv["up_fraction"] = float(up.mean())
    inv["label_matches_realised"] = bool(
        np.nanmax((df["target_return"] - df["realised"]).abs().to_numpy()) < 1e-6
    )
    inv["max_label_vs_realised_gap"] = float(
        np.nanmax((df["target_return"] - df["realised"]).abs().to_numpy())
    )
    inv["purge_matches_horizon"] = bool(
        all(m.get("purge", bars) == bars for m in metas if "purge" in m)
    )
    inv["no_nan_target"] = bool(np.isfinite(df["target_return"].to_numpy(dtype=float)).all())
    inv["no_silent_deletion"] = inv["no_nan_target"]
    counts = df.groupby("ts")["symbol"].nunique()
    inv["min_cross_section"] = int(counts.min())
    inv["cross_section_at_least_2_topn"] = bool((counts >= 2 * TOP_N).all())
    inv["n_fit_errors"] = sum(1 for m in metas if "error" in m)
    out["invariants"] = inv
    return out


def gate_horizon(m: dict[str, Any], *, rt_cost_bps: float) -> dict[str, Any]:
    """Apply the advancement rule mechanically, so the verdict is not a judgement call.

    A horizon advances only on the *conjunction* of ordering, monotonicity,
    separation from the null, and economics. A lone AUC blip is explicitly
    insufficient, and so is a large expected move with no ordering to exploit it.
    """
    se = m["signal_economics"]
    dec = m["deciles"]
    # A cross-section too thin to form a non-degenerate head/tail split yields no
    # signal economics at all. That is a refusal to compute, not a zero, so it must
    # not be silently promoted into a passing or failing economics verdict.
    if "error" in se:
        return {
            "checks": {},
            "substantive_bar_met": False,
            "advances": False,
            "not_assessable": se["error"],
            "required_gross_edge_bps": float(rt_cost_bps),
            "observed_gross_bps": float("nan"),
            "shortfall_bps": float("nan"),
        }
    checks = {
        "ordering": bool(
            np.isfinite(m["rank_corr_score"])
            and m["rank_corr_score"] > 0
            and m["perm_score"]["p"] < 0.05
        ),
        "monotonicity": bool(np.isfinite(dec["monotonicity"]) and dec["monotonicity"] > 0),
        "separated_from_null": bool(m.get("oos_auc_score_excludes_null", False)),
        "beats_ablation_baseline": bool(
            np.isfinite(se["gross_expectancy_bps_per_signal"])
            and se["gross_expectancy_bps_per_signal"] > ABLATION_GROSS_BPS_PER_RT
        ),
        "covers_realistic_cost": bool(se["covers_cost"]),
        "covers_required_edge": bool(se["covers_required_edge"]),
    }
    # Ordering plus economics is the substantive bar. Monotonicity and null
    # separation must not contradict it, but neither can rescue a null result on
    # their own, so they are recorded as supporting rather than sufficient.
    substantive = (
        checks["ordering"] and checks["beats_ablation_baseline"] and checks["covers_realistic_cost"]
    )
    return {
        "checks": checks,
        "substantive_bar_met": bool(substantive),
        "advances": bool(substantive and checks["monotonicity"]),
        "required_gross_edge_bps": se["required_gross_edge_bps"],
        "observed_gross_bps": se["gross_expectancy_bps_per_signal"],
        "shortfall_bps": (se["required_gross_edge_bps"] - se["gross_expectancy_bps_per_signal"]),
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
    p.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    p.add_argument("--symbols", default="", help="comma list; default is production 25")
    p.add_argument("--capital", type=float, default=1_000.0)
    p.add_argument("--top-k", type=int, default=TOP_N)
    p.add_argument("--prob-bar", type=float, default=0.0)
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--spread-bps", type=float, default=3.0)
    p.add_argument(
        "--safety-margin-bps",
        type=float,
        default=24.0,
        help="added to the round-trip cost to form required_gross_edge",
    )
    p.add_argument("--rebalance-band-bps", type=float, default=0.0)
    p.add_argument("--min-hold-bars", type=int, default=0)
    p.add_argument("--max-fills-per-day", type=int, default=None)
    p.add_argument("--gate-mode", choices=("flat", "hold"), default="flat")
    p.add_argument(
        "--out-predictions", type=Path, default=Path("data/artifacts/reports/horizon_discovery")
    )
    p.add_argument(
        "--out-report", type=Path, default=Path("data/artifacts/reports/horizon_discovery.json")
    )
    args = p.parse_args()

    horizons = sorted({int(x) for x in args.horizons.split(",") if x.strip()})
    if not horizons or min(horizons) < 1:
        p.error("horizons must be positive integers")
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()] or list(ab.PRODUCTION_UNIVERSE)
    store = ParquetStore(settings=Settings())
    rt_cost = 2 * (args.fee_bps + args.slippage_bps + args.spread_bps)
    max_h = max(horizons)

    print("HORIZON DISCOVERY -- horizon is the only variable")
    print(
        f"  interval={args.interval} cutoff={args.cutoff} seed={args.seed} "
        f"epochs={args.epochs} seq_len={args.seq_len} aux_weight={args.aux_weight}"
    )
    print(f"  horizons={horizons} bars (purge = h for each, per the leakage rule)")
    print(f"  universe={len(syms)} symbols (pinned); features/arch/optim/loss unchanged")
    print(
        f"  costs: {args.fee_bps}+{args.slippage_bps}+{args.spread_bps}="
        f"{args.fee_bps + args.slippage_bps + args.spread_bps}bps/fill -> "
        f"{rt_cost}bps round trip; required gross edge "
        f"{rt_cost + args.safety_margin_bps}bps"
    )
    print(f"  ablation baseline to beat: {ABLATION_GROSS_BPS_PER_RT}bps gross/round trip")
    print()

    args.out_predictions.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "design": {
            "held_fixed": [
                "feature set",
                "architecture",
                "optimiser",
                "loss",
                "epochs",
                "seed",
                "universe",
                "data source",
                "val_ratio",
                "selection rule",
                "evaluation implementation",
            ],
            "excluded": [
                "hyperparameter search",
                "feature selection",
                "feature engineering",
                "portfolio optimisation",
                "expected-move gate",
                "min-hold tuning",
                "top-k tuning",
            ],
            "purge_rule": "purge = horizon bars, embargo = 0, applied within each horizon",
            "row_alignment": (
                "every frame trimmed once by the largest horizon, so all "
                "horizons are scored on identical timestamps"
            ),
            "ablation_baseline_gross_bps_per_rt": ABLATION_GROSS_BPS_PER_RT,
        },
        "errors": [],
        "horizons": {},
    }

    # Prepared once per symbol at the largest horizon, then reused for every
    # horizon. Trimming by max_h is what makes log(P[t+h]/P[t]) finite for all h on
    # every remaining row.
    prepared: dict[str, Any] = {}
    print("  preparing universe ...", flush=True)
    for sym in syms:
        try:
            prepared[sym] = prepare_symbol(
                store,
                sym,
                args.interval,
                warmup_bars=args.cutoff,
                trade_bars=10**9,
                refit_bars=720,
                horizon=max_h,
                min_test_bars=20,
            )
        except Exception as exc:  # noqa: BLE001
            report["errors"].append({"symbol": sym, "stage": "prepare", "error": str(exc)[:200]})
            print(f"    {sym:<14} PREPARE FAILED: {str(exc)[:70]}")
    ok_syms = [s for s in syms if s in prepared]
    print(f"  prepared {len(ok_syms)}/{len(syms)} symbols at horizon {max_h}")
    print()

    for bars in horizons:
        pieces: list[pd.DataFrame] = []
        metas: list[dict[str, Any]] = []
        t0 = time.time()
        for i, sym in enumerate(ok_syms, 1):
            prep = prepared[sym]
            frame = horizon_frame(prep.frame, bars)
            # The binary label: exact zeros are NaN, never folded into a class.
            frame["horizon_direction"] = directional_labels(frame["target_return"])
            try:
                df_m, m = fit_horizon(
                    sym,
                    frame,
                    min(prep.test_from, len(frame)),
                    bars,
                    feature_columns=prep.feature_columns,
                    epochs=args.epochs,
                    seed=args.seed,
                    val_ratio=args.val_ratio,
                    seq_len=args.seq_len,
                    aux_weight=args.aux_weight,
                )
            except Exception as exc:  # noqa: BLE001
                m = {"symbol": sym, "horizon_bars": bars, "error": str(exc)[:200]}
                report["errors"].append(
                    {"symbol": sym, "horizon_bars": bars, "stage": "fit", "error": str(exc)[:200]}
                )
            metas.append(m)
            if "error" not in m and not df_m.empty:
                pieces.append(df_m)
            if i % 8 == 0 or i == len(ok_syms):
                print(
                    f"    [{bars:>2}h] {i:>2}/{len(ok_syms)} fitted ({time.time() - t0:.0f}s)",
                    flush=True,
                )
        if not pieces:
            print(f"  [{bars}h] no usable predictions; skipping")
            continue

        raw_df = pd.concat(pieces, ignore_index=True)
        df = realised_forward(raw_df, store, args.interval, bars)
        df["prob"] = la_norm_cdf(df["score"].to_numpy(dtype=float))
        finite = (
            np.isfinite(df["score"])
            & np.isfinite(df["realised"])
            & np.isfinite(df["target_return"])
            & np.isfinite(df["target_volatility"])
            & np.isfinite(df["predicted_return"])
            & np.isfinite(df["predicted_volatility"])
        )
        df = df.loc[finite].reset_index(drop=True)
        df.to_parquet(args.out_predictions / f"horizon_{bars}h.parquet", index=False)

        metrics = evaluate_horizon(df, bars, metas, store=store, interval=args.interval, args=args)
        metrics["gate"] = gate_horizon(metrics, rt_cost_bps=rt_cost)
        report["horizons"][str(bars)] = metrics
        # Written after every horizon so a long run is never all-or-nothing.
        args.out_report.write_text(json.dumps(report, indent=2, default=str))

        se = metrics["signal_economics"]
        dec = metrics["deciles"]
        if "error" in se:
            print(
                f"  [{bars:>2}h] rows {len(df):>6,} auc {metrics['oos_auc_score']:.4f} "
                f"[{metrics['oos_auc_score_ci'][0]:.4f},"
                f"{metrics['oos_auc_score_ci'][1]:.4f}] "
                f"rho {metrics['rank_corr_score']:+.5f} "
                f"perm p {metrics['perm_score']['p']:.4f}"
            )
            print(f"       signal economics not measurable: {se['error']}")
            print("       ADVANCES=False (not assessable)", flush=True)
            continue
        print(
            f"  [{bars:>2}h] rows {len(df):>6,} auc {metrics['oos_auc_score']:.4f} "
            f"[{metrics['oos_auc_score_ci'][0]:.4f},{metrics['oos_auc_score_ci'][1]:.4f}] "
            f"diracc {metrics['oos_directional_accuracy']:.4f} "
            f"rho {metrics['rank_corr_score']:+.5f}Â±{metrics['rank_corr_se_score']:.5f} "
            f"perm p {metrics['perm_score']['p']:.4f}"
        )
        print(
            f"       move {metrics['mean_abs_target_bps']:7.1f}bps  "
            f"gross {se['gross_expectancy_bps_per_signal']:+7.2f}bps/signal  "
            f"req {se['required_gross_edge_bps']:.0f}bps  "
            f"net {se['net_expectancy_bps_per_signal']:+8.2f}bps  "
            f"mono {dec['monotonicity']:+.3f}  d1-d10 {dec['top_minus_bottom_bps']:+.2f}bps"
        )
        print(
            f"       replay: gross {metrics['portfolio_replay']['gross_total_return']:+.4f} "
            f"net {metrics['portfolio_replay']['net_total_return']:+.4f} "
            f"grossSh {metrics['portfolio_replay']['gross_sharpe']} "
            f"{metrics['portfolio_replay']['round_trips_per_day']:.1f} rt/day  "
            f"ADVANCES={metrics['gate']['advances']}",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # cross-horizon comparison
    # ------------------------------------------------------------------ #
    done = report["horizons"]
    if done:
        report["alignment"] = {
            "rows_per_horizon": {h: m["n_rows"] for h, m in done.items()},
            "identical_across_horizons": len({m["n_rows"] for m in done.values()}) == 1,
            "timestamps_per_horizon": {h: m["n_timestamps"] for h, m in done.items()},
        }
        report["summary"] = [
            {
                "horizon_bars": m["horizon_bars"],
                "oos_auc_score": m["oos_auc_score"],
                "oos_auc_score_ci": m["oos_auc_score_ci"],
                "oos_auc_direction_head": m.get("oos_auc_direction_head"),
                "oos_directional_accuracy": m["oos_directional_accuracy"],
                "rank_corr": m["rank_corr_score"],
                "rank_corr_se": m["rank_corr_se_score"],
                "perm_p": m["perm_score"]["p"],
                "decile_monotonicity": m["deciles"]["monotonicity"],
                "top_minus_bottom_bps": m["deciles"]["top_minus_bottom_bps"],
                "expected_move_bps": m["mean_abs_target_bps"],
                "gross_bps_per_signal": se_get(m, "gross_expectancy_bps_per_signal"),
                "required_gross_edge_bps": se_get(m, "required_gross_edge_bps"),
                "net_bps_per_signal": se_get(m, "net_expectancy_bps_per_signal"),
                "gross_sharpe": m["portfolio_replay"].get("gross_sharpe"),
                "gross_total_return": m["portfolio_replay"].get("gross_total_return"),
                "net_total_return": m["portfolio_replay"].get("net_total_return"),
                "round_trips_per_day": m["portfolio_replay"].get("round_trips_per_day"),
                "horizon_implied_rt_per_day": m["portfolio_replay"].get(
                    "horizon_implied_rt_per_day"
                ),
                "compatible_with_5_per_day": m["portfolio_replay"].get("compatible_with_5_per_day"),
                "covers_cost": bool(se_get(m, "covers_cost", False)),
                "advances": m["gate"]["advances"],
            }
            for m in done.values()
        ]
        report["verdict"] = {
            "any_horizon_advances": any(m["gate"]["advances"] for m in done.values()),
            "advancing_horizons": [
                m["horizon_bars"] for m in done.values() if m["gate"]["advances"]
            ],
            "best_gross_bps_per_signal": max(
                (se_get(m, "gross_expectancy_bps_per_signal") for m in done.values()),
                default=float("nan"),
            ),
            "baseline_gross_bps_per_signal": ABLATION_GROSS_BPS_PER_RT,
            "case": "A" if any(m["gate"]["advances"] for m in done.values()) else "B",
            "single_fold_caveat": (
                "one cutoff and one seed per horizon: this screens "
                "horizons, it does not establish an edge"
            ),
        }
    args.out_report.write_text(json.dumps(report, indent=2, default=str))
    print()
    if "verdict" in report:
        v = report["verdict"]
        print(f"  CASE {v['case']}: any_horizon_advances={v['any_horizon_advances']}")
        print(
            f"  best gross {v['best_gross_bps_per_signal']:+.2f}bps/signal vs "
            f"required {rt_cost + args.safety_margin_bps}bps "
            f"(baseline {ABLATION_GROSS_BPS_PER_RT}bps)"
        )
    print(f"wrote {args.out_report}")


def la_norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF, matching the traded probability in production."""
    from scipy.stats import norm

    return norm.cdf(x)


if __name__ == "__main__":
    main()
