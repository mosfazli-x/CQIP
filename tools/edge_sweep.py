"""Sweep trading-logic policy on saved forecasts. No refitting.

This replays a persisted weight matrix (and, when available, the per-bar
predictions behind it) under different *restraint* settings, so the question
"is the problem turnover or is it the signal?" can be answered without paying
for another training run.

Three things are deliberately reported side by side for every configuration:

* **gross** -- the same signals with fees and slippage set to zero. That is the
  edge the model actually produces, isolated from execution.
* **net** -- the same signals paying 10bps of fee and 5bps of slippage a side.
* **excess** -- net return against equal-weight buy-and-hold on identical bars.
  A positive net number that does not beat holding the universe is not a result.

Round trips, not fills, are the unit of "a trade". A position trimmed twenty
times in a day is one managed trade, and counting it as twenty is how a churning
book looks busy instead of expensive.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import one_year_sim as sim  # noqa: E402

from cqip.config import Settings  # noqa: E402


def load_predictions(path: Path | None) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Return (move_grid, signal_diagnostics) from a saved predictions parquet."""
    if path is None or not path.exists():
        return None, {}
    frame = pd.read_parquet(path)
    move = frame.pivot_table(
        index="ts", columns="symbol", values="predicted_return", aggfunc="last"
    )
    move.index = pd.DatetimeIndex(move.index)

    diagnostics: dict[str, Any] = {}
    if {"score", "target_return"} <= set(frame.columns):
        from one_year_sim import _bootstrap_auc, _normal_cdf

        score = frame["score"].to_numpy(dtype=float)
        target = frame["target_return"].to_numpy(dtype=float)
        finite = np.isfinite(score) & np.isfinite(target)
        score, target = score[finite], target[finite]
        labels = (target > 0).astype(float)
        point, low, high = _bootstrap_auc(score, labels, n_boot=400, seed=7)
        claimed = _normal_cdf(score)
        diagnostics = {
            "oos_bars": int(len(score)),
            "auc": float(point),
            "auc_ci_low": float(low),
            "auc_ci_high": float(high),
            "directional_accuracy": float(np.mean(np.sign(target) == np.sign(score))),
            "correlation": float(np.corrcoef(target, score)[0, 1]),
        }
        # Expected calibration error over claimed-probability deciles: the average
        # gap between what the model asserts and what the tape delivered.
        bins = pd.qcut(pd.Series(claimed), 10, labels=False, duplicates="drop")
        ece = 0.0
        for bucket in pd.unique(bins):
            mask = bins == bucket
            if mask.any():
                ece += float(mask.mean()) * abs(
                    float(claimed[mask].mean()) - float(labels[mask].mean())
                )
        diagnostics["calibration_error"] = ece
        diagnostics["top_decile_claimed"] = float(claimed[pd.Series(claimed).rank(pct=True) >= 0.9].mean())
        top = pd.Series(claimed).rank(pct=True) >= 0.9
        diagnostics["top_decile_observed"] = float(labels[top.to_numpy()].mean())
    return move, diagnostics


def summarise(
    label: str,
    gross: dict[str, Any],
    net: dict[str, Any],
    *,
    interval: str,
    capital: float,
    equity_bench: dict[str, Any] | None,
    btc_bench: dict[str, Any] | None,
    signal: dict[str, Any],
) -> dict[str, Any]:
    metrics = net["metrics"]
    trips = net["round_trips"]
    bars = int(metrics["n_periods"])
    hours = sim._hours_per_bar(interval)
    days = bars * hours / 24.0
    months = days / 30.44

    def _stats(values: list[float]) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        arr = np.asarray(values, dtype=float)
        return float(arr.mean()), float(np.median(arr))

    trade_ret_mean, trade_ret_median = _stats([t["net_return"] for t in trips])
    trade_gross_mean, _ = _stats([t["gross_return"] for t in trips])
    hold_mean, hold_median = _stats([t["hours_held"] for t in trips])

    net_return = float(metrics["total_return"])
    bench_return = (
        float(equity_bench["metrics"]["total_return"]) if equity_bench else float("nan")
    )
    btc_return = float(btc_bench["metrics"]["total_return"]) if btc_bench else float("nan")

    return {
        "config": label,
        "policy": net["policy"],
        # --- headline -------------------------------------------------------
        "total_return": net_return,
        "annualised_return": float(metrics["cagr"]),
        "sharpe": float(metrics["sharpe"]),
        "sortino": float(metrics["sortino"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "win_rate_bars": float(metrics["win_rate"]),
        "profit_factor": float(metrics["profit_factor"]),
        # --- trade-level ----------------------------------------------------
        "round_trips": len(trips),
        "round_trips_per_day": len(trips) / days if days else float("nan"),
        "avg_trade_return": trade_ret_mean,
        "med_trade_return": trade_ret_median,
        "avg_trade_gross_return": trade_gross_mean,
        "avg_hold_hours": hold_mean,
        "med_hold_hours": hold_median,
        # --- activity -------------------------------------------------------
        "fills": int(net["n_trades"]),
        "fills_per_day": float(net["n_trades"] / days) if days else float("nan"),
        "turnover_per_month_x": float(net["turnover_x_initial"] / months) if months else float("nan"),
        "turnover_per_year_x": float(net["turnover_x_initial"]),
        "mean_gross_exposure": float(net["mean_gross_exposure"]),
        # --- cost decomposition --------------------------------------------
        "gross_return_before_costs": float(gross["metrics"]["total_return"]),
        "costs_paid": float(net["total_fees"] + net["total_slippage"]),
        "fees_paid": float(net["total_fees"]),
        "slippage_paid": float(net["total_slippage"]),
        "net_return_after_costs": net_return,
        # How much of the cost bill the signal actually paid for, measured *inside*
        # the costed run. Comparing a cost-free run's turnover against a costed
        # run's P&L would flatter the strategy: the cost-free account compounds, so
        # its notional -- and therefore its apparent turnover -- is larger than what
        # the same signals trade once the account is actually losing money. The
        # implied gross here is net plus the costs actually paid on the net path.
        "implied_gross_in_net_run": net_return + float(
            (net["total_fees"] + net["total_slippage"]) / capital
        ),
        "cost_coverage_ratio": (
            (net_return + float((net["total_fees"] + net["total_slippage"]) / capital))
            / float((net["total_fees"] + net["total_slippage"]) / capital)
            if (net["total_fees"] + net["total_slippage"]) > 0 else float("nan")
        ),
        # --- benchmarks -----------------------------------------------------
        "equal_weight_return": bench_return,
        "excess_vs_equal_weight": net_return - bench_return,
        "btc_return": btc_return,
        "excess_vs_btc": net_return - btc_return,
        # --- signal (invariant to policy) ----------------------------------
        **{f"signal_{k}": v for k, v in signal.items()},
    }


def constraint_check(row: dict[str, Any], *, max_trips_per_day: float) -> dict[str, Any]:
    """The user's stated success criterion, evaluated mechanically."""
    trips = row["round_trips_per_day"]
    fills = row["fills_per_day"]
    beats_bench = row["excess_vs_equal_weight"] > 0
    return {
        "trips_per_day_ok": bool(trips <= max_trips_per_day),
        "fills_per_day_ok": bool(fills <= 10.0),
        "beats_equal_weight": bool(beats_bench),
        "positive_sharpe_net": bool(row["sharpe"] > 0),
        "passes_all": bool(
            trips <= max_trips_per_day and fills <= 10.0
            and beats_bench and row["sharpe"] > 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--min-hold-bars", default="0,2,3,4,8")
    parser.add_argument("--band-bps", default="0,25,100")
    parser.add_argument("--max-fills-per-day", default="none,10,6")
    parser.add_argument("--min-expected-move-bps", default="0")
    parser.add_argument("--max-trips-per-day", type=float, default=10.0)
    parser.add_argument("--gate-mode", choices=("flat", "hold"), default="flat")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    settings = Settings(log_level="ERROR", allow_model_download=False)
    store = sim.ParquetStore(settings=settings)
    weights = pd.read_parquet(args.weights)
    symbols = list(weights.columns)
    stamps = weights.index

    move_grid, signal = load_predictions(
        Path(args.predictions) if args.predictions else None
    )
    if signal:
        print("signal quality (invariant to trading policy):")
        print(f"  bars {signal['oos_bars']:,}  AUC {signal['auc']:.4f} "
              f"[{signal['auc_ci_low']:.4f}, {signal['auc_ci_high']:.4f}]  "
              f"acc {signal['directional_accuracy']:.4f}  "
              f"corr {signal['correlation']:+.4f}")
        print(f"  calibration error {signal['calibration_error']:.4f}   "
              f"top decile claims {signal['top_decile_claimed']:.3f} / "
              f"delivers {signal['top_decile_observed']:.3f}")
    else:
        print("signal quality: no --predictions supplied; reporting execution only")

    equity_bench = sim.benchmark_equal_weight(
        store, symbols, args.interval, stamps,
        initial_capital=args.capital, fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    btc_bench = sim.benchmark_buy_and_hold(
        store, "BTC_USDT", args.interval, stamps,
        initial_capital=args.capital, fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    ) if "BTC_USDT" in symbols else None
    bench_return = float(equity_bench["metrics"]["total_return"])
    print(f"\nequal-weight buy-and-hold over the same bars: {bench_return:+.2%}")
    if btc_bench:
        print(f"BTC buy-and-hold:                          {float(btc_bench['metrics']['total_return']):+.2%}")

    def _int_list(text: str) -> list[int | None]:
        return [None if t.strip().lower() in {"none", "null", ""} else int(t) for t in text.split(",")]

    grid = list(itertools.product(
        [float(x) for x in args.min_expected_move_bps.split(",")],
        [int(x) for x in args.min_hold_bars.split(",")],
        [float(x) for x in args.band_bps.split(",")],
        _int_list(args.max_fills_per_day),
    ))

    rows: list[dict[str, Any]] = []
    print(f"\nsweeping {len(grid)} configurations "
          f"(each is run twice: gross at 0bps, net at {args.fee_bps:g}+{args.slippage_bps:g}bps)\n")
    for move_bps, hold_bars, band_bps, budget in grid:
        common = dict(
            initial_capital=args.capital,
            rebalance_band_bps=band_bps,
            expected_move=move_grid,
            min_expected_move_bps=move_bps,
            min_hold_bars=hold_bars,
            max_fills_per_day=budget,
            gate_mode=args.gate_mode,
        )
        label = (
            f"move>={move_bps:g}bps hold>={hold_bars} band={band_bps:g} "
            f"budget={'none' if budget is None else budget}"
        )
        net = sim.run_portfolio(
            store, symbols, args.interval, weights,
            fee_bps=args.fee_bps, slippage_bps=args.slippage_bps, **common,
        )
        gross = sim.run_portfolio(
            store, symbols, args.interval, weights,
            fee_bps=0.0, slippage_bps=0.0, **common,
        )
        if "error" in net or "error" in gross:
            print(f"  {label}: ERROR {net.get('error') or gross.get('error')}")
            continue
        row = summarise(
            label, gross, net, interval=args.interval, capital=args.capital,
            equity_bench=equity_bench, btc_bench=btc_bench, signal=signal,
        )
        row.update(constraint_check(row, max_trips_per_day=args.max_trips_per_day))
        rows.append(row)
        print(
            f"  {label:<52} gross {row['gross_return_before_costs']:>+7.2%}  "
            f"net {row['total_return']:>+7.2%}  sharpe {row['sharpe']:>+6.2f}  "
            f"trips/day {row['round_trips_per_day']:>5.1f}  "
            f"fills/day {row['fills_per_day']:>6.1f}  "
            f"vs EW {row['excess_vs_equal_weight']:>+7.2%}"
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        print("no configurations produced a result")
        return 1

    # Ranking is deliberately not by headline return: a configuration is only an
    # improvement if it clears the cost, holds the activity budget, and beats the
    # benchmark it could simply have bought.
    # Ranked on constraint compliance first, then on how much of the cost bill
    # the signal pays for, then on benchmark excess. Headline return is deliberately
    # absent: it is the number that most rewards a configuration for overtrading.
    frame["rank_score"] = (
        frame["passes_all"].astype(int) * 1000.0
        + frame["cost_coverage_ratio"].clip(-5, 5) * 50.0
        + frame["excess_vs_equal_weight"].clip(lower=-1.0) * 100.0
    )
    ranked = frame.sort_values("rank_score", ascending=False)

    cols = [
        "config", "gross_return_before_costs", "total_return", "annualised_return",
        "sharpe", "max_drawdown", "profit_factor", "round_trips_per_day",
        "fills_per_day", "avg_trade_return", "med_trade_return", "avg_hold_hours",
        "med_hold_hours", "turnover_per_month_x", "turnover_per_year_x",
        "costs_paid", "excess_vs_equal_weight", "excess_vs_btc",
        "trips_per_day_ok", "fills_per_day_ok", "beats_equal_weight",
        "positive_sharpe_net", "passes_all",
    ]
    print("\n" + "=" * 150)
    print("RANKED BY CONSTRAINT COMPLIANCE, NOT HEADLINE RETURN")
    print("=" * 150)
    with pd.option_context("display.width", 400, "display.max_columns", 100,
                           "display.float_format", lambda v: f"{v:,.4g}"):
        print(ranked[cols].to_string(index=False))

    best = ranked.iloc[0]
    print("\n" + "=" * 150)
    print("BEST CONFIGURATION UNDER THE STATED CONSTRAINTS")
    print("=" * 150)
    for key in cols:
        print(f"  {key:<30} {best[key]}")
    print(f"\n  configs meeting every constraint: {int(frame['passes_all'].sum())} of {len(frame)}")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.sort_values("rank_score", ascending=False).to_csv(path, index=False)
        print(f"\nsaved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
