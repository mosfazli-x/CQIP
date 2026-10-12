"""Validation-only event evaluation for one symbol.

Standalone research driver. Trains on the train slice, scores the *validation*
slice only, and reports event selectivity. The holdout slice is never touched:
reading it before a strategy has been selected would spend the one sample that
cannot be reused.

Usage:
    python tools/event_study.py BTC_USDT --threshold 0.02
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cqip import events as E  # noqa: E402
from cqip.backtest import conviction_threshold_sweep  # noqa: E402
from cqip.config import Settings  # noqa: E402
from cqip.features import split_boundaries  # noqa: E402
from cqip.forecaster import ForecasterTrainer  # noqa: E402
from cqip.pipeline import DatasetBuilder  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--kind", default="mlp")
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--k", type=float, default=None)
    parser.add_argument(
        "--no-save", action="store_true",
        help="print only; do not persist a report for the dashboard",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    settings = Settings(log_level="ERROR")
    builder = DatasetBuilder(settings)
    bundle = builder.build(
        args.symbol, args.interval, horizon=args.horizon,
        target="move_event", move_event_threshold=args.threshold,
        move_event_k=args.k,
    )
    frame = bundle.frame.reset_index(drop=True)
    settings_ratios = settings.backtest_train_ratio, settings.backtest_val_ratio
    splits = split_boundaries(
        len(frame), train_ratio=settings_ratios[0], val_ratio=settings_ratios[1],
        horizon=args.horizon, seq_len=args.seq_len,
        embargo_bars=getattr(settings, "embargo_bars", None),
    )
    train = frame.iloc[: splits.val_start].reset_index(drop=True)
    val = frame.iloc[splits.val_start : splits.val_stop].reset_index(drop=True)
    print(f"{args.symbol} {args.interval}: {len(train)} train / {len(val)} val bars "
          f"(holdout of {splits.total - splits.test_start} bars left untouched)")

    trainer = ForecasterTrainer(settings)
    model, info = trainer.train(
        train, val, bundle.feature_columns,
        target_column=bundle.target_spec.regression,
        direction_column=bundle.target_spec.direction,
        aux_direction_weight=0.3,
        seq_len=args.seq_len, horizon=args.horizon,
        kind=args.kind, epochs=args.epochs,
    )
    preds = trainer.predict(model, val, bundle.feature_columns, info, seq_len=args.seq_len)
    preds = preds.dropna(subset=["predicted_return"]).reset_index(drop=True)

    # A sequence model needs `seq_len` bars of history per row, so `predict`
    # returns len(val) - seq_len + 1 rows that line up with the *tail* of the
    # frame. Slicing from the head would pair every prediction with the wrong
    # label and quietly corrupt both the event report and the backtest.
    aligned = val.iloc[-len(preds):].reset_index(drop=True)
    vol = aligned["target_volatility"].to_numpy(dtype=float)
    conviction = (preds["predicted_return"].to_numpy(dtype=float)
                  / np.where(vol > 0, vol, np.nan))
    label = aligned[bundle.target_spec.direction].to_numpy(dtype=float)

    report = E.event_report(conviction, label)
    print(f"\nevent rate on validation: {report['event_rate']:.3f} "
          f"({report['n_events']}/{report['n']} scorable bars)")
    print(f"verdict: {report['verdict']}\n")
    print(f"{'k':>6} {'taken':>7} {'prec':>7} {'CI low':>8} {'CI high':>8} {'lift':>7} {'sig':>5}")
    for row in report["by_k"]:
        print(f"{row['k']:>6.2f} {row['n_taken']:>7} {row['precision']:>7.3f} "
              f"{row['precision_ci_low']:>8.3f} {row['precision_ci_high']:>8.3f} "
              f"{(row['lift'] or 0):>7.2f} {str(row['significant']):>5}")

    ohlc = aligned[["ts", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    rows = conviction_threshold_sweep(ohlc, np.nan_to_num(conviction), settings=settings,
                                      thresholds=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0))
    print(f"\n{'thr':>6} {'cover':>7} {'trades':>7} {'sharpe':>8} {'sortino':>8} "
          f"{'ret%':>9} {'excess%':>9}")
    for r in rows:
        print(f"{r['threshold']:>6.2f} {100 * r['coverage']:>6.1f}% {r['trades']:>7} "
              f"{(r['sharpe'] or 0):>8.2f} {(r['sortino'] or 0):>8.2f} "
              f"{100 * (r['total_return'] or 0):>8.1f}% {100 * (r['excess_return'] or 0):>8.1f}%")

    if not args.no_save:
        record = E.event_study_record(
            symbol=args.symbol, interval=args.interval, model_kind=args.kind,
            horizon=args.horizon, seq_len=args.seq_len, epochs=args.epochs,
            event=report, conviction=rows,
            val_rows=len(val), train_rows=len(train),
            holdout_rows=splits.total - splits.test_start,
            threshold=None if args.k else args.threshold, k=args.k,
        )
        path = E.save_event_report(record, settings.resolved_artifact_dir / "reports")
        print(f"\nsaved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
