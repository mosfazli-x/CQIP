"""Multi-timeframe, multi-scenario research sweeps.

One command runs a matrix of ``interval x model architecture`` cells, so a full
comparative study takes a single instruction instead of a dozen hand-run commands.
Each cell is trained, walk-forward backtested and scored against explicit quality
gates, and the whole matrix is persisted so a sweep can be resumed or compared
against a later run.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from cqip.config import Settings
from cqip.ingest import CANDLES_DATASET, Ingestor
from cqip.logging_config import get_logger
from cqip.pipeline import TrainingPipeline
from cqip.storage import ParquetStore
from cqip.utils import normalize_symbol

logger = get_logger(__name__)

#: Trading-timeframe label -> pandas resample rule.
INTERVAL_RULES: dict[str, str] = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1D",
    "1w": "1W",
}

#: Bar duration in minutes, used to detect partially-filled buckets.
INTERVAL_MINUTES: dict[str, int] = {
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "12h": 720,
    "1d": 1_440,
    "1w": 10_080,
}

#: Coarse labels must always reduce to a whole number of base bars.
COARSEN_ONLY: frozenset[str] = frozenset({"2h", "4h", "6h", "12h", "1d", "1w"})

CellStatus = Literal["ok", "skipped", "insufficient_data", "failed"]


def interval_minutes(interval: str) -> int:
    try:
        return INTERVAL_MINUTES[interval]
    except KeyError:
        raise ValueError(
            f"Unsupported interval {interval!r}; expected one of {sorted(INTERVAL_MINUTES)}"
        ) from None


def resample_ohlcv(
    frame: pd.DataFrame,
    *,
    interval: str,
    symbol: str,
    exchange: str,
    base_interval: str = "1h",
) -> pd.DataFrame:
    """Aggregate a base-interval OHLCV series into a coarser interval.

    Buckets that are not fully covered by base bars are **dropped**. A partial bar
    would carry a close taken mid-move, which both misstates the series and leaks
    the future into anything trained on it. The first and last buckets are the
    usual casualties.
    """
    if frame.empty:
        return frame.iloc[0:0].copy()

    rule = INTERVAL_RULES[interval]
    ratio = interval_minutes(interval) / interval_minutes(base_interval)
    if ratio < 1 or not float(ratio).is_integer():
        raise ValueError(
            f"Cannot build {interval} bars from {base_interval} data: "
            f"{interval} is not a whole multiple of {base_interval}"
        )

    working = frame.copy()
    working["ts"] = pd.to_datetime(working["ts"], utc=True)
    working = working.sort_values("ts").drop_duplicates(subset="ts", keep="last")
    grouped = working.set_index("ts")

    aggregations: dict[str, tuple[str, str]] = {
        "open": ("open", "first"),
        "high": ("high", "max"),
        "low": ("low", "min"),
        "close": ("close", "last"),
    }
    for column in ("volume", "quote_volume", "trades"):
        if column in grouped.columns:
            aggregations[column] = (column, "sum")

    resampled = grouped.resample(rule).agg(**aggregations)
    filled = grouped.resample(rule).size()
    complete = filled.index[filled >= ratio]
    resampled = resampled.loc[complete].dropna(subset=["close"]).reset_index()

    resampled["symbol"] = normalize_symbol(symbol)
    resampled["exchange"] = exchange
    resampled["interval"] = interval
    return resampled


@dataclass
class QualityGate:
    """Thresholds a cell must clear before its result means anything.

    The gate fails **closed**: a required metric that is missing or non-finite
    blocks the verdict rather than being skipped. A gate that passes on absent
    data is worse than no gate, because it manufactures confidence.

    ``evaluate`` is given an explicit *metric source* rather than reading a
    cell's fields directly. Selection must run on validation data: ranking on the
    final holdout turns the holdout into a training signal, and every subsequent
    holdout number is then contaminated by the fact that it was chosen on.

    Which thresholds apply depends on the source, and the split is deliberate:

    * **validation** enforces forecast quality only -- directional accuracy,
      correlation, and the accuracy interval. These are the numbers a sweep is
      allowed to rank on, and they are computable without ever touching the
      holdout.
    * **holdout** additionally enforces the trading thresholds (trade count,
      drawdown, excess over buy-and-hold), because those only mean anything out
      of sample, against a real fee-aware fill.

    Strategy metrics used to be passed through from whichever full-sample
    backtest happened to exist, which meant a "validation" verdict was silently
    being decided partly by holdout-period drawdown. Ranking on the holdout is
    the specific failure this whole refactor exists to prevent.
    """

    min_direction_accuracy: float = 0.55
    min_correlation: float = 0.05
    min_trades: int = 30
    min_excess_sharpe: float = 0.0
    max_drawdown_floor: float = -0.5
    #: When set, a cell whose accuracy confidence interval includes 0.5 is not
    #: promoted, because the point estimate is then indistinguishable from a coin
    #: flip and the sample is too small to say otherwise.
    require_ci_above_chance: bool = True
    min_ci_lower: float | None = None

    def evaluate(
        self, cell: dict[str, Any], *, source: str = "validation"
    ) -> tuple[str, str]:
        metrics = cell_metrics_for(cell, source)
        accuracy = metrics.get("direction_accuracy")
        correlation = metrics.get("correlation")
        trades = metrics.get("trades")
        drawdown = metrics.get("max_drawdown")
        excess = metrics.get("excess_vs_buy_and_hold")
        # Risk and return are only assertable out of sample. Demanding them at
        # validation time would either force a holdout read or, worse, quietly
        # substitute a number computed over the holdout.
        check_strategy = source == "holdout"

        required = [
            ("direction_accuracy", accuracy),
            ("correlation", correlation),
        ]
        if check_strategy:
            required += [
                ("trades", trades),
                ("max_drawdown", drawdown),
            ]
        missing = [name for name, value in required if value is None]
        if missing:
            # Fails closed, but the verdict says *why*. A cell with no metrics
            # has not been shown to lack an edge; it has not been shown to have
            # one either, and "no-edge" would claim a negative result that was
            # never measured.
            return "insufficient_data", f"missing required metrics: {', '.join(missing)}"

        reasons: list[str] = []
        if accuracy < 0.5:
            reasons.append(f"dir_acc {accuracy:.3f} < 0.5 (worse than a coin flip)")
        elif accuracy < self.min_direction_accuracy:
            reasons.append(f"dir_acc {accuracy:.3f} < {self.min_direction_accuracy}")
        if correlation < self.min_correlation:
            reasons.append(f"corr {correlation:+.4f} < {self.min_correlation}")
        if check_strategy:
            if trades < self.min_trades:
                reasons.append(f"{trades} trades < {self.min_trades} (not significant)")
            if drawdown < self.max_drawdown_floor:
                reasons.append(
                    f"drawdown {drawdown:.1%} worse than {self.max_drawdown_floor:.0%}"
                )
            if excess is not None and excess < self.min_excess_sharpe:
                reasons.append(f"underperforms buy-and-hold by {-excess:.2f} Sharpe")

        interval = metrics.get("direction_accuracy_ci") or {}
        lower = _coerce_float(interval.get("lower"))
        if self.require_ci_above_chance:
            # A point estimate of 0.56 whose interval spans 0.50 is not evidence
            # of skill. Without this check a sweep promotes whichever cells got
            # the luckiest small sample, which is the whole failure mode.
            #
            # A *missing* interval is treated as a failure rather than a pass.
            # The old `lower is not None` guard let a cell with no CI at all
            # clear the gate, so the one scenario that needs scrutiny most was
            # the one that sailed through.
            if lower is None:
                reasons.append(
                    f"dir_acc has no 95% CI ({source}); cannot establish edge above chance"
                )
            elif lower < 0.5:
                reasons.append(
                    f"dir_acc 95% CI [{lower:.3f}, {_fmt_upper(interval)}] "
                    "includes 0.50 (indistinguishable from chance)"
                )
        floor = self.min_ci_lower if self.min_ci_lower is not None else self.min_direction_accuracy
        if lower is not None and lower < floor:
            reasons.append(f"dir_acc CI lower {lower:.3f} < {floor}")

        if not reasons:
            return "passes", f"all gates cleared ({source})"
        return "no-edge", "; ".join(reasons)


@dataclass
class SweepCell:
    interval: str
    kind: str
    symbol: str
    status: CellStatus = "ok"
    verdict: str = ""
    reason: str = ""
    model_id: str | None = None
    rows: int | None = None
    direction_accuracy: float | None = None
    correlation: float | None = None
    interval_coverage: float | None = None
    sharpe: float | None = None
    total_return: float | None = None
    max_drawdown: float | None = None
    trades: int | None = None
    consistency: float | None = None
    excess_vs_buy_and_hold: float | None = None
    buy_and_hold_sharpe: float | None = None
    buy_and_hold_return: float | None = None
    duration_seconds: float = 0.0

    # -- validation (used for selection) ------------------------------------
    #: Metrics from the inner validation split, which is the only data selection
    #: is permitted to look at.
    #: Field names mirror the metric keys produced by the pipeline
    #: (``val_directional_accuracy``, ``test_directional_accuracy``, ...). A
    #: near-miss such as ``val_direction_accuracy`` is not cosmetic: the holdout
    #: absorb step copies metrics by name, so a mismatched field is silently
    #: dropped and a holdout verdict is computed from an empty metric set.
    val_directional_accuracy: float | None = None
    val_correlation: float | None = None
    val_mae: float | None = None
    val_directional_accuracy_ci: dict[str, float] | None = None
    val_correlation_ci: dict[str, float] | None = None
    val_ensemble_spread: float | None = None
    #: Validation-window trading statistics. Populated only when a backtest was
    #: run *restricted to the validation window*; left None during ordinary
    #: screening, because the available full-sample backtest spans the holdout.
    val_trades: int | None = None
    val_max_drawdown: float | None = None
    val_excess_vs_buy_and_hold: float | None = None

    # -- final holdout (reported, never selected on) -------------------------
    #: Metrics from the untouched final test split. Recorded for the report, and
    #: evaluated by the gate only when explicitly asked for a holdout verdict.
    test_directional_accuracy: float | None = None
    test_correlation: float | None = None
    test_mae: float | None = None
    test_directional_accuracy_ci: dict[str, float] | None = None
    test_correlation_ci: dict[str, float] | None = None
    #: Holdout-window trading statistics, from a backtest restricted to the test
    #: window. These are the only risk/return figures a holdout verdict may cite.
    test_trades: int | None = None
    test_max_drawdown: float | None = None
    test_excess_vs_buy_and_hold: float | None = None
    #: True only once the holdout has actually been scored. It is set by
    #: `ResearchSweep.evaluate_holdout`, which calls
    #: `TrainingPipeline.score_holdout`; nothing else may set it, because a flag
    #: that can be flipped by hand is not evidence of anything.
    holdout_scored: bool = False

    @property
    def key(self) -> str:
        return f"{self.symbol}|{self.interval}|{self.kind}"

    @property
    def selection_score(self) -> float:
        """Ranking score. Validation-only by construction.

        Deliberately not overridable: any hook that let a caller rank on the
        holdout would reintroduce exactly the leak this guards against.
        """
        accuracy = self.val_directional_accuracy
        correlation = self.val_correlation
        if accuracy is None or correlation is None:
            return float("-inf")
        # Accuracy dominates, correlation breaks ties and rewards genuine
        # magnitude agreement rather than sign agreement alone.
        return float(accuracy) + float(correlation)


def cell_metrics_for(cell: dict[str, Any], source: str) -> dict[str, Any]:
    """Map a cell to the metric set the gate should read.

    ``source="validation"`` is the default and the only correct choice for
    ranking. ``source="holdout"`` is for reporting a selected model's final
    performance. ``source="auto"`` prefers validation and falls back to the
    flat legacy fields, which only makes sense for reports written before this
    split existed.

    Every metric is namespaced by split, including the strategy ones
    (``trades``, drawdown, excess over buy-and-hold). They are deliberately
    **not** filled in from the un-prefixed flat fields: those come from a
    walk-forward backtest spanning the entire frame, holdout included, so
    reading them under ``source="validation"`` would let holdout-period drawdown
    decide a validation verdict. Ranking on the holdout is the specific failure
    this whole refactor exists to prevent.
    """
    if source == "validation":
        return {
            "direction_accuracy": cell.get("val_directional_accuracy"),
            "correlation": cell.get("val_correlation"),
            "direction_accuracy_ci": cell.get("val_directional_accuracy_ci"),
            "correlation_ci": cell.get("val_correlation_ci"),
            "trades": cell.get("val_trades"),
            "max_drawdown": cell.get("val_max_drawdown"),
            "excess_vs_buy_and_hold": cell.get("val_excess_vs_buy_and_hold"),
        }
    if source == "holdout":
        return {
            "direction_accuracy": cell.get("test_directional_accuracy"),
            "correlation": cell.get("test_correlation"),
            "direction_accuracy_ci": cell.get("test_directional_accuracy_ci"),
            "correlation_ci": cell.get("test_correlation_ci"),
            "trades": cell.get("test_trades"),
            "max_drawdown": cell.get("test_max_drawdown"),
            "excess_vs_buy_and_hold": cell.get("test_excess_vs_buy_and_hold"),
        }
    if source == "auto":
        if cell.get("val_directional_accuracy") is not None:
            return cell_metrics_for(cell, "validation")
        if cell.get("test_directional_accuracy") is not None:
            return cell_metrics_for(cell, "holdout")
        # Pre-split legacy reports only ever had flat fields.
        return {
            "direction_accuracy": cell.get("direction_accuracy"),
            "correlation": cell.get("correlation"),
        }
    raise ValueError(f"unknown metric source {source!r}")


@dataclass
class SweepReport:
    symbols: list[str]
    intervals: list[str]
    kinds: list[str]
    folds: int
    epochs: int
    started_at: str
    finished_at: str = ""
    duration_seconds: float = 0.0
    cells: list[SweepCell] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cells"] = [asdict(cell) for cell in self.cells]
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @property
    def passed(self) -> list[SweepCell]:
        return [cell for cell in self.cells if cell.verdict == "passes"]


def _coerce_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _absorb_test_metrics(cell: SweepCell, metrics: dict[str, Any]) -> None:
    """Copy a ``score_holdout`` payload onto a cell's ``test_*`` fields.

    Only keys that already exist as declared cell fields are written, so an
    unexpected metric cannot silently become part of the cell's state.

    A ``test_*`` key with no matching field is almost never intentional: it means
    the cell and the pipeline disagree about a metric's name, and the gate would
    then judge a holdout on a half-empty metric set while reporting a confident
    verdict. Those keys are therefore collected and logged, so a rename that
    breaks the mapping is loud rather than invisible.
    """
    unmatched: list[str] = []
    for key, value in metrics.items():
        if not str(key).startswith("test_"):
            continue
        if not hasattr(cell, key):
            unmatched.append(str(key))
            continue
        if str(key).endswith("_ci"):
            setattr(cell, key, _as_ci(value))
        else:
            setattr(cell, key, value)
    if unmatched:
        logger.warning(
            "sweep_holdout_metric_unmapped",
            extra={
                "cell": cell.key,
                "unmatched": sorted(unmatched)[:10],
                "reason": "metric key has no matching SweepCell field; "
                          "it was dropped rather than gated on",
            },
        )


def _fmt_upper(interval: dict[str, Any]) -> str:
    """Render an interval's upper bound without ever printing a bare ``nan``."""
    upper = _coerce_float(interval.get("upper"))
    return "?" if upper is None else f"{upper:.3f}"


def _as_ci(value: Any) -> dict[str, float] | None:
    """Normalise a confidence-interval payload, dropping non-finite bounds."""
    if not isinstance(value, dict):
        return None
    lower = _coerce_float(value.get("lower"))
    upper = _coerce_float(value.get("upper"))
    if lower is None or upper is None:
        return None
    return {"lower": lower, "upper": upper}


class ResearchSweep:
    """Run a matrix of timeframe x architecture studies over the stored data."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        engine: Any = None,
        store: ParquetStore | None = None,
    ) -> None:
        from cqip.db import get_engine

        self.settings = settings or Settings()
        self.engine = engine or get_engine(self.settings)
        # Pass the resolved path explicitly: a bare None would make ParquetStore
        # fall back to the *global* settings, bypassing injected configuration.
        self.store = store or ParquetStore(self.settings.resolved_parquet_dir)
        self.ingestor = Ingestor(self.settings, engine=self.engine, store=self.store)
        self.pipeline = TrainingPipeline(settings=self.settings, engine=self.engine)
        self.gate = QualityGate()

    # -- data preparation ---------------------------------------------------

    def ensure_interval(
        self,
        symbol: str,
        interval: str,
        *,
        base_interval: str = "1h",
        exchange: str | None = None,
    ) -> int:
        """Make sure ``interval`` bars exist for ``symbol``, resampling if needed.

        Coarse timeframes are derived from the base series rather than fetched, so
        a sweep is reproducible offline and every timeframe sees identical prices.
        Returns the number of rows available at ``interval``.
        """
        wanted = normalize_symbol(symbol)
        stored = self.store.read(
            CANDLES_DATASET,
            filters=[("symbol", "==", wanted), ("interval", "==", interval)],
        )
        if not stored.empty:
            return len(stored)

        if interval_minutes(interval) <= interval_minutes(base_interval):
            raise ValueError(
                f"No {interval} data for {wanted} and it cannot be derived from "
                f"{base_interval} data. Ingest {interval} explicitly."
            )

        base = self.store.read(
            CANDLES_DATASET,
            filters=[("symbol", "==", wanted), ("interval", "==", base_interval)],
        )
        if base.empty:
            raise ValueError(
                f"No {base_interval} data for {wanted}. Ingest candles first."
            )

        provider = exchange or self.settings.default_exchange
        primary = base[base["exchange"] == provider]
        if not primary.empty:
            base = primary

        resampled = resample_ohlcv(
            base,
            interval=interval,
            symbol=wanted,
            exchange=str(base["exchange"].iloc[0]),
            base_interval=base_interval,
        )
        if resampled.empty:
            return 0
        self.store.write(
            CANDLES_DATASET,
            resampled,
            partition_by=("year", "month"),
            dedupe_on=("ts", "symbol", "exchange", "interval"),
            mode="upsert",
        )
        return len(resampled)

    # -- execution ----------------------------------------------------------

    def _required_bars(self, horizon: int = 1) -> int:
        """Estimate bars needed before a cell is worth training.

        Feature construction drops a leading lookback, the chronological split
        keeps only 70% for training, and each sample consumes `seq_len` bars plus
        the prediction horizon. The test split must also hold at least one window.
        """
        seq_len = int(self.settings.forecaster_seq_len)
        # Longest rolling window in build_feature_frame; rows before it cannot
        # produce a feature value and are dropped.
        lookback = 60
        # The chronological split is 70/15/15, so val and test each keep
        # `tail_ratio` of the usable rows. The binding constraint is that the
        # *smallest* of the two (test) still has to fit a full window plus the
        # horizon, otherwise evaluation silently has nothing to score.
        tail_ratio = 0.15
        usable_needed = int(-(-seq_len // tail_ratio))  # ceil division
        return lookback + horizon + usable_needed

    def run_cell(
        self,
        symbol: str,
        interval: str,
        kind: str,
        *,
        epochs: int,
        folds: int,
        horizon: int = 1,
        threshold: float = 0.0,
        min_rows: int = 300,
        evaluate_holdout: bool = False,
    ) -> SweepCell:
        """Train and score one matrix cell.

        ``evaluate_holdout`` defaults to **False**. Screening compares hundreds
        of configurations, and a holdout score produced for each of them is a
        holdout that has been used hundreds of times to choose between models.
        The holdout is measured later, once, by :meth:`evaluate_holdout` on the
        single cell that selection chose.
        """
        cell = SweepCell(interval=interval, kind=kind, symbol=normalize_symbol(symbol))
        started = time.perf_counter()
        try:
            rows = self.ensure_interval(symbol, interval)
            cell.rows = rows
            if rows < min_rows:
                cell.status = "insufficient_data"
                cell.verdict = "insufficient_data"
                cell.reason = (
                    f"{rows} {interval} bars < {min_rows} minimum; "
                    f"ingest more history or lower --min-rows"
                )
                cell.duration_seconds = time.perf_counter() - started
                return cell

            # Raw bars are not usable rows: feature lookback discards the head and
            # the 70/15/15 split then has to fit `seq_len` windows in each part.
            # Checking this up front keeps a coarse timeframe from burning a full
            # training run before failing on window construction.
            required = self._required_bars(horizon)
            if rows < required:
                cell.status = "insufficient_data"
                cell.verdict = "insufficient_data"
                cell.reason = (
                    f"{rows} {interval} bars cannot support training: about {required} "
                    f"are needed after feature lookback, the chronological split and a "
                    f"{self.settings.forecaster_seq_len}-bar window; ingest more history"
                )
                cell.duration_seconds = time.perf_counter() - started
                return cell

            trained = self.pipeline.train_forecaster(
                symbol, interval, horizon=horizon, kind=kind, epochs=epochs,
                evaluate_holdout=evaluate_holdout,
            )
            cell.model_id = str(trained["model_id"])

            # Validation metrics drive selection; holdout metrics are recorded but
            # never ranked on. `train_forecaster` already splits 70/15/15 with a
            # purged boundary, so the test frame it reports on is disjoint from
            # everything the trainer read.
            # `val_metrics` carries two different vocabularies: unprefixed
            # training-time metrics from the trainer, and `val_`-prefixed metrics
            # from the scoring pass that also owns the confidence intervals.
            # The prefixed ones must be preferred -- reading the unprefixed
            # `directional_accuracy` yields a point estimate with no interval
            # attached, which is unusable now that the gate fails closed.
            val = dict(trained.get("val_metrics") or {})
            cell.val_directional_accuracy = _coerce_float(
                val.get("val_directional_accuracy", val.get("directional_accuracy"))
            )
            cell.val_correlation = _coerce_float(
                val.get("val_correlation", val.get("correlation"))
            )
            cell.val_mae = _coerce_float(val.get("val_mae", val.get("mae")))
            cell.val_directional_accuracy_ci = _as_ci(val.get("val_directional_accuracy_ci"))
            cell.val_correlation_ci = _as_ci(val.get("val_correlation_ci"))
            ensemble_info = dict(trained.get("ensemble") or {})
            cell.val_ensemble_spread = _coerce_float(ensemble_info.get("std_directional_accuracy"))

            test = dict(trained.get("test_metrics") or {})
            cell.test_directional_accuracy = _coerce_float(test.get("test_directional_accuracy"))
            cell.test_correlation = _coerce_float(test.get("test_correlation"))
            cell.test_mae = _coerce_float(test.get("test_mae"))
            cell.test_directional_accuracy_ci = _as_ci(
                test.get("test_directional_accuracy_ci")
            )
            cell.test_correlation_ci = _as_ci(test.get("test_correlation_ci"))

            # Legacy flat fields mirror validation when it exists so older
            # consumers and saved reports keep meaning "the selection metric".
            cell.direction_accuracy = cell.val_directional_accuracy
            cell.correlation = cell.val_correlation
            cell.interval_coverage = _coerce_float(
                val.get("val_68pct_interval_coverage", test.get("test_68pct_interval_coverage"))
            )

            # The walk-forward backtest refits on each fold's training window and
            # trades the remainder, which means it *reads the holdout*. Running it
            # during screening would both leak and cost a refit per fold per cell,
            # so it is deferred to the holdout stage alongside the other
            # out-of-sample measurements. Screening ranks on validation forecast
            # metrics only.
            if evaluate_holdout:
                # Holdout risk figures must come from a backtest restricted to
                # the test window. `backtest_predictions` is a walk-forward run
                # over the *whole* frame that refits per fold, so its drawdown
                # and trade count span train and validation too; writing those
                # into `test_*` would present holdout-inclusive risk as
                # holdout-window risk, which is the exact confusion the split
                # exists to prevent.
                holdout = self.pipeline.score_holdout(
                    cell.model_id, threshold=threshold
                )
                cell.test_trades = int(holdout.get("test_trades") or 0)
                cell.test_max_drawdown = _coerce_float(holdout.get("test_max_drawdown"))
                cell.test_excess_vs_buy_and_hold = _coerce_float(
                    holdout.get("test_excess_vs_buy_and_hold")
                )

                result = self.pipeline.backtest_predictions(
                    cell.model_id, threshold=threshold, n_folds=folds
                )
                backtest = dict(result.get("metrics") or {})
                failure = result.get("error")
                if failure:
                    cell.status = "failed"
                    cell.verdict = "failed"
                    cell.reason = f"backtest produced no results: {failure}"
                    cell.duration_seconds = time.perf_counter() - started
                    return cell
                # Un-prefixed fields stay full-sample walk-forward statistics and
                # are kept only for backward compatibility with older reports and
                # the CLI table. The gate reads `test_*`/`val_*` only, so these
                # cannot influence a verdict.
                cell.sharpe = _coerce_float(backtest.get("sharpe"))
                cell.total_return = _coerce_float(backtest.get("total_return"))
                cell.consistency = _coerce_float(backtest.get("consistency"))
                cell.max_drawdown = _coerce_float(backtest.get("max_drawdown"))
                cell.excess_vs_buy_and_hold = _coerce_float(
                    backtest.get("excess_vs_buy_and_hold")
                )
                cell.trades = int(backtest.get("trades") or 0)

                benchmarks = dict(result.get("benchmarks") or {})
                hold = dict(benchmarks.get("buy_and_hold") or {})
                cell.buy_and_hold_sharpe = _coerce_float(hold.get("sharpe"))
                cell.buy_and_hold_return = _coerce_float(hold.get("total_return"))

            cell.verdict, cell.reason = self.gate.evaluate(
                asdict(cell), source="holdout" if evaluate_holdout else "validation"
            )
        except ValueError as exc:
            # A timeframe that cannot supply windows is a data limitation, not a
            # defect, so it must not be reported as a failed cell.
            message = str(exc)
            if "not enough rows" in message.lower() or "window" in message.lower():
                cell.status = "insufficient_data"
                cell.verdict = "insufficient_data"
                cell.reason = (
                    f"{rows} {interval} bars do not survive feature lookback and the "
                    f"train/val/test split ({message[:160]}); ingest more history"
                )
                logger.warning(
                    "sweep_cell_insufficient_data",
                    extra={"symbol": symbol, "interval": interval, "kind": kind},
                )
            else:
                cell.status = "failed"
                cell.verdict = "failed"
                cell.reason = f"{type(exc).__name__}: {message[:200]}"
                logger.error(
                    "sweep_cell_failed",
                    extra={"symbol": symbol, "interval": interval, "kind": kind,
                           "error": message[:200]},
                )
        except Exception as exc:  # noqa: BLE001 - one cell must not kill the sweep
            cell.status = "failed"
            cell.verdict = "failed"
            cell.reason = f"{type(exc).__name__}: {str(exc)[:200]}"
            logger.error(
                "sweep_cell_failed",
                extra={"symbol": symbol, "interval": interval, "kind": kind,
                       "error": str(exc)[:200]},
            )
        cell.duration_seconds = time.perf_counter() - started
        return cell

    def run(
        self,
        *,
        symbols: Sequence[str],
        intervals: Sequence[str],
        kinds: Sequence[str],
        epochs: int = 5,
        folds: int = 3,
        horizon: int = 1,
        threshold: float = 0.0,
        min_rows: int = 300,
        resume_from: SweepReport | None = None,
        progress: Any = None,
    ) -> SweepReport:
        """Execute every ``symbol x interval x kind`` combination."""
        intervals = list(dict.fromkeys(intervals))
        for interval in intervals:
            if interval not in INTERVAL_RULES:
                raise ValueError(
                    f"Unsupported interval {interval!r}; expected one of "
                    f"{sorted(INTERVAL_RULES)}"
                )

        report = SweepReport(
            symbols=[normalize_symbol(s) for s in symbols],
            intervals=intervals,
            kinds=list(kinds),
            folds=folds,
            epochs=epochs,
            started_at=datetime.now(UTC).isoformat(),
        )
        completed: dict[str, SweepCell] = {}
        if resume_from is not None:
            completed = {cell.key: cell for cell in resume_from.cells if cell.status == "ok"}
            logger.info("sweep_resume", extra={"reusable": len(completed)})

        started = time.perf_counter()
        for symbol in report.symbols:
            for interval in intervals:
                for kind in kinds:
                    cell = SweepCell(
                        interval=interval, kind=kind, symbol=symbol
                    )
                    if cell.key in completed:
                        logger.info("sweep_cell_skipped", extra={"cell": cell.key})
                        report.cells.append(completed[cell.key])
                        continue
                    logger.info("sweep_cell_start", extra={"cell": cell.key})
                    result = self.run_cell(
                        symbol, interval, kind,
                        epochs=epochs, folds=folds, horizon=horizon,
                        threshold=threshold, min_rows=min_rows,
                    )
                    report.cells.append(result)
                    if progress is not None:
                        progress(result)
        report.duration_seconds = time.perf_counter() - started
        report.finished_at = datetime.now(UTC).isoformat()
        return report

    def run_staged(
        self,
        *,
        symbols: Sequence[str],
        intervals: Sequence[str],
        kinds: Sequence[str],
        screen_epochs: int = 1,
        epochs: int = 30,
        folds: int = 3,
        horizon: int = 1,
        threshold: float = 0.0,
        min_rows: int = 300,
        survivors: int = 5,
        progress: Any = None,
    ) -> tuple[SweepReport, SweepReport]:
        """Screen every cell cheaply, then train only the promising ones.

        A full matrix over many symbols and timeframes is tens of hours on CPU,
        and almost all of it goes to cells that cannot clear the gates because
        the features carry no signal for that symbol. Screening costs one epoch
        per cell and ranks the survivors by directional accuracy, so the real
        training budget goes where it can change the outcome.

        Neither stage measures the holdout: both call ``run_cell`` with
        ``evaluate_holdout=False``. The returned reports are validation reports.
        Call :meth:`evaluate_holdout` on the final selection to score the test
        split, once.

        Returns the screening report and the full-training report.
        """
        screen = self.run(
            symbols=symbols, intervals=intervals, kinds=kinds,
            epochs=screen_epochs, folds=folds, horizon=horizon,
            threshold=threshold, min_rows=min_rows, progress=progress,
        )
        self.save(screen, label="screen")

        ranked = [
            c for c in screen.cells
            if c.status == "ok"
            and c.val_directional_accuracy is not None
            and c.val_correlation is not None
            # Below chance accuracy is a result, not a near miss, so it cannot
            # be rescued by more epochs.
            and c.val_directional_accuracy >= 0.5
        ]
        # Rank on validation only. `selection_score` reads `val_*` exclusively, and
        # screening never measures the holdout, so the promotion list is decided
        # without the test split entering the process at all.
        ranked.sort(key=lambda c: c.selection_score, reverse=True)
        chosen = ranked[:max(0, survivors)]

        logger.info(
            "sweep_screen_complete",
            extra={"screened": len(screen.cells), "viable": len(ranked),
                   "promoted": len(chosen)},
        )

        full = SweepReport(
            symbols=list(dict.fromkeys(c.symbol for c in chosen)),
            intervals=list(intervals),
            kinds=list(kinds),
            folds=folds,
            epochs=epochs,
            started_at=datetime.now(UTC).isoformat(),
        )
        started = time.perf_counter()
        for cell in chosen:
            logger.info("sweep_cell_promote", extra={"cell": cell.key})
            result = self.run_cell(
                cell.symbol, cell.interval, cell.kind,
                epochs=epochs, folds=folds, horizon=horizon,
                threshold=threshold, min_rows=min_rows,
            )
            full.cells.append(result)
            if progress is not None:
                progress(result)
        full.duration_seconds = time.perf_counter() - started
        full.finished_at = datetime.now(UTC).isoformat()
        return screen, full

    def evaluate_holdout(
        self, selected: Sequence[SweepCell], *, threshold: float = 0.0,
        source: str = "holdout",
    ) -> list[SweepCell]:
        """Score the final holdout for models already chosen on validation.

        This is the only place a holdout verdict is produced, and it runs after
        selection has frozen. Screening and promotion both leave
        ``holdout_scored`` False, so a report claiming a passing holdout gate can
        only have come through here.

        The holdout is *measured* here, not merely re-read: ``score_holdout``
        rebuilds the test split with the recorded horizon and sequence length and
        scores the immutable saved artifact. The previous version only re-gated
        numbers that screening had already computed, which meant every cell had
        already touched the holdout long before selection finished.

        ``source`` defaults to ``"holdout"``. Defaulting it to ``"validation"``
        would mark ``holdout_scored`` True while judging validation metrics,
        which is a flag that lies about what was measured.

        ``threshold`` is the decision rule the holdout is traded with. Pass the
        same value used for the backtest: grading a different rule than the one
        under evaluation would report risk figures for a strategy nobody runs.
        """
        if source != "holdout":
            raise ValueError(
                "evaluate_holdout must judge the holdout; pass source='holdout'. "
                "A validation verdict here would set holdout_scored=True on "
                "numbers that never touched the test split."
            )
        if not getattr(self, "pipeline", None):
            raise RuntimeError(
                "ResearchSweep has no pipeline attached, so it cannot score a "
                "holdout; pass pipeline=... when constructing the sweep"
            )

        scored: list[SweepCell] = []
        for cell in selected:
            if not cell.model_id:
                logger.warning(
                    "sweep_holdout_skipped", extra={"cell": cell.key, "reason": "no model_id"}
                )
                scored.append(cell)
                continue
            try:
                metrics = self.pipeline.score_holdout(cell.model_id, threshold=threshold)
            except Exception as exc:  # noqa: BLE001
                # One unscorable model must not abort the report for the rest.
                cell.status = "failed"
                cell.verdict = "failed"
                cell.reason = f"holdout scoring failed: {str(exc)[:200]}"
                logger.error(
                    "sweep_holdout_failed",
                    extra={"cell": cell.key, "error": str(exc)[:300]},
                )
                scored.append(cell)
                continue

            _absorb_test_metrics(cell, metrics)
            verdict, reason = self.gate.evaluate(asdict(cell), source="holdout")
            cell.verdict = verdict
            cell.reason = reason
            cell.holdout_scored = True
            logger.info(
                "sweep_holdout_evaluated",
                extra={
                    "cell": cell.key,
                    "verdict": verdict,
                    "directional_accuracy": metrics.get("test_directional_accuracy"),
                },
            )
            scored.append(cell)
        return scored

    # -- persistence --------------------------------------------------------

    def report_dir(self) -> Path:
        path = self.settings.resolved_artifact_dir / "reports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, report: SweepReport, label: str | None = None) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        prefix = f"sweep-{label}-{stamp}" if label else f"sweep-{stamp}"
        target = self.report_dir() / f"{prefix}-{uuid.uuid4().hex[:6]}.json"
        target.write_text(report.to_json(), encoding="utf-8")
        # `sweep-latest.json` must keep tracking the most recent run, but a
        # screening pass must not overwrite the full result it is a filter for.
        if label != "screen":
            latest = self.report_dir() / "sweep-latest.json"
            latest.write_text(report.to_json(), encoding="utf-8")
        return target

    def load_latest(self) -> SweepReport | None:
        latest = self.report_dir() / "sweep-latest.json"
        if not latest.exists():
            return None
        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        cells = [SweepCell(**cell) for cell in payload.get("cells", [])]
        return SweepReport(
            symbols=payload.get("symbols", []),
            intervals=payload.get("intervals", []),
            kinds=payload.get("kinds", []),
            folds=payload.get("folds", 3),
            epochs=payload.get("epochs", 0),
            started_at=payload.get("started_at", ""),
            finished_at=payload.get("finished_at", ""),
            duration_seconds=payload.get("duration_seconds", 0.0),
            cells=cells,
        )
