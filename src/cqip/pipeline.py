"""End-to-end training and evaluation pipelines.

``TrainingPipeline`` composes ingestion -> features -> DL forecaster -> RL agent
-> walk-forward backtest, persisting every artifact and metric to the registry so
each run is reproducible and auditable.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cqip.backtest import (
    BacktestEngine,
    WalkForwardValidator,
    compute_metrics,
    forecast_metric_cis,
    periods_per_year,
)
from cqip.config import Settings, get_settings
from cqip.db import Engine, get_engine, to_jsonable
from cqip.features import (
    build_feature_frame,
    feature_quality_report,
    make_targets,
    split_boundaries,
    split_frame,
)
from cqip.forecaster import ForecasterArtifact, ForecasterTrainer, new_model_id
from cqip.ingest import Ingestor
from cqip.labeling import TripleBarrierConfig, build_target_frame
from cqip.logging_config import get_logger
from cqip.rl_env import TradingEnv
from cqip.tracking import ExperimentTracker, ModelRegistry

logger = get_logger(__name__)


def _excess_return(strategy: dict[str, Any], benchmark: dict[str, Any]) -> float | None:
    """Strategy minus buy-and-hold return, or None if either side is unusable.

    Reported in return units rather than a ratio, because "the strategy made 4%
    less than holding" is the sentence a reader needs; "it returned 0.98x the
    benchmark" invites the reader to do the subtraction themselves and get it
    subtly wrong.
    """
    strategy_return = strategy.get("total_return")
    benchmark_return = benchmark.get("total_return")
    if strategy_return is None or benchmark_return is None:
        return None
    try:
        gap = float(strategy_return) - float(benchmark_return)
    except (TypeError, ValueError):
        return None
    return gap if np.isfinite(gap) else None


def resolve_triple_barrier_config(
    settings: Settings,
    params: dict[str, Any] | None = None,
) -> TripleBarrierConfig | None:
    """Rebuild a run's triple-barrier configuration from a registry record.

    The per-fold refit has to reproduce the label definition the headline model
    was trained on, otherwise the backtest grades a different forecasting problem
    than the one that was validated. Records written before triple-barrier
    support have no ``target`` key and correctly yield ``None``.
    """
    params = params or {}
    if str(params.get("target") or "log_return") != "triple_barrier":
        return None
    # The barrier widths live in `target_detail`; a flat top-level key is
    # accepted as a fallback for hand-written records. Reading only the flat key
    # would silently rebuild the run with the *settings* barriers, and the
    # backtest would then grade a different labelling problem than the one that
    # was validated - which looks like a working pipeline, right up until the
    # numbers disagree with the headline model.
    detail = params.get("target_detail") or {}
    if not isinstance(detail, dict):
        detail = {}

    def pick(key: str, default: Any) -> Any:
        value = detail.get(key, params.get(key))
        return default if value is None else value

    return TripleBarrierConfig(
        take_profit_vol=float(
            pick("take_profit_vol", settings.forecaster_triple_barrier_take_profit_vol)
        ),
        stop_loss_vol=float(
            pick("stop_loss_vol", settings.forecaster_triple_barrier_stop_loss_vol)
        ),
        max_hold=pick("max_hold", settings.forecaster_triple_barrier_max_hold),
    )


def resolve_target_kind(
    settings: Settings, params: dict[str, Any] | None = None
) -> tuple[str, bool]:
    """Target kind and volatility-normalisation flag implied by a registry record."""
    params = params or {}
    kind = str(params.get("target") or "log_return")
    normalize = bool(
        params.get("normalize_volatility", settings.forecaster_normalize_volatility)
    )
    if kind == "vol_normalized":
        kind, normalize = "log_return", True
    if kind not in {"log_return", "triple_barrier", "move_event"}:
        kind = "log_return"
    return kind, normalize


def resolve_ensemble_size(settings: Settings, params: dict[str, Any] | None = None) -> int:
    """Ensemble width implied by a registry record, defaulting to the setting."""
    params = params or {}
    return max(1, int(params.get("ensemble_size") or settings.forecaster_ensemble_size))


@dataclass
class DatasetBundle:
    """Assembled feature matrix plus the metadata needed to train on it."""

    symbol: str
    interval: str
    frame: pd.DataFrame
    feature_columns: list[str]
    rows: int
    start: datetime | None = None
    end: datetime | None = None
    target_spec: Any = None
    feature_quality: dict[str, dict[str, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "rows": self.rows,
            "features": len(self.feature_columns),
            "start": self.start,
            "end": self.end,
            "target": self.target_spec.as_dict() if self.target_spec is not None else None,
            "warnings": self.warnings,
        }

    def sparse_features(self, threshold: float = 0.02) -> list[dict[str, float]]:
        """Columns carrying almost no signal, worst first.

        A feature that is zero for 98% of bars is not weak information, it is
        mostly an alignment artefact, and it still consumes a normalised input
        slot. Surfacing it is what makes an explicit ``min_active_fraction``
        pruning decision reviewable instead of a silent change to the schema.
        """
        weak = [
            {"column": name, **stats}
            for name, stats in self.feature_quality.items()
            if stats.get("active_fraction", 1.0) <= threshold
        ]
        return sorted(weak, key=lambda item: item["active_fraction"])


class DatasetBuilder:
    """Loads raw store data and produces a model-ready feature matrix."""

    def __init__(self, settings: Settings | None = None, ingestor: Ingestor | None = None) -> None:
        self.settings = settings or get_settings()
        self.ingestor = ingestor or Ingestor(self.settings)

    def build(
        self,
        symbol: str,
        interval: str = "1h",
        *,
        horizon: int = 1,
        start: datetime | None = None,
        end: datetime | None = None,
        use_news: bool = True,
        use_onchain: bool = True,
        min_rows: int = 300,
        target: str | None = None,
        normalize_volatility: bool | None = None,
    triple_barrier: TripleBarrierConfig | None = None,
    min_active_fraction: float | None = None,
    move_event_threshold: float | None = None,
    move_event_k: float | None = None,
) -> DatasetBundle:
        warnings: list[str] = []

        candles = self.ingestor.load_dataset(symbol, interval, start=start, end=end)
        if candles.empty:
            raise ValueError(f"No OHLCV data for {symbol} {interval}. Run 'cqip ingest' first.")
        if len(candles) < min_rows:
            warnings.append(f"only {len(candles)} rows available (minimum {min_rows})")

        news = self.ingestor.load_news() if use_news else pd.DataFrame()
        if use_news and (news.empty or "sentiment_score" not in news.columns):
            warnings.append("news unavailable or unscored; sentiment features disabled")
            news = pd.DataFrame()

        fear_greed = pd.DataFrame()
        onchain = pd.DataFrame()
        if use_onchain:
            fear_greed = self.ingestor.load_fear_greed()
            onchain = self.ingestor.load_onchain()
            if fear_greed.empty:
                warnings.append("fear & greed index unavailable")

        active_fraction = (
            self.settings.feature_min_active_fraction
            if min_active_fraction is None
            else float(min_active_fraction)
        )
        frame, feature_columns = build_feature_frame(
            candles, news, fear_greed, onchain,
            min_active_fraction=active_fraction,
        )
        if not feature_columns:
            raise ValueError("No usable features were produced; check source data quality")

        quality = feature_quality_report(frame, feature_columns)
        weak = [n for n, s in quality.items() if s["active_fraction"] <= 0.02]
        if weak:
            warnings.append(
                f"{len(weak)} near-constant features retained "
                f"(min_active_fraction={active_fraction:.3f}): {', '.join(sorted(weak)[:6])}"
            )

        kind = target or self.settings.forecaster_target
        normalize = (
            self.settings.forecaster_normalize_volatility
            if normalize_volatility is None
            else bool(normalize_volatility)
        )
        if triple_barrier is None and kind == "triple_barrier":
            triple_barrier = TripleBarrierConfig(
                take_profit_vol=self.settings.forecaster_triple_barrier_take_profit_vol,
                stop_loss_vol=self.settings.forecaster_triple_barrier_stop_loss_vol,
                max_hold=self.settings.forecaster_triple_barrier_max_hold,
            )

        frame = make_targets(frame, horizon=horizon)
        frame, spec = build_target_frame(
            frame,
            horizon=horizon,
            kind=kind,
            normalize_volatility=normalize,
            triple_barrier=triple_barrier,
            move_event_threshold=move_event_threshold,
            move_event_k=move_event_k,
        )

        # Only the regression label gates the dataset. The binary direction label
        # is NaN wherever the bar had no directional information (a triple-barrier
        # time exit, or an exactly flat return), and dropping those rows here
        # would throw away perfectly good regression targets — the aux head masks
        # its own NaNs in the loss.
        frame = frame.dropna(subset=[spec.regression]).reset_index(drop=True)

        if frame.empty:
            raise ValueError("Dataset empty after target construction")

        return DatasetBundle(
            symbol=symbol.upper(), interval=interval, frame=frame,
            feature_columns=feature_columns, rows=len(frame),
            start=frame["ts"].iloc[0].to_pydatetime(),
            end=frame["ts"].iloc[-1].to_pydatetime(),
            target_spec=spec, feature_quality=quality, warnings=warnings,
        )


def resolve_fold_epochs(model_params: dict[str, Any], settings: Settings) -> int:
    """Epoch budget for the per-fold refit inside a walk-forward backtest.

    Folds have to be refit for as long as the real model was trained, otherwise
    `--epochs` silently reaches only the headline model and the backtest grades a
    differently-trained forecaster. Falls back to half the configured budget when
    the record predates that field.
    """
    recorded = int(model_params.get("epochs") or 0)
    return recorded or max(5, settings.forecaster_epochs // 2)


class TrainingPipeline:
    """Runs the full train -> evaluate -> register flow for one symbol."""

    def __init__(
        self,
        settings: Settings | None = None,
        engine: Engine | None = None,
        registry: ModelRegistry | None = None,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.engine = engine or get_engine(self.settings)
        self.registry = registry or ModelRegistry(self.engine, self.settings)
        self.tracker = tracker or ExperimentTracker(self.engine, self.settings)
        self.builder = DatasetBuilder(self.settings)
        self.trainer = ForecasterTrainer(self.settings)

    def train_forecaster(
        self,
        symbol: str,
        interval: str = "1h",
        *,
        horizon: int = 1,
        kind: str | None = None,
        epochs: int | None = None,
        seq_len: int | None = None,
        experiment_name: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        target: str | None = None,
        normalize_volatility: bool | None = None,
        triple_barrier: TripleBarrierConfig | None = None,
        aux_direction_weight: float | None = None,
        ensemble_size: int | None = None,
        min_active_fraction: float | None = None,
        move_event_threshold: float | None = None,
        move_event_k: float | None = None,
        evaluate_holdout: bool = True,
    ) -> dict[str, Any]:
        """Train a forecaster and register it.

        ``evaluate_holdout=False`` trains, scores validation, saves the artifact
        and registers the model, but never looks at the final test split. That
        mode exists for *screening*: a sweep compares many configurations, and if
        each one reports a holdout score then the holdout has been used 600+
        times to make choices, which destroys the only genuinely untouched
        sample in the process. The selected model is then re-run with
        ``evaluate_holdout=True`` exactly once.
        """
        run_id = f"train-{uuid.uuid4().hex[:12]}"
        experiment = self.tracker.get_or_create_experiment(
            experiment_name or f"forecaster-{symbol}-{interval}",
            description=f"DL forecasting runs for {symbol} {interval}",
            tags={"symbol": symbol, "interval": interval, "task": "forecasting"},
        )

        resolved_seq_len = int(seq_len or self.settings.forecaster_seq_len)
        resolved_kind = kind or self.settings.forecaster_kind
        target_kind = target or self.settings.forecaster_target
        ensemble = max(1, int(
            self.settings.forecaster_ensemble_size if ensemble_size is None else ensemble_size
        ))
        direction_weight = float(
            self.settings.forecaster_aux_direction_weight
            if aux_direction_weight is None else aux_direction_weight
        )

        self.tracker.create_run(
            experiment.experiment_id, run_id=run_id, kind="train_forecaster",
            params={
                "symbol": symbol, "interval": interval, "horizon": horizon,
                "kind": resolved_kind, "epochs": epochs, "seq_len": seq_len,
                "target": target_kind, "ensemble_size": ensemble,
                "aux_direction_weight": direction_weight,
            },
        )

        try:
            bundle = self.builder.build(
                symbol, interval, horizon=horizon, start=start, end=end,
                target=target_kind, normalize_volatility=normalize_volatility,
                triple_barrier=triple_barrier,
                min_active_fraction=min_active_fraction,
                move_event_threshold=move_event_threshold,
                move_event_k=move_event_k,
            )
            logger.info("dataset_ready", extra=bundle.summary())

            # Purge and embargo are driven by the label horizon and the window
            # length, not by ratios: the tail of each block is exactly what would
            # otherwise carry prices the model has already seen.
            bounds = split_boundaries(
                len(bundle.frame), horizon=horizon, seq_len=resolved_seq_len
            )
            train_frame, val_frame, test_frame = split_frame(
                bundle.frame, horizon=horizon, seq_len=resolved_seq_len
            )
            if len(train_frame) < 100 or len(val_frame) < 20:
                raise ValueError(
                    f"Insufficient data: train={len(train_frame)} val={len(val_frame)} "
                    f"(purged/embargoed split of {len(bundle.frame)} rows)"
                )

            spec = bundle.target_spec
            train_kwargs: dict[str, Any] = {
                "target_column": spec.regression,
                "direction_column": spec.direction,
                "aux_direction_weight": direction_weight,
                "seq_len": resolved_seq_len,
                "horizon": horizon,
                "kind": resolved_kind,
                "epochs": epochs,
            }

            if ensemble > 1:
                members, training_info = self.trainer.train_ensemble(
                    train_frame, val_frame, bundle.feature_columns,
                    n_seeds=ensemble, **train_kwargs,
                )
                model, primary_info = members[0]
                training_info["history"] = primary_info["history"]
                training_info["member_metrics"] = [m[1]["metrics"] for m in members]
            else:
                model, training_info = self.trainer.train(
                    train_frame, val_frame, bundle.feature_columns, **train_kwargs
                )
                members = [(model, training_info)]

            # Validation is scored with the same function as the holdout so the
            # two are directly comparable, and so the selection metric carries an
            # interval the gate can act on.
            val_metrics = self._evaluate_forecaster(
                members, val_frame, bundle, training_info, prefix="val"
            )
            # During screening the holdout is left unopened. The metrics stay
            # absent rather than zero-filled so that no downstream code can
            # mistake "not measured" for "measured and found to be zero".
            test_metrics: dict[str, Any] = {}
            if evaluate_holdout:
                test_metrics = self._evaluate_forecaster(
                    members, test_frame, bundle, training_info, prefix="test"
                )
            else:
                logger.info(
                    "forecaster_holdout_deferred",
                    extra={"model_id_placeholder": True, "symbol": symbol},
                )

            model_id = new_model_id(symbol, resolved_kind, horizon)
            artifact = ForecasterArtifact(
                model_id, model, training_info["model_info"],
                {**training_info["metrics"], **val_metrics, **test_metrics},
                symbol.upper(), interval, members=members,
            )
            artifact_path = artifact.save(self.settings.resolved_artifact_dir / "models")

            register_params = {
                "horizon": horizon,
                "seq_len": training_info["model_info"]["seq_len"],
                "architecture": resolved_kind,
                "hidden_size": training_info["model_info"].get("hidden_size"),
                "epochs": int(training_info["metrics"].get("epochs_run") or epochs or 0),
                "target": spec.kind,
                "target_column": spec.regression,
                "normalize_volatility": bool(
                    normalize_volatility
                    if normalize_volatility is not None
                    else self.settings.forecaster_normalize_volatility
                ),
                "ensemble_size": len(members),
                "aux_direction_weight": direction_weight,
                "direction_column": spec.direction,
                "lr_schedule": training_info["model_info"].get("lr_schedule"),
                "target_detail": spec.detail,
                "purge_bars": bounds.purge,
                "embargo_bars": bounds.embargo,
            }

            self.registry.register(
                model_id=model_id, kind="forecaster", symbol=symbol.upper(), interval=interval,
                artifact_path=artifact_path, feature_schema=bundle.feature_columns,
                metrics={**training_info["metrics"], **val_metrics, **test_metrics},
                params=register_params,
                target=f"log_return_{horizon}", run_id=run_id,
                trained_from=bundle.start, trained_to=bundle.end,
            )

            result = {
                "run_id": run_id,
                "model_id": model_id,
                "symbol": symbol.upper(),
                "interval": interval,
                "dataset": bundle.summary(),
                "split": bounds.as_dict(),
                "val_metrics": {**training_info["metrics"], **val_metrics},
                "test_metrics": test_metrics,
                "ensemble": training_info.get("ensemble"),
                "history": training_info["history"],
                "artifact_path": str(artifact_path),
            }
            self.tracker.finish(
                run_id, "succeeded",
                {"val": {**training_info["metrics"], **val_metrics}, "test": test_metrics},
            )
            logger.info("forecaster_training_complete", extra={
                "model_id": model_id,
                "split": bounds.as_dict(),
                "ensemble_size": len(members),
                "val_dir_accuracy": round(val_metrics.get("val_directional_accuracy", 0.0), 4),
                "test_dir_accuracy": round(test_metrics.get("test_directional_accuracy", 0.0), 4),
            })
            return result

        except Exception as exc:  # noqa: BLE001
            self.tracker.finish(run_id, "failed", message=str(exc)[:500])
            logger.error("forecaster_training_failed", extra={"run_id": run_id, "error": str(exc)[:300]})
            raise

    def score_holdout(self, model_id: str, *, threshold: float = 0.0) -> dict[str, Any]:
        """Score a registered model's final test split, once, without retraining.

        This is the *only* sanctioned way to look at the holdout. A model
        trained with ``evaluate_holdout=False`` has an artifact and a validation
        score but no holdout numbers; this method produces them for the single
        model that selection chose, using the identical split geometry and the
        identical scoring function as the validation pass so the two are
        directly comparable.

        ``threshold`` must be the same decision rule the backtest and the sweep
        used for this model. Trading a different rule here would grade a
        strategy nobody intends to run.
        """
        record = self.registry.get_active(model_id)
        if record is None:
            raise ValueError(f"No active model found for id {model_id!r}")

        symbol = record.symbol
        interval = record.interval
        target_kind, normalize_volatility = resolve_target_kind(
            self.settings, record.params
        )
        members, _payload = ForecasterArtifact.load_ensemble(
            Path(record.artifact_path)
        )
        if not members:
            raise ValueError(f"model {model_id!r} has no ensemble members")
        model_info = members[0][1]
        horizon = int(record.params.get("horizon") or model_info.get("horizon") or 1)
        seq_len = int(record.params.get("seq_len") or model_info.get("seq_len") or 1)

        bundle = self.builder.build(
            symbol, interval, horizon=horizon,
            target=target_kind, normalize_volatility=normalize_volatility,
            triple_barrier=resolve_triple_barrier_config(self.settings, record.params),
            min_active_fraction=float(
                record.params.get("min_active_fraction") or 0.0
            ),
        )
        if not bundle.feature_columns:
            raise ValueError(
                f"no features could be built for {symbol} {interval}; cannot score holdout"
            )
        _train, _val, test_frame = split_frame(
            bundle.frame, horizon=horizon, seq_len=seq_len
        )
        if test_frame.empty:
            raise ValueError(
                f"holdout split is empty for {symbol} {interval} "
                f"(horizon={horizon}, seq_len={seq_len}, rows={len(bundle.frame)})"
            )

        metrics = self._evaluate_forecaster(
            members, test_frame, bundle, model_info, prefix="test"
        )
        if not metrics:
            raise ValueError(
                f"holdout produced no scorable rows for {symbol} {interval}; "
                "the test window is entirely unlabelled"
            )
        metrics.update(
            self._holdout_trading_metrics(
                members, test_frame, bundle, model_info, interval, threshold
            )
        )
        result = {
            "model_id": model_id,
            "symbol": symbol,
            "interval": interval,
            "horizon": horizon,
            "holdout_rows": int(len(test_frame)),
            "test_start": test_frame["ts"].iloc[0],
            "test_end": test_frame["ts"].iloc[-1],
            "annualisation": {
                "interval": interval,
                "bars_per_year": periods_per_year(interval),
                "return_horizon_bars": 1,
            },
            **metrics,
        }
        self.registry.record_metrics(model_id, metrics)
        logger.info(
            "forecaster_holdout_scored",
            extra={
                "model_id": model_id,
                "directional_accuracy": metrics.get("test_directional_accuracy"),
            },
        )
        return result

    def _holdout_trading_metrics(
        self,
        members: list[Any],
        test_frame: pd.DataFrame,
        bundle: DatasetBundle,
        model_info: dict[str, Any],
        interval: str,
        threshold: float = 0.0,
    ) -> dict[str, Any]:
        """Fee-aware trading statistics for the holdout window alone.

        Forecast metrics say the model ranks the future slightly better than
        chance. They say nothing about whether *trading* it survives costs, so a
        holdout verdict needs the drawdown, trade count and benchmark comparison
        that only a backtest produces.

        Three properties are deliberate:

        * The frozen artifact is traded as-is. Refitting per fold would be the
          walk-forward backtest, which fits on data including the window it then
          trades; here the model was fitted on train+validation and is applied,
          untouched, forward. That is the deployed configuration.
        * Only ``test_frame`` is traded. The engine is never handed the training
          or validation bars, so no risk figure quoted here can be traceable to
          data the model was fitted on.
        * Buy-and-hold is measured over the *same* window, so the excess figure
          compares two strategies on identical prices rather than a holdout
          against an all-sample average.
        """
        if test_frame.empty or "close" not in test_frame.columns:
            return {}
        predictions = self.trainer.predict_ensemble(
            members, test_frame, list(bundle.feature_columns)
        )
        if predictions.empty:
            return {}

        # `_evaluate_forecaster` restores labels to log-return space; the signal
        # comparison has to happen in the same units or a volatility-normalised
        # model would be compared against a threshold it was never trained on.
        predictions = self._to_log_return(predictions, bundle)
        series = pd.Series(
            (predictions["predicted_return"] > threshold).to_numpy(dtype=float),
            index=pd.DatetimeIndex(predictions["ts"]),
        )
        test_ts = pd.DatetimeIndex(test_frame["ts"])
        aligned = np.asarray(test_ts.isin(series.index), dtype=bool)
        signals = np.zeros(len(test_frame))
        if aligned.any():
            signals[aligned] = series.reindex(test_ts[aligned]).to_numpy()
        signals = np.nan_to_num(signals, nan=0.0)
        if not signals.any():
            # An all-flat signal means "never take a position". Reporting zero
            # drawdown and zero trades here would let a do-nothing model look
            # risk-free, so the window is reported as untraded instead.
            return {
                "test_trades": 0,
                "test_max_drawdown": None,
                "test_excess_vs_buy_and_hold": None,
                "test_strategy_sharpe": None,
                "test_strategy_return": None,
                "test_strategy_flat": True,
            }

        engine = BacktestEngine(
            self.settings, periods_per_year=periods_per_year(interval)
        )
        result = engine.run_signal_strategy(test_frame, signals)
        benchmark = engine.run_buy_and_hold(test_frame)
        # The engine nests its metric set under "metrics" and returns the trade
        # *log* separately as fill records, so a bare `result["trades"]` is a
        # list, not a count.
        strategy_metrics = dict(result.get("metrics") or {})
        fills = result.get("trades") or []
        benchmark_metrics = dict(benchmark.get("metrics") or {})
        return {
            "test_trades": len(fills),
            "test_max_drawdown": strategy_metrics.get("max_drawdown"),
            "test_strategy_sharpe": strategy_metrics.get("sharpe"),
            "test_strategy_return": strategy_metrics.get("total_return"),
            "test_benchmark_return": benchmark_metrics.get("total_return"),
            "test_benchmark_sharpe": benchmark_metrics.get("sharpe"),
            "test_excess_vs_buy_and_hold": _excess_return(
                strategy_metrics, benchmark_metrics
            ),
            "test_strategy_flat": False,
        }

    @staticmethod
    def _label_scale(bundle: DatasetBundle) -> pd.Series | None:
        """The causal trailing volatility used to scale this dataset's labels.

        Only a volatility-normalised target has one. It is the *trailing*
        volatility column, so using it at scoring time cannot introduce
        information the model did not have at prediction time.
        """
        if not bundle.target_spec.normalize_volatility:
            return None
        if "target_volatility" not in bundle.frame.columns:
            raise KeyError(
                "volatility-normalised target requested but `target_volatility` "
                "is missing from the dataset; scaled labels cannot be scored as "
                "returns without it"
            )
        return bundle.frame["target_volatility"].astype(float)

    def _to_log_return(
        self, predictions: pd.DataFrame, bundle: DatasetBundle
    ) -> pd.DataFrame:
        """Undo any target scaling so `predicted_return` can be exponentiated.

        A volatility-normalised model emits a move measured in sigmas, not a
        log return. Exponentiating that directly reports a price that is wrong
        by roughly the asset's volatility -- on hourly crypto, an order of
        magnitude. The conversion uses the same causal trailing sigma the
        target was built with, joined on the timestamp.
        """
        if not bundle.target_spec.normalize_volatility:
            return predictions
        sigma = self._label_scale(bundle)
        if sigma is None:
            return predictions
        lookup = pd.DataFrame({"ts": bundle.frame["ts"], "_sigma": sigma})
        joined = predictions.merge(lookup, on="ts", how="left")
        if len(joined) != len(predictions):
            raise KeyError(
                "cannot align trailing volatility to predictions; timestamp "
                f"types differ (predictions={predictions['ts'].dtype}, "
                f"frame={bundle.frame['ts'].dtype})"
            )
        values = joined["_sigma"].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            logger.warning(
                "prediction_scale_unavailable",
                extra={"symbol": bundle.symbol, "target": bundle.target_spec.kind},
            )
            return predictions
        out = predictions.copy()
        for column in ("predicted_return", "lower_68", "upper_68"):
            if column in out.columns:
                out[column] = np.asarray(out[column], dtype=float) * sigma
        return out

    def _evaluate_forecaster(
        self,
        members: list[tuple[Any, dict[str, Any]]],
        frame: pd.DataFrame,
        bundle: DatasetBundle,
        training_info: dict[str, Any],
        *,
        prefix: str = "test",
    ) -> dict[str, float]:
        """Score predictions against realised labels, with bootstrap intervals.

        Used for both the validation split (which drives selection) and the final
        holdout, so the two are measured identically. The `prefix` keeps the two
        sets of numbers from being silently merged in a report, and it is the
        reason selection cannot accidentally read a holdout field.
        """
        if not members:
            return {}
        spec = bundle.target_spec
        predictions = self.trainer.predict_ensemble(
            members, frame, bundle.feature_columns
        )
        if predictions.empty:
            return {}

        # Carry the causal trailing volatility along when the target is scaled,
        # so the label can be turned back into a realisable return below.
        carry = [spec.regression]
        if "target_volatility" in frame.columns:
            carry.append("target_volatility")
        merged = predictions.merge(
            frame[["ts", *carry]].rename(columns={spec.regression: "_actual"}),
            on="ts", how="inner",
        )
        if merged.empty:
            return {}

        actual = merged["_actual"].to_numpy(dtype=float)
        predicted = merged["predicted_return"].to_numpy(dtype=float)
        # Holdout/validation rows whose label never materialised cannot be
        # scored, but they must not silently shrink the sample either: record the
        # dropped count so the interval's n_obs can be sanity-checked.
        scorable = np.isfinite(actual) & np.isfinite(predicted)
        dropped = int((~scorable).sum())
        if not scorable.any():
            return {}
        if dropped:
            actual, predicted = actual[scorable], predicted[scorable]
            merged = merged.loc[scorable]

        mae = float(np.mean(np.abs(actual - predicted)))
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        direction = float(np.mean(np.sign(actual) == np.sign(predicted)))

        realized = np.asarray(actual, dtype=float)
        # This return series lives in *label* space, not price space. For a
        # volatility-normalised target `actual` is a move in trailing sigmas, and
        # compounding those as if they were log returns inflates every strategy
        # statistic by the average volatility (roughly 10x on hourly crypto).
        # Rescale by the label's own volatility before treating it as a return.
        if bundle.target_spec.normalize_volatility:
            scale = merged.get("target_volatility")
            if scale is None:
                raise KeyError(
                    "volatility-normalised target but predictions carry no "
                    "`target_volatility`; cannot convert the label to a return"
                )
            scale = np.asarray(scale, dtype=float)
            if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
                raise ValueError(
                    "trailing volatility is non-finite or non-positive; a scaled "
                    "label cannot be interpreted as a return"
                )
            realized = realized * scale
        # `log_return` and `triple_barrier` both produce log returns (tb_return is
        # log(exit/entry)), so they must be exponentiated before compounding.
        # A `move_event` run regresses on the raw forward log return and only
        # thresholds it for the auxiliary direction label, so its prediction is
        # in log space exactly like `log_return`.
        log_space = bundle.target_spec.kind in {"log_return", "triple_barrier", "move_event"}
        if bundle.target_spec.normalize_volatility:
            log_space = True
        if log_space:
            strategy = np.where(predicted > 0, np.expm1(realized), -np.expm1(realized))
        else:
            strategy = np.where(predicted > 0, realized, -realized)
        # One observation per label, and each label spans `horizon` bars, so the
        # annualisation divisor is the label horizon, not the raw bar count.
        strategy_metrics = compute_metrics(
            strategy,
            initial_capital=self.settings.backtest_initial_capital,
            periods_per_year=periods_per_year(
                bundle.interval, label_horizon=bundle.target_spec.horizon
            ),
        )

        covered = (actual >= merged["lower_68"].to_numpy(dtype=float)) & (
            actual <= merged["upper_68"].to_numpy(dtype=float)
        )
        coverage = float(np.mean(covered))

        report: dict[str, Any] = {
            f"{prefix}_mae": mae,
            f"{prefix}_rmse": rmse,
            f"{prefix}_directional_accuracy": direction,
            f"{prefix}_68pct_interval_coverage": coverage,
            f"{prefix}_correlation": float(np.corrcoef(actual, predicted)[0, 1])
            if len(actual) > 1 and np.std(predicted) > 1e-12 else 0.0,
            f"{prefix}_strategy_sharpe": strategy_metrics["sharpe"],
            f"{prefix}_strategy_return": strategy_metrics["total_return"],
            f"{prefix}_strategy_max_drawdown": strategy_metrics["max_drawdown"],
            f"{prefix}_samples": int(len(actual)),
        }
        if dropped:
            report[f"{prefix}_dropped_unlabelled"] = dropped

        # Without an interval a directional accuracy of 0.53 on 80 bars is
        # indistinguishable from a coin flip, and the gate would have no way to
        # know. The interval is what makes the point estimate actionable.
        intervals = forecast_metric_cis(
            actual, predicted,
            n_bootstrap=self.settings.bootstrap_iterations,
            confidence=self.settings.bootstrap_confidence,
            seed=self.settings.seed,
        )
        accuracy_ci = intervals.get("directional_accuracy") or {}
        correlation_ci = intervals.get("correlation") or {}
        mae_ci = intervals.get("mae") or {}

        if accuracy_ci:
            report[f"{prefix}_directional_accuracy_ci_low"] = float(accuracy_ci["lower"])
            report[f"{prefix}_directional_accuracy_ci_high"] = float(accuracy_ci["upper"])
            report[f"{prefix}_accuracy_block_size"] = float(accuracy_ci.get("block_size", 0.0))
            # Nested copy: the sweep gate reads a structured interval, and a flat
            # `*_ci_low` key is too easy to forget to check.
            report[f"{prefix}_directional_accuracy_ci"] = {
                "lower": float(accuracy_ci["lower"]),
                "upper": float(accuracy_ci["upper"]),
            }
        if correlation_ci:
            report[f"{prefix}_correlation_ci_low"] = float(correlation_ci["lower"])
            report[f"{prefix}_correlation_ci_high"] = float(correlation_ci["upper"])
            report[f"{prefix}_correlation_ci"] = {
                "lower": float(correlation_ci["lower"]),
                "upper": float(correlation_ci["upper"]),
            }
        if mae_ci:
            report[f"{prefix}_mae_ci_low"] = float(mae_ci["lower"])
            report[f"{prefix}_mae_ci_high"] = float(mae_ci["upper"])

        if "member_disagreement" in merged.columns:
            report[f"{prefix}_mean_member_disagreement"] = float(
                np.mean(merged["member_disagreement"].to_numpy(dtype=float))
            )
        return report

    def train_rl_agent(
        self,
        symbol: str,
        interval: str = "1h",
        *,
        timesteps: int | None = None,
        n_envs: int = 1,
        experiment_name: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, Any]:
        """Train PPO on the trading environment and register the resulting policy."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        run_id = f"train_rl-{uuid.uuid4().hex[:12]}"
        experiment = self.tracker.get_or_create_experiment(
            experiment_name or f"rl-{symbol}-{interval}",
            description=f"Reinforcement learning trading runs for {symbol} {interval}",
            tags={"symbol": symbol, "interval": interval, "task": "rl_trading"},
        )
        timesteps = timesteps or self.settings.rl_timesteps

        self.tracker.create_run(
            experiment.experiment_id, run_id=run_id, kind="train_rl",
            params={"symbol": symbol, "interval": interval, "timesteps": timesteps, "n_envs": n_envs},
        )

        try:
            bundle = self.builder.build(symbol, interval, horizon=1, start=start, end=end)
            if len(bundle.frame) < 500:
                logger.warning("rl_limited_data", extra={"rows": len(bundle.frame)})

            matrix = bundle.frame[bundle.feature_columns].to_numpy(dtype=np.float32)
            prices = bundle.frame["close"].to_numpy(dtype=np.float64)

            def make_env() -> TradingEnv:
                return TradingEnv(
                    matrix, prices, bundle.feature_columns, settings=self.settings,
                    reward_shaping=True,
                )

            env = DummyVecEnv([make_env for _ in range(max(1, n_envs))])
            model = PPO(
                "MlpPolicy", env,
                learning_rate=self.settings.rl_learning_rate,
                n_steps=min(2048, max(64, len(matrix) // 2)),
                batch_size=self.settings.rl_batch_size,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                policy_kwargs={"net_arch": [128, 64]},
                seed=self.settings.seed,
                verbose=0,
                device="cpu",
            )

            logger.info("rl_training_start", extra={"timesteps": timesteps, "rows": len(bundle.frame)})
            model.learn(total_timesteps=timesteps, progress_bar=False)
            env.close()

            eval_env = make_env()
            observations, _ = eval_env.reset(seed=self.settings.seed)
            episode_returns: list[float] = []

            for _ in range(10):
                done = False
                total_reward = 0.0
                while not done:
                    action, _states = model.predict(observations, deterministic=True)
                    observations, reward, terminated, truncated, info = eval_env.step(int(action[0]))
                    total_reward += float(reward[0])
                    done = bool(terminated[0] or truncated[0])
                episode_returns.append(total_reward)

            metrics = {
                "eval_episodes": len(episode_returns),
                "eval_mean_reward": float(np.mean(episode_returns)),
                "eval_std_reward": float(np.std(episode_returns)),
                "train_timesteps": timesteps,
                "env_rows": len(bundle.frame),
            }

            model_id = f"{symbol.replace('_', '-').lower()}-ppo-{uuid.uuid4().hex[:8]}"
            artifact_dir = self.settings.resolved_artifact_dir / "models"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / f"{model_id}.zip"
            model.save(str(artifact_path))

            self.registry.register(
                model_id=model_id, kind="rl_agent", symbol=symbol.upper(), interval=interval,
                artifact_path=artifact_path, feature_schema=bundle.feature_columns,
                metrics=metrics, params={"algorithm": "PPO", "timesteps": timesteps},
                target="portfolio_return", run_id=run_id, framework="stable_baselines3",
                trained_from=bundle.start, trained_to=bundle.end,
            )

            result = {
                "run_id": run_id, "model_id": model_id, "symbol": symbol.upper(),
                "interval": interval, "metrics": metrics,
                "dataset": bundle.summary(), "artifact_path": str(artifact_path),
            }
            self.tracker.finish(run_id, "succeeded", metrics)
            logger.info("rl_training_complete", extra={"model_id": model_id, **{
                k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)
            }})
            return result

        except Exception as exc:  # noqa: BLE001
            self.tracker.finish(run_id, "failed", message=str(exc)[:500])
            logger.error("rl_training_failed", extra={"error": str(exc)[:300]})
            raise

    def backtest_predictions(
        self,
        model_id: str,
        *,
        symbol: str | None = None,
        interval: str | None = None,
        threshold: float = 0.0,
        n_folds: int = 3,
        include_benchmarks: bool = True,
    ) -> dict[str, Any]:
        """Walk-forward backtest of a registered forecaster, refitting per fold."""

        run_id = f"backtest-{uuid.uuid4().hex[:12]}"
        record = self.registry.get_active(model_id)
        if record is None:
            raise ValueError(f"No active model found for id {model_id!r}")

        symbol = symbol or record.symbol
        interval = interval or record.interval
        model_params = record.params or {}
        architecture = str(model_params.get("architecture") or self.settings.forecaster_kind)
        refit_seq_len = int(model_params.get("seq_len") or self.settings.forecaster_seq_len)
        # The backtest must reproduce the horizon the model was trained on. A
        # model predicting 12-bar returns scored against 1-bar returns trades a
        # signal against the wrong horizon, which shows up as good directional
        # accuracy next to a destroyed Sharpe.
        model_horizon = int(model_params.get("horizon") or 1)
        # Folds should be refit for as long as the real model was trained,
        # otherwise `--epochs` silently does not reach the backtest.
        fold_epochs = resolve_fold_epochs(model_params, self.settings)
        # Reproduce the label definition, normalisation and ensemble width the
        # validated model used; anything less grades a different model.
        target_kind, normalize_volatility = resolve_target_kind(self.settings, model_params)
        barrier_config = resolve_triple_barrier_config(self.settings, model_params)
        ensemble = resolve_ensemble_size(self.settings, model_params)
        direction_weight = float(
            model_params.get("aux_direction_weight")
            if model_params.get("aux_direction_weight") is not None
            else self.settings.forecaster_aux_direction_weight
        )
        # Purge at least the label horizon, and embargo by the window length, so
        # no fold trains on bars whose label or feature window reaches into the
        # fold it is about to be graded on.
        purge_bars = max(
            int(self.settings.backtest_purge_bars or 0), model_horizon
        )
        embargo_bars = max(
            int(self.settings.backtest_embargo_bars or 0), refit_seq_len
        )

        experiment = self.tracker.get_or_create_experiment(
            f"backtest-{symbol}-{interval}",
            description=f"Walk-forward backtests for {symbol} {interval}",
            tags={"symbol": symbol, "interval": interval, "task": "backtest"},
        )
        self.tracker.create_run(
            experiment.experiment_id, run_id=run_id, kind="backtest",
            params={
                "model_id": model_id, "symbol": symbol, "interval": interval,
                "threshold": threshold, "n_folds": n_folds, "architecture": architecture,
                "target": target_kind, "ensemble_size": ensemble,
                "purge_bars": purge_bars, "embargo_bars": embargo_bars,
            },
        )

        try:
            bundle = self.builder.build(
                symbol, interval, horizon=model_horizon,
                target=target_kind, normalize_volatility=normalize_volatility,
                triple_barrier=barrier_config,
            )
            feature_columns = list(record.feature_schema or bundle.feature_columns)
            spec = bundle.target_spec

            def refit_and_predict(train_frame: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
                """Refit on the fold's train window, then forecast the unseen test window."""
                # The fold's own train window is split again to early-stop on, and
                # that inner split is purged too — the same leak applies one level
                # down, and early stopping on a leaking validation slice is how a
                # fold quietly reports a better model than the headline one.
                inner_horizon = min(12, max(1, int(len(train_frame) * 0.15)))
                inner_seq = min(refit_seq_len, max(1, inner_horizon))
                inner = split_frame(
                    train_frame, train_ratio=0.85, val_ratio=0.15,
                    horizon=inner_horizon, seq_len=inner_seq,
                )
                fit_frame, valid_frame, _discarded = inner
                if len(fit_frame) < 100 or len(valid_frame) < 10:
                    raise ValueError("fold too small to fit model")

                train_kwargs: dict[str, Any] = {
                    "target_column": spec.regression,
                    "direction_column": spec.direction,
                    "aux_direction_weight": direction_weight,
                    "epochs": fold_epochs,
                    "kind": architecture,
                    "seq_len": refit_seq_len,
                    "horizon": model_horizon,
                }
                if ensemble > 1:
                    members, info = self.trainer.train_ensemble(
                        fit_frame, valid_frame, feature_columns,
                        n_seeds=ensemble, **train_kwargs,
                    )
                else:
                    model, info = self.trainer.train(
                        fit_frame, valid_frame, feature_columns, **train_kwargs
                    )
                    members = [(model, info)]

                def predict(test_frame: pd.DataFrame) -> np.ndarray:
                    predictions = self.trainer.predict_ensemble(
                        members, test_frame, feature_columns
                    )
                    if predictions.empty:
                        return np.zeros(len(test_frame))
                    expected = float(params.get("threshold", threshold))
                    # `predict` returns a RangeIndex, so the signal series has to
                    # be indexed by timestamp before it can be aligned onto the
                    # test window. Reindexing it as-is yields all-NaN, which
                    # silently produced a strategy that never took a trade.
                    series = pd.Series(
                        (predictions["predicted_return"] > expected).to_numpy(dtype=float),
                        index=pd.DatetimeIndex(predictions["ts"]),
                    )
                    test_ts = pd.DatetimeIndex(test_frame["ts"])
                    aligned = np.asarray(test_ts.isin(series.index), dtype=bool)
                    signals = np.zeros(len(test_frame))
                    if aligned.any():
                        signals[aligned] = series.reindex(test_ts[aligned]).to_numpy()
                    return np.nan_to_num(signals, nan=0.0)

                return {"members": members, "info": info, "predict": predict}

            validator = WalkForwardValidator(self.settings)
            # The engine's return series is one entry per *bar* (the strategy
            # rebalances every bar), so annualisation uses the raw bar count for
            # this interval. Dividing by the label horizon here would be wrong:
            # that would rescale a per-bar equity curve.
            bar_periods = periods_per_year(interval)
            outcome = validator.validate(
                bundle.frame, refit_and_predict, n_folds=n_folds,
                params={"threshold": threshold},
                purge_bars=purge_bars, embargo_bars=embargo_bars,
                periods_per_year=bar_periods,
            )

            result: dict[str, Any] = {
                "run_id": run_id,
                "backtest_id": run_id,
                "model_id": model_id,
                "symbol": symbol,
                "interval": interval,
                "metrics": outcome.get("metrics", {}),
                "folds": outcome.get("folds", []),
                "strategy": "model_threshold_long",
                "threshold": threshold,
                "dataset": bundle.summary(),
                "split": {
                    "target": spec.as_dict(),
                    "ensemble_size": ensemble,
                    "purge_bars": purge_bars,
                    "embargo_bars": embargo_bars,
                },
                # Recorded so a report reader can tell whether Sharpe and CAGR
                # were annualised for the right bar size.
                "annualisation": {
                    "interval": interval,
                    "bars_per_year": bar_periods,
                    "return_horizon_bars": 1,
                },
            }
            for key in ("fold_distribution", "sharpe_ci"):
                if key in outcome:
                    result[key] = outcome[key]

            # A walk-forward run that produced nothing must say so. The old
            # behaviour returned an empty metrics dict that looked exactly like
            # "the model made no money", so a total failure in every fold was
            # indistinguishable from a flat strategy.
            failure = outcome.get("error")
            if failure:
                result["error"] = failure
                logger.warning("backtest_incomplete", extra={
                    "model_id": model_id, "error": failure,
                    "folds": len(outcome.get("folds", [])),
                })

            if include_benchmarks and "benchmark_metrics" in outcome:
                result["benchmarks"] = {
                    "buy_and_hold": outcome["benchmark_metrics"],
                    "excess_sharpe": result["metrics"].get("excess_vs_buy_and_hold"),
                }

            self._persist_backtest(result)
            self.tracker.finish(run_id, "succeeded", {"metrics": result.get("metrics", {})})
            logger.info("backtest_complete", extra={
                "model_id": model_id,
                "sharpe": round(result.get("metrics", {}).get("sharpe", 0.0), 4),
                "folds": len(result.get("folds", [])),
            })
            return result

        except Exception as exc:  # noqa: BLE001
            self.tracker.finish(run_id, "failed", message=str(exc)[:500])
            logger.error("backtest_failed", extra={"error": str(exc)[:300]})
            raise

    def _persist_backtest(self, result: dict[str, Any]) -> None:
        from cqip.db import BacktestResult

        with self.engine.session() as session:
            session.add(BacktestResult(
                backtest_id=result["backtest_id"],
                strategy=result.get("strategy", "unknown"),
                model_id=result.get("model_id"),
                symbol=result["symbol"],
                interval=result["interval"],
                params={"threshold": result.get("threshold")},
                metrics=to_jsonable(result.get("metrics", {})),
                folds=to_jsonable(result.get("folds", []))[:200],
                equity_curve=[],
                trades=[],
                run_id=result["run_id"],
            ))

    def predict_latest(self, model_id: str, *, symbol: str | None = None,
                       interval: str | None = None) -> dict[str, Any]:
        """Generate forward predictions for the most recent feature window."""

        record = self.registry.get_active(model_id)
        if record is None:
            raise ValueError(f"No active model found for id {model_id!r}")
        if record.kind != "forecaster":
            raise ValueError(f"Model {model_id!r} is a {record.kind}, not a forecaster")

        symbol = symbol or record.symbol
        interval = interval or record.interval

        target_kind, normalize_volatility = resolve_target_kind(self.settings, record.params)
        members, _payload = ForecasterArtifact.load_ensemble(Path(record.artifact_path))
        # Rebuild the dataset at the horizon the model was *trained* on. Using a
        # hardcoded 1 here silently mislabels a multi-bar model as a next-bar
        # one and desynchronises its triple-barrier geometry.
        model_horizon = int((members[0][1] if members else {}).get("horizon", 1) or 1)
        bundle = self.builder.build(
            symbol, interval, horizon=model_horizon,
            target=target_kind, normalize_volatility=normalize_volatility,
            triple_barrier=resolve_triple_barrier_config(self.settings, record.params),
        )
        feature_columns = list(record.feature_schema)
        predictions = self.trainer.predict_ensemble(
            members, bundle.frame, feature_columns
        )
        if predictions.empty:
            return {"model_id": model_id, "predictions": []}

        latest = predictions.iloc[-1]
        last_price = float(bundle.frame["close"].iloc[-1])
        model_info = members[0][1]
        horizon = int(model_info.get("horizon", 1))
        predictions = self._to_log_return(predictions, bundle)
        latest = predictions.iloc[-1]

        return {
            "model_id": model_id,
            "symbol": symbol,
            "interval": interval,
            "horizon": horizon,
            "as_of": bundle.frame["ts"].iloc[-1].to_pydatetime(),
            "last_price": last_price,
            "predicted_return": float(latest["predicted_return"]),
            "predicted_price": last_price * float(np.exp(latest["predicted_return"])),
            "lower_68": last_price * float(np.exp(latest["lower_68"])),
            "upper_68": last_price * float(np.exp(latest["upper_68"])),
            "confidence": float(latest["confidence"]),
            "ensemble_size": len(members),
            "feature_count": len(feature_columns),
            "metrics": record.metrics,
        }

    def forecast_series(self, model_id: str, *, symbol: str | None = None,
                        interval: str | None = None, limit: int = 100) -> pd.DataFrame:

        record = self.registry.get_active(model_id)
        if record is None:
            raise ValueError(f"No active model found for id {model_id!r}")

        symbol = symbol or record.symbol
        interval = interval or record.interval
        target_kind, normalize_volatility = resolve_target_kind(self.settings, record.params)
        members, _payload = ForecasterArtifact.load_ensemble(Path(record.artifact_path))
        model_horizon = int((members[0][1] if members else {}).get("horizon", 1) or 1)
        bundle = self.builder.build(
            symbol, interval, horizon=model_horizon,
            target=target_kind, normalize_volatility=normalize_volatility,
            triple_barrier=resolve_triple_barrier_config(self.settings, record.params),
        )
        predictions = self.trainer.predict_ensemble(
            members, bundle.frame, list(record.feature_schema)
        )
        if predictions.empty:
            return pd.DataFrame()
        last_price = float(bundle.frame["close"].iloc[-1])
        predictions = self._to_log_return(predictions, bundle)
        predictions["predicted_price"] = last_price * np.exp(predictions["predicted_return"])
        return predictions.tail(limit).reset_index(drop=True)


__all__ = [
    "DatasetBuilder",
    "DatasetBundle",
    "TrainingPipeline",
    "resolve_ensemble_size",
    "resolve_fold_epochs",
    "resolve_target_kind",
    "resolve_triple_barrier_config",
]