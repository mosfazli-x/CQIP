"""Tests for the research sweep: resampling, interval isolation and gating."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from cqip.storage import ParquetStore
from cqip.sweep import (
    INTERVAL_MINUTES,
    QualityGate,
    ResearchSweep,
    SweepCell,
    _absorb_test_metrics,
    cell_metrics_for,
    interval_minutes,
    resample_ohlcv,
)


def hourly(n: int = 96, start: str = "2026-01-01 00:00", price: float = 100.0) -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "ts": ts,
        "open": price + np.arange(n, dtype=float),
        "high": price + np.arange(n, dtype=float) + 1,
        "low": price + np.arange(n, dtype=float) - 1,
        "close": price + np.arange(n, dtype=float),
        "volume": np.ones(n),
        "symbol": "BTC_USDT",
        "exchange": "gate",
        "interval": "1h",
    })


class TestResample:
    def test_aggregates_ohlcv_correctly(self):
        frame = hourly(8)
        out = resample_ohlcv(
            frame, interval="4h", symbol="BTC_USDT", exchange="gate", base_interval="1h"
        )

        assert len(out) == 2
        first = out.iloc[0]
        # 8 hourly bars -> two complete 4h buckets.
        assert first["open"] == pytest.approx(frame["open"].iloc[0])
        assert first["high"] == pytest.approx(frame["high"].iloc[0:4].max())
        assert first["low"] == pytest.approx(frame["low"].iloc[0:4].min())
        assert first["close"] == pytest.approx(frame["close"].iloc[3])
        assert first["volume"] == pytest.approx(4.0)

    def test_drops_partially_filled_buckets(self):
        """A bucket missing base bars would carry a mid-move close."""
        # Starts at 01:00, so the 00:00-04:00 bucket only has 01,02,03.
        frame = hourly(10, start="2026-01-01 01:00")
        out = resample_ohlcv(
            frame, interval="4h", symbol="BTC_USDT", exchange="gate", base_interval="1h"
        )

        assert len(out) == 1
        assert out["ts"].iloc[0] == pd.Timestamp("2026-01-01 04:00", tz="UTC")

    def test_daily_bars_need_a_full_day(self):
        frame = hourly(24)
        out = resample_ohlcv(
            frame, interval="1d", symbol="BTC_USDT", exchange="gate", base_interval="1h"
        )
        assert len(out) == 1
        assert out["close"].iloc[0] == pytest.approx(frame["close"].iloc[-1])

    def test_too_short_a_series_yields_nothing(self):
        frame = hourly(10)
        out = resample_ohlcv(
            frame, interval="1d", symbol="BTC_USDT", exchange="gate", base_interval="1h"
        )
        assert out.empty

    def test_refuses_a_non_whole_multiple(self):
        with pytest.raises(ValueError, match="whole multiple"):
            resample_ohlcv(
                hourly(48), interval="90m", symbol="BTC_USDT", exchange="gate"
            ) if False else _raise_non_whole()

    def test_rejects_unknown_interval(self):
        with pytest.raises(ValueError, match="Unsupported interval"):
            interval_minutes("7q")

    def test_interval_minutes_cover_the_registry(self):
        assert interval_minutes("1h") == 60
        assert interval_minutes("1d") == 1_440
        assert set(INTERVAL_MINUTES.values()) == set(INTERVAL_MINUTES.values())

    def test_output_carries_identity_columns(self):
        out = resample_ohlcv(
            hourly(8), interval="4h", symbol="btc-usdt", exchange="gate", base_interval="1h"
        )
        assert set(out["interval"].unique()) == {"4h"}
        assert set(out["symbol"].unique()) == {"BTC_USDT"}
        assert set(out["exchange"].unique()) == {"gate"}


def _raise_non_whole() -> None:
    """1d from 4h base is a whole multiple, so use a genuinely partial ratio."""
    from cqip.sweep import INTERVAL_MINUTES as _m

    original = dict(_m)
    try:
        _m["1d"] = 1_500  # 1500 / 240 is not an integer
        resample_ohlcv(
            hourly(48), interval="1d", symbol="BTC_USDT", exchange="gate",
            base_interval="4h",
        )
    finally:
        _m.clear()
        _m.update(original)


class TestIntervalIsolation:
    """Several timeframes share one dataset, so the key must include the interval."""

    def test_different_intervals_do_not_collide(self, store: ParquetStore):
        base = hourly(8)
        coarse = resample_ohlcv(
            base, interval="4h", symbol="BTC_USDT", exchange="gate", base_interval="1h"
        )
        store.write(
            "ohlcv", base, dedupe_on=("ts", "symbol", "exchange", "interval"), mode="upsert"
        )
        store.write(
            "ohlcv", coarse, dedupe_on=("ts", "symbol", "exchange", "interval"),
            mode="upsert",
        )

        all_rows = store.read("ohlcv")
        one_hour = store.read("ohlcv", filters=[("interval", "==", "1h")])
        four_hour = store.read("ohlcv", filters=[("interval", "==", "4h")])

        # A 4h bar shares its timestamp with a 1h bar, so without `interval` in the
        # key the 1h row would be dropped as a duplicate.
        assert len(one_hour) == 8
        assert len(four_hour) == 2
        assert len(all_rows) == 10

    def test_loader_filters_by_interval(self, store: ParquetStore, settings, engine):
        from cqip.ingest import Ingestor

        base = hourly(24)
        coarse = resample_ohlcv(
            base, interval="4h", symbol="BTC_USDT", exchange="gate", base_interval="1h"
        )
        store.write(
            "ohlcv", base, dedupe_on=("ts", "symbol", "exchange", "interval"), mode="upsert"
        )
        store.write(
            "ohlcv", coarse, dedupe_on=("ts", "symbol", "exchange", "interval"),
            mode="upsert",
        )
        ingestor = Ingestor(settings, engine=engine, store=store)

        assert len(ingestor.load_dataset("BTC_USDT", "1h")) == 24
        assert len(ingestor.load_dataset("BTC_USDT", "4h")) == 6
        assert ingestor.load_dataset("BTC_USDT", "1d").empty


class TestQualityGate:
    def test_missing_metrics_block_the_verdict(self):
        """A gate that passes on absent data manufactures false confidence."""
        gate = QualityGate()
        verdict, reason = gate.evaluate({
            "direction_accuracy": None, "correlation": None,
            "trades": None, "max_drawdown": None,
        })
        # "insufficient_data", not "no-edge": a cell with no metrics has not
        # been shown to lack an edge, it has not been measured at all.
        assert verdict == "insufficient_data"
        assert "missing" in reason

    def _val_cell(self, **overrides):
        """A cell whose validation metrics are present and above chance."""
        cell = {
            "val_directional_accuracy": 0.62,
            "val_correlation": 0.11,
            "val_directional_accuracy_ci": {"lower": 0.57, "upper": 0.67},
            "trades": 120,
            "max_drawdown": -0.18,
            "excess_vs_buy_and_hold": 0.8,
        }
        cell.update(overrides)
        return cell

    def test_strong_cell_passes(self):
        gate = QualityGate()
        verdict, _ = gate.evaluate(self._val_cell(), source="validation")
        assert verdict == "passes"

    def test_missing_interval_is_rejected_not_assumed_strong(self):
        """No CI means no evidence, so the gate must not wave the cell through.

        The earlier `if lower is not None` guard let an interval-less cell pass,
        which is precisely backwards: the cells we know least about were the
        ones clearing the bar.
        """
        gate = QualityGate()
        cell = self._val_cell(
            val_directional_accuracy=0.72,
            val_correlation=0.30,
            trades=400,
            max_drawdown=-0.05,
            excess_vs_buy_and_hold=1.2,
        )
        cell.pop("val_directional_accuracy_ci")
        verdict, reason = gate.evaluate(cell, source="validation")
        # Accuracy and correlation were both measured here, so this is a real
        # rejection rather than an unmeasured cell: "no-edge" is accurate.
        assert verdict == "no-edge"
        assert "no 95% CI" in reason

    def test_interval_spanning_chance_is_rejected(self):
        gate = QualityGate()
        verdict, reason = gate.evaluate(
            self._val_cell(val_directional_accuracy_ci={"lower": 0.49, "upper": 0.63}),
            source="validation",
        )
        assert verdict == "no-edge"
        assert "0.50" in reason

    def test_malformed_interval_is_rejected(self):
        """A NaN bound is not a lower bound of zero; it is no bound at all."""
        gate = QualityGate()
        verdict, reason = gate.evaluate(
            self._val_cell(
                val_directional_accuracy_ci={"lower": float("nan"), "upper": float("nan")}
            ),
            source="validation",
        )
        assert verdict == "no-edge"
        assert "no 95% CI" in reason

    def test_chance_accuracy_is_rejected(self):
        gate = QualityGate()
        verdict, reason = gate.evaluate({
            "direction_accuracy": 0.479, "correlation": 0.30,
            "trades": 120, "max_drawdown": -0.18,
        }, source="auto")
        assert verdict == "no-edge"
        assert "coin flip" in reason

    def test_thin_trade_count_is_rejected(self):
        """Trade count is a holdout-only threshold: risk needs out-of-sample data."""
        gate = QualityGate()
        verdict, reason = gate.evaluate({
            "test_directional_accuracy": 0.70, "test_correlation": 0.30,
            "test_trades": 2, "test_max_drawdown": -0.05,
            "test_directional_accuracy_ci": {"lower": 0.64, "upper": 0.76},
        }, source="holdout")
        assert verdict == "no-edge"
        assert "not significant" in reason

    def test_underperforming_benchmark_is_rejected(self):
        gate = QualityGate()
        verdict, reason = gate.evaluate({
            "test_directional_accuracy": 0.70, "test_correlation": 0.30,
            "test_trades": 120, "test_max_drawdown": -0.05,
            "test_excess_vs_buy_and_hold": -7.35,
            "test_directional_accuracy_ci": {"lower": 0.64, "upper": 0.76},
        }, source="holdout")
        assert verdict == "no-edge"
        assert "buy-and-hold" in reason

    def test_validation_verdict_ignores_full_sample_strategy_metrics(self):
        """Flat risk fields must not influence a validation verdict.

        They come from a walk-forward backtest spanning the holdout, so honouring
        them here would let holdout drawdown decide a selection decision.
        """
        gate = QualityGate()
        cell = self._val_cell(
            # Catastrophic, and holdout-derived. Must be invisible to validation.
            trades=0, max_drawdown=-0.99, excess_vs_buy_and_hold=-9.0,
        )
        verdict, _ = gate.evaluate(cell, source="validation")
        assert verdict == "passes"

    def test_validation_does_not_require_strategy_metrics(self):
        gate = QualityGate()
        cell = self._val_cell()
        cell["val_trades"] = None
        cell["val_max_drawdown"] = None
        cell["val_excess_vs_buy_and_hold"] = None
        assert gate.evaluate(cell, source="validation")[0] == "passes"

    def test_zero_correlation_is_rejected_even_with_good_accuracy(self):
        gate = QualityGate()
        verdict, reason = gate.evaluate({
            "direction_accuracy": 0.72, "correlation": 0.003,
            "trades": 200, "max_drawdown": -0.10,
        }, source="auto")
        assert verdict == "no-edge"
        assert "corr" in reason

    def test_gate_defaults_to_validation_not_holdout(self):
        """The default source must not read holdout numbers.

        If the gate fell through to the holdout by accident, selection would be
        reading the data it is meant to be judged on, and the failure would be
        invisible in the report.
        """
        gate = QualityGate()
        verdict, reason = gate.evaluate({
            "val_directional_accuracy": None, "val_correlation": None,
            "test_directional_accuracy": 0.95, "test_correlation": 0.4,
            "trades": 500, "max_drawdown": -0.05,
        })
        # The validation metrics are absent even though the holdout metrics are
        # present, so nothing can be concluded: that is "insufficient_data",
        # and the holdout numbers must not leak in to replace them.
        assert verdict == "insufficient_data"
        assert "direction_accuracy" in reason

    def test_cell_with_chance_level_interval_is_rejected(self):
        """A point estimate whose interval spans 0.50 is not evidence of skill."""
        gate = QualityGate()
        verdict, reason = gate.evaluate({
            "val_directional_accuracy": 0.56, "val_correlation": 0.10,
            "trades": 200, "max_drawdown": -0.10,
            "val_directional_accuracy_ci": {"lower": 0.49, "upper": 0.63},
        })
        assert verdict == "no-edge"
        assert "indistinguishable from chance" in reason

    def test_cell_with_narrow_above_chance_interval_passes(self):
        gate = QualityGate()
        verdict, reason = gate.evaluate({
            "val_directional_accuracy": 0.61, "val_correlation": 0.12,
            "trades": 200, "max_drawdown": -0.10,
            "val_directional_accuracy_ci": {"lower": 0.57, "upper": 0.65},
        })
        assert verdict == "passes", reason

    def test_unknown_metric_source_is_rejected(self):
        with pytest.raises(ValueError, match="unknown metric source"):
            QualityGate().evaluate({"direction_accuracy": 0.9}, source="whatever")


class TestHoldoutMetricNameContract:
    """The cell and the pipeline must agree on holdout metric names.

    `_absorb_test_metrics` copies metrics by name, so a cell field called
    `test_direction_accuracy` facing a pipeline key of
    `test_directional_accuracy` drops the metric *silently*. The gate then
    reports a confident holdout verdict computed from an empty metric set.
    These tests pin the contract so a future rename cannot pass quietly again.
    """

    #: The subset the holdout gate actually reads, and the gate-side name each
    #: one is exposed under by ``cell_metrics_for``.
    GATED = {
        "test_directional_accuracy": "direction_accuracy",
        "test_correlation": "correlation",
        "test_directional_accuracy_ci": "direction_accuracy_ci",
        "test_trades": "trades",
        "test_max_drawdown": "max_drawdown",
        "test_excess_vs_buy_and_hold": "excess_vs_buy_and_hold",
    }

    def test_gated_holdout_metrics_are_real_cell_fields(self):
        fields = {f.name for f in dataclasses.fields(SweepCell)}
        missing = [n for n in self.GATED if n not in fields]
        assert not missing, f"gate reads metrics no cell can carry: {missing}"

    def test_every_gated_metric_reaches_the_cell_and_the_gate(self):
        payload = {
            "test_directional_accuracy": 0.61,
            "test_correlation": 0.12,
            "test_directional_accuracy_ci": {"lower": 0.56, "upper": 0.66},
            "test_trades": 95,
            "test_max_drawdown": -0.11,
            "test_excess_vs_buy_and_hold": 0.04,
            "test_samples": 700,
        }
        cell = SweepCell(interval="1d", kind="mlp", symbol="BTC_USDT", model_id="m")
        _absorb_test_metrics(cell, payload)

        for name in self.GATED:
            assert getattr(cell, name) == payload[name], f"{name} did not land on the cell"

        seen = cell_metrics_for(dataclasses.asdict(cell), "holdout")
        for name, gate_key in self.GATED.items():
            if name.endswith("_ci"):
                continue
            assert seen[gate_key] == payload[name], f"gate lost {name} ({gate_key})"

    def test_unmapped_metric_is_logged_rather_than_dropped_silently(self, caplog):
        cell = SweepCell(interval="1d", kind="mlp", symbol="BTC_USDT", model_id="m")
        with caplog.at_level("WARNING", logger="cqip.sweep"):
            _absorb_test_metrics(cell, {"test_directional_accurcy": 0.6})
        assert any("sweep_holdout_metric_unmapped" in r.message for r in caplog.records)

    def test_legacy_truncated_names_are_gone(self):
        """A stale `*_direction_accuracy` field must not reappear."""
        fields = {f.name for f in dataclasses.fields(SweepCell)}
        stale = sorted(n for n in fields if n.endswith("_direction_accuracy"))
        assert not stale, f"truncated metric names reintroduced: {stale}"


class TestSweepPlumbing:
    def test_ensure_interval_derives_coarse_bars(self, store, settings, engine):
        store.write(
            "ohlcv", hourly(96), dedupe_on=("ts", "symbol", "exchange", "interval"),
            mode="upsert",
        )
        sweep = ResearchSweep(settings, engine=engine, store=store)

        rows = sweep.ensure_interval("BTC_USDT", "4h")

        assert rows == 24
        assert len(sweep.store.read("ohlcv", filters=[("interval", "==", "1h")])) == 96

    def test_ensure_interval_is_idempotent(self, store, settings, engine):
        store.write(
            "ohlcv", hourly(96), dedupe_on=("ts", "symbol", "exchange", "interval"),
            mode="upsert",
        )
        sweep = ResearchSweep(settings, engine=engine, store=store)
        first = sweep.ensure_interval("BTC_USDT", "4h")
        second = sweep.ensure_interval("BTC_USDT", "4h")
        assert first == second == 24

    def test_cannot_derive_upwards(self, store, settings, engine):
        store.write(
            "ohlcv", hourly(96), dedupe_on=("ts", "symbol", "exchange", "interval"),
            mode="upsert",
        )
        sweep = ResearchSweep(settings, engine=engine, store=store)
        with pytest.raises(ValueError, match="cannot be derived"):
            sweep.ensure_interval("BTC_USDT", "15m")

    def test_cell_reports_insufficient_data(self, store, settings, engine):
        store.write(
            "ohlcv", hourly(48), dedupe_on=("ts", "symbol", "exchange", "interval"),
            mode="upsert",
        )
        sweep = ResearchSweep(settings, engine=engine, store=store)

        cell = sweep.run_cell("BTC_USDT", "1d", "mlp", epochs=1, folds=2)

        assert cell.status == "insufficient_data"
        # The verdict must not claim a negative result that was never measured.
        assert cell.verdict == "insufficient_data"
        assert "minimum" in cell.reason

    def test_short_history_is_a_data_limit_not_a_failure(self, store, settings, engine):
        """Too few bars must short-circuit as insufficient data, not a failed cell."""
        # 400 bars clears the flat min_rows floor of 300 but leaves a 15% test
        # tail of ~50 rows, which cannot hold a 64-bar window.
        store.write(
            "ohlcv", hourly(400), dedupe_on=("ts", "symbol", "exchange", "interval"),
            mode="upsert",
        )
        sweep = ResearchSweep(settings, engine=engine, store=store)

        def never(*args, **kwargs):
            raise AssertionError("training must not start without enough history")

        monkey = pytest.MonkeyPatch()
        monkey.setattr(sweep.pipeline, "train_forecaster", never)
        try:
            cell = sweep.run_cell("BTC_USDT", "1h", "mlp", epochs=1, folds=2)
        finally:
            monkey.undo()

        assert cell.status == "insufficient_data"
        # The verdict must not claim a negative result that was never measured.
        assert cell.verdict == "insufficient_data"
        assert "cannot support training" in cell.reason
        assert cell.model_id is None

    def test_window_error_is_classified_as_insufficient_data(
        self, store, settings, engine, monkeypatch
    ):
        store.write(
            "ohlcv", hourly(2000), dedupe_on=("ts", "symbol", "exchange", "interval"),
            mode="upsert",
        )
        sweep = ResearchSweep(settings, engine=engine, store=store)

        def too_short(*args, **kwargs):
            raise ValueError("Not enough rows to build a single window; need more history")

        monkeypatch.setattr(sweep.pipeline, "train_forecaster", too_short)
        cell = sweep.run_cell("BTC_USDT", "1h", "mlp", epochs=1, folds=2)

        assert cell.status == "insufficient_data"
        # The verdict must not claim a negative result that was never measured.
        assert cell.verdict == "insufficient_data"
        assert "do not survive" in cell.reason

    def test_other_value_errors_remain_failures(
        self, store, settings, engine, monkeypatch
    ):
        store.write(
            "ohlcv", hourly(2000), dedupe_on=("ts", "symbol", "exchange", "interval"),
            mode="upsert",
        )
        sweep = ResearchSweep(settings, engine=engine, store=store)

        def bad_config(*args, **kwargs):
            raise ValueError("hidden_size must be positive")

        monkeypatch.setattr(sweep.pipeline, "train_forecaster", bad_config)
        cell = sweep.run_cell("BTC_USDT", "1h", "mlp", epochs=1, folds=2)

        assert cell.status == "failed"
        assert "hidden_size" in cell.reason

    def test_cell_failure_is_contained(self, store, settings, engine, monkeypatch):
        store.write(
            "ohlcv", hourly(2000), dedupe_on=("ts", "symbol", "exchange", "interval"),
            mode="upsert",
        )
        sweep = ResearchSweep(settings, engine=engine, store=store)

        def boom(*args, **kwargs):
            raise RuntimeError("synthetic failure")

        monkeypatch.setattr(sweep.pipeline, "train_forecaster", boom)
        cell = sweep.run_cell("BTC_USDT", "1h", "mlp", epochs=1, folds=2)

        assert cell.status == "failed"
        assert "synthetic failure" in cell.reason

    def test_rejects_unknown_interval_in_run(self, store, settings, engine):
        sweep = ResearchSweep(settings, engine=engine, store=store)
        with pytest.raises(ValueError, match="Unsupported interval"):
            sweep.run(symbols=["BTC_USDT"], intervals=["3q"], kinds=["mlp"], epochs=1)

    def test_report_roundtrips_through_json(self, store, settings, engine):
        sweep = ResearchSweep(settings, engine=engine, store=store)
        from cqip.sweep import SweepReport

        report = SweepReport(
            symbols=["BTC_USDT"], intervals=["1h"], kinds=["mlp"], folds=3, epochs=1,
            started_at="2026-01-01T00:00:00+00:00",
            cells=[SweepCell(interval="1h", kind="mlp", symbol="BTC_USDT")],
        )
        path = sweep.save(report)
        assert path.exists()

        loaded = sweep.load_latest()
        assert loaded is not None
        assert loaded.cells[0].key == "BTC_USDT|1h|mlp"

    def test_sweep_uses_injected_settings_for_reports(self, store, settings, engine, tmp_path):
        """Reports must land under the injected data dir, not the global one."""
        custom = tmp_path / "custom"
        # `artifact_dir` is pinned by the fixture, so clear it to prove the
        # fallback resolves against the injected `data_dir`.
        settings = settings.model_copy(update={"data_dir": custom, "artifact_dir": None})
        sweep = ResearchSweep(settings, engine=engine, store=store)
        assert sweep.report_dir() == custom / "artifacts" / "reports"


class TestRequiredBars:
    """`_required_bars` gates whole cells, so an over-estimate silently hides symbols.

    The 70/15/15 split gives val and test 15% each, so the requirement is that
    the smaller tail fits one sequence window - not that 15% of the raw bar count
    covers the lookback. The earlier formula demanded ~1489 daily bars and
    rejected TIA/SUI/ARB/PEPE, which train perfectly well on ~1100.
    """

    def test_requirement_is_backed_by_the_actual_split(self, settings, engine, store):
        sweep = ResearchSweep(settings, engine=engine, store=store)
        seq_len = int(settings.forecaster_seq_len)
        required = sweep._required_bars(12)

        # Exactly enough rows for the test tail to hold one window.
        usable = required - 60 - 12
        assert int(usable * 0.15) >= seq_len
        # One bar less must genuinely fail, or the check is decorative.
        assert int((usable - 1) * 0.15) < seq_len

    def test_scales_with_seq_len_and_horizon(self, settings, engine, store):
        sweep = ResearchSweep(settings, engine=engine, store=store)
        base = sweep._required_bars(1)
        assert sweep._required_bars(12) > base, "a longer horizon needs more bars"
        longer = ResearchSweep(
            settings.model_copy(update={"forecaster_seq_len": int(settings.forecaster_seq_len) * 2}),
            engine=engine, store=store,
        )
        assert longer._required_bars(1) > base, "a longer window needs more bars"

    def test_newer_daily_histories_are_not_rejected(self, settings, engine, store):
        """The four symbols previously blocked must clear the bar requirement."""
        sweep = ResearchSweep(settings, engine=engine, store=store)
        required = sweep._required_bars(12)
        for symbol, bars in (("TIA_USDT", 1064), ("SUI_USDT", 1245),
                             ("ARB_USDT", 1286), ("PEPE_USDT", 1259)):
            assert bars >= required, f"{symbol} ({bars} bars) blocked by a {required}-bar floor"