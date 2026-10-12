"""Tests for symbol resolution, ingest depth planning and staged sweeps."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from cqip.universe import (
    DEFAULT_INTERVAL_DEPTH_DAYS,
    INSUFFICIENT_FOR_FORECASTING,
    RENAMED_SYMBOLS,
    TOP_CRYPTO_SYMBOLS,
    SymbolStatus,
    Universe,
    missing_targets,
    plan_backfill,
)


class FakeProvider:
    """Stands in for a live exchange listing without touching the network."""

    name = "fake"

    def __init__(self, pairs: set[str]) -> None:
        self.pairs = pairs
        self.settings = None

    def list_symbols(self):  # noqa: ANN201 - matches the provider contract
        return sorted(self.pairs)


def universe_from(tickers, listed):
    from cqip.universe import _resolve_one

    statuses = [_resolve_one(t, listed, "USDT") for t in tickers]
    return Universe(statuses=statuses, provider="fake", quote="USDT")


LISTED = {
    "BTC_USDT", "ETH_USDT", "SOL_USDT", "POL_USDT",
    "USDC_USDT", "USDT_BTC",
}


class TestResolution:
    def test_plain_symbol_resolves(self):
        universe = universe_from(["BTC", "sol"], LISTED)
        assert universe.tradable == ["BTC_USDT", "SOL_USDT"]

    def test_lowercase_and_underscore_forms_agree(self):
        a = universe_from(["BTC"], LISTED)
        b = universe_from(["btc_usdt"], LISTED)
        assert a.tradable == b.tradable == ["BTC_USDT"]

    def test_unlisted_symbol_is_reported_not_dropped(self):
        """A missing pair must be visible; silently omitting it hides the gap."""
        universe = universe_from(["BTC", "TONCOIN"], LISTED)
        assert universe.tradable == ["BTC_USDT"]
        assert [s.requested for s in universe.skipped] == ["TONCOIN"]
        assert "not a tradable" in universe.skipped[0].reason

    def test_usdt_against_itself_is_unavailable(self):
        universe = universe_from(["USDT"], LISTED)
        assert universe.tradable == []
        assert "USDT_USDT is not a tradable" in universe.skipped[0].reason

    def test_stablecoin_resolves_but_is_not_a_target(self):
        """A pegged price cannot be predicted, so it is flagged not trained."""
        status = universe_from(["USDC"], LISTED).statuses[0]
        assert status.available is True
        assert status.is_stablecoin is True
        assert status.usable_as_target is False
        assert "pegged" in status.reason

    def test_delisted_ticker_resolves_to_its_replacement(self):
        universe = universe_from(["MATIC"], LISTED)
        status = universe.statuses[0]
        assert status.symbol == "POL_USDT"
        assert status.renamed_from == "MATIC"
        assert "rename" in status.reason

    def test_gap_annotated_when_another_venue_lists_it(self, monkeypatch):
        """A delisting from one venue is recoverable, so the report must say so."""
        import cqip.universe as mod

        listed = {"gate": {"BTC_USDT"}, "bitget": {"BTC_USDT", "XMR_USDT"}}

        def fake_pairs(provider, quote):
            key = provider.name
            return listed.get(key, set())

        monkeypatch.setattr(mod, "_listed_pairs", fake_pairs)
        universe = mod.resolve_universe(
            ["BTC_USDT", "XMR_USDT"], provider_name="gate",
            fallback_providers=["bitget"],
        )
        xmr = next(s for s in universe.statuses if s.requested == "XMR_USDT")
        assert not xmr.available
        # A confirmed listing elsewhere is more actionable than the generic
        # note, so it must win.
        assert "bitget" in xmr.reason and "XMR_USDT" in xmr.reason
        assert "not a tradable" not in xmr.reason

    def test_unavailable_elsewhere_keeps_the_exchange_note(self, monkeypatch):
        import cqip.universe as mod

        def fake_pairs(provider, quote):
            return set()

        monkeypatch.setattr(mod, "_listed_pairs", fake_pairs)
        universe = mod.resolve_universe(
            ["TON"], provider_name="gate", fallback_providers=["bitget"],
        )
        assert "not listed on Gate" in universe.skipped[0].reason

    def test_mkr_resolves_to_its_replacement_sky(self):
        universe = universe_from(["MKR"], {"SKY_USDT"})
        assert universe.tradable == ["SKY_USDT"]
        assert universe.statuses[0].renamed_from == "MKR"

    def test_rename_is_not_applied_when_replacement_is_absent(self):
        universe = universe_from(["MATIC"], {"BTC_USDT"})
        assert universe.tradable == []
        assert "known rename: try POL_USDT" in universe.skipped[0].reason

    def test_targets_exclude_stablecoins_but_keep_the_rest(self):
        universe = universe_from(["BTC", "USDC", "ETH"], LISTED)
        assert universe.targets == ["BTC_USDT", "ETH_USDT"]
        assert [s.symbol for s in universe.stablecoins] == ["USDC_USDT"]

    def test_non_usdt_pairs_are_not_counted(self):
        universe = universe_from(["BTC"], {"USDT_BTC"})
        assert universe.tradable == []

    def test_missing_targets_lists_every_unusable_ticker(self):
        universe = universe_from(["BTC", "USDC", "USDT", "GHOST"], LISTED)
        assert sorted(missing_targets(universe, ["BTC", "USDC", "USDT", "GHOST"])) == [
            "GHOST", "USDC", "USDT",
        ]


class TestCatalog:
    def test_catalog_is_uppercase_tickers_without_quote(self):
        assert all(t == t.upper() and "_" not in t for t in TOP_CRYPTO_SYMBOLS)

    def test_catalog_has_no_duplicates(self):
        assert len(TOP_CRYPTO_SYMBOLS) == len(set(TOP_CRYPTO_SYMBOLS))

    def test_renames_point_at_themselves(self):
        """A rename mapping onto the old ticker would loop forever."""
        for old, new in RENAMED_SYMBOLS.items():
            assert new != old
            assert new not in RENAMED_SYMBOLS


class TestBackfillPlan:
    def test_daily_depth_is_unbounded_in_practice(self):
        """Daily bars reach listing, so the cap must not truncate them."""
        assert DEFAULT_INTERVAL_DEPTH_DAYS["1d"] > 10 * 365

    def test_hourly_depth_reflects_exchange_retention(self):
        assert DEFAULT_INTERVAL_DEPTH_DAYS["1h"] <= 400

    def test_sub_daily_intervals_are_marked_untrainable(self):
        for interval in ("5m", "15m", "30m"):
            assert interval in INSUFFICIENT_FOR_FORECASTING

    def test_plan_covers_every_symbol_and_interval(self):
        universe = universe_from(["BTC", "ETH"], LISTED)
        plan = plan_backfill(universe, intervals=("1d", "1h"))
        assert len(plan) == 4
        assert {p["symbol"] for p in plan} == {"BTC_USDT", "ETH_USDT"}
        assert all(p["trainable"] for p in plan)

    def test_plan_excludes_unavailable_symbols(self):
        universe = universe_from(["BTC", "GHOST"], LISTED)
        plan = plan_backfill(universe, intervals=("1d",))
        assert [p["symbol"] for p in plan] == ["BTC_USDT"]

    def test_plan_ignores_unknown_intervals(self):
        universe = universe_from(["BTC"], LISTED)
        plan = plan_backfill(universe, intervals=("1d", "7s"))
        assert len(plan) == 1


class TestIngestTiming:
    def test_result_duration_covers_the_fetch_not_just_the_write(self, store, settings, engine):
        """A duration that excludes the network round-trip always reads as instant."""
        from cqip.ingest import Ingestor
        from cqip.models import Candle

        ingestor = Ingestor(settings, engine=engine, store=store)
        start = datetime(2026, 1, 1, tzinfo=UTC)
        candles = [
            Candle(
                ts=start + timedelta(hours=i), open=100.0 + i, high=101.0 + i,
                low=99.0 + i, close=100.5 + i, volume=1.0,
            )
            for i in range(10)
        ]

        result = _ingest_with_stubbed_fetch(ingestor, candles)

        assert result.rows == 10
        assert result.finished_at is not None
        assert result.started_at <= result.finished_at

    def test_all_failed_combinations_mark_the_run_failed(self, store, settings, engine):
        """A total outage must not be recorded as a successful ingest."""
        import asyncio

        from cqip import ingest as ingest_mod
        from cqip.ingest import Ingestor

        ingestor = Ingestor(settings, engine=engine, store=store)
        asyncio.run(_ingest_returning_nothing(ingest_mod, ingestor))


def _ingest_with_stubbed_fetch(ingestor, candles):
    import asyncio

    from cqip import ingest as ingest_mod

    async def fake_fetch(symbols, interval, providers, **kwargs):
        return {(providers[0], s): list(candles) for s in symbols}

    original = ingest_mod.fetch_cross_exchange
    ingest_mod.fetch_cross_exchange = fake_fetch
    try:
        return asyncio.run(ingestor.ingest_candles(["BTC_USDT"], "1h", ["gate"]))[0]
    finally:
        ingest_mod.fetch_cross_exchange = original


async def _ingest_returning_nothing(ingest_mod, ingestor):
    from sqlalchemy import select

    from cqip.db import Run

    async def empty_fetch(symbols, interval, providers, **kwargs):
        return {(providers[0], s): [] for s in symbols}

    original = ingest_mod.fetch_cross_exchange
    ingest_mod.fetch_cross_exchange = empty_fetch
    try:
        await ingestor.ingest_candles(["BTC_USDT"], "1h", ["gate"])
    finally:
        ingest_mod.fetch_cross_exchange = original

    with ingestor.engine.session() as session:
        run = session.execute(
            select(Run).order_by(Run.started_at.desc()).limit(1)
        ).scalar_one()
    assert run.status == "failed", "a run where nothing was fetched is not a success"


class TestBackfillUniverse:
    def test_inserted_is_derived_from_write_not_dataset_recount(
        self, store, settings, engine
    ):
        """`ingest_candles` must trust `store.write`'s return value.

        Counting the dataset before and after the write reads every partition
        twice, which is ruinous on a multi-million-row store and is the
        dominant cost of an interval-wide backfill.
        """
        from cqip.ingest import Ingestor
        from cqip.models import Candle

        ingestor = Ingestor(settings, engine=engine, store=store)
        start = datetime(2026, 1, 1, tzinfo=UTC)
        candles = [
            Candle(
                ts=start + timedelta(hours=i), open=100.0 + i, high=101.0 + i,
                low=99.0 + i, close=100.5 + i, volume=1.0,
            )
            for i in range(10)
        ]

        class ExplodingCount:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, *args, **kwargs):
                self.calls += 1
                raise AssertionError("ingest must not recount the dataset to size its insert")

        counter = ExplodingCount()
        original_count = store.count
        store.count = counter
        try:
            result = _ingest_with_stubbed_fetch(ingestor, candles)
        finally:
            store.count = original_count

        assert counter.calls == 0
        assert result.inserted == len(candles)
        assert len(store.read(result.dataset)) == len(candles)

        # A re-run that only deduplicates must report zero genuinely new rows
        # instead of re-claiming the whole batch.
        store.count = counter
        try:
            again = _ingest_with_stubbed_fetch(ingestor, candles)
        finally:
            store.count = original_count

        assert counter.calls == 0
        assert again.inserted == 0
        assert len(store.read(again.dataset)) == len(candles)

    def test_every_symbol_interval_pair_is_ingested(self, store, settings, engine):
        import asyncio

        from cqip import ingest as ingest_mod
        from cqip.ingest import Ingestor

        ingestor = Ingestor(settings, engine=engine, store=store)
        seen: list[tuple[str, str]] = []

        async def fake_fetch(symbols, interval, providers, **kwargs):
            seen.append((tuple(symbols), interval))
            start = datetime(2020, 1, 1, tzinfo=UTC)
            from cqip.models import Candle

            return {
                (providers[0], s): [
                    Candle(
                        ts=start + timedelta(hours=i), open=1.0, high=1.0,
                        low=1.0, close=1.0, volume=1.0,
                    )
                    for i in range(5)
                ]
                for s in symbols
            }

        original = ingest_mod.fetch_cross_exchange
        ingest_mod.fetch_cross_exchange = fake_fetch
        try:
            summary = asyncio.run(ingestor.backfill_universe(
                ["BTC_USDT", "ETH_USDT"], intervals=("1d", "1h"),
            ))
        finally:
            ingest_mod.fetch_cross_exchange = original

        # One call per timeframe carrying every symbol, not one call per
        # (symbol, interval) pair: `store.write` rewrites the whole dataset, so
        # per-pair calls would make a universe backfill quadratic in size.
        assert seen == [
            (("BTC_USDT", "ETH_USDT"), "1d"),
            (("BTC_USDT", "ETH_USDT"), "1h"),
        ]
        assert summary["total_rows"] == 20
        assert summary["errors"] == []


class TestHoldoutRiskFieldProvenance:
    """``test_*`` risk fields must describe the test window, not the whole frame.

    ``backtest_predictions`` is a walk-forward run over the entire dataset that
    refits per fold, so its drawdown and trade count include train and
    validation. Populating ``test_*`` from it would let a holdout verdict quote
    risk figures the holdout never produced.
    """

    def test_holdout_risk_comes_from_score_holdout_not_walk_forward(
        self, settings, monkeypatch
    ):
        from cqip.sweep import ResearchSweep, cell_metrics_for

        class FakePipeline:
            def __init__(self):
                self.calls: list[str] = []

            def train_forecaster(self, *a, **k):
                self.calls.append("train")
                return {
                    "model_id": "m1", "symbol": "BTC_USDT", "interval": "1d",
                    "val_metrics": {
                        "val_directional_accuracy": 0.62,
                        "val_correlation": 0.20,
                        "val_directional_accuracy_ci": {"lower": 0.57, "upper": 0.67},
                    },
                    "test_metrics": {}, "ensemble": {},
                }

            def score_holdout(self, model_id, *, threshold=0.0):
                self.calls.append(f"score_holdout:{threshold}")
                return {
                    "test_trades": 40,
                    "test_max_drawdown": -0.12,
                    "test_excess_vs_buy_and_hold": 0.03,
                }

            def backtest_predictions(self, model_id, *, threshold=0.0, n_folds=3):
                # Whole-frame walk-forward: deliberately absurd values, so any
                # test asserting on them proves the wrong source was read.
                self.calls.append("backtest_predictions")
                return {"metrics": {
                    "sharpe": 99.0, "total_return": 99.0, "consistency": 1.0,
                    "max_drawdown": -0.99, "excess_vs_buy_and_hold": 99.0,
                    "trades": 9999,
                }, "benchmarks": {}}

        pipeline = FakePipeline()
        sweep = ResearchSweep(settings=settings)
        sweep.pipeline = pipeline
        # The point of this test is the wiring, not the dataset; the store check
        # is stubbed so `run_cell` reaches the branch under examination.
        monkeypatch.setattr(sweep, "ensure_interval", lambda s, i: 100_000)
        cell = sweep.run_cell(
            "BTC_USDT", "1d", "mlp", epochs=1, folds=2,
            evaluate_holdout=True,
        )

        assert cell.test_trades == 40, "test_trades came from the full-frame backtest"
        assert cell.test_max_drawdown == -0.12
        assert cell.test_excess_vs_buy_and_hold == 0.03
        # The full-sample mirrors may still carry walk-forward numbers, but the
        # gate must not be able to read them.
        assert cell.max_drawdown == -0.99
        gate_metrics = cell_metrics_for(dataclasses.asdict(cell), "holdout")
        assert gate_metrics["max_drawdown"] == -0.12
        assert "score_holdout" in " ".join(pipeline.calls)


class TestStagedSweep:
    def _cell(self, symbol, interval, kind, accuracy, correlation, status="ok",
              *, test_accuracy=None):
        """Build a screening cell whose *validation* metrics drive selection.

        The holdout fields are populated too whenever given, so the tests below
        can prove that promotion ignores them.
        """
        from cqip.sweep import SweepCell

        cell = SweepCell(interval=interval, kind=kind, symbol=symbol)
        cell.status = status
        cell.val_directional_accuracy = accuracy
        cell.val_correlation = correlation
        # Legacy flat mirrors, so the tests keep describing a complete cell.
        cell.direction_accuracy = accuracy
        cell.correlation = correlation
        if test_accuracy is not None:
            cell.test_directional_accuracy = test_accuracy
            cell.test_correlation = correlation
        return cell

    def test_promotes_only_cells_at_or_above_chance(self, store, settings, engine, monkeypatch):
        """Below-chance accuracy is a result, not a near miss to be retrained."""
        from cqip.sweep import ResearchSweep

        sweep = ResearchSweep(settings, engine=engine, store=store)
        screen_cells = [
            self._cell("BTC_USDT", "1d", "mlp", 0.61, 0.12),
            self._cell("ETH_USDT", "1d", "mlp", 0.50, 0.01),
            self._cell("SOL_USDT", "1d", "mlp", 0.44, 0.30),  # high corr, bad accuracy
        ]

        monkeypatch.setattr(sweep, "run", lambda **kwargs: _report(screen_cells))
        promoted: list[tuple[str, str, str]] = []
        monkeypatch.setattr(
            sweep, "run_cell",
            lambda s, i, k, **kw: promoted.append((s, i, k))
            or self._cell(s, i, k, 0.6, 0.1),
        )

        _, full = sweep.run_staged(
            symbols=["BTC_USDT", "ETH_USDT", "SOL_USDT"],
            intervals=["1d"], kinds=["mlp"], survivors=10,
        )

        assert sorted(promoted) == [("BTC_USDT", "1d", "mlp"), ("ETH_USDT", "1d", "mlp")]
        assert len(full.cells) == 2

    def test_respects_the_survivor_budget(self, store, settings, engine, monkeypatch):
        from cqip.sweep import ResearchSweep

        sweep = ResearchSweep(settings, engine=engine, store=store)
        screen_cells = [
            self._cell(f"C{i}_USDT", "1d", "mlp", 0.50 + i / 100, 0.01) for i in range(8)
        ]
        monkeypatch.setattr(sweep, "run", lambda **kwargs: _report(screen_cells))
        promoted: list[tuple[str, str, str]] = []
        monkeypatch.setattr(
            sweep, "run_cell",
            lambda s, i, k, **kw: promoted.append((s, i, k))
            or self._cell(s, i, k, 0.6, 0.1),
        )

        sweep.run_staged(
            symbols=[f"C{i}_USDT" for i in range(8)],
            intervals=["1d"], kinds=["mlp"], survivors=3,
        )

        # The three highest-accuracy cells survive, not an arbitrary three.
        assert [p[0] for p in promoted] == ["C7_USDT", "C6_USDT", "C5_USDT"]

    def test_skips_cells_without_metrics(self, store, settings, engine, monkeypatch):
        from cqip.sweep import ResearchSweep

        sweep = ResearchSweep(settings, engine=engine, store=store)
        screen_cells = [
            self._cell("BTC_USDT", "1d", "mlp", None, None, status="failed"),
            self._cell("ETH_USDT", "1d", "mlp", 0.62, 0.09),
        ]
        monkeypatch.setattr(sweep, "run", lambda **kwargs: _report(screen_cells))
        promoted: list[tuple[str, str, str]] = []
        monkeypatch.setattr(
            sweep, "run_cell",
            lambda s, i, k, **kw: promoted.append((s, i, k))
            or self._cell(s, i, k, 0.6, 0.1),
        )

        sweep.run_staged(
            symbols=["BTC_USDT", "ETH_USDT"], intervals=["1d"], kinds=["mlp"],
        )

        assert [p[0] for p in promoted] == ["ETH_USDT"]

    def test_promotion_ignores_holdout_metrics(self, store, settings, engine, monkeypatch):
        """Selection must not read the holdout, even when it looks far better.

        The holdout cell has the best holdout accuracy in the matrix and the worst
        validation accuracy. If promotion consulted it, the wrong cell would win
        and the final holdout would already be spent.
        """
        from cqip.sweep import ResearchSweep

        sweep = ResearchSweep(settings, engine=engine, store=store)
        screen_cells = [
            self._cell("GOOD_VAL_USDT", "1d", "mlp", 0.58, 0.10, test_accuracy=0.51),
            self._cell("BAD_VAL_USDT", "1d", "mlp", 0.51, 0.02, test_accuracy=0.99),
        ]
        monkeypatch.setattr(sweep, "run", lambda **kwargs: _report(screen_cells))
        promoted: list[str] = []
        monkeypatch.setattr(
            sweep, "run_cell",
            lambda s, i, k, **kw: promoted.append(s) or self._cell(s, i, k, 0.6, 0.1),
        )

        sweep.run_staged(
            symbols=["GOOD_VAL_USDT", "BAD_VAL_USDT"],
            intervals=["1d"], kinds=["mlp"], survivors=1,
        )

        assert promoted == ["GOOD_VAL_USDT"]

    def test_selection_score_reads_validation_only(self):
        """`selection_score` must not fall back to holdout or flat fields."""
        from cqip.sweep import SweepCell

        cell = SweepCell(interval="1d", kind="mlp", symbol="X_USDT")
        cell.direction_accuracy = 0.99
        cell.correlation = 0.9
        cell.test_directional_accuracy = 0.98
        cell.test_correlation = 0.8
        assert cell.selection_score == float("-inf")

        cell.val_directional_accuracy = 0.55
        cell.val_correlation = 0.08
        assert cell.selection_score == pytest.approx(0.63)

    def test_holdout_evaluation_scores_the_holdout(self, store, settings, engine):
        """`evaluate_holdout` must *measure* the holdout, not re-gate stored numbers.

        The old implementation just re-judged metrics that screening had already
        computed, which meant every cell had touched the test split long before
        selection finished. The holdout is now reached only through the pipeline.
        """
        from cqip.sweep import ResearchSweep

        sweep = ResearchSweep(settings, engine=engine, store=store)
        cell = self._cell("BTC_USDT", "1d", "mlp", 0.55, 0.10)
        cell.model_id = "btc-usdt-mlp-h1-test"
        cell.trades = 200
        cell.max_drawdown = -0.1
        assert cell.holdout_scored is False
        # Pre-existing holdout numbers that must be *replaced* by a real score.
        cell.test_directional_accuracy = 0.44

        seen: list[str] = []

        class FakePipeline:
            def score_holdout(self, model_id, *, threshold=0.0):
                seen.append(model_id)
                return {
                    "test_directional_accuracy": 0.61,
                    "test_correlation": 0.09,
                    "test_directional_accuracy_ci": {"lower": 0.56, "upper": 0.66},
                    "test_trades": 95,
                    "test_max_drawdown": -0.12,
                    "test_excess_vs_buy_and_hold": 0.4,
                }

        sweep.pipeline = FakePipeline()
        [scored] = sweep.evaluate_holdout([cell])

        assert seen == ["btc-usdt-mlp-h1-test"], "holdout was not actually scored"
        assert scored.holdout_scored is True
        assert scored.test_directional_accuracy == 0.61, "stale holdout value survived"
        assert scored.verdict == "passes"

    def test_holdout_evaluation_refuses_to_pretend_validation_is_holdout(
        self, store, settings, engine
    ):
        """`source="validation"` here would set `holdout_scored` on untested data."""
        from cqip.sweep import ResearchSweep

        sweep = ResearchSweep(settings, engine=engine, store=store)
        cell = self._cell("BTC_USDT", "1d", "mlp", 0.55, 0.10)
        cell.model_id = "some-model"
        with pytest.raises(ValueError, match="source='holdout'"):
            sweep.evaluate_holdout([cell], source="validation")
        assert cell.holdout_scored is False

    def test_holdout_failure_is_reported_per_cell_not_raised(self, store, settings, engine):
        from cqip.sweep import ResearchSweep

        sweep = ResearchSweep(settings, engine=engine, store=store)
        cell = self._cell("BTC_USDT", "1d", "mlp", 0.55, 0.10)
        cell.model_id = "broken-model"

        class BrokenPipeline:
            def score_holdout(self, _model_id, *, threshold=0.0):
                raise RuntimeError("artifact missing")

        sweep.pipeline = BrokenPipeline()
        [scored] = sweep.evaluate_holdout([cell])
        assert scored.holdout_scored is False
        assert scored.verdict == "failed"
        assert "artifact missing" in scored.reason

    def test_screening_never_overwrites_the_latest_full_report(
        self, store, settings, engine
    ):
        """A screen must not clobber the full result it is a filter for."""
        from cqip.sweep import ResearchSweep, SweepReport

        sweep = ResearchSweep(settings, engine=engine, store=store)
        full = SweepReport(
            symbols=["BTC_USDT"], intervals=["1d"], kinds=["mlp"],
            folds=1, epochs=1, started_at="2026-01-01T00:00:00+00:00",
        )
        sweep.save(full)
        assert "BTC_USDT" in (sweep.report_dir() / "sweep-latest.json").read_text()

        sweep.save(
            SweepReport(
                symbols=["SCREEN"], intervals=["1d"], kinds=["mlp"],
                folds=1, epochs=1, started_at="2026-01-01T00:00:00+00:00",
            ),
            label="screen",
        )
        assert "BTC_USDT" in (sweep.report_dir() / "sweep-latest.json").read_text()


def _report(cells) -> object:
    from cqip.sweep import SweepReport

    return SweepReport(
        symbols=[], intervals=["1d"], kinds=["mlp"], folds=1, epochs=1,
        started_at="2026-01-01T00:00:00+00:00", cells=list(cells),
    )


class TestStatusSerialisation:
    def test_status_round_trips_through_a_dict(self):
        status = SymbolStatus(
            requested="BTC", symbol="BTC_USDT", available=True,
            earliest_ts="2013-03-31", rows=4930,
        )
        payload = status.as_dict()
        assert payload["usable_as_target"] is True
        assert payload["earliest_ts"] == "2013-03-31"

    def test_unavailable_status_serialises_cleanly(self):
        status = SymbolStatus(
            requested="USDT", symbol=None, available=False, reason="not listed",
        )
        payload = status.as_dict()
        assert payload["symbol"] is None
        assert payload["usable_as_target"] is False
