"""Integration tests for the FastAPI surface.

Every test runs against an isolated temp data directory: the settings
dependency is overridden and the engine cache is reset, so nothing here can
touch the real data/ directory.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cqip import api as api_module
from cqip.api import create_app
from cqip.config import get_settings
from cqip.db import reset_engine


@pytest.fixture
def client(settings) -> Iterator[TestClient]:
    reset_engine()
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    reset_engine()


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    payload: dict = {}
    while time.monotonic() < deadline:
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["status"] in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {payload}")


class TestSystemEndpoints:
    def test_health_reports_database_and_providers(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] in {"ok", "degraded"}
        assert body["database"]["ok"] is True
        assert body["version"] == "0.1.0"
        assert isinstance(body["providers"], list)

    def test_health_uses_app_settings_not_globals(self, client, settings):
        """Regression: the lifespan used to re-read global settings."""
        assert client.get("/health").json()["data_dir"] == str(settings.data_dir)

    def test_openapi_schema_is_generated(self, client):
        response = client.get("/openapi.json")

        assert response.status_code == 200
        paths = response.json()["paths"]
        for expected in ("/health", "/data/candles", "/models", "/jobs", "/training/runs"):
            assert expected in paths, f"{expected} missing from OpenAPI paths"

    def test_dataset_stats_lists_every_dataset(self, client, seeded_store):
        response = client.get("/health/datasets")

        assert response.status_code == 200
        stats = {row["dataset"]: row for row in response.json()}
        assert set(stats) == {"ohlcv", "news", "onchain", "fear_greed"}
        assert stats["ohlcv"]["exists"] is True
        assert stats["ohlcv"]["rows"] == 900
        assert stats["ohlcv"]["size_bytes"] > 0
        assert stats["ohlcv"]["start"] < stats["ohlcv"]["end"]


class TestDataEndpoints:
    def test_candles_are_returned_for_known_symbol(self, client, seeded_store):
        response = client.get("/data/candles", params={"symbol": "BTC_USDT", "limit": 50})

        assert response.status_code == 200
        body = response.json()
        assert body["symbol"] == "BTC_USDT"
        assert body["count"] == 50
        assert len(body["candles"]) == 50
        assert set(body["candles"][0]) >= {"ts", "open", "high", "low", "close", "volume"}

    def test_candles_are_sorted_by_timestamp(self, client, seeded_store):
        body = client.get("/data/candles", params={"symbol": "BTC_USDT", "limit": 200}).json()

        times = pd.to_datetime([candle["ts"] for candle in body["candles"]], utc=True)
        assert times.is_monotonic_increasing

    def test_candles_are_404_for_unknown_symbol(self, client, seeded_store):
        response = client.get("/data/candles", params={"symbol": "DOGE_USDT"})

        assert response.status_code == 404
        assert "DOGE_USDT" in response.json()["detail"]

    def test_symbol_normalisation_accepts_dashed_input(self, client, seeded_store):
        response = client.get("/data/candles", params={"symbol": "btc-usdt", "limit": 5})

        assert response.status_code == 200
        assert response.json()["symbol"] == "BTC_USDT"

    def test_limit_is_validated(self, client, seeded_store):
        assert client.get("/data/candles", params={"limit": 0}).status_code == 422
        assert client.get("/data/candles", params={"limit": 99_999}).status_code == 422

    def test_symbols_endpoint_lists_ingested_pairs(self, client, seeded_store):
        response = client.get("/data/symbols")

        assert response.status_code == 200
        assert "BTC_USDT" in response.json()["symbols"]

    def test_news_endpoint_returns_scored_articles(self, client, seeded_store):
        response = client.get("/data/news", params={"limit": 10})

        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 10
        article = body["items"][0]
        assert "sentiment_score" in article
        assert -1.0 <= article["sentiment_score"] <= 1.0
        assert body["by_source"], "source rollup missing"

    def test_news_endpoint_filters_by_source(self, client, seeded_store):
        body = client.get("/data/news", params={"source": "coindesk", "limit": 100}).json()

        assert body["items"]
        assert {a["source"] for a in body["items"]} == {"coindesk"}

    def test_news_on_empty_dataset_returns_zero_count(self, client):
        body = client.get("/data/news").json()

        assert body["count"] == 0
        assert body["items"] == []

    def test_coverage_reports_recorded_ranges(self, client, settings, synthetic_candles):
        """Coverage is written by the ingestor, not by a raw Parquet write."""
        from cqip.db import get_engine, upsert_coverage

        engine = get_engine(settings)
        window = synthetic_candles["ts"]
        upsert_coverage(
            engine, "ohlcv", "BTC_USDT", "1h", "gate",
            window.min().to_pydatetime(), window.max().to_pydatetime(), len(synthetic_candles),
        )

        response = client.get("/data/coverage", params={"dataset": "ohlcv"})

        assert response.status_code == 200
        entries = response.json()["entries"]
        assert len(entries) == 1
        assert entries[0]["provider"] == "gate"
        assert entries[0]["rows"] == len(synthetic_candles)
        # The timezone-aware round trip must survive the UTCDateTime decorator.
        assert entries[0]["start"].endswith("Z") or "+00:00" in entries[0]["start"]


class TestModelEndpoints:
    def test_model_list_is_empty_before_training(self, client):
        response = client.get("/models")

        assert response.status_code == 200
        assert response.json() == {"count": 0, "models": []}

    def test_missing_model_is_404(self, client):
        assert client.get("/models/does-not-exist").status_code == 404
        assert client.get("/models/does-not-exist/predict").status_code == 404

    def test_train_submits_a_job(self, client, seeded_store):
        response = client.post("/models/train", json={
            "symbol": "BTC_USDT", "interval": "1h", "kind": "mlp", "epochs": 2, "seq_len": 16,
        })

        assert response.status_code == 202
        body = response.json()
        assert body["kind"] == "train_forecaster"
        assert body["status"] in {"queued", "running"}
        assert body["poll_url"] == f"/jobs/{body['job_id']}"

    def test_train_validates_payload(self, client, seeded_store):
        assert client.post("/models/train", json={}).status_code == 422
        assert client.post("/models/train", json={"symbol": "BTC_USDT", "horizon": 0}).status_code == 422

    def test_backtest_requires_a_model_id(self, client):
        assert client.post("/models/backtest", json={}).status_code == 422


class TestJobEndpoints:
    def test_submit_requires_kind(self, client):
        assert client.post("/jobs", json={"params": {}}).status_code == 400

    def test_submit_rejects_non_object_params(self, client):
        response = client.post("/jobs", json={"kind": "ingest", "params": []})

        assert response.status_code == 400
        assert "params" in response.json()["detail"]

    def test_submit_rejects_unknown_job_kind(self, client):
        response = client.post("/jobs", json={"kind": "not-a-real-kind", "params": {}})

        assert response.status_code == 400

    def test_job_lifecycle_reaches_a_terminal_state(self, client, seeded_store):
        submitted = client.post("/models/train", json={
            "symbol": "BTC_USDT", "interval": "1h", "kind": "mlp", "epochs": 2, "seq_len": 16,
        })
        job_id = submitted.json()["job_id"]

        finished = _wait_for_job(client, job_id)

        assert finished["status"] == "succeeded", finished
        assert finished["finished_at"] is not None

        listed = client.get("/jobs").json()
        assert listed["count"] >= 1
        assert {job["job_id"] for job in listed["jobs"]} >= {job_id}

    def test_jobs_can_be_filtered(self, client, seeded_store):
        client.post("/models/train", json={
            "symbol": "BTC_USDT", "interval": "1h", "kind": "mlp", "epochs": 2, "seq_len": 16,
        })

        assert client.get("/jobs", params={"kind": "train_forecaster"}).json()["count"] >= 1
        assert client.get("/jobs", params={"kind": "nothing-matches"}).json()["count"] == 0

    def test_unknown_job_is_404(self, client):
        assert client.get("/jobs/missing-job").status_code == 404

    def test_cancel_is_409_for_unknown_job(self, client):
        assert client.delete("/jobs/missing-job").status_code == 409


class TestTrainingEndpoints:
    def test_runs_are_recorded(self, client, seeded_store):
        submitted = client.post("/models/train", json={
            "symbol": "BTC_USDT", "interval": "1h", "kind": "mlp", "epochs": 2, "seq_len": 16,
        })
        _wait_for_job(client, submitted.json()["job_id"])

        response = client.get("/training/runs")

        assert response.status_code == 200
        assert response.json()["count"] >= 1


class TestIngestEndpoint:
    def test_ingest_is_accepted_and_queued(self, client, monkeypatch):
        calls: list[tuple] = []

        async def fake_ingest_all(self, *args, **kwargs):
            calls.append((args, kwargs))
            return {"candles": [], "news": {}, "onchain": []}

        monkeypatch.setattr(api_module.Ingestor, "ingest_all", fake_ingest_all, raising=False)

        response = client.post("/data/ingest", json={"symbols": ["BTC_USDT"], "interval": "1h"})

        assert response.status_code == 202
        body = response.json()
        assert body["news"]["status"] == "scheduled"
        assert calls, "background task never awaited ingest_all"
        assert calls[0][0][0] == ["BTC_USDT"]

    def test_ingest_failure_marks_the_run_failed(self, client, monkeypatch, settings):
        """A background ingestion error must be recorded, not silently dropped."""
        from cqip.db import get_engine
        from cqip.db import record_run as _record_run  # noqa: F401

        async def boom(self, *args, **kwargs):
            raise RuntimeError("provider exploded")

        monkeypatch.setattr(api_module.Ingestor, "ingest_all", boom, raising=False)

        body = client.post("/data/ingest", json={"symbols": ["BTC_USDT"]}).json()
        run_id = body["run_id"]

        import sqlalchemy as sa

        from cqip.db import Run  # noqa: F401

        engine = get_engine(settings)
        with engine.session() as session:
            row = session.execute(
                sa.text("SELECT status FROM runs WHERE run_id = :rid"), {"rid": run_id}
            ).scalar_one()

        assert row == "failed"

    def test_ingest_rejects_empty_symbols(self, client):
        assert client.post("/data/ingest", json={"symbols": []}).status_code == 422


class TestTrainedModelEndpoints:
    """Exercise the model routes against a genuinely trained artifact."""

    @pytest.fixture
    def trained(self, client, seeded_store) -> str:
        from cqip.db import get_engine
        from cqip.pipeline import TrainingPipeline

        pipeline = TrainingPipeline(settings=client.app.state.settings, engine=get_engine(client.app.state.settings))
        result = pipeline.train_forecaster("BTC_USDT", "1h", kind="mlp", epochs=2, seq_len=16)
        return str(result["model_id"])

    def test_model_appears_in_the_registry(self, client, trained):
        listing = client.get("/models").json()

        assert listing["count"] == 1
        assert listing["models"][0]["model_id"] == trained

    def test_model_detail_is_returned(self, client, trained):
        response = client.get(f"/models/{trained}")

        assert response.status_code == 200
        body = response.json()
        assert body["model_id"] == trained
        assert body["kind"] == "forecaster"
        assert body["symbol"] == "BTC_USDT"
        assert body["feature_count"] > 0, "feature schema not persisted with the model"
        assert body["metrics"], "training metrics not exposed on the model record"

    def test_predict_returns_a_distribution(self, client, trained):
        response = client.get(f"/models/{trained}/predict")

        assert response.status_code == 200
        body = response.json()
        assert body["symbol"] == "BTC_USDT"
        assert body["horizon"] >= 1
        assert body["lower_68"] <= body["upper_68"]
        assert 0.0 < body["last_price"] > 0
        assert body["as_of"]

    def test_forecast_returns_a_time_series(self, client, trained):
        response = client.get(f"/models/{trained}/forecast", params={"limit": 24})

        assert response.status_code == 200
        body = response.json()
        assert body["model_id"] == trained
        assert body["count"] > 0
        point = body["points"][0]
        assert {"ts", "predicted_return", "predicted_price", "lower_68", "upper_68"} <= set(point)
        assert point["lower_68"] <= point["upper_68"]
        assert point["predicted_price"] > 0

    def test_forecast_validates_limit(self, client, trained):
        assert client.get(f"/models/{trained}/forecast", params={"limit": 0}).status_code == 422
        assert client.get(f"/models/{trained}/forecast", params={"limit": 99_999}).status_code == 422

    def test_promote_advances_the_stage(self, client, trained):
        before = client.get(f"/models/{trained}").json()["stage"]

        response = client.post(f"/models/{trained}/promote", json={"stage": "production"})

        assert response.status_code == 200
        assert response.json()["stage"] == "production"
        assert response.json()["stage"] != before or before == "production"

    def test_delete_removes_the_model(self, client, trained):
        assert client.delete(f"/models/{trained}").status_code == 204
        assert client.get(f"/models/{trained}").status_code == 404
        assert client.delete(f"/models/{trained}").status_code == 404

    def test_registry_filters_by_kind_and_symbol(self, client, trained):
        assert client.get("/models", params={"kind": "forecaster"}).json()["count"] == 1
        assert client.get("/models", params={"kind": "rl_agent"}).json()["count"] == 0
        assert client.get("/models", params={"symbol": "ETH_USDT"}).json()["count"] == 0


class TestStreamingEndpoints:
    def test_price_stream_sends_a_tick(self, client, seeded_store):
        # Exactly one frame: the loop sleeps between updates, so reading more
        # than the first payload would block for the poll interval.
        with client.websocket_connect("/stream/prices?symbol=BTC_USDT") as socket:
            message = socket.receive_json()

        assert message["type"] == "prices"
        payload = message["data"]
        assert "ts" in payload
        assert "BTC_USDT" in payload["symbols"]
        tick = payload["symbols"]["BTC_USDT"]
        assert tick["close"] > 0

    def test_job_stream_sends_initial_status(self, client, seeded_store):
        submitted = client.post("/models/train", json={
            "symbol": "BTC_USDT", "interval": "1h", "kind": "mlp", "epochs": 2, "seq_len": 16,
        })
        job_id = submitted.json()["job_id"]

        with client.websocket_connect(f"/stream/jobs/{job_id}") as socket:
            message = socket.receive_json()

        assert message["type"] == "status"
        assert message["data"]["job_id"] == job_id
        assert message["data"]["status"] in {"queued", "running", "succeeded", "failed"}

    def test_job_stream_closes_immediately_for_finished_job(self, client, seeded_store):
        submitted = client.post("/models/train", json={
            "symbol": "BTC_USDT", "interval": "1h", "kind": "mlp", "epochs": 2, "seq_len": 16,
        })
        job_id = submitted.json()["job_id"]
        finished = _wait_for_job(client, job_id)
        assert finished["status"] == "succeeded", finished

        # A terminal job returns after the snapshot instead of holding the
        # connection open, so this read is guaranteed not to block.
        with client.websocket_connect(f"/stream/jobs/{job_id}") as socket:
            message = socket.receive_json()

        assert message["data"]["status"] == "succeeded"

    def test_sse_stream_emits_a_terminal_event(self, client, seeded_store):
        submitted = client.post("/models/train", json={
            "symbol": "BTC_USDT", "interval": "1h", "kind": "mlp", "epochs": 2, "seq_len": 16,
        })
        job_id = submitted.json()["job_id"]
        _wait_for_job(client, job_id)

        response = client.get(f"/jobs/{job_id}/stream")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "data:" in response.text
        assert "succeeded" in response.text

    def test_sse_stream_is_404_for_unknown_job(self, client):
        assert client.get("/jobs/nope/stream").status_code == 404


class TestResearchApi:
    """Regression tests for the dashboard research routes.

    These guard the failure mode that matters most here: a monitoring view that
    quietly reports a stale or invented number is worse than no view at all.
    """

    @staticmethod
    def _write_report(settings, name, cells, started="2026-01-01T00:00:00+00:00"):
        reports = settings.resolved_artifact_dir / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        payload = {
            "started_at": started, "folds": 3, "epochs": 20,
            "params": {}, "rows": [], "cells": cells,
        }
        (reports / name).write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def _cell(symbol="BTC_USDT", interval="1d", kind="mlp", **over):
        cell = {
            "symbol": symbol, "interval": interval, "kind": kind, "status": "ok",
            "model_id": "m-1", "rows": 1200,
            "val_directional_accuracy": 0.513,
            "val_directional_accuracy_ci": {"lower": 0.487, "upper": 0.539},
            "val_directional_correlation": 0.004,
            "test_directional_accuracy": None, "test_directional_accuracy_ci": None,
            "test_excess_return": None, "test_trades": None,
            "test_max_drawdown": None, "holdout_scored": False, "verdict": "no-edge",
            "reasons": ["val accuracy 0.513 overlaps 0.5"],
        }
        cell.update(over)
        return cell

    def test_overview_serves_empty_state_without_reports(self, client):
        response = client.get("/research/overview")

        assert response.status_code == 200
        body = response.json()
        assert body["cells_total"] == 0
        assert body["cells_passed"] == 0
        assert body["health"]["database_readable"] is True

    def test_overview_normalises_cells_and_keeps_missing_as_null(self, client, settings):
        self._write_report(settings, "sweep-a.json", [self._cell()])

        body = client.get("/research/overview").json()

        assert body["cells_total"] == 1
        row = body["symbol_table"][0]
        assert row["symbol"] == "BTC_USDT"
        assert row["verdict"] == "no-edge"
        # An unscored holdout must surface as null, never as 0.0.
        assert row["test_accuracy"] is None
        assert row["test_max_drawdown"] is None
        assert row["holdout_scored"] is False
        assert row["val_accuracy_ci"] == {"lower": 0.487, "upper": 0.539}

    def test_newest_report_supersedes_older_for_same_cell(self, client, settings):
        self._write_report(settings, "sweep-old.json", [self._cell(val_directional_accuracy=0.49)],
                           started="2026-01-01T00:00:00+00:00")
        self._write_report(settings, "sweep-new.json", [self._cell(val_directional_accuracy=0.57,
                                                                   verdict="passes")],
                           started="2026-02-01T00:00:00+00:00")

        body = client.get("/research/overview").json()

        # Regression: dedup used to keep the *oldest* sighting.
        assert body["cells_total"] == 1
        row = body["symbol_table"][0]
        assert row["val_accuracy"] == 0.57
        assert row["verdict"] == "passes"
        assert body["cells_passed"] == 1

    def test_holdout_reading_survives_a_later_rescreen(self, client, settings):
        """A re-validation cannot un-spend a holdout, so it must not be erased."""
        self._write_report(settings, "sweep-scored.json", [self._cell(
            holdout_scored=True, verdict="passes",
            test_directional_accuracy=0.5,
            test_directional_accuracy_ci={"lower": 0.42, "upper": 0.58},
            test_trades=7, test_max_drawdown=0.11,
        )], started="2026-01-01T00:00:00+00:00")
        self._write_report(settings, "sweep-rescreen.json", [self._cell(
            holdout_scored=False, verdict="no-edge",
        )], started="2026-02-01T00:00:00+00:00")

        body = client.get("/research/overview").json()

        assert body["cells_total"] == 1
        assert body["holdout_scored_cells"] == 1
        row = body["symbol_table"][0]
        # Primary record stays the newest run...
        assert row["val_accuracy"] == 0.513
        # ...but the spent holdout is carried forward and attributed.
        assert row["holdout_scored"] is True
        assert row["test_accuracy"] == 0.5
        assert row["test_trades"] == 7
        assert row["holdout_source"] == "sweep-scored.json"

    def test_screening_reports_excluded_from_cell_table(self, client, settings):
        self._write_report(settings, "sweep-a-screen-20260101T000000Z-abc.json", [self._cell()])

        body = client.get("/research/overview").json()

        assert body["cells_total"] == 0
        # Still listed as a report, just not as evidence.
        assert any(r["is_screen"] for r in body["reports"])

    def test_reports_listing_and_detail(self, client, settings):
        self._write_report(settings, "sweep-detail.json", [self._cell(), self._cell(symbol="ETH_USDT")])

        listing = client.get("/research/reports")
        assert listing.status_code == 200
        assert listing.json()["count"] == 1

        detail = client.get("/research/reports/sweep-detail.json")
        assert detail.status_code == 200
        body = detail.json()
        assert body["summary"]["cell_count"] == 2
        assert body["summary"]["verdicts"]["no-edge"] == 2

    @pytest.mark.parametrize("name", ["../secrets.json", "sub/evil.json", "nope.json"])
    def test_report_detail_rejects_traversal_and_non_sweep_files(self, client, name):
        assert client.get(f"/research/reports/{name}").status_code in {400, 404}

    def test_report_detail_reports_invalid_json_as_422(self, client, settings):
        reports = settings.resolved_artifact_dir / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "sweep-broken.json").write_text("{not json", encoding="utf-8")

        assert client.get("/research/reports/sweep-broken.json").status_code == 422

    def test_events_endpoint_returns_empty_list(self, client):
        body = client.get("/research/events").json()

        assert body["count"] == 0
        assert body["studies"] == []


class TestEventStudyRoutes:
    def test_event_summary_exposes_best_precision_interval(self, client, settings):
        """Regression: the summary passed scalars to a dict-coercion helper, so
        best_precision_ci was always null and the UI could not show the width of
        the headline selectivity number."""
        from cqip.events import event_study_record, save_event_report

        record = event_study_record(
            symbol="BTC_USDT", interval="1d", model_kind="mlp",
            horizon=1, seq_len=48, epochs=8,
            event={
                "verdict": "no-edge", "event_rate": 0.55, "n_events": 220, "n": 400,
                "threshold": 0.02, "k": 0.2,
                "by_k": [{
                    "k": 0.2, "precision": 0.70, "precision_ci_low": 0.61,
                    "precision_ci_high": 0.78, "lift": 1.27, "n_taken": 80,
                    "significant": True,
                }],
            },
            conviction=[], val_rows=400, train_rows=2000, holdout_rows=120,
        )
        save_event_report(record, settings.resolved_artifact_dir / "reports")

        body = client.get("/research/events").json()
        assert body["count"] == 1
        study = body["studies"][0]
        assert study["best_precision"] == 0.70
        assert study["best_lift"] == 1.27
        assert study["best_precision_ci"] == {"lower": 0.61, "upper": 0.78}
        assert study["split"]["holdout_rows_untouched"] == 120

    def test_event_study_without_significant_row_reports_no_best(self, client, settings):
        from cqip.events import event_study_record, save_event_report

        record = event_study_record(
            symbol="ETH_USDT", interval="4h", model_kind="lstm",
            horizon=1, seq_len=48, epochs=8,
            event={"verdict": "no-edge", "event_rate": 0.3, "n_events": 9, "n": 30,
                   "threshold": 0.05, "by_k": []},
            conviction=[], val_rows=30, train_rows=500, holdout_rows=0,
        )
        save_event_report(record, settings.resolved_artifact_dir / "reports")

        study = client.get("/research/events").json()["studies"][0]
        assert study["best_precision"] is None
        assert study["best_precision_ci"] is None
        assert study["n_events"] == 9


class TestDashboardRoute:
    def test_dashboard_and_root_serve_the_page(self, client):
        for path in ("/", "/dashboard"):
            response = client.get(path, follow_redirects=True)
            assert response.status_code == 200, path
            body = response.text
            assert "CQIP Research Monitor" in body
            # Self-contained by construction: no external asset references.
            for marker in ('src="http', 'href="http', "@import", "cdn.jsdelivr", "unpkg.com"):
                assert marker not in body, marker

    def test_prices_stream_honours_interval_filter(self, client, store, synthetic_candles):
        frame = synthetic_candles.copy()
        frame["interval"] = "1h"
        hourly = store.append_frame("ohlcv", frame)
        store.append_frame("ohlcv", frame.assign(interval="4h", close=999.0))

        assert hourly > 0
        with client.websocket_connect("/stream/prices?interval=4h") as ws:
            payload = ws.receive_json()

        data = payload["data"]
        assert data["interval"] == "4h"
        # Regression: the interval was accepted and then ignored, so a 1h close
        # was served as though the client had asked for 4h.
        assert data["symbols"], "expected at least one 4h symbol"
        assert all(sym["close"] == 999.0 for sym in data["symbols"].values())
