"""FastAPI application exposing data, models, training, backtests and jobs."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from cqip.config import Settings, get_settings
from cqip.db import Engine, get_coverage, get_engine
from cqip.ingest import (
    CANDLES_DATASET,
    NEWS_DATASET,
    ONCHAIN_DATASET,
    Ingestor,
)
from cqip.jobs import JobContext, JobManager, serialize_job
from cqip.logging_config import configure_logging, get_logger
from cqip.pipeline import TrainingPipeline
from cqip.providers import available_providers
from cqip.schemas import (
    BacktestRequest,
    CandleSeriesResponse,
    CoverageEntry,
    CoverageResponse,
    DatasetStats,
    ErrorResponse,
    ForecastPoint,
    ForecastSeriesResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    JobListResponse,
    JobResponse,
    JobSubmitResponse,
    ModelInfo,
    ModelListResponse,
    NewsItem,
    NewsResponse,
    PredictionResponse,
    RunInfo,
    RunListResponse,
    TrainForecasterRequest,
    TrainRLRequest,
)
from cqip.storage import ParquetStore
from cqip.tracking import ModelRegistry, serialize_model
from cqip.utils import utc_now

logger = get_logger(__name__)

state: dict[str, Any] = {"job_manager": None}


def get_store(settings: Settings = Depends(get_settings)) -> ParquetStore:
    return ParquetStore(settings=settings)


def get_ingestor(settings: Settings = Depends(get_settings)) -> Ingestor:
    return Ingestor(settings=settings)


def get_engine_dep(settings: Settings = Depends(get_settings)) -> Engine:
    return get_engine(settings)


def get_registry(engine: Engine = Depends(get_engine_dep)) -> ModelRegistry:
    return ModelRegistry(engine)


def get_pipeline(
    engine: Engine = Depends(get_engine_dep), settings: Settings = Depends(get_settings)
) -> TrainingPipeline:
    return TrainingPipeline(settings=settings, engine=engine)


def get_job_manager() -> JobManager:
    manager = state.get("job_manager")
    if manager is None:
        raise HTTPException(status_code=503, detail="Job manager unavailable")
    return manager


def register_job_handlers(manager: JobManager, settings: Settings, engine: Engine) -> None:
    def handle_ingest(context: JobContext, symbols: list[str], interval: str = "1h",
                       providers: list[str] | None = None, lookback_days: int = 730,
                       include_news: bool = True, include_onchain: bool = True) -> dict[str, Any]:
        ingestor = Ingestor(settings=settings, engine=engine)

        async def run() -> dict[str, Any]:
            context.set_stage("ingesting market data", 0.05)
            candles = await ingestor.ingest_candles(
                symbols, interval, providers, lookback_days=lookback_days
            )
            context.check_cancelled()
            context.set_progress(0.6, "market data ingested")

            news_result = None
            if include_news:
                context.set_stage("ingesting news", 0.65)
                news_result = await ingestor.ingest_news()

            onchain_results: list[Any] = []
            if include_onchain:
                context.set_stage("ingesting on-chain metrics", 0.8)
                onchain_results = await ingestor.ingest_onchain()

            return {
                "candles": [r.as_dict() for r in candles],
                "news": news_result.as_dict() if news_result else None,
                "onchain": [r.as_dict() for r in onchain_results],
            }

        # Job handlers run on a worker thread, so no event loop is active here.
        return asyncio.run(run())

    def handle_train_forecaster(context: JobContext, symbol: str, interval: str = "1h",
                                horizon: int = 1, kind: str | None = None,
                                epochs: int | None = None, seq_len: int | None = None,
                                start: str | None = None, end: str | None = None) -> dict[str, Any]:
        from datetime import datetime

        context.set_stage("building features", 0.1)
        pipeline = TrainingPipeline(settings=settings, engine=engine)
        result = pipeline.train_forecaster(
            symbol, interval, horizon=horizon, kind=kind, epochs=epochs, seq_len=seq_len,
            start=datetime.fromisoformat(start) if start else None,
            end=datetime.fromisoformat(end) if end else None,
        )
        context.set_progress(1.0, "training complete")
        return result

    def handle_train_rl(context: JobContext, symbol: str, interval: str = "1h",
                        timesteps: int | None = None, n_envs: int = 1,
                        start: str | None = None, end: str | None = None) -> dict[str, Any]:
        from datetime import datetime

        context.set_stage("building features", 0.1)
        pipeline = TrainingPipeline(settings=settings, engine=engine)
        result = pipeline.train_rl_agent(
            symbol, interval, timesteps=timesteps, n_envs=n_envs,
            start=datetime.fromisoformat(start) if start else None,
            end=datetime.fromisoformat(end) if end else None,
        )
        context.set_progress(1.0, "training complete")
        return result

    def handle_backtest(context: JobContext, model_id: str, symbol: str | None = None,
                        interval: str | None = None, threshold: float = 0.0,
                        n_folds: int = 3) -> dict[str, Any]:
        context.set_stage("running walk-forward backtest", 0.15)
        pipeline = TrainingPipeline(settings=settings, engine=engine)
        result = pipeline.backtest_predictions(
            model_id, symbol=symbol, interval=interval,
            threshold=threshold, n_folds=n_folds,
        )
        context.set_progress(1.0, "backtest complete")
        return result

    manager.register("ingest", handle_ingest)
    manager.register("train_forecaster", handle_train_forecaster)
    manager.register("train_rl", handle_train_rl)
    manager.register("backtest", handle_backtest)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Bind the settings the app was created with. Reading get_settings()
        # again here ignored create_app(settings=...) and pointed the job
        # manager at the global data directory instead of the caller's.
        configure_logging(settings)
        settings.ensure_directories()

        engine = get_engine(settings)
        manager = JobManager(settings=settings, engine=engine, max_workers=2)
        register_job_handlers(manager, settings, engine)
        state["job_manager"] = manager
        app.state.settings = settings
        app.state.engine = engine
        app.state.job_manager = manager

        logger.info("api_startup", extra={
            "environment": settings.environment, "providers": available_providers(),
        })

        try:
            yield
        finally:
            manager.shutdown(wait=False)
            logger.info("api_shutdown")

    app = FastAPI(
        title="Crypto Quant Intelligence Platform",
        description=(
            "Market, news and on-chain data ingestion, deep-learning forecasting, "
            "reinforcement-learning trading and walk-forward backtesting."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    health = APIRouter(prefix="/health", tags=["system"])
    data = APIRouter(prefix="/data", tags=["data"])
    models = APIRouter(prefix="/models", tags=["models"])
    training = APIRouter(prefix="/training", tags=["training"])
    jobs = APIRouter(prefix="/jobs", tags=["jobs"])
    stream = APIRouter(prefix="/stream", tags=["streaming"])

    @health.get("", response_model=HealthResponse)
    @health.get("/", response_model=HealthResponse, include_in_schema=False)
    async def read_health(engine: Engine = Depends(get_engine_dep)) -> HealthResponse:
        db_status = engine.healthcheck()
        sentiment_backend = None
        try:
            manager = state.get("job_manager")
            del manager
            from cqip.sentiment import SentimentScorer

            scorer = SentimentScorer(settings)
            scorer._load()
            sentiment_backend = scorer.backend
        except Exception:  # noqa: BLE001
            sentiment_backend = "unknown"

        return HealthResponse(
            status="ok" if db_status.get("ok") else "degraded",
            version="0.1.0",
            environment=settings.environment,
            database=db_status,
            providers=available_providers(),
            sentiment_backend=sentiment_backend,
            data_dir=str(settings.data_dir),
        )

    @health.get("/datasets", response_model=list[DatasetStats])
    async def read_datasets(store: ParquetStore = Depends(get_store)) -> list[DatasetStats]:
        out: list[DatasetStats] = []
        for dataset in (CANDLES_DATASET, NEWS_DATASET, ONCHAIN_DATASET, "fear_greed"):
            path = store.dataset_path(dataset)
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0
            start, end = store.date_range(dataset)
            out.append(DatasetStats(
                dataset=dataset, exists=store.exists(dataset), rows=store.row_count(dataset),
                columns=store.columns(dataset), start=start, end=end, size_bytes=size,
            ))
        return out

    @data.get("/candles", response_model=CandleSeriesResponse)
    async def read_candles(
        symbol: str = Query(default="BTC_USDT"),
        interval: str = Query(default="1h"),
        limit: int = Query(default=500, ge=1, le=20_000),
        exchange: str | None = Query(default=None),
        store: ParquetStore = Depends(get_store),
    ) -> CandleSeriesResponse:
        normalized = symbol.replace("/", "_").replace("-", "_").upper()
        filters = [("symbol", "==", normalized)]
        if exchange:
            filters.append(("exchange", "==", exchange))
        # The interval must be part of the filter. Without it a symbol that has
        # both 1h and 1d history comes back with the two interleaved by
        # timestamp, so a chart of "BTC 1d" would silently draw a mixture of
        # hourly and daily bars.
        filters.append(("interval", "==", interval))

        frame = store.read(CANDLES_DATASET, filters=filters)
        if frame.empty:
            raise HTTPException(status_code=404, detail=f"No candles for {normalized} {interval}")

        # `limit` means the most recent N bars, not the oldest N. ParquetStore
        # applies head() after sorting ascending, so passing the limit straight
        # through silently returns a chart of stale history from the start of
        # the dataset. Tail first, then re-sort so the response stays ascending.
        frame = frame.sort_values("ts").tail(limit).reset_index(drop=True)
        columns = ["ts", "open", "high", "low", "close", "volume"]
        available = [c for c in columns if c in frame.columns]

        coverage = None
        try:
            engine = get_engine(settings)
            entries = get_coverage(engine, CANDLES_DATASET, normalized, interval)
            if entries:
                entry = entries[0]
                coverage = {"provider": entry.provider, "start": entry.start, "end": entry.end,
                            "rows": entry.rows}
        except Exception:  # noqa: BLE001
            coverage = None

        return CandleSeriesResponse(
            symbol=normalized, interval=interval,
            exchange=str(frame["exchange"].iloc[-1]) if "exchange" in frame.columns else None,
            count=len(frame),
            candles=frame[available].to_dict(orient="records"),
            coverage=coverage,
        )

    @data.get("/symbols")
    async def read_symbols(store: ParquetStore = Depends(get_store)) -> dict[str, Any]:
        symbols = store.symbols(CANDLES_DATASET)
        return {"count": len(symbols), "symbols": symbols}

    @data.get("/news", response_model=NewsResponse)
    async def read_news(
        limit: int = Query(default=50, ge=1, le=500),
        source: str | None = Query(default=None),
        category: str | None = Query(default=None),
        store: ParquetStore = Depends(get_store),
    ) -> NewsResponse:
        frame = store.read(NEWS_DATASET, limit=limit * 3)
        if frame.empty:
            return NewsResponse(count=0, items=[], by_source={}, by_category={})

        if source:
            frame = frame[frame["source"] == source]
        if category:
            frame = frame[frame["category"] == category]
        frame = frame.sort_values("ts", ascending=False).head(limit)

        by_source = frame["source"].value_counts().to_dict()
        by_category = frame["category"].value_counts().to_dict()

        sentiment = None
        if "sentiment_score" in frame.columns:
            sentiment = {
                "mean": float(frame["sentiment_score"].mean()),
                "std": float(frame["sentiment_score"].std()) if len(frame) > 1 else 0.0,
                "positive_ratio": float(frame["is_positive"].mean()) if "is_positive" in frame else None,
                "negative_ratio": float(frame["is_negative"].mean()) if "is_negative" in frame else None,
            }

        items = [
            NewsItem(
                uid=row.get("uid", ""), source=str(row.get("source", "")),
                title=str(row.get("title", "")), url=str(row.get("url", "")),
                ts=row["ts"], category=str(row.get("category", "general")),
                sentiment_score=float(row["sentiment_score"]) if pd.notna(row.get("sentiment_score")) else None,
                summary=str(row.get("summary", ""))[:280] or None,
            )
            for _, row in frame.iterrows()
        ]
        return NewsResponse(count=len(items), items=items, sentiment=sentiment,
                            by_source=by_source, by_category=by_category)

    @data.get("/coverage", response_model=CoverageResponse)
    async def read_coverage(
        dataset: str = Query(default=CANDLES_DATASET),
        symbol: str | None = Query(default=None),
        interval: str | None = Query(default=None),
        engine: Engine = Depends(get_engine_dep),
    ) -> CoverageResponse:
        entries = get_coverage(engine, dataset, (symbol or "").upper(), interval or "")
        return CoverageResponse(
            count=len(entries),
            entries=[CoverageEntry(
                dataset=e.dataset, symbol=e.symbol, interval=e.interval, provider=e.provider,
                start=e.start, end=e.end, rows=e.rows,
            ) for e in entries],
        )

    @data.post("/ingest", response_model=IngestResponse, status_code=202)
    async def trigger_ingest(
        payload: IngestRequest,
        background: BackgroundTasks,
        engine: Engine = Depends(get_engine_dep),
    ) -> IngestResponse:
        import uuid

        from cqip.db import record_run

        run_id = f"ingest_{uuid.uuid4().hex[:12]}"
        record_run(engine, run_id, "ingest", payload.model_dump(mode="json"))

        async def run_ingestion() -> None:
            # Background tasks attached to an async route run on the request's
            # event loop, so this must be awaited rather than merely constructed.
            from cqip.db import finish_run

            try:
                ingestor = Ingestor(settings=settings, engine=engine)
                await ingestor.ingest_all(
                    payload.symbols, payload.interval, payload.providers,
                    lookback_days=payload.lookback_days,
                    news=payload.include_news, onchain=payload.include_onchain,
                )
            except Exception as exc:  # noqa: BLE001
                finish_run(engine, run_id, "failed", message=str(exc)[:500])
                logger.error("ingest_background_failed", extra={
                    "run_id": run_id, "error": str(exc)[:300],
                })
            else:
                finish_run(engine, run_id, "succeeded", {})

        background.add_task(run_ingestion)
        return IngestResponse(
            run_id=run_id, candles=[], onchain=[], total_rows=0,
            news={"status": "scheduled", "run_id": run_id},
        )

    @models.get("", response_model=ModelListResponse)
    @models.get("/", response_model=ModelListResponse, include_in_schema=False)
    async def read_models(
        kind: str | None = Query(default=None),
        symbol: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
        registry: ModelRegistry = Depends(get_registry),
    ) -> ModelListResponse:
        # `kind` is accepted as either the task kind ("forecaster") or the
        # architecture ("mlp"/"lstm"/"transformer"). Registry `kind` is the
        # literal "forecaster" for every deep-learning model, so filtering it by
        # architecture used to return an empty list for every real query.
        architectures = {"mlp", "lstm", "transformer"}
        architecture = kind.lower() if kind and kind.lower() in architectures else None
        task_kind = kind if kind and architecture is None else None
        records = registry.list_models(
            kind=task_kind, symbol=symbol, architecture=architecture, limit=limit
        )
        return ModelListResponse(
            count=len(records),
            models=[ModelInfo(**serialize_model(record)) for record in records],
        )

    @models.post("/train", response_model=JobSubmitResponse, status_code=202)
    async def train_forecaster(
        payload: TrainForecasterRequest, manager: JobManager = Depends(get_job_manager)
    ) -> JobSubmitResponse:
        record = manager.submit("train_forecaster", payload.model_dump(mode="json"))
        return JobSubmitResponse(
            job_id=record.job_id, kind=record.kind, status=record.status,
            poll_url=f"/jobs/{record.job_id}",
        )

    @models.post("/train-rl", response_model=JobSubmitResponse, status_code=202)
    async def train_rl(
        payload: TrainRLRequest, manager: JobManager = Depends(get_job_manager)
    ) -> JobSubmitResponse:
        record = manager.submit("train_rl", payload.model_dump(mode="json"))
        return JobSubmitResponse(
            job_id=record.job_id, kind=record.kind, status=record.status,
            poll_url=f"/jobs/{record.job_id}",
        )

    @models.post("/backtest", response_model=JobSubmitResponse, status_code=202)
    async def start_backtest(
        payload: BacktestRequest, manager: JobManager = Depends(get_job_manager)
    ) -> JobSubmitResponse:
        record = manager.submit("backtest", payload.model_dump(mode="json"))
        return JobSubmitResponse(
            job_id=record.job_id, kind=record.kind, status=record.status,
            poll_url=f"/jobs/{record.job_id}",
        )

    @models.get("/{model_id}/predict", response_model=PredictionResponse)
    async def predict(model_id: str, pipeline: TrainingPipeline = Depends(get_pipeline)) -> PredictionResponse:
        try:
            outcome = await asyncio.to_thread(pipeline.predict_latest, model_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc
        return PredictionResponse(**outcome)

    @models.get("/{model_id}/forecast", response_model=ForecastSeriesResponse)
    async def forecast(
        model_id: str,
        limit: int = Query(default=100, ge=1, le=1000),
        pipeline: TrainingPipeline = Depends(get_pipeline),
    ) -> ForecastSeriesResponse:
        try:
            frame = await asyncio.to_thread(pipeline.forecast_series, model_id, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc

        if frame.empty:
            return ForecastSeriesResponse(model_id=model_id, symbol="", interval="", count=0, points=[])

        return ForecastSeriesResponse(
            model_id=model_id,
            symbol=str(frame.get("symbol", pd.Series([""])).iloc[0]) or "",
            interval="",
            count=len(frame),
            points=[ForecastPoint(**row) for row in frame.to_dict(orient="records")],
        )

    @models.get("/{model_id}", response_model=ModelInfo)
    async def read_model(model_id: str, registry: ModelRegistry = Depends(get_registry)) -> ModelInfo:
        record = registry.get_active(model_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found")
        return ModelInfo(**serialize_model(record))

    @models.post("/{model_id}/promote", response_model=ModelInfo)
    async def promote(model_id: str, registry: ModelRegistry = Depends(get_registry)) -> ModelInfo:
        if not registry.promote(model_id, "production"):
            raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found")
        record = registry.get_active(model_id)
        return ModelInfo(**serialize_model(record))

    @models.delete("/{model_id}", status_code=204)
    async def delete_model(model_id: str, registry: ModelRegistry = Depends(get_registry)) -> None:
        if not registry.remove(model_id):
            raise HTTPException(status_code=404, detail=f"Model {model_id!r} not found")

    @jobs.post("", response_model=JobSubmitResponse, status_code=202)
    @jobs.post("/", response_model=JobSubmitResponse, status_code=202, include_in_schema=False)
    async def submit_job(
        payload: dict[str, Any] = Body(
            ...,
            description='Job request: {"kind": "ingest", "params": {...}}',
            examples=[{"kind": "ingest", "params": {"symbols": ["BTC_USDT"], "interval": "1h"}}],
        ),
        manager: JobManager = Depends(get_job_manager),
    ) -> JobSubmitResponse:
        kind = payload.get("kind")
        if not kind:
            raise HTTPException(status_code=400, detail="Field 'kind' is required")
        params = payload.get("params")
        if params is None:
            params = {}
        # `params or {}` would turn any falsy value (an empty list, 0, "")
        # into {} and silently skip the type check.
        if not isinstance(params, dict):
            raise HTTPException(status_code=400, detail="Field 'params' must be an object")

        try:
            record = manager.submit(str(kind), params)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JobSubmitResponse(
            job_id=record.job_id, kind=record.kind, status=record.status,
            poll_url=f"/jobs/{record.job_id}",
        )

    @jobs.get("", response_model=JobListResponse)
    @jobs.get("/", response_model=JobListResponse, include_in_schema=False)
    async def list_jobs(
        kind: str | None = Query(default=None),
        status: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=500),
        manager: JobManager = Depends(get_job_manager),
    ) -> JobListResponse:
        records = manager.list(kind=kind, status=status, limit=limit)
        return JobListResponse(
            count=len(records), jobs=[JobResponse(**serialize_job(r)) for r in records]
        )

    @jobs.get("/{job_id}", response_model=JobResponse)
    async def read_job(job_id: str, manager: JobManager = Depends(get_job_manager)) -> JobResponse:
        record = manager.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
        return JobResponse(**serialize_job(record))

    @jobs.delete("/{job_id}", response_model=JobResponse)
    async def cancel_job(job_id: str, manager: JobManager = Depends(get_job_manager)) -> JobResponse:
        if not manager.cancel(job_id):
            raise HTTPException(status_code=409, detail="Job not found or already terminal")
        record = manager.get(job_id)
        return JobResponse(**serialize_job(record))

    @jobs.get("/{job_id}/stream")
    async def stream_job(job_id: str, manager: JobManager = Depends(get_job_manager)) -> StreamingResponse:
        record = manager.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")

        async def event_stream() -> AsyncIterator[str]:
            queue = manager.subscribe(job_id)
            try:
                yield f"data: {json.dumps({'status': record.status, 'progress': record.progress})}\n\n"
                if record.is_terminal:
                    return
                while True:
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=30.0)
                    except TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    yield f"data: {json.dumps(payload, default=str)}\n\n"
                    if payload.get("status") in {"succeeded", "failed", "cancelled"}:
                        break
            finally:
                manager.unsubscribe(job_id, queue)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @training.get("/runs", response_model=RunListResponse)
    async def list_runs(
        kind: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=500),
        engine: Engine = Depends(get_engine_dep),
    ) -> RunListResponse:
        from cqip.db import list_runs as fetch_runs

        runs = fetch_runs(engine, kind=kind, limit=limit)
        return RunListResponse(count=len(runs), runs=[RunInfo(
            run_id=r.run_id, kind=r.kind, status=r.status, params=r.params or {},
            metrics=r.metrics or {}, message=r.message, started_at=r.started_at,
            finished_at=r.finished_at, duration_seconds=r.duration_seconds,
        ) for r in runs])

    @training.get("/runs/compare")
    async def compare_runs(
        run_ids: list[str] = Query(min_length=1),
        engine: Engine = Depends(get_engine_dep),
    ) -> dict[str, Any]:
        from cqip.tracking import ExperimentTracker

        tracker = ExperimentTracker(engine, settings)
        return tracker.compare_runs(run_ids)

    @stream.websocket("/prices")
    async def stream_prices(websocket: WebSocket, interval: str = "1h") -> None:
        await websocket.accept()
        store = ParquetStore(settings=settings)
        try:
            while True:
                symbols = store.symbols(CANDLES_DATASET)
                payload: dict[str, Any] = {
                    "ts": utc_now().isoformat(), "interval": interval, "symbols": {},
                }
                for symbol in symbols[:50]:
                    # The interval is part of the candles key, so a bar from
                    # another timeframe must be excluded rather than silently
                    # served as if it were this one.
                    frame = store.read(
                        CANDLES_DATASET,
                        filters=[("symbol", "==", symbol), ("interval", "==", interval)],
                        limit=1,
                    )
                    if not frame.empty:
                        row = frame.iloc[-1]
                        payload["symbols"][symbol] = {
                            "ts": str(row["ts"]), "close": float(row.get("close", 0.0)),
                            "volume": float(row.get("volume", 0.0)),
                        }
                await websocket.send_json({"type": "prices", "data": payload})
                await asyncio.sleep(5.0)
        except WebSocketDisconnect:
            logger.info("ws_prices_disconnected")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ws_prices_error", extra={"error": str(exc)[:200]})
            with contextlib.suppress(Exception):
                await websocket.close()

    @stream.websocket("/jobs/{job_id}")
    async def stream_job_ws(websocket: WebSocket, job_id: str) -> None:
        await websocket.accept()
        manager = state.get("job_manager")
        if manager is None:
            await websocket.close()
            return

        queue = manager.subscribe(job_id)
        try:
            record = manager.get(job_id)
            if record is None:
                await websocket.send_json({"type": "error", "data": {"detail": "job not found"}})
                return

            await websocket.send_json({
                "type": "status",
                "data": {"job_id": job_id, "status": record.status, "progress": record.progress},
            })
            if record.is_terminal:
                return

            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=30.0)
                except TimeoutError:
                    await websocket.send_json({"type": "ping", "data": {}})
                    continue
                await websocket.send_json({"type": "progress", "data": payload})
                if payload.get("status") in {"succeeded", "failed", "cancelled"}:
                    break
        except WebSocketDisconnect:
            logger.info("ws_job_disconnected", extra={"job_id": job_id})
        finally:
            manager.unsubscribe(job_id, queue)

    app.include_router(health)
    app.include_router(data)
    app.include_router(models)
    app.include_router(training)
    app.include_router(jobs)
    app.include_router(stream)

    # Research monitor. Mounted last so `/research/*` never shadows a core route.
    from cqip.research_api import router as research_router

    app.include_router(research_router)

    # The dashboards are single self-contained HTML files served straight off
    # disk. No build step and no CDN, so they work offline and cannot break
    # because a node toolchain or a chart library changed underneath it.
    web_root = Path(__file__).resolve().parents[2] / "web"

    def serve_html(path: Path) -> Any:
        """Serve a dashboard file, revalidating on every request.

        These pages are edited in place, so a cached copy means a reload that
        silently serves the previous version of the UI.
        """
        from fastapi.responses import FileResponse

        return FileResponse(
            path,
            media_type="text/html",
            headers={"Cache-Control": "no-cache, must-revalidate"},
        )

    dashboard_path = web_root / "dashboard.html"
    if dashboard_path.is_file():
        @app.get("/", include_in_schema=False)
        async def dashboard() -> Any:
            return serve_html(dashboard_path)

        @app.get("/dashboard", include_in_schema=False)
        async def dashboard_alias() -> Any:
            from fastapi.responses import RedirectResponse

            return RedirectResponse(url="/")

    # The probability workbench is a separate page rather than another tab of
    # the research monitor: it reads live model inference (/models/{id}/predict),
    # which the monitor deliberately never does, since scoring a holdout twice
    # would spend the one read it is allowed.
    analysis_path = web_root / "analysis.html"
    if analysis_path.is_file():
        @app.get("/analysis", include_in_schema=False)
        async def analysis() -> Any:
            return serve_html(analysis_path)

    @app.exception_handler(KeyError)
    async def handle_key_error(_request: Any, exc: KeyError) -> Any:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=400, content=ErrorResponse(
            error="KeyError", detail=str(exc), status_code=400
        ).model_dump())

    @app.exception_handler(ValueError)
    async def handle_value_error(_request: Any, exc: ValueError) -> Any:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=400, content=ErrorResponse(
            error="ValueError", detail=str(exc), status_code=400
        ).model_dump())

    return app


app = create_app()


def main() -> None:  # pragma: no cover - entrypoint
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "cqip.api:app", host=settings.api_host, port=settings.api_port,
        reload=not settings.is_production, log_level=settings.log_level.lower(),
    )


__all__ = ["app", "create_app"]