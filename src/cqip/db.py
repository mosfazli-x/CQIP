"""SQLAlchemy metadata store for runs, models, predictions, experiments and jobs."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
    func,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)
from sqlalchemy.types import TypeDecorator

from cqip.config import Settings, get_settings
from cqip.logging_config import get_logger
from cqip.utils import utc_now

logger = get_logger(__name__)


class UTCDateTime(TypeDecorator):
    """Persist datetimes as naive UTC and always return tz-aware UTC.

    SQLite has no native timestamp type, so a plain ``DateTime`` column silently
    drops ``tzinfo`` on write. Callers legitimately pass tz-aware pandas
    timestamps (``frame["ts"].min()``), and comparing one of those against a
    naive value read back from the database raises ``TypeError: can't compare
    offset-naive and offset-aware datetimes``. Normalising in both directions
    here keeps the rest of the codebase on aware datetimes only.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(
        self, value: datetime | None, dialect: Any
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(
        self, value: datetime | None, dialect: Any
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return utc_now()


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )


class Run(Base, TimestampMixin):
    """One execution of an ingestion, training or backtest job."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="running", index=True, nullable=False)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    duration_seconds: Mapped[float | None] = mapped_column(Float)

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"


class ModelVersion(Base, TimestampMixin):
    """Registry entry for a trained model artifact."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(96), index=True, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    interval: Mapped[str] = mapped_column(String(8), nullable=False)
    framework: Mapped[str] = mapped_column(String(32), default="torch")
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    feature_schema: Mapped[list[str]] = mapped_column(JSON, default=list)
    target: Mapped[str] = mapped_column(String(64), default="log_return_1")
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    stage: Mapped[str] = mapped_column(String(16), default="production", index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    trained_from: Mapped[datetime | None] = mapped_column(UTCDateTime)
    trained_to: Mapped[datetime | None] = mapped_column(UTCDateTime)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.run_id"), nullable=True)

    __table_args__ = (
        Index("ix_model_versions_lookup", "model_id", "kind", "symbol", "is_active"),
    )


class Prediction(Base, TimestampMixin):
    """Persisted model output for an auditable prediction history."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(96), index=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    ts: Mapped[datetime] = mapped_column(UTCDateTime, index=True, nullable=False)
    horizon: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    predicted_return: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_price: Mapped[float] = mapped_column(Float)
    lower: Mapped[float | None] = mapped_column(Float)
    upper: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    features: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    __table_args__ = (
        Index("ix_predictions_model_ts", "model_id", "ts"),
        Index("ix_predictions_symbol_ts", "symbol", "ts"),
    )


class Experiment(Base, TimestampMixin):
    """MLflow-style experiment grouping a set of runs."""

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(96), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    tags: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    description: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class BacktestResult(Base, TimestampMixin):
    """Walk-forward backtest outcome with full metrics and trade log summary."""

    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    backtest_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    strategy: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    model_id: Mapped[str | None] = mapped_column(String(96), index=True, nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    interval: Mapped[str] = mapped_column(String(8), nullable=False)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    folds: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    equity_curve: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    trades: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    period_start: Mapped[datetime | None] = mapped_column(UTCDateTime)
    period_end: Mapped[datetime | None] = mapped_column(UTCDateTime)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.run_id"), nullable=True)


class JobRecord(Base, TimestampMixin):
    """Durable record of a background job for status polling and audit."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(48), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True, nullable=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    stage: Mapped[str | None] = mapped_column(String(64))
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    queued_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    @property
    def is_terminal(self) -> bool:
        return self.status in {"succeeded", "failed", "cancelled"}


class DataCoverage(Base, TimestampMixin):
    """Tracks ingested coverage so incremental runs skip completed ranges."""

    __tablename__ = "data_coverage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    interval: Mapped[str] = mapped_column(String(8), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    start: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    end: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    rows: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        Index("ix_data_coverage_key", "dataset", "symbol", "interval", "provider", unique=True),
    )


class Engine:
    """Thin wrapper owning the SQLAlchemy engine and session lifecycle."""

    def __init__(self, settings: Settings | None = None, url: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.url = url or self.settings.resolved_db_url
        connect_args: dict[str, Any] = {}
        if self.url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            connect_args["timeout"] = 30
        self.engine = create_engine(
            self.url,
            future=True,
            echo=False,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        if self.url.startswith("sqlite"):

            @event.listens_for(self.engine, "connect")
            def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        self.session_factory = sessionmaker(
            bind=self.engine, expire_on_commit=False, future=True
        )
        self.create_all()

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def healthcheck(self) -> dict[str, Any]:
        try:
            with self.session() as session:
                session.execute(select(func.count()).select_from(Run))
            return {"ok": True, "url": self.url.split("://")[0]}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "url": self.url.split("://")[0], "error": str(exc)[:200]}

    def dispose(self) -> None:
        self.engine.dispose()


_default_engine: Engine | None = None
_engines: dict[str, Engine] = {}


def get_engine(settings: Settings | None = None) -> Engine:
    """Return a process-wide engine, one per database URL.

    A single global engine that ignores `settings` meant the first caller won:
    a test run or a worker configured with a different data_dir silently kept
    using whatever database happened to be opened first.
    """
    global _default_engine
    if settings is None:
        if _default_engine is None:
            _default_engine = Engine()
        return _default_engine

    key = settings.resolved_db_url
    engine = _engines.get(key)
    if engine is None:
        engine = Engine(settings=settings)
        _engines[key] = engine
    return engine


def reset_engine() -> None:
    global _default_engine
    if _default_engine is not None:
        _default_engine.dispose()
    _default_engine = None
    for engine in list(_engines.values()):
        engine.dispose()
    _engines.clear()


def record_run(
    engine: Engine,
    run_id: str,
    kind: str,
    params: dict[str, Any] | None = None,
    status: str = "running",
) -> Run:
    with engine.session() as session:
        run = Run(run_id=run_id, kind=kind, params=params or {}, status=status)
        session.add(run)
        session.flush()
        return run


def finish_run(
    engine: Engine,
    run_id: str,
    status: str,
    metrics: dict[str, Any] | None = None,
    message: str | None = None,
) -> None:
    with engine.session() as session:
        run = session.execute(
            select(Run).where(Run.run_id == run_id)
        ).scalar_one_or_none()
        if run is None:
            return
        run.status = status
        run.metrics = metrics or {}
        run.message = message
        run.finished_at = _utcnow()
        run.duration_seconds = (run.finished_at - run.started_at).total_seconds()


def upsert_coverage(
    engine: Engine,
    dataset: str,
    symbol: str,
    interval: str,
    provider: str,
    start: datetime,
    end: datetime,
    rows: int,
) -> None:
    with engine.session() as session:
        existing = session.execute(
            select(DataCoverage).where(
                DataCoverage.dataset == dataset,
                DataCoverage.symbol == symbol,
                DataCoverage.interval == interval,
                DataCoverage.provider == provider,
            )
        ).scalar_one_or_none()
        if existing is None:
            session.add(DataCoverage(
                dataset=dataset, symbol=symbol, interval=interval, provider=provider,
                start=start, end=end, rows=rows,
            ))
        else:
            existing.start = min(existing.start, start)
            existing.end = max(existing.end, end)
            existing.rows = max(existing.rows, rows)


def to_jsonable(value: Any) -> Any:
    """Coerce a value into something a JSON column can persist.

    Handler results and backtest payloads carry datetimes and numpy scalars,
    which raise "Object of type datetime is not JSON serializable" on flush and
    would otherwise turn a successful job into a failed one.
    """
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return value


def get_coverage(engine: Engine, dataset: str, symbol: str = "", interval: str = "") -> list[DataCoverage]:
    with engine.session() as session:
        stmt = select(DataCoverage).where(DataCoverage.dataset == dataset)
        if symbol:
            stmt = stmt.where(DataCoverage.symbol == symbol)
        if interval:
            stmt = stmt.where(DataCoverage.interval == interval)
        return list(session.execute(stmt).scalars())


def list_runs(engine: Engine, kind: str | None = None, limit: int = 50) -> list[Run]:
    with engine.session() as session:
        stmt = select(Run).order_by(Run.started_at.desc()).limit(limit)
        if kind:
            stmt = stmt.where(Run.kind == kind)
        return list(session.execute(stmt).scalars())


__all__ = [
    "BacktestResult",
    "Base",
    "DataCoverage",
    "Engine",
    "Experiment",
    "JobRecord",
    "ModelVersion",
    "Prediction",
    "Run",
    "finish_run",
    "get_coverage",
    "get_engine",
    "list_runs",
    "record_run",
    "reset_engine",
    "to_jsonable",
    "upsert_coverage",
]