"""In-process background job manager with SQLite-backed status and progress.

Training is CPU-bound and long-running, so it must not block the event loop.
Jobs run in a dedicated worker thread with cooperative cancellation, and all
status transitions are persisted so the API can report progress across restarts.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from sqlalchemy import select

from cqip.config import Settings, get_settings
from cqip.db import Engine, JobRecord, get_engine, to_jsonable
from cqip.logging_config import get_logger
from cqip.utils import utc_now

logger = get_logger(__name__)

TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


class Cancelled(Exception):
    """Raised inside a job body when a cancellation is requested."""


class JobContext:
    """Passed to job callables for progress reporting and cancellation checks."""

    def __init__(self, job_id: str, engine: Engine) -> None:
        self.job_id = job_id
        self.engine = engine
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        if self._cancelled:
            return True
        with self.engine.session() as session:
            record = session.execute(
                select(JobRecord.status).where(JobRecord.job_id == self.job_id)
            ).scalar_one_or_none()
        self._cancelled = record == "cancelled"
        return self._cancelled

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise Cancelled(f"Job {self.job_id} was cancelled")

    def set_progress(self, progress: float, stage: str | None = None) -> None:
        self._persist(progress=min(max(progress, 0.0), 1.0), stage=stage)

    def set_stage(self, stage: str, progress: float | None = None) -> None:
        self._persist(stage=stage, progress=progress)

    def _persist(self, **fields: Any) -> None:
        with self.engine.session() as session:
            record = session.execute(
                select(JobRecord).where(JobRecord.job_id == self.job_id)
            ).scalar_one_or_none()
            if record is None:
                return
            for key, value in fields.items():
                if value is not None:
                    setattr(record, key, value)

    def log(self, message: str) -> None:
        logger.info("job_progress", extra={"job_id": self.job_id, "stage": message})
        self.set_stage(message)


JobCallable = Callable[..., Any]


class JobManager:
    """Thread-pool backed job runner with durable status tracking."""

    def __init__(
        self,
        settings: Settings | None = None,
        engine: Engine | None = None,
        max_workers: int = 2,
    ) -> None:
        self.settings = settings or get_settings()
        self.engine = engine or get_engine(self.settings)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cqip-job")
        self._handlers: dict[str, JobCallable] = {}
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._lock = threading.Lock()

    def register(self, kind: str, handler: JobCallable) -> None:
        self._handlers[kind] = handler

    def registered_kinds(self) -> list[str]:
        return sorted(self._handlers)

    def submit(self, kind: str, params: dict[str, Any] | None = None,
               *, job_id: str | None = None) -> JobRecord:
        if kind not in self._handlers:
            raise KeyError(f"No handler registered for job kind {kind!r}. "
                           f"Available: {self.registered_kinds()}")

        job_id = job_id or f"job-{uuid.uuid4().hex[:14]}"
        record = JobRecord(
            job_id=job_id, kind=kind, status="queued",
            params=params or {}, queued_at=utc_now().replace(tzinfo=None),
        )
        with self.engine.session() as session:
            session.add(record)

        handler = self._handlers[kind]
        self.executor.submit(self._execute, job_id, handler, params or {})
        logger.info("job_submitted", extra={"job_id": job_id, "kind": kind})
        return record

    def _execute(self, job_id: str, handler: JobCallable, params: dict[str, Any]) -> None:
        context = JobContext(job_id, self.engine)
        started = utc_now().replace(tzinfo=None)

        with self.engine.session() as session:
            record = session.execute(
                select(JobRecord).where(JobRecord.job_id == job_id)
            ).scalar_one_or_none()
            if record is None:
                logger.error("job_missing_at_execution", extra={"job_id": job_id})
                return
            record.status = "running"
            record.started_at = started

        self._publish(job_id, {"status": "running", "stage": "starting"})

        try:
            result = handler(context, **params)
            status = "cancelled" if context.cancelled else "succeeded"
            self._finalize(job_id, status, result=result, started=started)
        except Cancelled:
            self._finalize(job_id, "cancelled", result={"message": "cancelled"}, started=started)
            logger.info("job_cancelled", extra={"job_id": job_id})
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"
            logger.error("job_failed", extra={
                "job_id": job_id, "error": detail[:300],
                "traceback": traceback.format_exc()[-1200:],
            })
            self._finalize(job_id, "failed", error=detail, started=started)

    def _finalize(self, job_id: str, status: str, *, result: Any = None,
                  error: str | None = None, started: datetime | None = None) -> None:
        finished = utc_now().replace(tzinfo=None)
        with self.engine.session() as session:
            record = session.execute(
                select(JobRecord).where(JobRecord.job_id == job_id)
            ).scalar_one_or_none()
            if record is None:
                return
            record.status = status
            record.finished_at = finished
            record.progress = 1.0 if status == "succeeded" else record.progress
            record.stage = status
            if error:
                record.error = error[:2000]
            if isinstance(result, dict):
                record.result = to_jsonable(result)
            elif result is not None:
                record.result = {"summary": str(result)[:4000]}
            if started:
                record.result = {
                    **(record.result or {}),
                    "duration_seconds": round((finished - started).total_seconds(), 3),
                }
            payload = {
                "status": status,
                "progress": record.progress,
                "result": record.result,
                "error": record.error,
            }

        self._publish(job_id, payload)
        logger.info("job_finished", extra={"job_id": job_id, "status": status})

    def get(self, job_id: str) -> JobRecord | None:
        with self.engine.session() as session:
            return session.execute(
                select(JobRecord).where(JobRecord.job_id == job_id)
            ).scalar_one_or_none()

    def list(self, *, kind: str | None = None, status: str | None = None,
             limit: int = 50) -> list[JobRecord]:
        with self.engine.session() as session:
            stmt = select(JobRecord).order_by(JobRecord.queued_at.desc()).limit(limit)
            if kind:
                stmt = stmt.where(JobRecord.kind == kind)
            if status:
                stmt = stmt.where(JobRecord.status == status)
            return list(session.execute(stmt).scalars())

    def cancel(self, job_id: str) -> bool:
        record = self.get(job_id)
        if record is None or record.status in TERMINAL_STATES:
            return False
        with self.engine.session() as session:
            target = session.execute(
                select(JobRecord).where(JobRecord.job_id == job_id)
            ).scalar_one_or_none()
            if target is not None:
                target.status = "cancelled"
                target.finished_at = utc_now().replace(tzinfo=None)
        logger.info("job_cancel_requested", extra={"job_id": job_id})
        return True

    def wait(self, job_id: str, timeout: float | None = None) -> JobRecord | None:
        import time as _time

        deadline = _time.monotonic() + timeout if timeout else None
        while True:
            record = self.get(job_id)
            if record is None:
                return None
            if record.status in TERMINAL_STATES:
                return record
            if deadline and _time.monotonic() > deadline:
                return record
            _time.sleep(0.1)

    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        with self._lock:
            self._subscribers.setdefault(job_id, set()).add(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            listeners = self._subscribers.get(job_id)
            if listeners:
                listeners.discard(queue)
                if not listeners:
                    self._subscribers.pop(job_id, None)

    def _publish(self, job_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            listeners = list(self._subscribers.get(job_id, set()))
        for queue in listeners:
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(payload)

    def shutdown(self, wait: bool = False) -> None:
        self.executor.shutdown(wait=wait, cancel_futures=not wait)


def serialize_job(record: JobRecord) -> dict[str, Any]:
    return {
        "job_id": record.job_id,
        "kind": record.kind,
        "status": record.status,
        "progress": record.progress,
        "stage": record.stage,
        "params": record.params,
        "result": record.result,
        "error": record.error,
        "queued_at": record.queued_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
    }


__all__ = ["Cancelled", "JobContext", "JobManager", "serialize_job"]