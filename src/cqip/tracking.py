"""Experiment tracking and model registry backed by SQLite."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select, update

from cqip.config import Settings, get_settings
from cqip.db import Engine, Experiment, ModelVersion, Run, get_engine
from cqip.logging_config import get_logger

logger = get_logger(__name__)

STAGE_RANK = {"staging": 0, "production": 1, "archived": 2}


class ExperimentTracker:
    """MLflow-style tracking: named experiments grouping parameterized runs."""

    def __init__(self, engine: Engine | None = None, settings: Settings | None = None) -> None:
        self.engine = engine or get_engine(settings)
        self.settings = settings or get_settings()

    def get_or_create_experiment(self, name: str, *, description: str | None = None,
                                 tags: dict[str, Any] | None = None) -> Experiment:
        experiment_id = f"exp-{uuid.uuid4().hex[:16]}"
        with self.engine.session() as session:
            existing = session.execute(
                select(Experiment).where(Experiment.name == name, Experiment.is_active.is_(True))
            ).scalar_one_or_none()
            if existing:
                return existing
            experiment = Experiment(
                experiment_id=experiment_id, name=name,
                description=description, tags=tags or {},
            )
            session.add(experiment)
            session.flush()
            return experiment

    def create_run(self, experiment_id: str, *, run_id: str | None = None,
                   kind: str = "train", params: dict[str, Any] | None = None) -> Run:
        from cqip.db import record_run

        run_id = run_id or f"{kind}-{uuid.uuid4().hex[:12]}"
        return record_run(self.engine, run_id, kind, params or {})

    def log_metrics(self, run_id: str, metrics: dict[str, Any]) -> None:
        with self.engine.session() as session:
            run = session.execute(select(Run).where(Run.run_id == run_id)).scalar_one_or_none()
            if run is None:
                logger.warning("log_metrics_missing_run", extra={"run_id": run_id})
                return
            run.metrics = {**(run.metrics or {}), **metrics}

    def finish(self, run_id: str, status: str = "succeeded",
               metrics: dict[str, Any] | None = None, message: str | None = None) -> None:
        from cqip.db import finish_run

        finish_run(self.engine, run_id, status, metrics, message)

    def list_runs(self, experiment_id: str | None = None, limit: int = 100) -> list[Run]:
        with self.engine.session() as session:
            stmt = select(Run).order_by(Run.started_at.desc()).limit(limit)
            if experiment_id:
                stmt = stmt.where(Run.params.like(f"%{experiment_id}%"))
            return list(session.execute(stmt).scalars())

    def compare_runs(self, run_ids: Sequence[str]) -> dict[str, Any]:
        metrics_by_run: dict[str, dict[str, Any]] = {}
        params_by_run: dict[str, dict[str, Any]] = {}
        statuses: dict[str, str] = {}

        with self.engine.session() as session:
            for run_id in run_ids:
                run = session.execute(
                    select(Run).where(Run.run_id == run_id)
                ).scalar_one_or_none()
                if run is None:
                    continue
                metrics_by_run[run_id] = run.metrics or {}
                params_by_run[run_id] = run.params or {}
                statuses[run_id] = run.status

        keys: set[str] = set()
        for metrics in metrics_by_run.values():
            keys.update(k for k, v in metrics.items() if isinstance(v, (int, float, str, bool)))

        comparison = {
            key: {run_id: metrics_by_run.get(run_id, {}).get(key) for run_id in run_ids}
            for key in sorted(keys)
        }
        best_key = None
        if "sharpe" in comparison:
            best_key = "sharpe"
        elif "directional_accuracy" in comparison:
            best_key = "directional_accuracy"

        best_run = None
        if best_key:
            candidates = {
                run_id: metrics_by_run[run_id][best_key]
                for run_id in run_ids
                if isinstance(metrics_by_run.get(run_id, {}).get(best_key), (int, float))
            }
            if candidates:
                best_run = max(candidates, key=lambda r: candidates[r])

        return {
            "metrics": comparison,
            "params": params_by_run,
            "statuses": statuses,
            "primary_metric": best_key,
            "best_run": best_run,
        }


class ModelRegistry:
    """Tracks model artifacts, versions, stages and active promotions."""

    def __init__(self, engine: Engine | None = None, settings: Settings | None = None) -> None:
        self.engine = engine or get_engine(settings)
        self.settings = settings or get_settings()
        self.artifact_dir = self.settings.resolved_artifact_dir / "models"

    def register(
        self,
        *,
        model_id: str,
        kind: str,
        symbol: str,
        interval: str,
        artifact_path: Path | str,
        feature_schema: Sequence[str],
        metrics: dict[str, Any],
        params: dict[str, Any] | None = None,
        target: str = "log_return_1",
        run_id: str | None = None,
        stage: str = "production",
        trained_from: datetime | None = None,
        trained_to: datetime | None = None,
        framework: str = "torch",
    ) -> ModelVersion:
        artifact_path = Path(artifact_path)
        with self.engine.session() as session:
            latest = session.execute(
                select(ModelVersion).where(ModelVersion.model_id == model_id)
                .order_by(ModelVersion.version.desc())
            ).scalars().first()
            version = (latest.version + 1) if latest else 1

            if stage == "production":
                session.execute(
                    update(ModelVersion)
                    .where(ModelVersion.model_id == model_id, ModelVersion.id != 0)
                    .values(is_active=False, stage="archived")
                )

            record = ModelVersion(
                model_id=model_id, version=version, kind=kind, symbol=symbol,
                interval=interval, framework=framework,
                artifact_path=str(artifact_path),
                feature_schema=list(feature_schema),
                target=target, params=params or {}, metrics=metrics,
                stage=stage, is_active=True, run_id=run_id,
                trained_from=trained_from, trained_to=trained_to,
            )
            session.add(record)
            session.flush()
            logger.info("model_registered", extra={
                "model_id": model_id, "version": version, "kind": kind,
                "symbol": symbol, "stage": stage,
            })
            return record

    def get_active(self, model_id: str) -> ModelVersion | None:
        with self.engine.session() as session:
            return session.execute(
                select(ModelVersion)
                .where(ModelVersion.model_id == model_id, ModelVersion.is_active.is_(True))
                .order_by(ModelVersion.version.desc())
            ).scalars().first()

    def record_metrics(
        self, model_id: str, metrics: dict[str, Any], *, merge: bool = True
    ) -> ModelVersion:
        """Attach metrics to the active version of a model.

        Needed for deferred evaluation: a model registered during screening
        carries only validation metrics, and the holdout numbers are computed
        later against the same immutable artifact. Re-recording the same
        artifact's holdout score is idempotent and leaks nothing, because the
        model and its split geometry are already fixed.
        """
        with self.engine.session() as session:
            record = session.execute(
                select(ModelVersion)
                .where(ModelVersion.model_id == model_id, ModelVersion.is_active.is_(True))
                .order_by(ModelVersion.version.desc())
            ).scalars().first()
            if record is None:
                raise ValueError(f"No active model found for id {model_id!r}")
            payload = dict(record.metrics or {}) if merge else {}
            payload.update(metrics)
            record.metrics = payload
            session.commit()
            session.refresh(record)
            return record

    def resolve(self, *, model_id: str | None = None, kind: str | None = None,
                symbol: str | None = None, interval: str | None = None) -> ModelVersion | None:
        """Find the best active model matching the supplied filters."""
        with self.engine.session() as session:
            stmt = select(ModelVersion).where(ModelVersion.is_active.is_(True))
            if model_id:
                stmt = stmt.where(ModelVersion.model_id == model_id)
            if kind:
                stmt = stmt.where(ModelVersion.kind == kind)
            if symbol:
                stmt = stmt.where(ModelVersion.symbol == symbol.upper())
            if interval:
                stmt = stmt.where(ModelVersion.interval == interval)
            stmt = stmt.order_by(ModelVersion.version.desc())
            return session.execute(stmt).scalars().first()

    def list_models(self, *, kind: str | None = None, symbol: str | None = None,
                    architecture: str | None = None,
                    limit: int = 100) -> list[ModelVersion]:
        with self.engine.session() as session:
            stmt = select(ModelVersion).order_by(
                ModelVersion.created_at.desc()
            ).limit(limit)
            if kind:
                stmt = stmt.where(ModelVersion.kind == kind)
            if symbol:
                stmt = stmt.where(ModelVersion.symbol == symbol.upper())
            if architecture:
                # `kind` is the *task* ("forecaster" for every DL model), so it
                # can never select between mlp/lstm/transformer. The architecture
                # lives inside the JSON params blob, and the filter is applied in
                # SQL rather than after the LIMIT so a filtered query still
                # returns up to `limit` genuine matches instead of however many
                # survived an unfiltered page.
                stmt = stmt.where(
                    ModelVersion.params.contains(
                        {"architecture": architecture.lower()}
                    )
                )
            return list(session.execute(stmt).scalars())

    def promote(self, model_id: str, stage: str = "production") -> bool:
        if stage not in STAGE_RANK:
            raise ValueError(f"Unknown stage {stage!r}; valid: {sorted(STAGE_RANK)}")
        with self.engine.session() as session:
            target = session.execute(
                select(ModelVersion)
                .where(ModelVersion.model_id == model_id)
                .order_by(ModelVersion.version.desc())
            ).scalars().first()
            if target is None:
                return False
            if stage == "production":
                session.execute(
                    update(ModelVersion)
                    .where(ModelVersion.model_id == model_id)
                    .values(is_active=False, stage="archived")
                )
            target.stage = stage
            target.is_active = stage == "production"
            return True

    def remove(self, model_id: str) -> bool:
        with self.engine.session() as session:
            record = session.execute(
                select(ModelVersion)
                .where(ModelVersion.model_id == model_id, ModelVersion.is_active.is_(True))
            ).scalars().first()
            if record is None:
                return False
            path = Path(record.artifact_path)
            if path.exists():
                try:
                    path.unlink()
                except OSError as exc:
                    logger.warning("model_artifact_delete_failed", extra={"error": str(exc)})
            session.delete(record)
            return True


def serialize_run(run: Run) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "kind": run.kind,
        "status": run.status,
        "params": run.params,
        "metrics": run.metrics,
        "message": run.message,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "duration_seconds": run.duration_seconds,
    }


def serialize_model(record: ModelVersion) -> dict[str, Any]:
    return {
        "model_id": record.model_id,
        "version": record.version,
        "kind": record.kind,
        "symbol": record.symbol,
        "interval": record.interval,
        "framework": record.framework,
        "artifact_path": record.artifact_path,
        "feature_count": len(record.feature_schema or []),
        "target": record.target,
        "params": record.params,
        "metrics": record.metrics,
        "stage": record.stage,
        "is_active": record.is_active,
        "trained_from": record.trained_from,
        "trained_to": record.trained_to,
        "run_id": record.run_id,
        "created_at": record.created_at,
    }


__all__ = [
    "Experiment",
    "ExperimentTracker",
    "ModelRegistry",
    "serialize_model",
    "serialize_run",
]