"""Parquet-backed dataset store with Hive partitioning and safe upsert semantics."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from cqip.config import Settings, get_settings
from cqip.logging_config import get_logger

logger = get_logger(__name__)

NULL_INT64 = pa.field("ts", pa.timestamp("us", tz="UTC"))
DS_FORMAT = "parquet"


def _as_utc_timestamp(value: Any) -> pd.Timestamp:
    """Normalize a datetime-like value to a tz-aware UTC pandas Timestamp."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


class ParquetStore:
    """Columnar store organized as ``<root>/<dataset>/<partition columns>/``.

    All timestamps are normalized to UTC. Writes are atomic: data is staged to a
    temporary file then moved into place, so a crashed job never leaves a
    partially written partition behind.
    """

    def __init__(self, root: Path | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.root = Path(root) if root else self.settings.resolved_parquet_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def dataset_path(self, dataset: str) -> Path:
        return self.root / dataset

    def exists(self, dataset: str) -> bool:
        path = self.dataset_path(dataset)
        return path.exists() and any(path.rglob("*.parquet"))

    def _partition_values(self, row: dict[str, Any], partition_by: Sequence[str]) -> dict[str, str]:
        values: dict[str, str] = {}
        for column in partition_by:
            raw = row.get(column)
            if isinstance(raw, datetime):
                dt = raw if raw.tzinfo else raw.replace(tzinfo=UTC)
                values[column] = dt.strftime("%Y-%m-%d")
            elif raw is None:
                values[column] = "unknown"
            else:
                values[column] = str(raw)
        return values

    def write(
        self,
        dataset: str,
        frame: pd.DataFrame,
        *,
        partition_by: Sequence[str] = ("year", "month"),
        dedupe_on: Sequence[str] | None = None,
        mode: str = "upsert",
    ) -> int:
        """Persist ``frame``, merging with existing data.

        ``upsert`` unions on ``dedupe_on`` keys keeping the incoming row,
        ``overwrite`` replaces the entire dataset, ``append`` skips merging.
        Returns the number of rows from ``frame`` that reached the dataset --
        i.e. genuinely new keys, not the untouched rows carried over from the
        existing data.
        """
        if frame.empty:
            return 0

        target = self.dataset_path(dataset)
        target.mkdir(parents=True, exist_ok=True)

        payload = frame.copy()
        if "ts" in payload.columns:
            payload["ts"] = pd.to_datetime(payload["ts"], utc=True)

        written = len(frame)
        if mode == "upsert":
            existing = self.read(dataset)
            if not existing.empty and dedupe_on:
                keys = [k for k in dedupe_on if k in payload.columns and k in existing.columns]
                if keys:
                    # Count how many incoming keys are absent from the stored data.
                    # Counting the merged frame instead would report the whole
                    # dataset as "inserted" on every incremental write.
                    seen = {tuple(row) for row in existing[keys].itertuples(index=False, name=None)}
                    written = 0
                    for row in payload[keys].itertuples(index=False, name=None):
                        if row not in seen:
                            seen.add(row)
                            written += 1

                    combined = pd.concat([existing, payload], ignore_index=True)
                    payload = combined.drop_duplicates(subset=keys, keep="last")
                else:
                    payload = pd.concat([existing, payload], ignore_index=True)

        # Every mode below rewrites the *whole* payload, so files left over from
        # a previous partition layout would linger and be read back as duplicate
        # rows. Clearing first makes writes idempotent and self-healing.
        for parquet in target.rglob("*.parquet"):
            parquet.unlink()

        # Partition columns must be derived *after* the merge. `read` strips
        # year/month, so rows carried over from the existing dataset would be
        # concatenated with nulls and land in __HIVE_DEFAULT_PARTITION__,
        # quietly reshaping the partition layout on every incremental write.
        if "ts" in payload.columns:
            payload = payload.dropna(subset=["ts"]).reset_index(drop=True)

        for column in partition_by:
            if column == "year":
                payload["year"] = payload["ts"].dt.year.astype("int16")
            elif column == "month":
                payload["month"] = payload["ts"].dt.month.astype("int8")
            elif column not in payload.columns:
                raise KeyError(f"Partition column {column!r} absent from frame")

        if payload.empty:
            logger.info("parquet_write_skipped", extra={"dataset": dataset, "mode": mode})
            return 0

        table = pa.Table.from_pandas(payload, preserve_index=False)

        tmp_dir = Path(tempfile.mkdtemp(prefix="cqip_write_", dir=self.root))
        try:
            ds.write_dataset(
                table,
                base_dir=str(target),
                format=DS_FORMAT,
                partitioning=partition_by if partition_by else None,
                partitioning_flavor="hive",
                existing_data_behavior="overwrite_or_ignore",
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        logger.info(
            "parquet_write",
            extra={"dataset": dataset, "rows_written": written, "rows_total": len(payload), "mode": mode},
        )
        return written

    def count(self, dataset: str) -> int:
        """Number of stored rows, used to measure net growth across a write."""
        frame = self.read(dataset)
        return int(len(frame))

    def read(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        filters: Sequence[tuple[str, str, Any]] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        symbols: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Read a dataset with optional column selection, filters and time range."""
        path = self.dataset_path(dataset)
        if not path.exists():
            return pd.DataFrame()

        parquet_files = sorted(path.rglob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()

        available = self.columns(dataset)
        select = list(columns) if columns else None
        if select:
            select = [c for c in select if c in available]

        predicate_parts: list[ds.Expression] = []
        if filters:
            predicate_parts.extend(
                ds.field(column) == value for column, _op, value in filters
            )
        if start is not None:
            predicate_parts.append(ds.field("ts") >= _as_utc_timestamp(start))
        if end is not None:
            predicate_parts.append(ds.field("ts") < _as_utc_timestamp(end))
        if symbols and "symbol" in available:
            predicate_parts.append(ds.field("symbol").isin(list(symbols)))

        predicate = None
        if predicate_parts:
            predicate = predicate_parts[0]
            for part in predicate_parts[1:]:
                predicate = predicate & part

        dataset = ds.dataset(str(path), format=DS_FORMAT, partitioning="hive")
        table = dataset.to_table(columns=select, filter=predicate)
        frame = table.to_pandas()
        if frame.empty:
            return frame

        if "ts" in frame.columns and not pd.api.types.is_datetime64tz_dtype(frame["ts"]):
            frame["ts"] = pd.to_datetime(frame["ts"], utc=True)

        if "year" in frame.columns and not (select and "year" in select):
            frame = frame.drop(columns=["year", "month"], errors="ignore")

        if "ts" in frame.columns:
            frame = frame.sort_values("ts").reset_index(drop=True)
        if limit:
            frame = frame.head(limit).reset_index(drop=True)
        return frame

    def columns(self, dataset: str) -> list[str]:
        path = self.dataset_path(dataset)
        if not path.exists():
            return []
        parquet_files = sorted(path.rglob("*.parquet"))
        if not parquet_files:
            return []
        schema = pq.read_schema(parquet_files[0])
        return list(schema.names)

    def row_count(self, dataset: str) -> int:
        frame = self.read(dataset, columns=["ts"])
        return 0 if frame.empty else len(frame)

    def date_range(self, dataset: str) -> tuple[datetime | None, datetime | None]:
        frame = self.read(dataset, columns=["ts"])
        if frame.empty:
            return None, None
        return frame["ts"].min().to_pydatetime(), frame["ts"].max().to_pydatetime()

    def symbols(self, dataset: str) -> list[str]:
        frame = self.read(dataset, columns=["symbol"])
        if frame.empty or "symbol" not in frame.columns:
            return []
        return sorted(frame["symbol"].dropna().unique().tolist())

    def append_frame(self, dataset: str, frame: pd.DataFrame) -> int:
        return self.write(dataset, frame, mode="append")

    def merge_frames(self, *frames: pd.DataFrame) -> pd.DataFrame:
        valid = [f for f in frames if f is not None and not f.empty]
        if not valid:
            return pd.DataFrame()
        return pd.concat(valid, ignore_index=True).sort_values("ts").reset_index(drop=True)


def rows_to_frame(rows: Iterable[Any], columns: Sequence[str] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if columns:
        for column in columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        frame = frame[list(columns)]
    if "ts" in frame.columns and not frame.empty:
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame