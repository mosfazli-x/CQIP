"""Structured logging setup shared by CLI, workers and API."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

from cqip.config import Settings, get_settings

_CONFIGURED = False
_console = Console(stderr=False)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload, default=str)


class ContextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extras = getattr(record, "extra_fields", {})
        base = super().format(record)
        if extras:
            rendered = " ".join(f"{k}={v}" for k, v in extras.items())
            return f"{base}  [{rendered}]"
        return base


def bind_fields(logger: logging.Logger, **fields: Any) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logger, {"extra_fields": fields})


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    settings = settings or get_settings()
    level = getattr(logging, settings.log_level, logging.INFO)

    handler: logging.Handler
    formatter: logging.Formatter
    if settings.log_format == "json":
        handler = logging.StreamHandler(sys.stderr)
        formatter = JsonFormatter()
    else:
        handler = RichHandler(
            console=_console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=False,
        )
        formatter = ContextFormatter("%(message)s")

    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    for noisy in ("httpx", "httpcore", "urllib3", "transformers", "filelock", "gymnasium"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)