"""Shared helpers: time handling, rate limiting, retries and HTTP client construction."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

import httpx

from cqip.config import Settings
from cqip.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
}

_MS_INTERVALS = {"1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m",
                 "60min": "1h", "4hour": "4h", "1day": "1d", "1week": "1w"}


def utc_now() -> datetime:
    return datetime.now(UTC)


def to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Unparseable datetime: {value!r}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def floor_to_interval(dt: datetime, interval: str) -> datetime:
    seconds = INTERVAL_SECONDS.get(interval)
    if seconds is None:
        raise ValueError(f"Unsupported interval: {interval}")
    epoch = int(dt.timestamp())
    return datetime.fromtimestamp(epoch - (epoch % seconds), tz=UTC)


def normalize_symbol(symbol: str, separator: str = "_") -> str:
    cleaned = symbol.replace("/", separator).replace("-", separator).replace(":", separator)
    cleaned = cleaned.upper().strip()
    return cleaned.replace(separator * 2, separator)


def split_symbol(symbol: str) -> tuple[str, str]:
    for sep in ("_", "/", "-"):
        if sep in symbol:
            base, quote = symbol.split(sep, 1)
            return base.upper(), quote.upper()
    return symbol.upper(), "USDT"


class RateLimiter:
    """Async token bucket limiting request rate while allowing short bursts."""

    def __init__(self, rate_per_second: float, burst: int | None = None) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be positive")
        self.rate = float(rate_per_second)
        self.capacity = float(burst if burst is not None else max(1.0, rate_per_second))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, amount: float = 1.0) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._updated
                self._updated = now
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                deficit = amount - self._tokens
                wait = deficit / self.rate
            await asyncio.sleep(min(wait, 5.0))


class RetryPolicy:
    def __init__(
        self,
        max_attempts: int = 4,
        backoff_base: float = 0.8,
        backoff_max: float = 30.0,
        jitter: float = 0.3,
    ) -> None:
        self.max_attempts = max(1, max_attempts)
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.jitter = jitter

    def delay(self, attempt: int) -> float:
        raw = min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_max)
        spread = raw * self.jitter
        return raw - spread + random.random() * 2 * spread


class HttpError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def create_http_client(settings: Settings) -> httpx.AsyncClient:
    limits = httpx.Limits(
        max_connections=settings.http_max_connections,
        max_keepalive_connections=settings.http_max_connections,
    )
    timeout = httpx.Timeout(settings.http_timeout, connect=min(settings.http_timeout, 10.0))
    return httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        follow_redirects=True,
        headers={
            "User-Agent": settings.user_agent,
            "Accept": "application/json, text/xml, */*",
            "Accept-Language": "en-US,en;q=0.9",
        },
        http2=False,
    )


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    policy: RetryPolicy | None = None,
    limiter: RateLimiter | None = None,
    expect_json: bool = True,
) -> Any:
    policy = policy or RetryPolicy()
    last_error: Exception | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            if limiter:
                await limiter.acquire()
            response = await client.get(url, params=params)
            if response.status_code == 429 or response.status_code >= 500:
                raise HttpError(
                    f"{url} -> HTTP {response.status_code}",
                    status_code=response.status_code,
                )
            response.raise_for_status()
            if not expect_json:
                return response.text
            if not response.content:
                return None
            return response.json()
        except (httpx.TransportError, HttpError, ValueError) as exc:
            last_error = exc
            if attempt == policy.max_attempts:
                break
            wait = policy.delay(attempt)
            logger.warning(
                "http_retry",
                extra={"url": url, "attempt": attempt, "wait": round(wait, 2),
                       "error": str(exc)[:200]},
            )
            await asyncio.sleep(wait)

    raise HttpError(f"GET {url} failed after {policy.max_attempts} attempts: {last_error}")


async def get_text(
    client: httpx.AsyncClient,
    url: str,
    *,
    policy: RetryPolicy | None = None,
    limiter: RateLimiter | None = None,
) -> str:
    return await get_json(client, url, policy=policy, limiter=limiter, expect_json=False)


async def gather_with_concurrency(
    factories: Sequence[Callable[[], Awaitable[T]]],
    concurrency: int = 4,
    return_exceptions: bool = True,
) -> list[Any]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run(factory: Callable[[], Awaitable[T]]) -> Any:
        async with semaphore:
            try:
                return await factory()
            except Exception as exc:  # noqa: BLE001
                if return_exceptions:
                    return exc
                raise

    return list(await asyncio.gather(*(run(f) for f in factories)))


def chunked(items: Iterable[T], size: int) -> Iterable[list[T]]:
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def humanize_duration(seconds: float) -> str:
    delta = timedelta(seconds=int(seconds))
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)