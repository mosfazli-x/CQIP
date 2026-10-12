"""Async market data providers.

Every provider implements :class:`MarketDataProvider` returning
:class:`~cqip.models.Candle` objects. Providers are registered by name so the
ingestion orchestrator and API can resolve them dynamically.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import httpx

from cqip.config import Settings, get_settings
from cqip.logging_config import get_logger
from cqip.models import Candle, Exchange, Interval
from cqip.utils import (
    RateLimiter,
    RetryPolicy,
    floor_to_interval,
    get_json,
    normalize_symbol,
    split_symbol,
    utc_now,
)

logger = get_logger(__name__)


class MarketDataProvider(ABC):
    name: str = "base"
    interval_map: dict[str, str] = {}

    def __init__(self, settings: Settings | None = None, rate_limit: float = 5.0) -> None:
        self.settings = settings or get_settings()
        self.client: httpx.AsyncClient | None = None
        self.limiter = RateLimiter(rate_limit)
        self.policy = RetryPolicy(
            max_attempts=self.settings.http_max_retries,
            backoff_base=self.settings.http_backoff_base,
            backoff_max=self.settings.http_backoff_max,
        )

    async def __aenter__(self) -> MarketDataProvider:
        from cqip.utils import create_http_client

        self.client = create_http_client(self.settings)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self.client:
            await self.client.aclose()
            self.client = None

    def _require_client(self) -> httpx.AsyncClient:
        if self.client is None:
            from cqip.utils import create_http_client

            self.client = create_http_client(self.settings)
        return self.client

    def map_interval(self, interval: str) -> str:
        return self.interval_map.get(interval, interval)

    @abstractmethod
    async def fetch_candles(
        self, symbol: str, interval: str, *, start: datetime | None = None,
        end: datetime | None = None, limit: int = 1000,
    ) -> list[Candle]: ...

    @abstractmethod
    async def list_symbols(self) -> list[str]: ...

    async def fetch_candles_safe(self, *args: Any, **kwargs: Any) -> list[Candle]:
        try:
            return await self.fetch_candles(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("provider_fetch_failed", extra={"provider": self.name, "error": str(exc)[:200]})
            return []


class GateProvider(MarketDataProvider):
    name = Exchange.GATE
    base_url = "https://api.gateio.ws/api/v4"
    interval_map = {
        "10s": "10s", "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "4h": "4h", "8h": "8h", "1d": "1d", "7d": "7d",
    }

    async def fetch_candles(
        self, symbol: str, interval: str, *, start: datetime | None = None,
        end: datetime | None = None, limit: int = 1000,
    ) -> list[Candle]:
        pair = normalize_symbol(symbol)
        mapped = self.map_interval(interval)
        client = self._require_client()
        collected: dict[int, Candle] = {}
        cursor = int(end.timestamp()) if end else int(utc_now().timestamp())

        for _ in range(60):
            payload = await get_json(
                client,
                f"{self.base_url}/spot/candlesticks",
                params={
                    "currency_pair": pair,
                    "interval": mapped,
                    "limit": min(limit, 1000),
                    "to": cursor,
                },
                policy=self.policy,
                limiter=self.limiter,
            )
            if not payload:
                break
            rows = list(payload)
            for row in rows:
                # Gate v4 array order is
                # [timestamp, quote_volume, close, high, low, open, base_volume, is_closed].
                # It is NOT OHLC-ordered, so index positions must be read literally.
                if len(row) < 7:
                    continue
                # Partial bars are still forming; storing them bakes intrabar noise
                # into downstream features and targets.
                if len(row) > 7 and str(row[7]).lower() not in {"true", "1"}:
                    continue
                ts = datetime.fromtimestamp(int(row[0]), tz=utc_now().tzinfo)
                candle = Candle(
                    ts=ts,
                    open=float(row[5]), high=float(row[3]), low=float(row[4]),
                    close=float(row[2]), volume=float(row[6]),
                    quote_volume=float(row[1]),
                )
                if candle.high < candle.low or candle.low <= 0:
                    logger.warning("gate_candle_rejected", extra={
                        "symbol": pair, "ts": ts.isoformat(),
                        "open": candle.open, "high": candle.high, "low": candle.low,
                    })
                    continue
                collected[int(ts.timestamp())] = candle
            oldest = min(int(row[0]) for row in rows)
            if start is not None and datetime.fromtimestamp(oldest, tz=utc_now().tzinfo) <= start:
                break
            if oldest >= cursor:
                break
            cursor = oldest - 1
            await asyncio.sleep(0.12)

        candles = [candle for ts, candle in sorted(collected.items())]
        if start:
            cutoff = start.timestamp()
            candles = [c for c in candles if c.ts.timestamp() >= cutoff]
        return candles

    async def list_symbols(self) -> list[str]:
        client = self._require_client()
        payload = await get_json(
            client, f"{self.base_url}/spot/currency_pairs", policy=self.policy, limiter=self.limiter
        )
        return [
            row["id"]
            for row in payload
            if row.get("trade_status") == "tradable" and str(row.get("quote", "")).upper() == "USDT"
        ]

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        client = self._require_client()
        payload = await get_json(
            client, f"{self.base_url}/spot/tickers",
            params={"currency_pair": normalize_symbol(symbol)},
            policy=self.policy, limiter=self.limiter,
        )
        return payload[0] if payload else {}


class HTXProvider(MarketDataProvider):
    name = Exchange.HTX
    base_url = "https://api.huobi.pro"
    interval_map = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
                    "1h": "60min", "4h": "4hour", "1d": "1day", "1w": "1week"}
    _INTERVAL_LABELS = {"1min", "5min", "15min", "30min", "60min", "4hour", "1day", "1week"}

    async def fetch_candles(
        self, symbol: str, interval: str, *, start: datetime | None = None,
        end: datetime | None = None, limit: int = 2000,
    ) -> list[Candle]:
        base, quote = split_symbol(symbol)
        period = self.map_interval(interval)
        client = self._require_client()
        collected: dict[int, Candle] = {}
        cursor = int((end or utc_now()).timestamp())

        for _ in range(40):
            params: dict[str, Any] = {
                "symbol": f"{base.lower()}{quote.lower()}",
                "period": period,
                "size": min(limit, 2000),
            }
            if start is not None:
                params["to"] = cursor
            payload = await get_json(
                client, f"{self.base_url}/market/history/kline",
                params=params, policy=self.policy, limiter=self.limiter,
            )
            rows = payload.get("data") or []
            if not rows:
                break
            oldest = None
            for row in rows:
                ts = datetime.fromtimestamp(int(row["id"]), tz=utc_now().tzinfo)
                collected[int(ts.timestamp())] = Candle(
                    ts=ts, open=float(row["open"]), high=float(row["high"]),
                    low=float(row["low"]), close=float(row["close"]),
                    volume=float(row.get("amount", 0.0)),
                    quote_volume=float(row.get("vol", 0.0)),
                    trades=int(row["count"]) if "count" in row else None,
                )
                oldest = ts.timestamp() if oldest is None else min(oldest, ts.timestamp())
            if start is None or oldest is None or oldest <= start.timestamp():
                break
            if oldest >= cursor:
                break
            cursor = int(oldest) - 1
            await asyncio.sleep(0.12)

        candles = [c for _, c in sorted(collected.items())]
        if start:
            cutoff = start.timestamp()
            candles = [c for c in candles if c.ts.timestamp() >= cutoff]
        return candles

    async def list_symbols(self) -> list[str]:
        client = self._require_client()
        payload = await get_json(
            client, f"{self.base_url}/v1/common/symbols", policy=self.policy, limiter=self.limiter
        )
        symbols = []
        for row in payload.get("data", []):
            if row.get("state") != "online":
                continue
            if str(row.get("quote-currency", "")).upper() != "USDT":
                continue
            symbols.append(f"{str(row['base-currency']).upper()}_{str(row['quote-currency']).upper()}")
        return symbols


class BitgetProvider(MarketDataProvider):
    name = Exchange.BITGET
    base_url = "https://api.bitget.com/api/v2/spot/market"
    interval_map = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
                    "1h": "1h", "4h": "4h", "1d": "1day", "1w": "1w"}

    async def fetch_candles(
        self, symbol: str, interval: str, *, start: datetime | None = None,
        end: datetime | None = None, limit: int = 300,
    ) -> list[Candle]:
        base, quote = split_symbol(symbol)
        granularity = self.map_interval(interval)
        client = self._require_client()
        collected: dict[int, Candle] = {}
        end_ms = int((end or utc_now()).timestamp() * 1000)

        for _ in range(40):
            params: dict[str, Any] = {
                "symbol": f"{base}{quote}",
                "granularity": granularity,
                "limit": 1000,
                "endTime": end_ms,
            }
            payload = await get_json(
                client, f"{self.base_url}/candles",
                params=params, policy=self.policy, limiter=self.limiter,
            )
            rows = payload.get("data") or []
            if not rows:
                break
            oldest = None
            for row in rows:
                ts = datetime.fromtimestamp(int(row[0]) / 1000, tz=utc_now().tzinfo)
                collected[int(ts.timestamp())] = Candle(
                    ts=ts, open=float(row[1]), high=float(row[2]), low=float(row[3]),
                    close=float(row[4]), volume=float(row[5]), quote_volume=float(row[6]),
                )
                oldest = ts.timestamp() if oldest is None else min(oldest, ts.timestamp())
            if start is None or oldest is None or oldest <= start.timestamp():
                break
            if oldest >= end_ms / 1000:
                break
            end_ms = int(oldest) * 1000 - 1
            await asyncio.sleep(0.12)

        candles = [c for _, c in sorted(collected.items())]
        if start:
            cutoff = start.timestamp()
            candles = [c for c in candles if c.ts.timestamp() >= cutoff]
        return candles

    async def list_symbols(self) -> list[str]:
        client = self._require_client()
        symbols: list[str] = []
        for offset in range(0, 2000, 100):
            payload = await get_json(
                client, "https://api.bitget.com/api/v2/spot/public/symbols",
                params={"offset": offset, "limit": 100},
                policy=self.policy, limiter=self.limiter,
            )
            rows = payload.get("data") or []
            if not rows:
                break
            for row in rows:
                if row.get("status") not in {"online", "normal"}:
                    continue
                if str(row.get("quoteCoin", "")).upper() != "USDT":
                    continue
                symbols.append(f"{str(row['baseCoin']).upper()}_{str(row['quoteCoin']).upper()}")
        return symbols


class BinanceProvider(MarketDataProvider):
    name = Exchange.BINANCE
    base_url = "https://api.binance.com/api/v3"
    interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                    "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"}

    async def fetch_candles(
        self, symbol: str, interval: str, *, start: datetime | None = None,
        end: datetime | None = None, limit: int = 1000,
    ) -> list[Candle]:
        base, quote = split_symbol(symbol)
        client = self._require_client()
        collected: dict[int, Candle] = {}
        cursor = int((start or utc_now() - timedelta(days=365)).timestamp() * 1000)
        stop = int((end or utc_now()).timestamp() * 1000)

        while cursor < stop:
            payload = await get_json(
                client, f"{self.base_url}/klines",
                params={
                    "symbol": f"{base}{quote}",
                    "interval": self.map_interval(interval),
                    "startTime": cursor,
                    "limit": 1000,
                },
                policy=self.policy, limiter=self.limiter,
            )
            if not payload:
                break
            for row in payload:
                ts = datetime.fromtimestamp(int(row[0]) / 1000, tz=utc_now().tzinfo)
                collected[int(ts.timestamp())] = Candle(
                    ts=ts, open=float(row[1]), high=float(row[2]), low=float(row[3]),
                    close=float(row[4]), volume=float(row[5]), quote_volume=float(row[7]),
                    trades=int(row[8]),
                )
            last = int(payload[-1][0])
            if last <= cursor:
                break
            cursor = last + 1
            await asyncio.sleep(0.12)

        candles = [c for _, c in sorted(collected.items())]
        if start:
            cutoff = start.timestamp()
            candles = [c for c in candles if c.ts.timestamp() >= cutoff]
        return candles

    async def list_symbols(self) -> list[str]:
        client = self._require_client()
        payload = await get_json(
            client, f"{self.base_url}/exchangeInfo", policy=self.policy, limiter=self.limiter
        )
        return [
            f"{row['baseAsset']}_{row['quoteAsset']}"
            for row in payload.get("symbols", [])
            if row.get("status") == "TRADING" and row.get("quoteAsset") == "USDT"
        ]


class CoinGeckoProvider(MarketDataProvider):
    name = Exchange.COINGECKO
    base_url = "https://api.coingecko.com/api/v3"
    interval_map = {"1d": "daily", "1h": "hourly"}

    async def fetch_candles(
        self, symbol: str, interval: str, *, start: datetime | None = None,
        end: datetime | None = None, limit: int = 1000,
    ) -> list[Candle]:
        base, _quote = split_symbol(symbol)
        coin_id = COINGECKO_IDS.get(base.lower(), base.lower())
        client = self._require_client()
        days = self._resolve_days(start, end)
        if days is None:
            return []

        granularity = "hourly" if interval in {"1h", "4h"} else "daily"
        payload = await get_json(
            client, f"{self.base_url}/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": days, "interval": granularity},
            policy=self.policy, limiter=self.limiter,
        )
        if not isinstance(payload, dict):
            return []

        prices = payload.get("prices") or []
        volumes = payload.get("total_volumes") or []
        volume_by_ts = {int(v[0]): float(v[1]) for v in volumes}

        candles: dict[int, Candle] = {}
        for point in prices:
            ts_ms, price = int(point[0]), float(point[1])
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=utc_now().tzinfo)
            bucket = int(floor_to_interval(ts, "1d" if granularity == "daily" else "1h").timestamp())
            entry = candles.get(bucket)
            if entry is None:
                candles[bucket] = Candle(
                    ts=datetime.fromtimestamp(bucket, tz=utc_now().tzinfo),
                    open=price, high=price, low=price, close=price,
                    volume=volume_by_ts.get(ts_ms, 0.0),
                )
            else:
                candles[bucket] = Candle(
                    ts=entry.ts, open=entry.open, high=max(entry.high, price),
                    low=min(entry.low, price), close=price,
                    volume=entry.volume + volume_by_ts.get(ts_ms, 0.0),
                )

        result = [c for _, c in sorted(candles.items())]
        if start:
            cutoff = start.timestamp()
            result = [c for c in result if c.ts.timestamp() >= cutoff]
        return result

    def _resolve_days(self, start: datetime | None, end: datetime | None) -> str | None:
        if start is None:
            return "365"
        span_days = ((end or utc_now()) - start).total_seconds() / 86400
        if span_days <= 1:
            return "1"
        if span_days <= 90:
            return "90"
        if span_days <= 365:
            return "365"
        return None

    async def list_symbols(self) -> list[str]:
        return [f"{coin.upper()}_USDT" for coin in COINGECKO_IDS]


COINGECKO_IDS: dict[str, str] = {
    "btc": "bitcoin", "eth": "ethereum", "sol": "solana", "bnb": "binancecoin",
    "xrp": "ripple", "ada": "cardano", "doge": "dogecoin", "avax": "avalanche-2",
    "dot": "polkadot", "link": "chainlink", "matic": "matic-network", "ltc": "litecoin",
    "trx": "tron", "atom": "cosmos", "uni": "uniswap", "xlm": "stellar",
    "near": "near", "apt": "aptos", "arb": "arbitrum", "op": "optimism",
    "fil": "filecoin", "algo": "algorand", "hbar": "hedera-hashgraph",
    "inj": "injective-protocol", "sui": "sui", "sei": "sei-network", "render": "render-token",
}


_PROVIDERS: dict[str, type[MarketDataProvider]] = {
    Exchange.GATE: GateProvider,
    Exchange.HTX: HTXProvider,
    Exchange.BITGET: BitgetProvider,
    Exchange.BINANCE: BinanceProvider,
    Exchange.COINGECKO: CoinGeckoProvider,
}


def get_provider(name: str, settings: Settings | None = None, **kwargs: Any) -> MarketDataProvider:
    key = name.lower()
    if key not in _PROVIDERS:
        raise KeyError(f"Unknown provider {name!r}. Available: {sorted(_PROVIDERS)}")
    return _PROVIDERS[key](settings=settings, **kwargs)


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


async def fetch_cross_exchange(
    symbols: Sequence[str],
    interval: str,
    providers: Sequence[str],
    *,
    settings: Settings | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    concurrency: int = 3,
) -> dict[tuple[str, str], list[Candle]]:
    """Fetch candles for many symbol/provider pairs, isolating failures."""
    from cqip.utils import gather_with_concurrency

    settings = settings or get_settings()
    jobs: list[tuple[tuple[str, str], Any]] = []
    results: dict[tuple[str, str], list[Candle]] = {}

    async def job(provider_name: str, symbol: str) -> list[Candle]:
        async with get_provider(provider_name, settings) as provider:
            return await provider.fetch_candles(symbol, interval, start=start, end=end)

    for provider_name in providers:
        for symbol in symbols:
            jobs.append(((provider_name, symbol), lambda p=provider_name, s=symbol: job(p, s)))

    outcomes = await gather_with_concurrency([f for _, f in jobs], concurrency)
    for (key, _), outcome in zip(jobs, outcomes, strict=True):
        if isinstance(outcome, Exception):
            logger.warning("cross_exchange_failed", extra={"key": list(key), "error": str(outcome)[:200]})
            results[key] = []
        else:
            results[key] = outcome
    return results


__all__ = [
    "BitgetProvider",
    "BinanceProvider",
    "CoinGeckoProvider",
    "GateProvider",
    "HTXProvider",
    "Interval",
    "MarketDataProvider",
    "available_providers",
    "fetch_cross_exchange",
    "get_provider",
]