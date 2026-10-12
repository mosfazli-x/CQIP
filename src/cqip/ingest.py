"""Ingestion orchestration: fetch -> normalize -> persist to Parquet + coverage."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from cqip.config import Settings, get_settings
from cqip.db import Engine, finish_run, get_engine, record_run, upsert_coverage
from cqip.logging_config import get_logger
from cqip.models import Candle, IngestResult, NewsArticle, OnChainSnapshot
from cqip.news_providers import (
    CoinGeckoGlobalProvider,
    MempoolProvider,
    SentimentIndexProvider,
    fetch_all_news,
)
from cqip.providers import fetch_cross_exchange
from cqip.sentiment import SentimentScorer
from cqip.storage import ParquetStore
from cqip.universe import DEFAULT_INTERVAL_DEPTH_DAYS
from cqip.utils import gather_with_concurrency, normalize_symbol, utc_now

logger = get_logger(__name__)

CANDLES_DATASET = "ohlcv"
NEWS_DATASET = "news"
FEAR_GREED_DATASET = "fear_greed"
ONCHAIN_DATASET = "onchain"


def candles_to_frame(
    candles: Sequence[Candle], symbol: str, exchange: str, interval: str
) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=["ts", "symbol", "exchange", "interval", "open", "high",
                                     "low", "close", "volume", "quote_volume", "trades"])
    frame = pd.DataFrame([c.as_row() for c in candles])
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame["symbol"] = normalize_symbol(symbol)
    frame["exchange"] = exchange
    frame["interval"] = interval
    frame = frame.sort_values("ts").drop_duplicates(subset=["ts", "symbol", "exchange"], keep="last")
    return frame.reset_index(drop=True)


def articles_to_frame(articles: Sequence[NewsArticle]) -> pd.DataFrame:
    if not articles:
        return pd.DataFrame()
    frame = pd.DataFrame([
        {
            "ts": article.published_at,
            "uid": article.uid,
            "source": str(article.source),
            "title": article.title,
            "url": article.url,
            "summary": article.summary,
            "tags": ",".join(article.tags),
            "category": str(article.category),
            "content_hash": article.content_hash,
        }
        for article in articles
    ])
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame.drop_duplicates(subset=["uid"], keep="last").sort_values("ts").reset_index(drop=True)


def snapshots_to_frame(snapshots: Sequence[OnChainSnapshot]) -> pd.DataFrame:
    if not snapshots:
        return pd.DataFrame()
    frame = pd.DataFrame([
        {"ts": s.ts, "asset": s.asset, "metric": s.metric, "value": s.value} for s in snapshots
    ])
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame.sort_values("ts").drop_duplicates(subset=["ts", "asset", "metric"], keep="last").reset_index(drop=True)


class Ingestor:
    """Coordinates ingestion across providers into the Parquet store."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: ParquetStore | None = None,
        engine: Engine | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or ParquetStore(settings=self.settings)
        self.engine = engine or get_engine(self.settings)
        self.sentiment = SentimentScorer(self.settings)

    async def ingest_candles(
        self,
        symbols: Sequence[str],
        interval: str = "1h",
        providers: Sequence[str] | None = None,
        *,
        lookback_days: int = 730,
        start: datetime | None = None,
        end: datetime | None = None,
        concurrency: int = 3,
        run_id: str | None = None,
    ) -> list[IngestResult]:
        run_id = run_id or f"ingest_candles_{uuid.uuid4().hex[:12]}"
        provider_names = list(providers or [self.settings.default_exchange])
        start = start or (utc_now() - timedelta(days=lookback_days))
        end = end or utc_now()
        # Timed from before the fetch, not from result construction. The network
        # round-trip dominates; timing only the write reports a duration that
        # looks instant no matter how slow the fetch actually was.
        fetch_started = utc_now()

        record_run(self.engine, run_id, "ingest_candles", {
            "symbols": list(symbols), "interval": interval,
            "providers": provider_names, "start": start.isoformat(), "end": end.isoformat(),
        })

        logger.info("ingest_candles_start", extra={
            "run_id": run_id, "symbols": list(symbols),
            "interval": interval, "providers": provider_names,
        })

        fetched = await fetch_cross_exchange(
            symbols, interval, provider_names,
            settings=self.settings, start=start, end=end, concurrency=concurrency,
        )

        # `store.write` rewrites the whole dataset on every call, so writing once
        # per symbol would make a multi-symbol backfill quadratic in dataset
        # size. Accumulate everything, then merge in a single pass.
        pending: list[pd.DataFrame] = []
        results: list[IngestResult] = []
        for (provider, symbol), candles in sorted(fetched.items()):
            result = IngestResult(
                provider=provider, dataset=CANDLES_DATASET, symbol=symbol,
                started_at=fetch_started,
            )
            try:
                frame = candles_to_frame(candles, symbol, provider, interval)
                result.rows = len(frame)
                if frame.empty:
                    result.errors.append("no rows returned")
                    results.append(result)
                    continue
                pending.append(frame)
            except Exception as exc:  # noqa: BLE001
                result.errors.append(str(exc)[:300])
                logger.error("ingest_candles_failed", extra={
                    "provider": provider, "symbol": symbol, "error": str(exc)[:300],
                })
            result.finished_at = utc_now()
            results.append(result)

        if pending:
            combined = pd.concat(pending, ignore_index=True)
            try:
                # `interval` is part of the key. Without it a 4h bar sharing a
                # timestamp with a 1h bar would be treated as a duplicate and
                # silently drop one of them from the dataset.
                # `write` already reports how many keys were genuinely new, so
                # counting the dataset before and after would read every
                # partition of a multi-million-row store twice for nothing.
                added = max(0, self.store.write(
                    CANDLES_DATASET, combined,
                    partition_by=("year", "month"),
                    dedupe_on=("ts", "symbol", "exchange", "interval"),
                    mode="upsert",
                ))
                # Split the net growth across the incoming frames in proportion
                # to their size. Reporting `len(frame)` would claim every row
                # was new on a re-run that only deduplicated, which hides how
                # much of a backfill was genuinely new.
                for frame in pending:
                    result = next(
                        (r for r in results if normalize_symbol(r.symbol or "") == frame["symbol"].iloc[0]),
                        None,
                    )
                    if result is None:
                        continue
                    result.inserted = int(round(added * len(frame) / len(combined)))
                    upsert_coverage(
                        self.engine, CANDLES_DATASET, frame["symbol"].iloc[0], interval,
                        frame["exchange"].iloc[0],
                        frame["ts"].min().to_pydatetime(),
                        frame["ts"].max().to_pydatetime(), len(frame),
                    )
            except Exception as exc:  # noqa: BLE001
                for result in results:
                    if not result.errors:
                        result.errors.append(f"write failed: {str(exc)[:200]}")
                logger.error("ingest_candles_write_failed", extra={"error": str(exc)[:300]})

        # A run where every provider failed is not a success. Recording it as
        # "succeeded" would make a total outage look like a clean ingest.
        status = "failed" if results and all(r.errors for r in results) else "succeeded"
        finish_run(self.engine, run_id, status, {"results": [r.as_dict() for r in results]})
        return results

    async def backfill_universe(
        self,
        symbols: Sequence[str],
        *,
        intervals: Sequence[str] = ("1d", "1h"),
        providers: Sequence[str] | None = None,
        concurrency: int = 3,
        on_progress=None,
    ) -> dict[str, Any]:
        """Ingest every (symbol, interval) pair at the depth the exchange allows.

        Each interval carries its own retention limit, so asking for a single
        lookback across several timeframes would either truncate daily history
        or hammer the API for hourly bars that do not exist.
        """
        summary: dict[str, Any] = {
            "results": [], "errors": [], "intervals": list(intervals),
            "total_rows": 0, "total_inserted": 0,
        }
        # Iterate intervals, not symbols. `store.write` rewrites the entire
        # dataset on every call, so one call per (symbol, interval) would make
        # a universe-wide backfill quadratic in dataset size. `ingest_candles`
        # already merges every symbol it is given into a single write, so
        # batching by interval keeps the number of rewrites at one per timeframe.
        planned = [i for i in intervals if i in DEFAULT_INTERVAL_DEPTH_DAYS]
        skipped_intervals = [i for i in intervals if i not in DEFAULT_INTERVAL_DEPTH_DAYS]
        summary["skipped_intervals"] = skipped_intervals
        for index, interval in enumerate(planned, start=1):
            depth = DEFAULT_INTERVAL_DEPTH_DAYS[interval]
            if on_progress is not None:
                on_progress(index, len(planned),
                            {"interval": interval, "symbols": len(symbols),
                             "lookback_days": depth})
            results = await self.ingest_candles(
                list(symbols), interval, providers,
                lookback_days=depth, concurrency=concurrency,
            )
            for result in results:
                summary["results"].append(result.as_dict())
                summary["total_rows"] += result.rows
                summary["total_inserted"] += result.inserted
                if result.errors:
                    summary["errors"].append(
                        f"{result.symbol} {interval}: {result.errors[0]}"
                    )
        return summary


    async def ingest_news(
        self,
        *,
        with_sentiment: bool = True,
        limit_per_source: int = 100,
        run_id: str | None = None,
    ) -> IngestResult:
        run_id = run_id or f"ingest_news_{uuid.uuid4().hex[:12]}"
        record_run(self.engine, run_id, "ingest_news", {"with_sentiment": with_sentiment})
        result = IngestResult(provider="rss", dataset=NEWS_DATASET)

        try:
            articles = await fetch_all_news(
                settings=self.settings, limit_per_source=limit_per_source
            )
            frame = articles_to_frame(articles)
            result.rows = len(frame)

            if with_sentiment and not frame.empty:
                logger.info("scoring_sentiment", extra={"articles": len(frame)})
                frame = self.sentiment.score_frame(frame)

            existing = self.store.read(NEWS_DATASET, columns=["uid"])
            known_uids = set(existing["uid"]) if not existing.empty else set()
            fresh = frame[~frame["uid"].isin(known_uids)]

            if fresh.empty:
                result.skipped = len(frame)
            else:
                result.inserted = self.store.write(
                    NEWS_DATASET, fresh,
                    partition_by=("year", "month"),
                    dedupe_on=("uid",), mode="upsert",
                )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(str(exc)[:300])
            logger.error("ingest_news_failed", extra={"error": str(exc)[:300]})

        result.finished_at = utc_now()
        finish_run(self.engine, run_id,
                   "failed" if result.errors else "succeeded", result.as_dict())
        logger.info("ingest_news_done", extra=result.as_dict())
        return result

    async def ingest_onchain(
        self, *, with_sentiment_index: bool = True, run_id: str | None = None
    ) -> list[IngestResult]:
        run_id = run_id or f"ingest_onchain_{uuid.uuid4().hex[:12]}"
        record_run(self.engine, run_id, "ingest_onchain", {})
        results: list[IngestResult] = []

        jobs: list[tuple[str, Any]] = [("mempool", None), ("coingecko_global", None)]

        async def run_mempool() -> list[OnChainSnapshot]:
            return await MempoolProvider(self.settings).fetch_all()

        async def run_global() -> list[OnChainSnapshot]:
            return await CoinGeckoGlobalProvider(self.settings).fetch()

        async def run_fng() -> list[OnChainSnapshot]:
            return await SentimentIndexProvider(self.settings).fetch_history()

        runners = [run_mempool, run_global]
        if with_sentiment_index:
            jobs.append(("alternative_me", None))
            runners.append(run_fng)

        outcomes = await gather_with_concurrency(runners, 3)
        for (name, _), outcome in zip(jobs, outcomes, strict=True):
            # Fear & Greed is kept separate from the on-chain series: it is a
            # sentiment indicator with its own history, not a chain metric.
            dataset = FEAR_GREED_DATASET if name == "alternative_me" else ONCHAIN_DATASET
            result = IngestResult(provider=name, dataset=dataset)
            try:
                if isinstance(outcome, Exception):
                    raise outcome
                snapshots = outcome
                frame = snapshots_to_frame(snapshots)
                result.rows = len(frame)
                if not frame.empty:
                    result.inserted = self.store.write(
                        dataset, frame,
                        partition_by=("year", "month"),
                        dedupe_on=("ts", "asset", "metric"), mode="upsert",
                    )
            except Exception as exc:  # noqa: BLE001
                result.errors.append(str(exc)[:300])
            result.finished_at = utc_now()
            results.append(result)

        finish_run(self.engine, run_id, "succeeded", {"results": [r.as_dict() for r in results]})
        return results

    async def ingest_all(
        self,
        symbols: Sequence[str],
        interval: str = "1h",
        providers: Sequence[str] | None = None,
        *,
        lookback_days: int = 730,
        news: bool = True,
        onchain: bool = True,
    ) -> dict[str, Any]:
        candle_results = await self.ingest_candles(
            symbols, interval, providers, lookback_days=lookback_days
        )
        news_result = await self.ingest_news() if news else None
        onchain_results = await self.ingest_onchain() if onchain else []

        return {
            "candles": [r.as_dict() for r in candle_results],
            "news": news_result.as_dict() if news_result else None,
            "onchain": [r.as_dict() for r in onchain_results],
            "total_rows": (
                sum(r.inserted for r in candle_results)
                + (news_result.inserted if news_result else 0)
                + sum(r.inserted for r in onchain_results)
            ),
        }

    def load_dataset(
        self,
        symbol: str,
        interval: str = "1h",
        *,
        exchanges: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Load a single-symbol OHLCV series, preferring the best available exchange."""
        frame = self.store.read(
            CANDLES_DATASET,
            filters=[
                ("symbol", "==", normalize_symbol(symbol)),
                # Several timeframes live in one dataset, so the interval must be
                # part of the filter. Omitting it silently returns a mixture of
                # 1h, 4h and 1d rows sharing timestamps.
                ("interval", "==", interval),
            ],
            start=start, end=end,
        )
        if frame.empty:
            return frame

        if exchanges:
            frame = frame[frame["exchange"].isin(list(exchanges))]
        if frame.empty:
            return frame

        preferred = self.settings.default_exchange
        has_primary = frame[frame["exchange"] == preferred]
        if not has_primary.empty:
            frame = has_primary
        else:
            coverage = frame.groupby("exchange").size().sort_values(ascending=False)
            if not coverage.empty:
                frame = frame[frame["exchange"] == coverage.index[0]]

        frame = frame.sort_values("ts").reset_index(drop=True)
        deduped = frame.drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
        return deduped

    def load_news(self, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        return self.store.read(NEWS_DATASET, start=start, end=end)

    def load_fear_greed(self) -> pd.DataFrame:
        return self.store.read(FEAR_GREED_DATASET)

    def load_onchain(self, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        return self.store.read(ONCHAIN_DATASET, start=start, end=end)


__all__ = [
    "CANDLES_DATASET",
    "FEAR_GREED_DATASET",
    "Ingestor",
    "NEWS_DATASET",
    "ONCHAIN_DATASET",
    "articles_to_frame",
    "candles_to_frame",
    "snapshots_to_frame",
]