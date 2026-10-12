"""News, on-chain and sentiment providers (all keyless)."""

from __future__ import annotations

import hashlib
import re
from abc import ABC
from datetime import UTC, datetime
from typing import Any
from xml.etree import ElementTree

import httpx

from cqip.config import Settings, get_settings
from cqip.logging_config import get_logger
from cqip.models import MacroCategory, NewsArticle, NewsSource, OnChainSnapshot
from cqip.utils import (
    RateLimiter,
    RetryPolicy,
    get_json,
    get_text,
    utc_now,
)

logger = get_logger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_NS_RE = re.compile(r"\{[^}]+\}")


def strip_html(text: str) -> str:
    cleaned = _TAG_RE.sub(" ", text or "")
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    return cleaned


def _parse_datetime(raw: str | None) -> datetime:
    if not raw:
        return utc_now()
    from dateutil import parser as date_parser

    try:
        parsed = date_parser.parse(raw)
    except (ValueError, TypeError):
        return utc_now()
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class NewsProvider(ABC):  # noqa: B024 - base for concrete feed subclasses
    name: str = "base"
    source: NewsSource = NewsSource.GENERAL
    url: str = ""
    default_category: MacroCategory = MacroCategory.GENERAL

    def __init__(self, settings: Settings | None = None, rate_limit: float = 0.5) -> None:
        self.settings = settings or get_settings()
        self.client: httpx.AsyncClient | None = None
        self.limiter = RateLimiter(rate_limit)
        self.policy = RetryPolicy(max_attempts=3, backoff_base=1.0)

    async def __aenter__(self) -> NewsProvider:
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

    async def fetch(self, limit: int = 100) -> list[NewsArticle]:
        try:
            raw = await get_text(
                self._require_client(), self.url, policy=self.policy, limiter=self.limiter
            )
            articles = self.parse(raw)
            return articles[:limit]
        except Exception as exc:  # noqa: BLE001
            logger.warning("news_fetch_failed", extra={"source": self.name, "error": str(exc)[:200]})
            return []

    def parse(self, raw: str) -> list[NewsArticle]:
        try:
            root = ElementTree.fromstring(raw.encode("utf-8", errors="ignore"))
        except ElementTree.ParseError as exc:
            logger.warning("rss_parse_failed", extra={"source": self.name, "error": str(exc)})
            return []

        articles: list[NewsArticle] = []
        for node in root.iter():
            tag = _NS_RE.sub("", node.tag).lower()
            if tag not in {"item", "entry"}:
                continue
            article = self._parse_entry(node)
            if article:
                articles.append(article)
        return articles

    def _parse_entry(self, node: ElementTree.Element) -> NewsArticle | None:
        fields: dict[str, str] = {}
        links: list[str] = []
        for child in node:
            tag = _NS_RE.sub("", child.tag).lower()
            text = (child.text or "").strip()
            if tag == "link":
                href = child.attrib.get("href")
                links.append(href or text)
            fields.setdefault(tag, text)
            if tag == "content" and not fields.get("description"):
                fields["description"] = text

        title = strip_html(fields.get("title", ""))
        if not title:
            return None

        url = links[0] if links else fields.get("guid", fields.get("link", ""))
        published = _parse_datetime(
            fields.get("pubdate") or fields.get("published") or fields.get("updated")
        )
        summary = strip_html(fields.get("description") or fields.get("summary") or "")[:1200]
        tags = tuple(
            t.strip() for t in (fields.get("category", "") or "").split(",") if t.strip()
        )

        uid = hashlib.sha1(f"{self.name}|{url or title}".encode()).hexdigest()[:24]
        content_hash = hashlib.sha1(f"{title}|{summary}".encode()).hexdigest()

        return NewsArticle(
            uid=uid,
            source=self.source,
            title=title,
            url=url,
            published_at=published,
            summary=summary,
            tags=tags,
            category=classify_article(title, summary, self.default_category),
            content_hash=content_hash,
        )


class RSSNewsProvider(NewsProvider):
    def __init__(
        self,
        name: str,
        source: NewsSource,
        url: str,
        category: MacroCategory = MacroCategory.GENERAL,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self.name = name
        self.source = source
        self.url = url
        self.default_category = category


MACRO_KEYWORDS: list[tuple[MacroCategory, tuple[str, ...]]] = [
    (MacroCategory.ETF, ("etf", "spot bitcoin etf", "ether etf", "blackrock", "grayscale")),
    (MacroCategory.REGULATORY, ("sec ", "cftc", "regulation", "regulator", "lawsuit",
                                 "subpoena", "indictment", "congress", "senate", "ban",
                                 "clarity act", "mica", "doj")),
    (MacroCategory.SECURITY, ("hack", "exploit", "breach", "rug pull", "stolen",
                              "vulnerability", "attack", "phishing", "scam")),
    (MacroCategory.MACRO, ("federal reserve", "fomc", "interest rate", "inflation",
                           "cpi", "jobs report", "unemployment", "powell", "treasury",
                           "recession", "rate cut", "rate hike", "gdp")),
    (MacroCategory.TECHNOLOGY, ("upgrade", "hard fork", "mainnet", "layer 2", "sharding",
                                "staking", "merge", "protocol upgrade")),
    (MacroCategory.DEFI, ("defi", "decentralized finance", "liquidity pool", "yield",
                          "dex", "amm", "lending protocol")),
    (MacroCategory.ADOPTION, ("institutional", "adoption", "treasury buys", "etf inflows",
                              "corporate", "payment", "custody")),
]


def classify_article(
    title: str, summary: str, default: MacroCategory = MacroCategory.GENERAL
) -> MacroCategory:
    blob = f"{title} {summary}".lower()
    for category, keywords in MACRO_KEYWORDS:
        if any(keyword in blob for keyword in keywords):
            return category
    return default


FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=0&format=json"


class SentimentIndexProvider:
    """Alternative.me Crypto Fear & Greed index (daily history, keyless)."""

    name = "alternative_me"

    def __init__(self, settings: Settings | None = None, rate_limit: float = 0.5) -> None:
        self.settings = settings or get_settings()
        self.limiter = RateLimiter(rate_limit)
        self.policy = RetryPolicy(max_attempts=3)

    async def fetch_history(self, limit: int = 0) -> list[OnChainSnapshot]:
        from cqip.utils import create_http_client

        async with create_http_client(self.settings) as client:
            payload = await get_json(
                client,
                FEAR_GREED_URL,
                params={"limit": limit} if limit else None,
                policy=self.policy,
                limiter=self.limiter,
            )
        return self.parse(payload)

    def parse(self, payload: dict[str, Any]) -> list[OnChainSnapshot]:
        snapshots: list[OnChainSnapshot] = []
        for row in payload.get("data", []):
            try:
                ts = datetime.fromtimestamp(int(row["timestamp"]), tz=UTC)
                snapshots.append(
                    OnChainSnapshot(
                        ts=ts, asset="MARKET", metric="fear_greed_index",
                        value=float(row["value"]),
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue
        return snapshots


class MempoolProvider:
    """Bitcoin network state from mempool.space."""

    name = "mempool_space"
    endpoints = {
        "recommended_fees": "https://mempool.space/api/v1/fees/recommended",
        "mempool_info": "https://mempool.space/api/mempool",
        "difficulty_adjustment": "https://mempool.space/api/v1/difficulty-adjustment",
        "hashrate": "https://mempool.space/api/v1/mining/hashrate/3m",
        "mining_pools": "https://mempool.space/api/v1/mining/pools/1m",
    }

    def __init__(self, settings: Settings | None = None, rate_limit: float = 1.0) -> None:
        self.settings = settings or get_settings()
        self.limiter = RateLimiter(rate_limit)
        self.policy = RetryPolicy(max_attempts=3)

    async def fetch_all(self) -> list[OnChainSnapshot]:
        from cqip.utils import create_http_client

        metrics: list[OnChainSnapshot] = []
        async with create_http_client(self.settings) as client:
            for key, url in self.endpoints.items():
                try:
                    payload = await get_json(
                        client, url, policy=self.policy, limiter=self.limiter
                    )
                    metrics.extend(self._extract(key, payload))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("onchain_fetch_failed", extra={"endpoint": key, "error": str(exc)[:160]})
        return metrics

    def _extract(self, endpoint: str, payload: Any) -> list[OnChainSnapshot]:
        ts = utc_now()
        out: list[OnChainSnapshot] = []
        if not isinstance(payload, dict):
            return out
        if endpoint == "recommended_fees":
            for key, value in payload.items():
                out.append(OnChainSnapshot(ts, "BTC", f"fee_{key}_sat_vb", float(value)))
        elif endpoint == "mempool_info":
            for key, value in payload.items():
                if isinstance(value, (int, float)):
                    out.append(OnChainSnapshot(ts, "BTC", f"mempool_{key}", float(value)))
        elif endpoint == "difficulty_adjustment":
            for key, value in payload.items():
                if isinstance(value, (int, float)):
                    out.append(OnChainSnapshot(ts, "BTC", f"difficulty_{key}", float(value)))
        elif endpoint == "hashrate":
            for bucket in payload.get("hashrates", []) or []:
                # mempool.space reports avgHashrate in H/s. The metric is named
                # `_eh_s`, so convert here: storing raw H/s both mislabels the
                # series and overflows the float32 feature scaler (1e21^2 > 3.4e38).
                out.append(OnChainSnapshot(
                    datetime.fromtimestamp(int(bucket["timestamp"]), tz=UTC),
                    "BTC", "hashrate_eh_s", float(bucket["avgHashrate"]) / 1e18,
                ))
        elif endpoint == "mining_pools":
            for pool in payload.get("pools", []) or []:
                out.append(OnChainSnapshot(
                    ts, "BTC", f"pool_share_{pool.get('name', 'unknown')}",
                    float(pool.get("share", 0.0)),
                ))
        return out


class CoinGeckoGlobalProvider:
    """Global market metrics: total market cap, volume, BTC dominance."""

    name = "coingecko_global"
    url = "https://api.coingecko.com/api/v3/global"

    def __init__(self, settings: Settings | None = None, rate_limit: float = 0.3) -> None:
        self.settings = settings or get_settings()
        self.limiter = RateLimiter(rate_limit)
        self.policy = RetryPolicy(max_attempts=3)

    async def fetch(self) -> list[OnChainSnapshot]:
        from cqip.utils import create_http_client

        async with create_http_client(self.settings) as client:
            payload = await get_json(client, self.url, policy=self.policy, limiter=self.limiter)
        data = (payload or {}).get("data") or {}
        ts = datetime.fromtimestamp(int(data.get("updated_at", utc_now().timestamp())), tz=UTC)
        out: list[OnChainSnapshot] = []
        mapping = {
            "total_market_cap_usd": data.get("total_market_cap", {}).get("usd"),
            "total_volume_usd": data.get("total_volume", {}).get("usd"),
            "market_cap_change_24h_pct": data.get("market_cap_change_percentage_24h_usd"),
            "active_cryptocurrencies": data.get("active_cryptocurrencies"),
        }
        for metric, value in mapping.items():
            if value is not None:
                out.append(OnChainSnapshot(ts=ts, asset="MARKET", metric=metric, value=float(value)))
        dominance = data.get("market_cap_percentage") or {}
        for coin, share in dominance.items():
            out.append(OnChainSnapshot(
                ts=ts, asset=coin.upper(), metric="dominance_pct", value=float(share)
            ))
        return out


NEWS_FEEDS: list[tuple[str, NewsSource, str, MacroCategory]] = [
    ("coindesk", NewsSource.COINDESK, "https://www.coindesk.com/arc/outboundfeeds/rss/", MacroCategory.MARKET),
    ("cointelegraph", NewsSource.COINTELEGRAPH, "https://cointelegraph.com/rss", MacroCategory.MARKET),
    ("decrypt", NewsSource.DECRYPT, "https://decrypt.co/feed", MacroCategory.TECHNOLOGY),
    ("theblock", NewsSource.THEBLOCK, "https://www.theblock.co/rss.xml", MacroCategory.MARKET),
    ("bitcoinmagazine", NewsSource.BITCOINMAGAZINE, "https://bitcoinmagazine.com/feed", MacroCategory.MARKET),
    ("cryptoslate", NewsSource.CRYPTOSLATE, "https://cryptoslate.com/feed/", MacroCategory.MARKET),
    ("newsbtc", NewsSource.NEWSBTC, "https://www.newsbtc.com/feed/", MacroCategory.MARKET),
    ("ambcrypto", NewsSource.AMBCRYPTO, "https://ambcrypto.com/feed/", MacroCategory.MARKET),
    ("beincrypto", NewsSource.BEINCRYPTO, "https://beincrypto.com/feed/", MacroCategory.MARKET),
    ("cryptonews", NewsSource.CRYPTONEWS, "https://cryptonews.com/news/feed/", MacroCategory.MARKET),
    ("sec", NewsSource.SEC, "https://www.sec.gov/news/pressreleases.rss", MacroCategory.REGULATORY),
    ("fed", NewsSource.FED, "https://www.federalreserve.gov/feeds/press_all.xml", MacroCategory.MACRO),
]


def build_news_providers(settings: Settings | None = None) -> list[NewsProvider]:
    return [
        RSSNewsProvider(name, source, url, category, settings=settings)
        for name, source, url, category in NEWS_FEEDS
    ]


async def fetch_all_news(
    settings: Settings | None = None, limit_per_source: int = 100, concurrency: int = 6
) -> list[NewsArticle]:
    from cqip.utils import gather_with_concurrency

    providers = build_news_providers(settings)

    async def run(provider: NewsProvider) -> list[NewsArticle]:
        async with provider:
            return await provider.fetch(limit=limit_per_source)

    outcomes = await gather_with_concurrency(
        [lambda p=p: run(p) for p in providers], concurrency
    )
    articles: list[NewsArticle] = []
    for provider, outcome in zip(providers, outcomes, strict=True):
        if isinstance(outcome, Exception):
            logger.warning("news_source_failed", extra={"source": provider.name, "error": str(outcome)[:160]})
        else:
            articles.extend(outcome)

    seen: set[str] = set()
    unique: list[NewsArticle] = []
    for article in articles:
        if article.uid in seen:
            continue
        seen.add(article.uid)
        unique.append(article)
    return sorted(unique, key=lambda a: a.published_at, reverse=True)


__all__ = [
    "CoinGeckoGlobalProvider",
    "MempoolProvider",
    "NEWS_FEEDS",
    "NewsProvider",
    "RSSNewsProvider",
    "SentimentIndexProvider",
    "build_news_providers",
    "classify_article",
    "fetch_all_news",
    "strip_html",
]