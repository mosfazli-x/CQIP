"""Domain models shared across ingestion, storage, features and modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class Exchange(StrEnum):
    GATE = "gate"
    HTX = "htx"
    BITGET = "bitget"
    BINANCE = "binance"
    COINGECKO = "coingecko"


class Interval(StrEnum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class NewsSource(StrEnum):
    COINDESK = "coindesk"
    COINTELEGRAPH = "cointelegraph"
    DECRYPT = "decrypt"
    THEBLOCK = "theblock"
    BITCOINMAGAZINE = "bitcoinmagazine"
    CRYPTOSLATE = "cryptoslate"
    NEWSBTC = "newsbtc"
    AMBCRYPTO = "ambcrypto"
    BEINCRYPTO = "beincrypto"
    CRYPTONEWS = "cryptonews"
    SEC = "sec"
    FED = "fed"
    GENERAL = "general"


class MacroCategory(StrEnum):
    REGULATORY = "regulatory"
    ETF = "etf"
    MACRO = "macro"
    SECURITY = "security"
    TECHNOLOGY = "technology"
    ADOPTION = "adoption"
    DEFI = "defi"
    MARKET = "market"
    GENERAL = "general"


class SentimentLabel(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


@dataclass(slots=True, frozen=True)
class Candle:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float | None = None
    trades: int | None = None

    def as_row(self) -> dict[str, object]:
        return {
            "ts": self.ts,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
            "trades": self.trades,
        }


@dataclass(slots=True, frozen=True)
class NewsArticle:
    uid: str
    source: NewsSource
    title: str
    url: str
    published_at: datetime
    summary: str = ""
    author: str | None = None
    tags: tuple[str, ...] = ()
    category: MacroCategory = MacroCategory.GENERAL
    content_hash: str = ""


@dataclass(slots=True, frozen=True)
class OnChainSnapshot:
    ts: datetime
    asset: str
    metric: str
    value: float


@dataclass(slots=True)
class IngestResult:
    provider: str
    dataset: str
    symbol: str | None = None
    rows: int = 0
    inserted: int = 0
    skipped: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    finished_at: datetime | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "dataset": self.dataset,
            "symbol": self.symbol,
            "rows": self.rows,
            "inserted": self.inserted,
            "skipped": self.skipped,
            "duration_seconds": round(self.duration_seconds, 3),
            "errors": self.errors,
        }