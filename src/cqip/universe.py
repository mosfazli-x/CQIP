"""A curated cryptocurrency universe and resolution against live exchange listings.

The point of this module is to avoid silently ingesting nothing. A symbol that
does not trade on the chosen exchange, a stablecoin that cannot be a prediction
target, and a delisted ticker all look identical to a caller that just hands a
string to the API. Resolution makes each of those explicit.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from cqip.config import Settings, get_settings
from cqip.logging_config import get_logger
from cqip.providers import MarketDataProvider, get_provider
from cqip.utils import normalize_symbol

logger = get_logger(__name__)

#: Popular tickers by standing, roughly ordered by market capitalisation.
#: These are base assets; the quote asset is appended during resolution.
TOP_CRYPTO_SYMBOLS: list[str] = [
    "BTC", "ETH", "USDT", "BNB", "XRP", "USDC", "SOL", "TRX", "ZEC", "HYPE",
    "DOGE", "XMR", "LINK", "ADA", "LEO", "XLM", "BCH", "NEAR", "UNI", "LTC",
    "USDE", "DAI", "AVAX", "CC", "USD1", "SUI", "HBAR", "TON", "TAO",
    "SHIB", "USDG", "CRO", "M", "PYUSD", "XAUT", "ONDO", "OKB", "RLUSD",
    "ENA", "AAVE", "MNT", "DOT", "ASTER", "PAXG", "PEPE", "PUMP", "WLFI",
    "ICP", "SKY", "WLD", "USDD", "VVV", "MORPHO", "ETC", "ARB", "USTB",
    "BGB", "LIT", "POL", "GT", "KAS", "QNT", "ALGO", "KCS", "JUP", "PI",
    "RENDER", "ATOM", "JST", "CAKE", "FIL", "DASH", "INJ", "VET", "STABLE",
    "APT", "AERO", "ETHFI", "XDC", "FLR", "PENGU", "TRUMP", "STX", "RAY",
    "NEXO", "ZRO", "CRV", "PYTH", "FET", "TUSD", "VIRTUAL", "PIE", "SEI",
    "EURC", "TIA", "PENDLE", "FF", "BSV", "PONS", "SPX", "NIGHT", "BTT",
    "LDO", "XTZ", "BONK", "SUN", "KITE", "FDUSD", "DCR", "XPL", "IMX",
    "LUNC", "OP", "GNO", "ENS", "MON", "AR", "STRK", "GRT", "CFX",
    "USELESS", "FLOKI", "SYRUP", "JTO", "TWT", "COMP", "AINFT", "WIF",
    "IOTA", "THETA", "EIGEN", "ZAMA", "KAIA", "KMNO", "RUNE", "CVX",
    "BUILD", "ZBCN", "AKT", "XCN", "EDX", "2Z", "AXS", "MINA", "FARTCOIN",
    "TRAC", "NEO", "XEC", "MET", "MANA", "CHZ", "MX", "V", "A", "DEXE",
    "CAT", "TEL", "SENT", "AIOZ", "APE", "SFP", "GOMINING", "1INCH", "ZEN",
    "BAT", "EGLD", "ZK", "H", "AWE", "GLM", "SAND", "GEOD", "PLUME",
    "GENIUS", "SOON", "ATH", "S", "FLUID", "FOUR", "MARS", "DEEP", "DYDX",
    "SUPER", "WPEAQ", "DOG", "PEAQ", "RSR", "MELANIA", "GRASS", "QTUM",
    "SEEKER", "BEAM", "CHEEMS", "ZRX", "PROM", "GALA", "WEMIX", "ORDI",
    "MKR", "SNX", "SUSHI", "YFI", "GMX", "HNT", "RPL", "KAVA", "ZIL", "EOS",
    "IOTX", "ENJ", "ANKR", "ONE", "CELO", "ROSE", "KSM", "BAND", "API3",
    "LPT", "MASK", "SSV", "BLUR", "MAGIC", "ACH", "STORJ", "SKL", "ICX",
    "ONT", "WOO", "AUDIO", "HOT", "RVN", "SC", "GLMR", "CELR", "CTSI",
    "CHR", "COTI", "FLUX", "DUSK", "ILV", "STG", "AGIX", "OCEAN",
]

#: Assets whose price is pegged to a reference currency. They barely move, so a
#: directional-accuracy model trained on them reports noise near 50%. They are
#: still useful as a *quote* asset, and useful for on-chain context, but they
#: must not be treated as prediction subjects.
STABLECOINS: frozenset[str] = frozenset({
    "DAI", "EURC", "FDUSD", "PYUSD", "RLUSD", "STABLE", "TUSD", "USD1",
    "USDC", "USDD", "USDE", "USDG", "USDT", "USTB",
})

#: Tickers that no longer trade under that name. Mapping to the replacement
#: keeps a stale list usable instead of dropping the asset entirely.
RENAMED_SYMBOLS: dict[str, str] = {
    "MATIC": "POL",  # Polygon rebranded; MATIC_USDT is delisted on Gate
    "MKR": "SKY",  # Maker rebranded; MKR_USDT is delisted, SKY_USDT trades
}

#: Symbols that are not Gate spot pairs. Kept so the gap report explains *why*
#: rather than printing a bare "unavailable", which reads like a bug when the
#: real cause is a different venue, a stablecoin base, or a perp-only token.
EXCHANGE_NOTES: dict[str, str] = {
    "USDT": "the quote asset itself, so no USDT/USDT pair exists",
    "XMR": "delisted on Gate; deep privacy-coin history sits on Kraken/Binance",
    "TON": "The Open Network; not listed on Gate",
    "LEO": "UNUS SED LEO is a Bitfinex token with no Gate pair",
    "HYPE": "Hyperliquid is a perp DEX token, not Gate spot",
    "TAO": "Bittensor; not listed on Gate",
    "KCS": "KuCoin Shares; no Gate pair",
    "BGB": "Bitget token; no Gate pair",
    "MX": "MEXC's MX Token; no Gate pair",
    "PI": "Pi Network; not listed on Gate",
    "EOS": "delisted on Gate",
    "ICX": "ICON; delisted on Gate",
    "AGIX": "SingularityNET; delisted on Gate",
    "OCEAN": "Ocean Protocol; delisted on Gate",
}

#: Depth we can actually obtain, established by probing Gate's paginated
#: candlestick endpoint. These are server-side retention limits, not client
#: caps: requesting an older cursor returns INVALID_PARAM_VALUE.
#:
#: Daily history is the one interval with no practical ceiling, so the value is
#: set high enough to always reach the original listing date rather than to
#: encode an arbitrary cutoff. The provider stops when its pages run out.
DEFAULT_INTERVAL_DEPTH_DAYS: dict[str, int] = {
    "1d": 20000,  # ~54 years; in practice bounded by the listing date
    "1w": 20000,
    "4h": 1000,
    "1h": 365,    # hourly history is truncated to roughly the last year
    "30m": 60,
    "15m": 30,
    "5m": 7,
    "1m": 1,
}

#: Intervals whose data cannot support the forecaster's window construction on
#: the retention-limited depths above. A sweep over these would be a waste of
#: CPU: 365 hourly bars is under the ~1405 bars training requires.
INSUFFICIENT_FOR_FORECASTING: frozenset[str] = frozenset({"5m", "15m", "30m"})


@dataclass(slots=True)
class SymbolStatus:
    """Resolution outcome for one requested base asset."""

    requested: str
    symbol: str | None
    available: bool
    reason: str = ""
    is_stablecoin: bool = False
    renamed_from: str | None = None
    earliest_ts: str | None = None
    rows: int = 0

    @property
    def usable_as_target(self) -> bool:
        """Whether this symbol is a sensible subject for price prediction."""
        return self.available and not self.is_stablecoin

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "symbol": self.symbol,
            "available": self.available,
            "usable_as_target": self.usable_as_target,
            "is_stablecoin": self.is_stablecoin,
            "renamed_from": self.renamed_from,
            "earliest_ts": self.earliest_ts,
            "rows": self.rows,
            "reason": self.reason,
        }


@dataclass(slots=True)
class Universe:
    """A resolved set of tradable symbols plus the ones that were dropped."""

    statuses: list[SymbolStatus] = field(default_factory=list)
    provider: str = "gate"
    quote: str = "USDT"

    @property
    def tradable(self) -> list[str]:
        return [s.symbol for s in self.statuses if s.available and s.symbol]

    @property
    def targets(self) -> list[str]:
        """Tradable symbols that are meaningful prediction subjects."""
        return [s.symbol for s in self.statuses if s.usable_as_target and s.symbol]

    @property
    def skipped(self) -> list[SymbolStatus]:
        return [s for s in self.statuses if not s.available]

    @property
    def stablecoins(self) -> list[SymbolStatus]:
        return [s for s in self.statuses if s.available and s.is_stablecoin]


def resolve_universe(
    tickers: Sequence[str] | None = None,
    *,
    provider_name: str = "gate",
    quote: str = "USDT",
    settings: Settings | None = None,
    fallback_providers: Sequence[str] | None = None,
) -> Universe:
    """Map base tickers onto tradable pairs on a live exchange.

    Reports rather than raises: an unavailable symbol, a stablecoin and a
    delisted ticker each get a distinct reason so the gap is legible instead of
    turning into a silent empty result later.

    `fallback_providers` are checked only for the symbols the primary venue
    rejected. Several majors (XMR, TON, EOS, AGIX, KAS) were delisted from Gate
    but still trade on other exchanges, so a bare "unavailable" would hide
    recoverable history from anyone reading the report.
    """
    settings = settings or get_settings()
    tickers = list(tickers or TOP_CRYPTO_SYMBOLS)
    quote = quote.upper()

    provider = get_provider(provider_name, settings)
    listed = _listed_pairs(provider, quote)

    statuses: list[SymbolStatus] = []
    for ticker in tickers:
        statuses.append(_resolve_one(ticker, listed, quote))

    universe = Universe(statuses=statuses, provider=provider_name, quote=quote)

    candidates = [p for p in (fallback_providers or ("htx", "bitget", "binance"))
                  if p != provider_name]
    for name in candidates:
        gaps = [s for s in universe.statuses if not s.available]
        if not gaps:
            break
        try:
            elsewhere = _listed_pairs(get_provider(name, settings), quote)
        except Exception as exc:  # noqa: BLE001
            logger.warning("universe_fallback_failed",
                           extra={"provider": name, "error": str(exc)[:200]})
            continue
        for status in gaps:
            # `elsewhere` holds normalized pairs, so compare the pair form
            # rather than the bare ticker or every check silently misses.
            base = status.requested.split("_")[0]
            pair = f"{base}_{quote}"
            if pair in elsewhere:
                status.reason = (
                    f"{normalize_symbol(pair)} trades on {name}, "
                    f"not on {provider_name}"
                )
    return universe


def _resolve_one(ticker: str, listed: set[str], quote: str) -> SymbolStatus:
    requested = ticker.strip().upper()
    renamed_from: str | None = None

    # Accept an already-quoted pair such as "BTC_USDT" alongside a bare ticker,
    # so a caller can pass through whatever they hold without pre-splitting it.
    if "_" in requested:
        base, _, given_quote = requested.partition("_")
        candidate = base
        quote = given_quote or quote
    else:
        candidate = requested

    if f"{candidate}_{quote}" not in listed:
        # A renamed asset is a recoverable gap, so try the replacement before
        # giving up on the ticker entirely.
        replacement = RENAMED_SYMBOLS.get(candidate)
        if replacement and f"{replacement}_{quote}" in listed:
            renamed_from, candidate = candidate, replacement

    symbol = f"{candidate}_{quote}"
    if symbol not in listed:
        # Even when the replacement is unlisted too, naming the known rename
        # turns "unknown symbol" into an actionable next step.
        hint = ""
        replacement = RENAMED_SYMBOLS.get(candidate)
        if replacement:
            hint = f" (known rename: try {replacement}_{quote})"
        note = EXCHANGE_NOTES.get(candidate)
        if note:
            hint = f" - {note}"
        return SymbolStatus(
            requested=requested,
            symbol=None,
            available=False,
            reason=f"{symbol} is not a tradable {quote} pair on this exchange{hint}",
        )

    return SymbolStatus(
        requested=requested,
        symbol=normalize_symbol(symbol),
        available=True,
        is_stablecoin=candidate in STABLECOINS,
        renamed_from=renamed_from,
        reason=(
            "stablecoin: price is pegged, so directional accuracy is not meaningful"
            if candidate in STABLECOINS
            else (f"trades under {candidate} after rename from {renamed_from}"
                  if renamed_from else "")
        ),
    )


async def _listed_pairs_async(provider: MarketDataProvider, quote: str) -> set[str]:
    raw = await provider.list_symbols()
    pairs: set[str] = set()
    for item in raw:
        symbol = normalize_symbol(str(item))
        if symbol.endswith(f"_{quote}"):
            pairs.add(symbol)
    return pairs


def _listed_pairs(provider: MarketDataProvider, quote: str) -> set[str]:
    return asyncio.run(_listed_pairs_async(provider, quote))


def plan_backfill(
    universe: Universe,
    *,
    intervals: Sequence[str] = ("1d", "1h"),
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Build the per-(symbol, interval) ingest plan with honest depth limits.

    Depth comes from what the exchange will actually serve, not from a
    requested lookback: asking hourly data for ten years silently returns the
    last year, which is far more damaging than a shorter, explicit plan.
    """
    settings = settings or get_settings()
    plan: list[dict[str, Any]] = []

    for symbol in universe.tradable:
        for interval in intervals:
            depth = DEFAULT_INTERVAL_DEPTH_DAYS.get(interval)
            if depth is None:
                continue
            note = ""
            if interval in INSUFFICIENT_FOR_FORECASTING:
                note = (
                    "too little retained history to fit the forecaster window; "
                    "usable for context, not for training"
                )
            plan.append({
                "symbol": symbol,
                "interval": interval,
                "lookback_days": depth,
                "trainable": interval not in INSUFFICIENT_FOR_FORECASTING,
                "note": note,
            })
    return plan


def missing_targets(universe: Universe, requested: Iterable[str] | None = None) -> list[str]:
    """Tickers that were asked for but produced no usable prediction subject."""
    wanted = {t.strip().upper() for t in (requested or TOP_CRYPTO_SYMBOLS)}
    return [s.requested for s in universe.statuses if s.requested in wanted and not s.usable_as_target]


__all__ = [
    "DEFAULT_INTERVAL_DEPTH_DAYS",
    "EXCHANGE_NOTES",
    "INSUFFICIENT_FOR_FORECASTING",
    "RENAMED_SYMBOLS",
    "STABLECOINS",
    "TOP_CRYPTO_SYMBOLS",
    "SymbolStatus",
    "Universe",
    "missing_targets",
    "plan_backfill",
    "resolve_universe",
]
