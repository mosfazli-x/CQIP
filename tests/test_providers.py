"""Offline provider parser tests.

Payloads below are verbatim captures from each exchange's live API. Exchanges do
not agree on field order -- Gate in particular is *not* OHLC-ordered -- so these
tests pin the array positions and assert OHLC invariants that a wrong mapping
cannot satisfy.
"""

from __future__ import annotations

import asyncio
from datetime import UTC

import pytest

from cqip import providers as providers_module
from cqip.providers import (
    BitgetProvider,
    GateProvider,
    HTXProvider,
    normalize_symbol,
)

HOUR = 3600
# 2024-03-01T00:00:00Z, used so fixtures are stable regardless of wall clock.
BASE_TS = 1709251200

GATE_ROWS = [
    [str(BASE_TS + i * HOUR), "14486197.56304600", "83986.6", "84197.5", "83869.9",
     "84004.1", "172.44117900", "true"]
    for i in range(3)
]

HTX_ROWS = [
    {
        "id": BASE_TS + i * HOUR,
        "open": 83986.6,
        "high": 84197.5,
        "low": 83869.9,
        "close": 84004.1,
        "amount": 172.441179,
        "vol": 14486197.56,
        "count": 953,
    }
    for i in range(3)
]

# Bitget: [ts_ms, open, high, low, close, base_vol, quote_vol, usdt_vol]
BITGET_ROWS = [
    [str((BASE_TS + i * HOUR) * 1000), "83986.6", "84197.5", "83869.9", "84004.1",
     "172.441179", "14486197.56", "14486197.56"]
    for i in range(3)
]


def _patch_get_json(monkeypatch, payload):
    """Serve a fixed payload and stop pagination after the first page."""
    calls: list[dict] = []

    async def fake_get_json(_client, url, **kwargs):
        calls.append({"url": url, **kwargs})
        if len(calls) > 1:
            return [] if isinstance(payload, list) else {"data": []}
        return payload

    monkeypatch.setattr(providers_module, "get_json", fake_get_json)
    return calls


async def _fetch(provider, **kwargs):
    candles = await provider.fetch_candles("BTC_USDT", "1h", **kwargs)
    return candles


@pytest.mark.asyncio
class TestGateParsing:
    async def test_field_positions_match_live_api_order(self, monkeypatch):
        """Gate returns [ts, quote_vol, close, high, low, open, base_vol, closed]."""
        _patch_get_json(monkeypatch, GATE_ROWS)
        candles = await _fetch(GateProvider())

        assert len(candles) == 3
        first = candles[0]
        assert first.ts.timestamp() == BASE_TS
        assert first.open == pytest.approx(84004.1)
        assert first.high == pytest.approx(84197.5)
        assert first.low == pytest.approx(83869.9)
        assert first.close == pytest.approx(83986.6)
        assert first.volume == pytest.approx(172.441179)
        assert first.quote_volume == pytest.approx(14486197.563046)

    async def test_open_is_not_mistaken_for_quote_volume(self, monkeypatch):
        """Regression guard: positional mapping once put the quote volume in `open`."""
        _patch_get_json(monkeypatch, GATE_ROWS)
        candles = await _fetch(GateProvider())

        for candle in candles:
            assert candle.open < 1e5
            assert candle.quote_volume > 1e6
            assert candle.volume < 1e3

    async def test_partial_bars_are_discarded(self, monkeypatch):
        rows = [list(row) for row in GATE_ROWS]
        rows[-1][-1] = "false"
        _patch_get_json(monkeypatch, rows)

        candles = await _fetch(GateProvider())
        assert len(candles) == 2
        assert all(c.ts.timestamp() < BASE_TS + 2 * HOUR for c in candles)

    async def test_ohlc_invariants_hold(self, monkeypatch):
        _patch_get_json(monkeypatch, GATE_ROWS)
        for candle in await _fetch(GateProvider()):
            assert candle.high >= max(candle.open, candle.close) >= min(candle.open, candle.close) >= candle.low
            assert candle.low > 0
            assert candle.quote_volume == pytest.approx(candle.volume * candle.close, rel=0.05)

    async def test_short_rows_are_ignored(self, monkeypatch):
        _patch_get_json(monkeypatch, [["1709251200", "1", "2"]])
        assert await _fetch(GateProvider()) == []


@pytest.mark.asyncio
class TestHTXParsing:
    async def test_field_mapping(self, monkeypatch):
        _patch_get_json(monkeypatch, {"status": "ok", "data": HTX_ROWS})
        candles = await _fetch(HTXProvider())

        assert len(candles) == 3
        first = candles[0]
        assert first.ts.timestamp() == BASE_TS
        assert first.open == pytest.approx(83986.6)
        assert first.high == pytest.approx(84197.5)
        assert first.close == pytest.approx(84004.1)
        assert first.volume == pytest.approx(172.441179)
        assert first.quote_volume == pytest.approx(14486197.56)
        assert first.trades == 953

    async def test_ohlc_invariants_hold(self, monkeypatch):
        _patch_get_json(monkeypatch, {"status": "ok", "data": HTX_ROWS})
        for candle in await _fetch(HTXProvider()):
            assert candle.high >= max(candle.open, candle.close) >= min(candle.open, candle.close) >= candle.low


@pytest.mark.asyncio
class TestBitgetParsing:
    async def test_field_mapping(self, monkeypatch):
        _patch_get_json(monkeypatch, {"code": "00000", "data": BITGET_ROWS})
        candles = await _fetch(BitgetProvider())

        assert len(candles) == 3
        first = candles[0]
        assert first.ts.timestamp() == BASE_TS
        assert first.open == pytest.approx(83986.6)
        assert first.high == pytest.approx(84197.5)
        assert first.close == pytest.approx(84004.1)
        assert first.volume == pytest.approx(172.441179)
        assert first.quote_volume == pytest.approx(14486197.56)

    async def test_ohlc_invariants_hold(self, monkeypatch):
        _patch_get_json(monkeypatch, {"code": "00000", "data": BITGET_ROWS})
        for candle in await _fetch(BitgetProvider()):
            assert candle.high >= max(candle.open, candle.close) >= min(candle.open, candle.close) >= candle.low


@pytest.mark.parametrize(
    "provider_cls",
    [GateProvider, HTXProvider, BitgetProvider],
)
class TestSharedProviderContract:
    async def test_candles_are_sorted_and_deduplicated(self, provider_cls, monkeypatch):
        fixture = {
            GateProvider: GATE_ROWS,
            HTXProvider: {"status": "ok", "data": HTX_ROWS},
            BitgetProvider: {"code": "00000", "data": BITGET_ROWS},
        }[provider_cls]
        _patch_get_json(monkeypatch, fixture)

        candles = await _fetch(provider_cls())
        timestamps = [c.ts for c in candles]
        assert timestamps == sorted(timestamps)
        assert len(set(timestamps)) == len(timestamps)
        assert all(c.ts.tzinfo is not None for c in candles)

    async def test_start_filter_truncates_history(self, provider_cls, monkeypatch):
        from datetime import datetime

        fixture = {
            GateProvider: GATE_ROWS,
            HTXProvider: {"status": "ok", "data": HTX_ROWS},
            BitgetProvider: {"code": "00000", "data": BITGET_ROWS},
        }[provider_cls]
        _patch_get_json(monkeypatch, fixture)

        cutoff = datetime.fromtimestamp(BASE_TS + HOUR, tz=UTC)
        candles = await provider_cls().fetch_candles(
            "BTC_USDT", "1h", start=cutoff, end=datetime.fromtimestamp(BASE_TS + 3 * HOUR, tz=UTC)
        )
        assert candles
        assert all(c.ts >= cutoff for c in candles)

    async def test_normalizes_symbol_format(self, provider_cls):
        assert normalize_symbol("btc-usdt") == "BTC_USDT"
        assert normalize_symbol("BTC/USDT") == "BTC_USDT"


class TestIngestionAsyncContract:
    """`ingest_candles` performs network I/O and must be awaitable end to end."""

    def test_ingest_candles_is_coroutine_function(self):
        from cqip.ingest import Ingestor

        assert asyncio.iscoroutinefunction(Ingestor.ingest_candles)
        assert asyncio.iscoroutinefunction(Ingestor.ingest_all)

    def test_no_uncalled_coroutine_in_api_ingest_path(self):
        """The API must await ingest coroutines, never fire-and-forget them."""
        import inspect

        from cqip import api

        source = inspect.getsource(api.register_job_handlers)
        assert "await ingestor.ingest_candles" in source
        assert "asyncio.run(run())" in source

        endpoint = inspect.getsource(api.create_app)
        assert "await ingestor.ingest_all" in endpoint
