"""Shared pytest fixtures with isolated temp storage per test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def settings(tmp_path: Path):
    from cqip.config import Settings

    config = Settings(
        data_dir=tmp_path / "data",
        parquet_dir=tmp_path / "data" / "parquet",
        artifact_dir=tmp_path / "data" / "artifacts",
        db_url=f"sqlite:///{(tmp_path / 'data' / 'test.db').as_posix()}",
        allow_model_download=False,
        log_level="WARNING",
        seed=7,
    )
    config.ensure_directories()
    return config


@pytest.fixture
def engine(settings):
    from cqip.db import Engine

    return Engine(settings=settings)


@pytest.fixture
def store(settings):
    from cqip.storage import ParquetStore

    return ParquetStore(settings=settings)


@pytest.fixture
def synthetic_candles() -> pd.DataFrame:
    """Deterministic GBM price series with realistic intrabar structure."""
    rng = np.random.default_rng(42)
    n = 900
    start = pd.Timestamp("2024-01-01", tz="UTC")

    drift = 0.0002
    volatility = 0.012
    returns = rng.normal(drift, volatility, n)
    close = 30_000.0 * np.exp(np.cumsum(returns))

    open_price = np.concatenate([[30_000.0], close[:-1]])
    spread = np.abs(rng.normal(0, volatility / 2, n)) * close
    high = np.maximum(open_price, close) + spread
    low = np.minimum(open_price, close) - spread
    volume = rng.lognormal(10, 0.8, n)

    times = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "ts": times,
        "symbol": "BTC_USDT",
        "exchange": "gate",
        "interval": "1h",
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "quote_volume": volume * close,
        "trades": rng.integers(100, 5000, n),
    })


@pytest.fixture
def seeded_store(store, synthetic_candles):
    """Store populated with synthetic OHLCV plus scored news and on-chain rows."""
    store.write("ohlcv", synthetic_candles, dedupe_on=("ts", "symbol", "exchange"), mode="overwrite")

    rng = np.random.default_rng(11)
    n = 400
    news_times = pd.date_range("2024-01-01", periods=n, freq="3h", tz="UTC")
    news = pd.DataFrame({
        "ts": news_times,
        "uid": [f"uid-{i:04d}" for i in range(n)],
        "source": rng.choice(["coindesk", "cointelegraph", "sec", "fed"], n),
        "title": [f"Market update {i}" for i in range(n)],
        "url": [f"https://example.com/{i}" for i in range(n)],
        "summary": ["Bitcoin surges to new highs on institutional demand"] * n,
        "category": rng.choice(["market", "regulatory", "macro", "security"], n),
        "content_hash": [f"hash-{i:04d}" for i in range(n)],
        "sentiment_score": rng.normal(0.15, 0.4, n),
        "sentiment_confidence": rng.uniform(0.3, 1.0, n),
        "is_positive": (rng.random(n) > 0.5).astype(float),
        "is_negative": (rng.random(n) < 0.3).astype(float),
        "is_neutral": 0.0,
    })
    store.write("news", news, dedupe_on=("uid",), mode="overwrite")

    fg_times = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    fear_greed = pd.DataFrame({
        "ts": fg_times, "asset": "MARKET",
        "metric": "fear_greed_index", "value": rng.integers(10, 90, n).astype(float),
    })
    store.write("fear_greed", fear_greed, dedupe_on=("ts", "asset", "metric"), mode="overwrite")

    onchain_times = pd.date_range("2024-01-01", periods=48, freq="1D", tz="UTC")
    onchain = pd.DataFrame({
        "ts": onchain_times,
        "asset": ["MARKET"] * 24 + ["BTC"] * 24,
        "metric": ["total_market_cap_usd"] * 24 + ["dominance_pct"] * 24,
        "value": np.concatenate([rng.uniform(8e11, 2e12, 24), rng.uniform(40, 55, 24)]),
    })
    store.write("onchain", onchain, dedupe_on=("ts", "asset", "metric"), mode="overwrite")

    return store