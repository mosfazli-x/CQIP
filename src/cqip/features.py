"""Feature engineering: technical indicators, microstructure and sentiment features.

All transformations are causal — every feature at index ``i`` uses only data at
or before ``i``, so the same code is safe for both training and live inference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-12

TECHNICAL_FEATURES: list[str] = [
    "returns_1", "returns_3", "returns_6", "returns_12", "returns_24",
    "log_volume_change", "volume_zscore_20", "obv_slope_10", "dollar_volume_zscore_20",
    "momentum_5", "momentum_10", "momentum_20", "momentum_60",
    "ema_fast_slow_spread", "sma_fast_slow_spread",
    "rsi_7", "rsi_14", "rsi_21",
    "macd_hist", "macd_line", "macd_signal",
    "bb_percent_b", "bb_bandwidth", "bb_width",
    "atr_14_pct", "high_low_range_14", "close_position_14",
    "realized_vol_5", "realized_vol_10", "realized_vol_20", "realized_vol_60",
    "vol_of_vol_20",
    "drawdown_60", "zscore_close_20", "hurst_approx",
]

SENTIMENT_FEATURES: list[str] = [
    "news_sentiment_1h", "news_sentiment_24h", "news_sentiment_7d",
    "news_volume_1h", "news_volume_24h",
    "news_sentiment_momentum_24h", "news_negative_ratio_24h",
    "news_positive_ratio_24h", "news_ewma_sentiment",
    "fear_greed_index", "fear_greed_change_7d",
    "sec_regulatory_sentiment_24h", "macro_sentiment_24h",
]

ONCHAIN_FEATURES: list[str] = [
    "btc_dominance", "total_market_cap_usd_log", "total_volume_usd_log",
    "market_cap_change_24h_pct", "mempool_fee_median", "mempool_vsize",
      "btc_hashrate_eh_s", "btc_difficulty_progress",
]

MARKET_FEATURES: list[str] = [
    "open", "high", "low", "close", "volume",
    "vwap_deviation", "upper_wick_ratio", "lower_wick_ratio", "body_ratio",
    "gap_from_prev_close", "range_expansion_10",
]


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=max(1, window // 2)).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = _safe_div(avg_gain, avg_loss)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger(series: pd.Series, window: int = 20, deviations: float = 2.0):
    middle = sma(series, window)
    std = series.rolling(window, min_periods=max(1, window // 2)).std()
    upper = middle + deviations * std
    lower = middle - deviations * std
    bandwidth = _safe_div(upper - lower, middle)
    percent_b = _safe_div(series - lower, upper - lower)
    return middle, upper, lower, percent_b, bandwidth


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(
        alpha=1 / window, adjust=False, min_periods=window
    ).mean()


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def realized_volatility(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=max(2, window // 2)).std() * np.sqrt(365)


def hurst_approx(series: pd.Series, window: int = 100) -> pd.Series:
    """Rolling Hurst exponent via rescaled range on log prices.

    ``NaN`` while insufficient history. Values < 0.5 mean mean-reversion,
    > 0.5 trending. Computed on a decimated series for speed.
    """
    values = series.values.astype(float)
    out = np.full(len(values), np.nan)
    for end in range(window, len(values) + 1):
        window_values = values[end - window : end]
        if not np.all(np.isfinite(window_values)) or window_values[0] <= 0:
            continue
        log_prices = np.log(window_values)
        cumulative = np.cumsum(log_prices - log_prices.mean())
        r = np.ptp(cumulative)
        s = log_prices.std()
        if s <= 0:
            continue
        out[end - 1] = np.clip(np.log(r / s) / np.log(window), 0.0, 1.0)
    return pd.Series(out, index=series.index)


def compute_technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build causal technical features from an OHLCV frame sorted by ``ts``."""
    if frame.empty:
        return pd.DataFrame()

    df = frame.sort_values("ts").reset_index(drop=True).copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)

    log_close = np.log(close.replace(0, np.nan))
    log_returns = log_close.diff()

    out = pd.DataFrame(index=df.index)

    for window in (1, 3, 6, 12, 24):
        out[f"returns_{window}"] = log_returns.rolling(window).sum()

    out["log_volume_change"] = np.log1p(volume).diff()
    volume_mean = sma(volume, 20)
    volume_std = volume.rolling(20, min_periods=10).std()
    out["volume_zscore_20"] = _safe_div(volume - volume_mean, volume_std)
    dollar_volume = volume * close
    dollar_mean = sma(dollar_volume, 20)
    dollar_std = dollar_volume.rolling(20, min_periods=10).std()
    out["dollar_volume_zscore_20"] = _safe_div(dollar_volume - dollar_mean, dollar_std)
    obv = on_balance_volume(close, volume)
    out["obv_slope_10"] = obv.diff(10)

    for window in (5, 10, 20, 60):
        out[f"momentum_{window}"] = _safe_div(close - close.shift(window), close.shift(window))

    ema_9, ema_21 = ema(close, 9), ema(close, 21)
    out["ema_fast_slow_spread"] = _safe_div(ema_9 - ema_21, ema_21)
    sma_10, sma_30 = sma(close, 10), sma(close, 30)
    out["sma_fast_slow_spread"] = _safe_div(sma_10 - sma_30, sma_30)

    for window in (7, 14, 21):
        out[f"rsi_{window}"] = rsi(close, window)

    macd_line, macd_signal, macd_hist = macd(close)
    out["macd_line"] = _safe_div(macd_line, close)
    out["macd_signal"] = _safe_div(macd_signal, close)
    out["macd_hist"] = _safe_div(macd_hist, close)

    _mid, _up, _low, percent_b, bandwidth = bollinger(close, 20)
    out["bb_percent_b"] = percent_b
    out["bb_bandwidth"] = bandwidth
    out["bb_width"] = _safe_div(_up - _low, _mid)

    atr_14 = atr(high, low, close, 14)
    out["atr_14_pct"] = _safe_div(atr_14, close)

    rolling_high = high.rolling(14, min_periods=7).max()
    rolling_low = low.rolling(14, min_periods=7).min()
    out["high_low_range_14"] = _safe_div(rolling_high - rolling_low, close)
    out["close_position_14"] = _safe_div(close - rolling_low, rolling_high - rolling_low).fillna(0.5)

    for window in (5, 10, 20, 60):
        out[f"realized_vol_{window}"] = realized_volatility(log_returns, window)
    vol_20 = out["realized_vol_20"]
    out["vol_of_vol_20"] = _safe_div(vol_20.diff(5).abs(), vol_20)

    rolling_max = close.rolling(60, min_periods=10).max()
    out["drawdown_60"] = _safe_div(close - rolling_max, rolling_max)

    close_mean = sma(close, 20)
    close_std = close.rolling(20, min_periods=10).std()
    out["zscore_close_20"] = _safe_div(close - close_mean, close_std)
    out["hurst_approx"] = hurst_approx(close, 100)

    typical_price = (high + low + close) / 3
    vwap = _safe_div((typical_price * volume).cumsum(), volume.cumsum())
    out["vwap_deviation"] = _safe_div(close - vwap, vwap)

    candle_range = (high - low).replace(0, np.nan)
    out["upper_wick_ratio"] = _safe_div(high - np.maximum(open_, close), candle_range).fillna(0.0)
    out["lower_wick_ratio"] = _safe_div(np.minimum(open_, close) - low, candle_range).fillna(0.0)
    out["body_ratio"] = _safe_div(close - open_, candle_range).fillna(0.0)
    out["gap_from_prev_close"] = _safe_div(open_ - close.shift(1), close.shift(1))
    range_10 = (high - low).rolling(10, min_periods=5).mean()
    out["range_expansion_10"] = _safe_div((high - low) - range_10, range_10)

    for column in ("open", "high", "low", "close", "volume"):
        out[column] = df[column].astype(float)

    return out


def _resample_sentiment(news: pd.DataFrame, rule: str) -> pd.DataFrame:
    if news.empty or "ts" not in news.columns:
        return pd.DataFrame()
    indexed = news.set_index("ts").sort_index()
    rule_map = {"1h": "1h", "24h": "1D", "7d": "7D"}
    frame = indexed.resample(rule_map.get(rule, rule)).agg(
        sentiment=("sentiment_score", "mean"),
        volume=("uid", "count"),
        negative=("is_negative", "sum"),
        positive=("is_positive", "sum"),
    )
    return frame


def _align_to(frame: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill a coarsely resampled frame onto a finer time grid."""
    if frame.empty:
        return frame
    return frame.reindex(grid.union(frame.index)).ffill().reindex(grid)


def compute_sentiment_features(
    news: pd.DataFrame,
    fear_greed: pd.DataFrame | None = None,
    price_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Aggregate article sentiment onto a price time grid.

    All aggregations use ``closed='left'`` so a bar only ever sees articles
    published strictly before its close. Sentiment is carried forward between
    events, which matches how a live system would observe it.
    """
    frames: list[pd.DataFrame] = []

    # Sentiment is only ever consumed on the price grid, so build it there
    # directly. Deriving the grid from the article timestamps instead would
    # leave gaps, and a dict-of-Series constructor over per-window indexes has
    # to reindex the raw (non-unique) article index when two articles share a
    # timestamp, which raises "cannot reindex on an axis with duplicate labels".
    grid = pd.DatetimeIndex(price_index).drop_duplicates().sort_values() if price_index is not None else None

    if news is not None and not news.empty:
        news = news.copy()
        news["ts"] = pd.to_datetime(news["ts"], utc=True)
        news = news.sort_values("ts")

        raw_hourly = _resample_sentiment(news, "1h")
        if raw_hourly.empty:
            return pd.DataFrame()
        if grid is None:
            grid = raw_hourly.index

        hourly = _align_to(raw_hourly, grid)
        daily = _align_to(_resample_sentiment(news, "24h"), grid)
        weekly = _align_to(_resample_sentiment(news, "7d"), grid)

        sent_1h = hourly["sentiment"].ffill()
        sent_24h = daily["sentiment"].ffill()
        sent_7d = weekly["sentiment"].ffill()

        vol_1h = hourly["volume"].fillna(0.0)
        vol_24h = daily["volume"].ffill().fillna(0.0)

        frames.append(pd.DataFrame({
            "news_sentiment_1h": sent_1h,
            "news_sentiment_24h": sent_24h,
            "news_sentiment_7d": sent_7d,
            "news_volume_1h": vol_1h,
            "news_volume_24h": vol_24h,
            "news_sentiment_momentum_24h": sent_24h - sent_24h.shift(1),
            "news_negative_ratio_24h": _safe_div(daily["negative"], daily["volume"]),
            "news_positive_ratio_24h": _safe_div(daily["positive"], daily["volume"]),
            "news_ewma_sentiment": sent_1h.ewm(span=24, adjust=False, min_periods=1).mean(),
        }, index=grid))

    if fear_greed is not None and not fear_greed.empty:
        fg = fear_greed.copy()
        fg["ts"] = pd.to_datetime(fg["ts"], utc=True)
        fg = fg.sort_values("ts").set_index("ts")["value"].astype(float)
        fg = fg[~fg.index.duplicated(keep="last")].resample("1D").last().ffill()
        frame = pd.DataFrame({
            "fear_greed_index": fg,
            "fear_greed_change_7d": fg.diff(7),
        })
        frames.append(_align_to(frame, grid) if grid is not None else frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1).sort_index()
    if combined.index.has_duplicates:
        combined = combined[~combined.index.duplicated(keep="last")]
    if grid is not None:
        combined = combined.reindex(grid)
    return combined.fillna(0.0) if grid is not None else combined
    return combined


def compute_sentiment_features_by_category(
    news: pd.DataFrame, categories: Sequence[str], price_index: pd.DatetimeIndex
) -> dict[str, pd.DataFrame]:
    if news.empty or "category" not in news.columns:
        return {}
    out: dict[str, pd.DataFrame] = {}
    for category in categories:
        subset = news[news["category"] == category]
        if subset.empty:
            continue
        features = compute_sentiment_features(subset, price_index=price_index)
        for column in features.columns:
            features = features.rename(columns={column: f"{column}_{category}"})
        out[category] = features
    return out


def compute_onchain_features(onchain: pd.DataFrame, price_index: pd.DatetimeIndex) -> pd.DataFrame:
    if onchain is None or onchain.empty:
        return pd.DataFrame()
    frame = onchain.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    wide = frame.pivot_table(index="ts", columns="metric", values="value", aggfunc="last")
    wide = wide.sort_index().resample("1h").last().ffill()
    out = pd.DataFrame(index=wide.index)
    if "dominance_pct" in frame.columns:
        pass
    dominance = onchain[(onchain["metric"] == "dominance_pct") & (onchain["asset"] == "BTC")]
    if not dominance.empty:
        dom = dominance.set_index("ts")["value"].sort_index().resample("1h").last().ffill()
        out["btc_dominance"] = dom
    if "total_market_cap_usd" in wide.columns:
        out["total_market_cap_usd_log"] = np.log1p(wide["total_market_cap_usd"].clip(lower=0))
    if "total_volume_usd" in wide.columns:
        out["total_volume_usd_log"] = np.log1p(wide["total_volume_usd"].clip(lower=0))
    if "market_cap_change_24h_pct" in wide.columns:
        out["market_cap_change_24h_pct"] = wide["market_cap_change_24h_pct"]
    if "fee_median" in wide.columns:
        out["mempool_fee_median"] = wide["fee_median"]
    if "mempool_vsize" in wide.columns:
        out["mempool_vsize"] = wide["mempool_vsize"]
    if "hashrate_eh_s" in wide.columns:
        out["btc_hashrate_eh_s"] = wide["hashrate_eh_s"]
    # mempool.space names this metric difficulty_progressPercent.
    difficulty_progress = next(
        (c for c in wide.columns if c.endswith("difficulty_progressPercent")
         or c.endswith("difficulty_progress")),
        None,
    )
    if difficulty_progress is not None:
        out["btc_difficulty_progress"] = wide[difficulty_progress]
    if out.empty:
        return pd.DataFrame()
    return out.reindex(price_index.union(out.index)).ffill().reindex(price_index)


def sanitize_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    *,
    min_active_fraction: float = 0.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop non-finite values and useless columns, returning a clean matrix.

    ``min_active_fraction`` additionally drops columns that are (near) zero for
    more than ``1 - min_active_fraction`` of rows. Such columns are typically
    neutral-filled alignment artefacts — a sentiment aggregate over a window in
    which no article was published, forward-filled from a non-null sentinel
    elsewhere. They are not constant, so the existing std filter keeps them, and
    they spend a normalised input slot on noise. The threshold defaults to off
    so existing schemas and artifacts are unaffected unless pruning is asked
    for; the dropped names are returned so the caller can record the decision.
    """
    if frame.empty:
        return frame, list(feature_columns or [])

    columns = list(feature_columns) if feature_columns else [
        c for c in frame.columns
        if c not in {"ts", "symbol", "year", "month"}
    ]
    columns = [c for c in columns if c in frame.columns]

    clean = frame[columns].replace([np.inf, -np.inf], np.nan)
    all_nan = clean.columns[clean.isna().all()].tolist()
    columns = [c for c in columns if c not in all_nan]

    clean = clean[columns].ffill().fillna(0.0)
    std = clean.std()
    constant = std[(std == 0) | std.isna()].index.tolist()
    columns = [c for c in columns if c not in constant]

    if min_active_fraction > 0.0 and columns:
        clean = clean[columns]
        active = (clean != 0.0).sum() / max(1, len(clean))
        sparse = set(
            active[(active < float(min_active_fraction))].index.tolist()
        )
        columns = [c for c in columns if c not in sparse]

    if not columns:
        return clean.iloc[:, :0].astype(np.float32), columns

    return clean[columns].astype(np.float32), columns



def build_feature_frame(
    candles: pd.DataFrame,
    news: pd.DataFrame | None = None,
    fear_greed: pd.DataFrame | None = None,
    onchain: pd.DataFrame | None = None,
    *,
    min_active_fraction: float = 0.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the full feature matrix for one symbol and report kept columns."""
    if candles.empty:
        return pd.DataFrame(), []

    candles = candles.sort_values("ts").reset_index(drop=True)
    technical = compute_technical_features(candles)
    index = pd.DatetimeIndex(candles["ts"])
    technical.index = index

    parts: list[pd.DataFrame] = [technical]
    sentiment = compute_sentiment_features(news, fear_greed, price_index=index)
    if not sentiment.empty:
        parts.append(sentiment)
    if onchain is not None and not onchain.empty:
        onchain_features = compute_onchain_features(onchain, index)
        if not onchain_features.empty:
            parts.append(onchain_features)

    combined = pd.concat(parts, axis=1).reindex(index).ffill().fillna(0.0)
    combined = combined.replace([np.inf, -np.inf], np.nan)

    clean, columns = sanitize_features(combined, min_active_fraction=min_active_fraction)
    clean.insert(0, "ts", index)
    clean.insert(1, "symbol", candles["symbol"].iloc[0] if "symbol" in candles.columns else "UNKNOWN")
    return clean.reset_index(drop=True), columns


def make_targets(frame: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Add forward-looking targets; final ``horizon`` rows are dropped later.

    ``target_volatility`` is the **trailing** volatility of one-bar log returns
    scaled to ``sqrt(horizon)``, i.e. the expected size of this target under the
    current regime. It used to be a rolling standard deviation of the forward
    return itself, which is built from future prices: harmless while the column
    stayed out of the feature matrix, but a live look-ahead bomb the moment
    anything regressed on it or used it to normalise a target.
    """
    out = frame.copy()
    close = out["close"].astype(float)
    out["target_return"] = np.log(close.shift(-horizon) / close)
    out["target_price"] = close.shift(-horizon)
    out["target_direction"] = (out["target_return"] > 0).astype(float)
    per_bar = np.log(close.replace(0, np.nan)).diff().rolling(
        24, min_periods=4
    ).std(ddof=1)
    out["target_volatility"] = per_bar * float(np.sqrt(max(1, horizon)))
    return out


@dataclass(frozen=True)
class SplitBoundaries:
    """Row geometry of a purged, embargoed chronological split.

    The gaps are not cosmetic. A training row at index ``i`` carries a label
    covering ``(i, i + horizon]``, so without purging the last ``horizon`` train
    rows their labels read validation prices and the model is graded on data it
    already saw. Validation needs the same treatment because early stopping reads
    it: a val label reaching into the test window hands the test period to the
    early-stopping rule. The embargo is a second, deliberately conservative gap
    of ``seq_len`` bars covering the overlap between adjacent sliding windows,
    whose serial correlation otherwise leaks across the boundary.
    """

    train_start: int
    train_stop: int
    val_start: int
    val_stop: int
    test_start: int
    total: int
    purge: int
    embargo: int

    @property
    def n_train(self) -> int:
        return max(0, self.train_stop - self.train_start)

    @property
    def n_val(self) -> int:
        return max(0, self.val_stop - self.val_start)

    @property
    def n_test(self) -> int:
        return max(0, self.total - self.test_start)

    def as_dict(self) -> dict[str, int]:
        return {
            "train_start": self.train_start,
            "train_stop": self.train_stop,
            "val_start": self.val_start,
            "val_stop": self.val_stop,
            "test_start": self.test_start,
            "total": self.total,
            "purge": self.purge,
            "embargo": self.embargo,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
        }


def resolve_purge_embargo(
    *, horizon: int = 0, seq_len: int = 0, embargo_bars: int | None = None
) -> tuple[int, int]:
    """Pick the purge and embargo widths for a split.

    ``purge`` defaults to the label horizon and ``embargo`` to the window
    length, which are the two quantities that actually straddle a boundary.
    Callers that pass neither get a plain 70/15/15 split, so existing behaviour
    is preserved when no label/window information is available.
    """
    purge = max(0, int(horizon or 0))
    embargo = max(0, int(seq_len or 0)) if embargo_bars is None else max(0, int(embargo_bars))
    return purge, embargo


def split_boundaries(
    total: int,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    horizon: int = 0,
    seq_len: int = 0,
    embargo_bars: int | None = None,
) -> SplitBoundaries:
    """Row indices for a purged, embargoed split, with the gaps enforced.

    The embargo can consume the entire val or test block on short series, so both
    boundaries are clamped to keep every block non-empty whenever the raw ratios
    leave any room at all. A caller that asks for more gap than the data can
    spare gets a visibly smaller block rather than a silently mislabelled one.
    """
    total = max(0, int(total))
    purge, embargo = resolve_purge_embargo(
        horizon=horizon, seq_len=seq_len, embargo_bars=embargo_bars
    )
    gap = purge + embargo

    raw_train_stop = int(total * train_ratio)
    raw_val_stop = raw_train_stop + int(total * val_ratio)

    # Never let the gaps consume a block down to nothing while another block
    # still has rows: shrink the gap symmetrically in that case.
    min_val, min_test = 1, 1
    room = max(0, raw_val_stop - raw_train_stop) + max(0, total - raw_val_stop)
    if gap > 0 and room >= min_val + min_test and (raw_val_stop - raw_train_stop - gap) < min_val:
        gap = max(0, (raw_val_stop - raw_train_stop) - min_val)
    if gap > 0 and (total - raw_val_stop - gap) < min_test:
        gap = max(0, (total - raw_val_stop) - min_test)

    train_stop = max(0, min(total, raw_train_stop - gap))
    val_start = min(total, train_stop + gap)
    val_stop = max(val_start, min(total, raw_val_stop - gap))
    test_start = min(total, max(val_stop, val_stop + gap))

    return SplitBoundaries(
        train_start=0,
        train_stop=train_stop,
        val_start=val_start,
        val_stop=val_stop,
        test_start=test_start,
        total=total,
        purge=purge,
        embargo=embargo,
    )


def split_frame(
    frame: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    *,
    horizon: int = 0,
    seq_len: int = 0,
    embargo_bars: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split — never shuffle, to avoid temporal leakage.

    With ``horizon=0`` and ``seq_len=0`` this is the original contiguous
    70/15/15 split. Passing the label horizon and window length turns on
    purging and embargoing, which is what every real run should do: without
    them the tail of each block is scored against prices the model was trained
    on, and the reported test metrics are optimistic by an unknown margin.
    """
    bounds = split_boundaries(
        len(frame),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        horizon=horizon,
        seq_len=seq_len,
        embargo_bars=embargo_bars,
    )
    return (
        frame.iloc[bounds.train_start : bounds.train_stop].copy(),
        frame.iloc[bounds.val_start : bounds.val_stop].copy(),
        frame.iloc[bounds.test_start :].copy(),
    )


def feature_quality_report(
    frame: pd.DataFrame, feature_columns: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Per-column coverage and activity, to justify dropping weak features.

    A column that is zero (or neutral-filled) for 99% of rows carries almost no
    gradient signal while still consuming a model input slot and a normaliser
    slot, so it acts as noise dilution rather than information. Reporting the
    active fraction next to the zero fraction makes that decision auditable
    instead of a judgement call.
    """
    report: dict[str, dict[str, float]] = {}
    for column in feature_columns:
        if column not in frame.columns:
            continue
        values = frame[column].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            report[column] = {"rows": 0.0, "null_fraction": 1.0, "zero_fraction": 1.0,
                              "active_fraction": 0.0, "std": 0.0}
            continue
        zero_fraction = float(np.mean(np.isclose(finite, 0.0)))
        report[column] = {
            "rows": float(finite.size),
            "null_fraction": float(1.0 - finite.size / max(1, values.size)),
            "zero_fraction": zero_fraction,
            "active_fraction": float(1.0 - zero_fraction),
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        }
    return report
