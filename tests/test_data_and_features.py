"""Tests for storage, feature engineering and sentiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cqip.features import (
    build_feature_frame,
    compute_sentiment_features,
    compute_technical_features,
    make_targets,
    sanitize_features,
    split_boundaries,
    split_frame,
)
from cqip.labeling import (
    DIRECTION_COLUMN,
    TripleBarrierConfig,
    add_volatility_normalized_target,
    build_target_frame,
    directional_labels,
    make_triple_barrier_labels,
    trailing_volatility,
)
from cqip.sentiment import SentimentScorer, lexicon_sentiment
from cqip.storage import ParquetStore


class TestParquetStore:
    def test_write_and_read_roundtrip(self, store: ParquetStore, synthetic_candles):
        store.write("ohlcv", synthetic_candles, dedupe_on=("ts", "symbol", "exchange"), mode="overwrite")
        frame = store.read("ohlcv")

        assert len(frame) == len(synthetic_candles)
        assert set(["open", "high", "low", "close", "volume"]).issubset(frame.columns)
        assert pd.api.types.is_datetime64tz_dtype(frame["ts"])

    def test_upsert_deduplicates_on_key(self, store: ParquetStore, synthetic_candles):
        store.write("ohlcv", synthetic_candles, dedupe_on=("ts", "symbol", "exchange"), mode="overwrite")
        original = store.row_count("ohlcv")

        modified = synthetic_candles.copy()
        modified["close"] = modified["close"] * 1.01
        store.write("ohlcv", modified, dedupe_on=("ts", "symbol", "exchange"), mode="upsert")

        assert store.row_count("ohlcv") == original
        reread = store.read("ohlcv")
        assert np.allclose(reread["close"].to_numpy(), modified["close"].to_numpy())

    def test_symbol_filtering(self, store: ParquetStore, synthetic_candles):
        other = synthetic_candles.copy()
        other["symbol"] = "ETH_USDT"
        combined = pd.concat([synthetic_candles, other], ignore_index=True)
        store.write("ohlcv", combined, dedupe_on=("ts", "symbol", "exchange"), mode="overwrite")

        btc = store.read("ohlcv", filters=[("symbol", "==", "BTC_USDT")])
        assert set(btc["symbol"].unique()) == {"BTC_USDT"}
        assert sorted(store.symbols("ohlcv")) == ["BTC_USDT", "ETH_USDT"]

    def test_time_range_filter(self, store: ParquetStore, synthetic_candles):
        store.write("ohlcv", synthetic_candles, dedupe_on=("ts", "symbol", "exchange"), mode="overwrite")
        start = synthetic_candles["ts"].iloc[100]
        end = synthetic_candles["ts"].iloc[200]

        frame = store.read("ohlcv", start=start, end=end)
        assert not frame.empty
        assert frame["ts"].min() >= start
        assert frame["ts"].max() < end

    def test_empty_dataset_returns_empty_frame(self, store: ParquetStore):
        assert store.read("nonexistent").empty
        assert store.row_count("nonexistent") == 0
        assert store.columns("nonexistent") == []
        assert not store.exists("nonexistent")

    def test_upsert_reports_only_new_rows(self, store: ParquetStore, synthetic_candles):
        """Re-ingesting identical data must report 0 inserted, not the dataset size."""
        store.write("ohlcv", synthetic_candles, dedupe_on=("ts", "symbol", "exchange"), mode="overwrite")
        again = synthetic_candles.copy()
        written = store.write("ohlcv", again, dedupe_on=("ts", "symbol", "exchange"), mode="upsert")
        assert written == 0

        fresh = synthetic_candles.iloc[:10].copy()
        written = store.write("ohlcv", fresh, dedupe_on=("ts", "symbol", "exchange"), mode="upsert")
        assert written == 0, "rows already stored must not be counted as new"

        extended = pd.concat([synthetic_candles, synthetic_candles.iloc[[0]].assign(ts=pd.Timestamp("2030-01-01", tz="UTC"))], ignore_index=True)
        written = store.write("ohlcv", extended, dedupe_on=("ts", "symbol", "exchange"), mode="upsert")
        assert written == 1, "only the genuinely new row counts"

    def test_upsert_accepts_aware_datetimes_against_stored_data(
        self, store: ParquetStore, synthetic_candles
    ):
        """Regression: mixing aware/naive timestamps must not raise.

        SQLite drops tzinfo, so a value read back can be naive while a pandas
        timestamp passed in is aware. Comparing the two used to raise
        TypeError deep inside the upsert.
        """
        store.write("ohlcv", synthetic_candles, dedupe_on=("ts", "symbol", "exchange"), mode="overwrite")
        reread = store.read("ohlcv")
        assert pd.api.types.is_datetime64tz_dtype(reread["ts"])

        store.write("ohlcv", reread, dedupe_on=("ts", "symbol", "exchange"), mode="upsert")
        assert store.row_count("ohlcv") == len(synthetic_candles)

    def test_upsert_preserves_partition_layout(self, store: ParquetStore, synthetic_candles):
        """Regression: incremental upserts must not dump rows into a null partition.

        `read` strips the year/month partition columns, so merging stored rows
        with incoming rows used to null them out and write the bulk of the
        dataset into __HIVE_DEFAULT_PARTITION__ on every incremental write.
        """
        first_half = synthetic_candles.iloc[:450]
        second_half = synthetic_candles.iloc[450:]

        store.write("ohlcv", first_half, dedupe_on=("ts", "symbol", "exchange"), mode="overwrite")
        years = {p.parent.parent.name for p in store.dataset_path("ohlcv").rglob("*.parquet")}
        assert years == {"year=2024"}

        store.write("ohlcv", second_half, dedupe_on=("ts", "symbol", "exchange"), mode="upsert")

        years = {p.parent.parent.name for p in store.dataset_path("ohlcv").rglob("*.parquet")}
        assert years == {"year=2024"}, f"unexpected partitions: {years}"
        assert not any("HIVE_DEFAULT" in str(p) for p in store.dataset_path("ohlcv").rglob("*"))

        reread = store.read("ohlcv")
        assert len(reread) == len(synthetic_candles)
        assert reread["ts"].notna().all()
        assert reread["ts"].is_monotonic_increasing

    def test_atomic_write_leaves_no_partial_files(self, store: ParquetStore, synthetic_candles):
        store.write("ohlcv", synthetic_candles, dedupe_on=("ts", "symbol", "exchange"), mode="overwrite")
        stray = [p for p in store.root.rglob("*") if p.is_file() and "cqip_write_" in str(p)]
        assert not stray


class TestSentimentFeatures:
    def test_tolerates_duplicate_article_timestamps(self):
        """Regression: articles sharing a timestamp must not break assembly.

        A dict-of-Series constructor unions the per-window indexes and then
        reindexes the raw (non-unique) article index, which raised
        "cannot reindex on an axis with duplicate labels".
        """
        base = pd.Timestamp("2024-01-01", tz="UTC")
        news = pd.DataFrame(
            {
                "ts": [base, base, base + pd.Timedelta(hours=2), base + pd.Timedelta(hours=2)],
                "uid": ["a", "b", "c", "d"],
                "sentiment_score": [0.5, -0.5, 0.25, 0.25],
                "is_negative": [0.0, 1.0, 0.0, 0.0],
                "is_positive": [1.0, 0.0, 1.0, 1.0],
            }
        )
        grid = pd.date_range(base, periods=6, freq="1h", tz="UTC")

        features = compute_sentiment_features(news, price_index=grid)

        assert list(features.index) == list(grid)
        assert features.notna().all().all(), features
        assert features["news_sentiment_1h"].iloc[0] == pytest.approx(0.0)  # (0.5 + -0.5) / 2
        assert features["news_sentiment_1h"].iloc[3] == pytest.approx(0.25)
        assert features["news_ewma_sentiment"].iloc[-1] < 0.25  # EWMA lags the step up

    def test_fear_greed_aligns_to_price_grid(self):
        base = pd.Timestamp("2024-01-01", tz="UTC")
        fear_greed = pd.DataFrame(
            {
                "ts": [base, base + pd.Timedelta(days=1), base + pd.Timedelta(days=2)],
                "value": [25.0, 50.0, 75.0],
            }
        )
        grid = pd.date_range(base, periods=72, freq="1h", tz="UTC")  # 3 full days

        features = compute_sentiment_features(
            pd.DataFrame(), fear_greed=fear_greed, price_index=grid
        )

        assert list(features.index) == list(grid)
        assert features["fear_greed_index"].iloc[0] == 25.0  # Jan 1
        assert features["fear_greed_index"].iloc[23] == 25.0  # still Jan 1
        assert features["fear_greed_index"].iloc[24] == 50.0  # Jan 2 reading lands at midnight
        assert features["fear_greed_index"].iloc[48] == 75.0  # Jan 3 reading
        assert features["fear_greed_index"].iloc[-1] == 75.0
        # The 7-day change needs 7 prior daily readings, so it stays neutral
        # rather than leaking a partial-window diff into the features.
        assert (features["fear_greed_change_7d"] == 0.0).all()


class TestTechnicalFeatures:
    def test_all_expected_features_produced(self, synthetic_candles):
        features = compute_technical_features(synthetic_candles)
        expected = [
            "returns_1", "rsi_14", "macd_hist", "bb_percent_b", "atr_14_pct",
            "realized_vol_20", "zscore_close_20", "hurst_approx", "obv_slope_10",
            "momentum_20", "drawdown_60", "vwap_deviation",
        ]
        for column in expected:
            assert column in features.columns, f"missing {column}"

    def test_no_infinite_values(self, synthetic_candles):
        features = compute_technical_features(synthetic_candles)
        numeric = features.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.to_numpy()).any()

    def test_causality_no_lookahead(self, synthetic_candles):
        """Truncating the future must not change past feature values."""
        full = compute_technical_features(synthetic_candles)
        truncated_input = synthetic_candles.iloc[:500].copy()
        truncated = compute_technical_features(truncated_input)

        for column in ["returns_1", "rsi_14", "macd_hist", "bb_percent_b", "realized_vol_20"]:
            full_values = full[column].iloc[:450].to_numpy()
            truncated_values = truncated[column].iloc[:450].to_numpy()
            valid = ~(np.isnan(full_values) | np.isnan(truncated_values))
            assert np.allclose(full_values[valid], truncated_values[valid], atol=1e-8), (
                f"{column} leaks future information"
            )

    def test_rsi_bounded_zero_to_hundred(self, synthetic_candles):
        features = compute_technical_features(synthetic_candles)
        rsi = features["rsi_14"].dropna()
        assert rsi.between(0, 100).all()

    def test_volatility_positive(self, synthetic_candles):
        features = compute_technical_features(synthetic_candles)
        vol = features["realized_vol_20"].dropna()
        assert (vol > 0).all()

    def test_handles_short_input(self):
        minimal = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=30, freq="1h", tz="UTC"),
            "open": np.linspace(100, 110, 30), "high": np.linspace(101, 111, 30),
            "low": np.linspace(99, 109, 30), "close": np.linspace(100, 110, 30),
            "volume": np.full(30, 1000.0),
        })
        features = compute_technical_features(minimal)
        assert not features.empty


class TestTargets:
    def test_forward_return_is_correct(self, synthetic_candles):
        frame, _columns = build_feature_frame(synthetic_candles)
        frame = make_targets(frame, horizon=1)

        row = frame.iloc[10]
        expected = np.log(synthetic_candles["close"].iloc[11] / synthetic_candles["close"].iloc[10])
        assert row["target_return"] == pytest.approx(expected, rel=1e-5)

    def test_last_row_target_is_nan(self, synthetic_candles):
        frame, _ = build_feature_frame(synthetic_candles)
        frame = make_targets(frame, horizon=1)
        assert np.isnan(frame["target_return"].iloc[-1])

    def test_split_is_chronological_and_disjoint(self, synthetic_candles):
        frame, _ = build_feature_frame(synthetic_candles)
        train, val, test = split_frame(frame, 0.7, 0.15)

        assert len(train) + len(val) + len(test) == len(frame)
        assert train["ts"].iloc[-1] <= val["ts"].iloc[0]
        assert val["ts"].iloc[-1] <= test["ts"].iloc[0]

    def test_purge_keeps_train_labels_out_of_validation(self, synthetic_candles):
        """A train label must not read a validation price.

        The last `horizon` train rows carry labels that resolve inside the
        validation block, so dropping them is what stops the model being graded
        on data it has already seen.
        """
        horizon, seq_len = 8, 16
        frame, _ = build_feature_frame(synthetic_candles)
        bounds = split_boundaries(len(frame), horizon=horizon, seq_len=seq_len)
        train, val, test = split_frame(frame, horizon=horizon, seq_len=seq_len)

        assert bounds.purge >= horizon
        # Contiguous, non-overlapping blocks, with a gap at each boundary.
        assert bounds.train_stop <= bounds.val_start
        assert bounds.val_stop <= bounds.test_start
        gap = bounds.purge + bounds.embargo
        assert bounds.val_start - bounds.train_stop == gap
        assert bounds.test_start - bounds.val_stop == gap
        # The surviving train tail is a full horizon clear of the val block.
        assert bounds.val_start - bounds.train_stop >= horizon
        # Purging and embargoing removes rows; it must never invent them.
        assert len(train) + len(val) + len(test) == len(frame) - 2 * gap
        assert train["ts"].iloc[-1] < val["ts"].iloc[0]
        assert val["ts"].iloc[-1] < test["ts"].iloc[0]
        # The dropped rows are exactly the gap, sitting at each boundary.
        assert train["ts"].iloc[-1] < frame["ts"].iloc[bounds.train_stop]
        assert val["ts"].iloc[0] == frame["ts"].iloc[bounds.val_start]

    def test_split_geometry_is_reported_consistently(self, synthetic_candles):
        frame, _ = build_feature_frame(synthetic_candles)
        bounds = split_boundaries(len(frame), horizon=4, seq_len=8)
        reported = bounds.as_dict()

        assert reported["n_train"] == bounds.train_stop - bounds.train_start
        assert reported["n_val"] == bounds.val_stop - bounds.val_start
        assert reported["n_test"] == bounds.total - bounds.test_start
        assert reported["purge"] == bounds.purge
        assert reported["embargo"] == bounds.embargo


class TestTargetLabelling:
    def test_direction_column_is_binary_for_a_regression_target(self):
        """The aux head needs 0/1, and must exist for plain log-return targets.

        Feeding a {-1, 0, +1} triple-barrier label to binary cross-entropy is
        accepted silently and then optimised as if it were a probability.
        """
        frame = pd.DataFrame({
            "target_return": [0.01, -0.02, 0.0, 0.03, -0.01],
        })
        labels = directional_labels(frame["target_return"])

        assert set(labels.dropna().unique()) == {0.0, 1.0}
        assert labels.iloc[0] == 1.0
        assert labels.iloc[1] == 0.0
        # An exactly flat bar carries no direction and must not become a class.
        assert np.isnan(labels.iloc[2])

    def test_build_target_frame_always_emits_a_direction_column(self):
        frame = pd.DataFrame({
            "close": np.linspace(100.0, 130.0, 200),
            "high": np.linspace(101.0, 131.0, 200),
            "low": np.linspace(99.0, 129.0, 200),
            "target_return": np.r_[np.full(198, 0.001), [np.nan, np.nan]],
        })
        _out, spec = build_target_frame(frame, horizon=2, kind="log_return")

        assert spec.direction == DIRECTION_COLUMN
        assert spec.regression == "target_return"

    def test_volatility_normalised_target_scales_with_horizon(self):
        """Without the sqrt(horizon) term the 'normalised' target still learns the horizon."""
        rng = np.random.default_rng(7)
        returns = rng.normal(0.0, 0.01, 200)
        close = 100.0 * np.exp(np.cumsum(returns))
        target_return = np.r_[np.log(close[4:] / close[:-4]), [np.nan] * 4]
        frame = pd.DataFrame({"close": close, "target_return": target_return})

        one = add_volatility_normalized_target(frame, window=20, horizon=1, out="scaled_1")
        four = add_volatility_normalized_target(frame, window=20, horizon=4, out="scaled_4")

        # Dividing a scaled target by the raw one recovers 1/(sigma*sqrt(h)) for
        # that bar. The same bar must therefore differ between horizons by
        # exactly sqrt(4) - that is the term under test.
        ratio_one = (one["scaled_1"] / frame["target_return"]).replace([np.inf, -np.inf], np.nan)
        ratio_four = (four["scaled_4"] / frame["target_return"]).replace([np.inf, -np.inf], np.nan)
        valid = ratio_one.notna() & ratio_four.notna()

        assert valid.sum() > 50
        assert (ratio_one[valid] / ratio_four[valid]).median() == pytest.approx(2.0, rel=1e-9)

    def test_trailing_volatility_uses_no_future_data(self):
        """A label built at bar i must not move when the future is rewritten."""
        rng = np.random.default_rng(0)
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
        baseline = trailing_volatility(close, window=20)

        tampered = close.copy()
        tampered.iloc[200:] *= 5.0
        after = trailing_volatility(tampered, window=20)

        # Everything strictly before the tamper point must be identical.
        assert np.allclose(
            baseline.iloc[:200].to_numpy(), after.iloc[:200].to_numpy(), equal_nan=True
        )

    def _barrier_frame(self, *, low_multiplier: float) -> pd.DataFrame:
        """A frame whose bar 13 has a wide high and a configurable low.

        Bar 13 is the first *future* bar of label 12, which is the first bar the
        scan can touch, and it is well past the volatility warm-up.
        """
        rng = np.random.default_rng(3)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 80)))
        frame = pd.DataFrame({
            "close": close,
            "high": close * 1.001,
            "low": close * 0.999,
        })
        frame.loc[13, "high"] = close[13] * 1.08
        frame.loc[13, "low"] = close[13] * low_multiplier
        return frame

    def test_triple_barrier_assumes_stop_first_on_ambiguous_bars(self):
        """When one bar spans both barriers, the label must be the pessimistic one."""
        # 0.92x reaches the stop, 1.08x reaches the take-profit: genuinely ambiguous.
        out = make_triple_barrier_labels(
            self._barrier_frame(low_multiplier=0.92),
            TripleBarrierConfig(vol_window=8),
            horizon=6,
        )
        assert out["tb_label"].iloc[12] == -1.0
        assert out["tb_barrier"].iloc[12] == "stop_loss"
        assert out["tb_exit_bars"].iloc[12] == 1

    def test_triple_barrier_marks_a_take_profit_hit(self):
        """The ambiguous case must not be the only one the test can observe."""
        # 0.9995x never reaches the stop, so the take-profit must win.
        out = make_triple_barrier_labels(
            self._barrier_frame(low_multiplier=0.9995),
            TripleBarrierConfig(vol_window=8),
            horizon=6,
        )
        assert out["tb_label"].iloc[12] == 1.0
        assert out["tb_barrier"].iloc[12] == "take_profit"

    def test_triple_barrier_leaves_the_tail_unlabelled(self):
        frame = pd.DataFrame({
            "close": np.linspace(100.0, 120.0, 80),
            "high": np.linspace(101.0, 121.0, 80),
            "low": np.linspace(99.0, 119.0, 80),
        })
        out = make_triple_barrier_labels(frame, TripleBarrierConfig(vol_window=20), horizon=5)

        assert out["tb_return"].iloc[-1:].isna().all()
        assert out["tb_return"].notna().any()


class TestSanitize:
    def test_drops_constant_and_all_nan_columns(self):
        frame = pd.DataFrame({
            "good": [1.0, 2.0, 3.0, 4.0],
            "constant": [5.0, 5.0, 5.0, 5.0],
            "empty": [np.nan, np.nan, np.nan, np.nan],
            "infinite": [np.inf, 1.0, 2.0, -np.inf],
        })
        clean, columns = sanitize_features(frame)
        assert "constant" not in columns
        assert "empty" not in columns
        assert "good" in columns
        assert np.isfinite(clean.to_numpy()).all()


class TestSentiment:
    def test_lexicon_detects_direction(self):
        positive, _ = lexicon_sentiment("Bitcoin surges to record high after ETF approval")
        negative, _ = lexicon_sentiment("Exchange hacked as attackers steal millions in exploit")
        assert positive > 0
        assert negative < 0

    def test_lexicon_handles_negation(self):
        plain, _ = lexicon_sentiment("The upgrade is confirmed")
        negated, _ = lexicon_sentiment("The upgrade is not confirmed")
        assert negated <= plain

    def test_unknown_text_is_neutral(self):
        score, coverage = lexicon_sentiment("zzz qqq xyzzy")
        assert score == 0.0
        assert coverage == 0.0

    def test_scorer_falls_back_without_model_download(self, settings):
        scorer = SentimentScorer(settings)
        results = scorer.score_texts([
            "Bitcoin price crashes amid exchange hack",
            "Major exchange confirms successful security upgrade",
        ])
        assert len(results) == 2
        assert all({"sentiment_score", "sentiment_confidence"} <= set(r) for r in results)

    def test_score_frame_adds_columns(self, settings):
        scorer = SentimentScorer(settings)
        news = pd.DataFrame({
            "title": ["Bitcoin surges past resistance", "Exchange hacked, funds stolen"],
            "summary": ["Bullish news for crypto markets", "Security incident reported"],
        })
        scored = scorer.score_frame(news)
        for column in ("sentiment_score", "is_positive", "is_negative"):
            assert column in scored.columns
        assert scored["sentiment_score"].iloc[0] > scored["sentiment_score"].iloc[1]

class TestCoverageTracking:
    def test_upsert_widens_range_with_aware_datetimes(self, engine, synthetic_candles):
        """Regression: SQLite returns naive timestamps, breaking aware comparisons.

        Feeding a second, later window through upsert_coverage used to raise
        "can't compare offset-naive and offset-aware datetimes" and silently
        mark the whole ingestion run as failed.
        """
        from cqip.db import get_coverage, upsert_coverage

        first = synthetic_candles["ts"].iloc[0]
        second = synthetic_candles["ts"].iloc[500]
        third = synthetic_candles["ts"].iloc[-1]

        upsert_coverage(engine, "candles", "BTC_USDT", "1h", "gate", first, second, 500)
        upsert_coverage(engine, "candles", "BTC_USDT", "1h", "gate", second, third, 400)

        rows = get_coverage(engine, "candles")
        assert len(rows) == 1
        record = rows[0]
        assert record.start == first
        assert record.end == third
        assert record.start.tzinfo is not None, "datetimes must come back tz-aware"
        assert record.end.tzinfo is not None

    def test_coverage_is_tracked_per_exchange(self, engine, synthetic_candles):
        from cqip.db import get_coverage, upsert_coverage

        ts = synthetic_candles["ts"]
        upsert_coverage(engine, "candles", "BTC_USDT", "1h", "gate", ts.iloc[0], ts.iloc[-1], 900)
        upsert_coverage(engine, "candles", "BTC_USDT", "1h", "htx", ts.iloc[0], ts.iloc[400], 400)

        providers = {row.provider for row in get_coverage(engine, "candles")}
        assert providers == {"gate", "htx"}
