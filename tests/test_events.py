"""Tests for event labelling, selective evaluation, and conviction trading.

The bias these tests are written against: an event system that *looks* selective
while carrying no information. A sweep over conviction thresholds will always
produce a row that looks great on the data it was chosen from, so the tests
below pin down the behaviours that keep that illusion from being reported as an
edge -- interval width at thin counts, rejection of noise, censoring of the
ambiguous middle band, and validation-only threshold selection.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from cqip import events as E
from cqip.backtest import (
    BacktestEngine,
    conviction_threshold_sweep,
)
from cqip.features import build_feature_frame, make_targets
from cqip.labeling import (
    EVENT_COLUMN,
    build_target_frame,
    make_move_event_label,
)


def _assert_labels(actual, expected) -> None:
    """Compare label vectors treating NaN as equal to NaN.

    Censored rows are the whole point of these labels, so the tests must assert
    on them -- and ``nan != nan`` under plain ``==`` would make every
    censored-middle-band assertion fail for the wrong reason.
    """
    np.testing.assert_array_equal(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    )


def _noisy_signal(seed: int, n: int = 4000, strength: float = 0.9):
    """Scores with a real monotonic relationship to the event labels."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1, n)
    events = rng.random(n) < 1.0 / (1.0 + np.exp(-(strength * signal)))
    return signal, events


class TestWilsonInterval:
    def test_thin_counts_produce_a_wide_interval(self):
        low, high = E._wilson_interval(3, 4)
        # 75% observed, but the honest interval spans roughly a third to
        # near-certain -- which is why the count has to be reported next to it.
        assert low < 0.35
        assert high > 0.95
        assert low < 0.75 < high

    def test_more_evidence_narrows_the_interval(self):
        narrow_low, narrow_high = E._wilson_interval(750, 1000)
        assert narrow_low > 0.70
        assert narrow_high < 0.78
        assert (narrow_high - narrow_low) < 0.10

    def test_zero_denominator_is_nan_not_zero(self):
        low, high = E._wilson_interval(0, 0)
        assert np.isnan(low) and np.isnan(high)

    def test_interval_stays_inside_unit_range(self):
        low, high = E._wilson_interval(0, 5)
        assert low == 0.0
        assert high < 1.0
        low, high = E._wilson_interval(5, 5)
        assert high == 1.0


class TestPrecisionAtK:
    def test_perfect_ranking_recovers_every_event(self):
        scores = np.arange(100, dtype=float)
        events = np.zeros(100, dtype=bool)
        events[90:] = True  # events sit at the top of the ranking
        result = E.precision_at_k(scores, events, 0.1)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["lift"] > 1.0
        assert result["n_taken"] == 10

    def test_inverted_ranking_is_worse_than_random(self):
        scores = np.arange(100, dtype=float)
        events = np.zeros(100, dtype=bool)
        events[:10] = True  # events sit at the *bottom* of the ranking
        result = E.precision_at_k(scores, events, 0.1)
        assert result["precision"] == pytest.approx(0.0)
        assert result["lift"] == pytest.approx(0.0)
        assert result["significant"] is False

    def test_lift_is_relative_to_base_rate_not_absolute(self):
        # Identical precision of 0.4 at identical 20% coverage, against base
        # rates of 10% and 90%. Reporting precision alone would make these look
        # like the same result; they differ by 9x in lift.
        scores = -np.arange(100, dtype=float)  # low score == high rank
        rare_events = np.zeros(100, dtype=bool)
        rare_events[:8] = True
        rare_events[90:92] = True  # 10 of 100 -> base rate 0.10
        common_events = np.zeros(100, dtype=bool)
        common_events[:8] = True
        common_events[20:] = True  # 88 of 100 -> base rate 0.88

        rare = E.precision_at_k(scores, rare_events, 0.2)
        common = E.precision_at_k(scores, common_events, 0.2)
        assert rare["precision"] == common["precision"] == pytest.approx(0.4)
        assert rare["lift"] == pytest.approx(4.0)
        assert common["lift"] == pytest.approx(0.4 / 0.88)

    def test_constant_score_ties_break_deterministically(self):
        scores = np.full(100, 1.0)
        events = np.arange(100) < 20
        first = E.precision_at_k(scores, events, 0.2)
        second = E.precision_at_k(scores, events, 0.2)
        assert first == second

    def test_k_outside_unit_interval_raises(self):
        with pytest.raises(ValueError, match="k must be"):
            E.precision_at_k(np.arange(10), np.arange(10) < 5, 0.0)
        with pytest.raises(ValueError, match="k must be"):
            E.precision_at_k(np.arange(10), np.arange(10) < 5, 1.5)

    def test_nan_scores_are_dropped_not_treated_as_low_confidence(self):
        scores = np.array([5.0, 4.0, np.nan, 3.0, 2.0, 1.0])
        events = np.array([True, True, True, True, True, False])
        result = E.precision_at_k(scores, events, 1.0)
        # A NaN score has no opinion, so its event must not be counted either
        # way -- otherwise dropping the score would inflate or deflate lift.
        assert result["n"] == 5
        assert result["n_events"] == 4

    def test_empty_input_is_safe(self):
        result = E.precision_at_k([], [], 0.1)
        assert result["n"] == 0
        assert result["precision"] is None
        assert result["significant"] is False


class TestSelectivityCurve:
    def test_omits_slices_too_thin_to_interpret(self):
        # 200 bars over 40 coverage levels: the first two levels hold 5 and 10
        # observations, so the 5-bar slice must be dropped rather than reported
        # as a spectacular 100%-precision point.
        scores, events = _noisy_signal(3, n=200)
        curve = E.selectivity_curve(scores, events, n_points=40, min_taken=10)
        assert all(p["n_taken"] >= 10 for p in curve)
        assert len(curve) < 40

    def test_precision_decays_as_coverage_grows(self):
        scores, events = _noisy_signal(4)
        curve = E.selectivity_curve(scores, events, n_points=10, min_taken=10)
        precisions = [p["precision"] for p in curve]
        assert precisions == sorted(precisions, reverse=True)

    def test_empty_input_returns_empty_curve(self):
        assert E.selectivity_curve([], []) == []


class TestEventReportMultiplicity:
    def test_pure_noise_is_reported_as_no_edge(self):
        # The critical regression: sweeping ~25 coverage levels and taking the
        # best finds a "significant" result by chance almost every time. This
        # test failed loudly before the Bonferroni + contiguity correction.
        rng = np.random.default_rng(11)
        flagged = 0
        for _ in range(60):
            scores = rng.normal(0, 1, 3000)
            events = rng.random(3000) < 0.3
            if E.event_report(scores, events)["verdict"].startswith("candidate-edge"):
                flagged += 1
        assert flagged / 60 <= 0.10, f"{flagged}/60 noise trials claimed an edge"

    def test_real_signal_is_detected(self):
        scores, events = _noisy_signal(7)
        report = E.event_report(scores, events)
        assert report["verdict"].startswith("candidate-edge")
        assert report["significant_k"]
        assert len(report["significant_k"]) >= 2  # a plateau, not a lone spike

    def test_isolated_spike_is_not_enough(self):
        points = [
            {"k": 0.1, "significant": False},
            {"k": 0.2, "significant": True},
            {"k": 0.3, "significant": False},
            {"k": 0.4, "significant": True},
            {"k": 0.5, "significant": False},
        ]
        assert E._longest_true_run(points) == []

    def test_adjacent_run_is_accepted(self):
        points = [
            {"k": 0.1, "significant": True},
            {"k": 0.2, "significant": True},
            {"k": 0.3, "significant": False},
        ]
        assert len(E._longest_true_run(points)) == 2

    def test_bonferroni_widens_the_interval(self):
        single = E._bonferroni_z(0.05, 1)
        many = E._bonferroni_z(0.05, 25)
        assert many > single

    def test_window_without_events_is_named(self):
        scores, _ = _noisy_signal(8)
        report = E.event_report(scores, np.zeros(4000, dtype=bool))
        assert report["verdict"] == "no-events-in-window"
        assert report["n_events"] == 0

    def test_empty_window_is_named(self):
        assert E.event_report([], [])["verdict"] == "no-scored-bars"

    def test_event_rate_matches_labels(self):
        scores, events = _noisy_signal(9)
        report = E.event_report(scores, events)
        assert report["event_rate"] == pytest.approx(events.mean())


class TestCalibration:
    def test_brier_rewards_accurate_probabilities(self):
        events = np.array([True, False, True, False])
        good = np.array([0.9, 0.1, 0.8, 0.2])
        bad = np.array([0.55, 0.45, 0.55, 0.45])
        assert E.brier_score(good, events) < E.brier_score(bad, events)

    def test_brier_rejects_a_ranking_score(self):
        # A z-score is not a probability; scoring one as if it were produces a
        # confident-looking number that means nothing.
        with pytest.raises(ValueError, match="not a probability"):
            E.brier_score(np.array([-2.0, 0.5, 3.0]), np.array([True, False, True]))

    def test_brier_of_empty_input_is_none(self):
        assert E.brier_score([], []) is None

    def test_calibration_table_recovers_a_known_rate(self):
        rng = np.random.default_rng(5)
        probability = rng.uniform(0, 1, 200_000)
        events = rng.random(200_000) < probability
        rows = E.calibration_table(probability, events, bins=10)
        for row in rows:
            assert row["observed"] == pytest.approx(row["predicted"], abs=0.03)

    def test_calibration_table_rejects_out_of_range(self):
        assert E.calibration_table([2.0, 3.0], [True, False]) == []

    def test_calibration_covers_the_full_probability_range(self):
        rows = E.calibration_table([0.0, 0.5, 1.0], [True, True, True], bins=2)
        assert len(rows) == 2
        assert rows[0]["lower"] == 0.0
        assert rows[-1]["upper"] == 1.0


class TestMoveEventLabel:
    def test_absolute_threshold_censors_the_middle_band(self):
        returns = pd.Series([0.05, 0.001, -0.04, 0.0])
        label = make_move_event_label(
            pd.DataFrame({"target_return": returns}), threshold=0.02
        )
        _assert_labels(label, [1.0, np.nan, 0.0, np.nan])

    def test_censoring_is_not_silently_becoming_a_negative_class(self):
        # A +0.1% day is not a failure to reach +2%; treating it as one teaches
        # the model to predict the noise between thresholds.
        returns = pd.Series([0.001] * 10)
        label = make_move_event_label(
            pd.DataFrame({"target_return": returns}), threshold=0.02
        )
        assert label.isna().all()
        assert (label == 0).sum() == 0

    def test_standardized_threshold_scales_with_symbol_volatility(self):
        frame = pd.DataFrame({
            "target_return": [0.02, -0.02, 0.02, -0.02],
            "target_volatility": [0.01, 0.01, 0.10, 0.10],
        })
        label = make_move_event_label(frame, k=2.0)
        # The same 2% move is a 2-sigma event for the calm symbol and a
        # 0.2-sigma nothing for the volatile one.
        _assert_labels(label, [1.0, 0.0, np.nan, np.nan])

    def test_spec_records_the_event_parameters(self):
        frame = pd.DataFrame({
            "close": [100.0, 101.0, 99.0, 100.0],
            "target_return": [0.02, -0.02, 0.001, 0.0],
            "target_volatility": [0.01, 0.01, 0.01, 0.01],
        })
        _, spec = build_target_frame(frame, horizon=1, kind="move_event", move_event_k=2.0)
        assert spec.kind == "move_event"
        assert spec.direction == EVENT_COLUMN
        assert spec.detail["k"] == 2.0
        assert spec.detail["threshold"] is None

    def test_event_target_is_not_overwritten_by_the_sign_label(self):
        # Regression stays the raw return so every bar remains usable, but the
        # direction column must remain the *thresholded* event. Clobbering it
        # with sign(target_return) would silently reduce the event problem back
        # to the plain direction problem.
        frame = pd.DataFrame({
            "close": [100.0, 101.0, 99.0, 100.0],
            "target_return": [0.02, -0.02, 0.001, 0.0],
            "target_volatility": [0.01, 0.01, 0.01, 0.01],
        })
        out, spec = build_target_frame(
            frame, horizon=1, kind="move_event", move_event_threshold=0.02
        )
        assert out[EVENT_COLUMN].isna().sum() == 2
        assert spec.direction == EVENT_COLUMN
        assert "target_direction_binary" not in out.columns

    def test_vol_normalized_target_still_materialises_direction(self):
        # The volatility scaler needs a long enough history for its rolling
        # window to produce a value at all; five rows yields an all-NaN scale.
        n = 200
        rng = np.random.default_rng(13)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        frame = pd.DataFrame({
            "close": close,
            "target_return": np.roll(rng.normal(0, 0.01, n), -1),
            "target_volatility": np.full(n, 0.01),
        })
        out, spec = build_target_frame(frame, horizon=1, normalize_volatility=True)
        assert spec.direction == "target_direction_binary"
        assert "target_direction_binary" in out.columns
        assert out["target_direction_binary"].notna().any()


class TestConvictionStrategy:
    def test_flat_below_the_threshold(self, settings, synthetic_candles):
        engine = BacktestEngine(settings)
        scores = np.full(len(synthetic_candles), 0.4)
        result = engine.run_conviction_strategy(synthetic_candles, scores, threshold=1.0)
        assert result["coverage"] == pytest.approx(0.0)
        assert result["metrics"]["trades"] == 0

    def test_engaged_above_the_threshold(self, settings, synthetic_candles):
        engine = BacktestEngine(settings)
        scores = np.full(len(synthetic_candles), 2.0)
        result = engine.run_conviction_strategy(synthetic_candles, scores, threshold=1.0)
        assert result["coverage"] == pytest.approx(1.0)
        assert result["metrics"]["trades"] > 0

    def test_zero_threshold_is_rejected(self, settings, synthetic_candles):
        # A conviction strategy with no bar to clear is just always-in-market;
        # accepting it silently would hide that from the caller.
        with pytest.raises(ValueError, match="threshold must be"):
            BacktestEngine(settings).run_conviction_strategy(
                synthetic_candles, np.ones(len(synthetic_candles)), threshold=0.0
            )

    def test_coverage_decreases_as_the_bar_rises(self, settings, synthetic_candles):
        engine = BacktestEngine(settings)
        scores = np.linspace(-3.0, 3.0, len(synthetic_candles))
        coverages = [
            engine.run_conviction_strategy(
                synthetic_candles, scores, threshold=t
            )["coverage"]
            for t in (0.5, 1.0, 2.0, 3.0)
        ]
        assert coverages == sorted(coverages, reverse=True)
        assert coverages[-1] == pytest.approx(0.0)

    def test_sign_follows_the_score_direction(self, settings, synthetic_candles):
        engine = BacktestEngine(settings)
        bullish = np.full(len(synthetic_candles), 2.0)
        bearish = np.full(len(synthetic_candles), -2.0)
        up = engine.run_conviction_strategy(synthetic_candles, bullish, threshold=1.0)
        down = engine.run_conviction_strategy(synthetic_candles, bearish, threshold=1.0)
        assert up["metrics"]["final_equity"] != pytest.approx(
            down["metrics"]["final_equity"]
        )

    def test_higher_bar_smaller_position(self, settings, synthetic_candles):
        engine = BacktestEngine(settings)
        scores = np.full(len(synthetic_candles), 1.5)
        low = engine.run_conviction_strategy(synthetic_candles, scores, threshold=1.0)
        high = engine.run_conviction_strategy(synthetic_candles, scores, threshold=3.0)
        # 1.5 against a bar of 1.0 is half size; against a bar of 3.0 it is flat.
        assert low["metrics"]["trades"] > 0
        assert high["metrics"]["trades"] == 0

    def test_leverage_is_reported_and_flagged(self, settings, synthetic_candles):
        engine = BacktestEngine(settings)
        scores = np.full(len(synthetic_candles), 2.0)
        unlevered = engine.run_conviction_strategy(synthetic_candles, scores, threshold=1.0)
        levered = engine.run_conviction_strategy(
            synthetic_candles, scores, threshold=1.0, leverage=2.0
        )
        assert levered["leverage"] == 2.0
        assert levered["leverage_capped"] is True
        assert unlevered["leverage_capped"] is False
        # Greater exposure must mean a bigger return, or `leverage` is a no-op.
        assert abs(levered["metrics"]["total_return"]) > abs(
            unlevered["metrics"]["total_return"]
        )

    def test_length_mismatch_raises(self, settings, synthetic_candles):
        with pytest.raises(ValueError, match="length mismatch"):
            BacktestEngine(settings).run_conviction_strategy(
                synthetic_candles, np.ones(5), threshold=1.0
            )


class TestConvictionThresholdSweep:
    @pytest.fixture
    def trend_window(self, settings):
        rng = np.random.default_rng(21)
        n = 800
        drift = np.concatenate([np.full(n // 2, 0.002), np.full(n - n // 2, -0.002)])
        close = 100.0 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, n)))
        return pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "open": close * (1.0 + rng.normal(0, 0.001, n)),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.uniform(1e3, 1e4, n),
        })

    def _momentum_score(self, frame: pd.DataFrame) -> np.ndarray:
        close = frame["close"].astype(float)
        momentum = np.log(close).diff(20)
        vol = np.log(close).diff().rolling(20, min_periods=5).std(ddof=1)
        return (
            (momentum / vol).fillna(0.0).replace([np.inf, -np.inf], 0.0).to_numpy()
        )

    def test_sweep_reports_every_threshold_with_a_benchmark(self, settings, trend_window):
        rows = conviction_threshold_sweep(
            trend_window, self._momentum_score(trend_window), settings=settings
        )
        assert len(rows) == 10
        for row in rows:
            assert "total_return" in row
            assert row["buy_hold_return"] is not None
            assert 0.0 <= row["coverage"] <= 1.0

    def test_baseline_row_is_always_in_market(self, settings, trend_window):
        rows = conviction_threshold_sweep(
            trend_window, self._momentum_score(trend_window), settings=settings
        )
        baseline = next(r for r in rows if r["threshold"] == 0.0)
        # The 0.0 row is the always-in comparison every other row is judged
        # against; it must not come back flat.
        assert baseline["coverage"] > 0.5
        assert baseline["effective_threshold"] > 0.0

    def test_coverage_falls_monotonically_with_the_bar(self, settings, trend_window):
        rows = conviction_threshold_sweep(
            trend_window, self._momentum_score(trend_window), settings=settings
        )
        coverages = [r["coverage"] for r in rows]
        assert coverages == sorted(coverages, reverse=True)

    def test_benchmark_is_identical_across_rows(self, settings, trend_window):
        rows = conviction_threshold_sweep(
            trend_window, self._momentum_score(trend_window), settings=settings
        )
        assert len({r["buy_hold_return"] for r in rows}) == 1

    def test_excess_return_is_against_the_same_window(self, settings, trend_window):
        rows = conviction_threshold_sweep(
            trend_window, self._momentum_score(trend_window), settings=settings
        )
        for row in rows:
            assert row["excess_return"] == pytest.approx(
                row["total_return"] - row["buy_hold_return"]
            )

    def test_length_mismatch_raises(self, settings, trend_window):
        with pytest.raises(ValueError, match="length mismatch"):
            conviction_threshold_sweep(trend_window, np.ones(3), settings=settings)


class TestMoveEventWiring:
    def test_settings_accept_the_move_event_target(self):
        from cqip.config import Settings

        assert Settings(forecaster_target="move_event").forecaster_target == "move_event"

    def test_unknown_target_is_still_rejected(self):
        from cqip.config import Settings

        with pytest.raises(ValueError, match="move_event"):
            Settings(forecaster_target="crystal_ball")

    def test_resolve_target_kind_preserves_move_event(self):
        # The registry restore path rewrites an unknown kind to log_return. If
        # it did that to `move_event`, a trained event model would be graded
        # against a different label than the one it was fitted on.
        from cqip.config import Settings
        from cqip.pipeline import resolve_target_kind

        kind, normalize = resolve_target_kind(
            Settings(), {"target": "move_event", "normalize_volatility": False}
        )
        assert kind == "move_event"
        assert normalize is False

    def test_move_event_regression_is_treated_as_log_space(self):
        # Its regression target is the raw forward log return, so it must be
        # exponentiated before compounding, exactly like `log_return`. If
        # `move_event` were missing from this set the prediction would be read
        # as a simple return and every metric would be quietly wrong.
        from cqip.pipeline import TrainingPipeline

        source = inspect.getsource(TrainingPipeline._evaluate_forecaster)
        assert '"move_event"' in source
        assert "log_space" in source

    def test_event_params_reach_the_dataset_builder(self):
        from cqip.pipeline import DatasetBuilder

        params = inspect.signature(DatasetBuilder.build).parameters
        assert "move_event_threshold" in params
        assert "move_event_k" in params

    def test_train_forecaster_exposes_event_params(self):
        from cqip.pipeline import TrainingPipeline

        params = inspect.signature(TrainingPipeline.train_forecaster).parameters
        assert "move_event_threshold" in params
        assert "move_event_k" in params


class TestConvictionOnRealFeatures:
    def test_event_target_builds_on_a_real_feature_frame(self, synthetic_candles):
        frame, _ = build_feature_frame(synthetic_candles)
        frame = make_targets(frame, horizon=1)
        out, spec = build_target_frame(
            frame, horizon=1, kind="move_event", move_event_threshold=0.05
        )
        assert spec.kind == "move_event"
        assert EVENT_COLUMN in out.columns
        # Every label is one of: long event, short event, or censored.
        assert set(out[EVENT_COLUMN].dropna().unique()).issubset({0.0, 1.0})

    def test_conviction_score_runs_over_feature_horizon(self, settings, synthetic_candles):
        frame, _ = build_feature_frame(synthetic_candles)
        frame = make_targets(frame, horizon=1).dropna(subset=["target_return"])
        score = frame["target_volatility"].to_numpy() * 0.5  # trailing vol is causal
        result = BacktestEngine(settings).run_conviction_strategy(
            frame, score, threshold=0.001
        )
        assert result["coverage"] > 0.0
        assert "max_drawdown" in result["metrics"]
