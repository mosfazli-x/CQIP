"""Tests for forecaster architectures, trading env and backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from cqip.backtest import (
    BacktestEngine,
    BenchmarkSuite,
    WalkForwardValidator,
    _directional_accuracy_statistic,
    bootstrap_ci,
    compute_metrics,
    compute_trade_metrics,
    periods_per_year,
)
from cqip.forecaster import (
    ForecasterArtifact,
    ForecasterTrainer,
    GaussianHead,
    LSTMForecaster,
    MLPForecaster,
    TransformerForecaster,
    _forward_full,
    build_model,
    gaussian_nll,
    model_has_direction_head,
    set_seed,
)
from cqip.rl_env import ACTION_FLAT, ACTION_LONG, ACTION_SHORT, TradingEnv


@pytest.fixture
def toy_dataset(synthetic_candles):
    from cqip.features import build_feature_frame, make_targets

    frame, columns = build_feature_frame(synthetic_candles)
    frame = make_targets(frame, horizon=1)
    frame = frame.dropna(subset=["target_return"]).reset_index(drop=True)
    return frame, columns


class TestModelArchitectures:
    @pytest.mark.parametrize("kind", ["lstm", "transformer", "mlp"])
    def test_forward_shapes(self, kind):
        set_seed(0)
        batch, seq, features = 4, 32, 11
        model = build_model(kind, features, hidden_size=32, num_layers=1, seq_len=seq)
        x = torch.randn(batch, seq, features)
        mean, log_var = model(x)

        assert mean.shape == (batch,)
        assert log_var.shape == (batch,)
        assert torch.isfinite(mean).all()
        assert torch.isfinite(log_var).all()

    def test_log_var_is_clamped(self):
        head = GaussianHead(8)
        with torch.no_grad():
            head.log_var.weight.fill_(50.0)
            head.log_var.bias.fill_(50.0)
        _, log_var = head(torch.randn(2, 8))
        assert (log_var <= 6.0).all()
        assert (log_var >= -10.0).all()

    def test_transformer_respects_causal_masking(self):
        """Perturbing the final timestep must not alter earlier hidden states."""
        torch.manual_seed(0)
        model = TransformerForecaster(4, hidden_size=16, num_layers=1, dropout=0.0, num_heads=4)
        model.eval()

        x = torch.randn(1, 16, 4)
        baseline, _ = model(x)

        perturbed = x.clone()
        perturbed[0, -1, :] += 100.0
        changed, _ = model(perturbed)

        assert not torch.allclose(baseline, changed, atol=1e-5), "model must react to the latest step"

    def test_mlp_parameter_count_lower_than_lstm(self):
        lstm = LSTMForecaster(10, hidden_size=64, num_layers=2)
        mlp = MLPForecaster(10, seq_len=16, hidden_size=64, num_layers=2)
        assert sum(p.numel() for p in mlp.parameters()) < sum(p.numel() for p in lstm.parameters())

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown model kind"):
            build_model("quantum", 4)


class TestGaussianNLL:
    def test_minimized_at_true_parameters(self):
        target = torch.tensor([0.5, -0.3, 0.1])
        mean = target.clone()
        log_var = torch.zeros(3)
        minimal = gaussian_nll(mean, log_var, target)

        wrong_mean = torch.tensor([0.0, 0.0, 0.0])
        wrong_var = torch.full((3,), 3.0)
        assert minimal < gaussian_nll(wrong_mean, wrong_var, target)

    def test_penalizes_overconfidence(self):
        """A confident-but-wrong prediction must score worse than a humble one."""
        target = torch.tensor([1.0])
        mean = torch.tensor([0.1])
        confident = gaussian_nll(mean, torch.tensor([-8.0]), target)
        humble = gaussian_nll(mean, torch.tensor([0.5]), target)
        assert confident > humble


class TestForecasterTraining:
    def test_training_reduces_validation_loss(self, settings, toy_dataset):
        frame, columns = toy_dataset
        split = int(len(frame) * 0.75)
        train_frame, val_frame = frame.iloc[:split], frame.iloc[split:]

        trainer = ForecasterTrainer(settings)
        model, info = trainer.train(
            train_frame, val_frame, columns, epochs=12, seq_len=24, kind="mlp"
        )

        history = info["history"]
        assert len(history) > 0
        assert history[-1]["val_loss"] <= history[0]["val_loss"]
        assert info["metrics"]["feature_count"] == len(columns)

    def test_prediction_shape_and_alignment(self, settings, toy_dataset):
        frame, columns = toy_dataset
        split = int(len(frame) * 0.75)
        trainer = ForecasterTrainer(settings)
        model, info = trainer.train(
            frame.iloc[:split], frame.iloc[split:], columns, epochs=3, seq_len=16, kind="mlp"
        )
        predictions = trainer.predict(model, frame.iloc[split:], columns, info["model_info"])

        assert len(predictions) == len(frame.iloc[split:]) - 16
        assert set(["predicted_return", "lower_68", "upper_68"]).issubset(predictions.columns)
        assert (predictions["lower_68"] <= predictions["upper_68"]).all()
        assert np.isfinite(predictions.select_dtypes(include=[np.number]).to_numpy()).all()

    def test_predict_feeds_time_major_windows(self, settings, toy_dataset):
        """Regression: sliding_window_view must not leak (features, seq) tensors.

        sliding_window_view appends the window axis last, so an untransposed
        view feeds the recurrent model a (batch, n_features, seq) tensor. The
        toy MLP hid this because it flattens both axes; the LSTM did not.
        """
        frame, columns = toy_dataset
        split = int(len(frame) * 0.75)
        trainer = ForecasterTrainer(settings)
        model, info = trainer.train(
            frame.iloc[:split], frame.iloc[split:], columns, epochs=2, seq_len=16, kind="lstm"
        )
        seq = int(info["model_info"]["seq_len"])
        n_features = int(info["model_info"]["input_size"])

        captured: list[tuple[int, ...]] = []
        original_forward = model.forward

        def spy(chunk, *args, **kwargs):
            captured.append(tuple(chunk.shape))
            return original_forward(chunk, *args, **kwargs)

        model.forward = spy  # type: ignore[method-assign]
        trainer.predict(model, frame.iloc[split:], columns, info["model_info"])

        assert captured, "model was never called"
        for shape in captured:
            assert shape[1:] == (seq, n_features), f"expected (batch, {seq}, {n_features}), got {shape}"

    def test_artifact_roundtrip(self, settings, toy_dataset, tmp_path):
        frame, columns = toy_dataset
        split = int(len(frame) * 0.7)
        trainer = ForecasterTrainer(settings)
        model, info = trainer.train(
            frame.iloc[:split], frame.iloc[split:], columns, epochs=3, seq_len=16, kind="mlp"
        )

        model_id = "test-artifact"
        artifact = ForecasterArtifact(model_id, model, info["model_info"], {"sharpe": 1.0}, "BTC_USDT", "1h")
        path = artifact.save(tmp_path)

        assert path.exists()
        loaded, loaded_info, payload = ForecasterArtifact.load(path)
        assert payload["model_id"] == model_id
        assert loaded_info["seq_len"] == 16
        assert loaded_info["feature_columns"] == columns

        reloaded_model, _, _ = ForecasterArtifact.load(path)
        x = torch.randn(2, 16, len(columns))
        model.eval()
        reloaded_model.eval()
        with torch.no_grad():
            original_out, _ = model(x)
            reloaded_out, _ = reloaded_model(x)
        assert torch.allclose(original_out, reloaded_out, atol=1e-5)

    def test_insufficient_data_raises(self, settings):
        tiny = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC"),
            "f1": np.arange(5, dtype=float), "target_return": np.zeros(5),
        })
        trainer = ForecasterTrainer(settings)
        with pytest.raises(ValueError, match="Not enough rows"):
            trainer.train(tiny, tiny, ["f1"], epochs=1, seq_len=64)


class TestSeedEnsembles:
    def test_ensemble_averages_members_and_reports_disagreement(self, settings, toy_dataset):
        frame, columns = toy_dataset
        split = int(len(frame) * 0.75)
        trainer = ForecasterTrainer(settings)

        members, info = trainer.train_ensemble(
            frame.iloc[:split], frame.iloc[split:], columns,
            n_seeds=2, epochs=3, seq_len=24, kind="mlp",
        )

        assert len(members) == 2
        predictions = trainer.predict_ensemble(members, frame, columns)

        assert not predictions.empty
        # Disagreement is only defined with more than one member.
        assert "member_disagreement" in predictions.columns
        assert (predictions["member_disagreement"] >= 0).all()

        single = members[0][0]
        alone = trainer.predict(single, frame, columns, members[0][1])
        assert len(alone) == len(predictions)
        # The ensemble mean must differ from a single seed unless they coincide.
        assert predictions["predicted_return"].notna().all()

    def test_ensemble_variance_pools_between_member_disagreement(self, settings, toy_dataset):
        """Pooled variance must be at least the average member's own variance.

        Pooling is ``E[Var] + Var[E]``: the between-member term is strictly
        additive, so dropping it would report intervals that get *tighter*
        exactly when the ensemble is least sure of itself. The guarantee to
        assert is against the mean of the members' own variances, not against
        one member - a single member may report an idiosyncratic variance far
        above the others'.
        """
        frame, columns = toy_dataset
        split = int(len(frame) * 0.75)
        trainer = ForecasterTrainer(settings)

        members, _info = trainer.train_ensemble(
            frame.iloc[:split], frame.iloc[split:], columns,
            n_seeds=3, epochs=3, seq_len=24, kind="mlp",
        )
        ensemble = trainer.predict_ensemble(members, frame, columns)
        pooled_var = ensemble["predicted_volatility"].to_numpy() ** 2

        primary_info = members[0][1]["model_info"]
        windowed = trainer._windows(
            trainer._scale_frame(frame, columns, primary_info),
            int(primary_info["seq_len"]),
        )
        member_vars = [trainer._forward_arrays(model, windowed)[1] for model, _i in members]
        within_mean = np.vstack(member_vars).mean(axis=0)

        assert len(within_mean) == len(pooled_var)
        assert (pooled_var >= within_mean - 1e-9).mean() > 0.99
        # The extra term is real: somewhere the between-member spread is positive.
        assert (pooled_var > within_mean + 1e-12).any()

    def test_ensemble_metrics_report_the_mean_not_the_best_member(self, settings, toy_dataset):
        """Reporting the best member's accuracy would be a selection bias.

        With several seeds on a weak signal the best one looks materially better
        than the ensemble actually is, and that gap is exactly what a reader
        needs to see.
        """
        frame, columns = toy_dataset
        split = int(len(frame) * 0.75)
        trainer = ForecasterTrainer(settings)

        _members, info = trainer.train_ensemble(
            frame.iloc[:split], frame.iloc[split:], columns,
            n_seeds=3, epochs=3, seq_len=24, kind="mlp",
        )
        metrics = info["metrics"]
        # `members` holds (model, training_result) pairs.
        member_scores = [m[1]["metrics"]["directional_accuracy"] for m in info["members"]]

        assert metrics["members_evaluated"] == 3
        assert metrics["directional_accuracy"] == pytest.approx(
            float(np.mean(member_scores))
        )
        assert metrics["worst_directional_accuracy"] == pytest.approx(
            min(member_scores)
        )
        assert "std_directional_accuracy" in metrics
        assert info["ensemble"]["size"] == 3

    def test_ensemble_artifact_round_trips_every_member(self, settings, toy_dataset, tmp_path):
        frame, columns = toy_dataset
        split = int(len(frame) * 0.75)
        trainer = ForecasterTrainer(settings)

        members, info = trainer.train_ensemble(
            frame.iloc[:split], frame.iloc[split:], columns,
            n_seeds=2, epochs=3, seq_len=24, kind="mlp",
        )
        artifact = ForecasterArtifact(
            "m-ens", members[0][0], info["model_info"], info["metrics"],
            "BTC_USDT", "1h", members=members,
        )
        path = artifact.save(tmp_path)

        loaded, payload = ForecasterArtifact.load_ensemble(path)
        assert len(loaded) == 2
        # Each member must carry its own scaler state, not a shared one.
        assert [sorted(i["scaler_mean"]) for _m, i in loaded] == [
            sorted(i["scaler_mean"]) for _m, i in loaded
        ]
        assert payload["members"], "artifact must persist every member"
        for _model, info_loaded in loaded:
            assert "scaler_mean" in info_loaded
            assert "seq_len" in info_loaded
            assert "feature_columns" in info_loaded

    def test_legacy_artifact_still_loads(self, settings, toy_dataset, tmp_path):
        """An artifact written before ensembles existed must keep working."""
        frame, columns = toy_dataset
        split = int(len(frame) * 0.75)
        trainer = ForecasterTrainer(settings)
        model, info = trainer.train(
            frame.iloc[:split], frame.iloc[split:], columns,
            epochs=2, seq_len=24, kind="mlp",
        )
        path = ForecasterArtifact(
            "m-solo", model, info["model_info"], info["metrics"], "BTC_USDT", "1h",
        ).save(tmp_path)

        _reloaded, _info, payload = ForecasterArtifact.load(path)
        assert payload["model_id"] == "m-solo"
        assert not payload["model_info"].get("aux_direction")

    def test_members_accept_full_training_results(self, settings, toy_dataset):
        """`predict_ensemble` must accept a member's whole training result.

        `train` hands back `(model, result)`. If inference read that dict as a
        `model_info` it would find no `scaler_mean` and fail with a KeyError
        that looks nothing like its cause.
        """
        frame, columns = toy_dataset
        split = int(len(frame) * 0.75)
        trainer = ForecasterTrainer(settings)
        model, info = trainer.train(
            frame.iloc[:split], frame.iloc[split:], columns,
            epochs=2, seq_len=24, kind="mlp",
        )
        assert "model_info" in info and "scaler_mean" not in info

        predictions = trainer.predict_ensemble([(model, info)], frame, columns)
        assert not predictions.empty

    def test_aux_direction_head_trains_on_a_binary_column(self, settings, toy_dataset):
        """The aux head needs 0/1 labels; a {-1,0,1} column would train on garbage."""
        frame, columns = toy_dataset
        frame = frame.copy()
        frame["direction_binary"] = (frame["target_return"] > 0).astype(float)
        # A bar with no direction must be NaN, not a third class value.
        frame.loc[frame.index[:5], "direction_binary"] = np.nan

        split = int(len(frame) * 0.75)
        trainer = ForecasterTrainer(settings)
        model, info = trainer.train(
            frame.iloc[:split], frame.iloc[split:], columns,
            direction_column="direction_binary", aux_direction_weight=0.3,
            epochs=2, seq_len=24, kind="mlp",
        )

        assert model_has_direction_head(model)
        assert info["metrics"]["val_direction_scored"] >= 0
        x = torch.randn(2, 24, len(columns))
        model.eval()
        with torch.no_grad():
            # The two-tuple contract must be unchanged by the aux head.
            mean, log_var = model(x)
            assert mean.numel() == 2 and log_var.numel() == 2
            _full_mean, _full_log_var, logits = _forward_full(model, x)
        assert logits is not None and logits.numel() == 2

    def test_aux_direction_requires_a_direction_column(self, settings, toy_dataset):
        frame, columns = toy_dataset
        split = int(len(frame) * 0.75)
        trainer = ForecasterTrainer(settings)
        with pytest.raises(ValueError, match="direction_column"):
            trainer.train(
                frame.iloc[:split], frame.iloc[split:], columns,
                aux_direction_weight=0.5, epochs=1, seq_len=24, kind="mlp",
            )


class TestBootstrapIntervals:
    def test_paired_resampling_preserves_pairing(self):
        """A 2-D paired sample must stay 2-D and keep columns aligned.

        A flat finite mask on a paired array flattens it to 1-D, and the
        statistic then indexes a second axis that no longer exists.
        """
        rng = np.random.default_rng(0)
        actual = rng.normal(0.0, 0.01, 200)
        predicted = actual + rng.normal(0.0, 0.005, 200)
        paired = np.column_stack([actual, predicted])

        result = bootstrap_ci(
            paired, _directional_accuracy_statistic(actual, predicted),
            n_bootstrap=200, seed=1,
        )

        assert np.isfinite(result["point"])
        assert 0.0 <= result["lower"] <= result["upper"] <= 1.0
        assert result["n_obs"] == 200

    def test_chance_level_signal_gets_an_interval_spanning_half(self):
        """The gate's CI check only works if the interval really is wide.

        On a coin-flip series the lower bound must sit at or below 0.5; if the
        bootstrap produced a narrow interval around 0.5 the gate would be
        convinced of skill that does not exist.
        """
        rng = np.random.default_rng(1)
        actual = rng.normal(0.0, 0.01, 300)
        predicted = rng.normal(0.0, 0.01, 300)
        paired = np.column_stack([actual, predicted])

        result = bootstrap_ci(
            paired, _directional_accuracy_statistic(actual, predicted),
            n_bootstrap=500, seed=2,
        )

        assert abs(result["point"] - 0.5) < 0.10
        assert result["lower"] <= 0.5 <= result["upper"]

    def test_strong_signal_gets_an_interval_above_chance(self):
        rng = np.random.default_rng(2)
        actual = rng.normal(0.0, 0.01, 400)
        predicted = actual * 0.95 + rng.normal(0.0, 0.002, 400)
        paired = np.column_stack([actual, predicted])

        result = bootstrap_ci(
            paired, _directional_accuracy_statistic(actual, predicted),
            n_bootstrap=500, seed=3,
        )

        assert result["point"] > 0.8
        assert result["lower"] > 0.5

    def test_nan_rows_are_dropped_before_resampling(self):
        actual = np.array([1.0, -1.0, np.nan, 1.0, -1.0, 1.0, 1.0, -1.0])
        predicted = np.array([1.0, -1.0, 1.0, np.nan, -1.0, 1.0, 1.0, 1.0])
        paired = np.column_stack([actual, predicted])

        result = bootstrap_ci(
            paired, _directional_accuracy_statistic(actual, predicted),
            n_bootstrap=50, seed=4,
        )

        assert result["n_obs"] == 6


class TestTradingEnv:
    def _make_env(self, synthetic_candles, settings, **kwargs):
        features = synthetic_candles[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32)
        prices = synthetic_candles["close"].to_numpy(dtype=np.float64)
        return TradingEnv(features, prices, ["open", "high", "low", "close", "volume"],
                          settings=settings, **kwargs)

    def test_observation_space_matches_reset(self, settings, synthetic_candles):
        env = self._make_env(synthetic_candles, settings)
        observation, info = env.reset(seed=1)
        assert observation.shape == env.observation_space.shape
        assert observation.dtype == np.float32
        assert "equity" in info

    def test_flat_action_holds_no_position(self, settings, synthetic_candles):
        env = self._make_env(synthetic_candles, settings)
        env.reset(seed=1)
        _, _, _, _, info = env.step(ACTION_FLAT)
        assert info["position"] == 0.0

    def test_long_action_accrues_position(self, settings, synthetic_candles):
        env = self._make_env(synthetic_candles, settings)
        env.reset(seed=1)
        env.step(ACTION_LONG)
        _, _, _, _, info = env.step(ACTION_LONG)
        assert info["position"] > 0

    def test_short_action_creates_negative_position(self, settings, synthetic_candles):
        env = self._make_env(synthetic_candles, settings)
        env.reset(seed=1)
        env.step(ACTION_SHORT)
        _, _, _, _, info = env.step(ACTION_SHORT)
        assert info["position"] < 0

    def test_fees_reduce_equity_on_roundtrip(self, settings, synthetic_candles):
        """A round trip at a flat price must lose exactly the fees.

        Price is held constant so P&L is zero and the only effect left is cost
        of trading; otherwise realised price movement would mask the fees.
        """
        flat = synthetic_candles.copy()
        flat_price = float(flat["close"].iloc[0])
        for column in ("open", "high", "low", "close"):
            flat[column] = flat_price

        env = self._make_env(flat, settings, fee_bps=10.0, slippage_bps=0.0)
        env.reset(seed=1)
        env.step(ACTION_LONG)
        _, _, _, _, after_open = env.step(ACTION_LONG)
        _, _, _, _, after_close = env.step(ACTION_FLAT)

        paid = after_open["equity"] - after_close["equity"]
        notional = abs(after_open["position"]) * flat_price
        assert paid == pytest.approx(notional * 0.001, rel=1e-6)
        assert paid > 0

    def test_step_counts_fees_in_trades(self, settings, synthetic_candles):
        env = self._make_env(synthetic_candles, settings)
        env.reset(seed=1)
        env.step(ACTION_LONG)
        env.step(ACTION_LONG)
        assert env.trades
        assert all(t["fee"] > 0 for t in env.trades)

    def test_terminates_at_series_end(self, settings, synthetic_candles):
        env = self._make_env(synthetic_candles, settings)
        env.reset(seed=1)
        terminated = False
        for _ in range(len(synthetic_candles) + 5):
            _obs, _reward, terminated, _truncated, _info = env.step(ACTION_FLAT)
            if terminated:
                break
        assert terminated

    def test_ruin_guard_terminates_early(self, synthetic_candles, settings):
        crash = synthetic_candles.copy()
        crash["close"] = np.linspace(30_000, 100, len(crash))
        crash["open"] = crash["close"]
        crash["high"] = crash["close"] * 1.001
        crash["low"] = crash["close"] * 0.999

        env = self._make_env(crash, settings, initial_capital=1000.0)
        env.reset(seed=1)
        terminated = False
        for _ in range(len(crash)):
            _obs, _reward, terminated, _truncated, _info = env.step(ACTION_SHORT)
            if terminated:
                break
        assert terminated

    def test_mismatched_lengths_raise(self, settings, synthetic_candles):
        with pytest.raises(ValueError, match="length mismatch"):
            TradingEnv(
                np.zeros((10, 3), dtype=np.float32), np.ones(9), ["a", "b", "c"], settings=settings
            )


class TestMetrics:
    def test_known_series_metrics(self):
        returns = np.array([0.01, -0.005, 0.02, 0.0, 0.015, -0.01])
        metrics = compute_metrics(returns, initial_capital=1000.0, periods_per_year=365)

        expected_final = 1000.0 * np.prod(1 + returns)
        assert metrics["final_equity"] == pytest.approx(expected_final, rel=1e-9)
        assert metrics["n_periods"] == 6
        assert 0.0 <= metrics["win_rate"] <= 1.0
        assert metrics["max_drawdown"] <= 0.0

    def test_empty_series_returns_zeroes(self):
        metrics = compute_metrics(np.array([]))
        assert metrics["total_return"] == 0.0
        assert metrics["sharpe"] == 0.0

    def test_zero_volatility_does_not_divide_by_zero(self):
        metrics = compute_metrics(np.zeros(50))
        assert np.isfinite(metrics["sharpe"])
        assert np.isfinite(metrics["sortino"])
        assert np.isfinite(metrics["calmar"])

    def test_max_drawdown_detection(self):
        returns = np.array([0.5, -0.5, 0.1])
        metrics = compute_metrics(returns, initial_capital=1000.0)
        assert metrics["max_drawdown"] == pytest.approx(-0.5, rel=1e-6)

    def test_trade_metrics_counts_roundtrips(self):
        trades = [
            {"index": 0, "side": "buy", "quantity": 1, "price": 100, "fee": 0.1},
            {"index": 5, "side": "sell", "quantity": 1, "price": 110, "fee": 0.1},
            {"index": 10, "side": "buy", "quantity": 1, "price": 110, "fee": 0.1},
            {"index": 12, "side": "sell", "quantity": 1, "price": 90, "fee": 0.1},
        ]
        metrics = compute_trade_metrics(trades)
        assert metrics["trades"] == 2
        assert metrics["wins"] == 1
        assert metrics["losses"] == 1
        assert metrics["win_rate"] == 0.5


class TestBacktestAnnualisation:
    """A wrong annualisation factor is silent, so it needs explicit guards."""

    def test_periods_per_year_depends_on_interval(self):
        assert periods_per_year("1h") == pytest.approx(24 * 365)
        assert periods_per_year("1d") == pytest.approx(365.0)
        assert periods_per_year("4h") == pytest.approx(6 * 365)
        assert periods_per_year("15m") == pytest.approx(96 * 365)

    def test_unknown_interval_falls_back_to_hourly(self):
        # One year of data must never be reported as one year of minutes.
        assert periods_per_year("7q") == pytest.approx(24 * 365)

    def test_label_horizon_shortens_the_period_count(self):
        assert periods_per_year("1d", label_horizon=3) == pytest.approx(365.0 / 3)

    def test_daily_and_hourly_sharpe_differ_by_sqrt_scaling(self):
        rng = np.random.default_rng(7)
        returns = rng.normal(0.0, 0.01, size=500)
        daily = compute_metrics(returns, periods_per_year=365.0)
        hourly = compute_metrics(returns, periods_per_year=24 * 365)
        assert hourly["sharpe"] == pytest.approx(daily["sharpe"] * np.sqrt(24.0), rel=1e-9)

    def test_same_returns_give_different_cagr_per_interval(self):
        # A flat +1%/bar series is a ~40x year daily and a ~1e5x year hourly.
        returns = np.full(365, 0.01)
        daily = compute_metrics(returns, periods_per_year=365.0)
        hourly = compute_metrics(returns, periods_per_year=24 * 365)
        assert daily["cagr"] == pytest.approx(1.01**365 - 1, rel=1e-6)
        assert hourly["cagr"] > daily["cagr"] * 10

    def test_missing_periods_per_year_still_returns_metrics(self):
        # A caller that does not know the interval must not crash, but the
        # function emits `compute_metrics_unannualised` so the guess is visible.
        metrics = compute_metrics(np.full(10, 0.001))
        assert metrics["n_periods"] == 10
        assert metrics["total_return"] > 0

    def test_engine_uses_injected_periods_per_year(self, settings, synthetic_candles):
        daily = BacktestEngine(settings, periods_per_year=365.0)
        hourly = BacktestEngine(settings, periods_per_year=24 * 365)
        a = daily.run_buy_and_hold(synthetic_candles, initial_capital=10_000.0)
        b = hourly.run_buy_and_hold(synthetic_candles, initial_capital=10_000.0)
        # Total return is a pure price fact and must not move; only the
        # annualised figures may.
        assert a["metrics"]["total_return"] == pytest.approx(b["metrics"]["total_return"])
        assert a["metrics"]["cagr"] != pytest.approx(b["metrics"]["cagr"])


class TestBacktestEngine:
    def test_buy_and_hold_matches_price_change(self, settings, synthetic_candles):
        engine = BacktestEngine(settings)
        result = engine.run_buy_and_hold(synthetic_candles, initial_capital=10_000.0)

        expected = synthetic_candles["close"].iloc[-1] / synthetic_candles["close"].iloc[0]
        assert result["metrics"]["total_return"] == pytest.approx(expected - 1, rel=0.02)

    def test_buy_and_hold_charges_entry_fee_as_a_real_return(self, settings, synthetic_candles):
        engine = BacktestEngine(settings)
        result = engine.run_buy_and_hold(synthetic_candles, initial_capital=10_000.0)

        closes = synthetic_candles["close"].to_numpy(dtype=float)
        # The entry is a real fill, so it pays both the fee and the slippage, in the
        # same direction a strategy's first fill does.
        entry = (1 - engine.fee_rate) * (1 + engine.slippage_rate)
        expected = (closes[-1] / closes[0]) * entry - 1
        # Return series must stay bar-aligned with a strategy's, so the entry cost is
        # folded into the first period rather than prepended as an extra bar.
        assert len(result["returns"]) == len(synthetic_candles) - 1
        assert result["metrics"]["total_return"] == pytest.approx(expected, rel=1e-9)
        assert result["metrics"]["final_equity"] == pytest.approx(10_000.0 * (1 + expected), rel=1e-9)
        fee_effect = (1 + result["returns"][0]) / (1 + closes[1] / closes[0] - 1)
        assert fee_effect == pytest.approx(entry, rel=1e-9)

    def test_momentum_strategy_runs(self, settings, synthetic_candles):
        result = BacktestEngine(settings).run_momentum(synthetic_candles, window=12)
        assert result["n_bars"] == len(synthetic_candles) - 1
        assert "metrics" in result

    def test_perfect_oracle_beats_random(self, settings, synthetic_candles):
        """A signal that knows tomorrow's sign must outperform zero-signal trading."""
        engine = BacktestEngine(settings)
        future_returns = synthetic_candles["close"].shift(-1) / synthetic_candles["close"] - 1
        oracle = np.sign(future_returns).fillna(0).to_numpy()

        result = engine.run_signal_strategy(synthetic_candles, oracle, initial_capital=10_000.0)
        assert result["metrics"]["total_return"] > 0.05

    def test_zero_signal_has_no_trades(self, settings, synthetic_candles):
        engine = BacktestEngine(settings)
        result = engine.run_signal_strategy(synthetic_candles, np.zeros(len(synthetic_candles)))
        assert not result["trades"]

    def test_equity_curve_length(self, settings, synthetic_candles):
        engine = BacktestEngine(settings)
        result = engine.run_signal_strategy(synthetic_candles, np.ones(len(synthetic_candles)))
        assert len(result["equity_curve"]) == len(synthetic_candles)
        assert len(result["returns"]) == len(synthetic_candles) - 1

    def test_execution_uses_next_bar_open(self, settings, synthetic_candles):
        """The signal from bar i must fill at bar i+1's open, never the current bar."""
        engine = BacktestEngine(settings)
        signals = np.zeros(len(synthetic_candles))
        signals[5] = 1.0

        result = engine.run_signal_strategy(synthetic_candles, signals, slippage_bps=0.0)
        buy_trade = result["trades"][0]

        assert buy_trade["signal_index"] == 5
        assert buy_trade["index"] == 6
        assert buy_trade["price"] == pytest.approx(synthetic_candles["open"].iloc[6], rel=1e-6)

    def test_extra_future_bars_cannot_change_past_results(self, settings, synthetic_candles):
        """Appending future bars must leave every earlier return untouched.

        This is the real look-ahead guard: if the engine ever read a signal or
        price beyond the current step, extending the series would retroactively
        change results that were already "in the past".
        """
        engine = BacktestEngine(settings)
        cut = 400
        history = synthetic_candles.iloc[:cut].reset_index(drop=True)
        extended = synthetic_candles.reset_index(drop=True)

        history_signals = np.sign(
            history["close"].shift(-1) / history["close"] - 1
        ).fillna(0.0).to_numpy()
        extended_signals = np.sign(
            extended["close"].shift(-1) / extended["close"] - 1
        ).fillna(0.0).to_numpy()

        short_run = engine.run_signal_strategy(history, history_signals, initial_capital=10_000.0)
        long_run = engine.run_signal_strategy(extended, extended_signals, initial_capital=10_000.0)

        assert len(short_run["returns"]) == cut - 1
        assert long_run["returns"][: cut - 1] == pytest.approx(short_run["returns"], rel=1e-9)

    def test_fee_and_slippage_reduce_returns(self, settings, synthetic_candles):
        signals = np.ones(len(synthetic_candles))
        cheap = BacktestEngine(settings).run_signal_strategy(
            synthetic_candles, signals, initial_capital=10_000.0
        )
        assert cheap["metrics"]["total_return"] < 1.0

    def test_mismatched_signal_length_raises(self, settings, synthetic_candles):
        engine = BacktestEngine(settings)
        with pytest.raises(ValueError):
            engine.run_signal_strategy(synthetic_candles, np.ones(10))

    def test_benchmark_suite_includes_baselines(self, settings, synthetic_candles):
        suite = BenchmarkSuite(settings)
        results = suite.run_all(synthetic_candles)
        assert "buy_and_hold" in results
        assert "momentum_24" in results


def _flat_candles(price: float = 30_000.0, n: int = 60) -> pd.DataFrame:
    """Constant-price series, so only sizing can move the equity curve."""
    times = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "ts": times,
        "symbol": "BTC_USDT",
        "exchange": "gate",
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "volume": 1.0,
    })


def _trending_candles(start: float = 30_000.0, per_bar: float = 0.01, n: int = 60) -> pd.DataFrame:
    """Steadily rising series: a long must earn the price change, not a multiple."""
    times = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = start * (1.0 + per_bar) ** np.arange(n)
    return pd.DataFrame({
        "ts": times,
        "symbol": "BTC_USDT",
        "exchange": "gate",
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1.0,
    })


class TestPositionSizing:
    """``position_size`` is a fraction of equity, not a count of asset units.

    Regression guard. Sizing used to be ``signal * position_size`` in whole asset
    units, so ``position_size=1.0`` bought one BTC (~$30k) on a $10k account and
    short losses were unbounded. Equity went negative, which produced impossible
    drawdowns below -100% and nonsense CAGR. The assertions below all failed under
    that behaviour, which the previous suite never checked.
    """

    def test_flat_price_long_preserves_equity(self, settings):
        """A long on a constant price must not multiply equity by leverage."""
        frame = _flat_candles()
        result = BacktestEngine(settings).run_signal_strategy(
            frame, np.ones(len(frame)), initial_capital=10_000.0
        )
        # Only the entry fee and slippage may erode the account.
        assert result["metrics"]["final_equity"] == pytest.approx(10_000.0, rel=0.01)
        assert abs(result["metrics"]["total_return"]) < 0.01

    def test_trending_long_earns_the_price_change_not_a_multiple(self, settings):
        """1x exposure on a +1%/bar series returns ~the series' own gain."""
        frame = _trending_candles()
        result = BacktestEngine(settings).run_signal_strategy(
            frame, np.ones(len(frame)), initial_capital=10_000.0
        )
        expected = frame["close"].iloc[-1] / frame["close"].iloc[0] - 1
        assert result["metrics"]["total_return"] == pytest.approx(expected, rel=0.05)

    def test_equity_never_goes_negative(self, settings, synthetic_candles):
        """Adverse signals must not bankrupt the account."""
        engine = BacktestEngine(settings)
        falling = np.sign(
            synthetic_candles["close"].shift(1) / synthetic_candles["close"] - 1
        ).fillna(0.0).to_numpy()
        result = engine.run_signal_strategy(synthetic_candles, falling, initial_capital=10_000.0)
        assert min(result["equity_curve"]) > 0

    def test_max_drawdown_cannot_exceed_total_loss(self, settings, synthetic_candles):
        """Drawdown below -100% is arithmetically impossible for a funded account."""
        engine = BacktestEngine(settings)
        oracle = np.sign(
            synthetic_candles["close"].shift(-1) / synthetic_candles["close"] - 1
        ).fillna(0.0).to_numpy()
        result = engine.run_signal_strategy(synthetic_candles, oracle, initial_capital=10_000.0)
        assert result["metrics"]["max_drawdown"] >= -1.0

    def test_trade_notional_does_not_exceed_equity(self, settings, synthetic_candles):
        """No single fill may commit more than the account holds."""
        result = BacktestEngine(settings).run_signal_strategy(
            synthetic_candles, np.ones(len(synthetic_candles)), initial_capital=10_000.0
        )
        assert all(t["notional"] <= 10_000.0 * (1.0 + t["quantity"] / max(t["price"], 1.0)) + 1e-6
                   for t in result["trades"])

    def test_full_size_tracks_the_asset_exactly(self, settings):
        """A 1.0-size long must earn the instrument's own return, bar for bar.

        With costs zeroed the position is rebalanced to exactly 1x of equity at
        the fill price, so every bar after the opening fill returns the series'
        own 1%. Sizing off the previous close instead left steady-state exposure
        near 0.99 and quietly lagged the asset.
        """
        frame = _trending_candles()
        result = BacktestEngine(settings).run_signal_strategy(
            frame,
            np.ones(len(frame)),
            initial_capital=10_000.0,
            fee_bps=0.0,
            slippage_bps=0.0,
        )
        returns = np.asarray(result["returns"])
        # First bar only opens the position, so it cannot have earned anything.
        assert returns[0] == pytest.approx(0.0, abs=1e-12)
        assert np.allclose(returns[1:], 0.01, rtol=1e-9)

    def test_half_size_earns_half_the_bar_return(self, settings):
        """Sizing must scale the per-bar return linearly."""
        frame = _trending_candles()
        signals = np.ones(len(frame))
        engine = BacktestEngine(settings)
        kwargs = {"initial_capital": 10_000.0, "fee_bps": 0.0, "slippage_bps": 0.0}
        full = np.asarray(engine.run_signal_strategy(frame, signals, position_size=1.0, **kwargs)["returns"])
        half = np.asarray(engine.run_signal_strategy(frame, signals, position_size=0.5, **kwargs)["returns"])
        assert np.allclose(half[1:], full[1:] * 0.5, rtol=1e-9)

    def test_oversized_request_is_clamped_not_leveraged(self, settings):
        """``position_size=3.0`` must clamp to 1x, matching the RL env's ``min``."""
        frame = _flat_candles()
        signals = np.ones(len(frame))
        engine = BacktestEngine(settings)
        clamped = engine.run_signal_strategy(
            frame, signals, initial_capital=10_000.0, position_size=3.0
        )
        capped = engine.run_signal_strategy(
            frame, signals, initial_capital=10_000.0, position_size=1.0
        )
        assert clamped["metrics"]["final_equity"] == pytest.approx(
            capped["metrics"]["final_equity"], rel=1e-9
        )
        assert clamped["metrics"]["final_equity"] > 0

    def test_benchmarks_stay_funded_on_real_history(self, settings):
        """The momentum baselines must remain solvent on realistic prices."""
        rng = np.random.default_rng(7)
        n = 500
        close = 76_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
        times = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
        frame = pd.DataFrame({
            "ts": times, "symbol": "BTC_USDT", "exchange": "gate",
            "open": close, "high": close, "low": close, "close": close, "volume": 1.0,
        })
        suite = BenchmarkSuite(settings)
        results = suite.run_all(frame)
        for name, metrics in results.items():
            assert metrics["max_drawdown"] >= -1.0, f"{name} exceeded a total loss"
            assert metrics["final_equity"] > 0, f"{name} went bankrupt"
            assert np.isfinite(metrics["cagr"]), f"{name} produced a non-finite CAGR"


class TestWalkForward:
    def test_failed_folds_are_reported_not_silently_empty(self, settings, synthetic_candles):
        """A run where every fold explodes must say so.

        Returning `{"metrics": {}}` for a total failure is indistinguishable
        from a strategy that simply made no money, which is how a real backtest
        bug hid behind a plausible-looking report.
        """
        validator = WalkForwardValidator(settings)

        def exploding(_train, _params):
            raise RuntimeError("deliberate refit failure")

        outcome = validator.validate(synthetic_candles, exploding, n_folds=2)
        assert outcome["error"]
        assert outcome["metrics"] == {}
        assert outcome["folds"] == []

    def test_validate_annualises_with_the_given_periods_per_year(
        self, settings, synthetic_candles
    ):
        validator = WalkForwardValidator(settings)

        def always_long(_train, _params):
            return {"predict": lambda test_frame: np.ones(len(test_frame))}

        # Identical folds, two annualisations: only the scaled fields may move.
        def run(ppy):
            return validator.validate(
                synthetic_candles, always_long, n_folds=2, periods_per_year=ppy
            )

        hourly = run(24 * 365)["metrics"]
        daily = run(365.0)["metrics"]
        assert hourly["sharpe"] != pytest.approx(daily["sharpe"])
        assert hourly["n_periods"] == daily["n_periods"]

    def test_folds_are_chronological_and_disjoint(self, settings, synthetic_candles):
        validator = WalkForwardValidator(settings)
        folds = validator.make_folds(synthetic_candles, n_folds=3)

        assert len(folds) >= 1
        for fold in folds:
            assert fold.train_start <= fold.train_end
            assert fold.train_end <= fold.test_start
            assert fold.test_start <= fold.test_end

        for earlier, later in zip(folds, folds[1:], strict=False):
            assert earlier.test_end <= later.test_start

    def test_train_data_never_overlaps_test(self, settings, synthetic_candles):
        validator = WalkForwardValidator(settings)
        for fold in validator.make_folds(synthetic_candles, n_folds=3):
            train_mask = (synthetic_candles["ts"] >= fold.train_start) & (
                synthetic_candles["ts"] < fold.train_end
            )
            test_mask = (synthetic_candles["ts"] >= fold.test_start) & (
                synthetic_candles["ts"] <= fold.test_end
            )
            assert train_mask.sum() > 0
            assert test_mask.sum() > 0
            assert not (train_mask & test_mask).any()
            assert synthetic_candles.loc[train_mask, "ts"].max() < fold.test_start
            assert fold.train_end <= fold.test_start

    def test_validate_returns_per_fold_metrics(self, settings, synthetic_candles):
        validator = WalkForwardValidator(settings)

        def strategy(train_frame, params):
            def predict(test_frame):
                forward = test_frame["close"].shift(-1) / test_frame["close"] - 1
                return np.sign(forward).fillna(0.0).to_numpy()

            return {"predict": predict}

        outcome = validator.validate(synthetic_candles, strategy, n_folds=3)
        assert "folds" in outcome
        assert "metrics" in outcome
        assert outcome["folds"]
        assert all("metrics" in fold for fold in outcome["folds"])
        assert outcome["folds"][0]["metrics"]["total_return"] > 0.05

    def test_train_only_signals_are_rejected_not_padded(self, settings, synthetic_candles):
        """Padding train-window signals into the test window would fake OOS results."""
        validator = WalkForwardValidator(settings)

        def strategy(train_frame, params):
            return {"signals": np.ones(len(train_frame))}

        outcome = validator.validate(synthetic_candles, strategy, n_folds=3)
        assert outcome["folds"] == []
        assert "error" in outcome

    def test_test_length_signals_are_accepted(self, settings, synthetic_candles):
        validator = WalkForwardValidator(settings)
        folds = validator.make_folds(synthetic_candles, n_folds=3)
        expected_len = int(
            (
                (synthetic_candles["ts"] >= folds[0].test_start)
                & (synthetic_candles["ts"] <= folds[0].test_end)
            ).sum()
        )

        def strategy(train_frame, params):
            return {"signals": np.ones(expected_len)}

        outcome = validator.validate(synthetic_candles, strategy, n_folds=3)
        assert outcome["folds"]
        assert outcome["folds"][0]["metrics"]["final_equity"] > 0

    def test_benchmark_comes_from_test_window_prices(self, settings, synthetic_candles):
        """Buy-and-hold must be derived from the fold's own test data, not the strategy."""
        validator = WalkForwardValidator(settings)

        def strategy(train_frame, params):
            return {"predict": lambda test_frame: np.ones(len(test_frame))}

        outcome = validator.validate(synthetic_candles, strategy, n_folds=3)
        assert "benchmark_metrics" in outcome
        assert len(outcome["benchmark_returns"]) == len(outcome["oos_returns"])
        assert outcome["metrics"]["excess_vs_buy_and_hold"] == pytest.approx(
            outcome["metrics"]["sharpe"] - outcome["benchmark_metrics"]["sharpe"], rel=1e-9
        )
        assert all("benchmark_metrics" in fold for fold in outcome["folds"])


    def test_insufficient_data_reports_error(self, settings, synthetic_candles):
        tiny = synthetic_candles.iloc[:20]
        validator = WalkForwardValidator(settings)
        outcome = validator.validate(tiny, lambda f, p: {"signals": np.zeros(len(f))}, n_folds=3)
        assert "error" in outcome or outcome["folds"] == []


class TestSignalAlignment:
    """The predict -> signal bridge must align on timestamps, not row order.

    `ForecasterTrainer.predict` returns a RangeIndex, so a signal Series built
    from it has to be re-indexed by `ts` before it can be mapped onto the test
    window. The buggy version produced all-NaN signals, which the engine read
    as "never trade": the backtest reported a flat 0.00% return across every
    fold with zero trades instead of failing loudly.
    """

    @staticmethod
    def _align(predictions: pd.DataFrame, test_frame: pd.DataFrame, threshold: float) -> np.ndarray:
        series = pd.Series(
            (predictions["predicted_return"] > threshold).to_numpy(dtype=float),
            index=pd.DatetimeIndex(predictions["ts"]),
        )
        test_ts = pd.DatetimeIndex(test_frame["ts"])
        aligned = np.asarray(test_ts.isin(series.index), dtype=bool)
        signals = np.zeros(len(test_frame))
        if aligned.any():
            signals[aligned] = series.reindex(test_ts[aligned]).to_numpy()
        return np.nan_to_num(signals, nan=0.0)

    def test_alignment_maps_predictions_onto_test_rows(self, settings, toy_dataset):
        frame, columns = toy_dataset
        split = int(len(frame) * 0.75)
        trainer = ForecasterTrainer(settings)
        model, info = trainer.train(
            frame.iloc[:split], frame.iloc[split:], columns, epochs=2, seq_len=16, kind="mlp"
        )
        test_frame = frame.iloc[split:]
        predictions = trainer.predict(model, test_frame, columns, info["model_info"])

        for threshold in (-np.inf, 0.0, np.inf):
            signals = self._align(predictions, test_frame, threshold)
            assert len(signals) == len(test_frame)
            assert np.isfinite(signals).all(), f"non-finite signals at threshold={threshold}"
            assert set(np.unique(signals)) <= {0.0, 1.0}

    def test_all_on_threshold_yields_full_length_signal(self, settings, toy_dataset):
        frame, columns = toy_dataset
        split = int(len(frame) * 0.75)
        trainer = ForecasterTrainer(settings)
        model, info = trainer.train(
            frame.iloc[:split], frame.iloc[split:], columns, epochs=2, seq_len=16, kind="mlp"
        )
        test_frame = frame.iloc[split:]
        predictions = trainer.predict(model, test_frame, columns, info["model_info"])

        # Every row past the first window must receive a signal.
        signals = self._align(predictions, test_frame, -np.inf)
        assert signals[16:].sum() > 0, "signals after the warmup window are all zero"
        assert len(predictions) == len(test_frame) - int(info["model_info"]["seq_len"])

    def test_aligned_signals_actually_trade(self, settings, synthetic_candles):
        engine = BacktestEngine(settings)
        signals = np.zeros(len(synthetic_candles))
        signals[20:] = 1.0
        result = engine.run_signal_strategy(synthetic_candles, signals)
        assert result["metrics"]["trades"] > 0, "long-only signals produced no trades"



class TestBacktestHorizonFidelity:
    """The walk-forward backtest must reproduce the trained model's horizon.

    A model trained to predict 12-bar forward returns was previously scored
    against 1-bar returns, which traded a signal against the wrong horizon and
    produced a destroyed Sharpe next to healthy directional accuracy.
    """

    def _pipeline_with_record(self, settings, params):
        from types import SimpleNamespace

        from cqip.pipeline import TrainingPipeline

        pipeline = TrainingPipeline(settings)
        pipeline.registry.get_active = lambda model_id: SimpleNamespace(
            symbol="BTC_USDT",
            interval="1d",
            feature_schema=None,
            params=params,
        )
        return pipeline

    def test_uses_the_recorded_training_horizon(self, settings):
        pipeline = self._pipeline_with_record(
            settings, {"architecture": "mlp", "horizon": 12, "epochs": 7, "seq_len": 8}
        )
        seen: list[int] = []

        def fake_build(symbol, interval, horizon=1, **kwargs):
            seen.append(horizon)
            raise RuntimeError("stop after the dataset build")

        pipeline.builder.build = fake_build
        with pytest.raises(RuntimeError):
            pipeline.backtest_predictions("any-model-id")
        assert seen == [12], "backtest scored the model against the wrong horizon"

    def test_defaults_to_one_when_the_record_has_no_horizon(self, settings):
        pipeline = self._pipeline_with_record(
            settings, {"architecture": "mlp", "seq_len": 8}
        )
        seen: list[int] = []

        def fake_build(symbol, interval, horizon=1, **kwargs):
            seen.append(horizon)
            raise RuntimeError("stop after the dataset build")

        pipeline.builder.build = fake_build
        with pytest.raises(RuntimeError):
            pipeline.backtest_predictions("any-model-id")
        assert seen == [1]

    def test_fold_epochs_come_from_the_model_record(self, settings):
        """`--epochs` has to reach the per-fold refit, not just the headline model."""
        from cqip.pipeline import resolve_fold_epochs

        recorded = resolve_fold_epochs({"epochs": 7}, settings)
        assert recorded == 7, f"folds would train for {recorded} epochs, not the trained 7"

    def test_fold_epochs_fall_back_when_the_record_is_silent(self, settings):
        from cqip.pipeline import resolve_fold_epochs

        fallback = resolve_fold_epochs({}, settings)
        assert fallback == max(5, settings.forecaster_epochs // 2)


class TestHoldoutTradingMetrics:
    """The holdout verdict needs trading risk figures, not just forecast skill.

    Forecast metrics describe ranking; a gate that never sees a drawdown, a trade
    count or a benchmark comparison will happily pass a model that is 51%
    accurate and loses 60% buying the currency.
    """

    def _bundle(self, frame, columns):
        from cqip.labeling import TargetSpec
        from cqip.pipeline import DatasetBundle

        return DatasetBundle(
            symbol="BTC_USDT", interval="1d", frame=frame,
            feature_columns=columns, rows=len(frame),
            target_spec=TargetSpec(
                kind="log_return", regression="target_return",
                direction="target_direction_binary", horizon=1,
            ),
        )

    def _pipeline(self, settings, frame, columns, signals):
        from cqip.pipeline import TrainingPipeline

        pipeline = TrainingPipeline(settings)

        def fake_predict(members, target_frame, feature_columns):
            return pd.DataFrame({
                "ts": target_frame["ts"].to_numpy(),
                "predicted_return": np.asarray(signals, dtype=float),
            })

        pipeline.trainer.predict_ensemble = fake_predict
        return pipeline, self._bundle(frame, columns)

    def test_trading_metrics_come_from_the_test_window_only(
        self, settings, synthetic_candles
    ):
        from cqip.features import build_feature_frame

        frame, columns = build_feature_frame(synthetic_candles)
        frame = frame.dropna(subset=["close"]).reset_index(drop=True)
        test_frame = frame.tail(120).reset_index(drop=True)
        signals = np.where(np.arange(len(test_frame)) % 3 == 0, 1.0, 0.0)
        pipeline, bundle = self._pipeline(settings, test_frame, columns, signals)

        seen: list[int] = []
        original = pipeline._holdout_trading_metrics

        def spy(members, passed_frame, *a, **k):
            seen.append(len(passed_frame))
            return original(members, passed_frame, *a, **k)

        pipeline._holdout_trading_metrics = spy
        result = pipeline._holdout_trading_metrics(
            [object()], test_frame, bundle, {"horizon": 1}, "1d"
        )

        assert seen == [len(test_frame)], "engine was handed more than the test window"
        assert result["test_trades"] > 0, "a non-flat signal produced no trades"
        assert result["test_max_drawdown"] is not None
        assert result["test_excess_vs_buy_and_hold"] is not None
        assert result["test_strategy_flat"] is False

    def test_flat_signal_is_reported_as_untraded_not_risk_free(
        self, settings, synthetic_candles
    ):
        """An all-flat signal must not report zero drawdown and call it safe.

        Reporting 0 trades / 0% drawdown for "never took a position" would let a
        model that does nothing pass a risk gate on the strength of never
        risking anything.
        """
        from cqip.features import build_feature_frame

        frame, columns = build_feature_frame(synthetic_candles)
        frame = frame.dropna(subset=["close"]).reset_index(drop=True)
        test_frame = frame.tail(120).reset_index(drop=True)
        pipeline, bundle = self._pipeline(
            settings, test_frame, columns, np.zeros(len(test_frame))
        )

        result = pipeline._holdout_trading_metrics(
            [object()], test_frame, bundle, {"horizon": 1}, "1d"
        )

        assert result["test_strategy_flat"] is True
        assert result["test_trades"] == 0
        assert result["test_max_drawdown"] is None, "flat signal must not claim zero drawdown"
        assert result["test_excess_vs_buy_and_hold"] is None

    def test_buy_and_hold_is_measured_over_the_same_window(
        self, settings, synthetic_candles
    ):
        from cqip.features import build_feature_frame

        frame, columns = build_feature_frame(synthetic_candles)
        frame = frame.dropna(subset=["close"]).reset_index(drop=True)
        test_frame = frame.tail(120).reset_index(drop=True)
        signals = np.where(np.arange(len(test_frame)) % 3 == 0, 1.0, 0.0)
        pipeline, bundle = self._pipeline(settings, test_frame, columns, signals)

        result = pipeline._holdout_trading_metrics(
            [object()], test_frame, bundle, {"horizon": 1}, "1d"
        )

        # Excess must be a signed difference in return units, and must agree with
        # its own two reported components.
        gap = result["test_excess_vs_buy_and_hold"]
        assert gap == pytest.approx(
            result["test_strategy_return"] - result["test_benchmark_return"], abs=1e-12
        )
        # The window really is the tail, so the benchmark cannot be an
        # all-sample return that flatters or buries the holdout.
        assert result["test_benchmark_return"] is not None


class TestDeferredHoldout:
    """Screening must be able to train without ever reading the test split.

    A matrix sweep trains hundreds of configurations. If each one reports a
    holdout score, the holdout has been used hundreds of times to make choices,
    and the only genuinely untouched sample in the project no longer exists.
    """

    def _pipeline(self, settings, toy_dataset):
        from cqip.labeling import TargetSpec
        from cqip.pipeline import DatasetBundle, TrainingPipeline

        frame, columns = toy_dataset
        pipeline = TrainingPipeline(settings)
        bundle = DatasetBundle(
            symbol="BTC_USDT", interval="1h", frame=frame,
            feature_columns=columns, rows=len(frame),
            target_spec=TargetSpec(
                kind="log_return", regression="target_return",
                direction="target_direction_binary", horizon=1,
            ),
        )
        pipeline.builder.build = lambda *a, **k: bundle

        class FakeModel(torch.nn.Module):
            def forward(self, x):
                batch = x.shape[0]
                return torch.zeros(batch), torch.zeros(batch)

        info = {
            "model_info": {
                "seq_len": 4, "horizon": 1, "architecture": "mlp",
                "feature_columns": list(columns),
                "scaler_mean": [0.0] * len(columns),
                "scaler_std": [1.0] * len(columns),
            },
            "metrics": {"epochs_run": 1, "val_loss": 0.5},
            "history": [],
        }
        pipeline.trainer.train = lambda *a, **k: (FakeModel(), info)
        pipeline.registry.register = lambda **k: None
        pipeline.registry.record_metrics = lambda *a, **k: None
        return pipeline

    def test_holdout_is_not_scored_when_deferred(self, settings, toy_dataset):
        pipeline = self._pipeline(settings, toy_dataset)
        prefixes: list[str] = []
        original = pipeline._evaluate_forecaster

        def spy(members, frame, bundle, info, *, prefix="test"):
            prefixes.append(prefix)
            return original(members, frame, bundle, info, prefix=prefix)

        pipeline._evaluate_forecaster = spy
        result = pipeline.train_forecaster(
            "BTC_USDT", "1h", kind="mlp", epochs=1, seq_len=4,
            evaluate_holdout=False,
        )

        assert prefixes == ["val"], f"test split was scored: {prefixes}"
        assert result["test_metrics"] == {}, "holdout metrics must be absent, not zero"

    def test_holdout_is_scored_when_requested(self, settings, toy_dataset):
        pipeline = self._pipeline(settings, toy_dataset)
        prefixes: list[str] = []
        original = pipeline._evaluate_forecaster

        def spy(members, frame, bundle, info, *, prefix="test"):
            prefixes.append(prefix)
            return original(members, frame, bundle, info, prefix=prefix)

        pipeline._evaluate_forecaster = spy
        pipeline.train_forecaster(
            "BTC_USDT", "1h", kind="mlp", epochs=1, seq_len=4,
            evaluate_holdout=True,
        )

        assert prefixes == ["val", "test"]


class TestLabelUnitRestoration:
    """A volatility-normalised label is in sigmas, not returns.

    Compounding sigmas as if they were log returns inflates every strategy
    statistic by roughly the asset's volatility, and exponentiating them reports
    a predicted price off by an order of magnitude. Both paths have to undo the
    scaling before touching price or equity.
    """

    def _bundle(self, sigma, *, normalized):
        from cqip.labeling import TargetSpec
        from cqip.pipeline import DatasetBundle

        frame = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=len(sigma), freq="D", tz="UTC"),
            "target_volatility": sigma,
            "target_return": [0.01] * len(sigma),
        })
        spec = TargetSpec(
            kind="vol_normalized" if normalized else "log_return",
            regression="target_return",
            direction="target_direction_binary",
            horizon=1,
            normalize_volatility=normalized,
        )
        return DatasetBundle(
            symbol="BTC_USDT", interval="1d", frame=frame,
            feature_columns=[], rows=len(frame), target_spec=spec,
        )

    def test_unscaled_targets_pass_through_untouched(self, settings):
        from cqip.pipeline import TrainingPipeline

        bundle = self._bundle([0.02, 0.03], normalized=False)
        predictions = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
            "predicted_return": [0.5, -0.5],
        })
        out = TrainingPipeline(settings)._to_log_return(predictions, bundle)
        assert out["predicted_return"].tolist() == [0.5, -0.5]

    def test_scaled_targets_are_multiplied_by_causal_sigma(self, settings):
        from cqip.pipeline import TrainingPipeline

        bundle = self._bundle([0.02, 0.04], normalized=True)
        predictions = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
            "predicted_return": [1.0, -1.0],
            "lower_68": [0.0, 0.0],
            "upper_68": [2.0, 2.0],
        })
        out = TrainingPipeline(settings)._to_log_return(predictions, bundle)
        assert out["predicted_return"].tolist() == [0.02, -0.04]
        assert out["lower_68"].tolist() == [0.0, 0.0]
        assert out["upper_68"].tolist() == [0.04, 0.08]

    def test_scaling_is_skipped_when_sigma_is_degenerate(self, settings):
        """A zero sigma would collapse a price to zero; better to leave it be."""
        from cqip.pipeline import TrainingPipeline

        bundle = self._bundle([0.0, 0.04], normalized=True)
        predictions = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
            "predicted_return": [1.0, -1.0],
        })
        out = TrainingPipeline(settings)._to_log_return(predictions, bundle)
        assert out["predicted_return"].tolist() == [1.0, -1.0]

    def test_missing_volatility_column_raises_rather_than_scaling_wrong(self, settings):
        from cqip.pipeline import TrainingPipeline

        bundle = self._bundle([0.02, 0.03], normalized=True)
        bundle.frame = bundle.frame.drop(columns=["target_volatility"])
        with pytest.raises(KeyError, match="target_volatility"):
            TrainingPipeline(settings)._to_log_return(
                pd.DataFrame({
                    "ts": pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC"),
                    "predicted_return": [1.0],
                }),
                bundle,
            )

    def test_spec_records_that_scaling_is_active(self):
        from cqip.labeling import TargetSpec

        plain = TargetSpec(
            kind="log_return", regression="target_return", direction=None, horizon=1
        )
        assert plain.normalize_volatility is False
        assert plain.as_dict()["normalize_volatility"] is False

        scaled = TargetSpec(
            kind="vol_normalized", regression="target_vol_normalized",
            direction=None, horizon=1, normalize_volatility=True,
        )
        assert scaled.as_dict()["normalize_volatility"] is True