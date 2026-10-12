"""Backtesting engine: walk-forward validation, benchmarks and performance metrics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from cqip.config import Settings, get_settings
from cqip.logging_config import get_logger

logger = get_logger(__name__)

TRADING_DAYS = 365.0

#: Minutes in a common crypto bar interval, for annualisation.
INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "3h": 180, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1_440, "3d": 4_320, "1w": 10_080,
}

#: Fallback when an interval is not in the table. Crypto trades continuously, so
#: the year is a full 365 days rather than 252 trading days.
MINUTES_PER_YEAR = 365 * 24 * 60


def periods_per_year(interval: str = "1h", *, label_horizon: int = 1) -> float:
    """How many return periods make up a year for this interval and label.

    This has to be explicit. The default of ``24 * 365`` is only correct for
    hourly bars, and it was being applied to every interval, so a daily backtest
    had its Sharpe inflated by ``sqrt(365 / 8760)`` and its CAGR by an order of
    magnitude. A backtest that reports a 200x annual return on daily data is not
    a good strategy, it is a units bug.

    ``label_horizon`` matters because the return series is one entry per *label*,
    not per bar: a 3-bar triple-barrier target produces one 3-bar return, so
    those periods are ``bars_per_year / 3`` long.
    """
    minutes = INTERVAL_MINUTES.get(str(interval).strip().lower())
    if minutes is None:
        minutes = 60
        logger.warning(
            "backtest_unknown_interval",
            extra={"interval": interval, "assumed_minutes": minutes},
        )
    bars = MINUTES_PER_YEAR / float(minutes)
    return max(1.0, bars / max(1, int(label_horizon)))


@dataclass
class Fold:
    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    purge_bars: int = 0
    embargo_bars: int = 0

    def as_dict(self) -> dict[str, Any]:
        # Serialise timestamps as ISO strings: this dict lands in a JSON column
        # and in API responses, and raw datetimes raise
        # "Object of type datetime is not JSON serializable" on flush.
        return {
            "index": self.index,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "purge_bars": self.purge_bars,
            "embargo_bars": self.embargo_bars,
        }


def _auto_block_size(n: int) -> int:
    """Block length for the moving-block bootstrap.

    Bar-level returns are strongly autocorrelated at the model window length, so
    an i.i.d. bootstrap would treat ``seq_len`` effectively independent draws as
    available and produce intervals far too narrow. A block length on the order
    of ``sqrt(n)`` is the standard conservative default when the true
    autocorrelation length is not known; callers can pass one explicitly.
    """
    return max(1, int(np.sqrt(max(1, n))))


def bootstrap_ci(
    values: np.ndarray | pd.Series,
    statistic: Callable[[np.ndarray], float] = lambda a: float(np.mean(a)),
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    block_size: int | None = None,
    seed: int = 0,
) -> dict[str, float | int]:
    """Moving-block bootstrap confidence interval for a statistic of a series.

    Resamples contiguous blocks rather than individual observations, which keeps
    the serial correlation that a window-based forecaster's predictions actually
    have. A point estimate with no interval is not evidence on these sample
    sizes: a directional accuracy of 0.53 on 80 bars is inside the noise band of
    a coin flip, and the gate needs to be able to see that.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[np.isfinite(array)]
    else:
        # Paired statistics (accuracy, correlation) are handed a 2-D array whose
        # columns move together. Dropping non-finite values with a flat boolean
        # mask would flatten it into 1-D and the statistic would then index a
        # nonexistent second axis. Filter whole rows instead, and resample rows
        # below, so the pairing inside a block survives.
        array = array[np.all(np.isfinite(array), axis=1)]
    series = array
    n = series.shape[0]
    if n == 0:
        return {"point": float("nan"), "lower": float("nan"), "upper": float("nan"),
                "n_bootstrap": 0, "n_obs": 0, "block_size": 0, "confidence": confidence}

    point = float(statistic(series))
    if n < 2:
        return {"point": point, "lower": point, "upper": point, "n_bootstrap": 0,
                "n_obs": int(n), "block_size": 0, "confidence": confidence}

    block = _auto_block_size(n) if block_size is None else max(1, int(block_size))
    block = min(block, n)
    n_blocks = int(np.ceil(n / block))
    max_start = n - block

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max_start + 1, size=(max(1, n_bootstrap), n_blocks))

    estimates = np.empty(max(1, n_bootstrap), dtype=float)
    for iteration in range(estimates.size):
        offsets = np.arange(block)
        rows = (starts[iteration][:, None] + offsets[None, :]).ravel()[:n]
        # `rows` indexes the first axis for both 1-D and 2-D input, so a block
        # keeps the actual/predicted pair aligned within each resample.
        sample = series[rows]
        try:
            estimates[iteration] = statistic(sample)
        except (ValueError, FloatingPointError, ZeroDivisionError, IndexError):
            estimates[iteration] = np.nan

    estimates = estimates[np.isfinite(estimates)]
    if estimates.size == 0:
        return {"point": point, "lower": float("nan"), "upper": float("nan"),
                "n_bootstrap": 0, "n_obs": int(n), "block_size": int(block),
                "confidence": confidence}

    alpha = 1.0 - float(confidence)
    lower = float(np.quantile(estimates, alpha / 2.0))
    upper = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    return {
        "point": point,
        "lower": lower,
        "upper": upper,
        "n_bootstrap": int(estimates.size),
        "n_obs": int(n),
        "block_size": int(block),
        "confidence": float(confidence),
    }


def _directional_accuracy_statistic(actual: np.ndarray, predicted: np.ndarray) -> Callable[[np.ndarray], float]:
    """Bootstrap statistic for accuracy, indexed by the rows of a paired array."""
    def statistic(sample: np.ndarray) -> float:
        chunk = sample[:, :2]
        if chunk.size == 0:
            return float("nan")
        return float(np.mean(np.sign(chunk[:, 0]) == np.sign(chunk[:, 1])))
    return statistic


def _correlation_statistic(actual: np.ndarray, predicted: np.ndarray) -> Callable[[np.ndarray], float]:
    def statistic(sample: np.ndarray) -> float:
        chunk = sample[:, :2]
        if chunk.shape[0] < 3:
            return float("nan")
        with np.errstate(invalid="ignore", divide="ignore"):
            matrix = np.corrcoef(chunk[:, 0], chunk[:, 1])
        value = float(matrix[0, 1]) if matrix.shape == (2, 2) else float("nan")
        return value if np.isfinite(value) else float("nan")
    return statistic


def _mae_statistic(actual: np.ndarray, predicted: np.ndarray) -> Callable[[np.ndarray], float]:
    def statistic(sample: np.ndarray) -> float:
        chunk = sample[:, :2]
        if chunk.size == 0:
            return float("nan")
        return float(np.mean(np.abs(chunk[:, 0] - chunk[:, 1])))
    return statistic


def forecast_metric_cis(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    block_size: int | None = None,
    seed: int = 0,
) -> dict[str, dict[str, float | int]]:
    """Bootstrap confidence intervals for the three forecast-quality metrics.

    ``actual`` and ``predicted`` are paired row-for-row, and the resampled index
    is shared across all three metrics so the intervals describe one draw from
    the same resampled series rather than three unrelated ones.
    """
    actual_array = np.asarray(actual, dtype=float).reshape(-1)
    predicted_array = np.asarray(predicted, dtype=float).reshape(-1)
    size = min(actual_array.size, predicted_array.size)
    if size == 0:
        return {}
    paired = np.column_stack([actual_array[:size], predicted_array[:size]])
    paired = paired[np.isfinite(paired).all(axis=1)]
    if paired.size == 0:
        return {}

    shared = {"n_bootstrap": n_bootstrap, "confidence": confidence,
              "block_size": block_size, "seed": seed}
    return {
        "directional_accuracy": bootstrap_ci(
            paired, _directional_accuracy_statistic(paired[:, 0], paired[:, 1]), **shared
        ),
        "correlation": bootstrap_ci(
            paired, _correlation_statistic(paired[:, 0], paired[:, 1]), **shared
        ),
        "mae": bootstrap_ci(
            paired, _mae_statistic(paired[:, 0], paired[:, 1]), **shared
        ),
    }


def compute_metrics(
    returns: np.ndarray | pd.Series,
    *,
    initial_capital: float = 10_000.0,
    periods_per_year: float | None = None,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Standard performance statistics for a return series.

    ``periods_per_year`` scales every annualised figure (CAGR, volatility,
    Sharpe, Sortino, Calmar). It has no safe default: a wrong value does not
    produce an error, it produces a confidently wrong Sharpe, which is the most
    dangerous failure mode this codebase can have. Callers that do not know the
    bar interval are warned and fall back to the hourly assumption.
    """
    series = np.asarray(returns, dtype=float)
    series = series[np.isfinite(series)]
    if periods_per_year is None:
        logger.warning(
            "compute_metrics_unannualised",
            extra={"assumption": "24*365 (hourly bars)", "n": int(series.size)},
        )
        periods_per_year = 24 * 365
    periods_per_year = max(1e-9, float(periods_per_year))
    if series.size == 0:
        return {
            "total_return": 0.0, "cagr": 0.0, "volatility": 0.0, "sharpe": 0.0,
            "sortino": 0.0, "calmar": 0.0, "max_drawdown": 0.0, "win_rate": 0.0,
            "profit_factor": 0.0, "skew": 0.0, "kurtosis": 0.0, "var_95": 0.0,
            "final_equity": initial_capital, "n_periods": 0, "best_period": 0.0,
            "worst_period": 0.0,
        }

    equity_curve = initial_capital * np.cumprod(1 + series)
    final_equity = float(equity_curve[-1])
    total_return = final_equity / initial_capital - 1

    elapsed = series.size / periods_per_year
    growth = (final_equity / initial_capital) if initial_capital else 1.0
    if elapsed > 0 and growth > 0:
        cagr = float(growth ** (1 / elapsed) - 1)
    elif elapsed > 0 and growth == 0:
        cagr = -1.0
    else:
        cagr = 0.0

    volatility = float(series.std(ddof=1)) if series.size > 1 else 0.0
    annualized_vol = volatility * np.sqrt(periods_per_year)

    excess = series - risk_free_rate / periods_per_year
    sharpe = (
        float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year))
        if series.size > 1 and excess.std(ddof=1) > 1e-12 else 0.0
    )

    downside = excess[excess < 0]
    downside_std = float(downside.std(ddof=1)) if downside.size > 1 else 0.0
    sortino = (
        float(excess.mean() / downside_std * np.sqrt(periods_per_year))
        if downside_std > 1e-12 else 0.0
    )

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_drawdown = float(drawdowns.min()) if drawdowns.size else 0.0
    calmar = float(cagr / abs(max_drawdown)) if abs(max_drawdown) > 1e-9 else 0.0

    wins = series[series > 0]
    losses = series[series < 0]
    win_rate = float(wins.size / series.size) if series.size else 0.0
    gross_profit = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(-losses.sum()) if losses.size else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else 0.0

    skew = float(pd.Series(series).skew())
    kurtosis = float(pd.Series(series).kurtosis())
    var_95 = float(np.percentile(series, 5))

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "volatility": float(annualized_vol),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "skew": skew,
        "kurtosis": kurtosis,
        "var_95": var_95,
        "final_equity": final_equity,
        "n_periods": int(series.size),
        "best_period": float(series.max()),
        "worst_period": float(series.min()),
    }


def compute_trade_metrics(trades: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not trades:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "avg_holding": 0.0}

    pnls: list[float] = []
    holding: list[int] = []
    open_trade: dict[str, Any] | None = None

    for trade in trades:
        side = trade.get("side")
        if side == "buy" and open_trade is None:
            open_trade = {"cost": trade["quantity"] * trade["price"], "index": trade["index"]}
        elif side == "sell" and open_trade is not None:
            proceeds = trade["quantity"] * trade["price"]
            pnls.append(proceeds - open_trade["cost"])
            holding.append(int(trade["index"]) - int(open_trade["index"]))
            open_trade = None

    if not pnls:
        return {"trades": len(trades), "wins": 0, "losses": 0, "win_rate": 0.0, "avg_holding": 0.0}

    pnl_array = np.asarray(pnls, dtype=float)
    wins = int((pnl_array > 0).sum())
    return {
        "trades": len(pnls),
        "wins": wins,
        "losses": len(pnls) - wins,
        "win_rate": wins / len(pnls),
        "avg_holding": float(np.mean(holding)) if holding else 0.0,
        "total_pnl": float(pnl_array.sum()),
    }


class BacktestEngine:
    """Vectorized fee-aware backtester for signal-based strategies."""

    def __init__(
        self, settings: Settings | None = None, *, periods_per_year: float | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.fee_rate = self.settings.backtest_fee_bps / 10_000
        self.slippage_rate = self.settings.backtest_slippage_bps / 10_000
        # An explicit override is the only way to annualise correctly; see
        # `periods_per_year` for why the default cannot be trusted.
        if periods_per_year is None:
            logger.warning("backtest_engine_default_annualisation")
            periods_per_year = 24 * 365
        self.periods_per_year = max(1e-9, float(periods_per_year))

    def run_signal_strategy(
        self,
        frame: pd.DataFrame,
        signals: np.ndarray,
        *,
        initial_capital: float | None = None,
        position_size: float = 1.0,
        shorting: bool = True,
        fee_bps: float | None = None,
        slippage_bps: float | None = None,
    ) -> dict[str, Any]:
        """Backtest a position series in {-1, 0, 1} with next-bar execution.

        ``signals[i]`` is decided using information available at the close of bar
        ``i`` and filled at the **open of bar i+1**. That ordering is what makes
        the backtest causal — reading ``signals[t+1]`` when filling bar ``t+1``
        would leak the future.

        ``position_size`` is the fraction of current equity committed as
        **notional** exposure, clamped to ``[0, 1]`` so the strategy can never
        borrow capital. Sizing off notional rather than whole asset units keeps
        this engine consistent with the RL environments, which cap the same way,
        and means a ``10_000`` account never risks more than it holds on a single
        bar. Exposure is rescaled every bar, so a winning long compounds into a
        larger notional position exactly as a rebalanced account would.
        """
        if frame.empty or len(signals) != len(frame):
            raise ValueError("frame and signals length mismatch or empty frame")

        capital = initial_capital or self.settings.backtest_initial_capital
        fee_rate = (fee_bps if fee_bps is not None else self.settings.backtest_fee_bps) / 10_000
        slippage_rate = (
            slippage_bps if slippage_bps is not None else self.settings.backtest_slippage_bps
        ) / 10_000
        opens = frame["open"].to_numpy(dtype=float)
        closes = frame["close"].to_numpy(dtype=float)
        times = frame["ts"].to_numpy()

        position = 0.0
        equity_curve = [capital]
        trades: list[dict[str, Any]] = []
        returns: list[float] = []

        # Fraction of equity committed as notional, clamped so that no borrowed
        # capital is ever implied.
        size = min(1.0, max(0.0, float(position_size)))

        for index in range(len(frame) - 1):
            execution_index = index + 1
            direction = float(signals[index])
            if not shorting:
                direction = max(0.0, direction)

            open_price = opens[execution_index]
            equity_before = capital + position * closes[index]
            if open_price > 0:
                # Size against equity marked at the price we are actually trading
                # at. Marking at the previous close would under-invest by exactly
                # the move since that close, so a 1.0-size position would settle
                # near 0.99 exposure and quietly lag the asset it tracks.
                equity_at_fill = capital + position * open_price
                # A non-positive equity flattens the book rather than taking a
                # position against a capital base that no longer exists.
                target_notional = size * direction * max(equity_at_fill, 0.0)
            else:
                target_notional = 0.0

            # Work in notional, not units, so the fill can be priced off the side
            # this particular trade is on. Deriving the slippage direction from the
            # signal instead would make a long exit fill at open*(1+slip) and a
            # short entry fill at open*(1-slip): both favourable, so a round trip
            # would pay no spread at all and the cost model would silently flatter
            # every strategy that trades.
            delta_notional = target_notional - position * open_price
            if abs(delta_notional) > 1e-9:
                fill_price = open_price * (1 + slippage_rate * np.sign(delta_notional))
                delta = delta_notional / fill_price
                fee = abs(delta) * fill_price * fee_rate
                capital -= delta * fill_price + fee
                trades.append({
                    "index": execution_index,
                    "ts": pd.Timestamp(times[execution_index]).isoformat(),
                    "signal_index": index,
                    "side": "buy" if delta > 0 else "sell",
                    "quantity": abs(delta),
                    "price": float(fill_price),
                    "fee": float(fee),
                    "notional": abs(delta) * fill_price,
                })
                position += delta

            equity_after = capital + position * closes[execution_index]
            period_return = (equity_after - equity_before) / equity_before if equity_before > 0 else 0.0
            returns.append(period_return)
            equity_curve.append(equity_after)

        returns_array = np.asarray(returns, dtype=float)
        metrics = compute_metrics(
            returns_array,
            initial_capital=initial_capital or self.settings.backtest_initial_capital,
            periods_per_year=self.periods_per_year,
        )
        metrics.update(compute_trade_metrics(trades))

        equity_series = pd.Series(equity_curve)
        drawdown = equity_series / equity_series.cummax() - 1

        return {
            "metrics": metrics,
            "equity_curve": equity_series.tolist(),
            "returns": returns_array.tolist(),
            "trades": trades,
            "drawdown": drawdown.tolist(),
            "n_bars": len(returns),
            "exposure": float((np.abs(signals[:-1]) > 0).mean()) if len(signals) > 1 else 0.0,
        }

    def run_buy_and_hold(self, frame: pd.DataFrame, *, initial_capital: float | None = None) -> dict[str, Any]:
        if frame.empty:
            raise ValueError("Cannot backtest empty frame")
        capital = initial_capital or self.settings.backtest_initial_capital
        closes = frame["close"].to_numpy(dtype=float)

        # Charge the entry fee inside the first period rather than prepending a
        # return, so the benchmark stays bar-for-bar aligned with a strategy's
        # return series and excess-return comparisons remain valid.
        price_returns = np.diff(closes) / closes[:-1]
        if price_returns.size:
            price_returns = price_returns.copy()
            price_returns[0] = (1 - self.fee_rate) * (1 + self.slippage_rate) * (1 + price_returns[0]) - 1
        returns = price_returns

        equity_curve = list(capital * np.cumprod(1 + returns))
        metrics = compute_metrics(
            returns, initial_capital=capital, periods_per_year=self.periods_per_year
        )
        return {
            "metrics": metrics,
            "equity_curve": equity_curve,
            "returns": returns.tolist(),
            "trades": [],
            "drawdown": (pd.Series(equity_curve) / pd.Series(equity_curve).cummax() - 1).tolist(),
            "n_bars": len(returns),
            "exposure": 1.0,
        }

    def run_momentum(self, frame: pd.DataFrame, window: int = 24, **kwargs: Any) -> dict[str, Any]:
        if len(frame) <= window:
            raise ValueError(f"Need more than {window} bars for momentum strategy")
        signals = np.sign(frame["close"].rolling(window).mean().diff().fillna(0.0).to_numpy())
        signals[np.abs(frame["close"].diff().rolling(window).mean().to_numpy()) < 1e-12] = 0
        return self.run_signal_strategy(frame, signals, **kwargs)

    def run_conviction_strategy(
        self,
        frame: pd.DataFrame,
        scores: np.ndarray,
        *,
        threshold: float = 0.0,
        max_weight: float = 1.0,
        leverage: float = 1.0,
        position_size: float = 1.0,
        initial_capital: float | None = None,
        fee_bps: float | None = None,
        slippage_bps: float | None = None,
    ) -> dict[str, Any]:
        """Trade only when conviction clears a bar, sizing by conviction.

        The difference from :meth:`run_signal_strategy` is the *decision rule*.
        A sign strategy is in the market on every bar it is not flat, so it pays
        a fee on every rebalance whether or not the model knows anything. Here a
        bar is traded only when ``|score| >= threshold``; everything else is
        cash. That distinction is the whole point of a selective system: being
        right is only worth something if being flat is also an option.

        ``scores`` is any ranking score where larger magnitude means stronger
        belief, canonically ``predicted_return / predicted_volatility``. The
        position is ``sign(score) * weight``, where ``weight`` ramps linearly from
        ``0`` at the threshold to ``max_weight`` at twice the threshold, so
        confidence maps to size continuously instead of snapping to full size on
        the first bar that clears the bar.

        ``leverage`` above 1.0 scales the resulting notional past equity. That is
        deliberate and dangerous: a leveraged result is not comparable with an
        unlevered one, and its drawdown compounds faster than its return. It is
        reported separately rather than folded into the metrics, because the
        default comparison everyone reaches for — versus buy-and-hold — is only
        meaningful at the same exposure.

        Execution and fee accounting are inherited unchanged: a signal decided at
        the close of bar ``i`` is filled at the open of bar ``i+1``.
        """
        if frame.empty or len(scores) != len(frame):
            raise ValueError("frame and scores length mismatch or empty frame")
        if threshold <= 0:
            raise ValueError(
                "threshold must be > 0: a conviction strategy with no bar to clear "
                "degenerates into always being in the market"
            )
        if max_weight <= 0:
            raise ValueError("max_weight must be > 0")
        if leverage <= 0:
            raise ValueError("leverage must be > 0")

        score_array = np.asarray(scores, dtype=float)
        # Linear ramp from flat at the threshold to full size at 2x the threshold.
        magnitude = np.clip(
            (np.abs(score_array) - threshold) / threshold, 0.0, None
        )
        magnitude = np.minimum(magnitude, 1.0) * float(max_weight) * float(leverage)
        magnitude[~np.isfinite(score_array)] = 0.0
        signals = np.sign(score_array) * magnitude
        # Below the bar, or exactly on it, the book is flat.
        signals[np.abs(score_array) < threshold] = 0.0

        result = self.run_signal_strategy(
            frame, signals, initial_capital=initial_capital,
            position_size=position_size, fee_bps=fee_bps, slippage_bps=slippage_bps,
        )
        engaged = np.abs(signals) > 0.0
        result["coverage"] = float(engaged.mean()) if engaged.size else 0.0
        result["threshold"] = float(threshold)
        result["leverage"] = float(leverage)
        result["leverage_capped"] = bool(
            np.abs(signals).max(initial=0.0) > 1.0 + 1e-9
        )
        metrics = result.get("metrics") or {}
        result["metrics"] = {
            **metrics,
            "coverage": result["coverage"],
            "leverage": float(leverage),
        }
        return result



class WalkForwardValidator:
    """Chronological train/validate/test splitting with anchored or rolling windows."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def make_folds(
        self,
        frame: pd.DataFrame,
        *,
        train_ratio: float | None = None,
        val_ratio: float | None = None,
        n_folds: int = 3,
        anchor: str | None = None,
        embargo_pct: float = 0.01,
        purge_bars: int = 0,
        embargo_bars: int | None = None,
    ) -> list[Fold]:
        """Chronological folds with a purge/embargo gap before each test window.

        ``purge_bars`` removes training rows whose forward label reaches into the
        test window — for a ``horizon``-bar target that is at least ``horizon``
        bars, and passing anything less silently re-introduces the leak the gate
        is supposed to catch. ``embargo_bars`` adds a further gap on top of the
        percentage-based one, sized to the model's window length by the pipeline
        so that overlapping windows cannot straddle the boundary.
        """
        train_ratio = train_ratio if train_ratio is not None else self.settings.backtest_train_ratio
        val_ratio = val_ratio if val_ratio is not None else self.settings.backtest_val_ratio
        anchor = anchor or self.settings.backtest_anchor
        test_ratio = max(0.05, 1.0 - train_ratio - val_ratio)
        n_folds = max(1, n_folds)

        total = len(frame)
        test_size = int(total * test_ratio / n_folds)
        purge = max(0, int(purge_bars))
        percentage_embargo = int(total * embargo_pct)
        explicit_embargo = max(0, int(embargo_bars)) if embargo_bars is not None else 0
        embargo = max(percentage_embargo, explicit_embargo)
        gap = purge + embargo
        folds: list[Fold] = []

        initial_train = int(total * train_ratio)
        for index in range(n_folds):
            test_start = initial_train + index * test_size
            test_end = test_start + test_size
            if test_end >= total:
                break

            train_end = test_start - gap
            train_start = max(0, train_end - initial_train) if anchor == "rolling" else 0

            if train_end - train_start < 50 or test_end - test_start < 10:
                continue

            folds.append(Fold(
                index=index,
                train_start=frame["ts"].iloc[train_start].to_pydatetime(),
                train_end=frame["ts"].iloc[train_end].to_pydatetime(),
                test_start=frame["ts"].iloc[test_start].to_pydatetime(),
                test_end=frame["ts"].iloc[test_end].to_pydatetime(),
                purge_bars=purge,
                embargo_bars=embargo,
            ))
        return folds

    def validate(
        self,
        frame: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame, dict[str, Any]], dict[str, Any]],
        *,
        n_folds: int = 3,
        params: dict[str, Any] | None = None,
        periods_per_year: float | None = None,
        **fold_kwargs: Any,
    ) -> dict[str, Any]:
        """Run ``strategy_fn`` per fold, refitting only on each fold's train slice.

        ``periods_per_year`` must describe the interval *and* the label horizon
        of the return series, or every annualised figure here is a units error.
        Callers that omit it get the engine default, which is only right for
        hourly single-bar targets.

        ``strategy_fn(train_frame, params)`` returns a dict that may contain:

        * ``predict``: a callable taking the **test** frame and returning a
          position series. This is the preferred contract, because the model is
          refit on train data and then genuinely forecasts unseen bars.
        * ``signals``: an already-computed position series. It is only accepted
          when its length matches the test slice exactly. Signals produced on the
          train window are *not* padded into the test window, because they carry
          no information about test-period behaviour and would silently turn the
          backtest into an in-sample one.
        """
        params = params or {}
        annualisation = (
            float(periods_per_year) if periods_per_year is not None else None
        )
        folds = self.make_folds(frame, n_folds=n_folds, **fold_kwargs)
        if not folds:
            return {"folds": [], "metrics": {}, "error": "insufficient data for folds"}

        fold_results: list[dict[str, Any]] = []
        all_returns: list[float] = []
        all_benchmark_returns: list[float] = []
        all_times: list[Any] = []

        for fold in folds:
            train_mask = (frame["ts"] >= fold.train_start) & (frame["ts"] < fold.train_end)
            test_mask = (frame["ts"] >= fold.test_start) & (frame["ts"] <= fold.test_end)
            train_frame = frame[train_mask].reset_index(drop=True)
            test_frame = frame[test_mask].reset_index(drop=True)

            if len(train_frame) < 60 or len(test_frame) < 5:
                continue

            try:
                outcome = strategy_fn(train_frame, params)
            except Exception as exc:  # noqa: BLE001
                logger.warning("walk_forward_fold_failed", extra={
                    "fold": fold.index, "error": str(exc)[:200],
                })
                continue

            signals: np.ndarray | None = None
            predictor = outcome.get("predict")
            if callable(predictor):
                try:
                    signals = np.asarray(predictor(test_frame), dtype=float)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("walk_forward_predict_failed", extra={
                        "fold": fold.index, "error": str(exc)[:200],
                    })
                    continue
            elif outcome.get("signals") is not None:
                candidate = np.asarray(outcome["signals"], dtype=float)
                if len(candidate) == len(test_frame):
                    signals = candidate
                else:
                    logger.warning("walk_forward_signal_length_mismatch", extra={
                        "fold": fold.index,
                        "signals": len(candidate),
                        "test_bars": len(test_frame),
                    })
                    continue

            if signals is None:
                logger.warning("walk_forward_no_signals", extra={"fold": fold.index})
                continue

            engine = BacktestEngine(self.settings, periods_per_year=annualisation)
            result = engine.run_signal_strategy(test_frame, signals)
            fold_metrics = result["metrics"]
            fold_results.append({
                **fold.as_dict(),
                "metrics": fold_metrics,
                "n_bars": result["n_bars"],
                "trades": len(result["trades"]),
            })
            all_returns.extend(result["returns"])
            all_times.extend(test_frame["ts"].iloc[1:].tolist())

            # A true benchmark must come from the same test window the strategy
            # traded, otherwise the comparison is meaningless.
            benchmark = engine.run_buy_and_hold(test_frame)
            all_benchmark_returns.extend(benchmark["returns"])
            fold_results[-1]["benchmark_metrics"] = benchmark["metrics"]
            logger.info("walk_forward_fold", extra={
                "fold": fold.index, "sharpe": round(fold_metrics["sharpe"], 4),
                "return": round(fold_metrics["total_return"], 4), "bars": result["n_bars"],
            })

        if not all_returns:
            return {"folds": fold_results, "metrics": {}, "error": "no folds produced returns"}

        combined = compute_metrics(
            np.asarray(all_returns, dtype=float), periods_per_year=annualisation
        )
        combined["folds_evaluated"] = len(fold_results)
        combined["consistency"] = (
            float(np.mean([f["metrics"]["total_return"] > 0 for f in fold_results]))
            if fold_results else 0.0
        )

        benchmark = compute_metrics(
            np.asarray(all_benchmark_returns, dtype=float),
            periods_per_year=annualisation,
        )
        combined["excess_vs_buy_and_hold"] = combined["sharpe"] - benchmark["sharpe"]
        combined["benchmark_total_return"] = benchmark["total_return"]
        # `compute_metrics` only sees the return series, so trade counts have to
        # be aggregated from the folds. Downstream quality gates read `trades`,
        # and omitting it made a strategy that never traded look unevaluable.
        combined["trades"] = int(sum(f.get("trades", 0) for f in fold_results))

        result: dict[str, Any] = {
            "folds": fold_results,
            "metrics": combined,
            "oos_returns": all_returns,
            "oos_times": [str(t) for t in all_times],
            "benchmark_returns": all_benchmark_returns,
            "benchmark_metrics": benchmark,
            "test_size": len(all_returns),
        }
        if len(fold_results) > 1:
            result["fold_distribution"] = _fold_distribution(fold_results)
            sharpe_ci = bootstrap_ci(
                np.asarray(all_returns, dtype=float),
                lambda sample: compute_metrics(
                    sample, periods_per_year=annualisation
                )["sharpe"],
                n_bootstrap=self.settings.bootstrap_iterations,
                confidence=self.settings.bootstrap_confidence,
                seed=self.settings.seed,
            )
            if np.isfinite(sharpe_ci["lower"]):
                result["sharpe_ci"] = sharpe_ci
        return result


def _fold_distribution(fold_results: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Spread of per-fold results, so a lucky single fold cannot hide.

    A headline Sharpe pooled across folds is a weighted average of per-fold
    outcomes, and one good fold can carry it. Reporting the mean and standard
    deviation of the per-fold metric alongside the pooled number makes the
    difference between a consistent edge and a lucky window visible without
    having to re-open every fold record.
    """
    def spread(key: str) -> float:
        values = [
            float((fold.get("metrics") or {})[key])
            for fold in fold_results
            if (fold.get("metrics") or {}).get(key) is not None
        ]
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    def mean(key: str) -> float:
        values = [
            float((fold.get("metrics") or {})[key])
            for fold in fold_results
            if (fold.get("metrics") or {}).get(key) is not None
        ]
        return float(np.mean(values)) if values else 0.0

    return {
        "folds": float(len(fold_results)),
        "mean_sharpe": mean("sharpe"),
        "std_sharpe": spread("sharpe"),
        "mean_total_return": mean("total_return"),
        "std_total_return": spread("total_return"),
        "mean_max_drawdown": mean("max_drawdown"),
        "positive_fold_fraction": float(
            np.mean([(fold.get("metrics") or {}).get("total_return", 0.0) > 0
                     for fold in fold_results])
        ),
    }


class BenchmarkSuite:
    """Compares a strategy against standard baselines on identical data."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.engine = BacktestEngine(settings)
        self.settings = settings or get_settings()

    def run_all(self, frame: pd.DataFrame, signals: np.ndarray | None = None) -> dict[str, Any]:
        results: dict[str, Any] = {
            "buy_and_hold": self.engine.run_buy_and_hold(frame)["metrics"],
        }
        if signals is not None and len(signals) == len(frame):
            results["strategy"] = self.engine.run_signal_strategy(frame, signals)["metrics"]
        if len(frame) > 30:
            results["momentum_24"] = self.engine.run_momentum(frame, 24)["metrics"]
            results["momentum_72"] = self.engine.run_momentum(frame, 72)["metrics"]

        for name, metrics in results.items():
            if name == "strategy" and "sharpe" in results.get("buy_and_hold", {}):
                excess = metrics["sharpe"] - results["buy_and_hold"]["sharpe"]
                results["strategy"]["excess_sharpe_vs_bh"] = excess
        return results


DEFAULT_CONVICTION_THRESHOLDS: tuple[float, ...] = (
    0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0,
)


def conviction_threshold_sweep(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    settings: Settings | None = None,
    thresholds: Sequence[float] = DEFAULT_CONVICTION_THRESHOLDS,
    leverage: float = 1.0,
    initial_capital: float | None = None,
) -> list[dict[str, Any]]:
    """Performance across conviction bars: "act only on my most confident X%".

    This is the table that answers whether a model is worth trading *selectively*
    rather than whether it is directionally accurate. Each row fixes a conviction
    bar, trades only the bars that clear it, and reports the full performance set
    alongside buy-and-hold over the identical window.

    A ``threshold`` of 0.0 is included on purpose as the always-in-market
    baseline. Every other row is only interesting relative to it: a selective
    rule that cannot beat being fully invested on the same bars was not selective
    in any useful sense.

    Selecting which row to deploy is a **validation-only** decision. Running this
    sweep on holdout bars and then trading the winner is fitting a parameter to
    the holdout, which spends the one sample that cannot be reused.
    """
    settings = settings or Settings()
    engine = BacktestEngine(settings)
    score_array = np.asarray(scores, dtype=float)
    if len(score_array) != len(frame):
        raise ValueError("frame and scores length mismatch")

    benchmark = engine.run_buy_and_hold(frame, initial_capital=initial_capital)
    benchmark_metrics = dict(benchmark.get("metrics") or {})
    benchmark_return = benchmark_metrics.get("total_return")
    benchmark_sharpe = benchmark_metrics.get("sharpe")

    rows: list[dict[str, Any]] = []
    positive = score_array[np.isfinite(score_array) & (score_array > 0)]
    usable = positive.size if positive.size else 0
    for threshold in thresholds:
        # The 0.0 row must still trade, so it is evaluated through the same
        # engine with the smallest representable bar rather than special-cased
        # out of the comparison.
        effective = float(threshold)
        if effective <= 0.0:
            effective = max(1e-9, float(np.nanmax(np.abs(score_array))) * 1e-6) if usable else 1e-9
        try:
            outcome = engine.run_conviction_strategy(
                frame, score_array, threshold=effective, leverage=leverage,
                initial_capital=initial_capital,
            )
        except ValueError:
            continue
        metrics = dict(outcome.get("metrics") or {})
        strategy_return = metrics.get("total_return")
        rows.append({
            "threshold": float(threshold),
            "effective_threshold": effective,
            "coverage": float(outcome.get("coverage") or 0.0),
            "trades": int(metrics.get("trades") or 0),
            "total_return": strategy_return,
            "sharpe": metrics.get("sharpe"),
            "sortino": metrics.get("sortino"),
            "calmar": metrics.get("calmar"),
            "profit_factor": metrics.get("profit_factor"),
            "win_rate": metrics.get("win_rate"),
            "max_drawdown": metrics.get("max_drawdown"),
            "buy_hold_return": benchmark_return,
            "buy_hold_sharpe": benchmark_sharpe,
            "excess_return": (
                float(strategy_return) - float(benchmark_return)
                if strategy_return is not None and benchmark_return is not None
                else None
            ),
            "excess_sharpe": (
                float(metrics.get("sharpe") or 0.0) - float(benchmark_sharpe or 0.0)
            ),
        })
    return rows


__all__ = [
    "BacktestEngine",
    "BenchmarkSuite",
    "DEFAULT_CONVICTION_THRESHOLDS",
    "Fold",
    "WalkForwardValidator",
    "bootstrap_ci",
    "compute_metrics",
    "compute_trade_metrics",
    "conviction_threshold_sweep",
    "forecast_metric_cis",
    "periods_per_year",
]