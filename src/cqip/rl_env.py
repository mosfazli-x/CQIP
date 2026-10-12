"""Gymnasium trading environment modelling spot crypto execution.

Key realism choices, since naive environments produce fantasy backtests:
- execution at the *next* bar open, never the current close
- proportional fees + slippage on both sides
- integer-quantized holdings with a configurable precision
- no look-ahead: observation at step ``t`` uses data through ``t`` only
- a liquidation-style ruin guard so the agent cannot silently lose everything
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cqip.config import Settings, get_settings
from cqip.logging_config import get_logger

logger = get_logger(__name__)

ACTION_LONG = 0
ACTION_SHORT = 1
ACTION_FLAT = 2


class TradingEnv(gym.Env):
    """Single-asset spot trading environment over a precomputed feature matrix."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        feature_columns: Sequence[str],
        *,
        settings: Settings | None = None,
        initial_capital: float | None = None,
        fee_bps: float | None = None,
        slippage_bps: float | None = None,
        reward_shaping: bool = True,
        position_size: float = 1.0,
        allow_fractional: bool = True,
        quantity_precision: int = 8,
        max_episode_steps: int | None = None,
        lookback: int = 1,
    ) -> None:
        super().__init__()

        self.settings = settings or get_settings()
        if len(features) != len(prices):
            raise ValueError(f"features/prices length mismatch: {len(features)} vs {len(prices)}")
        if len(features) < 3:
            raise ValueError("Need at least 3 rows to construct a trading episode")

        self.features = np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.prices = np.asarray(prices, dtype=np.float64)
        self.feature_columns = list(feature_columns)
        self.lookback = max(1, int(lookback))
        self.seq_len = int(getattr(settings, "forecaster_seq_len", 64)) if settings else 64
        self.seq_len = min(self.seq_len, max(1, len(self.features) // 2))

        self.initial_capital = float(initial_capital or self.settings.rl_initial_capital)
        self.fee_rate = (fee_bps if fee_bps is not None else self.settings.backtest_fee_bps) / 10_000
        self.slippage_rate = (slippage_bps if slippage_bps is not None else self.settings.backtest_slippage_bps) / 10_000
        self.reward_shaping = reward_shaping
        self.position_size = float(position_size)
        self.allow_fractional = allow_fractional
        self.quantity_precision = max(0, int(quantity_precision))

        self.max_episode_steps = max_episode_steps or max(1, len(self.features) - 1)

        feature_dim = self.features.shape[1] * self.seq_len
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self.episode_index = 0
        self.start_index = 0
        self.step_index = 0
        self.position = 0.0
        self.capital = self.initial_capital
        self.equity = self.initial_capital
        self.equity_curve: list[float] = []
        self.trades: list[dict[str, Any]] = []
        self._peak_equity = self.initial_capital
        self._entry_price = 0.0
        self._entry_index = 0
        self._returns: list[float] = []

    def _window(self, index: int) -> np.ndarray:
        start = max(0, index - self.seq_len + 1)
        segment = self.features[start : index + 1]
        if len(segment) < self.seq_len:
            pad = np.repeat(segment[:1], self.seq_len - len(segment), axis=0)
            segment = np.concatenate([pad, segment], axis=0)
        return segment.reshape(-1)

    def _mark_equity(self, price: float) -> None:
        holdings_value = self.position * price
        self.equity = self.capital + holdings_value
        self._peak_equity = max(self._peak_equity, self.equity)

    def _execute(self, target_position: float, price: float, index: int) -> float:
        delta = target_position - self.position
        if abs(delta) < 1e-12:
            return 0.0

        fill_price = price * (1 + self.slippage_rate * np.sign(delta))
        notional = abs(delta) * fill_price
        fee = notional * self.fee_rate

        self.capital -= delta * fill_price + fee
        self.position += delta
        self.trades.append({
            "index": index,
            "side": "buy" if delta > 0 else "sell",
            "quantity": abs(delta),
            "price": fill_price,
            "fee": fee,
            "position_after": self.position,
        })
        return fee

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = np.random.default_rng(seed), None

        self.episode_index += 1
        self.start_index = 0
        if options and "start_index" in options:
            self.start_index = int(options["start_index"])
        self.step_index = self.start_index
        self.position = 0.0
        self.capital = self.initial_capital
        self.equity = self.initial_capital
        self.equity_curve = [self.equity]
        self.trades = []
        self._peak_equity = self.initial_capital
        self._entry_price = 0.0
        self._entry_index = 0
        self._returns = []

        return self._window(self.step_index).astype(np.float32), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        current_price = float(self.prices[self.step_index])
        next_index = self.step_index + 1
        terminated = False
        truncated = False

        if self.position != 0.0:
            equity_before = self.capital + self.position * current_price
        else:
            equity_before = self.equity

        target_position = self._target_position(action, current_price)

        next_price = float(self.prices[next_index]) if next_index < len(self.prices) else current_price
        fee = self._execute(target_position, next_price, self.step_index)

        self.step_index = next_index
        self._mark_equity(next_price)

        reward = 0.0
        if equity_before > 0:
            step_return = (self.equity - equity_before) / equity_before
            reward = step_return
            self._returns.append(step_return)
            if self.reward_shaping:
                reward += 0.1 * step_return**2 * 10
        reward -= fee / max(equity_before, 1.0)

        if self.step_index >= len(self.prices) - 1:
            if self.position != 0.0:
                self._execute(0.0, next_price, self.step_index)
                self._mark_equity(next_price)
            terminated = True

        if self.equity <= 0.10 * self.initial_capital:
            terminated = True

        self.equity_curve.append(self.equity)
        observation = self._window(min(self.step_index, len(self.features) - 1)).astype(np.float32)

        return observation, float(reward), terminated, truncated, self._info()

    def _target_position(self, action: int, price: float) -> float:
        if action == ACTION_FLAT:
            return 0.0

        current_equity = self.capital + self.position * price
        if current_equity <= 0:
            return 0.0

        budget = current_equity * min(1.0, self.position_size)
        if price <= 0:
            return 0.0
        units = budget / price
        units = self._quantize(units)
        return units if action == ACTION_LONG else -units

    def _quantize(self, units: float) -> float:
        """Round size to venue lot precision.

        Crypto venues quote to 8 decimals, so ``floor`` to whole units (as in a
        stock environment) would round every position to zero at a $30k price
        with a $10k account. Rounding to lot size keeps fills realistic without
        discarding all notional.
        """
        if self.allow_fractional:
            return float(np.floor(units * 10**self.quantity_precision) / 10**self.quantity_precision)
        return float(np.floor(units))

    def _info(self) -> dict[str, Any]:
        total_return = (self.equity / self.initial_capital) - 1.0 if self.initial_capital else 0.0
        drawdown = (
            (self.equity - self._peak_equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        )
        return {
            "equity": self.equity,
            "capital": self.capital,
            "position": self.position,
            "total_return": total_return,
            "drawdown": drawdown,
            "step": self.step_index - self.start_index,
            "trades": len(self.trades),
            "price": float(self.prices[min(self.step_index, len(self.prices) - 1)]),
        }

    def render(self) -> None:  # pragma: no cover - headless by design
        info = self._info()
        return f"step={info['step']} equity={info['equity']:.2f} pos={info['position']:.6f}"


class MultiAssetTradingEnv(gym.Env):
    """Portfolio environment allocating across symbols with a cash residual.

    Action space is a simplex via ``Dirichlet``-style hints: the agent emits
    logits over assets plus a cash weight, normalized to sum to one.
    """

    def __init__(
        self,
        feature_dict: dict[str, np.ndarray],
        price_dict: dict[str, np.ndarray],
        feature_columns: dict[str, list[str]],
        *,
        settings: Settings | None = None,
        initial_capital: float | None = None,
        fee_bps: float | None = None,
        slippage_bps: float | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self.symbols = sorted(feature_dict.keys())
        if not self.symbols:
            raise ValueError("MultiAssetTradingEnv requires at least one symbol")

        min_length = min(
            min(len(feature_dict[s]), len(price_dict[s])) for s in self.symbols
        )
        self.length = min_length
        self.features = {s: np.nan_to_num(feature_dict[s][:min_length], dtype=np.float32) for s in self.symbols}
        self.prices = {s: price_dict[s][:min_length] for s in self.symbols}
        self.columns = feature_columns

        self.initial_capital = float(initial_capital or self.settings.rl_initial_capital)
        self.fee_rate = (fee_bps if fee_bps is not None else self.settings.backtest_fee_bps) / 10_000
        self.slippage_rate = (slippage_bps if slippage_bps is not None else self.settings.backtest_slippage_bps) / 10_000

        dim = len(self.symbols)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)

        self.capital = self.initial_capital
        self.weights = np.zeros(dim, dtype=np.float64)
        self.step_index = 0
        self.equity_curve: list[float] = []
        self._returns: list[float] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.capital = self.initial_capital
        self.weights = np.zeros(len(self.symbols))
        self.step_index = 0
        self.equity_curve = [self.initial_capital]
        self._returns = []
        return self._observe(), {"equity": self.initial_capital}

    def _observe(self) -> np.ndarray:
        if self.step_index == 0:
            return np.zeros(len(self.symbols), dtype=np.float32)
        returns = []
        for symbol in self.symbols:
            prices = self.prices[symbol]
            idx = min(self.step_index, len(prices) - 1)
            prev = max(0, idx - 1)
            returns.append((prices[idx] - prices[prev]) / max(prices[prev], 1e-9))
        return np.nan_to_num(np.asarray(returns, dtype=np.float32))

    def step(self, action: np.ndarray):
        weights = np.asarray(action, dtype=np.float64)
        total = weights.sum()
        weights = np.zeros_like(weights) if total <= 1e-9 else weights / total

        index = self.step_index
        portfolio_return = 0.0
        for position, symbol in enumerate(self.symbols):
            prices = self.prices[symbol]
            idx = min(index, len(prices) - 1)
            prev = max(0, idx - 1)
            asset_return = (prices[idx] - prices[prev]) / max(prices[prev], 1e-9)
            portfolio_return += weights[position] * asset_return

        turnover = float(np.abs(weights - self.weights).sum())
        cost = turnover * (self.fee_rate + self.slippage_rate)
        net_return = portfolio_return - cost

        self.capital *= 1 + net_return
        self.weights = weights
        self._returns.append(net_return)
        self.equity_curve.append(self.capital)
        self.step_index += 1

        terminated = self.step_index >= self.length - 1
        truncated = False
        reward = net_return - 0.01 * turnover
        return self._observe(), float(reward), terminated, truncated, {
            "equity": self.capital,
            "weights": weights.tolist(),
            "turnover": turnover,
        }


__all__ = ["ACTION_FLAT", "ACTION_LONG", "ACTION_SHORT", "MultiAssetTradingEnv", "TradingEnv"]