from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from one_year_sim import align_gate_to_decision, run_portfolio  # noqa: E402

SYMBOLS = ["AAA_USDT", "BBB_USDT", "CCC_USDT", "DDD_USDT"]
START = pd.Timestamp("2025-01-01", tz="UTC")
INTERVAL = "1d"
CAPITAL = 1000.0


def _stamp(day: int) -> str:
    """ISO timestamp of daily bar ``day``, counting bar 0 as 2025-01-01."""
    return (START + pd.Timedelta(days=day)).isoformat()


def _daily_ohlcv(symbols=SYMBOLS, n: int = 40) -> pd.DataFrame:
    """Daily bars on a flat tape, so weight changes are the only source of fills.

    With a constant price, every round trip's gross P&L is exactly the slippage it
    paid, which makes the cost assertions below exact rather than approximate.
    """
    times = pd.date_range(START, periods=n, freq="1D", tz="UTC")
    frames = []
    for i, symbol in enumerate(symbols):
        close = 100.0 * (1.0 + i)
        frames.append(pd.DataFrame({
            "ts": times, "symbol": symbol, "exchange": "gate", "interval": INTERVAL,
            "open": close, "high": close, "low": close, "close": close,
            "volume": 1.0, "quote_volume": close, "trades": 1,
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def daily_store(settings):
    from cqip.storage import ParquetStore

    store = ParquetStore(settings=settings)
    store.write("ohlcv", _daily_ohlcv(), dedupe_on=("ts", "symbol", "exchange"), mode="overwrite")
    return store


def _grid(values: dict[int, dict[str, float]], n: int = 40) -> pd.DataFrame:
    """Frame stamped on the bar grid; unspecified bars stay flat."""
    stamps = pd.date_range(START, periods=n, freq="1D", tz="UTC")
    frame = pd.DataFrame(0.0, index=stamps, columns=SYMBOLS)
    for bar, row in values.items():
        for symbol, value in row.items():
            frame.iloc[bar, frame.columns.get_loc(symbol)] = value
    return frame


def _run(store, weights, **kwargs):
    params = {"initial_capital": CAPITAL, "fee_bps": 10.0, "slippage_bps": 5.0}
    params.update(kwargs)
    return run_portfolio(store, SYMBOLS, INTERVAL, weights, **params)


def _fills(result, symbol: str) -> list[dict]:
    return [t for t in result["trades"] if t["symbol"] == symbol]


class TestGateAlignment:
    def test_row_t_carries_the_forecast_from_row_t_minus_one(self):
        stamps = pd.date_range(START, periods=4, freq="1D", tz="UTC")
        grid = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}, index=stamps)
        aligned = align_gate_to_decision(grid, stamps)

        assert np.isnan(aligned["A"].iloc[0])
        assert aligned["A"].iloc[1] == 1.0
        assert aligned["A"].iloc[3] == 3.0

    def test_gate_admits_a_fill_on_the_forecast_from_the_prior_bar(self, daily_store):
        """Weight row 1 is acted on at open(row 2), so row 1's forecast must admit it."""
        weights = _grid({1: {"AAA_USDT": 0.5}})
        moves = _grid({1: {"AAA_USDT": 0.01}, 2: {"AAA_USDT": 0.0}})
        # "hold" mode suppresses the exit, so the fill count isolates admission.
        result = _run(daily_store, weights, expected_move=moves,
                      min_expected_move_bps=50.0, gate_mode="hold")

        assert len(_fills(result, "AAA_USDT")) == 1
        assert _fills(result, "AAA_USDT")[0]["ts"] == _stamp(2)

    def test_a_forecast_published_after_the_fill_cannot_admit_it(self, daily_store):
        """The mirror case: a strong forecast on row 2 must not fill at open(row 2)."""
        weights = _grid({i: {"AAA_USDT": 0.5} for i in range(1, 9)})
        moves = _grid({1: {"AAA_USDT": 0.0}, **{i: {"AAA_USDT": 0.02} for i in range(2, 9)}})
        result = _run(daily_store, weights, expected_move=moves,
                      min_expected_move_bps=50.0, gate_mode="hold")

        fills = _fills(result, "AAA_USDT")
        # The first fill must be one bar later than a look-ahead would have given.
        assert fills[0]["ts"] == _stamp(3)
        assert not [t for t in fills if t["ts"] == _stamp(2)]

    def test_flat_gate_mode_closes_a_position_whose_conviction_dies(self, daily_store):
        weights = _grid({1: {"AAA_USDT": 0.5}})
        moves = _grid({1: {"AAA_USDT": 0.01}, 2: {"AAA_USDT": 0.0}})
        result = _run(daily_store, weights, expected_move=moves,
                      min_expected_move_bps=50.0, gate_mode="flat")

        assert [t["side"] for t in _fills(result, "AAA_USDT")] == ["buy", "sell"]
        assert result["n_round_trips"] == 1

    def test_hold_gate_mode_keeps_the_position(self, daily_store):
        weights = _grid({1: {"AAA_USDT": 0.5}})
        moves = _grid({1: {"AAA_USDT": 0.01}, 2: {"AAA_USDT": 0.0}})
        result = _run(daily_store, weights, expected_move=moves,
                      min_expected_move_bps=50.0, gate_mode="hold")

        assert [t["side"] for t in _fills(result, "AAA_USDT")] == ["buy"]

    def test_gate_rejects_a_position_the_forecast_does_not_support(self, daily_store):
        """A strong forecast everywhere still cannot open what the weights do not ask for."""
        weights = _grid({1: {"AAA_USDT": 0.0}})
        moves = _grid({i: {"AAA_USDT": 0.05} for i in range(1, 6)})
        result = _run(daily_store, weights, expected_move=moves, min_expected_move_bps=10.0)

        assert result.get("error") == "no active positions"

    def test_a_sub_threshold_forecast_blocks_the_entry_it_gates(self, daily_store):
        weights = _grid({1: {"AAA_USDT": 0.5}})
        moves = _grid({1: {"AAA_USDT": 0.0}, 2: {"AAA_USDT": 0.0}})
        result = _run(daily_store, weights, expected_move=moves,
                      min_expected_move_bps=50.0, gate_mode="hold")

        assert result["trades"] == []


class TestRestraintControls:
    def test_min_hold_bars_blocks_a_resize_before_the_position_ages(self, daily_store):
        # Buy on bar 1's decision, trim on bar 2's, exit on bar 3's.
        weights = _grid({1: {"AAA_USDT": 0.5}, 2: {"AAA_USDT": 0.25}})
        loose = _run(daily_store, weights, min_hold_bars=0)
        tight = _run(daily_store, weights, min_hold_bars=2)

        assert len(loose["trades"]) == 3          # buy, trim, exit
        assert len(tight["trades"]) == 2          # trim suppressed, exit allowed
        assert tight["restraint"]["blocked_by_min_hold"] >= 1

    def test_min_hold_bars_can_freeze_a_position_that_never_flattens(self, daily_store):
        weights = _grid({1: {"AAA_USDT": 0.5}, 2: {"AAA_USDT": 0.25}})
        frozen = _run(daily_store, weights, min_hold_bars=99)

        assert len(frozen["trades"]) == 1

    def test_fill_budget_is_a_ceiling_and_never_a_target(self, daily_store):
        weights = _grid({1: {"AAA_USDT": 0.2, "BBB_USDT": -0.2, "CCC_USDT": 0.2}})
        capped = _run(daily_store, weights, max_fills_per_day=2)
        unlimited = _run(daily_store, weights, max_fills_per_day=None)

        assert len(unlimited["trades"]) == 6      # three entries, three exits
        assert len(capped["trades"]) == 4         # two per day, on two separate days
        assert capped["restraint"]["blocked_by_budget"] >= 1

    def test_no_day_ever_exceeds_the_fill_budget(self, daily_store):
        weights = _grid({
            i: {"AAA_USDT": 0.2, "BBB_USDT": -0.2, "CCC_USDT": 0.2, "DDD_USDT": 0.2}
            for i in (1, 3, 5)
        })
        result = _run(daily_store, weights, max_fills_per_day=3)

        per_day: dict[str, int] = {}
        for trade in result["trades"]:
            day = trade["ts"][:10]
            per_day[day] = per_day.get(day, 0) + 1
        assert per_day, "expected some fills"
        assert max(per_day.values()) <= 3

    def test_budget_counts_fills_per_day_and_resets(self, daily_store):
        weights = _grid({1: {"AAA_USDT": 0.2}, 3: {"AAA_USDT": 0.4}})
        result = _run(daily_store, weights, max_fills_per_day=1)

        days = [t["ts"][:10] for t in result["trades"]]
        assert days == sorted(days)
        assert len(set(days)) == len(days)        # never two fills on one day

    def test_reducing_a_position_outranks_a_larger_opening(self, daily_store):
        """When the budget binds, an exit must never lose to a fresh entry.

        AAA's exit is a quarter the size of the CCC entry it competes with, so a
        size-ordered budget would spend the fill on the entry and strand the
        position. Costs are zeroed so that holding BBB constant produces no
        incidental trims to confuse the count.
        """
        weights = _grid({
            1: {"AAA_USDT": 0.05, "BBB_USDT": 0.05},
            2: {"BBB_USDT": 0.05, "CCC_USDT": 0.5},
        })
        result = _run(daily_store, weights, max_fills_per_day=1,
                      fee_bps=0.0, slippage_bps=0.0)

        second_day = [t for t in result["trades"] if t["ts"].startswith("2025-01-04")]
        assert len(second_day) == 1
        assert second_day[0]["symbol"] == "AAA_USDT"
        assert second_day[0]["side"] == "sell"

    def test_defaults_reproduce_the_ungated_run(self, daily_store):
        weights = _grid({1: {"AAA_USDT": 0.3}, 2: {"AAA_USDT": 0.3, "BBB_USDT": -0.3}})
        baseline = _run(daily_store, weights)
        explicit = _run(daily_store, weights, rebalance_band_bps=0.0,
                         min_expected_move_bps=0.0, min_hold_bars=0,
                         max_fills_per_day=None, gate_mode="flat")

        assert baseline["metrics"]["total_return"] == pytest.approx(
            explicit["metrics"]["total_return"]
        )
        assert baseline["total_fees"] == pytest.approx(explicit["total_fees"])

    def test_symbols_flat_for_the_whole_window_do_not_break_pricing(self, daily_store):
        """A never-held symbol still needs a price entry, or validation raises."""
        weights = _grid({1: {"AAA_USDT": 0.3}})
        result = _run(daily_store, weights)

        assert len(result["trades"]) == 2
        assert result["unpriceable_bar_symbols"] == 0


class TestRoundTripAccounting:
    def test_a_long_to_short_flip_closes_one_episode_and_opens_another(self, daily_store):
        """Crossing straight through zero is two decisions, not one round trip."""
        weights = _grid({1: {"AAA_USDT": 0.3}, 2: {"AAA_USDT": -0.3}})
        result = _run(daily_store, weights)

        trips = result["round_trips"]
        assert len(trips) == 2
        assert [t["side"] for t in trips] == ["long", "short"]
        assert trips[0]["exit_ts"] == trips[1]["entry_ts"]
        assert result["n_round_trips"] == 2

    def test_a_trim_does_not_create_a_second_round_trip(self, daily_store):
        weights = _grid({1: {"AAA_USDT": 0.5}, 2: {"AAA_USDT": 0.25}})
        result = _run(daily_store, weights)

        assert len(result["trades"]) == 3          # buy, trim, exit
        assert result["n_round_trips"] == 1

    def test_round_trip_return_is_net_of_the_fees_it_paid(self, daily_store):
        weights = _grid({1: {"AAA_USDT": 0.3}, 2: {"AAA_USDT": 0.0}})
        result = _run(daily_store, weights)

        trip = result["round_trips"][0]
        # A flat tape means the whole gross loss is slippage, and fees sit on top.
        assert trip["gross_pnl"] < 0.0
        assert trip["net_pnl"] < trip["gross_pnl"]
        assert trip["net_return"] < trip["gross_return"]

    def test_holding_period_is_reported_in_wall_clock_hours(self, daily_store):
        weights = _grid({1: {"AAA_USDT": 0.3}, 2: {"AAA_USDT": 0.0}})
        result = _run(daily_store, weights)

        trip = result["round_trips"][0]
        assert trip["bars_held"] == 1
        assert trip["hours_held"] == pytest.approx(24.0)
