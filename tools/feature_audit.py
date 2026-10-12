"""Feature / signal gap audit.

Diagnosis only. Nothing here trains a model, tunes a threshold, or optimises a
portfolio. The question is whether the features the model actually receives
carry information about forward returns at all, and if not, which information
domain is missing.

Every statistic is computed on the same corrected forward-return definition used
by the label ablation and horizon discovery (log(P[t+h]/P[t])), on the same OOS
window, with no row deletion beyond the trailing-horizon trim.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import ab_direction_head as ab  # noqa: E402

from cqip.features import build_feature_frame  # noqa: E402
from cqip.storage import ParquetStore  # noqa: E402

HORIZONS = [1, 2, 4, 6, 12, 24]
SIM_START = "2025-01-01"

# --------------------------------------------------------------------------- #
# family assignment for the features actually present
#
# Families are declared only for columns that exist. Empty families are reported
# explicitly, because "this domain is absent" is itself an audit finding.
# --------------------------------------------------------------------------- #
FAMILIES: dict[str, list[str]] = {
    "raw_returns": [
        "returns_1",
        "returns_3",
        "returns_6",
        "returns_12",
        "returns_24",
    ],
    "momentum_trend": [
        "momentum_5",
        "momentum_10",
        "momentum_20",
        "momentum_60",
        "ema_fast_slow_spread",
        "sma_fast_slow_spread",
        "macd_line",
        "macd_signal",
        "macd_hist",
    ],
    "mean_reversion": [
        "rsi_7",
        "rsi_14",
        "rsi_21",
        "bb_percent_b",
        "bb_bandwidth",
        "bb_width",
        "zscore_close_20",
        "close_position_14",
        "drawdown_60",
        "hurst_approx",
    ],
    "volatility": [
        "realized_vol_5",
        "realized_vol_10",
        "realized_vol_20",
        "realized_vol_60",
        "vol_of_vol_20",
        "atr_14_pct",
        "high_low_range_14",
        "range_expansion_10",
    ],
    "volume_flow": [
        "log_volume_change",
        "volume_zscore_20",
        "dollar_volume_zscore_20",
        "obv_slope_10",
    ],
    "bar_shape_vwap": [
        "vwap_deviation",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "body_ratio",
        "gap_from_prev_close",
    ],
    "raw_levels": ["open", "high", "low", "close", "volume"],
}

# Domains the audit must explicitly report as present-or-absent.
PROBED_ABSENT_DOMAINS = [
    "cross_sectional_relative_strength",
    "market_regime_btc_context",
    "market_breadth",
    "correlation_cross_asset",
    "sentiment_news",
    "onchain_network",
    "derivatives_funding_open_interest",
    "orderbook_microstructure",
    "time_calendar_seasonality",
]

FEATURE_META: dict[str, dict[str, str]] = {
    "returns_1": {"formula": "sum of 1-bar log returns", "window": "1h", "scope": "own-symbol"},
    "returns_3": {"formula": "sum of 3-bar log returns", "window": "3h", "scope": "own-symbol"},
    "returns_6": {"formula": "sum of 6-bar log returns", "window": "6h", "scope": "own-symbol"},
    "returns_12": {"formula": "sum of 12-bar log returns", "window": "12h", "scope": "own-symbol"},
    "returns_24": {"formula": "sum of 24-bar log returns", "window": "24h", "scope": "own-symbol"},
    "log_volume_change": {
        "formula": "diff of log1p(volume)",
        "window": "1h",
        "scope": "own-symbol",
    },
    "volume_zscore_20": {
        "formula": "(vol - sma20(vol)) / std20(vol)",
        "window": "20h",
        "scope": "own-symbol",
    },
    "dollar_volume_zscore_20": {
        "formula": "(vol*close - sma20) / std20",
        "window": "20h",
        "scope": "own-symbol",
    },
    "obv_slope_10": {
        "formula": "diff10 of cumulative signed volume",
        "window": "10h",
        "scope": "own-symbol",
    },
    "momentum_5": {"formula": "close/close[-5]-1", "window": "5h", "scope": "own-symbol"},
    "momentum_10": {"formula": "close/close[-10]-1", "window": "10h", "scope": "own-symbol"},
    "momentum_20": {"formula": "close/close[-20]-1", "window": "20h", "scope": "own-symbol"},
    "momentum_60": {"formula": "close/close[-60]-1", "window": "60h", "scope": "own-symbol"},
    "ema_fast_slow_spread": {
        "formula": "(ema9-ema21)/ema21",
        "window": "9/21h",
        "scope": "own-symbol",
    },
    "sma_fast_slow_spread": {
        "formula": "(sma10-sma30)/sma30",
        "window": "10/30h",
        "scope": "own-symbol",
    },
    "rsi_7": {"formula": "Wilder RSI", "window": "7h", "scope": "own-symbol"},
    "rsi_14": {"formula": "Wilder RSI", "window": "14h", "scope": "own-symbol"},
    "rsi_21": {"formula": "Wilder RSI", "window": "21h", "scope": "own-symbol"},
    "macd_line": {"formula": "(ema12-ema26)/close", "window": "12/26h", "scope": "own-symbol"},
    "macd_signal": {"formula": "ema9(macd_line)/close", "window": "9h", "scope": "own-symbol"},
    "macd_hist": {
        "formula": "(macd_line-signal)/close",
        "window": "12/26/9h",
        "scope": "own-symbol",
    },
    "bb_percent_b": {
        "formula": "(close-lower)/(upper-lower)",
        "window": "20h",
        "scope": "own-symbol",
    },
    "bb_bandwidth": {"formula": "(upper-lower)/mid", "window": "20h", "scope": "own-symbol"},
    "bb_width": {
        "formula": "(upper-lower)/mid (dup of bandwidth)",
        "window": "20h",
        "scope": "own-symbol",
    },
    "atr_14_pct": {"formula": "ATR(14)/close", "window": "14h", "scope": "own-symbol"},
    "high_low_range_14": {
        "formula": "(max14(high)-min14(low))/close",
        "window": "14h",
        "scope": "own-symbol",
    },
    "close_position_14": {
        "formula": "(close-min14)/(max14-min14)",
        "window": "14h",
        "scope": "own-symbol",
    },
    "realized_vol_5": {"formula": "std5(logret)*sqrt(365)", "window": "5h", "scope": "own-symbol"},
    "realized_vol_10": {
        "formula": "std10(logret)*sqrt(365)",
        "window": "10h",
        "scope": "own-symbol",
    },
    "realized_vol_20": {
        "formula": "std20(logret)*sqrt(365)",
        "window": "20h",
        "scope": "own-symbol",
    },
    "realized_vol_60": {
        "formula": "std60(logret)*sqrt(365)",
        "window": "60h",
        "scope": "own-symbol",
    },
    "vol_of_vol_20": {"formula": "|rv20.diff5| / rv20", "window": "20h", "scope": "own-symbol"},
    "drawdown_60": {
        "formula": "(close-max60(close))/max60(close)",
        "window": "60h",
        "scope": "own-symbol",
    },
    "zscore_close_20": {"formula": "(close-sma20)/std20", "window": "20h", "scope": "own-symbol"},
    "hurst_approx": {
        "formula": "rolling rescaled-range Hurst on log price",
        "window": "100h",
        "scope": "own-symbol",
    },
    "vwap_deviation": {
        "formula": "(close - expanding VWAP since series start)/VWAP",
        "window": "expanding",
        "scope": "own-symbol",
    },
    "upper_wick_ratio": {
        "formula": "(high-max(open,close))/(high-low)",
        "window": "1h",
        "scope": "own-symbol",
    },
    "lower_wick_ratio": {
        "formula": "(min(open,close)-low)/(high-low)",
        "window": "1h",
        "scope": "own-symbol",
    },
    "body_ratio": {"formula": "(close-open)/(high-low)", "window": "1h", "scope": "own-symbol"},
    "gap_from_prev_close": {
        "formula": "(open-close[-1])/close[-1]",
        "window": "1h",
        "scope": "own-symbol",
    },
    "range_expansion_10": {
        "formula": "((high-low)-sma10(high-low))/sma10",
        "window": "10h",
        "scope": "own-symbol",
    },
    "open": {"formula": "raw bar open (price level)", "window": "1h", "scope": "own-symbol"},
    "high": {"formula": "raw bar high (price level)", "window": "1h", "scope": "own-symbol"},
    "low": {"formula": "raw bar low (price level)", "window": "1h", "scope": "own-symbol"},
    "close": {"formula": "raw bar close (price level)", "window": "1h", "scope": "own-symbol"},
    "volume": {"formula": "raw bar base volume (level)", "window": "1h", "scope": "own-symbol"},
}

SAME_BAR_FEATURES = {"open", "high", "low", "close", "volume"}


# --------------------------------------------------------------------------- #
# fast rank-correlation machinery with day-block resampling
# --------------------------------------------------------------------------- #


@dataclass
class DayStats:
    """Sufficient statistics per UTC day for a rank-Pearson correlation.

    Spearman on pooled data equals Pearson on ranks, and Pearson is recoverable
    from additive sums, so a whole-day bootstrap or block permutation costs
    O(days) per draw instead of O(rows). That is what makes 276 tests x 2000
    draws affordable.
    """

    n: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    sxy: np.ndarray
    sxx: np.ndarray
    syy: np.ndarray

    @staticmethod
    def build(x: np.ndarray, y: np.ndarray, day: np.ndarray) -> DayStats:
        uniq, inv = np.unique(day, return_inverse=True)
        d = len(uniq)
        n = np.bincount(inv, minlength=d).astype(np.float64)
        sx = np.bincount(inv, weights=x, minlength=d)
        sy = np.bincount(inv, weights=y, minlength=d)
        sxy = np.bincount(inv, weights=x * y, minlength=d)
        sxx = np.bincount(inv, weights=x * x, minlength=d)
        syy = np.bincount(inv, weights=y * y, minlength=d)
        return DayStats(n, sx, sy, sxy, sxx, syy)

    @staticmethod
    def _r(n, sx, sy, sxy, sxx, syy):
        cov = n * sxy - sx * sy
        vx = n * sxx - sx * sx
        vy = n * syy - sy * sy
        with np.errstate(invalid="ignore", divide="ignore"):
            r = cov / np.sqrt(np.clip(vx * vy, 1e-300, None))
        return np.where((vx > 0) & (vy > 0), r, np.nan)

    def pooled(self) -> float:
        return float(
            self._r(
                self.n.sum(),
                self.sx.sum(),
                self.sy.sum(),
                self.sxy.sum(),
                self.sxx.sum(),
                self.syy.sum(),
            )
        )

    def bootstrap(self, draws: int, seed: int) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        d = len(self.n)
        idx = rng.integers(0, d, size=(draws, d))
        r = self._r(
            self.n[idx].sum(1),
            self.sx[idx].sum(1),
            self.sy[idx].sum(1),
            self.sxy[idx].sum(1),
            self.sxx[idx].sum(1),
            self.syy[idx].sum(1),
        )
        r = r[np.isfinite(r)]
        if r.size == 0:
            return float("nan"), float("nan")
        return float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))

    def permutation_p(self, draws: int, seed: int) -> float:
        """Deprecated: kept only so callers fail loudly rather than silently.

        Permuting day blocks and re-deriving Pearson from additive day sums is
        a no-op, because every additive sum in the formula is permutation
        invariant. Use :func:`block_permutation_p`, which permutes the rank
        series itself.
        """
        raise NotImplementedError(
            "additive day summaries cannot be block-permuted; use block_permutation_p"
        )


def block_permutation_p(
    xr: np.ndarray, yr: np.ndarray, day: np.ndarray, draws: int, seed: int
) -> float:
    """Day-block permutation p-value for a Spearman correlation.

    Whole UTC days of the target are reshuffled, which destroys the day-to-day
    alignment while preserving the serial correlation inside each day, so the
    null respects the fact that hourly crypto returns are not independent.

    The correlation is recomputed from the permuted rank series rather than from
    per-day additive summaries: permuting blocks and re-deriving Pearson from
    sums leaves every sum unchanged, which would make the null identical to the
    observation. Only the cross term changes under a permutation, and because
    both rank series have permutation-invariant mean and variance, each draw is a
    single dot product against precomputed centred ranks.
    """
    n = xr.size
    if n < 10 or yr.size != n:
        return float("nan")
    xc = xr - xr.mean()
    yc = yr - yr.mean()
    sdx = float(np.sqrt(np.dot(xc, xc)))
    sdy = float(np.sqrt(np.dot(yc, yc)))
    if sdx <= 0.0 or sdy <= 0.0:
        return float("nan")
    obs = abs(float(np.dot(xc, yc)) / (n * sdx * sdy))

    order = np.argsort(day, kind="stable")
    _, starts = np.unique(day[order], return_index=True)
    blocks = np.split(order, starts[1:])
    if len(blocks) < 2:
        return float("nan")

    rng = np.random.default_rng(seed)
    count = 0
    valid = 0
    for _ in range(draws):
        pick = rng.permutation(len(blocks))
        idx = blocks[pick[0]]
        for b in pick[1:]:
            idx = np.concatenate((idx, blocks[b]))
        r = abs(float(np.dot(xc, yc[idx])) / (n * sdx * sdy))
        if np.isfinite(r):
            valid += 1
            if r >= obs - 1e-15:
                count += 1
    if valid == 0:
        return float("nan")
    return (count + 1) / (valid + 1)


def rankdata(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks, NaN-preserving."""
    out = np.full(a.shape, np.nan)
    mask = np.isfinite(a)
    if mask.any():
        out[mask] = stats.rankdata(a[mask])
    return out


def auc_from_ranks(x: np.ndarray, y_positive: np.ndarray) -> float:
    """Mann-Whitney AUC of x predicting y, via ranks. NaN-safe, NaN excluded."""
    mask = np.isfinite(x) & np.isfinite(y_positive)
    if mask.sum() < 10:
        return float("nan")
    r = rankdata(x[mask])
    pos = y_positive[mask] > 0.5
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, NaN-preserving."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.clip(adj, 0.0, 1.0)
    out[ok] = res
    return out


# --------------------------------------------------------------------------- #
# panel construction
# --------------------------------------------------------------------------- #


@dataclass
class Panel:
    """Per-symbol OOS feature panel with corrected forward returns.

    Features are built exactly as production builds them (build_feature_frame on
    the same candle slice), so the audit measures the representation the model
    really receives rather than an idealised copy of it.
    """

    symbols: list[str]
    feature_columns: list[str]
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    close: dict[str, pd.Series] = field(default_factory=dict)
    train_stats: dict[str, pd.DataFrame] = field(default_factory=dict)
    btc_symbol: str = "BTC_USDT"

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        first = next(iter(self.frames.values()))
        return pd.DatetimeIndex(first["ts"].to_numpy())

    @property
    def union_timestamps(self) -> pd.DatetimeIndex:
        grids = [pd.DatetimeIndex(f["ts"].to_numpy()) for f in self.frames.values()]
        union = grids[0]
        for g in grids[1:]:
            union = union.union(g)
        return union.sort_values()

    @property
    def complete_timestamps(self) -> pd.DatetimeIndex:
        grids = [pd.DatetimeIndex(f["ts"].to_numpy()) for f in self.frames.values()]
        common = grids[0]
        for g in grids[1:]:
            common = common.intersection(g)
        return common.sort_values()

    def stacked(self) -> pd.DataFrame:
        parts = []
        for sym, frame in self.frames.items():
            part = frame.copy()
            part["symbol"] = sym
            parts.append(part)
        return pd.concat(parts, ignore_index=True)


def load_panel(
    store: ParquetStore,
    symbols: list[str],
    interval: str,
    max_horizon: int,
    cutoff: int,
    warmup: int = 600,
) -> Panel:
    """Build the OOS panel used by every statistic in the audit.

    The split mirrors production: train on everything before ``cutoff`` (the
    scaler and feature normalisation are fitted there and nowhere else), then
    evaluate strictly after it. Forward returns are recomputed from ``close`` for
    every horizon so no target column can leak between horizons.
    """
    panel = Panel(symbols=symbols, feature_columns=[])
    ref_cols: list[str] | None = None

    for sym in symbols:
        candles = store.read(
            "ohlcv",
            filters=[("interval", "==", interval)],
            symbols=[sym],
            columns=["ts", "open", "high", "low", "close", "volume", "symbol"],
        )
        if candles.empty:
            continue
        candles = candles.sort_values("ts").reset_index(drop=True)
        if interval == "1h":
            candles = candles[candles["ts"] >= SIM_START].reset_index(drop=True)
        if len(candles) < warmup + max_horizon + 100:
            continue

        frame, cols = build_feature_frame(candles)
        if ref_cols is None:
            ref_cols = list(cols)
            panel.feature_columns = [
                c
                for c in cols
                if c in FAMILIES["raw_levels"] or any(c in v for v in FAMILIES.values())
            ]
        missing = [c for c in ref_cols if c not in frame.columns]
        if missing:
            for c in missing:
                frame[c] = np.nan
        cols = [c for c in ref_cols if c in frame.columns]
        panel.feature_columns = list(cols)

        frame["close_f"] = frame["close"].astype(float)
        # Trim only the trailing max_horizon bars so every horizon is finite and
        # every symbol keeps an identical timestamp grid.
        frame = frame.iloc[: len(frame) - max_horizon].reset_index(drop=True)
        frame = frame.replace([np.inf, -np.inf], np.nan)

        # Per-symbol train-window statistics, used to z-score features without
        # letting the evaluation window inform the normalisation.
        train = frame.iloc[:cutoff]
        stats_df = pd.DataFrame(
            {
                "mean": train[cols].mean(),
                "std": train[cols].std(ddof=1).replace(0.0, 1.0),
            }
        )
        panel.train_stats[sym] = stats_df

        oos = frame.iloc[cutoff:].reset_index(drop=True)
        oos = oos.dropna(subset=["close_f"]).reset_index(drop=True)
        panel.frames[sym] = oos
        panel.close[sym] = oos["close_f"]

    return panel


def forward_returns(panel: Panel, horizon: int) -> dict[str, np.ndarray]:
    """log(P[t+h]/P[t]) for OOS rows, aligned to each symbol's frame length."""
    out: dict[str, np.ndarray] = {}
    for sym, close in panel.close.items():
        values = close.to_numpy(dtype=float)
        n = len(values)
        fwd = np.full(n, np.nan, dtype=float)
        if n > horizon:
            fwd[: n - horizon] = np.log(values[horizon:] / values[: n - horizon])
        out[sym] = fwd
    return out


def target_for(panel: Panel, pooled: pd.DataFrame, horizon: int) -> np.ndarray:
    """Forward returns aligned to ``pooled`` row order by an explicit join.

    Join is on (ts, symbol) because different symbols may have slightly
    different OOS lengths after cleaning (the stored panel is aligned by the
    union of finite rows for each symbol, not by forcing padding).
    """
    fwd = forward_returns(panel, horizon)
    parts = []
    for sym in panel.symbols:
        if sym not in fwd:
            continue
        ts = panel.frames[sym]["ts"].to_numpy()
        parts.append(pd.DataFrame({"ts": ts, "symbol": sym, "fwd": fwd[sym]}))
    if not parts:
        return np.full(len(pooled), np.nan)
    long = pd.concat(parts, ignore_index=True)
    merged = pooled[["ts", "symbol"]].merge(long, on=["ts", "symbol"], how="left", sort=False)
    if len(merged) != len(pooled):
        raise ValueError("target join changed the row count")
    return merged["fwd"].to_numpy(dtype=float)


def alignment_report(panel: Panel) -> dict[str, Any]:
    """Explicit OOS grid accounting so unequal symbol coverage is visible.

    Symbols in this universe can hold different numbers of finite OOS rows, so
    the audit never assumes the first symbol's grid is universal. Cross-sectional
    statistics use the complete-timestamp intersection and report how many
    timestamps that costs.
    """
    grids = {s: pd.DatetimeIndex(panel.frames[s]["ts"].to_numpy()) for s in panel.symbols}
    union = panel.union_timestamps
    complete = panel.complete_timestamps
    return {
        "rows_per_symbol": {s: int(len(g)) for s, g in grids.items()},
        "identical_grids": bool(len({tuple(g) for g in grids.values()}) == 1),
        "n_union_timestamps": int(len(union)),
        "n_complete_timestamps": int(len(complete)),
        "incomplete_timestamps": int(len(union) - len(complete)),
        "missing_rows_by_symbol": {s: int(len(union) - len(g)) for s, g in grids.items()},
    }


def wide_matrix(
    panel: Panel,
    values: dict[str, np.ndarray],
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Wide (timestamp x symbol) frame from per-symbol OOS arrays.

    Each symbol's array is indexed by its own timestamps, never by the first
    symbol's grid, because symbols can hold different numbers of OOS rows.
    ``index`` defaults to the sorted union of all timestamps so no row is
    silently dropped; callers that require complete cross-sections mask the
    resulting NaNs explicitly.
    """
    columns: dict[str, pd.Series] = {}
    all_ts: list[pd.DatetimeIndex] = []
    for sym in panel.symbols:
        if sym not in values:
            continue
        ts = pd.DatetimeIndex(panel.frames[sym]["ts"].to_numpy())
        all_ts.append(ts)
        columns[sym] = pd.Series(values[sym], index=ts, name=sym)
    if not columns:
        return pd.DataFrame()
    wide = pd.DataFrame(columns)
    if index is None:
        index = wide.index
    return wide.reindex(index)


def pooled_frame(panel: Panel) -> pd.DataFrame:
    """All OOS rows pooled, one row per (timestamp, symbol)."""
    parts = []
    for sym in panel.symbols:
        part = panel.frames[sym].copy()
        part["symbol"] = sym
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    df["day"] = pd.to_datetime(df["ts"], utc=True).dt.floor("D")
    return df.sort_values(["ts", "symbol"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# univariate information content
# --------------------------------------------------------------------------- #


def univariate_stats(
    panel: Panel, pooled: pd.DataFrame, horizon: int, draws: int, seed: int
) -> list[dict[str, Any]]:
    """Pooled time-series association of every feature with the h-bar return."""
    y = target_for(panel, pooled, horizon)
    day = pooled["day"].to_numpy()
    rows: list[dict[str, Any]] = []

    for col in panel.feature_columns:
        x = pooled[col].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x, d = x[finite], day[finite]
        if x.size < 1000 or np.nanstd(x) == 0:
            rows.append(
                {"feature": col, "horizon": horizon, "n": int(x.size), "note": "degenerate"}
            )
            continue
        yf = y[finite]
        xr = rankdata(x)
        yr_ = rankdata(yf)
        ds = DayStats.build(xr, yr_, d)
        rho = ds.pooled()
        ci = ds.bootstrap(draws, seed)
        pval = block_permutation_p(xr, yr_, d, draws, seed + 1)
        pear = float(np.corrcoef(x, yf)[0, 1])
        auc = auc_from_ranks(x, (yf > 0).astype(float))
        rows.append(
            {
                "feature": col,
                "horizon": horizon,
                "n": int(x.size),
                "spearman": rho,
                "spearman_ci": [ci[0], ci[1]],
                "perm_p": pval,
                "pearson": pear,
                "univariate_auc": auc,
                "std": float(np.nanstd(x)),
                "zero_frac": float(np.mean(np.isclose(x, 0.0))),
            }
        )
    return rows


def mutual_information_table(
    panel: Panel,
    pooled: pd.DataFrame,
    horizons: list[int],
    draws: int,
    seed: int,
    sample: int = 20000,
) -> list[dict[str, Any]]:
    """Discretised MI of feature vs forward return, with a label-shuffle null.

    A shuffle null is used rather than an asymptotic p-value because MI's null
    distribution is not chi-square for finite, autocorrelated samples.
    """
    from sklearn.feature_selection import mutual_info_regression

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    cols = panel.feature_columns
    null_draws = max(5, min(int(draws), 20))

    for horizon in horizons:
        y = target_for(panel, pooled, horizon)
        X = pooled[cols].to_numpy(dtype=float)

        idx = np.arange(len(y))
        if len(idx) > sample:
            idx = rng.choice(idx, size=sample, replace=False)
        Xs = X[idx]
        ys = y[idx]
        ok = np.isfinite(ys)
        Xs, ys = Xs[ok], ys[ok]

        mi_real = mutual_info_regression(Xs, ys, random_state=seed, discrete_features=False)
        # Several independent label shuffles so the reported p-value has usable
        # resolution; a single null draw can only ever return 0 or 1.
        null = np.empty((null_draws, len(cols)), dtype=float)
        for d in range(null_draws):
            perm = rng.permutation(ys)
            null[d] = mutual_info_regression(
                Xs, perm, random_state=seed + d, discrete_features=False
            )
        for i, col in enumerate(cols):
            col_null = null[:, i]
            col_null = col_null[np.isfinite(col_null)]
            if col_null.size == 0 or not np.isfinite(mi_real[i]):
                rows.append(
                    {
                        "feature": col,
                        "horizon": horizon,
                        "mi": float(mi_real[i]) if np.isfinite(mi_real[i]) else float("nan"),
                        "note": "null unavailable",
                    }
                )
                continue
            exceed = int(np.sum(col_null >= mi_real[i]))
            rows.append(
                {
                    "feature": col,
                    "horizon": horizon,
                    "mi": float(mi_real[i]),
                    "mi_null": float(np.mean(col_null)),
                    "mi_excess": float(mi_real[i] - np.mean(col_null)),
                    "mi_perm_p": float((exceed + 1) / (col_null.size + 1)),
                    "mi_null_draws": int(col_null.size),
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# cross-sectional vs time-series separation
# --------------------------------------------------------------------------- #


def cross_sectional_stats(
    panel: Panel, horizon: int, top_k: int, draws: int, seed: int
) -> list[dict[str, Any]]:
    """Within-timestamp rank IC: can features order coin A above coin B?

    This is the dimension the product actually trades, and it is computed
    independently of the pooled statistic so a good time-series fit cannot hide
    a total absence of cross-sectional ordering.
    """
    symbols = panel.symbols
    target = wide_matrix(panel, forward_returns(panel, horizon))
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    # Build the (time x symbol x feature) cube once. Rebuilding a wide frame per
    # feature dominated the runtime, and the cross-section validity mask only
    # depends on the target, so it is computed once here too.
    cube = np.empty((len(target.index), len(symbols), len(panel.feature_columns)))
    for j, col in enumerate(panel.feature_columns):
        Xf = wide_matrix(
            panel,
            {s: panel.frames[s][col].to_numpy(dtype=float) for s in symbols},
            index=target.index,
        )
        cube[:, :, j] = Xf.to_numpy(dtype=float)
    base_valid = target.notna().all(axis=1).to_numpy()

    for j, col in enumerate(panel.feature_columns):
        valid = base_valid & np.isfinite(cube[:, :, j]).all(axis=1)
        if int(valid.sum()) < 200:
            rows.append(
                {"feature": col, "horizon": horizon, "note": "too few complete cross-sections"}
            )
            continue
        Xv = cube[valid, :, j]
        Yv = target.to_numpy(dtype=float)[valid]
        n_t = len(Xv)

        Xr = np.apply_along_axis(rankdata, 1, Xv)
        Yr = np.apply_along_axis(rankdata, 1, Yv)
        Xc = Xr - Xr.mean(1, keepdims=True)
        Yc = Yr - Yr.mean(1, keepdims=True)
        num = (Xc * Yc).sum(1)
        den = np.sqrt((Xc**2).sum(1) * (Yc**2).sum(1))
        ic = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        ic = ic[np.isfinite(ic)]
        if ic.size < 100:
            rows.append({"feature": col, "horizon": horizon, "note": "degenerate cross-sections"})
            continue

        mean_ic = float(ic.mean())
        se = float(ic.std(ddof=1) / np.sqrt(ic.size))
        # Timestamp-block permutation: reshuffle the target across coins, which
        # breaks the pairing while preserving each cross-section's spread.
        # A fresh symbol permutation per draw is distributionally valid for the
        # rank IC and far cheaper than an independent permutation per row; with
        # many draws the null is indistinguishable from the per-row version.
        xc2 = (Xc**2).sum(1)
        yr_mean = Yr.mean(1, keepdims=True)
        n_sym = Yr.shape[1]
        null = np.empty(draws)
        for i in range(draws):
            Ypc = Yr[:, rng.permutation(n_sym)] - yr_mean
            n2 = (Xc * Ypc).sum(1)
            d2 = np.sqrt(xc2 * (Ypc**2).sum(1))
            v = np.where(d2 > 0, n2 / np.where(d2 > 0, d2, 1.0), np.nan)
            v = v[np.isfinite(v)]
            null[i] = v.mean() if v.size else np.nan
        null = null[np.isfinite(null)]
        pval = float((np.abs(null) >= abs(mean_ic) - 1e-15).mean()) if null.size else float("nan")

        # Bucket spread on ranks: top-k vs bottom-k mean forward return in bps.
        spread = float("nan")
        spread_se = float("nan")
        spread_p = float("nan")
        if len(symbols) >= 2 * top_k:
            k = top_k
            order = np.argsort(Xv, axis=1)
            top_ix = order[:, -k:]
            bot_ix = order[:, :k]
            diff = (
                np.take_along_axis(Yv, top_ix, axis=1).mean(1)
                - np.take_along_axis(Yv, bot_ix, axis=1).mean(1)
            ) * 10_000.0
            diff = diff[np.isfinite(diff)]
            if diff.size > 100:
                spread = float(diff.mean())
                spread_se = float(diff.std(ddof=1) / np.sqrt(diff.size))
                n_null = np.empty(draws)
                for i in range(draws):
                    Yp = Yv[:, rng.permutation(n_sym)]
                    t2 = np.take_along_axis(Yp, top_ix, axis=1).mean(1)
                    b2 = np.take_along_axis(Yp, bot_ix, axis=1).mean(1)
                    n_null[i] = float(np.nanmean(t2 - b2))
                spread_p = float((np.abs(n_null) >= abs(spread) - 1e-12).mean())

        # Overlap-aware significance. Consecutive hourly rows share h-1 of their h
        # forward-return hours, so the naive spread SE treats ~h strongly
        # dependent observations as independent and is understated by roughly
        # sqrt(h). The non-overlapping subsample keeps every h-th timestamp and
        # is the honest test of whether a gross spread is tradable.
        no_mean = float("nan")
        no_se = float("nan")
        no_t = float("nan")
        if np.isfinite(spread) and n_t >= horizon * 20:
            sub = diff[::horizon]
            if sub.size >= 20:
                no_mean = float(sub.mean())
                no_se = float(sub.std(ddof=1) / np.sqrt(sub.size))
                no_t = no_mean / no_se if no_se > 0 else float("nan")

        rows.append(
            {
                "feature": col,
                "horizon": horizon,
                "n_timestamps": int(n_t),
                "xs_ic_mean": mean_ic,
                "xs_ic_se": se,
                "xs_ic_t": mean_ic / se if se > 0 else float("nan"),
                "xs_ic_positive_frac": float(np.mean(ic > 0)),
                "xs_perm_p": pval,
                "topk_botk_bps": spread,
                "topk_botk_se": spread_se,
                "topk_botk_perm_p": spread_p,
                "topk_botk_noOverlap_mean": no_mean,
                "topk_botk_noOverlap_se": no_se,
                "topk_botk_noOverlap_t": no_t,
                # A top-minus-bottom spread is a TWO-leg position: the long leg and the short
                # leg each pay their own round trip. The production convention in
                # horizon_discovery.signal_economics charges 36bps per leg, so the
                # spread costs 72bps, not 36. Subtracting a single 36bps from a
                # spread understates cost by half and overstated the earlier
                # candidate by 36bps. Per-leg framing is reported alongside.
                "topk_botk_net_bps": spread - 72.0 if np.isfinite(spread) else float("nan"),
                "topk_botk_per_leg_gross_bps": spread / 2.0 if np.isfinite(spread) else float("nan"),
                "topk_botk_per_leg_net_bps": spread / 2.0 - 36.0 if np.isfinite(spread) else float("nan"),
                "per_leg_hurdle_bps": 60.0,
            }
        )
    return rows


def time_series_stats(panel: Panel, horizon: int, draws: int, seed: int) -> list[dict[str, Any]]:
    """Within-symbol rank IC: can a coin's own features predict its own return?"""
    fwd = forward_returns(panel, horizon)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    for col in panel.feature_columns:
        per_symbol: list[float] = []
        for sym in panel.symbols:
            x = panel.frames[sym][col].to_numpy(dtype=float)
            y = fwd[sym]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 500:
                continue
            xx, yy = rankdata(x[m]), rankdata(y[m])
            if np.nanstd(xx) == 0 or np.nanstd(yy) == 0:
                continue
            per_symbol.append(float(np.corrcoef(xx, yy)[0, 1]))
        if len(per_symbol) < 5:
            rows.append({"feature": col, "horizon": horizon, "note": "too few usable symbols"})
            continue
        arr = np.array(per_symbol)
        mean_ic = float(arr.mean())
        se = float(arr.std(ddof=1) / np.sqrt(arr.size))
        # Sign-flip randomisation. Permuting the per-symbol values cannot test
        # whether their mean is zero, because the mean is permutation
        # invariant; flipping random signs is the correct null for a symmetric
        # distribution of per-coin correlations.
        n_null = np.empty(draws)
        for i in range(draws):
            n_null[i] = float((arr * rng.choice((-1.0, 1.0), size=arr.size)).mean())
        pval = float((np.abs(n_null) >= abs(mean_ic) - 1e-15).mean())
        rows.append(
            {
                "feature": col,
                "horizon": horizon,
                "n_symbols": int(arr.size),
                "ts_ic_mean": mean_ic,
                "ts_ic_se": se,
                "ts_ic_t": mean_ic / se if se > 0 else float("nan"),
                "ts_symbol_positive_frac": float(np.mean(arr > 0)),
                "ts_sign_test_p": pval,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# redundancy / effective information
# --------------------------------------------------------------------------- #


def zscore_panel(panel: Panel, pooled: pd.DataFrame) -> pd.DataFrame:
    """Standardise each feature with its own symbol's TRAIN-window mean/std."""
    out = pd.DataFrame(index=pooled.index)
    for sym in panel.symbols:
        m = pooled["symbol"] == sym
        st = panel.train_stats[sym]
        sub = pooled.loc[m, panel.feature_columns]
        out.loc[m, panel.feature_columns] = (
            (sub - st["mean"].to_numpy()) / st["std"].to_numpy()
        ).to_numpy()
    return out


def redundancy_structure(
    pooled: pd.DataFrame, cols: list[str], threshold: float = 0.80
) -> dict[str, Any]:
    """How many genuinely independent signals are in the feature set?"""
    X = pooled[cols].to_numpy(dtype=float)
    X = X[np.all(np.isfinite(X), axis=1)]
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    off = np.abs(corr[~np.eye(len(cols), dtype=bool)])
    evals = np.linalg.eigvalsh(corr)
    evals = np.clip(evals, 0.0, None)
    total = evals.sum()
    participation = float(total**2 / max((evals**2).sum(), 1e-12))
    p = evals / total if total > 0 else evals
    entropy_rank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    cum = np.cumsum(evals) / total if total > 0 else evals
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n95 = int(np.searchsorted(cum, 0.95) + 1)

    # Greedy clustering on |rho| >= threshold: how many groups of near-duplicates.
    parent = list(range(len(cols)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr[i, j]) >= threshold:
                pairs.append((cols[i], cols[j], float(corr[i, j])))
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups: dict[int, list[str]] = {}
    for i, name in enumerate(cols):
        groups.setdefault(find(i), []).append(name)

    per_family = {}
    for fam, members in FAMILIES.items():
        present = [m for m in members if m in cols]
        if not present:
            per_family[fam] = {"n_features": 0, "present": []}
            continue
        idx = [cols.index(m) for m in present]
        sub = np.abs(corr[np.ix_(idx, idx)])
        np.fill_diagonal(sub, 0.0)
        per_family[fam] = {
            "n_features": len(present),
            "present": present,
            "max_abs_within": float(sub.max()) if sub.size else float("nan"),
            "mean_abs_within": float(sub.mean()) if sub.size else float("nan"),
            "pairs_above_threshold": int((sub >= threshold).sum() // 2),
        }

    return {
        "n_features": len(cols),
        "abs_corr_offdiag_mean": float(off.mean()),
        "abs_corr_offdiag_p95": float(np.quantile(off, 0.95)),
        "abs_corr_offdiag_max": float(off.max()),
        "pairs_above_threshold": len(pairs),
        "n_clusters_at_threshold": len(groups),
        "participation_ratio": participation,
        "entropy_effective_rank": entropy_rank,
        "n_components_90pct_var": n90,
        "n_components_95pct_var": n95,
        "threshold": threshold,
        "highest_corr_pairs": sorted(pairs, key=lambda t: -abs(t[2]))[:25],
        "clusters": [sorted(v) for v in groups.values() if len(v) > 1],
        "per_family": per_family,
    }


def family_composites(
    panel: Panel, horizon: int, top_k: int, draws: int, seed: int
) -> list[dict[str, Any]]:
    """Equal-weight composite of each family, scored exactly like a feature.

    A cheap way to ask whether a family carries joint information without
    training anything: standardise each member on train stats, average, then run
    the same cross-sectional and time-series statistics used for single features.
    """
    pooled = pooled_frame(panel)
    Z = zscore_panel(panel, pooled)
    symbols = panel.symbols
    target = wide_matrix(panel, forward_returns(panel, horizon))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for fam, members in FAMILIES.items():
        present = [m for m in members if m in panel.feature_columns]
        if not present:
            continue
        comp = Z[present].mean(axis=1)
        Cm = wide_matrix(
            panel,
            {s: comp[pooled["symbol"] == s].to_numpy(dtype=float) for s in symbols},
            index=target.index,
        )
        valid = Cm.notna().all(axis=1) & target.notna().all(axis=1)
        if int(valid.sum()) < 200:
            continue
        Xv = Cm.loc[valid].to_numpy(dtype=float)
        Yv = target.loc[valid].to_numpy(dtype=float)
        Xr = np.apply_along_axis(rankdata, 1, Xv)
        Yr = np.apply_along_axis(rankdata, 1, Yv)
        Xc, Yc = Xr - Xr.mean(1, keepdims=True), Yr - Yr.mean(1, keepdims=True)
        ic = (Xc * Yc).sum(1) / np.sqrt((Xc**2).sum(1) * (Yc**2).sum(1))
        ic = ic[np.isfinite(ic)]

        order = np.argsort(Xv, axis=1)
        top = np.take_along_axis(Yv, order[:, -top_k:], axis=1).mean(1)
        bot = np.take_along_axis(Yv, order[:, :top_k], axis=1).mean(1)
        spread_bps = float(((top - bot) * 10_000.0).mean())

        null = np.empty(draws)
        n_sym_f = Yr.shape[1]
        yr_mean_f = Yr.mean(1, keepdims=True)
        xc2_f = (Xc**2).sum(1)
        for i in range(draws):
            Ypc = Yr[:, rng.permutation(n_sym_f)] - yr_mean_f
            v = (Xc * Ypc).sum(1) / np.sqrt(xc2_f * (Ypc**2).sum(1))
            v = v[np.isfinite(v)]
            null[i] = v.mean() if v.size else np.nan
        null = null[np.isfinite(null)]
        rows.append(
            {
                "family": fam,
                "horizon": horizon,
                "n_members": len(present),
                "xs_ic_mean": float(ic.mean()),
                "xs_ic_t": float(ic.mean() / (ic.std(ddof=1) / np.sqrt(ic.size)))
                if ic.size > 1
                else float("nan"),
                "xs_perm_p": float((np.abs(null) >= abs(ic.mean()) - 1e-15).mean())
                if null.size
                else float("nan"),
                "topk_botk_bps": spread_bps,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# leakage / staleness / temporal alignment
# --------------------------------------------------------------------------- #


def prefix_invariance_audit(
    store: ParquetStore, symbols: list[str], interval: str, prefix: int = 3000
) -> list[dict[str, Any]]:
    """Causality proof: features computed on a truncated series must not change.

    If any feature used future bars (full-sample centring, backward fill, a
    two-sided rolling window, a scaler fitted on everything), truncating the
    input would change the earlier feature values. Exact equality therefore
    rules out look-ahead by construction rather than by inspection.
    """
    out: list[dict[str, Any]] = []
    for sym in symbols:
        candles = store.read(
            "ohlcv",
            filters=[("interval", "==", interval)],
            symbols=[sym],
            columns=["ts", "open", "high", "low", "close", "volume", "symbol"],
        )
        if candles.empty:
            continue
        candles = candles.sort_values("ts").reset_index(drop=True)
        if interval == "1h":
            candles = candles[candles["ts"] >= SIM_START].reset_index(drop=True)
        if len(candles) < prefix + 50:
            continue
        full, cols = build_feature_frame(candles)
        trunc, _ = build_feature_frame(candles.iloc[:prefix].copy())
        for col in cols:
            a = full[col].to_numpy(dtype=float)[:prefix]
            b = trunc[col].to_numpy(dtype=float)[:prefix]
            m = np.isfinite(a) & np.isfinite(b)
            if not m.any():
                out.append(
                    {
                        "symbol": sym,
                        "feature": col,
                        "comparable": 0,
                        "max_abs_diff": float("nan"),
                        "leak": False,
                    }
                )
                continue
            denom = max(np.nanmax(np.abs(a[m])), 1e-9)
            diff = float(np.max(np.abs(a[m] - b[m])) / denom)
            out.append(
                {
                    "symbol": sym,
                    "feature": col,
                    "comparable": int(m.sum()),
                    "max_abs_diff": diff,
                    "leak": diff > 1e-9,
                }
            )
    return out


def staleness_audit(pooled: pd.DataFrame, cols: list[str]) -> list[dict[str, Any]]:
    """How long does each feature sit unchanged, and how much is fallback zero?"""
    rows: list[dict[str, Any]] = []
    for col in cols:
        runs: list[int] = []
        zeros = 0
        total = 0
        n_pairs = 0
        n_repeat = 0
        for _, grp in pooled.groupby("symbol", sort=False):
            x = grp[col].to_numpy(dtype=float)
            x = np.where(np.isfinite(x), x, np.nan)
            zeros += int(np.sum(np.isclose(np.nan_to_num(x, nan=0.0), 0.0)))
            total += len(x)
            eq = np.isclose(x[1:], x[:-1], rtol=1e-9, atol=1e-12)
            eq = eq & np.isfinite(x[1:]) & np.isfinite(x[:-1])
            n_pairs += int(eq.size)
            n_repeat += int(eq.sum())
            if eq.size:
                # count consecutive-equal run lengths
                idx = np.flatnonzero(~eq) + 1
                bounds = np.concatenate(([0], idx, [len(x)]))
                for a, b in zip(bounds[:-1], bounds[1:], strict=False):
                    if b - a > 1:
                        runs.append(int(b - a))
        rows.append(
            {
                "feature": col,
                "stale_frac": float(n_repeat / n_pairs) if n_pairs else float("nan"),
                "max_run": int(max(runs)) if runs else 1,
                "mean_run": float(np.mean(runs)) if runs else 1.0,
                "zero_frac": zeros / max(total, 1),
                "n_distinct_median": float(pooled.groupby("symbol")[col].nunique().median()),
            }
        )
    return rows


def future_lead_audit(panel: Panel, pooled: pd.DataFrame, leads: list[int]) -> list[dict[str, Any]]:
    """Correlation with far-future returns as a leak canary.

    A genuinely causal technical feature has no business predicting a 72h-ahead
    return better than a 1h-ahead one. Persistent or growing |IC| at long leads
    is the signature of a feature that already contains future information.
    """
    rows: list[dict[str, Any]] = []
    for lead in leads:
        y = target_for(panel, pooled, lead)
        for col in panel.feature_columns:
            x = pooled[col].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 1000:
                continue
            xa, ya = rankdata(x[m]), rankdata(y[m])
            if np.nanstd(xa) == 0 or np.nanstd(ya) == 0:
                continue
            r = float(np.corrcoef(xa, ya)[0, 1])
            rows.append({"feature": col, "lead_bars": lead, "spearman": r})
    return rows


# --------------------------------------------------------------------------- #
# transparent baselines, including the context features the model never sees
# --------------------------------------------------------------------------- #

BASELINES: dict[str, str] = {
    "raw_momentum_24h": "returns_24",
    "raw_momentum_6h": "returns_6",
    "short_vs_long_momentum": "momentum_5 minus momentum_60",
    "realized_vol_20h": "realized_vol_20",
    "volume_change": "log_volume_change",
    "relative_volume": "volume_zscore_20",
    "mean_reversion_20h": "negative zscore_close_20",
    "trend_strength": "ema_fast_slow_spread",
    "cross_sectional_momentum_20h": "within-timestamp rank of momentum_20",
}


def build_baselines(panel: Panel) -> pd.DataFrame:
    """One column per transparent baseline, on the same OOS rows as everything else."""
    parts = []
    for sym in panel.symbols:
        f = panel.frames[sym]
        d = pd.DataFrame(
            {
                "ts": f["ts"].to_numpy(),
                "symbol": sym,
                "raw_momentum_24h": f["returns_24"].to_numpy(dtype=float),
                "raw_momentum_6h": f["returns_6"].to_numpy(dtype=float),
                "short_vs_long_momentum": (
                    f["momentum_5"].to_numpy(dtype=float) - f["momentum_60"].to_numpy(dtype=float)
                ),
                "realized_vol_20h": f["realized_vol_20"].to_numpy(dtype=float),
                "volume_change": f["log_volume_change"].to_numpy(dtype=float),
                "relative_volume": f["volume_zscore_20"].to_numpy(dtype=float),
                "mean_reversion_20h": -f["zscore_close_20"].to_numpy(dtype=float),
                "trend_strength": f["ema_fast_slow_spread"].to_numpy(dtype=float),
                "cross_sectional_momentum_20h": f["momentum_20"].to_numpy(dtype=float),
            }
        )
        parts.append(d)
    base = pd.concat(parts, ignore_index=True)
    # Turn absolute momentum into a genuine cross-sectional rank.
    base["cross_sectional_momentum_20h"] = base.groupby("ts")["cross_sectional_momentum_20h"].rank(
        pct=True
    )
    return base


def context_probe(panel: Panel, horizon: int, top_k: int, draws: int, seed: int) -> dict[str, Any]:
    """Test market context the feature set does NOT contain, without adding it.

    BTC's own trailing return, cross-sectional breadth, and an equal-weight
    market return are the cheapest possible context variables. If they carry
    information the model cannot see, that is direct evidence for the missing
    market-regime domain rather than an argument from intuition.
    """
    symbols = panel.symbols
    fwd = forward_returns(panel, horizon)

    btc = panel.btc_symbol if panel.btc_symbol in symbols else symbols[0]
    btc_ts = pd.DatetimeIndex(panel.frames[btc]["ts"].to_numpy())
    btc_ret = panel.frames[btc]["returns_24"].to_numpy(dtype=float)

    target = wide_matrix(panel, fwd)
    mkt = target.mean(axis=1)
    breadth = (target > 0).mean(axis=1)
    btc_fwd = pd.Series(fwd[btc], index=btc_ts)

    def _corr(a: pd.Series, b: pd.Series) -> tuple[float, float, float]:
        m = a.notna() & b.notna()
        if m.sum() < 200:
            return float("nan"), float("nan"), float("nan")
        idx = a.index[m]
        xa, ya = rankdata(a[m].to_numpy()), rankdata(b[m].to_numpy())
        r = float(np.corrcoef(xa, ya)[0, 1])
        # Day-block p-value, matching the rest of the audit: a plain row shuffle
        # and an independence SE both understate uncertainty for autocorrelated
        # hourly crypto series.
        day = pd.DatetimeIndex(idx).floor("D")
        p = block_permutation_p(xa, ya, day.to_numpy(), draws, int(seed))
        # Standard large-sample SE for a Pearson correlation on ranks.
        n_obs = len(xa)
        se = float(np.sqrt(max((1.0 - r * r) / max(n_obs - 3, 1), 0.0)))
        return r, se, p

    r_mkt, se_mkt, p_mkt = _corr(pd.Series(btc_ret, index=btc_ts), mkt)
    r_breadth, se_b, p_b = _corr(breadth, mkt)
    r_btcalt, se_ba, p_ba = _corr(btc_fwd, mkt)

    # Can an alt's return be ordered by BTC's trailing state?
    btcs = pd.Series(btc_ret, index=btc_ts)
    cross: list[float] = []
    for s in symbols:
        if s == btc:
            continue
        b = btcs.reindex(target.index)
        m = b.notna().to_numpy() & target[s].notna().to_numpy()
        if m.sum() < 200:
            continue
        cross.append(
            float(np.corrcoef(rankdata(b[m].to_numpy()), rankdata(target[s][m].to_numpy()))[0, 1])
        )
    xs_info = {
        "mean_per_symbol_corr": float(np.mean(cross)) if cross else float("nan"),
        "se": float(np.std(cross, ddof=1) / np.sqrt(len(cross)))
        if len(cross) > 1
        else float("nan"),
        "n_symbols": len(cross),
        "frac_positive": float(np.mean(np.array(cross) > 0)) if cross else float("nan"),
    }
    return {
        "horizon": horizon,
        "btc_trailing_24h_vs_market_return": {
            "rho": r_mkt,
            "se": se_mkt,
            "perm_p": p_mkt,
            "note": "informative: trailing BTC state vs mean forward return",
        },
        "breadth_vs_market_return": {
            "rho": r_breadth,
            "se": se_b,
            "perm_p": p_b,
            "note": (
                "CIRCULAR BY CONSTRUCTION: breadth is the fraction of symbols with a "
                "positive forward return and the market return is the mean of those same "
                "forward returns, so high correlation only verifies panel coherence and "
                "carries no predictive information"
            ),
        },
        "btc_forward_vs_market_return": {
            "rho": r_btcalt,
            "se": se_ba,
            "perm_p": p_ba,
            "note": (
                "CIRCULAR BY CONSTRUCTION: BTC's forward return is one of the terms in the "
                "mean forward return, so this measures BTC's weight in the basket"
            ),
        },
        "btc_trailing_state_vs_each_alt": xs_info,
    }


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #


def collect_domain_presence(panel: Panel) -> dict[str, Any]:
    """Which information domains are actually represented in the feature matrix?"""
    present = {
        fam: [c for c in members if c in panel.feature_columns] for fam, members in FAMILIES.items()
    }
    absent = {}
    for dom in PROBED_ABSENT_DOMAINS:
        if dom in present:
            continue
        absent[dom] = {
            "n_features": 0,
            "verdict": "absent from the feature matrix the model receives",
        }
    return {"present": present, "absent": absent, "n_features_total": len(panel.feature_columns)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default=",".join(ab.PRODUCTION_UNIVERSE))
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--cutoff", type=int, default=5100)
    ap.add_argument("--max-horizon", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--boot-draws", type=int, default=1000)
    ap.add_argument("--perm-draws", type=int, default=1000)
    ap.add_argument("--mi-sample", type=int, default=20000)
    ap.add_argument(
        "--out-report", type=Path, default=ROOT / "data/artifacts/reports/feature_audit.json"
    )
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    store = ParquetStore()

    print("FEATURE / SIGNAL GAP AUDIT -- diagnosis only, no training", flush=True)
    print(f"  interval={args.interval} cutoff={args.cutoff} seed={args.seed}", flush=True)
    print(f"  horizons={HORIZONS} universe={len(symbols)} symbols", flush=True)

    panel = load_panel(store, symbols, args.interval, args.max_horizon, args.cutoff)
    symbols = panel.symbols
    print(
        f"  panel: {len(panel.feature_columns)} features, "
        f"{len(panel.timestamps)} OOS timestamps, {len(symbols)} symbols",
        flush=True,
    )

    pooled = pooled_frame(panel)

    report: dict[str, Any] = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "domains": collect_domain_presence(panel),
        "panel_alignment": alignment_report(panel),
        "invariants": {
            "duplicate_ts_symbol": int(pooled.duplicated(subset=["ts", "symbol"]).sum()),
            "n_rows": int(len(pooled)),
            "n_timestamps_union": int(len(panel.union_timestamps)),
            "n_timestamps_complete": int(len(panel.complete_timestamps)),
            "up_fraction_by_horizon": {},
        },
    }

    # forward-return sign balance, both directions retained
    for h in HORIZONS:
        fwd = forward_returns(panel, h)
        vals = np.concatenate([fwd[s] for s in symbols])
        vals = vals[np.isfinite(vals)]
        report["invariants"]["up_fraction_by_horizon"][str(h)] = {
            "up_fraction": float(np.mean(vals > 0)),
            "down_fraction": float(np.mean(vals < 0)),
            "exactly_zero": int(np.sum(vals == 0.0)),
            "n": int(vals.size),
        }

    print("  univariate association vs forward returns ...", flush=True)
    uni: list[dict[str, Any]] = []
    for h in HORIZONS:
        uni.extend(univariate_stats(panel, pooled, h, args.boot_draws, args.seed + h))
    p_raw = np.array([r.get("perm_p", np.nan) for r in uni])
    for r, q in zip(uni, bh_fdr(p_raw), strict=True):
        r["perm_p_fdr"] = float(q) if np.isfinite(q) else float("nan")
    report["univariate"] = uni

    print("  mutual information ...", flush=True)
    report["mutual_information"] = mutual_information_table(
        panel, pooled, HORIZONS, args.perm_draws, args.seed, args.mi_sample
    )

    print("  cross-sectional (within-timestamp) ...", flush=True)
    xs: list[dict[str, Any]] = []
    for h in HORIZONS:
        xs.extend(cross_sectional_stats(panel, h, args.top_k, args.perm_draws, args.seed + h))
    report["cross_sectional"] = xs

    print("  time-series (within-symbol) ...", flush=True)
    ts: list[dict[str, Any]] = []
    for h in HORIZONS:
        ts.extend(time_series_stats(panel, h, args.perm_draws, args.seed + h))
    report["time_series"] = ts

    print("  redundancy / effective information ...", flush=True)
    report["redundancy"] = redundancy_structure(pooled, panel.feature_columns)

    print("  family composites ...", flush=True)
    fam: list[dict[str, Any]] = []
    for h in HORIZONS:
        fam.extend(family_composites(panel, h, args.top_k, args.perm_draws, args.seed + h))
    report["families"] = fam

    print("  leakage / staleness ...", flush=True)
    prefix = prefix_invariance_audit(store, symbols[:4], args.interval)
    leaks = [p for p in prefix if p.get("leak")]
    report["leakage"] = {
        "prefix_invariance": {
            "tested_symbols": sorted({p["symbol"] for p in prefix}),
            "n_features_tested": len({p["feature"] for p in prefix}),
            "n_violations": len(leaks),
            "violations": leaks,
            "verdict": "all features are invariant to truncation of future bars"
            if not leaks
            else "LEAK DETECTED",
        },
        "future_lead_correlation": future_lead_audit(panel, pooled, [1, 24, 72]),
        "staleness": staleness_audit(pooled, panel.feature_columns),
        "same_bar_features": sorted(SAME_BAR_FEATURES),
        "ts_convention": (
            "ohlcv ts is the Binance kline OPEN time (row[0]); the OHLC of bar t "
            "is only final at t+1. Features and targets are both indexed at bar t, "
            "so they are internally consistent provided execution occurs no earlier "
            "than the close of bar t."
        ),
    }

    print("  baselines ...", flush=True)
    base = build_baselines(panel)
    base_pooled = base.copy()
    base_pooled["day"] = pd.to_datetime(base_pooled["ts"], utc=True).dt.floor("D")
    base_rows: list[dict[str, Any]] = []
    for name in BASELINES:
        for h in HORIZONS:
            y = target_for(panel, pooled, h)
            x = base_pooled[name].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 1000:
                continue
            xr, yr = rankdata(x[m]), rankdata(y[m])
            rho = float(np.corrcoef(xr, yr)[0, 1])
            auc = auc_from_ranks(x[m], (y[m] > 0).astype(float))
            base_rows.append(
                {
                    "baseline": name,
                    "definition": BASELINES[name],
                    "horizon": h,
                    "n": int(m.sum()),
                    "spearman": rho,
                    "spearman_se": float(1.0 / np.sqrt(m.sum())),
                    "univariate_auc": auc,
                }
            )
    base_p = np.array([r["spearman"] for r in base_rows])
    se = np.array([r["spearman_se"] for r in base_rows])
    z = np.abs(base_p / se)
    from scipy.stats import norm as _norm

    pvals = 2 * (1 - _norm.cdf(z))
    for r, q, pv in zip(base_rows, bh_fdr(pvals), pvals, strict=True):
        r["perm_p_fdr"] = float(q)
        r["p_approx"] = float(pv)
    report["baselines"] = base_rows

    print("  market-context probe (NOT currently in the feature set) ...", flush=True)
    report["context_probe"] = [
        context_probe(panel, h, args.top_k, args.perm_draws, args.seed + h) for h in HORIZONS
    ]

    report["data_availability"] = dataset_coverage(store)

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.out_report}", flush=True)
    return 0


def dataset_coverage(store: ParquetStore) -> dict[str, Any]:
    """Row counts and date coverage per stored dataset, for feasibility."""
    out: dict[str, Any] = {}
    for ds in ("ohlcv", "news", "onchain", "fear_greed"):
        try:
            f = store.read(ds)
        except Exception as exc:  # noqa: BLE001
            out[ds] = {"error": str(exc)}
            continue
        entry: dict[str, Any] = {"rows": int(len(f)), "columns": list(f.columns)}
        if "ts" in f.columns and len(f):
            entry["ts_min"] = str(f["ts"].min())
            entry["ts_max"] = str(f["ts"].max())
        if ds == "ohlcv":
            entry["interval_counts"] = (
                f["interval"].value_counts().to_dict() if "interval" in f.columns else {}
            )
            entry["n_symbols"] = int(f["symbol"].nunique())
            if "interval" in f.columns:
                one = f[(f["interval"] == "1h") & (f["symbol"] == "BTC_USDT")]
                entry["btc_1h_bars"] = int(len(one))
        if ds == "onchain" and "metric" in f.columns:
            entry["metric_counts"] = f["metric"].value_counts().to_dict()
        out[ds] = entry
    out["_absent_datasets"] = [
        "funding_rate",
        "open_interest",
        "futures_basis",
        "liquidations",
        "order_book",
        "trades_tape",
        "borrow_lending",
    ]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
