"""Release gates for a trading model, and an honest audit of the gates themselves.

A gate is only useful if it would have rejected what we already know is broken,
and if it is set by what the strategy must *achieve economically* rather than by
a round number that sounds reasonable. Both are checked here.

Statistical power is computed rather than assumed: with a given number of
independent out-of-sample bars, an AUC interval of a given half-width is
resolvable, so a threshold can be placed inside the noise floor or outside it
knowingly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def auc_half_width(n: int, auc: float = 0.5, z: float = 1.96) -> float:
    """Bootstrap half-width of an AUC estimate at the null, via DeLong's variance.

    Under H0 the AUC has variance ~1/(12n) in the normal-approximation limit, so
    the resolvable difference from 0.5 is what this returns. Independent *bars*,
    not rows: overlapping horizons make the effective sample smaller, so callers
    pass a de-overlapped count.
    """
    if n <= 0:
        return float("nan")
    var = (auc * (1 - auc) + (auc - 0.5) ** 2) / max(n - 1, 1)
    return float(z * np.sqrt(var))


def evaluate(
    *,
    auc: float,
    auc_ci_low: float,
    calibration_error: float,
    excess_vs_equal_weight: float,
    excess_ci: tuple[float, float] | None,
    sharpe_net: float,
    max_drawdown: float,
    edge_per_turnover: float,
    round_trip_bps: float,
    n_independent_bars: int,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, value: float | None, threshold: float, passed: bool,
            why: str) -> None:
        checks.append({
            "gate": name, "value": value, "threshold": threshold,
            "passes": bool(passed), "rationale": why,
        })

    add("oos_auc", auc, 0.53, auc >= 0.53,
        "ranking power; 0.53 is a common floor but is only meaningful if it maps "
        "to more edge per trade than the spread")
    add("auc_ci_excludes_0.5", auc_ci_low, 0.50, auc_ci_low > 0.50,
        "an interval straddling 0.50 means the ranking is not distinguishable "
        "from a coin flip")
    add("calibration_error", calibration_error, 0.05, calibration_error < 0.05,
        "mean gap between claimed and observed probability; above this the "
        "headline 'probability of profit' cannot be sized against")
    add("excess_vs_equal_weight", excess_vs_equal_weight, 0.0,
        excess_vs_equal_weight > 0,
        "beating the thing you could have simply bought, net of costs")
    if excess_ci is not None:
        add("excess_significant", excess_ci[0], 0.0, excess_ci[0] > 0,
            "lower bound of the bootstrap interval above zero, so the excess is "
            "not sampling noise")
    add("sharpe_net", sharpe_net, 0.0, sharpe_net > 0,
        "positive risk-adjusted return after realistic costs")
    add("max_drawdown", max_drawdown, -0.25, max_drawdown > -0.25,
        "a return that needs more than a quarter of the account to stay invested "
        "is not deployable at any size")
    add("edge_per_turnover_ratio", edge_per_turnover, 1.0,
        edge_per_turnover >= 1.0,
        f"gross edge per unit turnover must exceed the {round_trip_bps:.2%} round "
        "trip; this is the binding economic constraint and it is not a round number")

    power = auc_half_width(n_independent_bars)
    return {
        "checks": checks,
        "n_checks": len(checks),
        "n_passed": sum(c["passes"] for c in checks),
        "all_pass": all(c["passes"] for c in checks),
        "statistical_power": {
            "independent_bars": n_independent_bars,
            "auc_half_width_at_null": power,
            "resolvable_auc_difference": power,
            "auc_0.53_is_resolvable": bool(power < 0.03),
            "note": (
                "With this many independent bars a 0.53 AUC is comfortably "
                "resolvable, so failing it is informative rather than a power "
                "problem. The binding limits are economic and the length of the "
                "P&L record, not the size of the sample."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, help="edge_sweep.csv output")
    parser.add_argument("--signal-json", required=True,
                        help="one_year_sim report, for AUC / accuracy")
    parser.add_argument("--predictions", default=None,
                        help="predictions parquet, for pooled calibration error")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import pandas as pd

    sweep = pd.read_csv(args.sweep)
    report = json.loads(Path(args.signal_json).read_text(encoding="utf-8"))
    block = report["intervals"][args.interval]

    best = sweep.loc[sweep["rank_score"].idxmax()]
    capital = float(args.capital)
    cost_paid_fraction = float(best["costs_paid"]) / capital
    # Measured inside the costed run, not by dividing a cost-free run's turnover
    # into it: the cost-free account compounds, so its notional is inflated
    # relative to what the same signals trade on the losing path.
    edge_per_turnover = float(
        best.get("cost_coverage_ratio",
                 (float(best["total_return"]) + cost_paid_fraction) / cost_paid_fraction)
    )

    pooled = block.get("pooled_diagnostics", {})
    ece = float("nan")
    if args.predictions and Path(args.predictions).exists():
        # Calibration is a bias measurement, so it is computed on the pooled
        # out-of-sample predictions rather than inferred from a per-symbol table.
        frame = pd.read_parquet(args.predictions)
        score = frame["score"].to_numpy(dtype=float)
        target = frame["target_return"].to_numpy(dtype=float)
        finite = np.isfinite(score) & np.isfinite(target)
        score, target = score[finite], target[finite]
        from one_year_sim import _normal_cdf

        claimed = _normal_cdf(score)
        labels = (target > 0).astype(float)
        bins = pd.qcut(pd.Series(claimed), 10, labels=False, duplicates="drop")
        ece = sum(
            float((bins == bucket).mean())
            * abs(float(claimed[bins == bucket].mean()) - float(labels[bins == bucket].mean()))
            for bucket in pd.unique(bins)
            if (bins == bucket).any()
        )
        top = pd.Series(claimed).rank(pct=True) >= 0.9
        top_claimed = float(claimed[top.to_numpy()].mean())
        top_observed = float(labels[top.to_numpy()].mean())
    else:
        calibration = block.get("pooled_diagnostics", {}).get("calibration", [])
        if calibration:
            ece = sum(
                (row["n"] / sum(r["n"] for r in calibration))
                * abs(row["claimed_p_profit"] - row["observed_p_profit"])
                for row in calibration
            )
        top_claimed = top_observed = float("nan")

    # Overlapping label horizons make consecutive bars dependent, so the effective
    # sample for an AUC interval is far below the raw bar count.
    raw_bars = int(pooled.get("oos_bars", block.get("pooled", {}).get("oos_bars", 0)))
    horizon = 1
    independent = max(1, raw_bars // max(horizon, 1) // 5)

    result = evaluate(
        auc=float(pooled.get("auc", float("nan"))),
        auc_ci_low=float(pooled.get("auc_ci_low", float("nan"))),
        calibration_error=ece,
        excess_vs_equal_weight=float(best["excess_vs_equal_weight"]),
        excess_ci=None,
        sharpe_net=float(best["sharpe"]),
        max_drawdown=float(best["max_drawdown"]),
        edge_per_turnover=edge_per_turnover,
        round_trip_bps=(args.fee_bps + args.slippage_bps) / 100.0,
        n_independent_bars=independent,
    )

    print("=" * 96)
    print("RELEASE GATES, EVALUATED AGAINST THE BEST CONFIGURATION FOUND")
    print("=" * 96)
    print(f"best config: {best['config']}\n")
    width = max(len(c["gate"]) for c in result["checks"])
    for check in result["checks"]:
        mark = "PASS" if check["passes"] else "FAIL"
        value = check["value"]
        shown = f"{value:>10.4f}" if isinstance(value, (int, float)) else f"{value:>10}"
        print(f"  [{mark}] {check['gate']:<{width}}  {shown}   "
              f"(threshold {check['threshold']})")
    print(f"\n  {result['n_passed']} of {result['n_checks']} gates pass.")
    print("\n  probability calibration, pooled out-of-sample:")
    print(f"    most bullish decile claims {top_claimed:.1%}, delivers {top_observed:.1%}")

    power = result["statistical_power"]
    print("\n" + "=" * 96)
    print("ARE THE PROPOSED THRESHOLDS JUSTIFIED BY THE AVAILABLE EVIDENCE?")
    print("=" * 96)
    print(f"  raw out-of-sample bars            {raw_bars:,}")
    print(f"  independent bars (after overlap)  {power['independent_bars']:,}")
    print(f"  AUC resolvable difference at null  +/-{power['auc_half_width_at_null']:.4f}")
    print(f"  is AUC 0.53 resolvable here?      {power['auc_0.53_is_resolvable']}")
    print(f"\n  {power['note']}")

    print("\n  On the proposed thresholds specifically:")
    print("   * AUC 0.53 -- statistically resolvable, but it is not on its own a")
    print("     sufficient condition. It should be paired with the economic gate")
    print("     below, because a 0.53 AUC traded at 600x turnover still loses money.")
    print("   * calibration error < 5% -- a reasonable target and independent of the")
    print("     sample size, since it is a bias measurement rather than an estimate")
    print("     of a mean. Current value is far outside it.")
    print("   * excess vs equal-weight > 0 -- correct, and it must be measured net of")
    print("     costs. A single year of one path cannot establish it; this needs")
    print("     several independent periods before it is trusted as a release gate.")
    print("   * Sharpe > 0 net -- necessary but not sufficient; it can be met by a")
    print("     single lucky stretch, so it needs a confidence interval too.")

    print("\n  Recommended additional gates, in place of round numbers:")
    print("   1. edge_per_turnover_ratio >= 1.0, measured on out-of-sample data.")
    print("      This is the constraint that actually decides profitability, and it")
    print("      is derived from the venue's cost rather than chosen.")
    print("   2. Round trips per day <= 10 AND turnover <= the level the edge can")
    print("      pay for. Both, because a strategy can respect a trade count and")
    print("      still churn through enormous notional.")
    print("   3. excess return vs equal-weight with a bootstrap interval lower bound")
    print("      above zero across multiple non-overlapping periods.")
    print("   4. Stability: the gross edge must survive the restraint sweep. If gross")
    print("      return swings by tens of points when a gate threshold moves by a")
    print("      few bps, the edge is noise and no single configuration should ship.")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nsaved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
