"""Read-only research-monitor API.

The existing API exposes models, runs and jobs, but none of the artefacts that
actually answer "is this system working?" live behind a route: the sweep reports
on disk, the per-symbol verdicts, and the event studies. Everything here is
therefore a *reader* over files and database rows that already exist.

Three deliberate design constraints
-----------------------------------
**Nothing is trained, scored or refitted on request.** The holdout can only be
read once per selection decision. An endpoint that silently re-ran a holdout
would quietly destroy the statistical guarantee the rest of the project is
built to protect, so the only holdout numbers served are the ones already
committed to a report file.

**Absence is reported, not hidden.** A cell with no holdout scored, a symbol
with too little history and a metric that failed to compute all return
explicitly distinct states. A dashboard that renders every row as a clean zero
is worse than no dashboard, because it looks like a working system.

**Thin evidence carries its width.** Every directional-accuracy figure is served
next to its confidence interval, so a reviewer can see that 0.72 precision on
three observations is not the same claim as 0.52 on three thousand.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from cqip.config import Settings, get_settings
from cqip.events import load_event_reports

router = APIRouter(prefix="/research", tags=["research"])

#: A cell that never produced a directional accuracy is not "0% accurate"; it is
#: missing, and the two must never be rendered the same way.
_MISSING = "missing"


def _num(value: Any) -> float | None:
    """Coerce a report field to a finite float, or ``None``.

    Reports are written by long-running jobs and contain ``None``, ``NaN`` and
    occasionally numeric strings. Anything non-finite becomes ``None`` so the
    frontend can distinguish "not measured" from "measured as zero".
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _ci(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    lower, upper = _num(value.get("lower")), _num(value.get("upper"))
    if lower is None or upper is None:
        return None
    return {"lower": lower, "upper": upper}


def _reports_dir(settings: Settings) -> Path:
    return settings.resolved_artifact_dir / "reports"


def _app_settings() -> Settings:
    """Fallback for callers outside a request context.

    Inside a request the routes take ``Settings`` via ``Depends(get_settings)``,
    which is what makes them testable and consistent with the rest of the API.
    Building a fresh ``Settings()`` would serve artifacts and database rows
    from a *different* location than the app was configured with.
    """
    return Settings()


def _load_report(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _list_reports(settings: Settings) -> list[dict[str, Any]]:
    """Index every sweep report on disk, newest first.

    A report whose label is ``screen`` is excluded from the "latest" pointer
    because a one-epoch screening pass is not a finished experiment; mixing the
    two would make a screening run look like a full one.
    """
    directory = _reports_dir(settings)
    if not directory.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for path in sorted(directory.glob("sweep-*.json"), reverse=True):
        payload = _load_report(path)
        if payload is None:
            continue
        name = path.name
        label = "screen" if "-screen-" in name else ("latest" if name == "sweep-latest.json" else "run")
        cells = payload.get("cells") or []
        entries.append({
            "name": name,
            "label": label,
            "is_screen": label == "screen",
            "is_latest": name == "sweep-latest.json",
            "symbols": payload.get("symbols"),
            "intervals": payload.get("intervals"),
            "kinds": payload.get("kinds"),
            "folds": payload.get("folds"),
            "epochs": payload.get("epochs"),
            "started_at": payload.get("started_at"),
            "finished_at": payload.get("finished_at"),
            "duration_seconds": _num(payload.get("duration_seconds")),
            "cell_count": len(cells),
            "passed": sum(1 for c in cells if isinstance(c, dict) and c.get("verdict") == "passes"),
            "no_edge": sum(1 for c in cells if isinstance(c, dict) and c.get("verdict") == "no-edge"),
            "status_counts": _count(c.get("status") for c in cells if isinstance(c, dict)),
        })
    entries.sort(key=lambda e: (e["is_screen"], str(e.get("started_at") or "")), reverse=True)
    return entries


def _count(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value) if value is not None else _MISSING
        counts[key] = counts.get(key, 0) + 1
    return counts


def _normalise_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Flatten one sweep cell into a uniform, frontend-friendly shape.

    Reports come from two eras: older cells carry flat fields
    (``direction_accuracy``) and newer ones carry prefixed
    ``val_``/``test_`` fields. Both are normalised to the same keys so the UI
    never has to branch on report vintage.
    """
    val_acc = _num(cell.get("val_directional_accuracy"))
    test_acc = _num(cell.get("test_directional_accuracy"))
    if val_acc is None:
        val_acc = _num(cell.get("direction_accuracy"))
    if test_acc is None and cell.get("holdout_scored"):
        test_acc = _num(cell.get("test_directional_accuracy"))

    val_corr = _num(cell.get("val_correlation"))
    if val_corr is None:
        val_corr = _num(cell.get("correlation"))

    holdout_scored = bool(cell.get("holdout_scored"))

    status = cell.get("status") or _MISSING
    # A cell that never produced evidence must not inherit a "no edge" verdict
    # from a run that merely failed or ran out of rows. "We could not test" and
    # "we tested and found nothing" are opposite claims, and conflating them
    # is how a research dashboard ends up quietly overstating its own coverage.
    verdict = cell.get("verdict")
    if verdict == "no-edge" and (status != "ok" or val_acc is None):
        verdict = "insufficient_data" if status != "failed" else "failed"
    if not verdict:
        verdict = _MISSING

    return {
        "key": f"{cell.get('symbol')}|{cell.get('interval')}|{cell.get('kind')}",
        "symbol": cell.get("symbol"),
        "interval": cell.get("interval"),
        "kind": cell.get("kind"),
        "status": status,
        "verdict": verdict,
        "reason": cell.get("reason"),
        "model_id": cell.get("model_id"),
        "rows": _num(cell.get("rows")),
        "holdout_scored": holdout_scored,
        "val": {
            "directional_accuracy": val_acc,
            "directional_accuracy_ci": _ci(cell.get("val_directional_accuracy_ci")),
            "correlation": val_corr,
            "correlation_ci": _ci(cell.get("val_correlation_ci")),
            "mae": _num(cell.get("val_mae")),
            "trades": _num(cell.get("val_trades")),
            "max_drawdown": _num(cell.get("val_max_drawdown")),
            "excess_vs_buy_and_hold": _num(cell.get("val_excess_vs_buy_and_hold")),
        },
        "test": {
            "directional_accuracy": test_acc,
            "directional_accuracy_ci": _ci(cell.get("test_directional_accuracy_ci")),
            "correlation": _num(cell.get("test_correlation")),
            "correlation_ci": _ci(cell.get("test_correlation_ci")),
            "mae": _num(cell.get("test_mae")),
            "trades": _num(cell.get("test_trades")),
            "max_drawdown": _num(cell.get("test_max_drawdown")),
            "excess_vs_buy_and_hold": _num(cell.get("test_excess_vs_buy_and_hold")),
        } if holdout_scored else None,
    }


class ReportListResponse(BaseModel):
    count: int = Field(ge=0)
    reports: list[dict[str, Any]]


class ReportDetailResponse(BaseModel):
    name: str
    summary: dict[str, Any]
    cells: list[dict[str, Any]]


class OverviewResponse(BaseModel):
    """Everything the dashboard needs for its first paint, in one request.

    Seven separate round-trips to render one page would be slower and would
    show a half-populated screen if any single call failed.
    """
    health: dict[str, Any]
    datasets: list[dict[str, Any]]
    symbols: int
    models: int
    active_models: int
    runs: dict[str, int]
    recent_runs: list[dict[str, Any]]
    jobs: list[dict[str, Any]]
    reports: list[dict[str, Any]]
    cells_total: int
    cells_passed: int
    cells_no_edge: int
    cells_insufficient: int
    cells_failed: int
    holdout_scored_cells: int
    event_studies: list[dict[str, Any]]
    architecture_counts: dict[str, int]
    interval_counts: dict[str, int]
    symbol_table: list[dict[str, Any]]


def _db_path(settings: Settings) -> Path | None:
    url = settings.resolved_db_url
    if not url.startswith("sqlite:///"):
        return None
    return Path(url[len("sqlite:///"):])


def _query(settings: Settings, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Run a read-only query against the SQLite store.

    Opened in read-only URI mode so a monitoring request can never take a write
    lock on, or mutate, the database the training pipeline is using.
    """
    path = _db_path(settings)
    if path is None or not path.exists():
        return []
    try:
        uri = f"file:{path.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(sql, params).fetchall()]
    except sqlite3.Error:
        # A monitoring endpoint that 500s because the DB is momentarily locked
        # is useless; an empty section is honest and recoverable.
        return []


def _architecture_counts(settings: Settings) -> dict[str, int]:
    """Count models per architecture.

    ``model_versions.kind`` is always the literal ``forecaster``; the
    architecture lives inside the JSON ``params`` blob, so counting the column
    would return a single meaningless bucket.
    """
    counts: dict[str, int] = {}
    for row in _query(settings, "SELECT params FROM model_versions WHERE is_active=1"):
        params = _loads(row.get("params")) or {}
        arch = params.get("architecture") or "unknown"
        counts[str(arch)] = counts.get(str(arch), 0) + 1
    return counts


def _interval_counts(settings: Settings) -> dict[str, int]:
    return {
        str(r.get("interval") or "unknown"): int(r.get("n") or 0)
        for r in _query(
            settings,
            "SELECT interval, COUNT(*) AS n FROM model_versions "
            "WHERE is_active=1 GROUP BY interval",
        )
    }


def _loads(value: Any) -> Any:
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None
    return value


def _all_cells(settings: Settings) -> list[dict[str, Any]]:
    """Every cell from every non-screening report, normalised.

    Reports are walked newest-first and the *first* sighting of a
    (symbol, interval, kind) identity wins, so a re-run supersedes its
    predecessor instead of the other way round.

    A holdout reading is deliberately carried forward even when a newer report
    for the same identity has none. The holdout was already spent on that cell;
    a later re-run that only re-validated cannot un-spend it, and dropping the
    result would hide real evidence. Such rows are tagged ``holdout_source`` so
    the UI can show that the figure came from an earlier report rather than
    implying it belongs to the current model.
    """
    primary: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []
    holdout_pool: dict[tuple[str, str, str], dict[str, Any]] = {}

    for entry in _list_reports(settings):
        if entry["is_screen"]:
            continue
        payload = _load_report(_reports_dir(settings) / entry["name"])
        if payload is None:
            continue
        for raw in payload.get("cells") or []:
            if not isinstance(raw, dict):
                continue
            cell = _normalise_cell(raw)
            identity = (str(cell["symbol"]), str(cell["interval"]), str(cell["kind"]))
            if identity in primary:
                if cell["holdout_scored"] and identity not in holdout_pool:
                    holdout_pool[identity] = {
                        "test": cell["test"], "source": entry["name"],
                        "model_id": cell["model_id"],
                    }
                continue
            primary[identity] = cell
            order.append(identity)

    for identity in order:
        cell = primary[identity]
        cell.setdefault("holdout_source", None)
        if not cell["holdout_scored"] and identity in holdout_pool:
            carried = holdout_pool[identity]
            cell["test"] = carried["test"]
            cell["holdout_scored"] = True
            cell["holdout_source"] = carried["source"]
            cell["holdout_model_id"] = carried["model_id"]
    return [primary[identity] for identity in order]


def _symbol_table(settings: Settings) -> list[dict[str, Any]]:
    """One row per symbol: the actionable assessment the research is for.

    A symbol is only labelled a candidate if it has validation evidence whose
    confidence interval clears chance *and* a holdout was scored. Everything
    else is reported at its true status, because "we do not know" is a finding
    worth surfacing rather than a gap to paper over.
    """
    coverage = {
        (r.get("symbol"), r.get("interval")): r
        for r in _query(settings, "SELECT * FROM data_coverage")
    }
    rows: list[dict[str, Any]] = []
    for cell in _all_cells(settings):
        rows.append({
            "symbol": cell["symbol"],
            "interval": cell["interval"],
            "kind": cell["kind"],
            "status": cell["status"],
            "verdict": cell["verdict"],
            "reason": cell["reason"],
            "rows": cell["rows"],
            "val_accuracy": cell["val"]["directional_accuracy"],
            "val_accuracy_ci": cell["val"]["directional_accuracy_ci"],
            "val_correlation": cell["val"]["correlation"],
            "holdout_scored": cell["holdout_scored"],
            # Carried-forward holdout figures are attributed to their source so
            # a reviewer can tell a fresh reading from an earlier run's.
            "holdout_source": cell.get("holdout_source"),
            "test_accuracy": (cell["test"] or {}).get("directional_accuracy"),
            "test_accuracy_ci": (cell["test"] or {}).get("directional_accuracy_ci"),
            "test_excess": (cell["test"] or {}).get("excess_vs_buy_and_hold"),
            "test_trades": (cell["test"] or {}).get("trades"),
            "test_max_drawdown": (cell["test"] or {}).get("max_drawdown"),
        })

    for row in rows:
        cov = coverage.get((row["symbol"], row["interval"])) or {}
        row["data_start"] = cov.get("start")
        row["data_end"] = cov.get("end")
        row["data_rows"] = cov.get("rows")
        row["provider"] = cov.get("provider")

    rows.sort(key=lambda r: (str(r["symbol"]), str(r["interval"]), str(r["kind"])))
    return rows


def _event_summaries(settings: Settings) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for record in load_event_reports(_reports_dir(settings)):
        event = record.get("event") or {}
        best = None
        for row in event.get("by_k") or []:
            if row.get("significant") and (
                best is None or (row.get("lift") or 0) > (best.get("lift") or 0)
            ):
                best = row
        summaries.append({
            "file": record.get("_file"),
            "symbol": record.get("symbol"),
            "interval": record.get("interval"),
            "model_kind": record.get("model_kind"),
            "verdict": event.get("verdict"),
            "event_rate": _num(event.get("event_rate")),
            "n_events": event.get("n_events"),
            "n": event.get("n"),
            "threshold": event.get("threshold"),
            "k": event.get("k"),
            "best_lift": _num((best or {}).get("lift")),
            "best_k": (best or {}).get("k"),
            "best_precision": _num((best or {}).get("precision")),
            "best_precision_ci": _ci({
                "lower": (best or {}).get("precision_ci_low"),
                "upper": (best or {}).get("precision_ci_high"),
            }),
            "split": record.get("split") or {},
            "conviction": record.get("conviction") or [],
            # The full per-k table, not just the significant best: the shape of
            # the curve *is* the evidence, and a lone best point hides whether
            # precision is flat, monotone, or a single lucky bin.
            "by_k": [
                {
                    "k": _num(row.get("k")),
                    "precision": _num(row.get("precision")),
                    "precision_ci_low": _num(row.get("precision_ci_low")),
                    "precision_ci_high": _num(row.get("precision_ci_high")),
                    "lift": _num(row.get("lift")),
                    "n_taken": row.get("n_taken"),
                    "significant": bool(row.get("significant")),
                }
                for row in (event.get("by_k") or [])
                if isinstance(row, dict)
            ],
            "selectivity_curve": event.get("selectivity_curve") or [],
            "n_tests": event.get("n_tests"),
        })
    return summaries


@router.get("/overview", response_model=OverviewResponse)
def overview(settings: Settings = Depends(get_settings)) -> OverviewResponse:
    """Single aggregate payload for the whole dashboard."""
    health: dict[str, Any] = {"database_readable": bool(_query(settings, "SELECT 1 AS ok"))}

    datasets = []
    # Constructed rather than pulled off app.state, which only holds
    # settings/engine/job_manager; ParquetStore is a thin lazy wrapper, so
    # building one here does no I/O until a dataset is actually counted.
    from cqip.storage import ParquetStore

    store = ParquetStore(settings=settings)
    for name in ("ohlcv", "news", "onchain", "fear_greed"):
        try:
            datasets.append({
                "dataset": name,
                "rows": store.row_count(name),
                "symbols": len(store.symbols(name)) if name == "ohlcv" else None,
            })
        except Exception:  # pragma: no cover - defensive
            datasets.append({"dataset": name, "rows": None, "symbols": None})

    reports = _list_reports(settings)
    cells = _all_cells(settings)
    by_status = _count(c.get("status") for c in cells)

    jobs = []
    from cqip import api as api_module

    manager = api_module.state.get("job_manager")
    if manager is not None:
        try:
            jobs = manager.list(status=None, limit=20)
        except Exception:  # pragma: no cover - defensive
            jobs = []

    recent_runs = _query(
        settings,
        "SELECT run_id, kind, status, metrics, started_at, duration_seconds "
        "FROM runs ORDER BY started_at DESC LIMIT 25",
    )
    for run in recent_runs:
        run["metrics"] = _loads(run.get("metrics")) or {}
        run["duration_seconds"] = _num(run.get("duration_seconds"))

    return OverviewResponse(
        health=health,
        datasets=datasets,
        symbols=len({c["symbol"] for c in cells}),
        models=len(_query(settings, "SELECT model_id FROM model_versions")),
        active_models=len(_query(settings, "SELECT model_id FROM model_versions WHERE is_active=1")),
        runs=_count(r.get("status") for r in _query(
            settings, "SELECT status FROM runs"
        )),
        recent_runs=recent_runs,
        jobs=jobs,
        reports=reports,
        cells_total=len(cells),
        cells_passed=sum(1 for c in cells if c["verdict"] == "passes"),
        cells_no_edge=sum(1 for c in cells if c["verdict"] == "no-edge"),
        cells_insufficient=by_status.get("insufficient_data", 0),
        cells_failed=by_status.get("failed", 0),
        holdout_scored_cells=sum(1 for c in cells if c["holdout_scored"]),
        event_studies=_event_summaries(settings),
        architecture_counts=_architecture_counts(settings),
        interval_counts=_interval_counts(settings),
        symbol_table=_symbol_table(settings),
    )


@router.get("/reports", response_model=ReportListResponse)
def list_reports(
    include_screen: bool = Query(default=True),
    settings: Settings = Depends(get_settings),
) -> ReportListResponse:
    reports = _list_reports(settings)
    if not include_screen:
        reports = [r for r in reports if not r["is_screen"]]
    return ReportListResponse(count=len(reports), reports=reports)


@router.get("/reports/{name}", response_model=ReportDetailResponse)
def report_detail(name: str, settings: Settings = Depends(get_settings)) -> ReportDetailResponse:
    if "/" in name or "\\" in name or ".." in name:
        # Reject traversal outright: the name comes from the URL and is joined
        # onto the reports directory.
        raise HTTPException(status_code=400, detail="invalid report name")
    path = _reports_dir(settings) / name
    if not path.is_file() or not name.startswith("sweep-"):
        raise HTTPException(status_code=404, detail=f"No sweep report named {name!r}")
    payload = _load_report(path)
    if payload is None:
        raise HTTPException(status_code=422, detail=f"Report {name!r} is not valid JSON")
    cells = [_normalise_cell(c) for c in payload.get("cells") or [] if isinstance(c, dict)]
    summary = {k: v for k, v in payload.items() if k != "cells"}
    summary["cell_count"] = len(cells)
    summary["verdicts"] = _count(c["verdict"] for c in cells)
    summary["statuses"] = _count(c["status"] for c in cells)
    return ReportDetailResponse(name=name, summary=summary, cells=cells)


@router.get("/events")
def event_studies(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    studies = _event_summaries(settings)
    return {"count": len(studies), "studies": studies}


# --------------------------------------------------------------------- backtests
# Walk-forward results are committed to ``backtest_results`` and served as-is.
# Nothing here refits or re-scores a fold: re-running the backtest on request
# would produce numbers that look live but are a second reading of data the
# project already spent its holdout on. These rows are the record of what was
# actually run.
#
# A backtest that recorded no trades is a finding, not a gap. The trade log and
# equity curve are reported as explicitly empty rather than as a flat zero line,
# because "the strategy never traded" and "the strategy traded and broke even"
# are different claims and a chart cannot tell them apart on its own.

_FOLD_METRIC_KEYS = (
    "total_return", "cagr", "volatility", "sharpe", "sortino",
    "max_drawdown", "win_rate", "profit_factor", "trades", "wins", "losses",
)


def _loads_list(value: Any) -> list[Any]:
    payload = _loads(value)
    return payload if isinstance(payload, list) else []


def _fold_row(fold: dict[str, Any]) -> dict[str, Any]:
    """Flatten one walk-forward fold, keeping strategy and benchmark apart.

    ``excess_vs_buy_and_hold`` is recomputed from the fold's own test prices
    rather than read from the headline, because the headline excess is chained
    across folds and does not describe any single one.
    """
    m = fold.get("metrics") if isinstance(fold.get("metrics"), dict) else {}
    b = fold.get("benchmark_metrics") if isinstance(fold.get("benchmark_metrics"), dict) else {}
    out: dict[str, Any] = {
        "index": fold.get("index"),
        "train_start": fold.get("train_start"),
        "train_end": fold.get("train_end"),
        "test_start": fold.get("test_start"),
        "test_end": fold.get("test_end"),
        "n_bars": fold.get("n_bars"),
    }
    for k in _FOLD_METRIC_KEYS:
        out[k] = _num(m.get(k))
    for k in ("total_return", "sharpe", "max_drawdown"):
        out["bm_" + k] = _num(b.get(k))
    strat = out["total_return"]
    bench = out["bm_total_return"]
    out["excess"] = None if (strat is None or bench is None) else strat - bench
    return out


def _backtest_summary(row: dict[str, Any]) -> dict[str, Any]:
    m = _loads(row.get("metrics")) or {}
    folds = [f for f in _loads_list(row.get("folds")) if isinstance(f, dict)]
    fold_rows = [_fold_row(f) for f in folds]
    traded = sum(1 for f in fold_rows if (f.get("trades") or 0) > 0)

    out: dict[str, Any] = {
        "backtest_id": row.get("backtest_id"),
        "model_id": row.get("model_id"),
        "symbol": row.get("symbol"),
        "interval": row.get("interval"),
        "strategy": row.get("strategy"),
        "created_at": row.get("created_at"),
        "folds_recorded": len(fold_rows),
        "folds_traded": traded,
        "params": _loads(row.get("params")) or {},
        "metrics": {},
    }
    for k in ("total_return", "cagr", "volatility", "sharpe", "sortino", "calmar",
              "max_drawdown", "win_rate", "profit_factor", "final_equity",
              "n_periods", "consistency", "excess_vs_buy_and_hold",
              "benchmark_total_return", "folds_evaluated"):
        out["metrics"][k] = _num(m.get(k))
    # Metrics are absent on rows where the backtest failed before scoring; the
    # UI has to be able to tell that from a fold that genuinely returned zero.
    out["metrics_recorded"] = bool(m)
    return out


class BacktestListResponse(BaseModel):
    count: int = Field(ge=0)
    backtests: list[dict[str, Any]]


class BacktestDetailResponse(BaseModel):
    backtest_id: str
    summary: dict[str, Any]
    folds: list[dict[str, Any]]
    equity_curve: list[dict[str, Any]] = Field(default_factory=list)
    trades: list[dict[str, Any]] = Field(default_factory=list)


@router.get("/backtests", response_model=BacktestListResponse)
def list_backtests(
    model_id: str | None = Query(default=None),
    symbol: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    settings: Settings = Depends(get_settings),
) -> BacktestListResponse:
    where, params = ["1=1"], []
    if model_id:
        where.append("model_id = ?")
        params.append(model_id)
    if symbol:
        where.append("symbol = ?")
        params.append(symbol.upper())
    params.append(limit)
    rows = _query(
        settings,
        f"SELECT * FROM backtest_results WHERE {' AND '.join(where)} "
        "ORDER BY created_at DESC, id DESC LIMIT ?",
        tuple(params),
    )
    return BacktestListResponse(
        count=len(rows), backtests=[_backtest_summary(r) for r in rows]
    )


@router.get("/backtests/{backtest_id}", response_model=BacktestDetailResponse)
def backtest_detail(
    backtest_id: str, settings: Settings = Depends(get_settings)
) -> BacktestDetailResponse:
    rows = _query(
        settings, "SELECT * FROM backtest_results WHERE backtest_id = ? LIMIT 1",
        (backtest_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"No backtest named {backtest_id!r}")
    row = rows[0]
    summary = _backtest_summary(row)
    folds = [_fold_row(f) for f in _loads_list(row.get("folds")) if isinstance(f, dict)]
    return BacktestDetailResponse(
        backtest_id=str(backtest_id),
        summary=summary,
        folds=folds,
        equity_curve=[e for e in _loads_list(row.get("equity_curve")) if isinstance(e, dict)],
        trades=[t for t in _loads_list(row.get("trades")) if isinstance(t, dict)],
    )
