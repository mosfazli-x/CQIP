"""Typer CLI for ingestion, training, backtesting and serving."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from cqip.backtest import BenchmarkSuite
from cqip.config import get_settings
from cqip.db import get_engine
from cqip.ingest import Ingestor
from cqip.logging_config import configure_logging
from cqip.pipeline import TrainingPipeline
from cqip.providers import available_providers
from cqip.sentiment import SentimentScorer
from cqip.storage import ParquetStore
from cqip.tracking import ModelRegistry, serialize_model

app = typer.Typer(
    name="cqip",
    help="Crypto Quant Intelligence Platform",
    no_args_is_help=True,
    add_completion=False,
)
ingest_app = typer.Typer(help="Ingest market, news and on-chain data", no_args_is_help=True)
train_app = typer.Typer(help="Train forecasting and RL models", no_args_is_help=True)
model_app = typer.Typer(help="Inspect and manage the model registry", no_args_is_help=True)
data_app = typer.Typer(help="Inspect stored datasets", no_args_is_help=True)

app.add_typer(ingest_app, name="ingest")
app.add_typer(train_app, name="train")
app.add_typer(model_app, name="model")
app.add_typer(data_app, name="data")

console = Console()


def _print_results(results: list[dict[str, Any]]) -> None:
    table = Table(title="Ingestion Results", show_lines=False)
    table.add_column("Provider", style="cyan")
    table.add_column("Symbol", style="magenta")
    table.add_column("Rows", justify="right")
    table.add_column("Written", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Errors", style="red")

    for result in results:
        table.add_row(
            str(result.get("provider", "")),
            str(result.get("symbol") or "-"),
            f"{result.get('rows', 0):,}",
            f"{result.get('inserted', 0):,}",
            f"{result.get('duration_seconds', 0):.2f}s",
            "; ".join(result.get("errors", []))[:60],
        )
    console.print(table)


@app.command("universe")
def universe_command(
    symbol: list[str] = typer.Option(
        None, "--symbol", "-s",
        help="Base ticker; repeat the flag. Defaults to the curated top list.",
    ),
    provider_name: str = typer.Option("gate", help="Exchange to resolve against"),
    quote: str = typer.Option("USDT", help="Quote asset"),
    json_out: bool = typer.Option(False, "--json", help="Emit raw JSON"),
) -> None:
    """Resolve a coin list against live exchange listings and report gaps.

    Stablecoins are flagged separately: a model trained on a pegged price
    reports noise near 50% directional accuracy, which reads as a broken model
    rather than a bad target.
    """
    from cqip.universe import TOP_CRYPTO_SYMBOLS, resolve_universe

    settings = get_settings()
    try:
        universe = resolve_universe(
            symbol or TOP_CRYPTO_SYMBOLS,
            provider_name=provider_name, quote=quote, settings=settings,
        )
    except KeyError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    if json_out:
        console.print_json(data={
            "provider": universe.provider,
            "quote": universe.quote,
            "tradable": universe.tradable,
            "targets": universe.targets,
            "statuses": [s.as_dict() for s in universe.statuses],
        })
        return

    table = Table(title=f"Symbol Universe ({universe.provider}, {universe.quote} quoted)")
    table.add_column("Requested", style="cyan")
    table.add_column("Pair", style="magenta")
    table.add_column("Trainable", justify="center")
    table.add_column("Note")

    for status in universe.statuses:
        if not status.available:
            table.add_row(status.requested, "-", "[red]no[/]", f"[red]{status.reason}[/]")
        elif status.is_stablecoin:
            table.add_row(
                status.requested, status.symbol, "[yellow]no[/]",
                f"[yellow]{status.reason}[/]",
            )
        else:
            note = status.reason or ""
            table.add_row(
                status.requested, status.symbol, "[green]yes[/]",
                f"[dim]{note}[/]" if note else "",
            )
    console.print(table)

    console.print(
        f"[bold]{len(universe.tradable)}[/] tradable, "
        f"[green]{len(universe.targets)}[/] usable as prediction targets, "
        f"[yellow]{len(universe.stablecoins)}[/] stablecoins, "
        f"[red]{len(universe.skipped)}[/] unavailable"
    )


@app.command("backfill")
def backfill_command(
    symbol: list[str] = typer.Option(
        None, "--symbol", "-s",
        help="Base ticker or BASE_QUOTE pair; repeat the flag. Defaults to the curated top list.",
    ),
    interval: list[str] = typer.Option(
        None, "--interval", "-i",
        help="Timeframe; repeat the flag. Defaults to 1d then 1h.",
    ),
    provider_name: str = typer.Option("gate", help="Exchange to ingest from"),
    concurrency: int = typer.Option(3, help="Parallel requests"),
    targets_only: bool = typer.Option(
        True, "--targets-only/--include-all",
        help="Skip stablecoins, which are not useful prediction subjects",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the plan without fetching"),
) -> None:
    """Backfill full available history for many symbols in one run.

    Daily bars are pulled back to the listing date; hourly bars are capped by
    the exchange's ~1 year retention, which is reported rather than hidden.
    """
    import asyncio

    from cqip.universe import DEFAULT_INTERVAL_DEPTH_DAYS, TOP_CRYPTO_SYMBOLS, resolve_universe

    settings = get_settings()
    requested = list(symbol) if symbol else list(TOP_CRYPTO_SYMBOLS)
    try:
        universe = resolve_universe(requested, provider_name=provider_name, settings=settings)
    except KeyError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    targets = universe.targets if targets_only else universe.tradable
    intervals = list(interval) if interval else ["1d", "1h"]

    if universe.skipped:
        console.print("[yellow]Skipped:[/] " + ", ".join(
            f"{s.requested} ({s.reason})" for s in universe.skipped
        ))
    if targets_only and universe.stablecoins:
        console.print(
            "[dim]Excluded as prediction targets: "
            + ", ".join(s.symbol or s.requested for s in universe.stablecoins)
            + "[/]"
        )
    if not targets:
        console.print("[red]No usable symbols resolved; nothing to backfill.[/]")
        raise typer.Exit(code=1)

    from cqip.ingest import Ingestor

    plan_table = Table(title="Backfill Plan", show_header=True)
    plan_table.add_column("Symbol", style="cyan")
    plan_table.add_column("TF", style="magenta")
    plan_table.add_column("Days", justify="right")
    for pair in targets:
        for iv in intervals:
            depth = DEFAULT_INTERVAL_DEPTH_DAYS.get(iv)
            plan_table.add_row(
                pair, iv, f"{depth:,}" if depth else "[red]unknown[/]"
            )
    console.print(plan_table)

    if dry_run:
        console.print("[dim]Dry run; nothing was fetched.[/]")
        return

    console.print(
        f"[dim]Fetching {len(targets) * len(intervals)} symbol/interval combinations...[/]"
    )
    ingestor = Ingestor(settings)
    planned = [i for i in intervals if i in DEFAULT_INTERVAL_DEPTH_DAYS]
    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("backfilling", total=len(planned))

        def on_step(done: int, total: int, item: dict[str, Any]) -> None:
            progress.update(
                task, completed=done,
                description=(
                    f"{item['interval']} x {item.get('symbols', '?')} symbols "
                    f"({done}/{total})"
                ),
            )

        summary = asyncio.run(ingestor.backfill_universe(
            targets, intervals=intervals, providers=[provider_name],
            concurrency=concurrency, on_progress=on_step,
        ))

    _print_results(summary["results"])
    console.print(
        f"[bold]{summary['total_rows']:,}[/] rows fetched, "
        f"[bold]{summary['total_inserted']:,}[/] written"
    )
    if summary["errors"]:
        console.print(f"[yellow]{len(summary['errors'])} errors:[/]")
        for error in summary["errors"][:10]:
            console.print(f"  [yellow]{error}[/]")


@app.command()
def version() -> None:
    """Show environment and provider configuration."""
    settings = get_settings()
    scorer = SentimentScorer(settings)
    scorer._load()

    table = Table(title="CQIP Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Environment", settings.environment)
    table.add_row("Data dir", str(settings.data_dir))
    table.add_row("Database", settings.resolved_db_url.split("://")[0])
    table.add_row("Providers", ", ".join(available_providers()))
    table.add_row("Default exchange", settings.default_exchange)
    table.add_row("Sentiment backend", scorer.backend)
    table.add_row("Torch device", "cpu" if not settings.debug else "debug-cpu")
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option(None, help="Bind host"),
    port: int = typer.Option(None, help="Bind port"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
) -> None:
    """Start the HTTP API server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "cqip.api:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


@app.command()
def check() -> None:
    """Verify database and storage connectivity."""
    settings = get_settings()
    engine = get_engine(settings)
    store = ParquetStore(settings=settings)

    health = engine.healthcheck()
    console.print(Panel(
        f"Database: {'OK' if health.get('ok') else 'FAIL'}\n"
        f"Parquet root: {store.root}\n"
        f"Parquet writable: {store.root.exists()}",
        title="Health Check",
        border_style="green" if health.get("ok") else "red",
    ))
    if not health.get("ok"):
        raise typer.Exit(code=1)


@ingest_app.command("candles")
def ingest_candles(
    symbols: list[str] = typer.Argument(..., help="Symbols e.g. BTC_USDT ETH_USDT"),
    interval: str = typer.Option("1h", help="Candle interval"),
    providers: list[str] = typer.Option(None, "--provider", "-p", help="Exchange providers"),
    lookback_days: int = typer.Option(730, help="Days of history to fetch"),
    concurrency: int = typer.Option(3, help="Parallel requests"),
) -> None:
    """Ingest OHLCV candles from exchange APIs."""
    settings = get_settings()
    ingestor = Ingestor(settings)

    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
        BarColumn(), TextColumn("{task.completed} files"),
        console=console, transient=False,
    ) as progress:
        task = progress.add_task("ingesting candles", total=len(symbols))
        results = asyncio.run(ingestor.ingest_candles(
            symbols, interval, providers, lookback_days=lookback_days, concurrency=concurrency
        ))
        progress.update(task, completed=len(symbols))

    _print_results([r.as_dict() for r in results])
    failures = [r for r in results if r.errors]
    if failures:
        console.print(f"[yellow]{len(failures)} provider/symbol combinations failed[/]")


@ingest_app.command("news")
def ingest_news(
    limit: int = typer.Option(100, help="Articles per source"),
    no_sentiment: bool = typer.Option(False, help="Skip sentiment scoring"),
) -> None:
    """Ingest news articles from RSS feeds."""
    settings = get_settings()
    ingestor = Ingestor(settings)

    result = asyncio.run(ingestor.ingest_news(
        with_sentiment=not no_sentiment, limit_per_source=limit
    ))
    console.print(Panel(
        f"Articles fetched: {result.rows}\n"
        f"New articles written: {result.inserted}\n"
        f"Duplicates skipped: {result.skipped}\n"
        f"Errors: {'; '.join(result.errors) or 'none'}",
        title="News Ingestion",
        border_style="green" if not result.errors else "yellow",
    ))


@ingest_app.command("onchain")
def ingest_onchain(no_fng: bool = typer.Option(False, help="Skip Fear & Greed index")) -> None:
    """Ingest on-chain and market-wide metrics."""
    settings = get_settings()
    ingestor = Ingestor(settings)
    results = asyncio.run(ingestor.ingest_onchain(with_sentiment_index=not no_fng))
    _print_results([r.as_dict() for r in results])


@ingest_app.command("all")
def ingest_all(
    symbols: list[str] = typer.Argument(None, help="Symbols (default: settings.default_symbols)"),
    interval: str = typer.Option("1h", help="Candle interval"),
    providers: list[str] = typer.Option(None, "--provider", "-p"),
    lookback_days: int = typer.Option(730, help="Days of history"),
) -> None:
    """Ingest candles, news and on-chain data in one pass."""
    settings = get_settings()
    ingestor = Ingestor(settings)
    symbols = symbols or settings.default_symbols

    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console) as progress:
        task = progress.add_task("ingesting all sources", total=None)
        summary = asyncio.run(ingestor.ingest_all(
            symbols, interval, providers, lookback_days=lookback_days
        ))
        progress.update(task, completed=True)

    console.print(Panel(f"Total rows written: {summary['total_rows']:,}",
                        title="Ingestion Complete", border_style="green"))


@train_app.command("forecaster")
def train_forecaster(
    symbol: str = typer.Argument(..., help="Symbol e.g. BTC_USDT"),
    interval: str = typer.Option("1h", help="Candle interval"),
    horizon: int = typer.Option(1, help="Prediction horizon in bars"),
    kind: str = typer.Option("lstm", help="lstm | transformer | mlp"),
    epochs: int = typer.Option(None, help="Max epochs"),
    seq_len: int = typer.Option(None, help="Sequence length"),
) -> None:
    """Train a deep-learning forecaster."""
    settings = get_settings()
    pipeline = TrainingPipeline(settings=settings)

    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console) as progress:
        task = progress.add_task(f"training {kind} on {symbol} {interval}", total=None)
        result = pipeline.train_forecaster(
            symbol, interval, horizon=horizon, kind=kind, epochs=epochs, seq_len=seq_len
        )
        progress.update(task, completed=True)

    console.print(Panel(
        f"Model ID: {result['model_id']}\n"
        f"Rows: {result['dataset']['rows']:,}\n"
        f"Features: {result['dataset']['features']}\n"
        f"Val directional accuracy: {result['val_metrics'].get('directional_accuracy', 0):.4f}\n"
        f"Test MAE: {result['test_metrics'].get('test_mae', 0):.6f}\n"
        f"Test directional accuracy: {result['test_metrics'].get('test_directional_accuracy', 0):.4f}\n"
        f"68% interval coverage: {result['test_metrics'].get('test_68pct_interval_coverage', 0):.4f}\n"
        f"Artifact: {result['artifact_path']}",
        title="Forecaster Training Complete",
        border_style="green",
    ))


@train_app.command("rl")
def train_rl(
    symbol: str = typer.Argument(..., help="Symbol e.g. BTC_USDT"),
    interval: str = typer.Option("1h", help="Candle interval"),
    timesteps: int = typer.Option(20_000, help="Training timesteps"),
    n_envs: int = typer.Option(1, help="Parallel environments"),
) -> None:
    """Train a PPO reinforcement-learning trading agent."""
    settings = get_settings()
    pipeline = TrainingPipeline(settings=settings)

    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console) as progress:
        task = progress.add_task(f"training PPO on {symbol} {interval}", total=None)
        result = pipeline.train_rl_agent(symbol, interval, timesteps=timesteps, n_envs=n_envs)
        progress.update(task, completed=True)

    metrics = result["metrics"]
    console.print(Panel(
        f"Model ID: {result['model_id']}\n"
        f"Episodes: {metrics['eval_episodes']}\n"
        f"Mean reward: {metrics['eval_mean_reward']:.4f}\n"
        f"Std reward: {metrics['eval_std_reward']:.4f}\n"
        f"Artifact: {result['artifact_path']}",
        title="RL Training Complete", border_style="green",
    ))


@train_app.command("backtest")
def run_backtest(
    model_id: str = typer.Argument(..., help="Model ID to backtest"),
    threshold: float = typer.Option(0.0, help="Signal threshold"),
    folds: int = typer.Option(3, help="Walk-forward folds"),
) -> None:
    """Run a walk-forward backtest of a forecaster."""
    settings = get_settings()
    pipeline = TrainingPipeline(settings=settings)

    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console) as progress:
        task = progress.add_task(f"backtesting {model_id}", total=None)
        result = pipeline.backtest_predictions(model_id, threshold=threshold, n_folds=folds)
        progress.update(task, completed=True)

    metrics = result["metrics"]
    table = Table(title=f"Walk-Forward Backtest: {result['symbol']} {result['interval']}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="white")
    for key in ("total_return", "cagr", "volatility", "sharpe", "sortino",
                "max_drawdown", "win_rate", "profit_factor", "final_equity",
                "folds_evaluated", "consistency"):
        if key in metrics:
            value = metrics[key]
            table.add_row(key, f"{value:,.4f}" if isinstance(value, float) else str(value))
    console.print(table)

    if result.get("folds"):
        fold_table = Table(title="Per-Fold Results")
        fold_table.add_column("Fold", justify="right")
        fold_table.add_column("Return", justify="right")
        fold_table.add_column("Sharpe", justify="right")
        fold_table.add_column("MaxDD", justify="right")
        fold_table.add_column("Trades", justify="right")
        for fold in result["folds"]:
            fold_metrics = fold["metrics"]
            fold_table.add_row(
                str(fold["index"]), f"{fold_metrics['total_return']:.2%}",
                f"{fold_metrics['sharpe']:.3f}", f"{fold_metrics['max_drawdown']:.2%}",
                str(fold_metrics.get("trades", 0)),
            )
        console.print(fold_table)


@train_app.command("benchmark")
def benchmark(
    symbol: str = typer.Argument(..., help="Symbol e.g. BTC_USDT"),
    interval: str = typer.Option("1h", help="Candle interval"),
) -> None:
    """Compare baseline strategies on historical data."""
    settings = get_settings()
    ingestor = Ingestor(settings)
    frame = ingestor.load_dataset(symbol, interval)

    if frame.empty:
        console.print(f"[red]No data for {symbol} {interval}. Run 'cqip ingest candles' first.[/]")
        raise typer.Exit(code=1)

    suite = BenchmarkSuite(settings)
    results = suite.run_all(frame)

    table = Table(title=f"Strategy Benchmarks: {symbol} {interval}")
    table.add_column("Strategy", style="cyan")
    for metric in ("total_return", "cagr", "sharpe", "max_drawdown", "win_rate"):
        table.add_column(metric, justify="right")

    for name, metrics in results.items():
        table.add_row(
            name,
            f"{metrics.get('total_return', 0):.2%}",
            f"{metrics.get('cagr', 0):.2%}",
            f"{metrics.get('sharpe', 0):.3f}",
            f"{metrics.get('max_drawdown', 0):.2%}",
            f"{metrics.get('win_rate', 0):.2%}",
        )
    console.print(table)


@model_app.command("list")
def list_models(
    kind: str = typer.Option(None, help="Filter by kind"),
    symbol: str = typer.Option(None, help="Filter by symbol"),
) -> None:
    """List registered models."""
    settings = get_settings()
    registry = ModelRegistry(get_engine(settings), settings)
    records = registry.list_models(kind=kind, symbol=symbol)

    if not records:
        console.print("[yellow]No models registered. Train one with 'cqip train forecaster'.[/]")
        return

    table = Table(title="Model Registry")
    table.add_column("Model ID", style="cyan")
    table.add_column("Ver", justify="right")
    table.add_column("Kind")
    table.add_column("Symbol")
    table.add_column("Interval")
    table.add_column("Stage")
    table.add_column("Sharpe/DirAcc", justify="right")

    for record in records:
        metrics = record.metrics or {}
        score = metrics.get("test_strategy_sharpe") or metrics.get("directional_accuracy") or 0
        table.add_row(
            record.model_id, str(record.version), record.kind, record.symbol,
            record.interval, record.stage,
            f"{score:.3f}" if isinstance(score, (int, float)) else "-",
        )
    console.print(table)


@model_app.command("show")
def show_model(model_id: str = typer.Argument(..., help="Model ID")) -> None:
    """Show full detail for one model."""
    settings = get_settings()
    registry = ModelRegistry(get_engine(settings), settings)
    record = registry.get_active(model_id)
    if record is None:
        console.print(f"[red]Model {model_id!r} not found[/]")
        raise typer.Exit(code=1)
    console.print_json(data=json.loads(json.dumps(serialize_model(record), default=str)))


@model_app.command("predict")
def predict(model_id: str = typer.Argument(..., help="Model ID")) -> None:
    """Generate a forecast for the latest data."""
    settings = get_settings()
    pipeline = TrainingPipeline(settings=settings)
    try:
        result = pipeline.predict_latest(model_id)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    table = Table(title=f"Prediction: {result['symbol']} {result['interval']}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("As of", str(result["as_of"]))
    table.add_row("Last price", f"{result['last_price']:,.4f}")
    table.add_row("Predicted return", f"{result['predicted_return']:+.4%}")
    table.add_row("Predicted price", f"{result['predicted_price']:,.4f}")
    table.add_row("68% lower", f"{result['lower_68']:,.4f}")
    table.add_row("68% upper", f"{result['upper_68']:,.4f}")
    table.add_row("Confidence", f"{result['confidence']:.4f}")
    table.add_row("Features", str(result["feature_count"]))
    console.print(table)


@model_app.command("promote")
def promote(model_id: str = typer.Argument(..., help="Model ID")) -> None:
    """Promote a model to production stage."""
    settings = get_settings()
    registry = ModelRegistry(get_engine(settings), settings)
    if registry.promote(model_id, "production"):
        console.print(f"[green]Promoted {model_id}[/]")
    else:
        console.print(f"[red]Model {model_id!r} not found[/]")
        raise typer.Exit(code=1)


@model_app.command("remove")
def remove_model(model_id: str = typer.Argument(..., help="Model ID")) -> None:
    """Unregister and delete a model artifact."""
    settings = get_settings()
    registry = ModelRegistry(get_engine(settings), settings)
    if registry.remove(model_id):
        console.print(f"[green]Removed {model_id}[/]")
    else:
        console.print(f"[red]Model {model_id!r} not found[/]")
        raise typer.Exit(code=1)


@data_app.command("info")
def data_info(dataset: str = typer.Argument("ohlcv", help="Dataset name")) -> None:
    """Show row counts, columns and time coverage for a dataset."""
    settings = get_settings()
    store = ParquetStore(settings=settings)
    start, end = store.date_range(dataset)
    path = store.dataset_path(dataset)
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0

    table = Table(title=f"Dataset: {dataset}")
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    table.add_row("Exists", str(store.exists(dataset)))
    table.add_row("Rows", f"{store.row_count(dataset):,}")
    table.add_row("Size", f"{size / 1_048_576:.2f} MB")
    table.add_row("Start", str(start))
    table.add_row("End", str(end))
    table.add_row("Columns", ", ".join(store.columns(dataset)[:20]))
    table.add_row("Symbols", ", ".join(store.symbols(dataset)[:20]))
    console.print(table)


@data_app.command("features")
def data_features(
    symbol: str = typer.Argument(..., help="Symbol e.g. BTC_USDT"),
    interval: str = typer.Option("1h", help="Candle interval"),
) -> None:
    """Preview the engineered feature matrix for a symbol."""
    settings = get_settings()
    from cqip.pipeline import DatasetBuilder

    builder = DatasetBuilder(settings)
    try:
        bundle = builder.build(symbol, interval)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    console.print(Panel(
        f"Symbol: {bundle.symbol} {bundle.interval}\n"
        f"Rows: {bundle.rows:,}\n"
        f"Features: {len(bundle.feature_columns)}\n"
        f"Range: {bundle.start} -> {bundle.end}\n"
        + (f"Warnings: {'; '.join(bundle.warnings)}" if bundle.warnings else ""),
        title="Dataset", border_style="cyan",
    ))

    table = Table(title=f"Feature Preview (last 5 rows of {len(bundle.feature_columns)} columns)")
    preview_columns = ["ts", "close", *bundle.feature_columns[:6]]
    preview = bundle.frame[preview_columns].tail(5)
    for column in preview_columns:
        table.add_column(str(column), style="cyan" if column in {"ts", "close"} else "")
    for _, row in preview.iterrows():
        table.add_row(*[
            str(row["ts"])[:16],
            f"{row['close']:.2f}",
            *[f"{row[c]:.4f}" if isinstance(row[c], (int, float)) else str(row[c])
              for c in bundle.feature_columns[:6]],
        ])
    console.print(table)


def _sweep_value(value: float | None, fmt: str = "{:+.4f}") -> str:
    return "n/a" if value is None else fmt.format(value)


def _print_sweep_table(report: Any, title: str = "Research Sweep") -> None:
    table = Table(title=title)
    table.add_column("Symbol", style="cyan")
    table.add_column("TF", style="cyan")
    table.add_column("Model", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("DirAcc", justify="right")
    table.add_column("Corr", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("MaxDD", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("vs B&H", justify="right")
    table.add_column("Verdict")

    for cell in report.cells:
        if cell.status == "ok":
            style = "green" if cell.verdict == "passes" else "yellow"
            verdict = cell.verdict
        else:
            style = "red"
            verdict = cell.status
        table.add_row(
            cell.symbol,
            cell.interval,
            cell.kind,
            f"{cell.rows:,}" if cell.rows is not None else "n/a",
            _sweep_value(cell.direction_accuracy, "{:.4f}"),
            _sweep_value(cell.correlation),
            _sweep_value(cell.sharpe, "{:+.3f}"),
            _sweep_value(cell.total_return, "{:+.2%}"),
            _sweep_value(cell.max_drawdown, "{:.2%}"),
            str(cell.trades) if cell.trades is not None else "n/a",
            _sweep_value(cell.excess_vs_buy_and_hold, "{:+.2f}"),
            f"[{style}]{verdict}[/]",
        )
    console.print(table)

    failures = [
        c for c in report.cells
        if c.status != "ok" or (c.verdict != "passes" and c.reason)
    ]
    if failures:
        detail = Table(title="Why Each Cell Failed", show_header=True)
        detail.add_column("Cell", style="cyan")
        detail.add_column("Reason")
        for cell in failures:
            detail.add_row(
                f"{cell.symbol} {cell.interval} {cell.kind}",
                cell.reason or cell.verdict,
            )
        console.print(detail)

    passing = report.passed
    console.print(Panel(
        f"Cells: {len(report.cells)}   passed gates: {len(passing)}   "
        f"elapsed: {report.duration_seconds:.1f}s\n"
        + ("No cell cleared the quality gates. Treat this as evidence the features "
           "or labels carry no signal, not as a tuning problem."
           if not passing else
           "Passing: " + ", ".join(f"{c.symbol} {c.interval} {c.kind}" for c in passing)),
        border_style="green" if passing else "yellow",
    ))


def _flatten_options(values: list[str] | None, defaults: tuple[str, ...]) -> list[str]:
    """Normalise repeated options, accepting a comma-joined value as well.

    Comma splitting stays supported for quoted input, but the repeatable flag is
    the reliable form on shells that treat ``1d`` as a numeric literal.
    """
    if not values:
        return list(defaults)
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in str(value).split(",") if part.strip())
    return out


@app.command("research")
def research(
    symbols: list[str] = typer.Argument(..., help="Symbols e.g. BTC_USDT ETH_USDT"),
    interval: list[str] = typer.Option(
        None, "--interval", "-i",
        help="Timeframe; repeat the flag for several. Also accepts '1h,4h,1d'.",
    ),
    model: list[str] = typer.Option(
        None, "--model", "-m",
        help="Scenario; repeat the flag for several. Also accepts 'lstm,mlp'.",
    ),
    epochs: int = typer.Option(5, help="Epochs per cell (screening default)"),
    folds: int = typer.Option(3, help="Walk-forward folds per cell"),
    horizon: int = typer.Option(1, help="Prediction horizon in bars"),
    threshold: float = typer.Option(0.0, help="Signal threshold"),
    min_rows: int = typer.Option(300, help="Minimum bars before a cell is run"),
    resume: bool = typer.Option(False, help="Reuse completed cells from the last sweep"),
    staged: bool = typer.Option(
        False, "--staged",
        help="Screen every cell with 1 epoch, then fully train only the top survivors",
    ),
    survivors: int = typer.Option(5, help="How many cells to promote in staged mode"),
) -> None:
    """Run a timeframe x scenario matrix and score every cell.

    Coarse timeframes are resampled from the stored base series, so a 1h, 4h and
    1d study of three architectures needs one command. Each cell is trained,
    walk-forward backtested, and judged against explicit quality gates; the matrix
    is written to ``artifacts/reports/`` for later comparison.

    Pass each timeframe as its own ``--interval`` flag. In PowerShell an unquoted
    ``1h,4h,1d`` is parsed as numbers before Python sees it, because ``d`` is a
    PowerShell numeric suffix.
    """
    from cqip.sweep import ResearchSweep

    settings = get_settings()
    sweep = ResearchSweep(settings)
    interval_list = _flatten_options(interval, ("1h", "4h", "1d"))
    model_list = _flatten_options(model, ("lstm", "transformer", "mlp"))

    previous = sweep.load_latest() if resume else None
    if resume and previous is None:
        console.print("[yellow]No previous sweep found; starting fresh.[/]")

    total = len(symbols) * len(interval_list) * len(model_list)
    console.print(Panel(
        f"Symbols:   {', '.join(symbols)}\n"
        f"Timeframes: {', '.join(interval_list)}\n"
        f"Scenarios: {', '.join(model_list)}\n"
        f"Cells: {total}   epochs: {epochs}   folds: {folds}",
        title="Research Sweep", border_style="cyan",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("sweeping", total=total)

        def on_cell(cell: Any) -> None:
            progress.update(task, advance=1)
            marker = "ok " if cell.status == "ok" else cell.status
            detail = cell.model_id or cell.reason
            progress.console.print(
                f"  [{'green' if cell.status == 'ok' else 'yellow'}]{marker}[/] "
                f"{cell.symbol} {cell.interval} {cell.kind}  {detail}"
            )

        try:
            if staged:
                screen, report = sweep.run_staged(
                    symbols=symbols,
                    intervals=interval_list,
                    kinds=model_list,
                    screen_epochs=1,
                    epochs=max(epochs, 30),
                    folds=folds,
                    horizon=horizon,
                    threshold=threshold,
                    min_rows=min_rows,
                    survivors=survivors,
                    progress=on_cell,
                )
            else:
                screen = None
                report = sweep.run(
                    symbols=symbols,
                    intervals=interval_list,
                    kinds=model_list,
                    epochs=epochs,
                    folds=folds,
                    horizon=horizon,
                    threshold=threshold,
                    min_rows=min_rows,
                    resume_from=previous,
                    progress=on_cell,
                )
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc

    if screen is not None:
        viable = [
            c for c in screen.cells
            if c.status == "ok" and (c.direction_accuracy or 0) >= 0.5
        ]
        console.print(Panel(
            f"Screened {len(screen.cells)} cells at 1 epoch; "
            f"{len(viable)} were at or above chance and {len(report.cells)} "
            f"were promoted to full training.",
            title="Screening", border_style="yellow",
        ))
        _print_sweep_table(screen, title="Screening (1 epoch)")

    path = sweep.save(report)
    _print_sweep_table(report)
    console.print(f"[dim]Report saved to {path}[/]")


def main() -> None:
    configure_logging()
    app()


if __name__ == "__main__":
    main()