"""Pydantic schemas for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    database: dict[str, Any]
    providers: list[str]
    sentiment_backend: str | None = None
    data_dir: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbols: list[str] = Field(min_length=1, max_length=25)
    interval: str = "1h"
    providers: list[str] | None = None
    lookback_days: int = Field(default=730, ge=1, le=4000)
    include_news: bool = True
    include_onchain: bool = True

    @field_validator("symbols")
    @classmethod
    def _normalize(cls, values: list[str]) -> list[str]:
        return [v.replace("/", "_").replace("-", "_").upper() for v in values]

    @field_validator("interval")
    @classmethod
    def _valid_interval(cls, value: str) -> str:
        allowed = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"}
        if value not in allowed:
            raise ValueError(f"interval must be one of {sorted(allowed)}")
        return value


class IngestResponse(BaseModel):
    run_id: str | None = None
    candles: list[dict[str, Any]]
    news: dict[str, Any] | None = None
    onchain: list[dict[str, Any]] = Field(default_factory=list)
    total_rows: int


class CandlePoint(BaseModel):
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandleSeriesResponse(BaseModel):
    symbol: str
    interval: str
    exchange: str | None = None
    count: int
    candles: list[CandlePoint]
    coverage: dict[str, Any] | None = None


class NewsItem(BaseModel):
    uid: str
    source: str
    title: str
    url: str
    ts: datetime
    category: str
    sentiment_score: float | None = None
    summary: str | None = None


class NewsResponse(BaseModel):
    count: int
    items: list[NewsItem]
    sentiment: dict[str, Any] | None = None
    by_source: dict[str, int] = Field(default_factory=dict)
    by_category: dict[str, int] = Field(default_factory=dict)


class TrainForecasterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    interval: str = "1h"
    horizon: int = Field(default=1, ge=1, le=48)
    kind: Literal["lstm", "transformer", "mlp"] | None = None
    epochs: int | None = Field(default=None, ge=1, le=500)
    seq_len: int | None = Field(default=None, ge=8, le=512)
    start: datetime | None = None
    end: datetime | None = None

    @field_validator("symbol")
    @classmethod
    def _normalize(cls, value: str) -> str:
        return value.replace("/", "_").replace("-", "_").upper()


class TrainRLRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    interval: str = "1h"
    timesteps: int | None = Field(default=None, ge=100, le=5_000_000)
    n_envs: int = Field(default=1, ge=1, le=8)
    start: datetime | None = None
    end: datetime | None = None

    @field_validator("symbol")
    @classmethod
    def _normalize(cls, value: str) -> str:
        return value.replace("/", "_").replace("-", "_").upper()


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    symbol: str | None = None
    interval: str | None = None
    threshold: float = Field(default=0.0, ge=-1.0, le=1.0)
    n_folds: int = Field(default=3, ge=1, le=20)


class ModelInfo(BaseModel):
    model_id: str
    version: int
    kind: str
    symbol: str
    interval: str
    framework: str
    feature_count: int
    target: str
    stage: str
    is_active: bool
    metrics: dict[str, Any]
    params: dict[str, Any] = Field(default_factory=dict)
    artifact_path: str | None = None
    trained_from: datetime | None = None
    trained_to: datetime | None = None
    created_at: datetime | None = None


class ModelListResponse(BaseModel):
    count: int
    models: list[ModelInfo]


class PredictionResponse(BaseModel):
    model_id: str
    symbol: str
    interval: str
    horizon: int
    as_of: datetime
    last_price: float
    predicted_return: float
    predicted_price: float
    lower_68: float
    upper_68: float
    confidence: float
    feature_count: int
    metrics: dict[str, Any] = Field(default_factory=dict)


class ForecastPoint(BaseModel):
    ts: datetime
    predicted_return: float
    predicted_price: float
    predicted_volatility: float
    lower_68: float
    upper_68: float
    confidence: float


class ForecastSeriesResponse(BaseModel):
    model_id: str
    symbol: str
    interval: str
    count: int
    points: list[ForecastPoint]


class RunInfo(BaseModel):
    run_id: str
    kind: str
    status: str
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None


class RunListResponse(BaseModel):
    count: int
    runs: list[RunInfo]


class BacktestMetrics(BaseModel):
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    final_equity: float
    n_periods: int


class BacktestResponse(BaseModel):
    run_id: str
    backtest_id: str
    model_id: str
    symbol: str
    interval: str
    strategy: str
    threshold: float
    metrics: dict[str, Any]
    folds: list[dict[str, Any]] = Field(default_factory=list)
    benchmarks: dict[str, Any] = Field(default_factory=dict)
    dataset: dict[str, Any] = Field(default_factory=dict)


class JobResponse(BaseModel):
    job_id: str
    kind: str
    status: str
    progress: float
    stage: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    queued_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class JobListResponse(BaseModel):
    count: int
    jobs: list[JobResponse]


class JobSubmitResponse(BaseModel):
    job_id: str
    kind: str
    status: str
    poll_url: str


class CoverageEntry(BaseModel):
    dataset: str
    symbol: str
    interval: str
    provider: str
    start: datetime
    end: datetime
    rows: int


class CoverageResponse(BaseModel):
    count: int
    entries: list[CoverageEntry]


class DatasetStats(BaseModel):
    dataset: str
    exists: bool
    rows: int
    columns: list[str]
    start: datetime | None = None
    end: datetime | None = None
    size_bytes: int = 0


class WebsocketMessage(BaseModel):
    type: str
    data: dict[str, Any]
    ts: datetime | None = None