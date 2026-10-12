"""Application configuration, loaded from environment with sane local defaults."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CQIP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "crypto-quant-intelligence-platform"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    log_format: str = "console"

    data_dir: Path = DEFAULT_DATA_DIR
    parquet_dir: Path | None = None
    artifact_dir: Path | None = None
    db_url: str | None = None

    http_timeout: float = 30.0
    http_max_connections: int = 20
    http_max_retries: int = 4
    http_backoff_base: float = 0.8
    http_backoff_max: float = 30.0
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )

    default_exchange: str = "gate"
    default_quote: str = "USDT"
    default_symbols: list[str] = Field(default_factory=lambda: ["BTC_USDT"])

    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_batch_size: int = 16
    sentiment_max_length: int = 256
    allow_model_download: bool = True

    forecaster_kind: str = "lstm"
    forecaster_hidden_size: int = 128
    forecaster_num_layers: int = 2
    forecaster_dropout: float = 0.1
    forecaster_seq_len: int = 64
    forecaster_epochs: int = 30
    forecaster_batch_size: int = 64
    forecaster_lr: float = 1e-3
    forecaster_patience: int = 6
    # `plateau` keeps the historical behaviour; `cosine` and `onecycle` are
    # opt-in and fit runs with a known, fixed epoch budget.
    forecaster_lr_schedule: str = "plateau"
    forecaster_weight_decay: float = 1e-4
    # Target construction. `log_return` is the historical fixed-horizon target;
    # the other two change what the model is being asked to predict.
    forecaster_target: str = "log_return"
    forecaster_normalize_volatility: bool = False
    forecaster_triple_barrier_take_profit_vol: float = 1.0
    forecaster_triple_barrier_stop_loss_vol: float = 1.0
    forecaster_triple_barrier_max_hold: int | None = None
    # Auxiliary binary direction head, and how strongly it contributes to loss.
    forecaster_aux_direction_weight: float = 0.0
    # Seed ensembling: 1 disables it, N trains N members and averages them.
    forecaster_ensemble_size: int = 1
    # Drop features that are (near) zero on more than 1 - this fraction of rows.
    # 0.0 disables pruning so existing feature schemas stay valid.
    feature_min_active_fraction: float = 0.0
    # Number of walk-forward folds and how many bars of purge/embargo to apply
    # around each boundary. `backtest_purge_bars` follows the label horizon and
    # `backtest_embargo_bars` the window length when left at 0.
    backtest_folds: int = 3
    backtest_purge_bars: int = 0
    backtest_embargo_bars: int = 0
    # Moving-block bootstrap settings for confidence intervals on test metrics.
    bootstrap_iterations: int = 1000
    bootstrap_confidence: float = 0.95
    seed: int = 42

    rl_algorithm: str = "PPO"
    rl_timesteps: int = 20_000
    rl_learning_rate: float = 3e-4
    rl_batch_size: int = 64
    rl_n_envs: int = 1
    rl_initial_capital: float = 10_000.0

    backtest_initial_capital: float = 10_000.0
    backtest_fee_bps: float = 10.0
    backtest_slippage_bps: float = 5.0
    backtest_train_ratio: float = 0.7
    backtest_val_ratio: float = 0.15
    backtest_anchor: str = "expanding"

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        return v.upper()

    @field_validator("log_format")
    @classmethod
    def _lower_log_format(cls, v: str) -> str:
        v = v.lower()
        return v if v in {"console", "json"} else "console"

    @field_validator("forecaster_lr_schedule")
    @classmethod
    def _known_lr_schedule(cls, v: str) -> str:
        v = v.lower()
        if v not in {"plateau", "cosine", "onecycle"}:
            raise ValueError(
                "forecaster_lr_schedule must be one of plateau, cosine, onecycle"
            )
        return v

    @field_validator("forecaster_target")
    @classmethod
    def _known_target(cls, v: str) -> str:
        v = v.lower()
        if v not in {"log_return", "vol_normalized", "triple_barrier", "move_event"}:
            raise ValueError(
                "forecaster_target must be one of log_return, vol_normalized, "
                "triple_barrier, move_event"
            )
        return v

    @field_validator("bootstrap_confidence")
    @classmethod
    def _unit_interval(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("bootstrap_confidence must be strictly between 0 and 1")
        return v

    @property
    def backtest_purge_bars_effective(self) -> int:
        return max(0, int(self.backtest_purge_bars))

    @property
    def resolved_parquet_dir(self) -> Path:
        return self.parquet_dir or self.data_dir / "parquet"

    @property
    def resolved_artifact_dir(self) -> Path:
        return self.artifact_dir or self.data_dir / "artifacts"

    @property
    def resolved_db_url(self) -> str:
        return self.db_url or f"sqlite:///{(self.data_dir / 'cqip.db').as_posix()}"

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.resolved_parquet_dir,
            self.resolved_artifact_dir,
            self.resolved_artifact_dir / "models",
            self.resolved_artifact_dir / "runs",
            self.data_dir / "cache",
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings