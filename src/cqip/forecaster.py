"""Sequential deep-learning forecaster for crypto returns.

Architecture choice: the module ships an LSTM and a Transformer encoder sharing
one training loop. The forecaster predicts a *distribution* over the forward
log-return (Gaussian NLL) rather than a point estimate, which yields calibrated
uncertainty intervals for downstream position sizing — a point regression alone
would produce unusable confidence bands.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cqip.config import Settings, get_settings
from cqip.logging_config import get_logger

logger = get_logger(__name__)

ModelKind = Literal["lstm", "transformer", "mlp"]


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


class GaussianHead(nn.Module):
    """Predicts mean and log-variance of a Gaussian target.

    When ``aux_direction`` is set the head also emits a single logit for a binary
    up/down classification of the same target. Direction is the quantity the
    quality gate actually scores, but a pure NLL objective optimises a regression
    fit, and on near-symmetric return distributions the sign of a well-calibrated
    mean is close to the sign of noise. The auxiliary term gives the trunk a
    direct gradient towards the decision that matters, which is the standard
    multi-task argument and the reason this stays opt-in: it is an architectural
    change and must be validated, not assumed.
    """

    def __init__(self, hidden_size: int, aux_direction: bool = False) -> None:
        super().__init__()
        self.mean = nn.Linear(hidden_size, 1)
        self.log_var = nn.Linear(hidden_size, 1)
        self.aux_direction = bool(aux_direction)
        self.direction = nn.Linear(hidden_size, 1) if self.aux_direction else None

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(context)
        log_var = torch.clamp(self.log_var(context), -10.0, 6.0)
        return mean.squeeze(-1), log_var.squeeze(-1)

    def forward_full(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        mean, log_var = self(context)
        logits = (
            self.direction(context).squeeze(-1) if self.direction is not None else None
        )
        return mean, log_var, logits


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        aux_direction: bool = False,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = GaussianHead(hidden_size, aux_direction=aux_direction)
        self.dropout = nn.Dropout(dropout)

    def context(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(x)
        return self.dropout(self.norm(outputs[:, -1, :]))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.context(x))

    def forward_full(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return self.head.forward_full(self.context(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.encoding[:, : x.size(1)])


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
        max_len: int = 512,
        aux_direction: bool = False,
    ) -> None:
        super().__init__()
        d_model = hidden_size
        if d_model % num_heads:
            d_model = num_heads * (d_model // num_heads + 1)

        self.input_proj = nn.Linear(input_size, d_model)
        self.positional = PositionalEncoding(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = GaussianHead(d_model, aux_direction=aux_direction)

    def context(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.positional(h)
        mask = torch.triu(
            torch.ones(h.size(1), h.size(1), device=h.device), diagonal=1
        )
        encoded = self.encoder(h, mask=mask)
        return self.norm(encoded[:, -1, :])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.context(x))

    def forward_full(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return self.head.forward_full(self.context(x))


class MLPForecaster(nn.Module):
    """Flattened-window baseline — deliberately simple to validate the pipeline."""

    def __init__(self, input_size: int, seq_len: int = 64, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.1,
                 aux_direction: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten(), nn.Dropout(dropout)]
        in_features = input_size * seq_len
        for _ in range(num_layers):
            layers += [
                nn.Linear(in_features, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_features = hidden_size
        self.body = nn.Sequential(*layers)
        self.head = GaussianHead(hidden_size, aux_direction=aux_direction)

    def context(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.context(x))

    def forward_full(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return self.head.forward_full(self.context(x))


def build_model(
    kind: ModelKind,
    input_size: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    seq_len: int = 64,
    aux_direction: bool = False,
) -> nn.Module:
    if kind == "lstm":
        return LSTMForecaster(
            input_size, hidden_size, num_layers, dropout, aux_direction=aux_direction
        )
    if kind == "transformer":
        return TransformerForecaster(
            input_size, hidden_size, num_layers, dropout,
            max_len=max(512, seq_len * 2), aux_direction=aux_direction,
        )
    if kind == "mlp":
        return MLPForecaster(
            input_size, seq_len, hidden_size, num_layers, dropout,
            aux_direction=aux_direction,
        )
    raise ValueError(f"Unknown model kind: {kind}")


def model_has_direction_head(model: nn.Module) -> bool:
    """Whether a model was built with the auxiliary direction output."""
    return callable(getattr(model, "forward_full", None)) and bool(
        getattr(getattr(model, "head", None), "aux_direction", False)
    )


def gaussian_nll(
    mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    variance = torch.exp(log_var)
    loss = 0.5 * (log_var + (target - mean) ** 2 / variance)
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def binary_direction_loss(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Binary cross-entropy on the up/down direction of the same target.

    ``labels`` must be strictly 0.0/1.0; NaN rows are masked out rather than
    coerced, so flat bars that carry no direction information do not become free
    wins for whichever class the model defaults to.
    """
    valid = torch.isfinite(labels)
    if not bool(valid.any()):
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    picked_logits = logits[valid]
    picked_labels = labels[valid].to(logits.dtype)
    per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
        picked_logits, picked_labels, reduction="none"
    )
    if weights is not None:
        per_sample = per_sample * weights[valid].to(per_sample.dtype)
    return per_sample.mean()


@dataclass
class WindowDataset(Dataset):
    features: np.ndarray
    target: np.ndarray
    seq_len: int
    direction: np.ndarray | None = None

    def __len__(self) -> int:
        return max(0, len(self.features) - self.seq_len)

    def _label(self, index: int) -> torch.Tensor:
        row = index + self.seq_len
        label = torch.tensor(np.float32(self.target[row]), dtype=torch.float32)
        if self.direction is None:
            return label
        direction = torch.tensor(np.float32(self.direction[row]), dtype=torch.float32)
        return torch.stack([label, direction])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.features[index : index + self.seq_len]
        return torch.from_numpy(window), self._label(index)


def _forward_full(
    model: nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run a model and always return a three-tuple, with ``None`` when headless.

    The two-tuple ``forward`` contract is load-bearing across the codebase and
    across saved artifacts, so the auxiliary output is exposed through a separate
    method rather than by widening the tuple every caller has to unpack.
    """
    if model_has_direction_head(model):
        return model.forward_full(x)
    mean, log_var = model(x)
    return mean, log_var, None


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ts", "predicted_return", "predicted_volatility",
            "lower_68", "upper_68", "confidence", "member_disagreement",
        ]
    )


def _member_model_info(info: dict[str, Any]) -> dict[str, Any]:
    """Accept either a bare ``model_info`` or a full training result.

    ``train``/``train_ensemble`` hand back ``(model, training_result)`` pairs
    because callers want the metrics and history alongside. Inference and
    artifact serialization only ever need ``model_info``. Normalising here keeps
    the two shapes interchangeable, so a member cannot be silently half-fed: the
    old symptom was a ``KeyError: 'seq_len'`` from a full training result being
    read as if it were a scaler/architecture description.
    """
    nested = info.get("model_info")
    if isinstance(nested, dict) and "feature_columns" in nested:
        return nested
    return info


def _member_info_pair(info: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    return info, info.get("model_info") if isinstance(info, dict) else None


def _aggregate_member_metrics(
    member_metrics: Sequence[dict[str, Any]], reference: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Summarise ensemble members, and expose the seed-to-seed spread.

    The quality keys (``final_val_loss``, ``directional_accuracy``, ``mae``)
    become the **mean** across members, never the best member's number, because
    reporting the best of N on a weak signal is a selection bias dressed up as a
    result. The spread is reported alongside as ``std_<key>``: if members disagree
    by more than the margin the quality gate asks for, a single member's number
    is uninformative and the run should not be trusted however well the best one
    scored. Structural keys (window counts, parameter counts, best epoch) are
    copied from the primary member since they describe the run, not its quality.
    """
    if not member_metrics:
        return dict(reference or {})

    quality_keys = (
        "final_val_loss", "directional_accuracy", "mae", "val_direction_accuracy",
    )
    structural = {
        key: value
        for key, value in (reference or member_metrics[0]).items()
        if key not in quality_keys
    }

    summary: dict[str, Any] = dict(structural)
    for key in quality_keys:
        values = [float(m[key]) for m in member_metrics if m.get(key) is not None]
        if not values:
            continue
        array = np.asarray(values, dtype=float)
        summary[key] = float(array.mean())
        summary[f"std_{key}"] = float(array.std(ddof=1)) if array.size > 1 else 0.0

    summary["members_evaluated"] = len(member_metrics)
    summary["worst_directional_accuracy"] = min(
        float(m["directional_accuracy"]) for m in member_metrics
        if m.get("directional_accuracy") is not None
    )
    return summary


class ForecasterTrainer:
    """Trains a forecaster with chronological splits and early stopping."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.settings.debug else "cpu"
        )

    def _make_loader(
        self,
        features: np.ndarray,
        target: np.ndarray,
        seq_len: int,
        batch_size: int,
        shuffle: bool,
        direction: np.ndarray | None = None,
    ) -> DataLoader:
        dataset = WindowDataset(features, target, seq_len, direction)
        if len(dataset) == 0:
            raise ValueError("Not enough rows to build a single window; need more history")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            generator=torch.Generator().manual_seed(self.settings.seed) if shuffle else None,
        )

    def _build_scheduler(
        self, name: str, optimizer: torch.optim.Optimizer, epochs: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Pick a learning-rate schedule.

        ``plateau`` reacts to validation loss and is the historical default, so
        nothing about existing runs changes. ``cosine`` anneals on a fixed epoch
        budget and ``onecycle`` adds a warmup phase; both are more appropriate
        when the run has a known, fixed epoch count, which is exactly the sweep
        case. They are opt-in because a schedule change alters the optimisation
        trajectory and therefore the meaning of a checkpoint.
        """
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs),
                eta_min=self.settings.forecaster_lr * 0.01,
            )
        if name == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.settings.forecaster_lr,
                epochs=max(1, epochs),
                steps_per_epoch=1,
                pct_start=0.3,
            )
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3
            )
        raise ValueError(
            f"Unknown forecaster_lr_schedule {name!r}; expected plateau, cosine or onecycle"
        )

    def train(
        self,
        train_frame: pd.DataFrame,
        val_frame: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        target_column: str = "target_return",
        direction_column: str | None = None,
        aux_direction_weight: float = 0.0,
        seq_len: int | None = None,
        horizon: int = 1,
        kind: ModelKind | None = None,
        epochs: int | None = None,
        seed: int | None = None,
        lr_schedule: str | None = None,
        weight_decay: float | None = None,
    ) -> tuple[nn.Module, dict[str, Any]]:
        model, info = self._train_one(
            train_frame,
            val_frame,
            feature_columns,
            target_column=target_column,
            direction_column=direction_column,
            aux_direction_weight=aux_direction_weight,
            seq_len=seq_len,
            horizon=horizon,
            kind=kind,
            epochs=epochs,
            seed=seed,
            lr_schedule=lr_schedule,
            weight_decay=weight_decay,
        )
        return model, info

    def train_ensemble(
        self,
        train_frame: pd.DataFrame,
        val_frame: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        n_seeds: int = 1,
        base_seed: int | None = None,
        **train_kwargs: Any,
    ) -> tuple[list[tuple[nn.Module, dict[str, Any]]], dict[str, Any]]:
        """Train ``n_seeds`` independently initialised members on identical data.

        Seed averaging is the cheapest honest variance reduction available here.
        A single run reports one draw from a stochastic training process whose
        run-to-run spread on a weak signal is comparable to the edge being
        measured, so a single "best" run is partly a record of its seed. The
        mean prediction of several members cancels that nuisance variance, and
        the spread across members becomes a usable disagreement signal instead of
        an unrecorded nuisance.

        Members are averaged, **not** stacked, so nothing here needs a new
        inference path: :meth:`predict_ensemble` consumes the same list.
        """
        n_seeds = max(1, int(n_seeds))
        origin = int(self.settings.seed if base_seed is None else base_seed)

        members: list[tuple[nn.Module, dict[str, Any]]] = []
        member_metrics: list[dict[str, Any]] = []
        for offset in range(n_seeds):
            member_seed = origin + offset
            model, info = self._train_one(
                train_frame, val_frame, feature_columns, seed=member_seed, **train_kwargs
            )
            info["model_info"]["seed"] = member_seed
            members.append((model, info))
            member_metrics.append(info["metrics"])
            logger.info("ensemble_member_trained", extra={
                "member": offset + 1, "of": n_seeds, "seed": member_seed,
                "val_loss": round(float(info["metrics"].get("final_val_loss", 0.0)), 6),
                "val_directional_accuracy": round(
                    float(info["metrics"].get("directional_accuracy", 0.0)), 4
                ),
            })

        summary = _aggregate_member_metrics(member_metrics, member_metrics[0])
        model_info = dict(members[0][1]["model_info"])
        model_info["ensemble_size"] = n_seeds
        model_info["ensemble_seeds"] = [m[1]["model_info"]["seed"] for m in members]

        return members, {
            "metrics": summary,
            "model_info": model_info,
            "members": members,
            "ensemble": {
                "size": n_seeds,
                "seeds": model_info["ensemble_seeds"],
                "mean_val_loss": summary.get("final_val_loss"),
                "std_val_loss": summary.get("std_final_val_loss"),
                "mean_directional_accuracy": summary.get("directional_accuracy"),
                "std_directional_accuracy": summary.get("std_directional_accuracy"),
                "worst_directional_accuracy": summary.get("worst_directional_accuracy"),
                "val_loss_spread": (
                    max(m["final_val_loss"] for m in member_metrics)
                    - min(m["final_val_loss"] for m in member_metrics)
                ) if member_metrics else None,
            },
        }

    def _train_one(
        self,
        train_frame: pd.DataFrame,
        val_frame: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        target_column: str = "target_return",
        direction_column: str | None = None,
        aux_direction_weight: float = 0.0,
        seq_len: int | None = None,
        horizon: int = 1,
        kind: ModelKind | None = None,
        epochs: int | None = None,
        seed: int | None = None,
        lr_schedule: str | None = None,
        weight_decay: float | None = None,
    ) -> tuple[nn.Module, dict[str, Any]]:
        run_seed = int(self.settings.seed if seed is None else seed)
        set_seed(run_seed)
        seq_len = seq_len or self.settings.forecaster_seq_len
        kind = kind or self.settings.forecaster_kind
        epochs = epochs or self.settings.forecaster_epochs
        schedule_name = lr_schedule or self.settings.forecaster_lr_schedule
        decay = self.settings.forecaster_weight_decay if weight_decay is None else weight_decay
        use_direction = direction_column is not None and aux_direction_weight > 0.0

        if aux_direction_weight > 0.0 and direction_column is None:
            raise ValueError(
                "aux_direction_weight needs a direction_column to supervise the "
                "auxiliary head; pass one or set the weight to 0"
            )

        columns = list(feature_columns)
        train_matrix = train_frame[columns].to_numpy(dtype=np.float32)
        val_matrix = val_frame[columns].to_numpy(dtype=np.float32)
        train_target = train_frame[target_column].to_numpy(dtype=np.float32)
        val_target = val_frame[target_column].to_numpy(dtype=np.float32)

        train_direction = (
            train_frame[direction_column].to_numpy(dtype=np.float32)
            if use_direction else None
        )
        val_direction = (
            val_frame[direction_column].to_numpy(dtype=np.float32)
            if use_direction else None
        )

        scaler_mean = train_matrix.mean(axis=0)
        # float32 squares overflow past ~1.8e19, so compute the spread in float64
        # and then clip: a single unscaled outlier must not poison the model.
        scaler_std = train_matrix.astype(np.float64).std(axis=0)
        scaler_std[scaler_std < 1e-8] = 1.0
        scaler_std = np.clip(scaler_std, 1e-8, 1e12)
        scaler_mean = np.nan_to_num(scaler_mean, nan=0.0, posinf=0.0, neginf=0.0)
        scaler_std = np.nan_to_num(scaler_std, nan=1.0, posinf=1.0, neginf=1.0)

        train_scaled = ((train_matrix - scaler_mean) / scaler_std).astype(np.float32)
        val_scaled = ((val_matrix - scaler_mean) / scaler_std).astype(np.float32)

        train_loader = self._make_loader(
            train_scaled, train_target, seq_len, self.settings.forecaster_batch_size,
            shuffle=True, direction=train_direction,
        )
        val_loader = self._make_loader(
            val_scaled, val_target, seq_len, self.settings.forecaster_batch_size,
            shuffle=False, direction=val_direction,
        )

        model = build_model(
            kind, len(columns),
            self.settings.forecaster_hidden_size,
            self.settings.forecaster_num_layers,
            self.settings.forecaster_dropout,
            seq_len,
            aux_direction=use_direction,
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.settings.forecaster_lr,
                                      weight_decay=decay)
        scheduler = self._build_scheduler(schedule_name, optimizer, epochs)

        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        patience_counter = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                if batch_y.dim() > 1:
                    batch_target, batch_direction = batch_y[:, 0], batch_y[:, 1]
                else:
                    batch_target, batch_direction = batch_y, None
                optimizer.zero_grad()
                mean, log_var, logits = _forward_full(model, batch_x)
                loss = gaussian_nll(mean, log_var, batch_target)
                if logits is not None and batch_direction is not None:
                    loss = loss + float(aux_direction_weight) * binary_direction_loss(
                        logits, batch_direction
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            val_metrics = self._evaluate(model, val_loader)
            if schedule_name == "plateau":
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

            entry = {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_directional_accuracy": val_metrics["directional_accuracy"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if "val_direction_accuracy" in val_metrics:
                entry["val_direction_accuracy"] = val_metrics["val_direction_accuracy"]
            history.append(entry)

            if val_metrics["loss"] < best_loss - 1e-6:
                best_loss = val_metrics["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            logger.info("forecaster_epoch", extra={
                "kind": kind, **entry,
                "patience": patience_counter,
            })

            if patience_counter >= self.settings.forecaster_patience:
                logger.info("forecaster_early_stop", extra={"epoch": epoch, "best_epoch": best_epoch})
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        metrics = self._evaluate(model, val_loader)
        metrics.update({
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "final_val_loss": best_loss,
            "train_windows": len(train_loader.dataset),
            "val_windows": len(val_loader.dataset),
            "feature_count": len(columns),
            "parameters": sum(p.numel() for p in model.parameters()),
        })

        model_info = {
            "kind": kind,
            "input_size": len(columns),
            "hidden_size": self.settings.forecaster_hidden_size,
            "num_layers": self.settings.forecaster_num_layers,
            "dropout": self.settings.forecaster_dropout,
            "seq_len": seq_len,
            "horizon": horizon,
            "feature_columns": columns,
            "target_column": target_column,
            "direction_column": direction_column if use_direction else None,
            "aux_direction": use_direction,
            "aux_direction_weight": float(aux_direction_weight) if use_direction else 0.0,
            "seed": run_seed,
            "lr_schedule": schedule_name,
            "weight_decay": float(decay),
            "scaler_mean": scaler_mean.tolist(),
            "scaler_std": scaler_std.tolist(),
        }
        return model, {"metrics": metrics, "history": history, "model_info": model_info}

    def _evaluate(self, model: nn.Module, loader: DataLoader) -> dict[str, float]:
        model.eval()
        losses, abs_errors, directions, magnitudes = [], [], [], []
        head_hits, head_scored = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                if batch_y.dim() > 1:
                    batch_target, batch_direction = batch_y[:, 0], batch_y[:, 1]
                else:
                    batch_target, batch_direction = batch_y, None
                mean, log_var, logits = _forward_full(model, batch_x)
                losses.append(gaussian_nll(mean, log_var, batch_target).item())
                abs_errors.append((mean - batch_target).abs().mean().item())
                directions.append((torch.sign(mean) == torch.sign(batch_target)).float().mean().item())
                magnitudes.append(mean.abs().mean().item())
                if logits is not None and batch_direction is not None:
                    scored = torch.isfinite(batch_direction)
                    if bool(scored.any()):
                        predicted = (logits[scored] > 0).to(batch_direction.dtype)
                        head_hits += int((predicted == batch_direction[scored]).sum().item())
                        head_scored += int(scored.sum().item())

        report = {
            "loss": float(np.mean(losses)) if losses else float("inf"),
            "mae": float(np.mean(abs_errors)) if abs_errors else float("nan"),
            "directional_accuracy": float(np.mean(directions)) if directions else 0.5,
            "mean_predicted_magnitude": float(np.mean(magnitudes)) if magnitudes else 0.0,
        }
        if head_scored:
            report["val_direction_accuracy"] = head_hits / head_scored
            report["val_direction_scored"] = head_scored
        return report

    def predict(
        self,
        model: nn.Module,
        frame: pd.DataFrame,
        feature_columns: Sequence[str],
        model_info: dict[str, Any],
        *,
        seq_len: int | None = None,
    ) -> pd.DataFrame:
        """Single-model prediction. Kept as the canonical windowing path."""
        return self.predict_ensemble(
            [(model, model_info)], frame, feature_columns, seq_len=seq_len
        )

    def predict_ensemble(
        self,
        members: Sequence[tuple[nn.Module, dict[str, Any]]],
        frame: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        seq_len: int | None = None,
    ) -> pd.DataFrame:
        """Average several trained members and pool their predictive variance.

        Means are averaged arithmetically. Variances are pooled by the law of
        total variance, ``E[Var] + Var[E]``: the first term is the average
        member's own uncertainty, the second is the disagreement between
        members. Ignoring the second term would report intervals that get
        *tighter* as a sign of confidence precisely when the ensemble disagrees
        most, which is backwards — disagreement is the epistemic part of the
        uncertainty and belongs in the interval.
        """
        if not members:
            return _empty_predictions()

        member_infos = [_member_model_info(info) for _model, info in members]
        primary_info = member_infos[0]
        seq = seq_len or int(primary_info.get("seq_len") or self.settings.forecaster_seq_len)
        if len(frame) <= seq:
            return _empty_predictions()

        reference = list(primary_info["feature_columns"])
        for info in member_infos[1:]:
            if list(info["feature_columns"]) != reference:
                logger.warning(
                    "ensemble_feature_schema_mismatch",
                    extra={"expected": len(reference), "got": len(info["feature_columns"])},
                )

        scaled = self._scale_frame(frame, reference, primary_info)
        windowed = self._windows(scaled, seq)

        member_means: list[np.ndarray] = []
        member_vars: list[np.ndarray] = []
        for model, _info in members:
            mean_out, var_out = self._forward_arrays(model, windowed)
            member_means.append(mean_out)
            member_vars.append(var_out)

        stacked = np.vstack(member_means)
        pred = stacked.mean(axis=0)
        within = np.vstack(member_vars).mean(axis=0)
        # A single member has no between-member spread to measure, so total
        # variance collapses to the mean within-member variance.
        between = stacked.var(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(pred)
        var = np.maximum(within + between, 1e-12)

        out = pd.DataFrame({"ts": frame["ts"].iloc[seq:].to_numpy()})
        out["predicted_return"] = pred
        out["predicted_volatility"] = np.sqrt(var)
        out["lower_68"] = pred - 0.9945 * np.sqrt(var)
        out["upper_68"] = pred + 0.9945 * np.sqrt(var)
        out["confidence"] = 1.0 / (1.0 + np.sqrt(var) * 100)
        if stacked.shape[0] > 1:
            out["member_disagreement"] = np.sqrt(between)
        return out.reset_index(drop=True)

    def _scale_frame(
        self, frame: pd.DataFrame, feature_columns: Sequence[str], model_info: dict[str, Any]
    ) -> np.ndarray:
        columns = [c for c in feature_columns if c in frame.columns]
        if len(columns) != len(feature_columns):
            missing = set(feature_columns) - set(columns)
            logger.warning("predict_missing_features_zeroed", extra={"missing": sorted(missing)[:10]})

        matrix = frame.reindex(columns=list(feature_columns)).to_numpy(dtype=np.float32)
        matrix = np.nan_to_num(matrix, nan=0.0)
        mean = np.asarray(model_info["scaler_mean"], dtype=np.float32)
        std = np.asarray(model_info["scaler_std"], dtype=np.float32)
        return (matrix - mean) / std

    @staticmethod
    def _windows(scaled: np.ndarray, seq: int) -> np.ndarray:
        # sliding_window_view appends the window axis at the end, giving
        # (n - seq + 1, n_features, seq). Training feeds (batch, seq, features),
        # so the window axis has to move back into the time position.
        windowed = np.lib.stride_tricks.sliding_window_view(scaled, seq, axis=0)
        windowed = np.ascontiguousarray(windowed.transpose(0, 2, 1))
        # Training exposes windows [i, i+seq) for i in [0, N-seq), so the extra
        # trailing window produced by sliding_window_view is never used at
        # inference either. Keeping it would emit one more prediction than there
        # are scorable rows and shift every timestamp by one bar.
        return windowed[: len(scaled) - seq]

    def _forward_arrays(
        self, model: nn.Module, windowed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        means, variances = [], []
        with torch.no_grad():
            for start in range(0, len(windowed), 256):
                chunk = torch.from_numpy(windowed[start : start + 256]).to(self.device)
                mean_out, log_var, _logits = _forward_full(model, chunk)
                means.append(mean_out.cpu().numpy())
                variances.append(torch.exp(log_var).cpu().numpy())
        if not means:
            empty = np.array([], dtype=np.float64)
            return empty, empty
        return (
            np.concatenate(means).astype(np.float64),
            np.concatenate(variances).astype(np.float64),
        )


class ForecasterArtifact:
    """Serializes a trained model with its normalization state and schema.

    A seed ensemble is stored as a list of members, each with its own scaler and
    architecture flag, because members genuinely differ and re-deriving them from
    one shared scaler would silently change their predictions. The first member
    is additionally mirrored at the top level so any older reader — including
    :meth:`load` itself — keeps working on an ensemble payload.
    """

    def __init__(
        self,
        model_id: str,
        model: nn.Module,
        model_info: dict[str, Any],
        metrics: dict[str, Any],
        symbol: str,
        interval: str,
        members: Sequence[tuple[nn.Module, dict[str, Any]]] | None = None,
    ) -> None:
        self.model_id = model_id
        self.model = model
        self.model_info = model_info
        self.metrics = metrics
        self.symbol = symbol
        self.interval = interval
        # Store only the model_info per member. Writing the whole training result
        # here would bloat the artifact with metrics/history and, on reload, hand
        # inference a dict that has no `scaler_mean`.
        self.members = (
            [(m, _member_model_info(info)) for m, info in members]
            if members
            else [(model, model_info)]
        )

    def save(self, directory: Path, trainer: ForecasterTrainer | None = None) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / f"{self.model_id}.pt"
        primary_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        torch.save(
            {
                "model_id": self.model_id,
                "state_dict": primary_state,
                "model_info": self.model_info,
                "members": [
                    {
                        "state_dict": {k: v.cpu() for k, v in member.state_dict().items()},
                        "model_info": info,
                    }
                    for member, info in self.members
                ],
                "metrics": self.metrics,
                "symbol": self.symbol,
                "interval": self.interval,
                "torch_version": torch.__version__,
                "created_at": pd.Timestamp.utcnow().isoformat(),
            },
            model_path,
        )
        (directory / f"{self.model_id}.json").write_text(
            json.dumps(
                {
                    "model_id": self.model_id,
                    "symbol": self.symbol,
                    "interval": self.interval,
                    "metrics": self.metrics,
                    "model_info": {
                        k: v for k, v in self.model_info.items()
                        if k not in {"scaler_mean", "scaler_std"}
                    },
                    "scaler_mean_len": len(self.model_info.get("scaler_mean", [])),
                    "ensemble_size": len(self.members),
                },
                indent=2, default=str,
            ),
            encoding="utf-8",
        )
        return model_path

    @staticmethod
    def load(path: Path, device: torch.device | None = None) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
        """Load the primary (first) member.

        On an ensemble payload this returns **one** member, not the ensemble
        average. Callers that want the ensemble must use :meth:`load_ensemble`;
        silently averaging here would be indistinguishable from a
        single-member load and would misreport which model produced a metric.
        """
        payload, members = ForecasterArtifact._load_payload(path, device)
        model, info = members[0]
        return model, info, payload

    @staticmethod
    def load_ensemble(
        path: Path, device: torch.device | None = None
    ) -> tuple[list[tuple[nn.Module, dict[str, Any]]], dict[str, Any]]:
        """Load every member, so predictions can be averaged as an ensemble."""
        payload, members = ForecasterArtifact._load_payload(path, device)
        return members, payload

    @staticmethod
    def _load_payload(
        path: Path, device: torch.device | None = None
    ) -> tuple[dict[str, Any], list[tuple[nn.Module, dict[str, Any]]]]:
        payload = torch.load(path, map_location=device or torch.device("cpu"), weights_only=False)
        raw_members = payload.get("members")
        if raw_members:
            members = [
                (_rebuild_model(entry["model_info"], entry["state_dict"]), entry["model_info"])
                for entry in raw_members
            ]
        else:
            info = payload["model_info"]
            members = [(_rebuild_model(info, payload["state_dict"]), info)]
        return payload, members


def _rebuild_model(info: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> nn.Module:
    model = build_model(
        info["kind"], info["input_size"], info["hidden_size"],
        info["num_layers"], info["dropout"], info["seq_len"],
        aux_direction=bool(info.get("aux_direction", False)),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def new_model_id(symbol: str, kind: str, horizon: int = 1) -> str:
    return f"{symbol.replace('_', '-').lower()}-{kind}-h{horizon}-{uuid.uuid4().hex[:8]}"


__all__ = [
    "ForecasterArtifact",
    "ForecasterTrainer",
    "LSTMForecaster",
    "MLPForecaster",
    "TransformerForecaster",
    "WindowDataset",
    "binary_direction_loss",
    "build_model",
    "gaussian_nll",
    "model_has_direction_head",
    "new_model_id",
    "set_seed",
]