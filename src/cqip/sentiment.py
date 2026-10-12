"""News sentiment scoring using a fine-tuned transformer with lexicon fallback."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import pandas as pd

from cqip.config import Settings, get_settings
from cqip.logging_config import get_logger

logger = get_logger(__name__)

CRYPTO_LEXICON: dict[str, float] = {
    "adoption": 0.6, "approved": 0.9, "approval": 0.8, "approve": 0.8, "surge": 0.7,
    "surges": 0.7, "soar": 0.9, "soars": 0.9, "rally": 0.7, "rallies": 0.7,
    "bullish": 0.8, "bull": 0.6, "breakout": 0.8, "record": 0.5, "high": 0.4,
    "gain": 0.5, "gains": 0.5, "profit": 0.6, "profits": 0.6, "upgrade": 0.6,
    "partnership": 0.6, "inflow": 0.6, "inflows": 0.6, "institutional": 0.4,
    "launch": 0.4, "launched": 0.4, "support": 0.4, "boost": 0.6, "growth": 0.5,
    "milestone": 0.5, "etf": 0.3, "blackrock": 0.3, "inclusion": 0.5,
    "crash": -0.9, "plunge": -0.9, "plunges": -0.9, "tumble": -0.7,
    "tumbles": -0.7, "bearish": -0.8, "bear": -0.6, "hack": -1.0, "hacked": -1.0,
    "exploit": -1.0, "breach": -0.9, "stolen": -0.9, "theft": -0.9, "fraud": -1.0,
    "scam": -1.0, "lawsuit": -0.6, "sue": -0.6, "sued": -0.6, "ban": -0.7,
    "banned": -0.7, "crackdown": -0.7, "warning": -0.5, "risk": -0.4,
    "decline": -0.5, "falls": -0.5, "fell": -0.5, "drop": -0.4, "loss": -0.6,
    "losses": -0.6, "sell-off": -0.7, "selloff": -0.7, "liquidation": -0.7,
    "liquidated": -0.7, "bankruptcy": -1.0, "insolvency": -0.9, "downgrade": -0.7,
    "reversal": -0.3, "correction": -0.4, "investigation": -0.6, "fine": -0.6,
    "penalty": -0.6, "delay": -0.5, "delayed": -0.5, "outage": -0.7, "halt": -0.6,
    "rejected": -0.7, "reject": -0.6, "concern": -0.4, "fear": -0.6, "uncertainty": -0.5,
}

_TOKEN_RE = re.compile(r"[a-z\-']+")

_NEGATIONS = {"not", "no", "never", "without", "denies", "denied", "against"}


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def lexicon_sentiment(text: str) -> tuple[float, float]:
    """Deterministic sentiment in [-1, 1] plus a confidence weight in [0, 1]."""
    tokens = _tokens(text)
    if not tokens:
        return 0.0, 0.0
    total = 0.0
    hits = 0
    for index, token in enumerate(tokens):
        score = CRYPTO_LEXICON.get(token)
        if score is None:
            continue
        if index > 0 and tokens[index - 1] in _NEGATIONS:
            score = -score
        total += score
        hits += 1
    if hits == 0:
        return 0.0, 0.0
    raw = total / hits
    normalized = math.tanh(raw * 1.8)
    coverage = min(1.0, hits / max(6.0, len(tokens) * 0.35))
    return normalized, coverage


class SentimentScorer:
    """Transformer sentiment scorer with graceful degradation.

    If the model cannot be downloaded (offline, no cache, or disabled via
    ``CQIP_ALLOW_MODEL_DOWNLOAD``) the scorer falls back to the domain lexicon
    so the pipeline keeps working.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._pipeline: Any = None
        self._backend = "lexicon"
        self._load_attempted = False

    @property
    def backend(self) -> str:
        return self._backend

    def _load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if not self.settings.allow_model_download:
            self._backend = "lexicon"
            return
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.settings.sentiment_model,
                truncation=True,
                max_length=self.settings.sentiment_max_length,
                device=-1,
            )
            probe = self._pipeline(["Bitcoin surges past key resistance"])
            if not probe or "label" not in probe[0]:
                raise RuntimeError("unexpected pipeline output shape")
            self._backend = "transformer"
            logger.info("sentiment_model_loaded", extra={"model": self.settings.sentiment_model})
        except Exception as exc:  # noqa: BLE001
            self._backend = "lexicon"
            logger.warning(
                "sentiment_model_unavailable_using_lexicon",
                extra={"model": self.settings.sentiment_model, "error": str(exc)[:200]},
            )

    def score_texts(self, texts: Sequence[str], batch_size: int | None = None) -> list[dict[str, float]]:
        if not texts:
            return []
        self._load()
        batch = batch_size or self.settings.sentiment_batch_size

        if self._backend == "transformer" and self._pipeline is not None:
            try:
                return self._score_transformer(list(texts), batch)
            except Exception as exc:  # noqa: BLE001
                logger.warning("sentiment_inference_failed_falling_back", extra={"error": str(exc)[:200]})
                self._backend = "lexicon"

        return [self._score_lexicon(text) for text in texts]

    def _score_transformer(self, texts: list[str], batch_size: int) -> list[dict[str, float]]:
        results: list[dict[str, float]] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            outputs = self._pipeline(chunk, batch_size=batch_size, truncation=True)
            for item in outputs:
                label = str(item["label"]).lower()
                raw = float(item["score"])
                polarity = raw if "positive" in label else -raw
                results.append({
                    "sentiment_score": polarity,
                    "sentiment_confidence": raw,
                    "is_positive": 1.0 if "positive" in label else 0.0,
                    "is_negative": 1.0 if "negative" in label else 0.0,
                    "is_neutral": 1.0 if "neutral" in label else 0.0,
                })
        return results

    def _score_lexicon(self, text: str) -> dict[str, float]:
        score, coverage = lexicon_sentiment(text)
        return {
            "sentiment_score": score,
            "sentiment_confidence": coverage,
            "is_positive": 1.0 if score > 0.1 else 0.0,
            "is_negative": 1.0 if score < -0.1 else 0.0,
            "is_neutral": 1.0 if abs(score) <= 0.1 else 0.0,
        }

    def score_frame(
        self, news: pd.DataFrame, text_columns: Sequence[str] = ("title", "summary")
    ) -> pd.DataFrame:
        """Attach sentiment columns to an articles frame."""
        if news.empty:
            return news.assign(
                sentiment_score=pd.Series(dtype=float),
                sentiment_confidence=pd.Series(dtype=float),
                is_positive=pd.Series(dtype=float),
                is_negative=pd.Series(dtype=float),
                is_neutral=pd.Series(dtype=float),
            )

        frame = news.copy()
        parts = [frame[column].fillna("").astype(str) for column in text_columns if column in frame.columns]
        if not parts:
            raise KeyError("No text columns available for sentiment scoring")
        combined = parts[0]
        for extra in parts[1:]:
            combined = combined + ". " + extra

        scores = self.score_texts(combined.tolist())
        for key in ("sentiment_score", "sentiment_confidence", "is_positive", "is_negative", "is_neutral"):
            frame[key] = [entry[key] for entry in scores]
        return frame


def aggregate_sentiment_by_window(news: pd.DataFrame, window: str = "1h") -> pd.Series:
    if news.empty:
        return pd.Series(dtype=float)
    frame = news.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame.set_index("ts")["sentiment_score"].resample(window).mean()


__all__ = ["SentimentScorer", "aggregate_sentiment_by_window", "lexicon_sentiment"]