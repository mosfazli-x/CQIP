"""Package initialization and public surface."""

from __future__ import annotations

__version__ = "0.1.0"

from cqip.config import Settings, get_settings

__all__ = ["Settings", "__version__", "get_settings"]