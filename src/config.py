"""Shared environment loading for local and deployed runs."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()


def require_env(name: str) -> str:
    """Fail early with a useful setup message because missing service keys otherwise surface as noisy tracebacks."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable {name}. Add it to {PROJECT_ROOT / '.env'} or your host dashboard.")
    return value
