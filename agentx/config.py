"""Settings. Secrets come from environment variables or a local .env file, never from code."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Hard-coded on purpose. There is no live-broker adapter anywhere in this project.
LIVE_TRADING_ENABLED = False


def load_env(path: str = ".env") -> None:
    """Tiny .env reader (no dependency). Existing environment variables win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        load_env()
        return cls(
            base_url=os.environ.get("AGENTX_LLM_BASE_URL", "http://localhost:8081/v1").rstrip("/"),
            api_key=os.environ.get("AGENTX_LLM_API_KEY", "").strip().strip('"').strip("'").strip(),  # tolerate pasted spaces/quotes
            model=os.environ.get("AGENTX_LLM_MODEL", "gemini-3.7-flash"),
            timeout=float(os.environ.get("AGENTX_LLM_TIMEOUT", "60")),
        )
