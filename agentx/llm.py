"""OpenAI-compatible chat client (stdlib only). Works with any /v1/chat/completions gateway."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .config import LLMConfig


class LLMError(Exception):
    pass


class HttpLLM:
    name = "llm"

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        host = urlparse(cfg.base_url).hostname or ""
        # Don't send localhost traffic through a system proxy.
        handlers = [urllib.request.ProxyHandler({})] if host in ("localhost", "127.0.0.1") else []
        self._opener = urllib.request.build_opener(*handlers)

    def _request(self, path: str, body: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.cfg.base_url + path, data=data, headers=headers,
                                     method="POST" if body is not None else "GET")
        try:
            with self._opener.open(req, timeout=self.cfg.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LLMError(f"HTTP {e.code} from {path}: {e.read().decode('utf-8', 'replace')[:200]}") from e
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            raise LLMError(f"cannot reach {self.cfg.base_url}{path}: {e}") from e

    def chat(self, messages: list) -> str:
        data = self._request("/chat/completions", {"model": self.cfg.model, "messages": messages, "temperature": 0.2})
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {str(data)[:200]}") from e

    def list_models(self) -> list:
        data = self._request("/models")
        return [m.get("id", "?") for m in data.get("data", [])]
