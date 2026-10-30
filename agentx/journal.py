"""Plain-language trade journal: one entry per executed trade, plus 'why I did NOT trade' entries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


class Journal:
    def __init__(self, md_path: Optional[str] = None, jsonl_path: Optional[str] = None):
        self.entries: list = []
        self.md_path = Path(md_path) if md_path else None
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        for p in (self.md_path, self.jsonl_path):
            if p:
                p.parent.mkdir(parents=True, exist_ok=True)
        if self.md_path:
            self.md_path.write_text("# Trade journal (paper trading, synthetic data)\n\n", encoding="utf-8")
        if self.jsonl_path:
            self.jsonl_path.write_text("", encoding="utf-8")

    def add(self, kind: str, date: str, symbol: str, text: str, **data) -> dict:
        entry = {"kind": kind, "date": date, "symbol": symbol, "text": text, **data}
        self.entries.append(entry)
        if self.md_path:
            with self.md_path.open("a", encoding="utf-8") as fh:
                tag = f"{kind.upper()} {symbol}".strip()
                fh.write(f"- **{date} - {tag}**: {text}\n")
        if self.jsonl_path:
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=float) + "\n")
        return entry

    def latest(self, n: int = 30) -> list:
        return list(reversed(self.entries[-n:]))
