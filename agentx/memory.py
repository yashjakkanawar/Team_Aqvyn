"""Persistent memory: realised results per strategy, so the agent learns across sessions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class Memory:
    def __init__(self, path: Optional[str] = None, keep: int = 200):
        self.path = Path(path) if path else None
        self.keep = keep
        self.trades: list = []
        if self.path and self.path.exists():
            try:
                self.trades = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.trades = []

    def record(self, strategy: str, symbol: str, pnl_pct: float, date: str) -> None:
        self.trades.append({"strategy": strategy, "symbol": symbol, "pnl_pct": round(float(pnl_pct), 5), "date": date})
        self.trades = self.trades[-self.keep:]
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.trades), encoding="utf-8")

    def summary(self) -> dict:
        out: dict = {}
        for t in self.trades:
            s = out.setdefault(t["strategy"], {"trades": 0, "wins": 0, "sum": 0.0})
            s["trades"] += 1
            s["wins"] += t["pnl_pct"] > 0
            s["sum"] += t["pnl_pct"]
        return {k: {"trades": v["trades"], "win_rate_pct": round(100 * v["wins"] / v["trades"], 1),
                    "avg_pnl_pct": round(100 * v["sum"] / v["trades"], 2)} for k, v in out.items()}
