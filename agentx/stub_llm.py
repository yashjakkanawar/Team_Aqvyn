"""Deterministic stand-in for a real LLM. Speaks the same JSON tool protocol as the real agent.

Used for offline development, tests and demos with no API key. It is rule-based, so it is NOT a
substitute for a real model when judging how smart the agent is; it proves the plumbing works.
"""
from __future__ import annotations

import json
import re

from .strategies import STRATEGIES

CANDIDATES = [k for k in STRATEGIES if k != "risk_off"]


def _results(messages) -> list:
    out = []
    for m in messages:
        if m["role"] == "user" and m["content"].startswith("TOOL_RESULT "):
            head, _, body = m["content"].partition(": ")
            try:
                out.append((head.split(" ", 1)[1], json.loads(body)))
            except json.JSONDecodeError:
                pass
    return out


def _call(name: str, **args) -> str:
    return json.dumps({"tool": name, "args": args})


class StubLLM:
    name = "stub"

    def chat(self, messages: list) -> str:
        system = messages[0]["content"]
        if "TASK:SELECT_STRATEGY" in system:
            return self._select(messages)
        if "TASK:REVIEW_SIGNAL" in system:
            return self._review(messages)
        return json.dumps({"final": {}})

    def _select(self, messages) -> str:
        res = _results(messages)
        names = [n for n, _ in res]
        if "get_market_overview" not in names:
            return _call("get_market_overview")
        if "get_memory" not in names:
            return _call("get_memory")
        done = {r["strategy"]: r for n, r in res if n == "run_backtest" and "strategy" in r}
        for s in CANDIDATES:
            if s not in done:
                return _call("run_backtest", strategy=s, lookback_days=120)
        memory = next((r for n, r in res if n == "get_memory"), {}).get("by_strategy", {})
        overview = next((r for n, r in res if n == "get_market_overview"), {})
        scored = {}
        for s, r in done.items():
            score = r["sharpe"] + 0.02 * r["excess_return_pct"]
            mem = memory.get(s)
            if mem and mem["trades"] >= 3 and mem["avg_pnl_pct"] < 0:
                score -= 0.5
            scored[s] = score
        best = max(scored, key=scored.get)
        b = done[best]
        regime = overview.get("regime_guess", "unknown")
        if b["total_return_pct"] <= 0:
            return json.dumps({"final": {"strategy": "risk_off", "reason":
                f"Regime looks like {regime} and no strategy made money in the last {b['days_tested']} days (best: {best} {b['total_return_pct']:+.1f}%). Staying in cash."}})
        return json.dumps({"final": {"strategy": best, "reason":
            f"Regime looks like {regime}. {best} returned {b['total_return_pct']:+.1f}% vs buy-and-hold {b['buy_and_hold_return_pct']:+.1f}% "
            f"(Sharpe {b['sharpe']:.2f}, max drawdown {b['max_drawdown_pct']:.1f}%) in the trailing backtest, the best risk-adjusted of the candidates."}})

    def _review(self, messages) -> str:
        user = messages[1]["content"]
        m = re.search(r"SIGNAL_JSON: (\{.*\})", user)
        sig = json.loads(m.group(1)) if m else {"symbol": "?"}
        res = _results(messages)
        news = next((r for n, r in res if n == "get_news"), None)
        if news is None:
            return _call("get_news", symbol=sig["symbol"], days=5)
        net = news.get("net_sentiment", 0.0)
        heads = "; ".join(i["headline"] for i in news.get("items", [])[:2])
        if net <= -0.5:
            return json.dumps({"final": {"decision": "veto", "conviction": 0.0,
                "rationale": f"Recent news is negative (net sentiment {net:+.1f}: {heads}). Not buying into that."}})
        conviction = 0.6 if net == 0 else min(1.0, 0.7 + 0.15 * net)
        why = f"Recent news supports it (net sentiment {net:+.1f}: {heads})." if net > 0 else "No news either way, so a smaller position."
        return json.dumps({"final": {"decision": "confirm", "conviction": round(conviction, 2), "rationale": why}})
