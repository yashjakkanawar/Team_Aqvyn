"""Check that your LLM gateway works with this project. Run it on the machine that hosts the gateway:

    python -m agentx.check_llm

It reads AGENTX_LLM_BASE_URL, AGENTX_LLM_API_KEY and AGENTX_LLM_MODEL from .env, lists models, asks for one JSON reply,
then runs one real strategy-selection and one signal-review through the agent.
"""
from __future__ import annotations

import sys

from .agent import Agent, AgentUnavailable, extract_json
from .config import LLMConfig
from .data import synthetic_market
from .llm import HttpLLM, LLMError
from .session import TradingSession
from .strategies import Signal


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = LLMConfig.from_env()
    print(f"gateway: {cfg.base_url} | model: {cfg.model} | key set: {'yes' if cfg.api_key else 'NO'}"
          f" (length {len(cfg.api_key)}, has spaces: {'yes' if ' ' in cfg.api_key else 'no'})")
    llm = HttpLLM(cfg)

    try:
        models = llm.list_models()
        print(f"[1/4] /models OK: {', '.join(models[:8])}{' ...' if len(models) > 8 else ''}")
        if cfg.model not in models and models:
            print(f"      note: '{cfg.model}' is not in that list. Set AGENTX_LLM_MODEL in .env to one of them if calls fail.")
    except LLMError as e:
        print(f"[1/4] /models failed (some gateways don't offer it, continuing): {e}")

    try:
        text = llm.chat([{"role": "user", "content": 'Reply with exactly this JSON and nothing else: {"final": {"ok": true}}'}])
        obj = extract_json(text)
        print(f"[2/4] chat OK, reply: {text.strip()[:100]!r}, parsed JSON: {'yes' if obj else 'NO (model ignores JSON instructions)'}")
    except LLMError as e:
        print(f"[2/4] chat FAILED: {e}")
        return 1

    market = synthetic_market(days=200, seed=7)
    s = TradingSession(market, agent=Agent(llm), start_idx=60)
    s.run(until=60)
    s.last_closes = {sym: float(market.close[sym][60]) for sym in market.symbols}
    agent = Agent(llm)
    try:
        r = agent.select_strategy(s)
        print(f"[3/4] strategy selection OK -> {r['strategy']}: {r['reason']}")
        for st in r["trace"]["steps"]:
            print("      tool:", st[:150])
    except AgentUnavailable as e:
        print(f"[3/4] strategy selection FAILED: {e}")
        return 1
    try:
        r = agent.review_signal(s, market.symbols[0], Signal(enter=True, stop=1.0, reason="test signal"))
        print(f"[4/4] signal review OK -> {r['decision']} (conviction {r['conviction']}): {r['rationale']}")
    except AgentUnavailable as e:
        print(f"[4/4] signal review FAILED: {e}")
        return 1
    print("\nAll good: run  python -m agentx.run_demo --llm http")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
