"""The decision agent: an LLM that plans with tools, but never touches the broker.

Protocol (works with any chat endpoint, no native function-calling needed). Each turn the model
replies with ONE JSON object:
    {"tool": "<name>", "args": {...}}    -> we run the tool and feed the result back
    {"final": {...}}                     -> the decision
Two jobs: pick a strategy (backtesting candidates first) and review an entry signal (read news first).
Principle: the agent can only say NO to risk-taking or shrink a position. It never enlarges one and
never blocks an exit.
"""
from __future__ import annotations

import json
from typing import Optional

from .llm import LLMError
from .strategies import STRATEGIES
from .tools import ToolBox


class AgentUnavailable(Exception):
    pass


def extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of model text (handles ```json fences and chatter)."""
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def _short(obj, n: int = 160) -> str:
    s = json.dumps(obj, default=str)
    return s if len(s) <= n else s[: n - 3] + "..."


SYSTEM = """You are the decision agent inside a PAPER-trading system (simulated money, mock data).
You NEVER place orders. Deterministic risk code sizes and approves every trade; you analyse, propose and explain.
Reply with exactly ONE JSON object per turn and nothing else:
  {{"tool": "<name>", "args": {{...}}}}   to call a tool
  {{"final": {{...}}}}                    when you are done
Tools:
{tools}

TASK:{task}
{instructions}"""

SELECT_INSTR = """Pick the strategy to run for the next few weeks.
1. Call get_market_overview, then get_memory, then run_backtest for each candidate strategy (not risk_off).
2. Judge each candidate on RISK-ADJUSTED quality: positive total return, Sharpe, and a max drawdown clearly smaller than buy-and-hold's. This portfolio deliberately keeps at least 20% cash and caps risk per trade, so in a rising market it will often trail buy-and-hold on raw return. That alone is NOT a reason to reject a strategy that made money with a better Sharpe and a shallower drawdown.
3. Distrust strategies with poor realised results in memory.
4. Choose "risk_off" (stay in cash) only if no candidate has a positive return and a meaningfully positive Sharpe, or if the market looks like a high-volatility downtrend.
Final format: {"final": {"strategy": "<name>", "reason": "<one or two plain sentences citing the numbers>"}}"""

REVIEW_INSTR = """A strategy wants to ENTER a position (see SIGNAL_JSON). Check recent news for the symbol (get_news) and anything else useful.
Veto if news or risk context clearly argues against it. Otherwise confirm. conviction is 0..1 and can only SHRINK the position size.
Final format: {"final": {"decision": "confirm" or "veto", "conviction": 0.0-1.0, "rationale": "<one or two plain sentences>"}}"""


class Agent:
    def __init__(self, llm, max_steps: int = 10, fail_limit: int = 3):
        self.llm = llm
        self.name = getattr(llm, "name", "llm")
        self.max_steps, self.fail_limit = max_steps, fail_limit
        self.fail_streak, self.offline, self.last_error = 0, False, ""

    def reset(self) -> None:
        self.fail_streak, self.offline, self.last_error = 0, False, ""

    def _fail(self, msg: str) -> None:
        self.last_error = msg
        self.fail_streak += 1
        if self.fail_streak >= self.fail_limit:
            self.offline = True  # circuit breaker: stop waiting on a dead model

    def _run(self, task: str, instructions: str, user: str, toolbox: ToolBox, required: tuple = ()):
        """`required` lists tool calls ("name" or "name:strategy_or_symbol") the model MUST make before
        its final answer is accepted. Models sometimes skip the tool and claim "no news"; we don't allow that."""
        if self.offline:
            raise AgentUnavailable(f"LLM circuit-breaker open ({self.last_error})")
        system = SYSTEM.format(tools=toolbox.describe(), task=task, instructions=instructions)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        steps, bad, nudges, called = [], 0, 0, set()
        for turn in range(self.max_steps + 1):
            if turn == self.max_steps:
                messages.append({"role": "user", "content": 'Step limit reached. Reply NOW with your {"final": ...} JSON.'})
            try:
                text = self.llm.chat(messages)
            except LLMError as e:
                self._fail(str(e))
                raise AgentUnavailable(str(e))
            obj = extract_json(text)
            if obj and isinstance(obj.get("final"), dict):
                missing = [r for r in required if r not in called]
                if missing and nudges < 2 and turn < self.max_steps:
                    nudges += 1
                    messages += [{"role": "assistant", "content": text},
                                 {"role": "user", "content": f"Rejected: you must call these tools before giving a final answer: {', '.join(missing)}. "
                                                              'Reply with a {"tool": ...} JSON request now.'}]
                    continue
                if missing:
                    self._fail(f"model skipped required tools: {', '.join(missing)}")
                    raise AgentUnavailable(f"model skipped required tools: {', '.join(missing)}")
                self.fail_streak = 0
                return obj["final"], steps
            if obj and "tool" in obj and turn < self.max_steps:
                name, args = str(obj["tool"]), obj.get("args") or {}
                if isinstance(args, dict):
                    called.add(name)
                    for k in ("strategy", "symbol"):
                        if k in args:
                            called.add(f"{name}:{args[k]}")
                result = toolbox.call(name, args if isinstance(args, dict) else {})
                steps.append(f"{name}({_short(args, 70)}) -> {_short(result, 110)}")
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": f"TOOL_RESULT {name}: {json.dumps(result, default=str)}"}]
                continue
            bad += 1
            if bad >= 2:
                self._fail("model did not return valid JSON")
                raise AgentUnavailable("model did not return valid JSON")
            messages += [{"role": "assistant", "content": text or "(empty)"},
                         {"role": "user", "content": 'Invalid. Reply with exactly one JSON object: {"tool": ...} or {"final": ...}.'}]
        self._fail("no final answer within step limit")
        raise AgentUnavailable("no final answer within step limit")

    # ------------------------------------------------------------------ jobs
    def select_strategy(self, session) -> dict:
        user = json.dumps({"date": session.date, "current_strategy": session.strategy,
                           "candidates": [k for k in STRATEGIES if k != "risk_off"]})
        required = tuple(f"run_backtest:{k}" for k in STRATEGIES if k != "risk_off")
        final, steps = self._run("SELECT_STRATEGY", SELECT_INSTR, user, ToolBox(session), required)
        name = str(final.get("strategy", ""))
        if name not in STRATEGIES:
            self._fail(f"model chose unknown strategy '{name}'")
            raise AgentUnavailable(f"model chose unknown strategy '{name}'")
        reason = str(final.get("reason", "")).strip() or "no reason given"
        return {"strategy": name, "reason": reason, "source": self.name,
                "trace": {"source": self.name, "steps": steps, "final": f"{name}: {reason}"}}

    def review_signal(self, session, symbol: str, sig) -> dict:
        signal = {"symbol": symbol, "side": "buy", "strategy": session.strategy, "reason": sig.reason,
                  "price": round(float(session.last_closes[symbol]), 2), "date": session.date}
        user = "SIGNAL_JSON: " + json.dumps(signal)
        final, steps = self._run("REVIEW_SIGNAL", REVIEW_INSTR, user, ToolBox(session), (f"get_news:{symbol}",))
        decision = str(final.get("decision", "")).lower()
        if decision not in ("confirm", "veto"):
            self._fail(f"bad decision '{decision}'")
            raise AgentUnavailable(f"bad decision '{decision}'")
        try:
            conviction = min(max(float(final.get("conviction", 1.0)), 0.0), 1.0)
        except (TypeError, ValueError):
            conviction = 1.0
        rationale = str(final.get("rationale", "")).strip() or "no rationale given"
        return {"decision": decision, "conviction": conviction, "rationale": rationale, "source": self.name,
                "trace": {"source": self.name, "steps": steps, "final": f"{decision} (conviction {conviction:.2f}): {rationale}"}}
