# AgentX Paper Trader

## 🚀 Live Demo

| | Link |
|---|---|
| 🌐 **Live Dashboard** | [agentx-paper-trader.vercel.app](https://agentx-paper-trader.vercel.app) |
| 🐍 **Backend API** | [team-aqvyn.onrender.com](https://team-aqvyn.onrender.com) |

> ⚠️ The backend runs on Render's free tier — it may take ~30 seconds to wake up on the first visit.

---

> **Status: idea-stage prototype for the AgentX Hackathon 2026, PS-02 (agentic paper-trading bot).**
> Built before the 24-hour round to validate the design and produce real numbers for the idea deck.
> It is not the final-round submission. Data is synthetic and all trading is simulated.

An autonomous agent that reads market data and news, picks and adapts a strategy by **backtesting candidates itself**, sizes and places trades in a **paper** account, and writes a plain-language journal for every trade, including the trades it chose not to take.

**Core idea: the LLM proposes, deterministic code disposes.** The model can analyse, suggest and explain. It cannot place orders. Position sizing, stop-losses, the daily loss limit and the human kill switch live in plain code that nothing the model outputs can override. A test (`test_llm_cannot_enlarge_or_bypass_risk`) runs a deliberately reckless model against the gatekeeper to prove it.

## Quick start (Windows PowerShell or any shell)

```
py -m pip install -r requirements.txt
py -m agentx.run_demo                 # full session, offline, stub LLM  -> ./out/
py -m agentx.dashboard                # live dashboard at http://localhost:8765
py -m agentx.robustness --markets 30  # results across 30 random markets, not one lucky seed
py -m unittest discover -s tests -t . # 25 tests
```

### Using your real LLM
1. Copy `.env.example` to `.env` and set `AGENTX_LLM_BASE_URL`, `AGENTX_LLM_API_KEY`, `AGENTX_LLM_MODEL` (any OpenAI-compatible `/v1/chat/completions` endpoint).
2. `py -m agentx.check_llm` pings the gateway, lists models, and runs one real strategy-selection and one signal-review.
3. `py -m agentx.run_demo --llm http` or `py -m agentx.dashboard --llm http`.

If the model fails or returns junk, the agent falls back to rule-based decisions at half size, logs it, and after 3 failures stops waiting on it (circuit breaker). A demo never crashes because the LLM did.

## How it maps to PS-02

| PS-02 requirement | Where |
|---|---|
| Reads market data and news | `tools.py`: `get_market_overview`, `get_news` |
| Picks or adapts a strategy, backtests it | `agent.py` + `tools.run_backtest`: trailing-window backtests of every candidate, re-run every 20 days |
| Paper account via a mock exchange API | `broker.py`: simulated broker with slippage and fees; no live path exists (`config.LIVE_TRADING_ENABLED = False`) |
| Position sizing, stop-losses | `risk.py`: 1% risk per trade, mandatory stop-loss, 20% single-position cap, 80% exposure cap, 5 positions max |
| Daily loss limit | `risk.py`: a day down 2% blocks all new entries |
| Human kill switch | dashboard button, or create a file named `KILL_SWITCH`. Blocks entries instantly, optionally flattens; exits still run |
| Plain-language journal for every trade | `journal.py` -> `out/journal.md`; also logs trades skipped and why |
| Backtest report vs buy-and-hold | `backtest.py`, `report.py` -> `out/report.md`, `out/equity_curve.svg` |
| Memory across sessions | `memory.py`: realised results per strategy, fed back into strategy selection (`--memory path.json`) |

```
 market data + news --> [ Agent (LLM, read-only tools) ] --> proposal (confirm/veto, conviction<=1)
                              |  get_market_overview, get_news, run_backtest,
                              |  get_portfolio, get_risk_status, get_memory
                              v
              [ Risk gatekeeper: sizing, stop, loss limit, kill switch ]  <-- human kill switch
                              v
                [ Paper broker: fills at next open ] --> [ Journal + Memory + Dashboard ]
```

Rules of thumb baked in: decisions at the close, fills at the **next** open (no look-ahead); the agent may only veto an entry or shrink it, never enlarge it, and never blocks an exit.

## Honest results (synthetic data, stub LLM)

`py -m agentx.robustness --markets 30` (see `sample_output/robustness.txt`):

- The agent beat buy-and-hold in only **9 of 30** markets. In rising markets it lags (+4.5% vs +24.2%). In falling markets it protects capital (-1.2% vs -11.7%). Max drawdown is about **a third** of buy-and-hold's (-5.6% vs -15.6%).
- Fixed strategies with no agent averaged *better* than the stub-driven agent, so these runs do **not** show that the agent adds return. The stub is rule-based; a real model may behave differently, which is what `check_llm` and `--llm http` are for.
- Synthetic headlines carry a small built-in signal so the news tool has something to read. Do not quote these returns as real-world evidence. They validate the plumbing and the guardrails.

The defensible pitch is **disciplined, explainable risk control with hard guardrails**, not "beats the market".

## Verified against a real model

The agent's JSON tool protocol was tested against `gemini-3.7-flash` through the local gateway (about 5 s per call):

- It follows the protocol cleanly: picks tools in a sensible order, returns parseable JSON every turn, and cites real numbers in its reasons.
- **Found and fixed: it skips tools.** On a bad-news case it answered "no news" without calling `get_news` in 5 of 6 runs. The agent now *requires* certain tool calls (`get_news` for the symbol, a backtest of every candidate) and bounces a final answer that skipped them. After the bounce it read the news and vetoed the trade in 6 of 6 runs. If it still refuses, the agent falls back to rules.
- **Found and fixed: a prompt flaw.** The original instruction "prefer strategies that beat buy-and-hold" made the model pick cash even when every candidate made money, because a portfolio that keeps 20% cash trails buy-and-hold in rising markets. The prompt now asks for risk-adjusted quality (return, Sharpe, drawdown).
- Not yet done: a full multi-day session driven by the real model from Python on your machine. Run `py -m agentx.check_llm`, then `py -m agentx.run_demo --llm http` (about 150 model calls, roughly 12 minutes).

## Using real prices

Put `SYMBOL.csv` files (Date,Open,High,Low,Close,Volume, Yahoo Finance style, 120+ shared days) in a folder and run `py -m agentx.run_demo --csv-dir path\to\folder`. CSVs carry no news, so the news tool returns nothing.

## Rules reminder

The hackathon rules say all building for the final round happens inside the 24-hour window (11-12 Nov). Treat this folder as the idea-stage prototype and be ready to say so. For the final repo, either rebuild inside the window or get the organizers' written OK (contact@hack4hack.com) to reuse this.

## Limitations

Long-only equities, daily bars, one broker model, equal-weight benchmark with no costs (so it is slightly generous to buy-and-hold), no intraday data, no real news feed.
