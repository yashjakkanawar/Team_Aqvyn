"""TradingSession: one engine used for live paper sessions, the dashboard AND backtests.

Using one engine means a backtest obeys exactly the same guardrails as the live agent.

Daily flow (decisions at the close, fills at the NEXT open, so there is no look-ahead):
  1. human kill switch is applied
  2. orders queued yesterday are filled at today's open (kill switch re-checked at fill time)
  3. stop-losses are checked against today's low
  4. portfolio and the buy-and-hold benchmark are marked at the close
  5. the agent re-selects a strategy every `reselect_days`
  6. signals -> agent review (entries only) -> deterministic risk gatekeeper -> queued orders
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .broker import PaperBroker, Position
from .journal import Journal, money
from .memory import Memory
from .metrics import curve_metrics, trade_stats
from .risk import KillSwitch, Proposal, RiskConfig, RiskGatekeeper
from .strategies import STRATEGIES


@dataclass
class Order:
    symbol: str
    side: str
    qty: int
    stop: Optional[float]
    strategy: str
    reason: str
    agent_note: str = ""
    conviction: float = 1.0
    decided_on: str = ""
    risk_notes: list = field(default_factory=list)


class TradingSession:
    def __init__(self, market, cash: float = 100_000.0, risk_cfg: Optional[RiskConfig] = None, agent=None,
                 strategy: str = "sma_crossover", strategy_params: Optional[dict] = None, reselect_days: int = 20,
                 start_idx: int = 60, journal: Optional[Journal] = None, memory: Optional[Memory] = None,
                 kill_switch: Optional[KillSwitch] = None, slippage_bps: float = 5.0, fee_bps: float = 2.0,
                 cooldown_days: int = 5):
        self.market = market
        self.symbols = list(market.symbols)
        self.initial_cash = cash
        self.broker = PaperBroker(cash, slippage_bps, fee_bps)
        self.kill = kill_switch or KillSwitch()
        self.gatekeeper = RiskGatekeeper(risk_cfg or RiskConfig(), self.kill)
        self.agent, self.journal, self.memory = agent, journal, memory
        self.strategy, self.strategy_params = strategy, strategy_params
        self.reselect_days, self.start_idx, self.cooldown_days = reselect_days, start_idx, cooldown_days
        self.t = start_idx - 1
        self.pending: list = []
        self.cooldown: dict = {}
        self.dates: list = []
        self.equity: list = []
        self.bench: list = []
        self.exposure: list = []
        self.prev_equity = cash
        self.day_pnl = 0.0
        self.bench_units: Optional[dict] = None
        self.last_select: Optional[int] = None
        self.strategy_log: list = []
        self.traces: list = []
        self.stats = {"signals": 0, "vetoed": 0, "blocked": 0, "agent_fallbacks": 0}
        self._flattened = False
        self.last_closes: dict = {}

    # ------------------------------------------------------------------ helpers
    @property
    def done(self) -> bool:
        return self.t >= len(self.market) - 1

    @property
    def date(self) -> str:
        return self.market.dates[max(self.t, 0)]

    @property
    def current_equity(self) -> float:
        return self.equity[-1] if self.equity else self.initial_cash

    def _j(self, kind: str, symbol: str, text: str, **data) -> None:
        if self.journal is not None:
            self.journal.add(kind, self.date, symbol, text, **data)

    def _strategy(self):
        fn, defaults, _ = STRATEGIES[self.strategy]
        return fn, {**defaults, **(self.strategy_params or {})}

    def run(self, until: Optional[int] = None) -> "TradingSession":
        while not self.done and (until is None or self.t < until):
            self.step()
        return self

    # ------------------------------------------------------------------ main step
    def step(self) -> None:
        if self.done:
            return
        self.t += 1
        t, m = self.t, self.market
        first = t == self.start_idx
        self._apply_kill_switch()
        self._fill_pending(t)
        self._check_stops(t)

        closes = {s: float(m.close[s][t]) for s in self.symbols}
        self.last_closes = closes
        eq = self.broker.equity(closes)
        if first or self.bench_units is None:
            share = self.initial_cash / len(self.symbols)
            self.bench_units = {s: share / closes[s] for s in self.symbols}
        self.day_pnl = 0.0 if first else eq / self.prev_equity - 1
        self.prev_equity = eq
        self.dates.append(m.dates[t])
        self.equity.append(eq)
        self.bench.append(sum(self.bench_units[s] * closes[s] for s in self.symbols))
        self.exposure.append(self.broker.gross(closes) / eq if eq else 0.0)

        if not self.done:
            self._decide(t, closes)

    # ------------------------------------------------------------------ kill switch
    def _apply_kill_switch(self) -> None:
        if self.kill.is_engaged():
            cancelled = [o for o in self.pending if o.side == "buy"]
            if cancelled:
                self._j("risk", "", f"Human kill switch engaged ({self.kill.reason}). Cancelled {len(cancelled)} pending entry order(s): "
                        + ", ".join(o.symbol for o in cancelled) + ".")
            self.pending = [o for o in self.pending if o.side == "sell"]
            if self.kill.flatten and not self._flattened:
                queued = {o.symbol for o in self.pending}
                for sym, pos in self.broker.positions.items():
                    if sym not in queued:
                        self.pending.append(Order(sym, "sell", pos.qty, None, pos.strategy,
                                                  "kill switch: flatten all positions", decided_on=self.date))
                self._flattened = True
                self._j("risk", "", "Kill switch asked to flatten: all open positions will be sold at the next open.")
        else:
            self._flattened = False

    # ------------------------------------------------------------------ fills
    def _fill_pending(self, t: int) -> None:
        m = self.market
        orders, self.pending = self.pending, []
        for o in orders:
            px = float(m.open[o.symbol][t])
            if o.side == "buy":
                if self.kill.is_engaged():
                    self._j("skip", o.symbol, f"Entry order for {o.symbol} cancelled at the open: kill switch is engaged.")
                    continue
                fill = self.broker.buy(o.symbol, o.qty, px, self.date, o.stop, o.strategy, o.reason, t)
                if fill is None:
                    self._j("skip", o.symbol, f"Entry order for {o.symbol} could not be filled (not enough cash).")
                    continue
                self._journal_buy(fill, o)
            else:
                fill = self.broker.sell(o.symbol, px, self.date, o.reason, t)
                if fill:
                    self._journal_sell(fill)

    def _check_stops(self, t: int) -> None:
        m = self.market
        for sym, pos in list(self.broker.positions.items()):
            if pos.stop is not None and float(m.low[sym][t]) <= pos.stop:
                ref = min(float(m.open[sym][t]), pos.stop)  # a gap below the stop fills at the open
                fill = self.broker.sell(sym, ref, self.date, "stop-loss hit", t)
                if fill:
                    self.cooldown[sym] = t + self.cooldown_days
                    self._journal_sell(fill)

    def _journal_buy(self, f, o: Order) -> None:
        eq = max(self.prev_equity, 1.0)
        stop_txt = ""
        if o.stop:
            risk = f.qty * (f.price - o.stop)
            stop_txt = (f" Safety net: stop-loss at {o.stop:.2f} ({(o.stop / f.price - 1) * 100:.1f}%); "
                        f"if it triggers I lose about {money(risk)} ({risk / eq * 100:.2f}% of the portfolio).")
        view = f" Agent's view: {o.agent_note}" if o.agent_note else ""
        self._j("buy", f.symbol,
                f"Bought {f.qty} shares at {f.price:.2f} (~{money(f.qty * f.price)}, {f.qty * f.price / eq * 100:.1f}% of portfolio). "
                f"Strategy: {f.strategy}. Why: {o.reason}.{view}{stop_txt} Sizing: {'; '.join(o.risk_notes)}.",
                qty=f.qty, price=f.price)

    def _journal_sell(self, f) -> None:
        if self.memory is not None and f.pnl_pct is not None:
            self.memory.record(f.strategy, f.symbol, f.pnl_pct, f.date)
        verb = "Lost" if f.pnl < 0 else "Made"
        self._j("sell", f.symbol,
                f"Sold {f.qty} shares at {f.price:.2f}. Reason: {f.reason}. {verb} {money(abs(f.pnl))} "
                f"({f.pnl_pct * 100:+.1f}%) after {f.held_days} trading day(s). Strategy: {f.strategy}.",
                qty=f.qty, price=f.price, pnl=f.pnl)

    # ------------------------------------------------------------------ decisions
    def _decide(self, t: int, closes: dict) -> None:
        if self.agent is not None and (self.last_select is None or t - self.last_select >= self.reselect_days):
            self._select_strategy(t)
        fn, params = self._strategy()
        queued = {o.symbol for o in self.pending}
        for sym in self.symbols:
            if sym in queued or t < self.cooldown.get(sym, -1):
                continue
            sig = fn(self.market.bars(sym, t), **params)
            held = sym in self.broker.positions
            if not held and sig.enter:
                self._consider(sym, "buy", sig, closes)
            elif held and sig.exit:
                self._consider(sym, "sell", sig, closes)

    def _select_strategy(self, t: int) -> None:
        self.last_select = t
        old = self.strategy
        try:
            res = self.agent.select_strategy(self)
        except Exception as e:  # AgentUnavailable or anything unexpected: keep going with the current strategy
            self.stats["agent_fallbacks"] += 1
            self._j("system", "", f"Could not ask the agent to pick a strategy ({e}). Keeping '{old}'.")
            self.strategy_log.append({"date": self.date, "strategy": old, "reason": f"agent unavailable ({e})", "source": "fallback"})
            return
        self.traces.append({"date": self.date, "task": "select_strategy", **res["trace"]})
        new = res["strategy"] if res["strategy"] in STRATEGIES else old
        self.strategy_log.append({"date": self.date, "strategy": new, "reason": res["reason"], "source": res["source"]})
        change = f"switched from '{old}' to '{new}'" if new != old else f"kept '{old}'"
        self._j("strategy", "", f"Strategy review: {change}. {res['reason']}", strategy=new)
        self.strategy, self.strategy_params = new, None

    def _projected(self, closes: dict):
        """Portfolio as it will look once today's already-approved entries fill, so limits can't be
        breached by several orders that were each sized against the same stale state."""
        positions, cash = dict(self.broker.positions), self.broker.cash
        for o in self.pending:
            if o.side == "buy":
                positions[o.symbol] = Position(o.symbol, o.qty, closes[o.symbol], o.stop, self.date, o.strategy)
                cash -= o.qty * closes[o.symbol]
        return positions, cash

    def _consider(self, sym: str, side: str, sig, closes: dict) -> None:
        self.stats["signals"] += 1
        conviction, note = 1.0, ""
        if side == "buy" and self.agent is not None:
            try:
                res = self.agent.review_signal(self, sym, sig)
            except Exception as e:
                self.stats["agent_fallbacks"] += 1
                res = {"decision": "confirm", "conviction": 0.5, "rationale": f"agent unavailable ({e}); rule-based fallback at half size",
                       "source": "fallback", "trace": {"steps": [], "final": "fallback"}}
            self.traces.append({"date": self.date, "task": f"review {sym}", **res["trace"]})
            conviction, note = res["conviction"], res["rationale"]
            if res["decision"] == "veto":
                self.stats["vetoed"] += 1
                self._j("skip", sym, f"Strategy '{self.strategy}' wanted to buy ({sig.reason}), but the agent vetoed it: {note}")
                return
        prop = Proposal(sym, side, closes[sym], sig.stop, conviction, self.strategy, sig.reason)
        positions, cash = self._projected(closes)
        dec = self.gatekeeper.review(prop, self.current_equity, cash, positions, closes, self.day_pnl)
        if not dec.approved:
            self.stats["blocked"] += 1
            self._j("skip", sym, f"Wanted to {side} ({sig.reason}) but the risk gatekeeper said no: " + "; ".join(dec.reasons) + ".")
            return
        self.pending.append(Order(sym, side, dec.qty, sig.stop, self.strategy, sig.reason, note, conviction, self.date, dec.notes))

    # ------------------------------------------------------------------ reporting
    def report(self, include_curves: bool = False) -> dict:
        out = {
            "strategy_final": self.strategy,
            "days": len(self.equity),
            "start_date": self.dates[0] if self.dates else None,
            "end_date": self.dates[-1] if self.dates else None,
            "agent": curve_metrics(self.equity),
            "buy_and_hold": curve_metrics(self.bench),
            "trades": trade_stats(self.broker.fills),
            "avg_exposure_pct": round(100 * float(np.mean(self.exposure)), 1) if self.exposure else 0.0,
            "stats": dict(self.stats),
        }
        out["excess_return_pct"] = round(out["agent"]["total_return_pct"] - out["buy_and_hold"]["total_return_pct"], 2)
        if include_curves:
            out["curves"] = {"dates": list(self.dates), "agent": list(self.equity), "buy_and_hold": list(self.bench)}
        return out
