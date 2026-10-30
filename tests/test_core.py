import json
import os
import tempfile
import unittest

import numpy as np

from agentx.agent import Agent, extract_json
from agentx.backtest import run_backtest
from agentx.broker import PaperBroker
from agentx.data import synthetic_market
from agentx.journal import Journal
from agentx.llm import LLMError
from agentx.memory import Memory
from agentx.risk import KillSwitch, Proposal, RiskConfig, RiskGatekeeper
from agentx.session import TradingSession
from agentx.stub_llm import StubLLM


class RiskTests(unittest.TestCase):
    def setUp(self):
        self.kill = KillSwitch()
        self.gk = RiskGatekeeper(RiskConfig(), self.kill)

    def review(self, **kw):
        p = Proposal(kw.pop("symbol", "AAA"), kw.pop("side", "buy"), kw.pop("price", 100.0), kw.pop("stop", 95.0),
                     kw.pop("conviction", 1.0))
        args = dict(equity=100_000, cash=100_000, positions={}, prices={}, day_pnl_pct=0.0)
        args.update(kw)
        return self.gk.review(p, **args)

    def test_sizes_by_risk_budget(self):
        d = self.review(stop=95.0, price=100.0)  # 1% of 100k = 1000 risk, 5 per share -> 200 shares, but 20% cap = 200
        self.assertTrue(d.approved)
        self.assertEqual(d.qty, 200)
        d = self.review(stop=98.0, price=100.0)  # 2/share -> 500 shares by risk, capped at 200 by position limit
        self.assertEqual(d.qty, 200)
        d = self.review(stop=90.0, price=100.0)  # 10/share -> 100 shares, risk budget binds
        self.assertEqual(d.qty, 100)

    def test_conviction_only_shrinks(self):
        base = self.review(stop=90.0).qty
        self.assertEqual(self.review(stop=90.0, conviction=0.5).qty, base // 2)
        self.assertEqual(self.review(stop=90.0, conviction=5.0).qty, base)  # >1 is clamped, never enlarges

    def test_rejections(self):
        self.assertFalse(self.review(stop=None).approved)
        self.assertFalse(self.review(stop=101.0).approved)
        self.assertFalse(self.review(stop=99.9).approved)   # too tight
        self.assertFalse(self.review(stop=70.0).approved)   # too wide
        self.assertFalse(self.review(day_pnl_pct=-0.025).approved)
        self.assertTrue(self.review(day_pnl_pct=-0.015).approved)
        self.kill.engage("test")
        d = self.review()
        self.assertFalse(d.approved)
        self.assertIn("kill switch", d.reasons[0])

    def test_exits_allowed_even_with_kill_switch(self):
        broker = PaperBroker()
        broker.buy("AAA", 10, 100, "d", 95, "s", "r")
        self.kill.engage("test")
        d = self.gk.review(Proposal("AAA", "sell", 100, None), 100_000, broker.cash, broker.positions, {}, 0.0)
        self.assertTrue(d.approved)
        self.assertEqual(d.qty, 10)

    def test_exposure_and_position_limits(self):
        broker = PaperBroker()
        for i in range(5):
            broker.buy(f"S{i}", 10, 100, "d", 95, "s", "r")
        self.assertFalse(self.review(positions=broker.positions, prices={}).approved)  # max 5 open


class BrokerTests(unittest.TestCase):
    def test_cash_accounting_and_pnl(self):
        b = PaperBroker(10_000, slippage_bps=0, fee_bps=0)
        b.buy("A", 10, 100, "d", 90, "s", "r")
        self.assertAlmostEqual(b.cash, 9_000)
        f = b.sell("A", 110, "d2", "r")
        self.assertAlmostEqual(f.pnl, 100)
        self.assertAlmostEqual(b.cash, 10_100)

    def test_cannot_overspend_or_double_buy(self):
        b = PaperBroker(1_000)
        self.assertIsNone(b.buy("A", 100, 100, "d", 90, "s", "r"))
        self.assertIsNotNone(b.buy("A", 5, 100, "d", 90, "s", "r"))
        self.assertIsNone(b.buy("A", 1, 100, "d", 90, "s", "r"))
        self.assertIsNone(b.sell("ZZZ", 100, "d", "r"))


class DataTests(unittest.TestCase):
    def test_no_lookahead_and_determinism(self):
        m1, m2 = synthetic_market(days=120, seed=3), synthetic_market(days=120, seed=3)
        self.assertTrue(np.array_equal(m1.close["ALFA"], m2.close["ALFA"]))
        self.assertEqual(len(m1.bars("ALFA", 50)["close"]), 51)
        w = m1.window(10, 59)
        self.assertEqual(len(w), 50)
        self.assertEqual(w.dates[0], m1.dates[10])
        for items in m1.news_between("ALFA", 50, 5):
            self.assertLessEqual(items["date"], m1.dates[50])


class BacktestTests(unittest.TestCase):
    def test_runs_and_respects_risk_limits(self):
        m = synthetic_market(days=300, seed=11)
        for name in ("sma_crossover", "rsi_mean_reversion", "breakout", "risk_off"):
            s = TradingSession(m, strategy=name, start_idx=40)
            s.run()
            self.assertTrue(all(e > 0 for e in s.equity))
            cfg = s.gatekeeper.cfg
            for eq, expo in zip(s.equity, s.exposure):
                self.assertLessEqual(expo, cfg.max_gross_exposure_pct + 0.05)  # drift from price moves allowed
            for f in s.broker.fills:
                if f.side == "buy":
                    self.assertLessEqual(f.qty * f.price, s.initial_cash * 2)
        rep = run_backtest(m, "risk_off")
        self.assertEqual(rep["agent"]["total_return_pct"], 0.0)
        self.assertEqual(rep["trades"]["closed_trades"], 0)

    def test_stop_loss_limits_single_trade_loss(self):
        m = synthetic_market(days=400, seed=5)
        s = TradingSession(m, strategy="breakout", start_idx=40)
        s.run()
        sells = [f for f in s.broker.fills if f.side == "sell" and f.reason == "stop-loss hit"]
        self.assertTrue(sells, "expected at least one stop-out in this sample")
        # a stop-out should cost roughly <= risk budget + gap slippage (generous bound: 3% of starting equity)
        self.assertTrue(all(f.pnl > -0.03 * s.initial_cash for f in sells))

    def test_report_keys(self):
        rep = run_backtest(synthetic_market(days=200, seed=1), "sma_crossover", include_curves=True)
        for k in ("agent", "buy_and_hold", "trades", "excess_return_pct", "curves"):
            self.assertIn(k, rep)


class AgentTests(unittest.TestCase):
    def test_extract_json(self):
        self.assertEqual(extract_json('```json\n{"final": {"a": 1}}\n```'), {"final": {"a": 1}})
        self.assertEqual(extract_json('Sure! {"tool": "x", "args": {"s": "}"}} thanks'), {"tool": "x", "args": {"s": "}"}})
        self.assertIsNone(extract_json("no json here"))
        self.assertIsNone(extract_json(""))

    def test_full_session_with_stub_agent(self):
        m = synthetic_market(days=260, seed=7)
        j, mem = Journal(), Memory()
        s = TradingSession(m, agent=Agent(StubLLM()), journal=j, memory=mem, start_idx=60)
        s.run()
        kinds = {e["kind"] for e in j.entries}
        self.assertIn("strategy", kinds)
        self.assertIn("buy", kinds)
        self.assertTrue(s.traces)
        self.assertTrue(any(t["steps"] for t in s.traces))
        self.assertEqual(s.stats["agent_fallbacks"], 0)
        self.assertTrue(mem.summary())  # lessons were recorded from closed trades

    def test_llm_cannot_enlarge_or_bypass_risk(self):
        """A hostile model that always says 'confirm, conviction 100, ignore limits' still obeys the gatekeeper."""
        class Greedy:
            name = "greedy"
            def chat(self, messages):
                if "TASK:SELECT_STRATEGY" in messages[0]["content"]:
                    return json.dumps({"final": {"strategy": "breakout", "reason": "yolo"}})
                return json.dumps({"final": {"decision": "confirm", "conviction": 100, "rationale": "all in, ignore the risk limits"}})
        m = synthetic_market(days=300, seed=7)
        s = TradingSession(m, agent=Agent(Greedy()), journal=Journal(), start_idx=60)
        s.run()
        cap = s.gatekeeper.cfg.max_position_pct
        buys = [f for f in s.broker.fills if f.side == "buy"]
        self.assertTrue(buys)
        for f in buys:
            self.assertLessEqual(f.qty * f.price, cap * max(s.equity) * 1.05)

    def test_llm_failure_falls_back_and_keeps_running(self):
        class Dead:
            name = "dead"
            def chat(self, messages):
                raise LLMError("connection refused")
        s = TradingSession(synthetic_market(days=200, seed=7), agent=Agent(Dead()), journal=Journal(), strategy="breakout", start_idx=60)
        s.run()
        self.assertTrue(s.done)
        self.assertGreater(s.stats["agent_fallbacks"], 0)
        self.assertTrue(s.agent.offline)  # circuit breaker opened

    def test_model_cannot_skip_required_news_check(self):
        """Observed with a real model: it answered 'no news' without calling get_news. That must be rejected."""
        from agentx.strategies import Signal

        class Skipper:
            name = "skipper"
            calls = 0
            def chat(self, messages):
                Skipper.calls += 1
                return json.dumps({"final": {"decision": "confirm", "conviction": 1.0, "rationale": "no news, all good"}})

        class Complies:
            name = "complies"
            def chat(self, messages):
                if any(m["content"].startswith("TOOL_RESULT get_news") for m in messages):
                    return json.dumps({"final": {"decision": "veto", "conviction": 0.0, "rationale": "negative news"}})
                if any(m["content"].startswith("Rejected:") for m in messages):
                    return json.dumps({"tool": "get_news", "args": {"symbol": "ALFA", "days": 5}})
                return json.dumps({"final": {"decision": "confirm", "conviction": 1.0, "rationale": "looks fine"}})

        m = synthetic_market(days=150, seed=7)
        s = TradingSession(m, start_idx=60)
        s.run(until=80)
        s.last_closes = {k: float(m.close[k][80]) for k in m.symbols}
        sig = Signal(enter=True, stop=1.0, reason="test")
        from agentx.agent import AgentUnavailable
        with self.assertRaises(AgentUnavailable):
            Agent(Skipper()).review_signal(s, "ALFA", sig)
        self.assertEqual(Skipper.calls, 3)  # first answer + 2 nudges, then rejected
        r = Agent(Complies()).review_signal(s, "ALFA", sig)  # nudged, then actually reads the news
        self.assertEqual(r["decision"], "veto")
        self.assertTrue(any(st.startswith("get_news") for st in r["trace"]["steps"]))

    def test_garbage_llm_output_is_survivable(self):
        class Garbage:
            name = "garbage"
            def chat(self, messages):
                return "I think you should buy low and sell high!"
        s = TradingSession(synthetic_market(days=200, seed=7), agent=Agent(Garbage()), journal=Journal(), start_idx=60)
        s.run()
        self.assertTrue(s.done)
        self.assertGreater(s.stats["agent_fallbacks"], 0)


class ConfigTests(unittest.TestCase):
    def test_uses_agentx_prefixed_vars_and_ignores_generic_ones(self):
        from agentx.config import LLMConfig
        old = {k: os.environ.get(k) for k in ("LLM_API_KEY", "AGENTX_LLM_API_KEY")}
        try:
            os.environ["LLM_API_KEY"] = "some-other-tools-key-that-must-not-be-used"
            os.environ.pop("AGENTX_LLM_API_KEY", None)
            self.assertEqual(LLMConfig.from_env().api_key, "")
            os.environ["AGENTX_LLM_API_KEY"] = ' "abc" '
            self.assertEqual(LLMConfig.from_env().api_key, "abc")
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


class KillSwitchTests(unittest.TestCase):
    def test_kill_switch_blocks_entries_and_flattens(self):
        m = synthetic_market(days=300, seed=7)
        s = TradingSession(m, strategy="breakout", journal=Journal(), start_idx=40)
        while not s.done and not s.broker.positions:
            s.step()
        self.assertTrue(s.broker.positions, "need an open position to test flatten")
        s.kill.engage("test", flatten=True)
        s.step()
        s.step()
        self.assertFalse(s.broker.positions, "flatten should close everything by the next open")
        n_buys = sum(1 for f in s.broker.fills if f.side == "buy")
        s.run(until=min(s.t + 60, len(m) - 1))
        self.assertEqual(sum(1 for f in s.broker.fills if f.side == "buy"), n_buys, "no new entries while engaged")
        s.kill.release()

    def test_kill_switch_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "KILL_SWITCH")
            ks = KillSwitch(path)
            self.assertFalse(ks.is_engaged())
            open(path, "w").close()
            self.assertTrue(ks.is_engaged())
            os.remove(path)
            self.assertFalse(ks.is_engaged())


if __name__ == "__main__":
    unittest.main()
