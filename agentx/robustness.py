"""Run the agent over many synthetic markets so results never rest on one lucky seed.

    python -m agentx.robustness --markets 30
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .agent import Agent
from .backtest import run_backtest
from .data import synthetic_market
from .session import TradingSession
from .stub_llm import StubLLM

FIXED = ("sma_crossover", "rsi_mean_reversion", "breakout")


def run(markets: int = 30, days: int = 310, make_agent=lambda: Agent(StubLLM())) -> str:
    rows = []
    for seed in range(1, markets + 1):
        m = synthetic_market(days=days, seed=seed)
        a = TradingSession(m, agent=make_agent(), start_idx=60).run().report()
        fixed = {n: run_backtest(m, n, start_idx=60)["agent"]["total_return_pct"] for n in FIXED}
        rows.append((a["agent"]["total_return_pct"], a["buy_and_hold"]["total_return_pct"],
                     a["agent"]["max_drawdown_pct"], a["buy_and_hold"]["max_drawdown_pct"], fixed))
    ag, bh = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    up, dn = bh > 0, bh <= 0
    out = [f"Robustness over {markets} synthetic markets ({days} days each, first 60 are warm-up)",
           f"- Agent: mean {ag.mean():+.2f}%, median {np.median(ag):+.2f}%, positive in {int((ag > 0).sum())}/{markets}",
           f"- Buy-and-hold: mean {bh.mean():+.2f}%, median {np.median(bh):+.2f}%, positive in {int((bh > 0).sum())}/{markets}",
           f"- Agent beat buy-and-hold in {int((ag > bh).sum())}/{markets} markets",
           f"- Rising markets ({int(up.sum())}): agent {ag[up].mean():+.2f}% vs buy-and-hold {bh[up].mean():+.2f}%" if up.any() else "- no rising markets",
           f"- Falling markets ({int(dn.sum())}): agent {ag[dn].mean():+.2f}% vs buy-and-hold {bh[dn].mean():+.2f}%" if dn.any() else "- no falling markets",
           f"- Max drawdown: agent {np.mean([r[2] for r in rows]):.1f}% vs buy-and-hold {np.mean([r[3] for r in rows]):.1f}% (mean)"]
    for n in FIXED:
        v = np.array([r[4][n] for r in rows])
        out.append(f"- Fixed '{n}' (no agent): mean {v.mean():+.2f}%, beat buy-and-hold in {int((v > bh).sum())}/{markets}")
    return "\n".join(out)


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", type=int, default=30)
    ap.add_argument("--days", type=int, default=310)
    a = ap.parse_args(argv)
    print(run(a.markets, a.days))


if __name__ == "__main__":
    main()
