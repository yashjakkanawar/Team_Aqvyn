"""Run a full agent session end to end and write report, journal, curves and chart to ./out.

    python -m agentx.run_demo                 # offline, stub LLM
    python -m agentx.run_demo --llm http      # your real LLM gateway (see .env)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .agent import Agent
from .backtest import compare_all, format_table
from .config import LLMConfig
from .data import load_csv_dir, synthetic_market
from .journal import Journal
from .llm import HttpLLM
from .memory import Memory
from .report import equity_svg, markdown_report
from .risk import KillSwitch
from .session import TradingSession
from .stub_llm import StubLLM


def build_agent(kind: str):
    if kind == "none":
        return None
    if kind == "http":
        return Agent(HttpLLM(LLMConfig.from_env()))
    return Agent(StubLLM())


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", choices=["stub", "http", "none"], default="stub")
    ap.add_argument("--days", type=int, default=310, help="total days of data (first 60 are warm-up)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--csv-dir", help="folder of SYMBOL.csv files instead of synthetic data")
    ap.add_argument("--memory", help="path to a memory JSON file (persists lessons across runs)")
    ap.add_argument("--out", default="out")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    market = load_csv_dir(a.csv_dir) if a.csv_dir else synthetic_market(days=a.days, seed=a.seed)
    journal = Journal(str(out / "journal.md"), str(out / "journal.jsonl"))
    session = TradingSession(market, agent=build_agent(a.llm), journal=journal,
                             memory=Memory(a.memory) if a.memory else Memory(),
                             kill_switch=KillSwitch("KILL_SWITCH"), start_idx=60)
    print(f"data: {market.source}, {len(market)} days | llm: {a.llm} | symbols: {', '.join(market.symbols)}")
    session.run()

    rep = session.report(include_curves=True)
    rep["data_source"] = market.source
    notes = ["Paper trading only: the broker is simulated and no live-trading path exists in the code.",
             "Decisions are made at the close and filled at the next open, with 5 bps slippage and 2 bps fees. The benchmark pays no costs.",
             "On synthetic data these numbers validate the pipeline; they are not evidence of real-world edge."]
    (out / "report.md").write_text(markdown_report(rep, "Agent session report", notes), encoding="utf-8")
    c = rep["curves"]
    (out / "equity_curve.svg").write_text(
        equity_svg(c["dates"], {"Agent": c["agent"], "Buy-and-hold": c["buy_and_hold"]}, "Agent vs equal-weight buy-and-hold"), encoding="utf-8")
    with (out / "equity_curve.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "agent", "buy_and_hold"])
        w.writerows(zip(c["dates"], c["agent"], c["buy_and_hold"]))

    print(f"\nAgent {rep['agent']['total_return_pct']:+.2f}% | Buy-and-hold {rep['buy_and_hold']['total_return_pct']:+.2f}% | "
          f"excess {rep['excess_return_pct']:+.2f} pts | max drawdown {rep['agent']['max_drawdown_pct']:.2f}% vs {rep['buy_and_hold']['max_drawdown_pct']:.2f}%")
    print(f"Trades: {rep['trades']['closed_trades']} closed, win rate {rep['trades']['win_rate_pct']}% | journal entries: {len(journal.entries)}")
    chosen = [s["strategy"] for s in session.strategy_log]
    switches = sum(1 for a_, b_ in zip(chosen, chosen[1:]) if a_ != b_)
    print(f"Strategy reviews: {len(chosen)}, switches: {switches} | agent fallbacks: {session.stats['agent_fallbacks']}")
    print("\nFixed-strategy backtests on the same data (no LLM):")
    print(format_table(compare_all(market, start_idx=60)))
    print(f"\nFiles written to {out.resolve()}")


if __name__ == "__main__":
    main()
