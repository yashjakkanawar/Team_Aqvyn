"""Backtest report (markdown) and a dependency-free SVG equity chart for the deck."""
from __future__ import annotations

from html import escape


def equity_svg(dates: list, series: dict, title: str = "Equity curve", w: int = 900, h: int = 380) -> str:
    colors = ["#16a34a", "#6b7280", "#2563eb", "#d97706"]
    pad_l, pad_r, pad_t, pad_b = 70, 20, 44, 40
    all_vals = [v for s in series.values() for v in s]
    lo, hi = min(all_vals), max(all_vals)
    span = (hi - lo) or 1.0
    lo, hi = lo - span * 0.05, hi + span * 0.05
    n = max(len(s) for s in series.values())

    def x(i): return pad_l + (w - pad_l - pad_r) * i / max(n - 1, 1)
    def y(v): return pad_t + (h - pad_t - pad_b) * (1 - (v - lo) / (hi - lo))

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="Segoe UI, Arial, sans-serif">',
           f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
           f'<text x="{pad_l}" y="26" font-size="16" font-weight="600" fill="#111827">{escape(title)}</text>']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        out.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{pad_l - 8}" y="{y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="#6b7280">${v:,.0f}</text>')
    out.append(f'<text x="{pad_l}" y="{h - 14}" font-size="11" fill="#6b7280">{escape(dates[0])}</text>')
    out.append(f'<text x="{w - pad_r}" y="{h - 14}" font-size="11" text-anchor="end" fill="#6b7280">{escape(dates[-1])}</text>')
    lx = w - pad_r - 10
    for idx, (name, vals) in enumerate(series.items()):
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        dash = ' stroke-dasharray="6 4"' if "hold" in name.lower() else ""
        out.append(f'<polyline fill="none" stroke="{colors[idx % 4]}" stroke-width="2"{dash} points="{pts}"/>')
        ly = 26 + idx * 0
        out.append(f'<text x="{lx - idx * 170}" y="{ly}" font-size="12" text-anchor="end" fill="{colors[idx % 4]}">{escape(name)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def markdown_report(rep: dict, title: str = "Backtest report", notes: list | None = None) -> str:
    a, b, t = rep["agent"], rep["buy_and_hold"], rep["trades"]
    lines = [f"# {title}", "",
             f"Period: {rep['start_date']} to {rep['end_date']} ({rep['days']} trading days). Paper trading on {rep.get('data_source', 'synthetic')} data.", "",
             "| Metric | Agent | Equal-weight buy-and-hold |", "|---|---:|---:|",
             f"| Total return | {a['total_return_pct']:+.2f}% | {b['total_return_pct']:+.2f}% |",
             f"| Sharpe (daily, annualised) | {a['sharpe']:.2f} | {b['sharpe']:.2f} |",
             f"| Max drawdown | {a['max_drawdown_pct']:.2f}% | {b['max_drawdown_pct']:.2f}% |",
             f"| Final equity | ${a['final_equity']:,.0f} | ${b['final_equity']:,.0f} |", "",
             f"**Excess return vs buy-and-hold: {rep['excess_return_pct']:+.2f} percentage points.**", "",
             "## Trades", "",
             f"- Closed trades: {t['closed_trades']}, win rate {t['win_rate_pct']}%, average win {t['avg_win_pct']}%, average loss {t['avg_loss_pct']}%, profit factor {t['profit_factor']}",
             f"- Average exposure: {rep['avg_exposure_pct']}% of equity",
             f"- Signals: {rep['stats']['signals']}, vetoed by the agent: {rep['stats']['vetoed']}, blocked by risk rules: {rep['stats']['blocked']}, agent fallbacks: {rep['stats']['agent_fallbacks']}", ""]
    if notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(lines)
