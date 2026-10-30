"""Market data: a synthetic regime-switching generator (default) and a CSV loader.

The synthetic world exists to exercise the pipeline offline. Results on it say nothing about
real-world edge. Headlines in it carry a small, deliberate signal so the news tool has something
to read, which is another reason not to quote its returns as evidence.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_SYMBOLS = ["ALFA", "BRVO", "CHRL", "DLTA", "ECHO", "FXTR"]

POS_HEADLINES = [
    "{s} beats quarterly estimates and raises guidance",
    "{s} wins large multi-year contract",
    "{s} announces buyback after strong cash flow",
    "Analysts upgrade {s} on improving margins",
]
NEG_HEADLINES = [
    "{s} misses estimates and cuts outlook",
    "{s} faces regulatory probe over disclosures",
    "{s} CFO resigns amid accounting questions",
    "Analysts downgrade {s} on weak demand",
]


@dataclass
class Market:
    symbols: list
    dates: list
    open: dict
    high: dict
    low: dict
    close: dict
    volume: dict
    news: dict  # news[symbol][t] -> list of {"headline", "sentiment"}
    source: str = "synthetic"

    def __len__(self) -> int:
        return len(self.dates)

    def bars(self, symbol: str, t: int) -> dict:
        """History up to and including day t. Nothing after t is ever exposed (no look-ahead)."""
        return {
            "open": self.open[symbol][: t + 1],
            "high": self.high[symbol][: t + 1],
            "low": self.low[symbol][: t + 1],
            "close": self.close[symbol][: t + 1],
            "volume": self.volume[symbol][: t + 1],
        }

    def news_between(self, symbol: str, t: int, days: int) -> list:
        out = []
        for i in range(max(0, t - days + 1), t + 1):
            for item in self.news.get(symbol, [[]] * len(self))[i]:
                out.append({"date": self.dates[i], **item})
        return out

    def window(self, start: int, end: int) -> "Market":
        """Inclusive slice [start, end]."""
        sl = slice(start, end + 1)
        return Market(
            symbols=list(self.symbols),
            dates=self.dates[sl],
            open={s: a[sl] for s, a in self.open.items()},
            high={s: a[sl] for s, a in self.high.items()},
            low={s: a[sl] for s, a in self.low.items()},
            close={s: a[sl] for s, a in self.close.items()},
            volume={s: a[sl] for s, a in self.volume.items()},
            news={s: n[sl] for s, n in self.news.items()},
            source=self.source,
        )


def _business_days(start: str, count: int) -> list:
    d = np.datetime64(start)
    out = []
    while len(out) < count:
        if np.is_busday(d):
            out.append(str(d))
        d = d + np.timedelta64(1, "D")
    return out


def synthetic_market(symbols=None, days: int = 500, seed: int = 7, start: str = "2025-01-01") -> Market:
    rng = np.random.default_rng(seed)
    symbols = list(symbols or DEFAULT_SYMBOLS)
    n = len(symbols)

    # Market regimes: bull, bear, chop (daily mean, daily vol) with sticky transitions.
    regimes = np.array([[0.0008, 0.009], [-0.0010, 0.017], [0.0001, 0.008]])
    trans = np.array([[0.985, 0.005, 0.010], [0.020, 0.960, 0.020], [0.020, 0.010, 0.970]])
    reg = np.zeros(days, dtype=int)
    for t in range(1, days):
        reg[t] = rng.choice(3, p=trans[reg[t - 1]])
    mkt = regimes[reg, 0] + regimes[reg, 1] * rng.standard_normal(days)

    beta = rng.uniform(0.7, 1.3, n)
    idio = rng.uniform(0.007, 0.013, n)
    drift = rng.uniform(-0.0002, 0.0004, n)
    rets = beta[None, :] * mkt[:, None] + idio[None, :] * rng.standard_normal((days, n)) + drift[None, :]

    news = {s: [[] for _ in range(days)] for s in symbols}
    for i, s in enumerate(symbols):
        for t in range(days - 1):
            if rng.random() < 0.05:
                sent = float(rng.choice([-1.0, -0.6, 0.6, 1.0]))
                pool = POS_HEADLINES if sent > 0 else NEG_HEADLINES
                headline = pool[int(rng.integers(len(pool)))].format(s=s)
                news[s][t].append({"headline": headline, "sentiment": sent})
                rets[t + 1, i] += 0.012 * sent * rng.uniform(0.5, 1.0)
    rets = np.clip(rets, -0.25, 0.25)

    o, h, l, c, v = {}, {}, {}, {}, {}
    for i, s in enumerate(symbols):
        p0 = float(rng.uniform(40, 400))
        close = p0 * np.cumprod(1 + rets[:, i])
        prev = np.concatenate([[p0], close[:-1]])
        opn = prev * (1 + 0.25 * rets[:, i] + 0.002 * rng.standard_normal(days))
        high = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.004, days)))
        low = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.004, days)))
        o[s], h[s], l[s], c[s] = opn, high, low, close
        v[s] = rng.lognormal(13, 0.4, days)
    return Market(symbols, _business_days(start, days), o, h, l, c, v, news, source="synthetic")


def load_csv_dir(path: str, symbols=None) -> Market:
    """Load SYMBOL.csv files (Date,Open,High,Low,Close,Volume - Yahoo Finance style).

    Dates are aligned on the intersection. No news is available from CSVs.
    """
    files = sorted(Path(path).glob("*.csv"))
    if symbols:
        files = [f for f in files if f.stem.upper() in {s.upper() for s in symbols}]
    if not files:
        raise FileNotFoundError(f"no CSV files found in {path}")
    raw = {}
    for f in files:
        rows = {}
        with f.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    rows[r["Date"][:10]] = [float(r[k]) for k in ("Open", "High", "Low", "Close")] + [
                        float(r.get("Volume") or 0)
                    ]
                except (KeyError, ValueError):
                    continue
        raw[f.stem.upper()] = rows
    dates = sorted(set.intersection(*[set(r) for r in raw.values()]))
    if len(dates) < 120:
        raise ValueError(f"only {len(dates)} shared dates across files; need at least 120")
    syms = list(raw)
    cols = {k: {s: np.array([raw[s][d][i] for d in dates]) for s in syms} for i, k in enumerate(["open", "high", "low", "close", "volume"])}
    return Market(syms, dates, cols["open"], cols["high"], cols["low"], cols["close"], cols["volume"],
                  {s: [[] for _ in dates] for s in syms}, source=f"csv:{path}")
