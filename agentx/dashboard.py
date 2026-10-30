"""Live dashboard (stdlib only):  python -m agentx.dashboard [--llm stub|http|none]  then open http://localhost:8765"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .agent import Agent
from .config import LLMConfig
from .data import synthetic_market
from .journal import Journal
from .llm import HttpLLM
from .memory import Memory
from .risk import KillSwitch
from .session import TradingSession
from .stub_llm import StubLLM


class App:
    def __init__(self, llm_kind: str, seed: int = 7, days: int = 360):
        self.llm_kind, self.seed, self.days = llm_kind, seed, days
        self.lock = threading.RLock()
        self.running, self.interval = False, 0.6
        self.queued, self.busy, self._cache = 0, False, None
        self.reset()

    def reset(self, seed=None) -> None:
        with self.lock:
            if seed is not None:
                self.seed = seed
            self.running = False
            market = synthetic_market(days=self.days, seed=self.seed)
            self.agent = None if self.llm_kind == "none" else Agent(HttpLLM(LLMConfig.from_env()) if self.llm_kind == "http" else StubLLM())
            self.kill = KillSwitch("KILL_SWITCH")
            self.session = TradingSession(market, agent=self.agent, journal=Journal(), memory=Memory(),
                                          kill_switch=self.kill, start_idx=60)
            self.queued = 0
            self.session.step()  # first mark so the chart has a point
            self._cache = self._snapshot()

    def step(self, n: int = 1) -> None:
        """Queue n days; the worker thread runs them so HTTP requests never wait on a slow LLM."""
        self.queued += n

    def loop(self) -> None:
        last = 0.0
        while True:
            time.sleep(0.05)
            go = self.queued > 0 or (self.running and time.time() - last >= self.interval)
            if not go:
                continue
            with self.lock:
                if self.session.done:
                    self.running, self.queued = False, 0
                    continue
                self.busy = True
                try:
                    self.session.step()
                finally:
                    self.busy = False
                self.queued = max(0, self.queued - 1)
                self._cache = self._snapshot()
            last = time.time()

    def state(self) -> dict:
        """Lock-free read of the last snapshot, with the live kill-switch status overlaid."""
        d = dict(self._cache or {})
        d["busy"] = self.busy
        d["running"] = self.running
        d["risk"] = {**d.get("risk", {}), "kill": self.kill.is_engaged(), "kill_reason": self.kill.reason}
        return d

    def _snapshot(self) -> dict:
        with self.lock:
            s = self.session
            c = s.gatekeeper.cfg
            closes = s.last_closes
            eq, bench = s.current_equity, (s.bench[-1] if s.bench else s.initial_cash)
            rep = s.report()
            positions = [{"symbol": sym, "qty": p.qty, "avg": round(p.avg_price, 2), "last": round(closes.get(sym, p.avg_price), 2),
                          "pnl_pct": round((closes.get(sym, p.avg_price) / p.avg_price - 1) * 100, 2),
                          "stop": round(p.stop, 2) if p.stop else None, "strategy": p.strategy}
                         for sym, p in s.broker.positions.items()]
            return {
                "date": s.date, "day": s.t - s.start_idx + 1, "total_days": len(s.market) - s.start_idx,
                "done": s.done, "running": self.running, "interval": self.interval,
                "llm": {"kind": self.llm_kind, "offline": bool(self.agent and self.agent.offline),
                        "last_error": self.agent.last_error if self.agent else ""},
                "curve": {"dates": s.dates, "agent": [round(x, 2) for x in s.equity], "bench": [round(x, 2) for x in s.bench]},
                "equity": round(eq, 2), "bench_equity": round(bench, 2), "cash": round(s.broker.cash, 2),
                "day_pnl_pct": round(s.day_pnl * 100, 2),
                "max_dd": rep["agent"]["max_drawdown_pct"], "bench_max_dd": rep["buy_and_hold"]["max_drawdown_pct"],
                "strategy": s.strategy, "positions": positions,
                "risk": {"kill": s.kill.is_engaged(), "kill_reason": s.kill.reason, "daily_limit": c.daily_loss_limit_pct * 100,
                         "risk_per_trade": c.risk_per_trade_pct * 100, "max_pos": c.max_position_pct * 100,
                         "max_open": c.max_open_positions, "gross_pct": round(100 * s.exposure[-1], 1) if s.exposure else 0,
                         "max_gross": c.max_gross_exposure_pct * 100,
                         "halted_today": s.day_pnl <= -c.daily_loss_limit_pct},
                "stats": s.stats, "trades": rep["trades"],
                "journal": s.journal.latest(40), "traces": list(reversed(s.traces[-6:])),
            }


class Handler(BaseHTTPRequestHandler):
    app: App
    wbufsize = 65536  # one packet per response (avoids Nagle/delayed-ACK stalls on Windows)

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/state":
            self._send(200, json.dumps(self.app.state(), default=float).encode(), "application/json")
        elif self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        a, p = self.app, self.path
        if p == "/api/step":
            a.step(int(body.get("n", 1)))
        elif p == "/api/play":
            a.running = True
        elif p == "/api/pause":
            a.running = False
        elif p == "/api/speed":
            a.interval = max(0.05, min(3.0, float(body.get("interval", 0.6))))
        elif p == "/api/kill":  # deliberately lock-free: the kill switch must never wait on the LLM
            a.kill.engage("human pressed the kill switch", flatten=bool(body.get("flatten")))
        elif p == "/api/release":
            a.kill.release()
        elif p == "/api/reset":
            a.reset(body.get("seed"))
        elif p == "/api/llm-reset" and a.agent:
            a.agent.reset()
        else:
            return self._send(404, b"not found", "text/plain")
        self._send(200, b"{}", "application/json")


def main(argv=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=["stub", "http", "none"], default="stub")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    app = App(a.llm, a.seed)
    Handler.app = app
    threading.Thread(target=app.loop, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"AgentX paper-trading dashboard: http://localhost:{a.port}  (llm={a.llm}, paper trading + synthetic data)  Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgentX Paper Trader</title>
<style>
:root{--bg:#0e1116;--card:#161b22;--line:#252c36;--text:#e6edf3;--mute:#8b96a5;--green:#3fb950;--red:#f85149;--blue:#58a6ff;--amber:#d29922}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 "Segoe UI",system-ui,sans-serif}
header{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:14px 20px;border-bottom:1px solid var(--line)}
h1{font-size:17px;margin:0 8px 0 0}.chip{font-size:11px;padding:3px 9px;border-radius:99px;border:1px solid var(--line);color:var(--mute)}
.chip.ok{color:var(--green);border-color:#285a33}.chip.warn{color:var(--amber);border-color:#5c4a12}.chip.bad{color:var(--red);border-color:#6b2420}
main{display:grid;grid-template-columns:minmax(0,2fr) minmax(0,1fr);gap:14px;padding:14px 20px}
@media(max-width:980px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute);margin:0 0 10px;font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px}
.kpi small{display:block;color:var(--mute);font-size:11px}.kpi b{font-size:20px;font-variant-numeric:tabular-nums}
.pos{color:var(--green)}.neg{color:var(--red)}
canvas{width:100%;height:280px;display:block}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px;align-items:center}
button,select{background:#21262d;color:var(--text);border:1px solid var(--line);border-radius:7px;padding:7px 12px;font:inherit;cursor:pointer}
button:hover{border-color:var(--blue)}button.primary{background:#1f6feb;border-color:#1f6feb}
.kill{width:100%;padding:14px;font-weight:700;font-size:15px;background:#b62324;border-color:#f85149;color:#fff}
.kill.engaged{background:#1a7f37;border-color:#3fb950}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:5px 6px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}th{color:var(--mute);font-weight:500;font-size:11px}
.row{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid var(--line)}.row:last-child{border:0}
.feed{max-height:420px;overflow:auto}.entry{padding:8px 0;border-bottom:1px solid var(--line)}
.tag{display:inline-block;font-size:10px;font-weight:700;padding:1px 7px;border-radius:5px;margin-right:6px;text-transform:uppercase}
.t-buy{background:#12361d;color:var(--green)}.t-sell{background:#3d1a18;color:var(--red)}.t-skip{background:#3a2f10;color:var(--amber)}
.t-strategy{background:#112d4e;color:var(--blue)}.t-risk,.t-system{background:#3d1a18;color:var(--red)}
.entry small{color:var(--mute)}.trace{font:12px/1.4 ui-monospace,Consolas,monospace;color:var(--mute);padding:6px 0;border-bottom:1px solid var(--line)}
.trace b{color:var(--text)}.note{color:var(--mute);font-size:12px;margin-top:8px}
</style></head><body>
<header><h1>AgentX Paper Trader</h1>
<span class="chip ok">PAPER ONLY</span><span class="chip warn">SYNTHETIC DATA</span><span class="chip" id="llm"></span><span class="chip" id="strategy"></span>
<span style="margin-left:auto;color:var(--mute)" id="clock"></span></header>
<main>
<div>
 <div class="card"><div class="controls">
  <button class="primary" id="play">Play</button><button id="step">Step 1 day</button><button id="step5">+5 days</button>
  <select id="speed"><option value="1.2">Slow</option><option value="0.6" selected>Normal</option><option value="0.15">Fast</option></select>
  <button id="reset">New market</button><span style="margin-left:auto;color:var(--mute)" id="progress"></span></div>
  <div class="kpis">
   <div class="kpi"><small>Agent equity</small><b id="eq">-</b></div>
   <div class="kpi"><small>Buy-and-hold</small><b id="bh">-</b></div>
   <div class="kpi"><small>Day P&amp;L</small><b id="day">-</b></div>
   <div class="kpi"><small>Max drawdown (agent / B&amp;H)</small><b id="dd">-</b></div></div>
  <canvas id="chart"></canvas></div>
 <div class="card" style="margin-top:14px"><h2>Trade journal (plain language)</h2><div class="feed" id="journal"></div></div>
</div>
<div>
 <div class="card"><h2>Human kill switch</h2><button class="kill" id="kill">KILL SWITCH: stop new trades</button>
  <label class="note"><input type="checkbox" id="flatten"> also sell all positions at the next open</label>
  <div class="note" id="killnote"></div></div>
 <div class="card" style="margin-top:14px"><h2>Risk guardrails (hard-coded)</h2><div id="risk"></div></div>
 <div class="card" style="margin-top:14px"><h2>Open positions</h2><table><thead><tr><th>Symbol</th><th>Qty</th><th>Avg</th><th>Last</th><th>P&amp;L</th><th>Stop</th></tr></thead><tbody id="pos"></tbody></table><div class="note" id="nopos"></div></div>
 <div class="card" style="margin-top:14px"><h2>Agent trace (tool calls)</h2><div id="traces"></div></div>
</div></main>
<script>
const $=id=>document.getElementById(id);let S=null;
const fmt=x=>'$'+Math.round(x).toLocaleString();const pct=x=>(x>0?'+':'')+x.toFixed(2)+'%';
const cls=x=>x>=0?'pos':'neg';
async function post(p,b){await fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});load()}
$('play').onclick=()=>post(S&&S.running?'/api/pause':'/api/play');$('step').onclick=()=>post('/api/step',{n:1});$('step5').onclick=()=>post('/api/step',{n:5});
$('reset').onclick=()=>post('/api/reset',{seed:Math.floor(Math.random()*1000)});$('speed').onchange=e=>post('/api/speed',{interval:+e.target.value});
$('kill').onclick=()=>post(S&&S.risk.kill?'/api/release':'/api/kill',{flatten:$('flatten').checked});
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function draw(c){const cv=$('chart'),dpr=devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;cv.width=w*dpr;cv.height=h*dpr;
 const g=cv.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,w,h);const a=c.agent,b=c.bench;if(a.length<2)return;
 const all=a.concat(b),lo=Math.min(...all),hi=Math.max(...all),pad=(hi-lo)*.08||1,L=62,R=10,T=10,B=22,n=a.length;
 const X=i=>L+(w-L-R)*i/(n-1),Y=v=>T+(h-T-B)*(1-(v-(lo-pad))/(hi-lo+2*pad));
 g.font='11px Segoe UI';g.fillStyle='#8b96a5';g.strokeStyle='#252c36';
 for(let k=0;k<5;k++){const v=lo-pad+(hi-lo+2*pad)*k/4;g.beginPath();g.moveTo(L,Y(v));g.lineTo(w-R,Y(v));g.stroke();g.textAlign='right';g.fillText(fmt(v),L-6,Y(v)+4)}
 g.textAlign='left';g.fillText(c.dates[0],L,h-6);g.textAlign='right';g.fillText(c.dates[n-1],w-R,h-6);
 const line=(s,col,dash)=>{g.beginPath();g.setLineDash(dash||[]);g.strokeStyle=col;g.lineWidth=2;s.forEach((v,i)=>i?g.lineTo(X(i),Y(v)):g.moveTo(X(i),Y(v)));g.stroke();g.setLineDash([])};
 line(b,'#8b96a5',[6,4]);line(a,'#3fb950');}
function render(s){S=s;$('clock').textContent=s.date;$('progress').textContent='Day '+s.day+' / '+s.total_days+(s.done?' (finished)':'')+(s.busy?'  - agent is thinking...':'');
 $('play').textContent=s.running?'Pause':'Play';
 const L=$('llm');L.textContent='LLM: '+s.llm.kind+(s.llm.offline?' (OFFLINE, using rules)':'');L.className='chip '+(s.llm.offline?'bad':'ok');
 $('strategy').textContent='Strategy: '+s.strategy;
 $('eq').textContent=fmt(s.equity);$('eq').className=cls(s.equity-100000);$('bh').textContent=fmt(s.bench_equity);$('bh').className=cls(s.bench_equity-100000);
 $('day').textContent=pct(s.day_pnl_pct);$('day').className=cls(s.day_pnl_pct);$('dd').textContent=s.max_dd.toFixed(1)+'% / '+s.bench_max_dd.toFixed(1)+'%';
 draw(s.curve);
 const k=$('kill');k.textContent=s.risk.kill?'KILL SWITCH ENGAGED. Click to release':'KILL SWITCH: stop new trades';k.className='kill'+(s.risk.kill?' engaged':'');
 $('killnote').textContent=s.risk.kill?'Reason: '+s.risk.kill_reason+'. You can also create a file named KILL_SWITCH next to the project to trigger it.':'Blocks all new entries instantly. Exits still run. Tick the box to flatten too.';
 const r=s.risk,rows=[['Risk per trade',r.risk_per_trade+'% of equity'],['Max single position',r.max_pos+'%'],['Max open positions',s.positions.length+' / '+r.max_open],
  ['Gross exposure',r.gross_pct+'% / '+r.max_gross+'% cap'],['Daily loss limit','-'+r.daily_limit+'%'+(r.halted_today?' <b class="neg">(HIT: entries halted)</b>':'')],
  ['Signals / vetoed / blocked',s.stats.signals+' / '+s.stats.vetoed+' / '+s.stats.blocked],['Closed trades / win rate',s.trades.closed_trades+' / '+s.trades.win_rate_pct+'%']];
 $('risk').innerHTML=rows.map(x=>'<div class="row"><span>'+x[0]+'</span><span>'+x[1]+'</span></div>').join('');
 $('pos').innerHTML=s.positions.map(p=>'<tr><td>'+p.symbol+'</td><td>'+p.qty+'</td><td>'+p.avg+'</td><td>'+p.last+'</td><td class="'+cls(p.pnl_pct)+'">'+pct(p.pnl_pct)+'</td><td>'+(p.stop??'-')+'</td></tr>').join('');
 $('nopos').textContent=s.positions.length?'':'No open positions.';
 $('journal').innerHTML=s.journal.map(e=>'<div class="entry"><span class="tag t-'+e.kind+'">'+e.kind+(e.symbol?' '+e.symbol:'')+'</span><small>'+e.date+'</small><div>'+esc(e.text)+'</div></div>').join('')||'<div class="note">Press Play or Step. Every trade, and every trade NOT taken, is explained here.</div>';
 $('traces').innerHTML=s.traces.map(t=>'<div class="trace"><b>'+esc(t.task)+'</b> <small>'+t.date+' ['+t.source+']</small><br>'+t.steps.map(x=>'&rarr; '+esc(x)).join('<br>')+(t.steps.length?'<br>':'')+'<b>decision:</b> '+esc(t.final)+'</div>').join('')||'<div class="note">The agent\'s tool calls show up here.</div>';}
async function load(){try{render(await (await fetch('/api/state')).json())}catch(e){}}
load();setInterval(load,700);addEventListener('resize',()=>S&&draw(S.curve));
</script></body></html>"""

if __name__ == "__main__":
    main()
