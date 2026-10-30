"""The dashboard and the kill switch must stay responsive while a slow LLM is thinking."""
import json
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from agentx.agent import Agent
from agentx.dashboard import App, Handler
from agentx.stub_llm import StubLLM


class SlowLLM(StubLLM):
    name = "slow"

    def chat(self, messages):
        time.sleep(0.4)
        return super().chat(messages)


def call(port, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method="POST" if body is not None else "GET")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=10) as r:
        out = json.loads(r.read())
    return out, time.time() - t0


class DashboardTests(unittest.TestCase):
    def test_state_and_kill_switch_never_wait_on_the_llm(self):
        app = App("stub", days=200)
        app.agent = Agent(SlowLLM())
        app.session.agent = app.agent
        app.session.last_select = None  # force a strategy review (many slow LLM calls) on the next step
        Handler.app = app
        threading.Thread(target=app.loop, daemon=True).start()
        srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            _, t_step = call(port, "/api/step", {"n": 1})
            self.assertLess(t_step, 0.5, "stepping must return immediately")
            time.sleep(0.6)  # the worker is now inside the slow LLM loop
            self.assertTrue(app.busy)
            st, t_state = call(port, "/api/state")
            self.assertLess(t_state, 0.5, "state must not wait for the LLM")
            self.assertTrue(st["busy"])
            _, t_kill = call(port, "/api/kill", {"flatten": False})
            self.assertLess(t_kill, 0.5, "kill switch must not wait for the LLM")
            st, _ = call(port, "/api/state")
            self.assertTrue(st["risk"]["kill"], "kill switch should show as engaged immediately")
            deadline = time.time() + 20
            while app.busy or app.queued:
                self.assertLess(time.time(), deadline, "step never finished")
                time.sleep(0.1)
            st, _ = call(port, "/api/state")
            self.assertGreaterEqual(st["day"], 2)
        finally:
            srv.shutdown()
            app.kill.release()


if __name__ == "__main__":
    unittest.main()
