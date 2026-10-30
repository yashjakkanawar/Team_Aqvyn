"""Exercise the real HTTP client + JSON tool protocol against a local mock OpenAI-compatible gateway.

This stands in for your localhost:8081 gateway when it can't be reached (e.g. in CI or the cloud).
"""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agentx.agent import Agent
from agentx.config import LLMConfig
from agentx.data import synthetic_market
from agentx.llm import HttpLLM, LLMError
from agentx.session import TradingSession
from agentx.stub_llm import StubLLM


class MockGateway(BaseHTTPRequestHandler):
    wbufsize = 65536  # send headers+body in one packet (avoids ~200 ms Nagle/delayed-ACK stalls on Windows)
    seen_auth = []
    mode = "ok"  # ok | fence | 500

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._json(200, {"data": [{"id": "mock-model"}]})
        else:
            self._json(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n))
        MockGateway.seen_auth.append(self.headers.get("Authorization"))
        if MockGateway.mode == "500":
            return self._json(500, {"error": "boom"})
        text = StubLLM().chat(body["messages"])
        if MockGateway.mode == "fence":  # chatty models wrap JSON in fences and prose
            text = "Sure, here is my answer:\n```json\n" + text + "\n```\nHope that helps!"
        self._json(200, {"choices": [{"message": {"role": "assistant", "content": text}}]})


class HttpLLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), MockGateway)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def llm(self):
        return HttpLLM(LLMConfig(f"http://127.0.0.1:{self.port}/v1", "test-key", "mock-model", timeout=5))

    def setUp(self):
        MockGateway.mode, MockGateway.seen_auth = "ok", []

    def test_models_and_auth_header(self):
        llm = self.llm()
        self.assertEqual(llm.list_models(), ["mock-model"])
        llm.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(MockGateway.seen_auth[-1], "Bearer test-key")

    def test_session_through_http_matches_direct_stub(self):
        market = synthetic_market(days=200, seed=7)
        direct = TradingSession(market, agent=Agent(StubLLM()), start_idx=60).run().report()
        via_http = TradingSession(market, agent=Agent(self.llm()), start_idx=60).run().report()
        self.assertEqual(direct["agent"], via_http["agent"])
        self.assertEqual(via_http["stats"]["agent_fallbacks"], 0)

    def test_fenced_chatty_json_is_parsed(self):
        MockGateway.mode = "fence"
        market = synthetic_market(days=200, seed=7)
        rep = TradingSession(market, agent=Agent(self.llm()), start_idx=60).run().report()
        self.assertEqual(rep["stats"]["agent_fallbacks"], 0)

    def test_server_errors_raise_llmerror_and_session_survives(self):
        MockGateway.mode = "500"
        with self.assertRaises(LLMError):
            self.llm().chat([{"role": "user", "content": "x"}])
        market = synthetic_market(days=150, seed=7)
        s = TradingSession(market, agent=Agent(self.llm()), strategy="breakout", start_idx=60).run()
        self.assertTrue(s.done)
        self.assertGreater(s.stats["agent_fallbacks"], 0)

    def test_unreachable_gateway(self):
        dead = HttpLLM(LLMConfig("http://127.0.0.1:1/v1", "k", "m", timeout=2))
        with self.assertRaises(LLMError):
            dead.chat([{"role": "user", "content": "x"}])


if __name__ == "__main__":
    unittest.main()
