"""A tiny fake 'company chatbot' with an HTTP API, so you can try the HTTP
adapter without deploying anything. No LLM: it returns the best-matching
passage. Its JSON shape is deliberately different from evalkit's, to show
that the adapter maps fields by config.

  python examples/mock_http_agent.py          # serves on http://localhost:8765/chat
  python -m evalkit run examples/http_agent.yaml
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evalkit.adapters.folder import FolderRetrieverAdapter  # noqa: E402

DOCS = FolderRetrieverAdapter(str(Path(__file__).resolve().parent.parent / "data" / "docs"))


def answer(message: str) -> dict:
    hits = DOCS.search(message, k=3)
    best = hits[0] if hits and (hits[0].score or 0) > 1.0 else None
    reply = f"According to {best.metadata['source']}: {best.text[:300]}" if best else "I don't know."
    return {
        "reply": {"text": reply},
        "debug": {"tool_calls": [{"name": "search", "arguments": {"q": message},
                                  "output": json.dumps([h.id for h in hits])}],
                  "sources": [{"doc_id": h.metadata["source"], "content": h.text} for h in hits]},
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/chat":
            self.send_error(404); return
        if self.headers.get("Authorization") != "Bearer demo-key":
            self.send_error(401, "missing or wrong API key"); return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or "{}")
        data = json.dumps(answer(body.get("message", ""))).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def make_server(port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"Mock agent on http://localhost:{port}/chat (API key: demo-key)")
    make_server(port).serve_forever()
