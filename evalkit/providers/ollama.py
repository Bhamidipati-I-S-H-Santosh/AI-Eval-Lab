"""Local models through Ollama's native REST API (no extra package; nothing
leaves the machine)."""
from __future__ import annotations

import requests

from ..plugins import HealthStatus, timed_health
from . import Completion, ModelProvider


class OllamaProvider(ModelProvider):
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 300,
                 default_model: str | None = None, think: bool | None = None, session=None, **_):
        self.base_url, self.timeout = base_url.rstrip("/"), timeout
        self.default_model, self.think = default_model, think
        self.session = session or requests.Session()

    def complete(self, messages, model, temperature=0.0, max_tokens=None):
        options = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens
        body = {"model": model, "messages": messages, "stream": False, "options": options}
        if self.think is not None:
            body["think"] = self.think
        r = self.session.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return Completion(((data.get("message") or {}).get("content") or "").strip(),
                          data.get("prompt_eval_count"), data.get("eval_count"))

    def health_check(self) -> HealthStatus:
        def probe():
            r = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            r.raise_for_status()
            names = [m["name"] for m in r.json().get("models", [])]
            if self.default_model and self.default_model not in names:
                raise RuntimeError(f"model '{self.default_model}' not pulled; available: {names}")
            return f"{len(names)} model(s) available"
        return timed_health(probe)

    def close(self):
        self.session.close()
