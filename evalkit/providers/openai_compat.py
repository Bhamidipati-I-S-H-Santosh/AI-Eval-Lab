"""Any OpenAI-compatible chat endpoint: OpenAI, Groq, Together, vLLM, LM Studio,
LiteLLM proxy, Ollama's /v1, Gemini's OpenAI endpoint."""
from __future__ import annotations

from ..auth import build_auth
from . import Completion, ModelProvider


class OpenAICompatibleProvider(ModelProvider):
    def __init__(self, base_url: str | None = None, api_key: str | None = None, auth: dict | None = None,
                 max_retries: int = 6, timeout: float = 120, default_model: str | None = None,
                 client=None, **_):
        if client is None:
            from openai import OpenAI
            authp = build_auth(auth)
            headers = authp.headers() if authp else None
            client = OpenAI(base_url=base_url or None, api_key=api_key or "none",
                            max_retries=max_retries, timeout=timeout, default_headers=headers)
        self.client, self.default_model = client, default_model

    def complete(self, messages, model, temperature=0.0, max_tokens=None):
        kw = {"max_tokens": max_tokens} if max_tokens else {}
        resp = self.client.chat.completions.create(model=model, messages=messages,
                                                   temperature=temperature, **kw)
        u = getattr(resp, "usage", None)
        return Completion((resp.choices[0].message.content or "").strip(),
                          getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None))

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass
