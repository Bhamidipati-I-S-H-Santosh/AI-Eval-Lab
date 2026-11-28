"""Anthropic Messages API (Claude models). Needs: pip install 'evalkit[anthropic]'."""
from __future__ import annotations

from . import Completion, ModelProvider, split_system


class AnthropicProvider(ModelProvider):
    def __init__(self, api_key: str | None = None, base_url: str | None = None, max_retries: int = 6,
                 timeout: float = 120, default_max_tokens: int = 1024, default_model: str | None = None,
                 client=None, **_):
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key or None, base_url=base_url or None,
                                         max_retries=max_retries, timeout=timeout)
        self.client, self.default_max_tokens, self.default_model = client, default_max_tokens, default_model

    def complete(self, messages, model, temperature=0.0, max_tokens=None):
        system, msgs = split_system(messages)
        kw = {"system": system} if system else {}
        resp = self.client.messages.create(model=model, messages=msgs, temperature=temperature,
                                           max_tokens=max_tokens or self.default_max_tokens, **kw)
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "text") == "text")
        u = getattr(resp, "usage", None)
        return Completion(text.strip(), getattr(u, "input_tokens", None), getattr(u, "output_tokens", None))
