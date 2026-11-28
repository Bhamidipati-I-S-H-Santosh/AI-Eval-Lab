"""Azure OpenAI. Auth with a key, or keyless with Microsoft Entra ID
(managed identity / workload identity / az login). Data stays in your Azure
tenant and region. `model` is your deployment name."""
from __future__ import annotations

from ..errors import ConfigError
from . import Completion, ModelProvider

_SCOPE = "https://cognitiveservices.azure.com/.default"


class AzureOpenAIProvider(ModelProvider):
    def __init__(self, azure_endpoint: str, api_version: str = "2024-10-21", api_key: str | None = None,
                 auth: str | dict | None = None, max_retries: int = 6, timeout: float = 120,
                 default_model: str | None = None, client=None, **_):
        if client is None:
            from openai import AzureOpenAI
            kw = dict(azure_endpoint=azure_endpoint, api_version=api_version,
                      max_retries=max_retries, timeout=timeout)
            if api_key:
                kw["api_key"] = api_key
            elif auth in (None, "entra_id") or (isinstance(auth, dict) and auth.get("type") == "entra_id"):
                from ..auth import EntraIdAuth
                scope = auth.get("scope", _SCOPE) if isinstance(auth, dict) else _SCOPE
                kw["azure_ad_token_provider"] = EntraIdAuth(scope=scope).token
            else:
                raise ConfigError("azure_openai: set api_key, or auth: entra_id for keyless access")
            client = AzureOpenAI(**kw)
        self.client, self.default_model = client, default_model

    def complete(self, messages, model, temperature=0.0, max_tokens=None):
        kw = {"max_tokens": max_tokens} if max_tokens else {}
        resp = self.client.chat.completions.create(model=model, messages=messages,
                                                   temperature=temperature, **kw)
        u = getattr(resp, "usage", None)
        return Completion((resp.choices[0].message.content or "").strip(),
                          getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None))
