"""Model providers: one interface for every LLM vendor, used by judges and
test generation. A provider is chosen in the judge config:

  judge:
    provider: azure_openai            # default: openai_compatible
    model: gpt-4.1-mini               # deployment / model id
    azure_endpoint: https://my-res.openai.azure.com
    auth: entra_id                    # or api_key: ${secret:aoai-key}
"""
from __future__ import annotations

from dataclasses import dataclass

from ..plugins import HealthStatus, Plugin, timed_health


@dataclass
class Completion:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ModelProvider(Plugin):
    """Chat completion. messages are OpenAI-style [{"role", "content"}]."""

    def complete(self, messages: list[dict], model: str, temperature: float = 0.0,
                 max_tokens: int | None = None) -> Completion:  # pragma: no cover - abstract
        raise NotImplementedError

    #: model used by health_check when none is configured
    default_model: str | None = None

    def health_check(self) -> HealthStatus:
        if not self.default_model:
            return HealthStatus(True, "configured (no model given for a live probe)")
        return timed_health(lambda: f"model replied: {self.complete([{'role': 'user', 'content': 'Reply OK'}], self.default_model, max_tokens=5).text[:20]!r}")


def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Separate system messages (Anthropic/Bedrock/Vertex take them apart)."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    return system, [m for m in messages if m["role"] != "system"]


def build_provider(cfg: dict) -> ModelProvider:
    from ..plugins import REGISTRY
    return REGISTRY.build("model", cfg)


__all__ = ["Completion", "ModelProvider", "build_provider", "split_system"]
