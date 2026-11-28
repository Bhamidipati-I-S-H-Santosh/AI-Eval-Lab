"""Google Vertex AI (Gemini and Model Garden models) through the google-genai
SDK in Vertex mode. Auth: Application Default Credentials / service account.
Needs: pip install 'evalkit[gcp]'."""
from __future__ import annotations

from . import Completion, ModelProvider, split_system


class VertexProvider(ModelProvider):
    def __init__(self, project: str | None = None, location: str = "us-central1",
                 default_model: str | None = None, client=None, **_):
        if client is None:
            from google import genai
            client = genai.Client(vertexai=True, project=project, location=location)
        self.client, self.default_model = client, default_model

    def complete(self, messages, model, temperature=0.0, max_tokens=None):
        system, msgs = split_system(messages)
        contents = [{"role": "model" if m["role"] == "assistant" else "user",
                     "parts": [{"text": m["content"]}]} for m in msgs]
        config = {"temperature": temperature}
        if max_tokens:
            config["max_output_tokens"] = max_tokens
        if system:
            config["system_instruction"] = system
        resp = self.client.models.generate_content(model=model, contents=contents, config=config)
        u = getattr(resp, "usage_metadata", None)
        return Completion((getattr(resp, "text", "") or "").strip(),
                          getattr(u, "prompt_token_count", None), getattr(u, "candidates_token_count", None))
