"""AWS Bedrock via the Converse API (works for Claude, Llama, Mistral, Nova,
Titan... on Bedrock). Auth: the standard AWS credential chain.
Needs: pip install 'evalkit[aws]'."""
from __future__ import annotations

from . import Completion, ModelProvider, split_system


class BedrockProvider(ModelProvider):
    def __init__(self, region: str | None = None, profile: str | None = None, default_model: str | None = None,
                 client=None, **_):
        if client is None:
            import boto3
            from botocore.config import Config
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            client = session.client("bedrock-runtime", region_name=region,
                                    config=Config(retries={"max_attempts": 8, "mode": "adaptive"}))
        self.client, self.default_model = client, default_model

    def complete(self, messages, model, temperature=0.0, max_tokens=None):
        system, msgs = split_system(messages)
        inference = {"temperature": temperature}
        if max_tokens:
            inference["maxTokens"] = max_tokens
        kw = {"system": [{"text": system}]} if system else {}
        resp = self.client.converse(
            modelId=model, inferenceConfig=inference, **kw,
            messages=[{"role": m["role"], "content": [{"text": m["content"]}]} for m in msgs])
        parts = resp["output"]["message"]["content"]
        u = resp.get("usage") or {}
        return Completion("".join(p.get("text", "") for p in parts).strip(),
                          u.get("inputTokens"), u.get("outputTokens"))
