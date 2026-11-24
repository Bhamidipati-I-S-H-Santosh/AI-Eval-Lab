"""One small wrapper around any OpenAI-compatible API.

Works with free options by changing three environment variables:
  Groq:    LLM_BASE_URL=https://api.groq.com/openai/v1
  Gemini:  LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
  Ollama:  LLM_BASE_URL=http://localhost:11434/v1   (LLM_API_KEY=ollama)
"""
from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=os.environ["LLM_API_KEY"],
            max_retries=6,  # SDK retries 429/5xx itself; covers the agent loop too
        )
    return _client


def chat(prompt: str, system: str = "", model: str | None = None, retries: int = 5) -> str:
    model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    for attempt in range(retries):
        try:
            resp = client().chat.completions.create(model=model, messages=messages, temperature=0)
            return (resp.choices[0].message.content or "").strip()
        except Exception:  # free tiers rate-limit often; back off and retry
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 2 ** attempt * 10))  # per-minute quotas need ~60s
    return ""
