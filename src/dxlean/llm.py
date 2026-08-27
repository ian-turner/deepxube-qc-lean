"""Minimal OpenAI-compatible chat client.

Works against any /v1/chat/completions server: ollama and LM Studio and
mlx_lm.server locally, vLLM on the CUDA server, or a hosted API. The rest of
dxlean only sees `chat(system, user) -> str`, so backends swap freely.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import httpx


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None,
                 timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("DXLEAN_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self._client = httpx.Client(timeout=timeout)
        self.n_calls = 0

    def chat(self, system: str, user: str, temperature: float = 0.7,
             max_tokens: int = 512) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        self.n_calls += 1
        resp = self._client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class FakeChatClient:
    """Deterministic stub for tests: `respond(system, user) -> str`."""

    def __init__(self, respond: Callable[[str, str], str]):
        self._respond = respond
        self.n_calls = 0

    def chat(self, system: str, user: str, temperature: float = 0.7,
             max_tokens: int = 512) -> str:
        self.n_calls += 1
        return self._respond(system, user)
