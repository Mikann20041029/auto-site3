from __future__ import annotations
import os
import requests
from typing import Any

DEEPSEEK_BASE = "https://api.deepseek.com"

class DeepSeekClient:
    def __init__(self, api_key: str | None = None, base_url: str = DEEPSEEK_BASE) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url.rstrip("/")

    def chat(self, model: str, messages: list[dict[str, Any]], temperature: float = 0.85, max_tokens: int = 2200) -> str:
        if not self.api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY (set GitHub Secrets: DEEPSEEK_API_KEY)")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens)
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()
