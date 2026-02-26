"""Unified LLM client supporting local Ollama and Claude API.

Low temperature (0.3) for deterministic editing decisions.
Retry logic: up to 3 attempts if JSON parsing fails.
On failure: returns None so the fallback engine can take over.
"""

from __future__ import annotations

import json
import logging
import time

import httpx

logger = logging.getLogger("baby_editor.brain")


class LLMClient:
    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2.5:32b",
        config: dict | None = None,
    ):
        self.provider = provider
        self.model = model
        self.config = config or {}
        self._anthropic = None

        if provider == "claude":
            from anthropic import Anthropic

            self._anthropic = Anthropic()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> dict | None:
        """Call the LLM and parse the JSON response.

        Returns parsed dict on success, None after 3 failed attempts.
        """
        backoff = self.config.get("backoff_seconds", 2)

        for attempt in range(1, 4):
            try:
                raw = self._call_llm(system_prompt, user_prompt, max_tokens, temperature)
                clean = raw.strip()
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[1]
                    clean = clean.rsplit("```", 1)[0]
                parsed = json.loads(clean)
                logger.info("LLM response parsed OK (attempt %d)", attempt)
                return parsed
            except json.JSONDecodeError as exc:
                logger.warning("JSON parse failed (attempt %d): %s", attempt, exc)
                time.sleep(backoff)
            except Exception as exc:
                logger.error("LLM call failed (attempt %d): %s", attempt, exc)
                time.sleep(backoff)

        logger.error("All 3 LLM attempts failed. Returning None for fallback.")
        return None

    def _call_llm(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if self.provider == "ollama":
            base_url = self.config.get("base_url", "http://localhost:11434")
            timeout = self.config.get("timeout", 120)
            resp = httpx.post(
                f"{base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": user,
                    "system": system,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()["response"]

        if self.provider == "claude":
            assert self._anthropic is not None
            msg = self._anthropic.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text

        raise ValueError(f"Unknown LLM provider: {self.provider}")
