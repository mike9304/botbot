"""Multi-provider LLM client for the AI Teacher system.

Supports multiple API providers with a unified interface:
- Anthropic Claude (Sonnet 4.5, Haiku 4.5)
- OpenAI (GPT-4o, GPT-4o-mini)
- Google Gemini (2.5 Flash, 2.5 Pro)
- DeepSeek (V3, R1)
- Groq (Llama 3.3 70B — free/cheap tier)

Each provider has different cost/quality tradeoffs.
The system defaults to the cheapest option and can be configured.

Cost estimates (per million tokens, as of 2025):
┌──────────────────────┬─────────┬──────────┐
│ Model                │ Input   │ Output   │
├──────────────────────┼─────────┼──────────┤
│ Claude Haiku 4.5     │ $0.80   │ $4.00    │
│ Claude Sonnet 4.5    │ $3.00   │ $15.00   │
│ GPT-4o-mini          │ $0.15   │ $0.60    │
│ GPT-4o               │ $2.50   │ $10.00   │
│ Gemini 2.5 Flash     │ $0.15   │ $0.60    │
│ Gemini 2.5 Pro       │ $1.25   │ $10.00   │
│ DeepSeek V3          │ $0.27   │ $1.10    │
│ DeepSeek R1          │ $0.55   │ $2.19    │
│ Groq Llama 3.3 70B   │ $0.06   │ $0.06    │
└──────────────────────┴─────────┴──────────┘
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    OFFLINE = "offline"  # Rule-based fallback (no API needed)


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""
    provider: LLMProvider = LLMProvider.OFFLINE
    model: str = ""  # Auto-selected based on provider if empty
    api_key: str = ""  # From env var if empty
    temperature: float = 0.3  # Low for consistent evaluations
    max_tokens: int = 500
    timeout: float = 30.0

    # Cost tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    # Rate limiting
    max_calls_per_minute: int = 30
    max_cost_per_day_usd: float = 5.0  # Safety cap

    def __post_init__(self) -> None:
        if not self.model:
            self.model = DEFAULT_MODELS.get(self.provider, "")
        if not self.api_key:
            self.api_key = self._get_api_key_from_env()

    def _get_api_key_from_env(self) -> str:
        env_vars = {
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.GOOGLE: "GOOGLE_API_KEY",
            LLMProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
            LLMProvider.GROQ: "GROQ_API_KEY",
        }
        var = env_vars.get(self.provider, "")
        return os.environ.get(var, "")


DEFAULT_MODELS = {
    LLMProvider.ANTHROPIC: "claude-haiku-4-5-20251001",
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.GOOGLE: "gemini-2.5-flash",
    LLMProvider.DEEPSEEK: "deepseek-chat",
    LLMProvider.GROQ: "llama-3.3-70b-versatile",
    LLMProvider.OFFLINE: "",
}

# Cost per million tokens (input, output)
TOKEN_COSTS: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.00),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    "llama-3.3-70b-versatile": (0.06, 0.06),
}

# OpenAI-compatible provider endpoints (DeepSeek, Groq use the same API format)
_OPENAI_COMPATIBLE_ENDPOINTS: dict[LLMProvider, str] = {
    LLMProvider.OPENAI: "https://api.openai.com/v1/chat/completions",
    LLMProvider.DEEPSEEK: "https://api.deepseek.com/v1/chat/completions",
    LLMProvider.GROQ: "https://api.groq.com/openai/v1/chat/completions",
}


@dataclass
class LLMResponse:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    provider: str = ""


class LLMClient:
    """Unified LLM client supporting multiple providers.

    Usage:
        client = LLMClient(LLMConfig(provider=LLMProvider.DEEPSEEK))
        response = await client.chat("Evaluate this agent's performance...")
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._call_count = 0
        self._daily_cost = 0.0
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def chat(self, prompt: str, system: str = "") -> LLMResponse:
        """Send a chat message to the configured LLM provider."""
        if self.config.provider == LLMProvider.OFFLINE:
            return LLMResponse(content="[OFFLINE MODE]", model="offline", provider="offline")

        # Cost safety check
        if self._daily_cost >= self.config.max_cost_per_day_usd:
            logger.warning(
                f"Daily cost limit reached (${self._daily_cost:.2f} >= "
                f"${self.config.max_cost_per_day_usd:.2f}). Falling back to offline."
            )
            return LLMResponse(
                content="[COST LIMIT REACHED - OFFLINE MODE]",
                model="offline", provider="offline",
            )

        # API key validation
        if not self.config.api_key:
            logger.warning(f"No API key for {self.config.provider.value}, falling back to offline.")
            return LLMResponse(
                content="[NO API KEY - OFFLINE MODE]",
                model="offline", provider="offline",
            )

        try:
            if self.config.provider == LLMProvider.ANTHROPIC:
                response = await self._call_anthropic(prompt, system)
            elif self.config.provider == LLMProvider.GOOGLE:
                response = await self._call_google(prompt, system)
            elif self.config.provider in _OPENAI_COMPATIBLE_ENDPOINTS:
                response = await self._call_openai_compatible(prompt, system)
            else:
                return LLMResponse(content="[UNKNOWN PROVIDER]")

            # Track costs
            self._update_costs(response)
            return response

        except aiohttp.ClientError as e:
            logger.error(f"Network error calling {self.config.provider}: {e}")
            return LLMResponse(content=f"[NETWORK ERROR: {e}]", model=self.config.model)
        except TimeoutError as e:
            logger.error(f"Timeout calling {self.config.provider}: {e}")
            return LLMResponse(content=f"[TIMEOUT ERROR: {e}]", model=self.config.model)
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected API response from {self.config.provider}: {e}")
            return LLMResponse(content=f"[PARSE ERROR: {e}]", model=self.config.model)
        except RuntimeError as e:
            logger.error(f"API error from {self.config.provider}: {e}")
            return LLMResponse(content=f"[API ERROR: {e}]", model=self.config.model)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def _call_anthropic(self, prompt: str, system: str) -> LLMResponse:
        """Call Anthropic Claude API."""
        session = await self._get_session()
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        async with session.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
        ) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise RuntimeError(f"Anthropic API error {resp.status}: {data}")
            return LLMResponse(
                content=data["content"][0]["text"],
                input_tokens=data["usage"]["input_tokens"],
                output_tokens=data["usage"]["output_tokens"],
                model=self.config.model,
                provider="anthropic",
            )

    async def _call_openai_compatible(self, prompt: str, system: str) -> LLMResponse:
        """Call any OpenAI-compatible API (OpenAI, DeepSeek, Groq)."""
        endpoint = _OPENAI_COMPATIBLE_ENDPOINTS[self.config.provider]
        provider_name = self.config.provider.value

        session = await self._get_session()
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        async with session.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
        ) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise RuntimeError(f"{provider_name} API error {resp.status}: {data}")
            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                input_tokens=data["usage"].get("prompt_tokens", 0),
                output_tokens=data["usage"].get("completion_tokens", 0),
                model=self.config.model,
                provider=provider_name,
            )

    async def _call_google(self, prompt: str, system: str) -> LLMResponse:
        """Call Google Gemini API."""
        session = await self._get_session()
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.model}:generateContent?key={self.config.api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        async with session.post(
            url, json=payload,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
        ) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise RuntimeError(f"Gemini API error {resp.status}: {data}")
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            return LLMResponse(
                content=content,
                input_tokens=usage.get("promptTokenCount", 0),
                output_tokens=usage.get("candidatesTokenCount", 0),
                model=self.config.model,
                provider="google",
            )

    def _update_costs(self, response: LLMResponse) -> None:
        """Track token usage and costs."""
        costs = TOKEN_COSTS.get(response.model, (0, 0))
        input_cost = response.input_tokens * costs[0] / 1_000_000
        output_cost = response.output_tokens * costs[1] / 1_000_000
        response.cost_usd = input_cost + output_cost

        self.config.total_input_tokens += response.input_tokens
        self.config.total_output_tokens += response.output_tokens
        self.config.total_cost_usd += response.cost_usd
        self._daily_cost += response.cost_usd
        self._call_count += 1

    def get_cost_report(self) -> dict:
        """Get a summary of API usage and costs."""
        return {
            "provider": self.config.provider.value,
            "model": self.config.model,
            "total_calls": self._call_count,
            "total_input_tokens": self.config.total_input_tokens,
            "total_output_tokens": self.config.total_output_tokens,
            "total_cost_usd": round(self.config.total_cost_usd, 4),
            "daily_cost_usd": round(self._daily_cost, 4),
            "daily_limit_usd": self.config.max_cost_per_day_usd,
        }

    async def close(self) -> None:
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None
