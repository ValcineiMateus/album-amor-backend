"""Cliente de IA direto (sem Emergent): Gemini, Groq, Claude (Anthropic) e ChatGPT (OpenAI).

Mantém a mesma interface que o server.py já usava:
    LlmChat(session_id=..., system_message=...).with_model(provider, model)
    async for ev in chat.stream_message(UserMessage(text=...)):  # TextDelta / StreamDone

Variáveis de ambiente (use as que tiver; basta UMA):
    GEMINI_API_KEY     -> Google Gemini (tem plano gratuito)
    GROQ_API_KEY       -> Groq (tem plano gratuito)
    ANTHROPIC_API_KEY  -> Claude (pago)
    OPENAI_API_KEY     -> ChatGPT (pago)
    GEMINI_MODEL       -> opcional, padrão "gemini-2.5-flash"
    GROQ_MODEL         -> opcional, padrão "llama-3.3-70b-versatile"
Cada botão do app usa o provedor dele se houver chave;
se não houver, cai no primeiro provedor que tiver chave (ordem: Gemini, Groq, Claude, ChatGPT).
"""
import json
import logging
import os
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import httpx

logger = logging.getLogger("album.llm")

GEMINI_BASE = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
GROQ_BASE = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
ANTHROPIC_BASE = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
OPENAI_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
DEFAULT_MODELS = {
    "gemini": os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    "groq": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-5.6-terra",
}
FALLBACK_ORDER = ("groq", "gemini", "anthropic", "openai")
MAX_TOKENS = 1500


@dataclass
class UserMessage:
    text: str


@dataclass
class TextDelta:
    content: str


class StreamDone:
    pass


class LLMError(Exception):
    """code: 'sem_credito' (conta sem saldo), 'limite_atingido' (cota/rate limit) ou 'indisponivel'."""

    def __init__(self, message: str, code: str = "indisponivel"):
        super().__init__(message)
        self.code = code


def _keys() -> dict:
    return {
        "gemini": os.environ.get("GEMINI_API_KEY", "").strip(),
        "groq": os.environ.get("GROQ_API_KEY", "").strip(),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", "").strip(),
        "openai": os.environ.get("OPENAI_API_KEY", "").strip(),
    }


def _code_for(provider: str, status: int, body: str) -> str:
    b = body.lower()
    if status == 402:
        return "sem_credito"
    if status == 429:
        return "limite_atingido"  # cota gratuita (Groq/Gemini) esgotada momentaneamente: é só esperar
    if provider in ("anthropic", "openai") and any(
        k in b for k in ("credit balance", "insufficient_quota", "billing", "exceeded your current quota")
    ):
        return "sem_credito"
    return "indisponivel"


async def _sse_data(resp: httpx.Response) -> AsyncIterator[dict]:
    async for line in resp.aiter_lines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


class LlmChat:
    def __init__(self, api_key: Optional[str] = None, session_id: str = "", system_message: str = ""):
        self.session_id = session_id
        self.system = system_message
        self.provider = "gemini"
        self.model = DEFAULT_MODELS["gemini"]

    def with_model(self, provider: str, model: str) -> "LlmChat":
        keys = _keys()
        if keys.get(provider):
            self.provider, self.model = provider, model
        else:  # sem chave do provedor pedido: usa o primeiro que tiver chave
            for p in FALLBACK_ORDER:
                if keys[p]:
                    self.provider, self.model = p, DEFAULT_MODELS[p]
                    break
            else:
                self.provider, self.model = provider, model  # sem nenhuma chave: falha em stream_message
        return self

    async def stream_message(self, msg: UserMessage) -> AsyncIterator[object]:
        key = _keys().get(self.provider, "")
        if not key:
            raise LLMError("nenhuma chave de IA configurada (GEMINI_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY ou OPENAI_API_KEY)")
        timeout = httpx.Timeout(30.0, read=120.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            if self.provider == "gemini":
                gen = self._gemini(client, key, msg.text)
            elif self.provider == "anthropic":
                gen = self._anthropic(client, key, msg.text)
            elif self.provider == "groq":
                gen = self._openai(client, key, msg.text, base=GROQ_BASE)
            else:
                gen = self._openai(client, key, msg.text)
            async for chunk in gen:
                if chunk:
                    yield TextDelta(chunk)
        yield StreamDone()

    async def _check(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        body = (await resp.aread()).decode("utf-8", "replace")
        logger.error("IA %s respondeu %s: %s", self.provider, resp.status_code, body[:400])
        raise LLMError(f"{self.provider} HTTP {resp.status_code}", _code_for(self.provider, resp.status_code, body))

    async def _gemini(self, client, key, text):
        url = f"{GEMINI_BASE}/v1beta/models/{self.model}:streamGenerateContent?alt=sse"
        body = {
            "systemInstruction": {"parts": [{"text": self.system}]},
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"temperature": 0.8, "maxOutputTokens": 4096},
        }
        async with client.stream("POST", url, headers={"x-goog-api-key": key}, json=body) as r:
            await self._check(r)
            async for ev in _sse_data(r):
                if ev.get("promptFeedback", {}).get("blockReason"):
                    raise LLMError("gemini bloqueou o conteúdo")
                for cand in ev.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        if part.get("text") and not part.get("thought"):
                            yield part["text"]

    async def _anthropic(self, client, key, text):
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        body = {
            "model": self.model, "max_tokens": MAX_TOKENS, "stream": True,
            "system": self.system, "messages": [{"role": "user", "content": text}],
        }
        async with client.stream("POST", f"{ANTHROPIC_BASE}/v1/messages", headers=headers, json=body) as r:
            await self._check(r)
            async for ev in _sse_data(r):
                if ev.get("type") == "content_block_delta" and ev.get("delta", {}).get("type") == "text_delta":
                    yield ev["delta"].get("text", "")
                elif ev.get("type") == "error":
                    raise LLMError("anthropic erro no stream")

    async def _openai(self, client, key, text, base=None):
        headers = {"Authorization": f"Bearer {key}"}
        body = {
            "model": self.model, "stream": True,
            "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": text}],
        }
        base_url = base or OPENAI_BASE
        async with client.stream("POST", f"{base_url}/v1/chat/completions", headers=headers, json=body) as r:
            await self._check(r)
            async for ev in _sse_data(r):
                for ch in ev.get("choices", []):
                    c = ch.get("delta", {}).get("content")
                    if c:
                        yield c
