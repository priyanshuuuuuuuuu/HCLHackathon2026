"""One function to call the chosen LLM (LLM_PROVIDER / LLM_MODEL in .env).
Anthropic has its own API; OpenAI, Gemini, Groq and Ollama all accept the OpenAI chat format."""

import os

import httpx

from app import config

OPENAI_STYLE_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    "ollama": os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/v1/chat/completions",
}
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY",
           "groq": "GROQ_API_KEY", "ollama": None}


class LLMError(Exception):
    pass


def status() -> dict:
    if config.MOCK_LLM:
        return {"provider": "mock", "model": None, "ready": True}
    key = KEY_ENV.get(config.LLM_PROVIDER)
    return {"provider": config.LLM_PROVIDER, "model": config.LLM_MODEL, "ready": key is None or bool(os.getenv(key))}


def call(system: str, user: str, max_tokens: int = 1500) -> tuple[str, int]:
    """Returns (text, tokens used). Raises LLMError so callers can fall back."""
    if config.MOCK_LLM:
        raise LLMError("MOCK_LLM is on")
    try:
        if config.LLM_PROVIDER == "anthropic":
            r = httpx.post("https://api.anthropic.com/v1/messages", timeout=60,
                           headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                           json={"model": config.LLM_MODEL, "max_tokens": max_tokens, "system": system,
                                 "messages": [{"role": "user", "content": user}]})
            r.raise_for_status()
            d = r.json()
            return d["content"][0]["text"], d["usage"]["input_tokens"] + d["usage"]["output_tokens"]
        key = os.getenv(KEY_ENV[config.LLM_PROVIDER] or "", "ollama")
        r = httpx.post(OPENAI_STYLE_URLS[config.LLM_PROVIDER], timeout=180,
                       headers={"Authorization": f"Bearer {key}"},
                       json={"model": config.LLM_MODEL, "max_tokens": max_tokens,
                             "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        d = r.json()
        return d["choices"][0]["message"]["content"], d.get("usage", {}).get("total_tokens", 0)
    except (httpx.HTTPError, KeyError) as e:
        raise LLMError(str(e)) from e
