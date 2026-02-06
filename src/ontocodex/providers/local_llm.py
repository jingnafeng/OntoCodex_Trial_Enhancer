from __future__ import annotations

import os
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from dotenv import find_dotenv, load_dotenv


class LocalLLMError(Exception):
    """Raised when the Local LLM provider fails."""
    pass


@dataclass
class LocalLLMConfig:
    """Configuration for LocalLLM."""
    base_url: str
    model: str
    api_type: str = "ollama"
    api_key: Optional[str] = None
    timeout: int = 120


class LocalLLM:
    """
    A simple client for a local LLM server (e.g., Ollama, LM Studio, LocalAI).
    Defaults to Ollama's API format.
    """
    def __init__(self, base_url: str, model: str, api_type: str = "ollama", api_key: Optional[str] = None, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_type = api_type
        self.api_key = api_key
        self.timeout = int(timeout) if timeout else 120

    @classmethod
    def from_options(cls, options: Dict[str, Any], prefix: str = "llm") -> LocalLLM:
        """
        Initialize from a configuration dictionary.
        
        Keys looked up (e.g. with prefix="decision"):
          - decision_base_url (default: http://localhost:11434)
          - decision_model (default: llama3)
          - decision_api_type (default: ollama)
        
        Fallbacks to generic "llm_*" keys if specific prefix not found.
        """
        # Load .env from current working tree when available.
        load_dotenv(find_dotenv(usecwd=True), override=False)

        # 1. Try specific prefix
        base_url = options.get(f"{prefix}_base_url")
        model = options.get(f"{prefix}_model")
        api_type = options.get(f"{prefix}_api_type")
        api_key = options.get(f"{prefix}_api_key")
        timeout = options.get(f"{prefix}_timeout")

        # 2. Fallback to generic 'llm_'
        if not base_url:
            base_url = options.get("llm_base_url", "http://localhost:11434")
        if not model:
            model = options.get("llm_model", "llama3")
        if not api_type:
            api_type = options.get("llm_api_type", "ollama")
        if not api_key:
            api_key = (
                options.get("llm_api_key")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("GENAI_GARDEN_KEY")
                or os.getenv("GEMINI_API_KEY")
            )
        if not timeout:
            timeout = options.get("llm_timeout") or os.getenv("ONTOCODEX_LLM_TIMEOUT") or 120

        if api_type == "gemini":
            if not base_url or base_url == "http://localhost:11434":
                base_url = "https://generativelanguage.googleapis.com"
            if not model or model == "llama3":
                model = "gemini-1.5-flash"

        return cls(base_url=base_url, model=model, api_type=api_type, api_key=api_key, timeout=int(timeout))

    def chat(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.0, 
        max_tokens: int = 512
    ) -> str:
        """
        Send a chat request to the local LLM.
        """
        if self.api_type == "ollama":
            return self._chat_ollama(messages, temperature, max_tokens)
        elif self.api_type == "openai_compatible":
            return self._chat_openai_compat(messages, temperature, max_tokens)
        elif self.api_type == "gemini":
            return self._chat_gemini(messages, temperature, max_tokens)
        else:
            raise LocalLLMError(f"Unknown api_type: {self.api_type}")

    def _chat_ollama(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            raise LocalLLMError(f"Ollama request failed: {e}")

    def _chat_openai_compat(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        # For LM Studio, LocalAI, vLLM serving OpenAI compatible endpoints
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("message", {}).get("content", "")
        except Exception as e:
            raise LocalLLMError(f"OpenAI-compatible request failed: {e}")

    def _chat_gemini(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        if not self.api_key:
            raise LocalLLMError("Gemini API key is missing. Set GENAI_GARDEN_KEY or llm_api_key.")

        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages if m.get("content"))
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            parts = (candidates[0].get("content") or {}).get("parts", [])
            text = "".join(str(p.get("text", "")) for p in parts)
            return text
        except Exception as e:
            raise LocalLLMError(f"Gemini request failed: {e}")
