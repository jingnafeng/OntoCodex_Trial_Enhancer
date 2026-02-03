from __future__ import annotations

import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class LocalLLMError(Exception):
    """Raised when the Local LLM provider fails."""
    pass


@dataclass
class LocalLLMConfig:
    """Configuration for LocalLLM."""
    base_url: str
    model: str
    api_type: str = "ollama"


class LocalLLM:
    """
    A simple client for a local LLM server (e.g., Ollama, LM Studio, LocalAI).
    Defaults to Ollama's API format.
    """
    def __init__(self, base_url: str, model: str, api_type: str = "ollama"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_type = api_type

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
        # 1. Try specific prefix
        base_url = options.get(f"{prefix}_base_url")
        model = options.get(f"{prefix}_model")
        api_type = options.get(f"{prefix}_api_type")

        # 2. Fallback to generic 'llm_'
        if not base_url:
            base_url = options.get("llm_base_url", "http://localhost:11434")
        if not model:
            model = options.get("llm_model", "llama3")
        if not api_type:
            api_type = options.get("llm_api_type", "ollama")

        return cls(base_url=base_url, model=model, api_type=api_type)

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
            resp = requests.post(url, json=payload, timeout=60)
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
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("message", {}).get("content", "")
        except Exception as e:
            raise LocalLLMError(f"OpenAI-compatible request failed: {e}")