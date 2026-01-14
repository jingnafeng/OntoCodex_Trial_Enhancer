from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Optional
from urllib import request, error


class LocalLLMError(RuntimeError):
    pass


@dataclass
class LocalLLMConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3"
    timeout_s: int = 60


class LocalLLM:
    def __init__(self, config: Optional[LocalLLMConfig] = None) -> None:
        self.config = config or LocalLLMConfig()

    @classmethod
    def from_env(cls) -> "LocalLLM":
        cfg = LocalLLMConfig(
            base_url=os.getenv("ONTOCODEX_LLM_BASE_URL", "http://localhost:11434"),
            model=os.getenv("ONTOCODEX_LLM_MODEL", "llama3"),
            timeout_s=int(os.getenv("ONTOCODEX_LLM_TIMEOUT_S", "60")),
        )
        return cls(cfg)

    @classmethod
    def from_options(cls, options: Dict[str, Any], prefix: Optional[str] = None) -> "LocalLLM":
        def _opt(key: str) -> Any:
            if prefix:
                prefixed = f"{prefix}_{key}"
                if prefixed in options:
                    return options.get(prefixed)
            return options.get(key)

        cfg = LocalLLMConfig(
            base_url=str(_opt("llm_base_url") or os.getenv("ONTOCODEX_LLM_BASE_URL", "http://localhost:11434")),
            model=str(_opt("llm_model") or os.getenv("ONTOCODEX_LLM_MODEL", "llama3")),
            timeout_s=int(_opt("llm_timeout_s") or os.getenv("ONTOCODEX_LLM_TIMEOUT_S", "60")),
        )
        return cls(cfg)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        return self._post_json("/api/chat", payload, content_path=("message", "content"))

    def complete(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        return self._post_json("/api/generate", payload, content_path=("response",))

    def _post_json(self, path: str, payload: Dict[str, Any], content_path: tuple) -> str:
        url = self.config.base_url.rstrip("/") + path
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with request.urlopen(req, timeout=self.config.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except error.URLError as exc:
            raise LocalLLMError(f"Local LLM request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LocalLLMError("Local LLM returned non-JSON response.") from exc

        cur: Any = data
        for key in content_path:
            if not isinstance(cur, dict) or key not in cur:
                raise LocalLLMError(f"Local LLM response missing field: {'.'.join(content_path)}")
            cur = cur[key]
        if not isinstance(cur, str):
            raise LocalLLMError("Local LLM response content is not a string.")
        return cur.strip()
