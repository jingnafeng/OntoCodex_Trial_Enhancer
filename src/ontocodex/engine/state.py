from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


class OntoCodexState(BaseModel):
    """
    Backward-compatible Pydantic version of your existing dataclass state.

    Keeps the same field names/types you already use:
      - candidates: List[Dict[str, Any]]
      - evidence: List[Dict[str, Any]]
      - mappings: List[Dict[str, Any]]
      - validations: List[Dict[str, Any]]

    So existing agents that read/write dicts will still work.
    """

    model_config = ConfigDict(extra="allow")

    # Run identity
    run_id: str
    task_type: str  # "enrich" | "map" | "extract"
    ontology_path: Optional[str] = None
    guideline_scope: List[str] = Field(default_factory=list)
    options: Dict[str, Any] = Field(default_factory=dict)

    # Ontology reader outputs
    ontology_summary: Optional[Dict[str, Any]] = None
    candidates: List[Dict[str, Any]] = Field(default_factory=list)

    # Decision outputs
    routing: Dict[str, Any] = Field(default_factory=dict)

    # KB evidence & mapping outputs
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    mappings: List[Dict[str, Any]] = Field(default_factory=list)

    # Validation outputs
    validations: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # Script outputs
    generated_script: Optional[str] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)

    # Optional metadata (handy for server runs; safe because extra="allow")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Convenience helpers
    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def get(self, key: str, default: Any = None) -> Any:
        value = getattr(self, key, None)
        if value is not None:
            return value
        extra = getattr(self, "model_extra", None) or {}
        return extra.get(key, default)
