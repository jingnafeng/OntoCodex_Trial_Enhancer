from __future__ import annotations

from typing import Any, Dict, List


class OntoCodexState(dict):
    """
    State object for the OntoCodex LangGraph workflow.
    Supports both dictionary access (state['key']) and attribute access (state.key).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize default fields
        self.setdefault("ontology_path", "")
        self.setdefault("task_type", "map")  # "map" or "enrich"
        self.setdefault("run_id", "")
        self.setdefault("options", {})
        self.setdefault("candidates", [])
        self.setdefault("mappings", [])
        self.setdefault("evidence", [])
        self.setdefault("routing", {})
        self.setdefault("artifacts", {})
        self.setdefault("ontology_summary", {})
        self.setdefault("errors", [])
        self.setdefault("warnings", [])

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value