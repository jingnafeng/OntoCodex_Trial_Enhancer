from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class OntoCodexState:
    # Run identity
    run_id: str
    task_type: str                 # "enrich" | "map" | "extract"
    ontology_path: Optional[str] = None
    guideline_scope: List[str] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)

    # Ontology reader outputs
    ontology_summary: Optional[Dict[str, Any]] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)   # terms/classes needing enrichment

    # Decision outputs
    routing: Dict[str, Any] = field(default_factory=dict)

    # KB evidence & mapping outputs
    evidence: List[Dict[str, Any]] = field(default_factory=list)     # serialized Evidence objects
    mappings: List[Dict[str, Any]] = field(default_factory=list)     # normalized code mappings

    # Validation outputs
    validations: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Script outputs
    generated_script: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)          # paths, summaries, etc.
