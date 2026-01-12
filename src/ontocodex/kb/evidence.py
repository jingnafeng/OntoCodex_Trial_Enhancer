# src/ontocodex/kb/evidence.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class Evidence:
    """
    Provenance object carried through OntoCodex.

    - source_type: "csv" | "xlsx" | "owl" | "obo" | "ttl" | "vector" | etc.
    - source_file: filename (or filename#sheet for xlsx)
    - id: row_id for tables or IRI for ontology entities
    - field/snippet: optional human-readable trace
    - extra: structured metadata (system, code, concept_id, etc.)
    """
    source_type: str
    source_file: str
    id: str
    field: Optional[str] = None
    snippet: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["extra"] = d["extra"] or {}
        return d
