from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_evidence_rows(run_id: str, mappings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert OntoCodex mappings into audit-friendly evidence rows.
    Expect each mapping to include:
      - term, system, code, target_iri
      - extra.concept_id (OMOP concept_id)
      - evidence (dict produced by Evidence.to_dict())
    """
    ts = utc_now_iso()
    out: List[Dict[str, Any]] = []

    for m in mappings:
        extra = m.get("extra", {}) or {}
        ev = m.get("evidence", {}) or {}

        out.append({
            "run_id": run_id,
            "timestamp": ts,
            "input_term": m.get("term"),
            "target_iri": m.get("target_iri"),
            "mapping": {
                "system": m.get("system"),
                "code": m.get("code"),
                "omop_concept_id": extra.get("concept_id"),
                "standard_concept": extra.get("standard_concept"),
            },
            "evidence": {
                "source_type": ev.get("source_type"),
                "source_file": ev.get("source_file"),
                "row_id": ev.get("id"),
                "snippet": ev.get("snippet"),
                # Optional: keep the full ev["extra"] for debugging
                "extra": ev.get("extra", {}),
            },
            "scores": {
                "kb_score": m.get("confidence") or m.get("score"),
            },
        })

    return out
