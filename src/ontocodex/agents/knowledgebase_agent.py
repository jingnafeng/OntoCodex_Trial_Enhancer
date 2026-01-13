from __future__ import annotations

from typing import Any, Dict, List, Optional

from ontocodex.engine.state import OntoCodexState
from ontocodex.kb.kb_api import KnowledgeBase

# Lazy singleton KB (local server friendly)
_KB: Optional[KnowledgeBase] = None


def _get_kb(data_dir: str = "data") -> KnowledgeBase:
    global _KB
    if _KB is None:
        _KB = KnowledgeBase.from_local_data(data_dir=data_dir, enable_vector=False)
    return _KB


def knowledgebase_node(state: OntoCodexState) -> OntoCodexState:
    """
    Term-level retrieval only:
      - For each candidate, lookup terminology hits in the KB
      - Store hits on the candidate as `kb_hits`
    """
    if not state.candidates:
        state.warnings.append("KnowledgeBaseAgent: no candidates to retrieve.")
        return state

    data_dir = state.options.get("data_dir", "data")
    system = state.options.get("kb_system")  # optional system constraint
    top_k = int(state.options.get("kb_top_k", 5))

    kb = _get_kb(data_dir=data_dir)

    for cand in state.candidates:
        term = cand.get("term") or cand.get("label") or cand.get("concept_name")
        if not term:
            continue
        hits = kb.lookup(term=term, system=system, k=top_k)
        cand["kb_hits"] = hits

    return state
