from __future__ import annotations

from ontocodex.engine.state import OntoCodexState
from ontocodex.kb.utils import get_kb


def _uniq_terms(terms):
    seen = set()
    out = []
    for t in terms:
        norm = str(t or "").strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


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

    kb = get_kb(data_dir=data_dir)

    for cand in state.candidates:
        term = cand.get("term") or cand.get("label") or cand.get("concept_name")
        if not term:
            continue
        hits = kb.term_store.lookup(term=term, system=system, k=top_k)
        if hits:
            cand["kb_hits"] = hits
            continue

        onto_hits = kb.onto_store.lookup(term=term, k=top_k)
        related_terms = []
        for h in onto_hits:
            label = h.get("term")
            iri = h.get("iri")
            if label:
                related_terms.append(label)
            if iri:
                for parent_iri in kb.onto_store.get_parents(iri, depth=1):
                    parent = kb.onto_store.get_entity(parent_iri)
                    if parent and parent.get("label"):
                        related_terms.append(parent["label"])
        related_terms = _uniq_terms(related_terms)
        cand["fallback_terms"] = related_terms

    return state
