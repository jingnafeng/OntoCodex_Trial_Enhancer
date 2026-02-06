from __future__ import annotations

import json

from ontocodex.engine.state import OntoCodexState
from ontocodex.kb.utils import get_kb
from ontocodex.providers.local_llm import LocalLLM, LocalLLMError


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


def _kb_systems(options):
    vals = options.get("kb_systems")
    if isinstance(vals, list) and vals:
        out = []
        for v in vals:
            s = str(v).strip()
            if s:
                out.append(s)
        return out or [None]
    one = options.get("kb_system")
    if one:
        return [str(one).strip()]
    return [None]


def _llm_expand_related_terms(state: OntoCodexState, term: str):
    llm = LocalLLM.from_options(state.options, prefix="knowledge")
    system = (
        "Given a clinical term or short request, return JSON with key related_terms "
        "(array of concise clinical terms, abbreviations, and parent terms). "
        "Return only JSON."
    )
    try:
        raw = llm.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": term},
            ],
            temperature=0.0,
            max_tokens=250,
        )
    except LocalLLMError as exc:
        state.warnings.append(f"KnowledgeBaseAgent LLM expansion failed: {exc}")
        return []
    try:
        data = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            data = json.loads(raw[start : end + 1])
        except Exception:
            return []
    vals = data.get("related_terms", []) if isinstance(data, dict) else []
    return _uniq_terms(vals if isinstance(vals, list) else [])


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
    systems = _kb_systems(state.options)
    top_k = int(state.options.get("kb_top_k", 5))

    kb = get_kb(data_dir=data_dir)

    for cand in state.candidates:
        term = cand.get("term") or cand.get("label") or cand.get("concept_name")
        if not term:
            continue
        hits = []
        for system in systems:
            hits = kb.term_store.lookup(term=term, system=system, k=top_k)
            if hits:
                break
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
        if not related_terms and bool(state.options.get("llm_kb_expansion", True)):
            related_terms = _llm_expand_related_terms(state, term)
        related_terms = _uniq_terms(related_terms)
        cand["fallback_terms"] = related_terms

    return state
