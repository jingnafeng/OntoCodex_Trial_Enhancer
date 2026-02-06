from ontocodex.engine.state import OntoCodexState
from ontocodex.kb.utils import get_kb

def _hit_score(hit: dict) -> float:
    try:
        return float(hit.get("score", 0.0))
    except Exception:
        return 0.0


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

def terminology_node(state: OntoCodexState) -> OntoCodexState:
    """
    Terminology Agent:
    Normalizes extracted concepts to standard vocabularies using CSV, OWL, or TTL datasets.
    
    Supported Standards & Vocabularies:
      - ICD-9 / ICD-10
      - SNOMED CT
      - RxNorm
      - ATC
      - LOINC
      - DOID
    """
    # Build a dedup set from existing mappings (in case node re-runs)
    seen = set()
    for m in state.mappings:
        extra = m.get("extra", {}) or {}
        seen.add((m.get("target_iri"), m.get("system"), m.get("code"), extra.get("concept_id")))

    data_dir = state.options.get("data_dir", "data")
    kb = get_kb(data_dir=data_dir)
    systems = _kb_systems(state.options)

    for cand in state.candidates:
        hits = cand.get("kb_hits", []) or []
        if not hits:
            fallback_terms = cand.get("fallback_terms", []) or []
            for term in fallback_terms:
                for system in systems:
                    hits.extend(kb.term_store.lookup(term, system=system, k=5))
        if not hits:
            continue

        # Choose best hit by score
        best = max(hits, key=_hit_score)

        term = cand.get("term") or cand.get("label") or cand.get("concept_name")
        if not term:
            # skip malformed candidate
            continue

        mapping = {
            "term": term,
            "system": best.get("system"),
            "code": best.get("code"),
            "target_iri": cand.get("target_iri"),     # should be set by ontology reader/matcher
            "confidence": best.get("score", 1.0),
            "evidence": best.get("evidence"),         # required for audit + validator
            "extra": best.get("extra", {}) or {},     # contains concept_id from OMOP mapping tables
        }

        extra = mapping["extra"]
        key = (mapping.get("target_iri"), mapping.get("system"), mapping.get("code"), extra.get("concept_id"))
        if key in seen:
            continue
        seen.add(key)

        state.mappings.append(mapping)

    return state
