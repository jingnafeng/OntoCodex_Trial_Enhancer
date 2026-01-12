from ontocodex.engine.state import OntoCodexState

def terminology_node(state: OntoCodexState) -> OntoCodexState:
    # Turn KB hits into canonical mappings: {term, system, code, confidence, evidence_ref}
    for cand in state.candidates:
        hits = cand.get("kb_hits", [])
        if not hits:
            continue
        best = hits[0]  # later: scoring
        state.mappings.append({
            "term": cand.get("term") or cand.get("label"),
            "system": best.get("system"),
            "code": best.get("code"),
            "confidence": best.get("score", 1.0),
            "evidence": best.get("evidence"),
        })
    return state
