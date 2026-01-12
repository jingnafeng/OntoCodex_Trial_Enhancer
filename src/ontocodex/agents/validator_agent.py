from ontocodex.engine.state import OntoCodexState

REQUIRED_EVIDENCE_FIELDS = ("source_file", "source_type", "id")

def validator_node(state: OntoCodexState) -> OntoCodexState:
    # Enforce: no mapping without evidence
    for m in state.mappings:
        ev = m.get("evidence") or {}
        missing = [k for k in REQUIRED_EVIDENCE_FIELDS if k not in ev]
        if missing:
            state.errors.append(f"Mapping missing evidence fields {missing}: {m}")
    return state
