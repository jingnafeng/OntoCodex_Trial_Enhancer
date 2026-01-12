from ontocodex.engine.state import OntoCodexState
from ontocodex.kb.kb_api import KnowledgeBase

KB: KnowledgeBase | None = None  # loaded once at startup

def knowledgebase_node(state: OntoCodexState) -> OntoCodexState:
    global KB
    if KB is None:
        KB = KnowledgeBase.from_local_data()

    for cand in state.candidates:
        term = cand.get("term") or cand.get("label")
        if not term:
            continue

        # Tier A/B first
        results = KB.lookup(term=term, system=state.options.get("system"))
        state.evidence.extend([r["evidence"] for r in results])
        cand["kb_hits"] = results

        # Optionally attach top supporting facts
    return state
