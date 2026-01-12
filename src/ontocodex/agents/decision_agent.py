from ontocodex.engine.state import OntoCodexState

def decision_node(state: OntoCodexState) -> OntoCodexState:
    # Simple deterministic routing first; LLM routing later if desired
    task = state.task_type.lower()
    if task == "map":
        state.routing = {"skip_kb": True, "needs_terminology": True}
    else:
        state.routing = {"skip_kb": False, "needs_terminology": True}
    return state
