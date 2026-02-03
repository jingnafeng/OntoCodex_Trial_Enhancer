import json
from typing import Any, Dict, Optional

from ontocodex.engine.state import OntoCodexState
from ontocodex.providers.local_llm import LocalLLM, LocalLLMError
from ontocodex.kb.utils import get_kb


def _parse_routing_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: grab the first JSON object in the response
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _llm_route(state: OntoCodexState) -> Optional[Dict[str, Any]]:
    llm = LocalLLM.from_options(state.options, prefix="decision")
    system = (
        "You are a routing assistant for an ontology mapping pipeline. "
        "Return a compact JSON object with keys: skip_kb (bool), "
        "needs_terminology (bool), needs_knowledge_agent (bool)."
    )
    user = {
        "task_type": state.task_type,
        "candidate_count": len(state.candidates),
        "options": state.options,
    }
    try:
        raw = llm.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)},
            ],
            temperature=0.1,
            max_tokens=200,
        )
    except LocalLLMError as exc:
        state.warnings.append(f"DecisionAgent LLM routing failed: {exc}")
        return None

    data = _parse_routing_json(raw)
    if not isinstance(data, dict):
        state.warnings.append("DecisionAgent LLM routing returned invalid JSON.")
        return None
    return data

def decision_node(state: OntoCodexState) -> OntoCodexState:
    """
    Decision Agent:
    Interprets enrichment goals and orchestrates agent interactions.
    """
    input_term = (state.options or {}).get("input_term")
    if isinstance(input_term, str) and input_term.strip():
        term = input_term.strip()
        data_dir = state.options.get("data_dir", "data")
        kb = get_kb(data_dir=data_dir)
        hits = kb.term_store.lookup(term, system=state.options.get("kb_system"))

        state.candidates = [{
            "term": term,
            "label": term,
            "target_iri": None,
            "kb_hits": hits,
        }]

        if hits:
            state.routing = {
                "skip_kb": True,
                "needs_terminology": True,
                "needs_knowledge_agent": False,
            }
            return state

        state.warnings.append(f"TerminologyStore: no hits for term '{term}'.")
        state.routing = {
            "skip_kb": False,
            "needs_terminology": True,
            "needs_knowledge_agent": False,
        }
        return state

    # Simple deterministic routing first; LLM routing optional
    llm_enabled = bool(state.options.get("llm_routing"))
    if llm_enabled:
        routed = _llm_route(state)
        if routed:
            state.routing = routed
            return state

    task = state.task_type.lower()
    if task == "map":
        state.routing = {
            "skip_kb": True,
            "needs_terminology": True,
            "needs_knowledge_agent": False,
        }
    else:
        state.routing = {
            "skip_kb": False,
            "needs_terminology": True,
            "needs_knowledge_agent": True,
        }
    return state
