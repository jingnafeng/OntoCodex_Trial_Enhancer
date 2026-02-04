import json
from pathlib import Path
from typing import Any, Dict, Optional

from ontocodex.engine.state import OntoCodexState
from ontocodex.providers.local_llm import LocalLLM, LocalLLMError
from ontocodex.kb.utils import get_kb


def _split_trial_sections(trial_text: str) -> list[Dict[str, Any]]:
    parts = [p.strip() for p in trial_text.replace("\r", "\n").split("\n") if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in trial_text.split(". ") if p.strip()]
    out = []
    for idx, text in enumerate(parts):
        low = text.lower()
        if "exclusion" in low:
            kind = "exclusion"
        elif "outcome" in low or "endpoint" in low:
            kind = "outcome"
        elif any(k in low for k in ["inclusion", "eligible", "patients with", "evidence of", "diagnosis"]):
            kind = "inclusion"
        else:
            kind = "other"

        route_agents = ["llm", "terminology"]
        if any(k in low for k in ["inhibitor", "chemotherapy", "metastatic", "phenotype", "combination"]):
            route_agents.append("knowledge")
        out.append(
            {
                "id": f"sec_{idx+1}",
                "criterion_type": kind,
                "text": text,
                "route_agents": route_agents,
            }
        )
    return out


def _build_trial_decision_plan(trial_text: str) -> Dict[str, Any]:
    text = (trial_text or "").lower()
    composite_defs = []

    if "inpatient oncology care" in text or ("inpatient" in text and "oncology care" in text):
        composite_defs.append(
            {
                "name": "Inpatient oncology care",
                "min_components": 2,
                "components": [
                    {
                        "role": "diagnosis",
                        "required_systems": ["ICD10CM", "ICD10"],
                        "description": "Oncology diagnosis term(s), e.g., NSCLC or malignancy.",
                    },
                    {
                        "role": "visit_type",
                        "required_systems": ["CPT4", "CPT"],
                        "description": "Inpatient visit / hospital encounter term.",
                    },
                ],
            }
        )
    if "outpatient oncology care" in text or ("outpatient" in text and "oncology care" in text):
        composite_defs.append(
            {
                "name": "Outpatient oncology care",
                "min_components": 2,
                "components": [
                    {
                        "role": "diagnosis",
                        "required_systems": ["ICD10CM", "ICD10"],
                        "description": "Oncology diagnosis term(s), e.g., NSCLC or malignancy.",
                    },
                    {
                        "role": "visit_type",
                        "required_systems": ["CPT4", "CPT"],
                        "description": "Outpatient office/oncology encounter term.",
                    },
                ],
            }
        )

    return {
        "mode": "trial_parser",
        "composite_term_strategy": {
            "enabled": True,
            "require_multi_term_for_composites": True,
            "default_min_components": 2,
        },
        "section_strategy": {
            "enabled": True,
            "split_by_paragraph_or_sentence": True,
            "agent_router": "decision_agent",
        },
        "criteria_sections": _split_trial_sections(trial_text),
        "composite_definitions": composite_defs,
    }


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


def _llm_extract_terms(state: OntoCodexState, user_text: str) -> Optional[Dict[str, Any]]:
    llm = LocalLLM.from_options(state.options, prefix="decision")
    system = (
        "Extract the primary clinical concept from the user's request. "
        "Return only JSON with keys: canonical_term (string), aliases (array of strings)."
    )
    try:
        raw = llm.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            temperature=0.0,
            max_tokens=200,
        )
    except LocalLLMError as exc:
        state.warnings.append(f"DecisionAgent LLM term extraction failed: {exc}")
        return None
    data = _parse_routing_json(raw)
    if not isinstance(data, dict):
        return None
    return data

def decision_node(state: OntoCodexState) -> OntoCodexState:
    """
    Decision Agent:
    Interprets enrichment goals and orchestrates agent interactions.
    """
    trial_text = (state.options or {}).get("trial_text")
    trial_text_file = (state.options or {}).get("trial_text_file")
    if isinstance(trial_text_file, str) and trial_text_file.strip():
        p = Path(trial_text_file.strip())
        if p.exists() and p.is_file():
            trial_text = p.read_text(encoding="utf-8", errors="ignore")
            state.artifacts["trial_document"] = {
                "source_file": str(p),
                "text": trial_text,
            }
        else:
            state.warnings.append(f"Trial text file not found: {trial_text_file}")

    input_term = (state.options or {}).get("input_term")
    if isinstance(trial_text, str) and trial_text.strip():
        state.artifacts["decision_plan"] = _build_trial_decision_plan(trial_text)
        strict_router = bool((state.options or {}).get("trial_strict_router"))
        state.routing = {
            "needs_trial_parser": not strict_router,
            "needs_trial_orchestrator": strict_router,
            "skip_kb": True,
            "needs_terminology": False,
            "needs_knowledge_agent": False,
        }
        return state
    if isinstance(input_term, str) and "clinical trial" in input_term.lower():
        state.artifacts["decision_plan"] = _build_trial_decision_plan(input_term)
        strict_router = bool((state.options or {}).get("trial_strict_router"))
        state.routing = {
            "needs_trial_parser": not strict_router,
            "needs_trial_orchestrator": strict_router,
            "skip_kb": True,
            "needs_terminology": False,
            "needs_knowledge_agent": False,
        }
        return state

    if isinstance(input_term, str) and input_term.strip():
        term = input_term.strip()
        llm_term_extraction = bool(state.options.get("llm_term_extraction", True))
        lookup_terms = [term]
        if llm_term_extraction:
            extracted = _llm_extract_terms(state, term)
            if extracted:
                canonical = str(extracted.get("canonical_term", "")).strip()
                aliases = extracted.get("aliases", []) or []
                if canonical:
                    lookup_terms.insert(0, canonical)
                for a in aliases:
                    a_str = str(a).strip()
                    if a_str:
                        lookup_terms.append(a_str)
        # de-dup while preserving order
        seen = set()
        lookup_terms = [t for t in lookup_terms if not (t in seen or seen.add(t))]

        data_dir = state.options.get("data_dir", "data")
        kb = get_kb(data_dir=data_dir)
        hits = []
        matched_term = term
        for q in lookup_terms:
            q_hits = kb.term_store.lookup(q, system=state.options.get("kb_system"))
            if q_hits:
                matched_term = q
                hits = q_hits
                break

        state.candidates = [{
            "term": matched_term,
            "input_term": term,
            "label": term,
            "target_iri": None,
            "kb_hits": hits,
            "lookup_terms": lookup_terms,
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
