from __future__ import annotations

import json
from typing import Any, Dict, List

from ontocodex.engine.state import OntoCodexState
from ontocodex.kb.terminology_store import normalize_term
from ontocodex.kb.trial_codes_store import TrialCodesStore
from ontocodex.kb.utils import get_kb
from ontocodex.providers.local_llm import LocalLLM, LocalLLMError


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return {}
    try:
        data = json.loads(text[s : e + 1])
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _llm_parse_section(state: OntoCodexState, text: str) -> Dict[str, Any]:
    llm = LocalLLM.from_options(state.options, prefix="decision")
    system = (
        "Extract criterion concepts and return only JSON with keys: "
        "inclusion_concepts (array), exclusion_concepts (array), "
        "temporal_constraints (array), outcomes (array)."
    )
    try:
        raw = llm.chat(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
            temperature=0.0,
            max_tokens=300,
        )
    except LocalLLMError as exc:
        state.warnings.append(f"TrialOrchestrator LLM section parse failed: {exc}")
        return {"inclusion_concepts": [], "exclusion_concepts": [], "temporal_constraints": [], "outcomes": []}
    data = _extract_json(raw)
    return {
        "inclusion_concepts": data.get("inclusion_concepts", []) if isinstance(data, dict) else [],
        "exclusion_concepts": data.get("exclusion_concepts", []) if isinstance(data, dict) else [],
        "temporal_constraints": data.get("temporal_constraints", []) if isinstance(data, dict) else [],
        "outcomes": data.get("outcomes", []) if isinstance(data, dict) else [],
    }


def _map_term(term: str, kb, trial_codes: TrialCodesStore) -> Dict[str, Any]:
    out: Dict[str, Any] = {k: None for k in ("ICD10CM", "ICD10", "RXNORM", "CPT4", "CPT")}
    dict_hits = trial_codes.lookup(term, systems=("ICD10CM", "ICD10", "RXNORM", "CPT4", "CPT"), k=5)
    for h in dict_hits:
        sys = str(h.get("system", "")).upper()
        if sys in out and out[sys] is None:
            out[sys] = {"code": h.get("code"), "term": h.get("term"), "score": h.get("score"), "evidence": h.get("evidence")}
    for sys in ("ICD10CM", "ICD10", "RXNORM", "CPT4", "CPT"):
        if out[sys] is None:
            hits = kb.term_store.lookup(term=term, system=sys, k=1)
            if hits:
                h = hits[0]
                out[sys] = {"code": h.get("code"), "term": h.get("term"), "score": h.get("score"), "evidence": h.get("evidence")}
    return out


def _ensure_orch(state: OntoCodexState) -> Dict[str, Any]:
    orch = (state.artifacts.get("trial_orchestrator", {}) or {})
    if orch:
        return orch
    plan = (state.artifacts.get("decision_plan", {}) or {})
    sections = plan.get("criteria_sections", []) or []
    orch = {
        "sections": sections,
        "section_idx": 0,
        "agent_idx": 0,
        "results": [
            {
                "section_id": s.get("id"),
                "criterion_type": s.get("criterion_type"),
                "text": s.get("text"),
                "route_agents": s.get("route_agents", []),
                "llm_parsed": {},
                "kb_hits": [],
                "terminology_mappings": [],
            }
            for s in sections
        ],
    }
    state.artifacts["trial_orchestrator"] = orch
    return orch


def init_trial_orchestrator_node(state: OntoCodexState) -> OntoCodexState:
    _ensure_orch(state)
    return state


def trial_orchestrator_route(state: OntoCodexState) -> str:
    orch = _ensure_orch(state)
    si = int(orch.get("section_idx", 0))
    sections = orch.get("sections", [])
    if si >= len(sections):
        return "to_finalize"
    ai = int(orch.get("agent_idx", 0))
    route_agents = sections[si].get("route_agents", []) or []
    if ai >= len(route_agents):
        return "to_advance"
    agent = str(route_agents[ai]).lower()
    if agent == "llm":
        return "to_llm"
    if agent == "knowledge":
        return "to_kb"
    if agent == "terminology":
        return "to_terminology"
    return "to_advance"


def trial_section_llm_node(state: OntoCodexState) -> OntoCodexState:
    orch = _ensure_orch(state)
    si = int(orch.get("section_idx", 0))
    sec = orch["sections"][si]
    parsed = _llm_parse_section(state, str(sec.get("text", "")))
    orch["results"][si]["llm_parsed"] = parsed
    orch["agent_idx"] = int(orch.get("agent_idx", 0)) + 1
    state.artifacts["trial_orchestrator"] = orch
    return state


def trial_section_kb_node(state: OntoCodexState) -> OntoCodexState:
    orch = _ensure_orch(state)
    si = int(orch.get("section_idx", 0))
    data_dir = state.options.get("data_dir", "data")
    kb = get_kb(data_dir=data_dir)
    sec_result = orch["results"][si]
    parsed = sec_result.get("llm_parsed", {}) or {}
    terms = []
    for key in ("inclusion_concepts", "exclusion_concepts", "outcomes"):
        for item in parsed.get(key, []) or []:
            if isinstance(item, str):
                terms.append(item)
            elif isinstance(item, dict):
                terms.append(item.get("term", ""))
    if not terms:
        terms = [sec_result.get("text", "")]
    kb_hits = []
    for t in terms[:8]:
        t = str(t).strip()
        if not t:
            continue
        kb_hits.extend(kb.lookup(term=t, system=None, k=2))
    sec_result["kb_hits"] = kb_hits[:12]
    orch["agent_idx"] = int(orch.get("agent_idx", 0)) + 1
    state.artifacts["trial_orchestrator"] = orch
    return state


def trial_section_terminology_node(state: OntoCodexState) -> OntoCodexState:
    orch = _ensure_orch(state)
    si = int(orch.get("section_idx", 0))
    data_dir = state.options.get("data_dir", "data")
    kb = get_kb(data_dir=data_dir)
    trial_codes = TrialCodesStore.from_dir(data_dir=data_dir)

    sec_result = orch["results"][si]
    parsed = sec_result.get("llm_parsed", {}) or {}
    terms = []
    for key in ("inclusion_concepts", "exclusion_concepts", "outcomes"):
        for item in parsed.get(key, []) or []:
            if isinstance(item, str):
                terms.append(item)
            elif isinstance(item, dict):
                terms.append(item.get("term", ""))
    # Add likely terms from KB hits as backup.
    for h in sec_result.get("kb_hits", []) or []:
        t = h.get("term")
        if t:
            terms.append(t)

    seen = set()
    mappings = []
    for term in terms:
        term = str(term).strip()
        norm = normalize_term(term)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        mappings.append(
            {
                "term": term,
                "mapped_codes": _map_term(term, kb=kb, trial_codes=trial_codes),
            }
        )
    sec_result["terminology_mappings"] = mappings
    orch["agent_idx"] = int(orch.get("agent_idx", 0)) + 1
    state.artifacts["trial_orchestrator"] = orch
    return state


def trial_advance_section_node(state: OntoCodexState) -> OntoCodexState:
    orch = _ensure_orch(state)
    orch["section_idx"] = int(orch.get("section_idx", 0)) + 1
    orch["agent_idx"] = 0
    state.artifacts["trial_orchestrator"] = orch
    return state


def trial_finalize_orchestrator_node(state: OntoCodexState) -> OntoCodexState:
    orch = _ensure_orch(state)
    inclusion = []
    exclusion = []
    temporal = []
    outcomes = []
    concept_mappings = []

    for r in orch.get("results", []):
        parsed = r.get("llm_parsed", {}) or {}
        inclusion.extend(parsed.get("inclusion_concepts", []) or [])
        exclusion.extend(parsed.get("exclusion_concepts", []) or [])
        temporal.extend(parsed.get("temporal_constraints", []) or [])
        outcomes.extend(parsed.get("outcomes", []) or [])
        for m in r.get("terminology_mappings", []) or []:
            concept_mappings.append(
                {
                    "criterion_type": r.get("criterion_type"),
                    "term": m.get("term"),
                    "mapped_codes": m.get("mapped_codes", {}),
                }
            )

    state.artifacts["trial_parser"] = {
        "input_text": ((state.artifacts.get("trial_document", {}) or {}).get("text")) or (state.options.get("trial_text") or ""),
        "decision_plan": state.artifacts.get("decision_plan", {}),
        "section_results": orch.get("results", []),
        "inclusion_concepts": inclusion,
        "exclusion_concepts": exclusion,
        "temporal_constraints": temporal,
        "outcomes": outcomes,
        "concept_mappings": concept_mappings,
        "composite_concepts": [],
    }
    return state

