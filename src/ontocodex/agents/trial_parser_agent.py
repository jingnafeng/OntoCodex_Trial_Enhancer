from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from ontocodex.engine.state import OntoCodexState
from ontocodex.kb.terminology_store import normalize_term
from ontocodex.kb.trial_codes_store import TrialCodesStore
from ontocodex.kb.utils import get_kb
from ontocodex.providers.local_llm import LocalLLM, LocalLLMError


def _extract_json_object(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _fallback_parse(trial_text: str) -> Dict[str, Any]:
    icd_codes = re.findall(r"\b[A-TV-Z][0-9]{2}(?:\.[0-9A-ZxX]+)?(?:[–-][A-TV-Z]?[0-9]{2}(?:\.[0-9A-ZxX]+)?)?\b", trial_text)
    tl = trial_text.lower()
    inclusion = []
    if "non" in tl and "small cell lung cancer" in tl:
        inclusion.append({"term": "Non-small cell lung cancer", "temporal_constraint": "prior to Index Date"})
    if "metastatic" in tl:
        inclusion.append({"term": "Metastatic disease", "temporal_constraint": "prior to Index Date"})
    if "outpatient oncology care" in tl:
        inclusion.append({"term": "Outpatient oncology care", "temporal_constraint": "prior to Index Date"})
    if "inpatient oncology care" in tl or re.search(r"\binpatient\b.*\boncology care\b", tl):
        inclusion.append({"term": "Inpatient oncology care", "temporal_constraint": "prior to Index Date"})
    if "platinum" in tl and "chemotherapy" in tl:
        inclusion.append({"term": "Platinum-based chemotherapy", "temporal_constraint": "first-line"})
    if "pd-1" in tl or "pd-l1" in tl:
        inclusion.append({"term": "PD-1/PD-L1 inhibitor therapy", "temporal_constraint": "first-line"})
    if "docetaxel" in tl:
        inclusion.append({"term": "Docetaxel-based treatment", "temporal_constraint": "second-line"})
    if "outpatient infusion" in tl:
        inclusion.append({"term": "Outpatient infusion setting", "temporal_constraint": "second-line initiation"})
    return {
        "inclusion_concepts": inclusion,
        "exclusion_concepts": [],
        "temporal_constraints": [
            x
            for x, cond in [
                ("prior to Index Date", "index date" in tl),
                ("first-line before second-line initiation", "prior to initiation of second-line" in tl),
                ("second-line initiation in routine care", "second-line treatment initiated" in tl),
            ]
            if cond
        ],
        "outcomes": [],
        "codes_mentioned": {"ICD": icd_codes},
    }


def _parse_trial_with_llm(state: OntoCodexState, trial_text: str) -> Dict[str, Any]:
    llm = LocalLLM.from_options(state.options, prefix="decision")
    prompt = (
        "Extract clinical trial structure and return only JSON with keys: "
        "inclusion_concepts (array of {term, temporal_constraint}), "
        "exclusion_concepts (array of {term, temporal_constraint}), "
        "temporal_constraints (array of strings), outcomes (array of strings), "
        "codes_mentioned (object with ICD/RXNORM/CPT arrays)."
    )
    try:
        raw = llm.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": trial_text},
            ],
            temperature=0.0,
            max_tokens=700,
        )
        data = _extract_json_object(raw)
        if data:
            return data
    except LocalLLMError as exc:
        state.warnings.append(f"TrialParser LLM parse failed: {exc}")
    return _fallback_parse(trial_text)


def _map_term_multi_system(kb, term: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {k: None for k in ("ICD10CM", "ICD10", "RXNORM", "CPT4", "CPT")}

    trial_store: TrialCodesStore = kb.get("trial_codes_store")
    if trial_store is not None:
        dict_hits = trial_store.lookup(term=term, systems=("ICD10CM", "ICD10", "RXNORM", "CPT4", "CPT"), k=5)
        for h in dict_hits:
            sys = str(h.get("system", "")).upper()
            if sys in out and out[sys] is None:
                out[sys] = {
                    "code": h.get("code"),
                    "term": h.get("term"),
                    "score": h.get("score"),
                    "evidence": h.get("evidence"),
                }

    for system in ("ICD10CM", "ICD10", "RXNORM", "CPT4", "CPT"):
        if out[system] is not None:
            continue
        hits = kb["kb"].term_store.lookup(term=term, system=system, k=1)
        if hits:
            h = hits[0]
            out[system] = {
                "code": h.get("code"),
                "term": h.get("term"),
                "score": h.get("score"),
                "evidence": h.get("evidence"),
            }
    return out


def _best_code_for_systems(mapped_codes: Dict[str, Any], systems: List[str]) -> Any:
    for s in systems:
        hit = mapped_codes.get(s)
        if hit:
            return hit
    return None


def _build_composite_concepts(
    trial_text: str,
    parsed: Dict[str, Any],
    decision_plan: Dict[str, Any],
    kb,
) -> List[Dict[str, Any]]:
    if not decision_plan:
        return []
    strat = decision_plan.get("composite_term_strategy", {}) or {}
    if not strat.get("enabled", False):
        return []

    inclusion_terms = []
    for item in parsed.get("inclusion_concepts", []) or []:
        if isinstance(item, str):
            inclusion_terms.append(item)
        else:
            inclusion_terms.append(str(item.get("term", "")))
    inclusion_terms = [t.strip() for t in inclusion_terms if t and str(t).strip()]

    diagnosis_candidates = [
        t for t in inclusion_terms if any(x in normalize_term(t) for x in ("cancer", "carcinoma", "neoplasm", "tumor"))
    ]
    if not diagnosis_candidates and "non-small cell lung cancer" in trial_text.lower():
        diagnosis_candidates = ["Non-small cell lung cancer"]

    composites: List[Dict[str, Any]] = []
    for cdef in decision_plan.get("composite_definitions", []) or []:
        name = cdef.get("name", "")
        lname = name.lower()
        if lname and lname not in trial_text.lower() and lname not in " ".join(inclusion_terms).lower():
            continue

        component_terms: List[Dict[str, Any]] = []
        for comp in cdef.get("components", []) or []:
            role = comp.get("role")
            systems = comp.get("required_systems", [])
            if role == "diagnosis":
                # Use one or more diagnosis terms to satisfy composite definition.
                for dterm in diagnosis_candidates[:2]:
                    mc = _map_term_multi_system(kb, dterm)
                    best = _best_code_for_systems(mc, systems)
                    if best:
                        component_terms.append(
                            {
                                "role": role,
                                "term": dterm,
                                "mapping": best,
                            }
                        )
            elif role == "visit_type":
                visit_terms = []
                if "inpatient" in lname:
                    visit_terms = ["Inpatient oncology care", "Inpatient visit"]
                elif "outpatient" in lname:
                    visit_terms = ["Outpatient oncology care", "Outpatient visit"]
                for vterm in visit_terms:
                    mc = _map_term_multi_system(kb, vterm)
                    best = _best_code_for_systems(mc, systems)
                    if best:
                        component_terms.append(
                            {
                                "role": role,
                                "term": vterm,
                                "mapping": best,
                            }
                        )
                        break

        min_components = int(cdef.get("min_components", strat.get("default_min_components", 2)))
        meets = len(component_terms) >= min_components
        composites.append(
            {
                "name": name,
                "min_components": min_components,
                "component_terms": component_terms,
                "meets_definition": meets,
            }
        )
    return composites


def trial_parser_node(state: OntoCodexState) -> OntoCodexState:
    trial_text = (
        (state.artifacts.get("trial_document", {}) or {}).get("text")
        or (state.options or {}).get("trial_text")
        or (state.options or {}).get("input_term")
        or ""
    )
    trial_text = str(trial_text).strip()
    if not trial_text:
        state.warnings.append("TrialParser: no trial_text/input_term provided.")
        return state

    decision_plan = (state.artifacts or {}).get("decision_plan", {}) or {}
    sections = decision_plan.get("criteria_sections", []) or []
    if sections:
        merged = {
            "inclusion_concepts": [],
            "exclusion_concepts": [],
            "temporal_constraints": [],
            "outcomes": [],
            "codes_mentioned": {"ICD": [], "RXNORM": [], "CPT": []},
        }
        section_results = []
        for sec in sections:
            sec_text = str(sec.get("text", "")).strip()
            sec_parsed = _parse_trial_with_llm(state, sec_text)
            for k in ("inclusion_concepts", "exclusion_concepts", "temporal_constraints", "outcomes"):
                merged[k].extend(sec_parsed.get(k, []) or [])
            cm = sec_parsed.get("codes_mentioned", {}) or {}
            for k in ("ICD", "RXNORM", "CPT"):
                merged["codes_mentioned"][k].extend(cm.get(k, []) or [])
            section_results.append({"section_id": sec.get("id"), "criterion_type": sec.get("criterion_type"), "parsed": sec_parsed})
        parsed = merged
        state.artifacts["trial_parser_sections"] = section_results
    else:
        parsed = _parse_trial_with_llm(state, trial_text)
    parsed.setdefault("inclusion_concepts", [])
    parsed.setdefault("exclusion_concepts", [])
    parsed.setdefault("temporal_constraints", [])
    parsed.setdefault("outcomes", [])
    parsed.setdefault("codes_mentioned", {})

    # Rule-based augmentation keeps key care-setting concepts even when LLM misses them.
    it_lower = trial_text.lower()
    if "outpatient oncology care" in it_lower:
        parsed["inclusion_concepts"].append({"term": "Outpatient oncology care", "temporal_constraint": "prior to Index Date"})
    if "inpatient oncology care" in it_lower or re.search(r"\binpatient\b.*\boncology care\b", it_lower):
        parsed["inclusion_concepts"].append({"term": "Inpatient oncology care", "temporal_constraint": "prior to Index Date"})
    # de-dup inclusion/exclusion concepts
    def _dedup(items):
        seen = set()
        out = []
        for item in items:
            term = item if isinstance(item, str) else item.get("term")
            key = normalize_term(str(term or ""))
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out
    parsed["inclusion_concepts"] = _dedup(parsed.get("inclusion_concepts", []) or [])
    parsed["exclusion_concepts"] = _dedup(parsed.get("exclusion_concepts", []) or [])

    data_dir = state.options.get("data_dir", "data")
    kb = {
        "kb": get_kb(data_dir=data_dir),
        "trial_codes_store": TrialCodesStore.from_dir(data_dir=data_dir),
    }

    concept_mappings: List[Dict[str, Any]] = []
    for bucket in ("inclusion_concepts", "exclusion_concepts"):
        for item in parsed.get(bucket, []) or []:
            if isinstance(item, str):
                term = item.strip()
                temporal = None
            else:
                term = str(item.get("term", "")).strip()
                temporal = item.get("temporal_constraint")
            if not term:
                continue
            concept_mappings.append(
                {
                    "criterion_type": "inclusion" if bucket == "inclusion_concepts" else "exclusion",
                    "term": term,
                    "temporal_constraint": temporal,
                    "mapped_codes": _map_term_multi_system(kb, term),
                }
            )

    # Preserve ICD codes explicitly provided by protocol text
    icd_mentions = parsed.get("codes_mentioned", {}).get("ICD", []) or []
    if icd_mentions:
        concept_mappings.append(
            {
                "criterion_type": "protocol_codes",
                "term": "ICD codes from protocol text",
                "temporal_constraint": None,
                "mapped_codes": {
                    "ICD10CM": {"code": icd_mentions, "term": "Protocol-specified ICD range", "score": 1.0, "evidence": None},
                    "ICD10": {"code": icd_mentions, "term": "Protocol-specified ICD range", "score": 1.0, "evidence": None},
                    "RXNORM": None,
                    "CPT4": None,
                    "CPT": None,
                },
            }
        )

    composite_concepts = _build_composite_concepts(
        trial_text=trial_text,
        parsed=parsed,
        decision_plan=decision_plan,
        kb=kb,
    )

    state.artifacts["trial_parser"] = {
        "input_text": trial_text,
        "decision_plan": decision_plan,
        "section_results": state.artifacts.get("trial_parser_sections", []),
        "inclusion_concepts": parsed.get("inclusion_concepts", []),
        "exclusion_concepts": parsed.get("exclusion_concepts", []),
        "temporal_constraints": parsed.get("temporal_constraints", []),
        "outcomes": parsed.get("outcomes", []),
        "concept_mappings": concept_mappings,
        "composite_concepts": composite_concepts,
    }
    return state
