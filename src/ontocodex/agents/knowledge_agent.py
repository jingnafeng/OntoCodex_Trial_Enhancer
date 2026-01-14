from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL

from ontocodex.engine.state import OntoCodexState
from ontocodex.kb.kb_api import KnowledgeBase
from ontocodex.kb.utils import get_kb
from ontocodex.utils.owl_utils import best_label, is_owl_class, load_owl, local_name


def _property_matches(p: URIRef, wanted_local_names: Set[str], wanted_full_iris: Set[str]) -> bool:
    ps = str(p)
    if ps in wanted_full_iris:
        return True
    return local_name(p) in wanted_local_names


def _extract_fillers_via_restrictions(
    g: Graph,
    disease_iri: str,
    property_local_names: List[str],
    property_full_iris: Optional[List[str]] = None,
) -> List[str]:
    """
    Find fillers in axioms like:
      disease rdfs:subClassOf [
         a owl:Restriction ;
         owl:onProperty <prop> ;
         owl:someValuesFrom <FILLER>
      ] .

    Returns list of filler class IRIs (strings).
    """
    subj = URIRef(disease_iri)
    wanted_local = set(property_local_names)
    wanted_full = set(property_full_iris or [])

    fillers: List[str] = []

    for restriction in g.objects(subj, RDFS.subClassOf):
        # restriction must be owl:Restriction
        if (restriction, RDF.type, OWL.Restriction) not in g:
            continue

        # onProperty must match
        props = list(g.objects(restriction, OWL.onProperty))
        if not props:
            continue
        p = props[0]
        if not isinstance(p, URIRef):
            continue
        if not _property_matches(p, wanted_local, wanted_full):
            continue

        # someValuesFrom must point to a class (URIRef)
        for filler in g.objects(restriction, OWL.someValuesFrom):
            if isinstance(filler, URIRef):
                fillers.append(str(filler))

    # de-dup preserving order
    seen = set()
    out = []
    for f in fillers:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def _kb_best_hit(kb: KnowledgeBase, term: str, system: Optional[str], k: int = 5) -> Optional[Dict[str, Any]]:
    hits = kb.lookup(term=term, system=system, k=k)
    if not hits:
        return None
    # safest: pick max score if present
    def sc(h: Dict[str, Any]) -> float:
        try:
            return float(h.get("score", 0.0))
        except Exception:
            return 0.0
    return max(hits, key=sc)


def knowledge_agent_node(state: OntoCodexState) -> OntoCodexState:
    """
    Knowledge Base Agent:
    Retrieves candidate concepts and relationships from curated sources such as:
      - MedlinePlus (API / cached TTL)
      - PubMed
      - NIH / CDC / ADA / ACC / AHA resources
      - Disease Ontology (DOID)
      3) Build enrichment payloads that will be applied as annotations to the disease class later

    Output:
      - state.artifacts["enrichments"] = List[dict]
      - state.evidence extended with evidence dicts from KB hits (for audit)
    """
    if not state.ontology_path:
        state.errors.append("KnowledgeAgent: ontology_path is missing.")
        return state

    g = load_owl(state.ontology_path)

    # Config knobs (override via state.options)
    data_dir = state.options.get("data_dir", "data")

    # Properties to interpret as "medication links" and "lab links"
    # Use LOCAL NAMES by default (works even if ontology uses relative IRIs)
    med_props_local = state.options.get("medication_properties_local", ["treated_by_medication"])
    lab_props_local = state.options.get("lab_properties_local", ["has_lab_test", "assessed_by_lab", "diagnosed_by_lab"])

    # Optional: exact full IRIs if you have them later
    med_props_full = state.options.get("medication_properties_full", [])
    lab_props_full = state.options.get("lab_properties_full", [])

    kb = get_kb(data_dir=data_dir)

    enrichments: List[Dict[str, Any]] = state.artifacts.get("enrichments", []) or []

    for cand in state.candidates:
        disease_iri = cand.get("target_iri")
        if not disease_iri:
            continue

        disease_uri = URIRef(disease_iri)
        if not is_owl_class(g, disease_uri):
            # classes only: skip anything that isn't a class
            continue

        disease_term = cand.get("term") or cand.get("label") or best_label(g, disease_uri)

        # --- 1) map disease itself (optional here; you may already do this earlier) ---
        disease_hit = _kb_best_hit(kb, term=disease_term, system=None)  # or system constraint if you want
        disease_mapping = None
        if disease_hit:
            disease_mapping = {
                "term": disease_term,
                "system": disease_hit.get("system"),
                "code": disease_hit.get("code"),
                "omop_concept_id": (disease_hit.get("extra") or {}).get("concept_id"),
                "confidence": disease_hit.get("score", 1.0),
                "evidence": disease_hit.get("evidence"),
                "extra": disease_hit.get("extra", {}) or {},
            }
            if disease_hit.get("evidence"):
                state.evidence.append(disease_hit["evidence"])

        # --- 2) extract linked medication class IRIs from OWL restrictions ---
        med_iris = _extract_fillers_via_restrictions(
            g=g,
            disease_iri=disease_iri,
            property_local_names=med_props_local,
            property_full_iris=med_props_full,
        )

        medications: List[Dict[str, Any]] = []
        for miri in med_iris:
            muri = URIRef(miri)
            if not is_owl_class(g, muri):
                continue

            m_label = best_label(g, muri)
            hit = _kb_best_hit(kb, term=m_label, system="RXNORM")
            item = {
                "term": m_label,
                "class_iri": miri,
                "mapping": None,
            }
            if hit:
                item["mapping"] = {
                    "system": hit.get("system"),  # RXNORM
                    "code": hit.get("code"),
                    "omop_concept_id": (hit.get("extra") or {}).get("concept_id"),
                    "confidence": hit.get("score", 1.0),
                    "evidence": hit.get("evidence"),
                    "extra": hit.get("extra", {}) or {},
                }
                if hit.get("evidence"):
                    state.evidence.append(hit["evidence"])
            medications.append(item)

        # --- 3) extract linked lab class IRIs from OWL restrictions ---
        lab_iris = _extract_fillers_via_restrictions(
            g=g,
            disease_iri=disease_iri,
            property_local_names=lab_props_local,
            property_full_iris=lab_props_full,
        )

        labs: List[Dict[str, Any]] = []
        for liri in lab_iris:
            luri = URIRef(liri)
            if not is_owl_class(g, luri):
                continue

            l_label = best_label(g, luri)
            hit = _kb_best_hit(kb, term=l_label, system="LOINC")
            item = {
                "term": l_label,
                "class_iri": liri,
                "mapping": None,
            }
            if hit:
                item["mapping"] = {
                    "system": hit.get("system"),  # LOINC
                    "code": hit.get("code"),
                    "omop_concept_id": (hit.get("extra") or {}).get("concept_id"),
                    "cui": (hit.get("extra") or {}).get("cui"),  # from LOINC_CUI if available
                    "confidence": hit.get("score", 1.0),
                    "evidence": hit.get("evidence"),
                    "extra": hit.get("extra", {}) or {},
                }
                if hit.get("evidence"):
                    state.evidence.append(hit["evidence"])
            labs.append(item)

        enrichments.append({
            "disease": {
                "term": disease_term,
                "target_iri": disease_iri,
                "mapping": disease_mapping,
            },
            "medications": medications,
            "labs": labs,
        })

    state.artifacts["enrichments"] = enrichments
    return state
