from __future__ import annotations

from typing import Dict, List, Sequence

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL

from ontocodex.engine.state import OntoCodexState
from ontocodex.agents.owl_utils import best_label, is_owl_class, load_owl


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _gather_class_iris(g: Graph) -> List[URIRef]:
    iris = set(g.subjects(RDF.type, OWL.Class))
    iris.update(s for s, _, _ in g.triples((None, RDFS.subClassOf, None)) if isinstance(s, URIRef))
    return list(iris)


def _candidates_from_iris(g: Graph, iris: Sequence[str]) -> List[dict]:
    out: List[dict] = []
    for iri in iris:
        uri = URIRef(iri)
        if not is_owl_class(g, uri):
            continue
        label = best_label(g, uri)
        out.append({
            "target_iri": str(uri),
            "label": label,
            "term": label,
        })
    return out


def ontology_reader_node(state: OntoCodexState) -> OntoCodexState:
    """
    Load ontology and build candidates for downstream mapping.

    Options:
      - target_iris: explicit list of IRIs to process
      - candidate_terms: list of terms to match to ontology labels
      - max_candidates: cap for full-ontology scan
    """
    if not state.ontology_path:
        state.errors.append("OntologyReaderAgent: ontology_path is missing.")
        return state

    g = load_owl(state.ontology_path)

    target_iris = state.options.get("target_iris")
    candidate_terms = state.options.get("candidate_terms")
    max_candidates = int(state.options.get("max_candidates", 200))

    candidates: List[dict] = []

    if target_iris:
        candidates = _candidates_from_iris(g, target_iris)
    elif candidate_terms:
        label_index: Dict[str, List[str]] = {}
        for uri in _gather_class_iris(g):
            if not is_owl_class(g, uri):
                continue
            label = best_label(g, uri)
            label_index.setdefault(_norm(label), []).append(str(uri))

        for term in candidate_terms:
            term_norm = _norm(term)
            iris = label_index.get(term_norm, [])
            if not iris:
                state.warnings.append(f"OntologyReaderAgent: no ontology match for term '{term}'.")
                candidates.append({"term": term, "label": term, "target_iri": None})
                continue
            candidates.extend(_candidates_from_iris(g, iris))
    else:
        for uri in _gather_class_iris(g)[:max_candidates]:
            if not is_owl_class(g, uri):
                continue
            label = best_label(g, uri)
            candidates.append({
                "target_iri": str(uri),
                "label": label,
                "term": label,
            })

    state.candidates = candidates
    state.ontology_summary = {
        "candidate_count": len(candidates),
    }
    return state
