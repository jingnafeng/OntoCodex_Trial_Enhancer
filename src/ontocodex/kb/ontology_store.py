from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL

from ontocodex.kb.evidence import Evidence


class OntologyStore:
    """
    Minimal ontology store for label-based lookup and basic traversal.
    """

    def __init__(self, graph: Graph, source_files: Optional[List[str]] = None) -> None:
        self._graph = graph
        self._source_files = source_files or []

    @classmethod
    def from_dir(cls, data_dir: str = "data") -> "OntologyStore":
        g = Graph()
        source_files: List[str] = []

        if not os.path.isdir(data_dir):
            return cls(graph=g, source_files=source_files)

        for fn in os.listdir(data_dir):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in {".owl", ".obo", ".ttl", ".rdf", ".xml"}:
                continue
            path = os.path.join(data_dir, fn)
            if not os.path.isfile(path):
                continue
            try:
                g.parse(path)
                source_files.append(fn)
            except Exception:
                continue

        return cls(graph=g, source_files=source_files)

    def lookup(self, term: str, k: int = 5) -> List[Dict[str, Any]]:
        if not term or not str(term).strip():
            return []

        term_norm = str(term).strip().lower()
        hits: List[Dict[str, Any]] = []

        for subj, label in self._graph.subject_objects(RDFS.label):
            if not isinstance(label, Literal):
                continue
            label_str = str(label).strip()
            label_norm = label_str.lower()
            if term_norm == label_norm:
                score = 1.0
            elif term_norm in label_norm or label_norm in term_norm:
                score = 0.7
            else:
                continue

            evidence = Evidence(
                source_type="ontology",
                source_file=self._source_files[0] if self._source_files else "ontology_store",
                id=str(subj),
                field="rdfs:label",
                snippet=label_str,
                extra={},
            ).to_dict()

            hits.append({
                "kind": "ontology",
                "term": label_str,
                "iri": str(subj),
                "score": score,
                "evidence": evidence,
            })

        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[: max(1, k)]

    def get_entity(self, iri: str) -> Optional[Dict[str, Any]]:
        if not iri:
            return None
        uri = URIRef(iri)
        label = None
        for o in self._graph.objects(uri, RDFS.label):
            if isinstance(o, Literal) and str(o).strip():
                label = str(o).strip()
                break
        if label is None:
            return None
        return {"iri": iri, "label": label}

    def get_parents(self, iri: str, depth: int = 1) -> List[str]:
        if not iri:
            return []
        parents: List[str] = []
        frontier = {URIRef(iri)}
        for _ in range(max(1, depth)):
            next_frontier = set()
            for node in frontier:
                for p in self._graph.objects(node, RDFS.subClassOf):
                    if isinstance(p, URIRef):
                        parents.append(str(p))
                        next_frontier.add(p)
            frontier = next_frontier
            if not frontier:
                break
        return parents

    def get_children(self, iri: str, depth: int = 1) -> List[str]:
        if not iri:
            return []
        children: List[str] = []
        frontier = {URIRef(iri)}
        for _ in range(max(1, depth)):
            next_frontier = set()
            for node in frontier:
                for child in self._graph.subjects(RDFS.subClassOf, node):
                    if isinstance(child, URIRef):
                        children.append(str(child))
                        next_frontier.add(child)
            frontier = next_frontier
            if not frontier:
                break
        return children
