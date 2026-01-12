from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL

from ontocodex.engine.state import OntoCodexState
from ontocodex.kb.kb_api import KnowledgeBase

# Lazy singleton KB + ontology graph cache (local server friendly)
_KB: Optional[KnowledgeBase] = None
_OWL_CACHE: Dict[str, Graph] = {}


def _get_kb(data_dir: str = "data") -> KnowledgeBase:
    global _KB
    if _KB is None:
        _KB = KnowledgeBase.from_local_data(data_dir=data_dir, enable_vector=False)
    return _KB


def _load_owl(path: str) -> Graph:
    if path in _OWL_CACHE:
        return _OWL_CACHE[path]
    g = Graph()
    g.parse(path)  # rdflib infers format from extension/content
    _OWL_CACHE[path] = g
    return g


def _local_name(uri: URIRef) -> str:
    s = str(uri)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    return s.rsplit("/", 1)[-1]


def _best_label(g: Graph, uri: URIRef) -> str:
    # Prefer rdfs:label; fallback to localname
    for o in g.objects(uri, RDFS.label):
        if isinstance(o, Literal) and str(o).strip():
            return str(o).strip()
    return _local_name(uri)


def _is_owl_class(g: Graph, uri: URIRef) -> bool:
    # strict: typed owl:Class OR appears as a subclass subject
