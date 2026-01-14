from __future__ import annotations

from typing import Dict

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL

_OWL_CACHE: Dict[str, Graph] = {}


def load_owl(path: str) -> Graph:
    """Loads an OWL file into an rdflib.Graph, using a cache."""
    if path in _OWL_CACHE:
        return _OWL_CACHE[path]
    g = Graph()
    g.parse(path)
    _OWL_CACHE[path] = g
    return g


def local_name(uri: URIRef) -> str:
    """Extracts the local name from a URI."""
    s = str(uri)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    return s.rsplit("/", 1)[-1]


def best_label(g: Graph, uri: URIRef) -> str:
    """
    Finds the best human-readable label for a URI, preferring rdfs:label
    and falling back to the local name.
    """
    for o in g.objects(uri, RDFS.label):
        if isinstance(o, Literal) and str(o).strip():
            return str(o).strip()
    return local_name(uri)


def is_owl_class(g: Graph, uri: URIRef) -> bool:
    """
    Checks if a URI represents an owl:Class, either by explicit type
    or by being the subject of an rdfs:subClassOf triple.
    """
    if (uri, RDF.type, OWL.Class) in g:
        return True
    if (uri, RDFS.subClassOf, None) in g:
        return True
    return False