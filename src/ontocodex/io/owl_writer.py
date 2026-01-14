from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, XSD

# Namespace for OntoCodex annotations
ONTOCODEX = Namespace("http://ontocodex.org/schema/")


@dataclass
class OmopAnnotationPayload:
    """Payload for OMOP mapping annotations."""
    concept_id: str
    vocabulary: Optional[str] = None
    concept_code: Optional[str] = None
    source: Optional[str] = None


def load_graph(path: str) -> Graph:
    """
    Loads an ontology into an rdflib.Graph.
    Unlike cached reader utilities, this returns a fresh graph instance
    suitable for modification.
    """
    g = Graph()
    g.parse(path)
    return g


def save_graph(g: Graph, path: str, fmt: str = "xml") -> None:
    """Saves the graph to the specified path in the given format."""
    g.serialize(destination=path, format=fmt)


def add_omop_annotations(g: Graph, entity_iri: str, payload: OmopAnnotationPayload) -> None:
    """
    Adds OMOP-related annotation properties to the specified entity IRI in the graph.
    """
    subject = URIRef(entity_iri)

    # Define properties
    prop_concept_id = ONTOCODEX.omop_concept_id
    prop_vocab_id = ONTOCODEX.omop_vocabulary_id
    prop_concept_code = ONTOCODEX.omop_concept_code
    prop_source = ONTOCODEX.mapping_source

    # Helper to add annotation property declaration and value
    def _add_ann(prop: URIRef, value: Optional[str]):
        if value:
            g.add((prop, RDF.type, OWL.AnnotationProperty))
            g.add((subject, prop, Literal(value, datatype=XSD.string)))

    _add_ann(prop_concept_id, payload.concept_id)
    _add_ann(prop_vocab_id, payload.vocabulary)
    _add_ann(prop_concept_code, payload.concept_code)
    _add_ann(prop_source, payload.source)