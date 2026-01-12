from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

# Namespaces
OMOP = Namespace("http://ontocodex.ai/omop#")  # your project namespace
OBOINOWL = Namespace("http://www.geneontology.org/formats/oboInOwl#")


@dataclass
class OmopAnnotationPayload:
    concept_id: str
    vocabulary: Optional[str] = None   # e.g., "LOINC", "SNOMEDCT", "RXNORM", "ATC"
    concept_code: Optional[str] = None # e.g., "10234-3"
    source: Optional[str] = None       # e.g., "loinc_omop.csv"


def load_graph(path: str) -> Graph:
    g = Graph()
    # RDFLib guesses format from extension; if needed you can pass format="xml"/"turtle"/"n3"
    g.parse(path)
    return g


def ensure_annotation_properties(g: Graph) -> None:
    """
    Declare OMOP annotation properties (optional but nice).
    OWL doesn’t require explicit declaration, but it’s cleaner.
    """
    for p in [OMOP.concept_id, OMOP.vocabulary, OMOP.concept_code, OMOP.source]:
        g.add((p, RDF.type, OWL.AnnotationProperty))


def add_omop_annotations(g: Graph, entity_iri: str, payload: OmopAnnotationPayload) -> None:
    """
    Attach OMOP mapping metadata as OWL annotations to an entity (class/property/individual).
    """
    ensure_annotation_properties(g)

    ent = URIRef(entity_iri)

    if payload.concept_id:
        g.add((ent, OMOP.concept_id, Literal(str(payload.concept_id), datatype=XSD.string)))
        # OBO-style xref (optional, widely used)
        g.add((ent, OBOINOWL.hasDbXref, Literal(f"OMOP:{payload.concept_id}", datatype=XSD.string)))

    if payload.vocabulary:
        vocab = str(payload.vocabulary).strip().upper()
        g.add((ent, OMOP.vocabulary, Literal(vocab, datatype=XSD.string)))

    if payload.concept_code:
        code = str(payload.concept_code).strip()
        g.add((ent, OMOP.concept_code, Literal(code, datatype=XSD.string)))
        if payload.vocabulary:
            g.add((ent, OBOINOWL.hasDbXref, Literal(f"{payload.vocabulary.upper()}:{code}", datatype=XSD.string)))

    if payload.source:
        g.add((ent, OMOP.source, Literal(str(payload.source), datatype=XSD.string)))


def save_graph(g: Graph, out_path: str, fmt: str = "xml") -> None:
    """
    fmt:
      - "xml" for RDF/XML (.owl)
      - "turtle" for .ttl
    """
    g.serialize(destination=out_path, format=fmt)
