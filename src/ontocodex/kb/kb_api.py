# src/ontocodex/kb/kb_api.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from ontocodex.kb.terminology_store import TerminologyStore
from ontocodex.kb.ontology_store import OntologyStore


class VectorStore(Protocol):
    """Optional semantic fallback interface."""
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        ...


@dataclass
class KnowledgeBase:
    """
    Unified KnowledgeBase facade for OntoCodex.

    Retrieval order (strict):
      1) TerminologyStore (deterministic CSV/XLSX mappings)
      2) OntologyStore (graph truth from OWL/OBO/TTL)
      3) VectorStore (optional semantic fallback; NOT authoritative)
    """
    term_store: TerminologyStore
    onto_store: OntologyStore
    vector_store: Optional[VectorStore] = None

    @classmethod
    def from_local_data(
        cls,
        data_dir: str = "data",
        enable_vector: bool = False,
        vector_index_dir: Optional[str] = None,
    ) -> "KnowledgeBase":
        """
        Construct KB from local filesystem.

        - data_dir points to your OntoCodexFramework/data folder (or a mounted/symlinked copy).
        - enable_vector loads vector_store if you’ve implemented it; otherwise ignored.
        """
        term_store = TerminologyStore.from_dir(data_dir)
        onto_store = OntologyStore.from_dir(data_dir)

        vector_store: Optional[VectorStore] = None
        if enable_vector:
            # Lazy import so you can start without vectors
            from ontocodex.kb.vector_store import LocalVectorStore  # you’ll implement later
            vector_store = LocalVectorStore.from_dir(
                data_dir=data_dir,
                index_dir=vector_index_dir or f"{data_dir}/.vector_index",
            )

        return cls(term_store=term_store, onto_store=onto_store, vector_store=vector_store)

    def lookup(
        self,
        term: str,
        system: Optional[str] = None,
        k: int = 5,
        allow_vector_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Lookup a term and return a list of candidate hits.

        Expected hit format (recommended):
          {
            "kind": "terminology" | "ontology" | "vector",
            "term": "...",
            "system": "...",              # for terminology hits
            "code": "...",                # for terminology hits
            "iri": "...",                 # for ontology hits
            "score": 0.0..1.0,
            "evidence": {...}             # provenance object (source_file, id/iri, snippet)
          }
        """
        # 1) deterministic terminology mapping
        hits = self.term_store.lookup(term=term, system=system, k=k)
        if hits:
            return hits

        # 2) ontology label/synonym lookup
        hits = self.onto_store.lookup(term=term, k=k)
        if hits:
            return hits

        # 3) optional semantic fallback
        if allow_vector_fallback and self.vector_store is not None:
            hits = self.vector_store.search(query=term, k=k)
            if hits:
                return hits

        return []

    def lookup_code(
        self,
        code: str,
        system: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Reverse lookup for terminology codes (e.g., SNOMED/RxNorm/LOINC/ATC).
        """
        return self.term_store.lookup_code(code=code, system=system)

    def get_ontology_entity(
        self,
        iri: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch annotations for an ontology entity (label, definition, synonyms, xrefs).
        """
        return self.onto_store.get_entity(iri)

    def get_parents(self, iri: str, depth: int = 1) -> List[str]:
        return self.onto_store.get_parents(iri=iri, depth=depth)

    def get_children(self, iri: str, depth: int = 1) -> List[str]:
        return self.onto_store.get_children(iri=iri, depth=depth)
