from __future__ import annotations

from typing import Any, Dict, List, Optional

from ontocodex.kb.ontology_store import OntologyStore
from ontocodex.kb.terminology_store import TerminologyStore


class KnowledgeBase:
    """
    Unified Knowledge Base API wrapping TerminologyStore and OntologyStore.
    """
    def __init__(self, term_store: TerminologyStore, onto_store: OntologyStore):
        self.term_store = term_store
        self.onto_store = onto_store

    @classmethod
    def from_local_data(cls, data_dir: str, enable_vector: bool = False) -> KnowledgeBase:
        # Initialize stores from data directory
        term_store = TerminologyStore.from_dir(data_dir)
        onto_store = OntologyStore.from_dir(data_dir)
        return cls(term_store, onto_store)

    def lookup(self, term: str, system: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
        """
        Federated lookup across terminology and ontology stores.
        """
        hits = []

        # 1. Terminology Store Lookup
        hits.extend(self.term_store.lookup(term, system=system, k=k))

        # 2. Ontology Store Lookup (simple fallback or concurrent search)
        hits.extend(self.onto_store.lookup(term, k=k))

        # Sort combined hits by score
        hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return hits[:k]