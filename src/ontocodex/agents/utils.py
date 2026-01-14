from __future__ import annotations

from typing import Optional

from ontocodex.kb.kb_api import KnowledgeBase

_KB: Optional[KnowledgeBase] = None


def get_kb(data_dir: str = "data") -> KnowledgeBase:
    """Lazy singleton for the KnowledgeBase."""
    global _KB
    if _KB is None:
        _KB = KnowledgeBase.from_local_data(data_dir=data_dir, enable_vector=False)
    return _KB