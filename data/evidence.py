from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

@dataclass
class Evidence:
    source_type: str            # "csv" | "xlsx" | "owl" | "obo" | "ttl" | "vector"
    source_file: str
    id: str                     # row_id or IRI
    field: Optional[str] = None
    snippet: Optional[str] = None
    extra: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["extra"] = d["extra"] or {}
        return d
