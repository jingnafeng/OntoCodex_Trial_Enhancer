from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ontocodex.kb.terminology_store import jaccard, normalize_term


@dataclass
class TrialCodeRecord:
    term: str
    term_norm: str
    system: str
    code: str
    source_file: str
    row_id: str


class TrialCodesStore:
    """
    Lightweight dictionary store for trial concept -> ICD/CPT/RxNorm mappings.

    Expected CSV schema (flexible column names):
      - term (or concept/label/name)
      - system (e.g., ICD10CM, ICD10, CPT4, CPT, RXNORM)
      - code
      - synonyms (optional, ';' or '|' delimited)
    """

    def __init__(self) -> None:
        self._term_index: Dict[str, List[TrialCodeRecord]] = {}
        self._loaded_sources: List[str] = []

    @classmethod
    def from_dir(cls, data_dir: str = "data") -> "TrialCodesStore":
        store = cls()
        for fn in ("trial_codes.csv", "icd_dictionary.csv", "cpt_dictionary.csv"):
            path = os.path.join(data_dir, fn)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                store._load_csv(path)
            else:
                store._loaded_sources.append(f"SKIP (missing/empty): {fn}")
        return store

    def lookup(self, term: str, systems: Optional[Tuple[str, ...]] = None, k: int = 3) -> List[Dict[str, Any]]:
        q = normalize_term(term)
        if not q:
            return []
        target_systems = {s.strip().upper() for s in systems} if systems else None

        candidates = list(self._term_index.get(q, []))
        if not candidates:
            q_tokens = set(q.split())
            for key, recs in self._term_index.items():
                if jaccard(q_tokens, set(key.split())) >= 0.5:
                    candidates.extend(recs)

        if target_systems:
            candidates = [r for r in candidates if r.system in target_systems]

        out: List[Dict[str, Any]] = []
        q_tokens = set(q.split())
        for rec in candidates:
            score = 1.0 if rec.term_norm == q else jaccard(q_tokens, set(rec.term_norm.split()))
            out.append(
                {
                    "system": rec.system,
                    "code": rec.code,
                    "term": rec.term,
                    "score": float(score),
                    "evidence": {
                        "source_type": "trial_dictionary",
                        "source_file": rec.source_file,
                        "id": rec.row_id,
                        "field": "term/code",
                        "snippet": f"term={rec.term} | code={rec.system}:{rec.code}",
                        "extra": {"system": rec.system, "code": rec.code},
                    },
                }
            )
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[: max(1, k)]

    def _load_csv(self, path: str) -> None:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df.columns = [c.strip() for c in df.columns]

        term_col = self._pick_col(df.columns, ["term", "concept", "label", "name"])
        system_col = self._pick_col(df.columns, ["system", "vocabulary", "vocabulary_id"])
        code_col = self._pick_col(df.columns, ["code", "concept_code", "vocabulary_code"])
        syn_col = self._pick_col(df.columns, ["synonyms", "alias", "aliases"])

        if not term_col or not system_col or not code_col:
            self._loaded_sources.append(
                f"SKIP (schema mismatch): {os.path.basename(path)} "
                f"(term_col={term_col}, system_col={system_col}, code_col={code_col})"
            )
            return

        for i, row in df.iterrows():
            term = str(row.get(term_col, "")).strip()
            system = str(row.get(system_col, "")).strip().upper()
            code = str(row.get(code_col, "")).strip()
            if not term or not system or not code:
                continue

            all_terms = [term]
            if syn_col:
                syn = str(row.get(syn_col, "")).strip()
                if syn:
                    parts = [x.strip() for x in syn.replace("|", ";").split(";")]
                    all_terms.extend([p for p in parts if p])

            for t in all_terms:
                t_norm = normalize_term(t)
                if not t_norm:
                    continue
                rec = TrialCodeRecord(
                    term=t,
                    term_norm=t_norm,
                    system=system,
                    code=code,
                    source_file=os.path.basename(path),
                    row_id=str(i),
                )
                self._term_index.setdefault(t_norm, []).append(rec)

        self._loaded_sources.append(f"LOADED: {os.path.basename(path)} ({len(df)} rows)")

    @staticmethod
    def _pick_col(columns: List[str], candidates: List[str]) -> Optional[str]:
        col_map = {str(c).lower(): str(c) for c in columns}
        for cand in candidates:
            if cand.lower() in col_map:
                return col_map[cand.lower()]
        return None

