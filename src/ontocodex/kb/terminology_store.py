# TerminologyStore (Layer 1: CSV/XLSX deterministic)
# #load: *_omop.csv, LOINC_CUI.xlsx, medDRA.xlsx, MedDRA_CTCAE_mapping_v5.xlsx
# build normalized indices

# src/ontocodex/kb/terminology_store.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from ontocodex.kb.evidence import Evidence


# -------------------------
# Normalization utilities
# -------------------------

_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_term(text: str) -> str:
    """
    Normalize a term for lookup:
    - split camelCase
    - lowercase
    - replace non-alphanum with spaces
    - collapse whitespace
    """
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    s = _CAMEL_SPLIT_RE.sub(" ", s)
    s = s.lower()
    s = _NON_ALNUM_RE.sub(" ", s)
    s = " ".join(s.split())
    return s


def token_set(s: str) -> set[str]:
    return set(normalize_term(s).split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0


# -------------------------
# Store implementation
# -------------------------

@dataclass(frozen=True)
class TermRecord:
    system: str
    code: str
    term: str
    term_norm: str
    source_file: str
    row_id: str
    extra: Dict[str, Any]


class TerminologyStore:
    """
    Deterministic terminology lookup store for OntoCodex.

    Supports:
      - term -> top-k records (scored)
      - code -> record
      - optional: synonyms, concept_id, etc. via 'extra'

    Returns dict hits with Evidence objects serialized as dicts.
    """

    def __init__(self) -> None:
        # term_norm -> list[TermRecord]
        self._term_index: Dict[str, List[TermRecord]] = {}
        # code_key -> TermRecord
        self._code_index: Dict[str, TermRecord] = {}
        # for debugging/inspection
        self._loaded_sources: List[str] = []

    # -------------------------
    # Construction
    # -------------------------

    @classmethod
    def from_dir(cls, data_dir: str = "data") -> "TerminologyStore":
        store = cls()

        # 1) Load *_omop.csv files (primary deterministic mapping layer)
        csv_files = [
            "snomed_omop.csv",
            "rxnorm_omop.csv",
            "loinc_omop.csv",
            "atc_omop.csv",
            "measurement_omop.csv",
        ]
        for fn in csv_files:
            path = os.path.join(data_dir, fn)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                system = fn.split("_")[0].upper()  # SNOMED/RXNORM/LOINC/ATC/MEASUREMENT
                store._load_omop_csv(path=path, system=system)
            else:
                # non-fatal (some are empty in repo)
                store._loaded_sources.append(f"SKIP (missing/empty): {fn}")

        # 2) Optional XLSX loaders (best-effort; won't break if schema differs)
        xlsx_files = [
            "LOINC_CUI.xlsx",
            "medDRA.xlsx",
            "MedDRA_CTCAE_mapping_v5.xlsx",
        ]
        for fn in xlsx_files:
            path = os.path.join(data_dir, fn)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                store._load_xlsx_best_effort(path=path)
            else:
                store._loaded_sources.append(f"SKIP (missing/empty): {fn}")

        return store

    # -------------------------
    # Public API
    # -------------------------

    def lookup(self, term: str, system: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
        """
        Lookup a term (string) and return top-k candidate records.

        system: optional filter (e.g. "SNOMED", "RXNORM", "LOINC", "ATC").
        """
        if not term or not str(term).strip():
            return []

        q_norm = normalize_term(term)
        if not q_norm:
            return []

        # Candidate pools:
        candidates: List[TermRecord] = []

        # Exact normalized match
        candidates.extend(self._term_index.get(q_norm, []))

        # If no exact hits, do lightweight fuzzy candidates:
        # - find keys that share tokens with query (bounded scan over keys)
        # This is deterministic but can be costly if huge. For large stores, replace
        # with an inverted index or SQLite FTS. For now, keep it simple.
        if not candidates:
            q_tokens = set(q_norm.split())
            if q_tokens:
                # bounded scan: only evaluate up to N keys to avoid worst-case blowups
                # (tune as you scale)
                MAX_KEYS = 20000
                for i, key in enumerate(self._term_index.keys()):
                    if i > MAX_KEYS:
                        break
                    key_tokens = set(key.split())
                    if q_tokens & key_tokens:
                        candidates.extend(self._term_index[key])

        # Apply system filter
        if system:
            sys_norm = system.strip().upper()
            candidates = [r for r in candidates if r.system.upper() == sys_norm]

        if not candidates:
            return []

        # Score candidates
        scored: List[Tuple[float, TermRecord]] = []
        q_tokens = set(q_norm.split())
        for r in candidates:
            score = self._score_record(query_norm=q_norm, query_tokens=q_tokens, rec=r)
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: max(1, k)]

        return [self._to_hit_dict(score=s, rec=r, query=term) for s, r in top if s > 0]

    def lookup_code(self, code: str, system: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Reverse lookup by code.
        system: optional filter.
        """
        if not code or not str(code).strip():
            return None
        code_str = str(code).strip()

        if system:
            key = self._code_key(system=system, code=code_str)
            rec = self._code_index.get(key)
            if not rec:
                return None
            return self._to_hit_dict(score=1.0, rec=rec, query=code_str)

        # no system: try any system match (first)
        # Prefer exact code match by scanning keys ending with "|<code>"
        suffix = "|" + code_str
        for k, rec in self._code_index.items():
            if k.endswith(suffix):
                return self._to_hit_dict(score=1.0, rec=rec, query=code_str)

        return None
    
    def lookup_code_hierarchy(self, code: str, system: str = "ATC") -> List[Dict[str, Any]]:
        """
        Hierarchical reverse lookup for ATC codes.
        Returns hits for the code and its parent codes (if present in store).
        """
        sys_norm = (system or "").strip().upper()
        if sys_norm != "ATC":
            # For non-ATC systems, just do normal reverse lookup
            hit = self.lookup_code(code=code, system=system)
            return [hit] if hit else []

        hits: List[Dict[str, Any]] = []
        for c in self._atc_hierarchy_codes(code):
            h = self.lookup_code(code=c, system="ATC")
            if h:
                hits.append(h)
        return hits

    def stats(self) -> Dict[str, Any]:
        return {
            "loaded_sources": self._loaded_sources,
            "term_index_keys": len(self._term_index),
            "code_index_keys": len(self._code_index),
            "total_records": sum(len(v) for v in self._term_index.values()),
        }

    # -------------------------
    # Internal loaders
    # -------------------------

    def _load_omop_csv(self, path: str, system: str) -> None:
        """
        Robust loader for OMOP-ish CSV mapping files.

        Supports:
        - Standard OMOP concept export: concept_code + concept_name (+ concept_id, vocabulary_id)
        - Vocab-specific code column exports like your rxnorm_omop.csv:
            concept_id, concept_name, domain_id, concept_class_id, RxNorm
            where "RxNorm" is the RXCUI/code column.
        """
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df.columns = [c.strip() for c in df.columns]

        # 1) Prefer explicit vocab-named code columns if present (e.g., "RxNorm")
        explicit_vocab_cols = [
                # RxNorm
                "RxNorm", "RXNORM", "rxcui", "RXCUI",
                # SNOMED variants
                "snomed_ct_us", "SNOMED_CT_US", "snomed", "SNOMED",
                "snomedct", "SNOMEDCT",
                # LOINC / ATC / NDC
                "loinc", "LOINC",
                "atc", "ATC",
                "ndc", "NDC",
            ]

        code_col = None
        for c in explicit_vocab_cols:
            if c in df.columns:
                code_col = c
                break

        # 2) Fallback to generic OMOP-ish code column names
        if code_col is None:
            code_col = self._pick_col(df.columns, [
                "concept_code", "code", "vocabulary_code", "conceptcode"
            ])

        # 3) Term/name column (must exist)
        term_col = self._pick_col(df.columns, [
            "concept_name", "name", "term", "label", "conceptname"
        ])

        # 4) Optional metadata columns
        concept_id_col = self._pick_col(df.columns, [
            "concept_id", "conceptid", "omop_concept_id"
        ])
        domain_col = self._pick_col(df.columns, ["domain_id", "domain"])
        class_col = self._pick_col(df.columns, ["concept_class_id", "concept_class"])
        vocab_col = self._pick_col(df.columns, ["vocabulary_id", "vocabulary", "system"])
        standard_col = self._pick_col(df.columns, ["standard_concept", "standardconcept"])


        # Validate essential columns
        if code_col is None or term_col is None:
            self._loaded_sources.append(
                f"SKIP (unrecognized OMOP CSV schema): {os.path.basename(path)} "
                f"(code_col={code_col}, term_col={term_col}, cols={list(df.columns)})"
            )
            return

        for idx, row in df.iterrows():
            code = str(row.get(code_col, "")).strip()
            term = str(row.get(term_col, "")).strip()
            if not code or not term:
                continue

            # Determine system:
            # (a) If the code column is literally a vocab name (RxNorm/SNOMED/LOINC/ATC), use that.
            # (b) Else if vocabulary_id/system column exists, use it.
            # (c) Else fallback to filename-derived system passed in.
            row_system = system.strip().upper()
            col = code_col.lower()
            if col in {"rxnorm", "rxcui"}:
                row_system = "RXNORM"
            elif col in {"snomed", "snomedct", "snomed_ct_us"}:
                row_system = "SNOMEDCT"
            elif col == "loinc":
                row_system = "LOINC"
            elif col == "atc":
                row_system = "ATC"
            elif col == "ndc":
                row_system = "NDC"

            elif vocab_col:
                v = str(row.get(vocab_col, "")).strip()
                if v:
                    row_system = v.upper()

            term_norm = normalize_term(term)
            if not term_norm:
                continue

            extra: Dict[str, Any] = {}

            if concept_id_col:
                cid = str(row.get(concept_id_col, "")).strip()
                if cid:
                    extra["concept_id"] = cid
            if domain_col:
                dv = str(row.get(domain_col, "")).strip()
                if dv:
                    extra["domain_id"] = dv
            if class_col:
                cv = str(row.get(class_col, "")).strip()
                if cv:
                    extra["concept_class_id"] = cv
            if standard_col:
                sv = str(row.get(standard_col, "")).strip()
                if sv:
                    extra["standard_concept"] = sv


            rec = TermRecord(
                system=row_system,
                code=code,
                term=term,
                term_norm=term_norm,
                source_file=os.path.basename(path),
                row_id=str(idx),
                extra=extra,
            )
            self._add_record(rec)

        self._loaded_sources.append(f"LOADED: {os.path.basename(path)} ({len(df)} rows)")


    def _load_xlsx_best_effort(self, path: str) -> None:
        """
        Best-effort XLSX loader.
        We do not know your exact sheets/columns, so we:
          - read all sheets
          - try to detect (term, code, system) columns
          - index if we can
        """
        xls = pd.ExcelFile(path)
        loaded_any = False

        for sheet in xls.sheet_names:
            try:
                df = xls.parse(sheet_name=sheet, dtype=str, keep_default_na=False)
            except Exception:
                continue

            df.columns = [str(c).strip() for c in df.columns]

            # Heuristics for term/code columns
            code_col = self._pick_col(df.columns, ["code", "concept_code", "loinc", "rxnorm", "snomed", "meddra", "cui"])
            term_col = self._pick_col(df.columns, ["term", "name", "label", "concept_name", "preferred_term", "pt", "llt"])

            if code_col is None or term_col is None:
                # Try alternative: two-code mapping sheets (e.g., MedDRA <-> CTCAE)
                # We won't index those generically without known semantics.
                continue

            # Try infer system from filename or sheet name
            base = os.path.basename(path).lower()
            system = "UNKNOWN"
            if "loinc" in base:
                system = "LOINC"
            elif "rxnorm" in base:
                system = "RXNORM"
            elif "snomed" in base:
                system = "SNOMED"
            elif "meddra" in base:
                system = "MEDDRA"

            for idx, row in df.iterrows():
                code = str(row.get(code_col, "")).strip()
                term = str(row.get(term_col, "")).strip()
                if not code or not term:
                    continue
                term_norm = normalize_term(term)
                if not term_norm:
                    continue

                rec = TermRecord(
                    system=system,
                    code=code,
                    term=term,
                    term_norm=term_norm,
                    source_file=os.path.basename(path) + f"#{sheet}",
                    row_id=str(idx),
                    extra={},
                )
                self._add_record(rec)
                loaded_any = True

        if loaded_any:
            self._loaded_sources.append(f"LOADED: {os.path.basename(path)} (best-effort)")
        else:
            self._loaded_sources.append(f"SKIP (no recognizable sheets): {os.path.basename(path)}")

    # -------------------------
    # Internal indexing helpers
    # -------------------------

    def _add_record(self, rec: TermRecord) -> None:
        self._term_index.setdefault(rec.term_norm, []).append(rec)
        self._code_index[self._code_key(system=rec.system, code=rec.code)] = rec

    @staticmethod
    def _code_key(system: str, code: str) -> str:
        return f"{system.strip().upper()}|{str(code).strip()}"
    
    @staticmethod
    def _atc_hierarchy_codes(code: str) -> List[str]:
        """
        ATC hierarchy: A, A01, A01A, A01AB, ...
        Return code plus all parents by truncating from the end.
        Example: "A01" -> ["A01", "A"]
                 "A01AB" -> ["A01AB", "A01A", "A01", "A"]
        """
        c = (code or "").strip().upper()
        if not c:
            return []
        parents = [c]
        # Keep truncating until length 1 (top level, e.g. "A")
        while len(c) > 1:
            c = c[:-1]
            parents.append(c)
        # De-dup while preserving order
        out = []
        seen = set()
        for x in parents:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out


    @staticmethod
    def _pick_col(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
        """
        Pick the first matching column name from candidates (case-insensitive).
        """
        col_map = {c.lower(): c for c in columns}
        for cand in candidates:
            if cand.lower() in col_map:
                return col_map[cand.lower()]
        return None

    @staticmethod
    def _score_record(query_norm: str, query_tokens: set[str], rec: TermRecord) -> float:
        """
        Deterministic scoring:
          - exact normalized match => 1.0
          - token Jaccard => [0..1)
          - small bonus if term startswith query or vice versa
        """
        if rec.term_norm == query_norm:
            return 1.0

        rec_tokens = set(rec.term_norm.split())
        base = jaccard(query_tokens, rec_tokens)

        # Prefix bonus for close variants
        if rec.term_norm.startswith(query_norm) or query_norm.startswith(rec.term_norm):
            base = min(1.0, base + 0.15)

        # Very short queries are risky; dampen slightly unless strong overlap
        if len(query_tokens) <= 1 and base < 0.8:
            base *= 0.75

        return float(base)

    @staticmethod
    def _to_hit_dict(score: float, rec: TermRecord, query: str) -> Dict[str, Any]:
        snippet = f"term={rec.term} | code={rec.system}:{rec.code}"
        ev = Evidence(
            source_type="csv/xlsx",
            source_file=rec.source_file,
            id=rec.row_id,
            field="term/code",
            snippet=snippet,
            extra={"system": rec.system, "code": rec.code, **(rec.extra or {})},
        ).to_dict()

        return {
            "kind": "terminology",
            "query": query,
            "term": rec.term,
            "system": rec.system,
            "code": rec.code,
            "score": float(score),
            "evidence": ev,
            "extra": rec.extra or {},
        }
