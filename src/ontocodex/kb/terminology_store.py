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


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0


# -------------------------
# Data record
# -------------------------

@dataclass
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

    Primary sources:
      - *_omop.csv (RxNorm/SNOMED/LOINC/ATC/measurement)
    Enrichment source for LOINC:
      - LOINC_CUI.csv (Class ID / Preferred Label / Synonyms / CUI)
    """

    def __init__(self) -> None:
        # term_norm -> list[TermRecord]
        self._term_index: Dict[str, List[TermRecord]] = {}
        # system|code -> TermRecord
        self._code_index: Dict[str, TermRecord] = {}
        self._loaded_sources: List[str] = []

        # LOINC enrichment
        self._loinc_meta: Dict[str, Dict[str, Any]] = {}
        self._loinc_syn_index: Dict[str, List[str]] = {}  # norm_syn -> [loinc_code]
        # Mayo CPT quick index: normalized test name -> CPT TermRecord list
        self._mayo_cpt_index: Dict[str, List[TermRecord]] = {}

    # -------------------------
    # Construction
    # -------------------------

    @classmethod
    def from_dir(cls, data_dir: str = "data") -> "TerminologyStore":
        store = cls()

        # 1) Load OMOP mapping CSVs (authoritative deterministic layer)
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
                # initial system label from filename (fallback only)
                system = fn.split("_")[0].upper()
                store._load_omop_csv(path=path, system=system)
            else:
                store._loaded_sources.append(f"SKIP (missing/empty): {fn}")

        # 2) Load LOINC_CUI enrichment CSV (optional)
        loinc_cui_csv = os.path.join(data_dir, "LOINC_CUI.csv")
        if os.path.exists(loinc_cui_csv) and os.path.getsize(loinc_cui_csv) > 0:
            store._load_loinc_cui_csv(loinc_cui_csv)
        else:
            store._loaded_sources.append("SKIP (missing/empty): LOINC_CUI.csv")

        # 3) Mayo CPT reference (first-line for lab/procedure lookups)
        mayo_cpt_csv = os.path.join(data_dir, "cpt-codes-mayo.csv")
        if os.path.exists(mayo_cpt_csv) and os.path.getsize(mayo_cpt_csv) > 0:
            store._load_mayo_cpt_csv(mayo_cpt_csv)
        else:
            store._loaded_sources.append("SKIP (missing/empty): cpt-codes-mayo.csv")

        # 4) Optional XLSX (best-effort; safe to keep but not required)
        for fn in ["medDRA.xlsx", "MedDRA_CTCAE_mapping_v5.xlsx"]:
            path = os.path.join(data_dir, fn)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                store._load_xlsx_best_effort(path)
            else:
                store._loaded_sources.append(f"SKIP (missing/empty): {fn}")

        # 5) Post-process: attach LOINC CUI metadata to any LOINC records loaded from loinc_omop.csv
        store._attach_loinc_meta_to_existing_records()

        return store

    # -------------------------
    # Public API
    # -------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "loaded_sources": self._loaded_sources,
            "term_index_keys": len(self._term_index),
            "code_index_keys": len(self._code_index),
            "total_records": sum(len(v) for v in self._term_index.values()),
            "loinc_meta_rows": len(self._loinc_meta),
            "loinc_syn_index_keys": len(self._loinc_syn_index),
        }

    def lookup(self, term: str, system: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
        """
        Deterministic lookup. For LOINC, optionally fallback to LOINC_CUI synonyms
        if no OMOP hit.
        """
        if not term or not str(term).strip():
            return []

        q_norm = normalize_term(term)
        if not q_norm:
            return []

        sys_norm = self._normalize_system(system) if system else None

        # First-line Mayo CPT reference for lab tests / diagnostic procedures.
        mayo_hits = self._lookup_mayo_cpt(term=term, system=sys_norm, k=k)
        if mayo_hits:
            return mayo_hits

        # 1) Exact normalized hits
        candidates: List[TermRecord] = []
        candidates.extend(self._term_index.get(q_norm, []))

        # 2) Lightweight fuzzy candidates if exact missing (bounded scan)
        if not candidates:
            q_tokens = set(q_norm.split())
            if q_tokens:
                MAX_KEYS = 20000
                for i, key in enumerate(self._term_index.keys()):
                    if i > MAX_KEYS:
                        break
                    key_tokens = set(key.split())
                    if q_tokens & key_tokens:
                        candidates.extend(self._term_index[key])

        # System filter
        if sys_norm:
            candidates = [r for r in candidates if r.system.upper() == sys_norm]

        if candidates:
            # Score and return
            scored: List[Tuple[float, TermRecord]] = []
            q_tokens = set(q_norm.split())
            for r in candidates:
                score = self._score_record(query_norm=q_norm, query_tokens=q_tokens, rec=r)
                scored.append((score, r))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[: max(1, k)]
            return [self._to_hit_dict(score=s, rec=r, query=term) for s, r in top if s > 0]

        # 3) LOINC synonym fallback (optional): only if system is LOINC or not specified
        if (sys_norm is None or sys_norm == "LOINC") and self._loinc_syn_index:
            codes = self._loinc_syn_index.get(q_norm, [])
            hits: List[Dict[str, Any]] = []
            for code in codes[: max(1, k)]:
                meta = self._loinc_meta.get(code, {})
                ev = Evidence(
                    source_type="csv",
                    source_file=meta.get("source_file", "LOINC_CUI.csv"),
                    id=str(meta.get("row_id", "")),
                    field="Preferred Label/Synonyms",
                    snippet=f"LOINC={code} | CUI={meta.get('cui','')}",
                    extra={"system": "LOINC", "code": code, "cui": meta.get("cui")},
                ).to_dict()

                hits.append({
                    "kind": "terminology",
                    "query": term,
                    "term": meta.get("preferred_label", term),
                    "system": "LOINC",
                    "code": code,
                    "score": 0.95,  # exact synonym normalization match
                    "evidence": ev,
                    "extra": meta,
                })
            if hits:
                return hits

        return []

    def _lookup_mayo_cpt(self, term: str, system: Optional[str], k: int) -> List[Dict[str, Any]]:
        q_norm = normalize_term(term)
        if not q_norm or not self._mayo_cpt_index:
            return []

        # Use Mayo CPT first for explicit CPT queries or lab/procedure-like phrases.
        q_tokens = set(q_norm.split())
        is_cpt_query = system in {"CPT", "CPT4", None}
        procedure_markers = {
            "lab", "test", "assay", "panel", "diagnostic", "diagnosis",
            "procedure", "infusion", "biopsy", "screen", "screening",
        }
        likely_procedure = bool(q_tokens & procedure_markers)
        if not (is_cpt_query and likely_procedure) and system not in {"CPT", "CPT4"}:
            return []

        candidates: List[TermRecord] = []
        candidates.extend(self._mayo_cpt_index.get(q_norm, []))
        if not candidates:
            for key, recs in self._mayo_cpt_index.items():
                if q_tokens & set(key.split()):
                    candidates.extend(recs)

        if not candidates:
            return []

        scored: List[Tuple[float, TermRecord]] = []
        for r in candidates:
            score = self._score_record(query_norm=q_norm, query_tokens=q_tokens, rec=r)
            # Mayo dictionary is designated first-line for this use-case.
            score = min(1.0, score + 0.1)
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: max(1, k)]
        return [self._to_hit_dict(score=s, rec=r, query=term) for s, r in top if s > 0]

    def lookup_code(self, code: str, system: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Reverse lookup by code.
        """
        if not code or not str(code).strip():
            return None
        code_str = str(code).strip()

        sys_norm = self._normalize_system(system) if system else None
        if sys_norm == "LOINC":
            code_str = self._normalize_loinc_code(code_str)

        if sys_norm:
            rec = self._code_index.get(self._code_key(system=sys_norm, code=code_str))
            if not rec:
                return None
            return self._to_hit_dict(score=1.0, rec=rec, query=code_str)

        # no system: find first match by suffix
        suffix = "|" + code_str
        for k, rec in self._code_index.items():
            if k.endswith(suffix):
                return self._to_hit_dict(score=1.0, rec=rec, query=code_str)

        return None

    # -------------------------
    # OMOP CSV Loader
    # -------------------------

    def _load_omop_csv(self, path: str, system: str) -> None:
        """
        Robust loader for OMOP-ish CSV mapping files.

        Handles your schemas:
          RxNorm: concept_id, concept_name, domain_id, concept_class_id, RxNorm
          SNOMED: concept_id, concept_name, domain_id, concept_class_id, standard_concept, snomed_ct_us
          ATC:    concept_id, concept_name, domain_id, concept_class_id, ATC
          LOINC:  concept_id, concept_name, domain_id, concept_class_id, LOINC
        """
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df.columns = [c.strip() for c in df.columns]

        # Prefer explicit vocab-named code columns if present
        explicit_vocab_cols = [
            "RxNorm", "RXNORM", "rxcui", "RXCUI",
            "snomed_ct_us", "SNOMED_CT_US", "snomed", "SNOMED", "snomedct", "SNOMEDCT",
            "LOINC", "loinc", "loinc_code", "LOINC_CODE",
            "ATC", "atc",
            "NDC", "ndc",
        ]
        code_col = None
        for c in explicit_vocab_cols:
            if c in df.columns:
                code_col = c
                break

        # Fallback to generic OMOP-ish code column names
        if code_col is None:
            code_col = self._pick_col(df.columns, ["concept_code", "code", "vocabulary_code", "conceptcode"])

        term_col = self._pick_col(df.columns, ["concept_name", "name", "term", "label", "conceptname"])

        concept_id_col = self._pick_col(df.columns, ["concept_id", "conceptid", "omop_concept_id"])
        domain_col = self._pick_col(df.columns, ["domain_id", "domain"])
        class_col = self._pick_col(df.columns, ["concept_class_id", "concept_class"])
        vocab_col = self._pick_col(df.columns, ["vocabulary_id", "vocabulary", "system"])
        standard_col = self._pick_col(df.columns, ["standard_concept", "standardconcept"])

        if code_col is None or term_col is None:
            self._loaded_sources.append(
                f"SKIP (unrecognized OMOP CSV schema): {os.path.basename(path)} "
                f"(code_col={code_col}, term_col={term_col}, cols={list(df.columns)})"
            )
            return

        loaded = 0
        for idx, row in df.iterrows():
            code = str(row.get(code_col, "")).strip()
            term = str(row.get(term_col, "")).strip()
            if not code or not term:
                continue

            # Determine canonical system from code column name if applicable
            row_system = self._normalize_system(system)
            if code_col:
                col = code_col.lower()
                if col in {"rxnorm", "rxcui"}:
                    row_system = "RXNORM"
                elif col in {"snomed", "snomedct", "snomed_ct_us"}:
                    row_system = "SNOMEDCT"
                elif col in {"loinc", "loinc_code"}:
                    row_system = "LOINC"
                elif col == "atc":
                    row_system = "ATC"
                elif col == "ndc":
                    row_system = "NDC"

            # If still generic and vocabulary_id exists, use it
            if vocab_col and row_system in {"RXNORM", "SNOMED", "SNOMEDCT", "LOINC", "ATC", "NDC", "MEASUREMENT"} is False:
                v = str(row.get(vocab_col, "")).strip()
                if v:
                    row_system = self._normalize_system(v)

            if row_system == "LOINC":
                code = self._normalize_loinc_code(code)

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
            loaded += 1

        self._loaded_sources.append(f"LOADED: {os.path.basename(path)} (rows_indexed={loaded})")

    # -------------------------
    # LOINC_CUI Loader + attachment
    # -------------------------

    @staticmethod
    def _normalize_loinc_code(code: str) -> str:
        c = (code or "").strip()
        if not c:
            return ""
        if "-" in c:
            return c
        if c.isdigit() and len(c) >= 2:
            return f"{c[:-1]}-{c[-1]}"
        return c

    def _load_loinc_cui_csv(self, path: str) -> None:
        """
        Load LOINC_CUI.csv with columns:
          Class ID, Preferred Label, Synonyms, CUI

        Builds:
          - _loinc_meta[loinc_code] = {cui, preferred_label, synonyms, class_id, source_file, row_id}
          - _loinc_syn_index[normalize_term(label_or_syn)] -> [loinc_code]
        """
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df.columns = [str(c).strip() for c in df.columns]

        class_id_col = self._pick_col(df.columns, ["Class ID", "ClassID", "class_id", "class id"])
        pref_col = self._pick_col(df.columns, ["Preferred Label", "PreferredLabel", "preferred_label", "label"])
        syn_col = self._pick_col(df.columns, ["Synonyms", "synonyms", "Synonym"])
        cui_col = self._pick_col(df.columns, ["CUI", "cui"])

        if not class_id_col or not pref_col or not cui_col:
            self._loaded_sources.append(
                f"SKIP (LOINC_CUI schema mismatch): {os.path.basename(path)} (cols={list(df.columns)})"
            )
            return

        loaded = 0
        for idx, row in df.iterrows():
            class_id = str(row.get(class_id_col, "")).strip()
            if not class_id:
                continue

            loinc_code = class_id.rsplit("/", 1)[-1].strip()
            loinc_code = self._normalize_loinc_code(loinc_code)
            if not loinc_code:
                continue

            pref = str(row.get(pref_col, "")).strip()
            cui = str(row.get(cui_col, "")).strip()
            syn_raw = str(row.get(syn_col, "")).strip() if syn_col else ""
            synonyms = [s.strip() for s in syn_raw.split("|") if s.strip()] if syn_raw else []

            self._loinc_meta[loinc_code] = {
                "cui": cui,
                "preferred_label": pref,
                "synonyms": synonyms,
                "class_id": class_id,
                "source_file": os.path.basename(path),
                "row_id": str(idx),
            }

            # Build synonym index (optional fallback lookup)
            for s in ([pref] if pref else []) + synonyms:
                ns = normalize_term(s)
                if ns:
                    self._loinc_syn_index.setdefault(ns, []).append(loinc_code)

            loaded += 1

        self._loaded_sources.append(f"LOADED: {os.path.basename(path)} (loinc_meta_rows={loaded})")

    def _load_mayo_cpt_csv(self, path: str) -> None:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df.columns = [str(c).strip() for c in df.columns]

        term_col = self._pick_col(df.columns, ["Test Name", "test_name", "test name", "name", "term", "label"])
        code_col = self._pick_col(df.columns, ["CPT Code", "cpt_code", "cpt code", "code"])
        mayo_id_col = self._pick_col(df.columns, ["Mayo Test ID", "mayo_test_id", "mayo test id", "test_id"])

        if not term_col or not code_col:
            self._loaded_sources.append(
                f"SKIP (Mayo CPT schema mismatch): {os.path.basename(path)} "
                f"(term_col={term_col}, code_col={code_col}, cols={list(df.columns)})"
            )
            return

        loaded = 0
        for idx, row in df.iterrows():
            term = str(row.get(term_col, "")).strip()
            code_raw = str(row.get(code_col, "")).strip()
            if not term or not code_raw:
                continue
            term_norm = normalize_term(term)
            if not term_norm:
                continue

            code_entries = self._parse_mayo_cpt_entries(code_raw)
            if not code_entries:
                continue

            extra = {}
            if mayo_id_col:
                mtid = str(row.get(mayo_id_col, "")).strip()
                if mtid:
                    extra["mayo_test_id"] = mtid
            if len(code_entries) > 1:
                extra["mayo_panel"] = True

            for i, entry in enumerate(code_entries):
                rec_extra = dict(extra)
                if entry.get("detail"):
                    rec_extra["cpt_detail"] = entry["detail"]
                if entry.get("multiplier"):
                    rec_extra["multiplier"] = entry["multiplier"]
                rec_extra["panel_index"] = i + 1

                rec = TermRecord(
                    system="CPT4",
                    code=entry["code"],
                    term=term,
                    term_norm=term_norm,
                    source_file=os.path.basename(path),
                    row_id=f"{idx}:{i}",
                    extra=rec_extra,
                )
                self._mayo_cpt_index.setdefault(term_norm, []).append(rec)
                self._add_record(rec)
                loaded += 1
        self._loaded_sources.append(f"LOADED: {os.path.basename(path)} (rows_indexed={loaded})")

    @staticmethod
    def _parse_mayo_cpt_entries(code_raw: str) -> List[Dict[str, Any]]:
        """
        Parse Mayo CPT field that may contain:
          - single code: "83916"
          - single code with quantity: "86765 x 2"
          - multiline code-detail panel:
              "88184-Flow cytometry ...\\n88185-..."
        Returns unique entries preserving first-seen order.
        """
        if not code_raw:
            return []

        text = str(code_raw).replace("\r\n", "\n").replace("\r", "\n")
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if not lines:
            lines = [text.strip()]

        out: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()

        for ln in lines:
            codes = re.findall(r"\b(\d{5})\b", ln)
            if not codes:
                continue

            # Prefer first code in a line if multiple are present.
            code = codes[0]
            detail = ""
            if "-" in ln:
                detail = ln.split("-", 1)[1].strip()
            mult = None
            m = re.search(r"\bx\s*(\d+)\b", ln, flags=re.IGNORECASE)
            if m:
                try:
                    mult = int(m.group(1))
                except Exception:
                    mult = None

            key = (code, detail)
            if key in seen:
                continue
            seen.add(key)
            out.append({"code": code, "detail": detail, "multiplier": mult})

        # If parser failed on line-wise pass, fallback to any unique 5-digit code in full text.
        if not out:
            for code in re.findall(r"\b(\d{5})\b", text):
                key = (code, "")
                if key in seen:
                    continue
                seen.add(key)
                out.append({"code": code, "detail": "", "multiplier": None})

        return out

    def _attach_loinc_meta_to_existing_records(self) -> None:
        """
        After both loinc_omop.csv and LOINC_CUI.csv are loaded, attach CUI/label info to LOINC records.
        """
        if not self._loinc_meta:
            return

        # Update code_index records for LOINC
        for key, rec in list(self._code_index.items()):
            if rec.system != "LOINC":
                continue
            meta = self._loinc_meta.get(rec.code)
            if not meta:
                continue
            # attach lightweight metadata
            rec.extra.setdefault("cui", meta.get("cui"))
            rec.extra.setdefault("preferred_label", meta.get("preferred_label"))
            rec.extra.setdefault("synonym_count", len(meta.get("synonyms") or []))
            rec.extra.setdefault("loinc_class_id", meta.get("class_id"))

    # -------------------------
    # XLSX best-effort loader (optional)
    # -------------------------

    def _load_xlsx_best_effort(self, path: str) -> None:
        """
        Best-effort XLSX loader.
        Not required for LOINC OMOP/CUI flow; safe to keep for MedDRA artifacts.
        """
        xls = pd.ExcelFile(path)
        loaded_any = False

        for sheet in xls.sheet_names:
            try:
                df = xls.parse(sheet_name=sheet, dtype=str, keep_default_na=False)
            except Exception:
                continue

            df.columns = [str(c).strip() for c in df.columns]
            code_col = self._pick_col(df.columns, ["code", "concept_code", "meddra", "cui", "loinc"])
            term_col = self._pick_col(df.columns, ["term", "name", "label", "concept_name", "preferred_term", "pt", "llt"])

            if code_col is None or term_col is None:
                continue

            base = os.path.basename(path).lower()
            system = "UNKNOWN"
            if "meddra" in base:
                system = "MEDDRA"
            elif "loinc" in base:
                system = "LOINC"

            for idx, row in df.iterrows():
                code = str(row.get(code_col, "")).strip()
                term = str(row.get(term_col, "")).strip()
                if not code or not term:
                    continue

                if system == "LOINC":
                    code = self._normalize_loinc_code(code)

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
    # Indexing & scoring
    # -------------------------

    def _add_record(self, rec: TermRecord) -> None:
        self._term_index.setdefault(rec.term_norm, []).append(rec)
        self._code_index[self._code_key(system=rec.system, code=rec.code)] = rec

    @staticmethod
    def _code_key(system: str, code: str) -> str:
        return f"{TerminologyStore._normalize_system(system)}|{str(code).strip()}"

    @staticmethod
    def _normalize_system(system: Optional[str]) -> str:
        sys = str(system or "").strip().upper()
        if sys == "SNOMED":
            return "SNOMEDCT"
        return sys

    @staticmethod
    def _pick_col(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
        col_map = {str(c).lower(): str(c) for c in columns}
        for cand in candidates:
            if cand.lower() in col_map:
                return col_map[cand.lower()]
        return None

    @staticmethod
    def _score_record(query_norm: str, query_tokens: set[str], rec: TermRecord) -> float:
        if rec.term_norm == query_norm:
            return 1.0

        rec_tokens = set(rec.term_norm.split())
        base = jaccard(query_tokens, rec_tokens)

        if rec.term_norm.startswith(query_norm) or query_norm.startswith(rec.term_norm):
            base = min(1.0, base + 0.15)

        if len(query_tokens) <= 1 and base < 0.8:
            base *= 0.75

        return float(base)

    def _to_hit_dict(self, score: float, rec: TermRecord, query: str) -> Dict[str, Any]:
        # Ensure LOINC has normalized code and enriched meta if available
        extra = dict(rec.extra or {})
        if rec.system == "LOINC":
            extra_code = self._normalize_loinc_code(rec.code)
            if extra_code != rec.code:
                extra["normalized_code"] = extra_code
            meta = self._loinc_meta.get(extra_code)
            if meta:
                extra.setdefault("cui", meta.get("cui"))
                extra.setdefault("preferred_label", meta.get("preferred_label"))
                extra.setdefault("synonym_count", len(meta.get("synonyms") or []))
                extra.setdefault("loinc_class_id", meta.get("class_id"))

        snippet = f"term={rec.term} | code={rec.system}:{rec.code}"
        ev = Evidence(
            source_type="csv/xlsx",
            source_file=rec.source_file,
            id=rec.row_id,
            field="term/code",
            snippet=snippet,
            extra={"system": rec.system, "code": rec.code, **extra},
        ).to_dict()

        return {
            "kind": "terminology",
            "query": query,
            "term": rec.term,
            "system": rec.system,
            "code": rec.code,
            "score": float(score),
            "evidence": ev,
            "extra": extra,
        }
