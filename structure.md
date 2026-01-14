OntoCodexLocal/
  README.md
  pyproject.toml
  .env.example
  .gitignore

  data/                          # your repo data folder (mounted or symlinked)
    DOID.owl
    hp.obo
    MEDLINEPLUS.ttl
    snomed_omop.csv
    rxnorm_omop.csv
    loinc_omop.csv
    atc_omop.csv
    measurement_omop.csv
    LOINC_CUI.xlsx
    medDRA.xlsx
    MedDRA_CTCAE_mapping_v5.xlsx
    OMOP-MCC/

  configs/
    data_manifest.yaml           # where large files come from + required flags
    ontocodex.yaml               # thresholds, source priority tree, code systems
    prompts/
      decision.md
      ontology_reader.md
      knowledgebase.md
      terminology.md
      validator.md
      script_agent.md

  src/
    ontocodex/
      __init__.py

      api/                       # optional: local server layer
        main.py                  # FastAPI entry
        routes_runs.py
        schemas.py

      engine/
        state.py                 # LangGraph state (Pydantic/dataclass)
        graph.py                 # build_graph() returns compiled LangGraph
        policies.py              # confidence thresholds, evidence rules
        artifacts.py             # artifact writing: scripts, mappings, patches
        tracing.py               # step logging (local)

      agents/                    # each OntoCodex role = a LangGraph node
        decision_agent.py
        ontology_reader_agent.py
        knowledgebase_agent.py
        terminology_agent.py
        validator_agent.py
        script_agent.py

      kb/                        # Knowledge layer (functions first, vectors optional)
        __init__.py
        normalization.py         # normalize_term(), camelCase split, etc.
        evidence.py              # Evidence dataclasses + serialization
        terminology_store.py     # CSV/XLSX deterministic lookups (Layer 1)
        ontology_store.py        # OWL/OBO/TTL graph store (Layer 2)
        vector_store.py          # optional FAISS/Chroma (Layer 3)
        kb_api.py                # unified KnowledgeBase facade

      providers/
        __init__.py
        llm_base.py              # LLMProvider interface
        gemini_dev.py            # Google Gemini API key provider (local)
        embeddings.py            # embedder interface + Gemini embeddings impl

      io/
        loader.py                # data_manifest validation + downloading hooks
        owl_writer.py            # apply patch scripts / write OWL/Turtle outputs

      cli/
        run_local.py             # run a single workflow from terminal
        build_indexes.py         # build KB indexes + optional vector index

      tests/
        test_kb_lookup.py
        test_graph_flow.py
        test_validator_gates.py

  scripts/
    dev.sh                       # convenience: run api or cli
    format.sh


┌─────────────────────────────┐
│ macOS system                │
│  ├─ Ollama.app              │  ← model + server
│  │   └─ models in ~/.ollama │
│  └─ localhost:11434         │
└─────────────▲───────────────┘
              │ HTTP
┌─────────────┴───────────────┐
│ Conda env (ontocodex)        │
│  ├─ Python                  │
│  ├─ requests                │
│  └─ OntoCodex code           │
└─────────────────────────────┘

4️⃣ Decide where LLM is allowed to run (important)
✅ Allowed nodes

KnowledgeAgent (reasoning only)

Optional: guideline parsing agent

❌ Forbidden nodes

KnowledgeBaseAgent

TerminologyAgent

ValidatorAgent

Script/OWL Writer

This keeps OntoCodex non-hallucinating.