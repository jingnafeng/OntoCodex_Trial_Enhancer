from ontocodex.io.owl_writer import load_graph, save_graph, add_omop_annotations, OmopAnnotationPayload
from ontocodex.engine.evidence_log import build_evidence_rows, write_jsonl

def script_node(state):
    """
    Script Generation Agent:
    Generates executable Python scripts to update ontologies with new classes, axioms, annotations, and provenance.
    """
    g = load_graph(state.ontology_path)

    for m in state.mappings:
        # You must know which ontology entity IRI you are annotating.
        # Typically this comes from OntologyReaderAgent / OntologyStore match.
        entity_iri = m.get("target_iri")
        if not entity_iri:
            continue

        extra = m.get("extra", {}) or {}
        payload = OmopAnnotationPayload(
            concept_id=str(extra.get("concept_id", "")),
            vocabulary=m.get("system"),
            concept_code=m.get("code"),
            source=m.get("evidence", {}).get("source_file"),
        )
        add_omop_annotations(g, entity_iri, payload)

    out_path = state.options.get("out_owl_path", "artifacts/enriched.owl")
    save_graph(g, out_path, fmt="xml")
    state.artifacts["enriched_owl"] = out_path
    
    evidence_path = state.options.get("evidence_path", f"artifacts/{state.run_id}/evidence.jsonl")
    rows = build_evidence_rows(run_id=state.run_id, mappings=state.mappings)
    write_jsonl(evidence_path, rows)
    state.artifacts["evidence_jsonl"] = evidence_path

    return state
