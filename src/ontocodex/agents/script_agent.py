from ontocodex.engine.state import OntoCodexState
from ontocodex.engine.artifacts import write_artifacts

def script_node(state: OntoCodexState) -> OntoCodexState:
    # Later: LLM-assisted script generation; start deterministic for skeleton
    script_lines = [
        "# Auto-generated OntoCodex enrichment script",
        "# TODO: implement owlready2/rdflib patch writing",
        "",
    ]
    for m in state.mappings:
        script_lines.append(f"# {m['term']} -> {m['system']}:{m['code']}")

    state.generated_script = "\n".join(script_lines)
    state.artifacts = write_artifacts(state)
    return state
