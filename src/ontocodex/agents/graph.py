from langgraph.graph import StateGraph, END

from ontocodex.engine.state import OntoCodexState

from .decision_agent import decision_node
from .knowledge_agent import knowledge_agent_node
from .knowledgebase_agent import knowledgebase_node
from .ontology_reader_agent import ontology_reader_node
from .script_agent import script_node
from .terminology_agent import terminology_node
from .trial_parser_agent import trial_parser_node
from .trial_orchestrator_agent import (
    init_trial_orchestrator_node,
    trial_advance_section_node,
    trial_finalize_orchestrator_node,
    trial_orchestrator_route,
    trial_section_kb_node,
    trial_section_llm_node,
    trial_section_terminology_node,
)
from .validator_agent import validator_node


# Start routing function
def route_from_start(state: OntoCodexState) -> str:
    """
    Choose pipeline at startup:
      - trial extraction path: go directly to decision agent
      - ontology normalization/enrichment path: read ontology first
    """
    opts = state.get("options", {}) or {}
    has_trial_text = bool(str(opts.get("trial_text", "")).strip()) or bool(str(opts.get("trial_text_file", "")).strip())
    task_type = str(state.get("task_type", "")).lower()
    if has_trial_text or task_type == "extract":
        return "to_decision"
    return "to_ontology_reader"


# Conditional routing function
def route_after_decision(state: OntoCodexState) -> str:
    """Routes to terminology, knowledge, or validation based on the decision."""
    routing_decision = state.get("routing", {})
    needs_term = routing_decision.get("needs_terminology", False)
    needs_know = routing_decision.get("needs_knowledge_agent", False)
    needs_trial = routing_decision.get("needs_trial_parser", False)
    needs_trial_orch = routing_decision.get("needs_trial_orchestrator", False)

    if needs_trial_orch:
        return "to_trial_orchestrator"

    if needs_trial:
        return "to_trial_parser"

    if needs_term:
        if routing_decision.get("skip_kb", False):
            return "to_terminology_only"
        return "to_terminology_path"
    if needs_know:
        return "to_knowledge_path"
    return "to_validator"  # Default path


def create_graph():
    """Creates the OntoCodex agent graph."""
    workflow = StateGraph(OntoCodexState)

    # Define the nodes
    workflow.add_node("start_router", lambda s: s)
    workflow.add_node("read_ontology", ontology_reader_node)
    workflow.add_node("make_decision", decision_node)
    workflow.add_node("lookup_kb", knowledgebase_node)
    workflow.add_node("map_terminology", terminology_node)
    workflow.add_node("enrich_knowledge", knowledge_agent_node)
    workflow.add_node("parse_trial", trial_parser_node)
    workflow.add_node("init_trial_orchestrator", init_trial_orchestrator_node)
    workflow.add_node("trial_section_llm", trial_section_llm_node)
    workflow.add_node("trial_section_kb", trial_section_kb_node)
    workflow.add_node("trial_section_terminology", trial_section_terminology_node)
    workflow.add_node("trial_advance_section", trial_advance_section_node)
    workflow.add_node("finalize_trial_orchestrator", trial_finalize_orchestrator_node)
    workflow.add_node("validate_mappings", validator_node)
    workflow.add_node("write_output", script_node)

    # Set the entrypoint
    workflow.set_entry_point("start_router")

    # Define the graph structure
    workflow.add_conditional_edges(
        "start_router",
        route_from_start,
        {
            "to_ontology_reader": "read_ontology",
            "to_decision": "make_decision",
        },
    )
    workflow.add_edge("read_ontology", "make_decision")

    # Branching from the decision node
    workflow.add_conditional_edges(
        "make_decision",
        route_after_decision,
        {
            "to_terminology_path": "lookup_kb",
            "to_terminology_only": "map_terminology",
            "to_knowledge_path": "enrich_knowledge",
            "to_trial_parser": "parse_trial",
            "to_trial_orchestrator": "init_trial_orchestrator",
            "to_validator": "validate_mappings",
        },
    )

    # Define remaining paths
    workflow.add_edge("lookup_kb", "map_terminology")
    workflow.add_edge("map_terminology", "validate_mappings")
    workflow.add_edge("enrich_knowledge", "validate_mappings")
    workflow.add_edge("parse_trial", "validate_mappings")
    workflow.add_conditional_edges(
        "init_trial_orchestrator",
        trial_orchestrator_route,
        {
            "to_llm": "trial_section_llm",
            "to_kb": "trial_section_kb",
            "to_terminology": "trial_section_terminology",
            "to_advance": "trial_advance_section",
            "to_finalize": "finalize_trial_orchestrator",
        },
    )
    workflow.add_conditional_edges(
        "trial_section_llm",
        trial_orchestrator_route,
        {
            "to_llm": "trial_section_llm",
            "to_kb": "trial_section_kb",
            "to_terminology": "trial_section_terminology",
            "to_advance": "trial_advance_section",
            "to_finalize": "finalize_trial_orchestrator",
        },
    )
    workflow.add_conditional_edges(
        "trial_section_kb",
        trial_orchestrator_route,
        {
            "to_llm": "trial_section_llm",
            "to_kb": "trial_section_kb",
            "to_terminology": "trial_section_terminology",
            "to_advance": "trial_advance_section",
            "to_finalize": "finalize_trial_orchestrator",
        },
    )
    workflow.add_conditional_edges(
        "trial_section_terminology",
        trial_orchestrator_route,
        {
            "to_llm": "trial_section_llm",
            "to_kb": "trial_section_kb",
            "to_terminology": "trial_section_terminology",
            "to_advance": "trial_advance_section",
            "to_finalize": "finalize_trial_orchestrator",
        },
    )
    workflow.add_conditional_edges(
        "trial_advance_section",
        trial_orchestrator_route,
        {
            "to_llm": "trial_section_llm",
            "to_kb": "trial_section_kb",
            "to_terminology": "trial_section_terminology",
            "to_advance": "trial_advance_section",
            "to_finalize": "finalize_trial_orchestrator",
        },
    )
    workflow.add_edge("finalize_trial_orchestrator", "validate_mappings")
    workflow.add_edge("validate_mappings", "write_output")
    workflow.add_edge("write_output", END)

    # Compile the graph
    app = workflow.compile()
    return app
