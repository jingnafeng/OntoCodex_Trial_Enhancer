from langgraph.graph import StateGraph, END

from ontocodex.engine.state import OntoCodexState

from .decision_agent import decision_node
from .knowledge_agent import knowledge_agent_node
from .knowledgebase_agent import knowledgebase_node
from .ontology_reader_agent import ontology_reader_node
from .script_agent import script_node
from .terminology_agent import terminology_node
from .validator_agent import validator_node


# Conditional routing function
def route_after_decision(state: OntoCodexState) -> str:
    """Routes to terminology, knowledge, or validation based on the decision."""
    routing_decision = state.get("routing", {})
    needs_term = routing_decision.get("needs_terminology", False)
    needs_know = routing_decision.get("needs_knowledge_agent", False)

    if needs_term and not routing_decision.get("skip_kb", False):
        return "to_terminology_path"
    if needs_know:
        return "to_knowledge_path"
    return "to_validator"  # Default path


def create_graph():
    """Creates the OntoCodex agent graph."""
    workflow = StateGraph(OntoCodexState)

    # Define the nodes
    workflow.add_node("read_ontology", ontology_reader_node)
    workflow.add_node("make_decision", decision_node)
    workflow.add_node("lookup_kb", knowledgebase_node)
    workflow.add_node("map_terminology", terminology_node)
    workflow.add_node("enrich_knowledge", knowledge_agent_node)
    workflow.add_node("validate_mappings", validator_node)
    workflow.add_node("write_output", script_node)

    # Set the entrypoint
    workflow.set_entry_point("read_ontology")

    # Define the graph structure
    workflow.add_edge("read_ontology", "make_decision")

    # Branching from the decision node
    workflow.add_conditional_edges(
        "make_decision",
        route_after_decision,
        {
            "to_terminology_path": "lookup_kb",
            "to_knowledge_path": "enrich_knowledge",
            "to_validator": "validate_mappings",
        },
    )

    # Define remaining paths
    workflow.add_edge("lookup_kb", "map_terminology")
    workflow.add_edge("map_terminology", "validate_mappings")
    workflow.add_edge("enrich_knowledge", "validate_mappings")
    workflow.add_edge("validate_mappings", "write_output")
    workflow.add_edge("write_output", END)

    # Compile the graph
    app = workflow.compile()
    return app