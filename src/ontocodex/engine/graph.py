from langgraph.graph import StateGraph, END
from ontocodex.engine.state import OntoCodexState

from ontocodex.agents.ontology_reader_agent import ontology_reader_node
from ontocodex.agents.decision_agent import decision_node
from ontocodex.agents.knowledgebase_agent import knowledgebase_node
from ontocodex.agents.terminology_agent import terminology_node
from ontocodex.agents.validator_agent import validator_node
from ontocodex.agents.script_agent import script_node
from ontocodex.agents.knowledge_agent import knowledge_agent_node



def build_graph():
    g = StateGraph(OntoCodexState)

    # Nodes (roles)
    g.add_node("ontology_reader", ontology_reader_node)
    g.add_node("decision", decision_node)
    g.add_node("knowledgebase", knowledgebase_node)
    g.add_node("terminology", terminology_node)
    g.add_node("knowledge_agent", knowledge_agent_node)
    g.add_node("validator", validator_node)
    g.add_node("script", script_node)

    # Default path
    g.set_entry_point("ontology_reader")
    g.add_edge("ontology_reader", "decision")

    # Conditional routing from decision
    def route(state: OntoCodexState) -> str:
        # decision_node sets state.routing
        if state.routing.get("skip_kb", False):
            return "terminology"
        return "knowledgebase"

    g.add_conditional_edges("decision", route, {
        "knowledgebase": "knowledgebase",
        "terminology": "terminology",
    })

    # Continue
    g.add_edge("knowledgebase", "terminology")
    g.add_edge("terminology", "knowledge_agent")
    g.add_edge("knowledge_agent", "validator")
    g.add_edge("terminology", "validator")

    # If validator passes -> script, else stop
    def after_validate(state: OntoCodexState) -> str:
        if state.errors:
            return END
        return "script"

    g.add_conditional_edges("validator", after_validate, {
        "script": "script",
        END: END,
    })

    g.add_edge("script", END)
    return g.compile()
