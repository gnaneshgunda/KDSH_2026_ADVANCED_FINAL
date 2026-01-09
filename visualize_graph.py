"""
Visualize LangGraph workflow
"""

from langgraph.graph import StateGraph, END
from models import GraphState
from nodes import (
    load_corpus_node,
    extract_claims_node,
    embed_claims_node,
    retrieve_supporting_node,
    retrieve_opposing_node,
    analyze_consistency_node,
    error_handler_node
)


def build_and_visualize():
    """Build graph and generate visualization"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("load_corpus", load_corpus_node)
    workflow.add_node("extract_claims", extract_claims_node)
    workflow.add_node("embed_claims", embed_claims_node)
    workflow.add_node("retrieve_supporting", retrieve_supporting_node)
    workflow.add_node("retrieve_opposing", retrieve_opposing_node)
    workflow.add_node("analyze_consistency", analyze_consistency_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # Define edges
    workflow.set_entry_point("load_corpus")
    workflow.add_edge("load_corpus", "extract_claims")
    workflow.add_edge("extract_claims", "embed_claims")
    workflow.add_edge("embed_claims", "retrieve_supporting")
    workflow.add_edge("retrieve_supporting", "retrieve_opposing")
    workflow.add_edge("retrieve_opposing", "analyze_consistency")
    
    # Conditional edge
    workflow.add_conditional_edges(
        "analyze_consistency",
        lambda state: "error_handler" if state.get('error') else END,
        {
            "error_handler": "error_handler",
            END: END
        }
    )
    workflow.add_edge("error_handler", END)
    
    # Compile
    app = workflow.compile()
    
    # Generate visualization
    try:
        from IPython.display import Image, display
        display(Image(app.get_graph().draw_mermaid_png()))
        print("Graph visualization generated!")
    except Exception as e:
        print(f"Visualization requires graphviz: {e}")
        print("\nGraph structure:")
        print("START -> load_corpus -> extract_claims -> embed_claims")
        print("      -> retrieve_supporting -> retrieve_opposing")
        print("      -> analyze_consistency -> [error_handler] -> END")


if __name__ == "__main__":
    build_and_visualize()
