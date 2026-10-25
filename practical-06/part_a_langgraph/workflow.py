"""LangGraph StateGraph construction and execution for Practical 6."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Literal

from common import LLMClient
from langgraph.graph import END, START, StateGraph
from part_a_langgraph.nodes import (
    finalize_node,
    finance_node,
    marketing_node,
    research_node,
    review_node,
    revision_node,
)
from part_a_langgraph.state import CampaignState

logger = logging.getLogger("prac06.langgraph.workflow")


def build_campaign_graph(llm: LLMClient) -> Any:
    """Build and compile the multi-agent marketing & finance StateGraph."""
    graph = StateGraph(CampaignState)

    # Wrap nodes with LLM injection
    def _research(s: CampaignState) -> Dict[str, Any]:
        return research_node(s, llm)

    def _marketing(s: CampaignState) -> Dict[str, Any]:
        return marketing_node(s, llm)

    def _finance(s: CampaignState) -> Dict[str, Any]:
        return finance_node(s, llm)

    def _review(s: CampaignState) -> Dict[str, Any]:
        return review_node(s, llm)

    def _revision(s: CampaignState) -> Dict[str, Any]:
        return revision_node(s, llm)

    def _finalize(s: CampaignState) -> Dict[str, Any]:
        return finalize_node(s)

    graph.add_node("research", _research)
    graph.add_node("marketing", _marketing)
    graph.add_node("finance", _finance)
    graph.add_node("review", _review)
    graph.add_node("revision", _revision)
    graph.add_node("finalize", _finalize)

    # Base linear flow: START -> research -> marketing -> finance -> review
    graph.add_edge(START, "research")
    graph.add_edge("research", "marketing")
    graph.add_edge("marketing", "finance")
    graph.add_edge("finance", "review")

    # Conditional router after review:
    # If REVISE and iteration < max_iterations -> revision -> finance -> review loop
    # If APPROVED or limit reached -> finalize -> END
    def route_decision(state: CampaignState) -> Literal["revision", "finalize"]:
        feedback = state.get("review_feedback", {})
        status = feedback.get("status", "APPROVED")
        it = state.get("iteration", 1)
        max_it = state.get("max_iterations", 3)
        if status == "REVISE" and it < max_it:
            logger.info(f"Conditional Router: Status={status}, Iteration={it}/{max_it} -> routing to 'revision'")
            return "revision"
        logger.info(f"Conditional Router: Status={status}, Iteration={it}/{max_it} -> routing to 'finalize'")
        return "finalize"

    graph.add_conditional_edges(
        "review",
        route_decision,
        {
            "revision": "revision",
            "finalize": "finalize",
        },
    )

    # Revision leads back to finance to recompute line items
    graph.add_edge("revision", "finance")
    graph.add_edge("finalize", END)

    compiled = graph.compile()
    return compiled


def get_mermaid_diagram() -> str:
    """Return Mermaid diagram definition of the LangGraph state machine."""
    return """```mermaid
flowchart TD
    START([User Campaign Input]) --> Research[Research Agent]
    Research --> Marketing[Marketing Agent]
    Marketing --> Finance[Finance Agent]
    Finance --> Review{Review / Auditor Agent}
    Review -- "Over Budget (REVISE)" --> Revision[Revision Agent]
    Revision --> Finance
    Review -- "Budget Compliant (APPROVED)" --> Finalize[Finalize Strategy]
    Finalize --> END([Final Executive Strategy])

    classDef agent fill:#0E7C86,stroke:#0F2A3D,stroke-width:2px,color:#fff;
    classDef decision fill:#E58A1F,stroke:#0F2A3D,stroke-width:2px,color:#fff;
    classDef io fill:#F4F7F9,stroke:#3A4856,stroke-width:2px,color:#0F2A3D;

    class Research,Marketing,Finance,Revision,Finalize agent;
    class Review decision;
    class START,END io;
```"""
