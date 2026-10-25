"""Typed state definition for the LangGraph Marketing & Finance Multi-Agent workflow."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class CampaignState(TypedDict, total=False):
    scenario: Dict[str, Any]
    research_data: Dict[str, Any]
    content_strategy: Dict[str, Any]
    financial_plan: Dict[str, Any]
    review_feedback: Dict[str, Any]
    iteration: int
    max_iterations: int
    revision_history: List[Dict[str, Any]]
    final_strategy: Dict[str, Any]
    node_metrics: Dict[str, Any]
    execution_log: List[Dict[str, Any]]
    status: str
