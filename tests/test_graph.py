"""
tests/test_graph.py - Unit tests for LangGraph StateGraph compilation and state definitions.
"""
import pytest
from agent.state import AgentState
from agent.graph import build_agent_graph
from agent.prompts import build_system_prompt
from config import GROQ_API_KEY


def test_agent_state_keys():
    """Verify AgentState TypedDict expected annotations."""
    annotations = AgentState.__annotations__
    assert "messages" in annotations
    assert "user_id" in annotations
    assert "selected_vehicle_id" in annotations
    assert "pending_confirmation" in annotations


def test_build_system_prompt():
    """Test system prompt formatting with vehicle context."""
    prompt = build_system_prompt(selected_vehicle_id=1)
    assert "Vehicle Maintenance AI Assistant" in prompt
    assert "ACTIVE VEHICLE IN UI" in prompt


def test_build_agent_graph():
    """Test LangGraph StateGraph compilation."""
    graph = build_agent_graph(GROQ_API_KEY)
    assert graph is not None
