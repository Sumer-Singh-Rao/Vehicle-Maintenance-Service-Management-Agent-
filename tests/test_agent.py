"""
tests/test_agent.py - Integration test for VehicleMaintenanceAgent powered by LangGraph.
"""
import pytest
from agent import VehicleMaintenanceAgent


def test_agent_configuration():
    """Verify VehicleMaintenanceAgent configuration and graph status."""
    agent = VehicleMaintenanceAgent()
    assert agent.model == "openai/gpt-oss-120b"
    assert agent.max_tool_calls == 12
    assert agent.is_configured() is True
