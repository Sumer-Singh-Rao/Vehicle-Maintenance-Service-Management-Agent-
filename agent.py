"""
agent.py - Compatibility module re-exporting VehicleMaintenanceAgent.
Delegates reasoning, multi-turn state management, and tool calls to LangGraph workflow graph.
"""
from agent import VehicleMaintenanceAgent, build_agent_graph, AgentState

__all__ = ["VehicleMaintenanceAgent", "build_agent_graph", "AgentState"]

if __name__ == "__main__":
    agent = VehicleMaintenanceAgent()
    print(f"LangGraph Agent ready ({agent.model}): {agent.is_configured()}")
