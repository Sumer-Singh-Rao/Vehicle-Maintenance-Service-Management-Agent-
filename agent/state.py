"""
agent/state.py - LangGraph State Definition.
"""
from typing import TypedDict, Annotated, Sequence, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Explicit State for VehicleCare AI Agent Graph.
    Contains message stream and structured context parameters.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: int
    selected_vehicle_id: Optional[int]
    pending_confirmation: bool
    pending_action: Optional[str]
    booking_details: Optional[Dict[str, Any]]
